import argparse
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import shortuuid
import uuid
from pathlib import Path
import re
import ast
import sys
sys.path.append('/home/haifeng/code/SVE-Math-Qwen2_5')

from gllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from gllava.conversation import conv_templates, SeparatorStyle
from gllava.model.builder import load_pretrained_model
from gllava.utils import disable_torch_init
from gllava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from qwen_vl_utils import process_vision_info

from PIL import Image, ImageDraw
import math
from torchvision import transforms as T

prompt_choice = {
    "none": "",
    "wo": "\nPlease directly answer the question.\nQuestion: ",
    "cot": "\nPlease first conduct reasoning, and then answer the question.\nQuestion: ",
    "multimath": "\nPlease reason step by step, and put your final answer within \\boxed{}.\nEach step is placed on a new line, using the following format: \nStep X (Mathematical theorem/basis used): Detailed solution steps. \nAnswer: \\boxed{}"
}
#and provide the correct option letter, e.g., A, B, C, D
#and provide the correct option letter, e.g., A, B, C, D, at the end
def is_bbox(line):
    pattern = r"^\[(\s*(?:0|1)(?:\.\d+)?,\s*)*(?:0|1)(?:\.\d+)?\s*\]$"
    return re.match(pattern, line) is not None

def rescale_bbox(bbox, image):
    h = image.height
    w = image.width
    print("h:{} w:{}".format(h, w))

    adjusted_bbox = bbox[:]  # 创建 bbox 的副本，避免修改原始列表

    if h > w:
        adjusted_bbox[0] *= h  # x1
        adjusted_bbox[2] *= h  # x2
        adjusted_bbox[0] -= (h - w) // 2
        adjusted_bbox[2] -= (h - w) // 2
        adjusted_bbox[1] *= h  # y1
        adjusted_bbox[3] *= h  # y2

    else:
        adjusted_bbox[0] *= w  # x1
        adjusted_bbox[2] *= w  # x2
        adjusted_bbox[1] *= w  # y1
        adjusted_bbox[3] *= w  # y2
        adjusted_bbox[1] -= (w - h) // 2
        adjusted_bbox[3] -= (w - h) // 2

    # 确保坐标非负 (模拟 ReLU)
    for i in range(len(adjusted_bbox)):
        adjusted_bbox[i] = int(max(0, adjusted_bbox[i]))

    return adjusted_bbox



def build_transform():
    PIXEL_MEAN=[103.53, 116.28, 123.675]
    PIXEL_STD=[57.375, 57.12, 58.395]
    to_bgr_transform = T.Lambda(lambda x: x * 255)

    normalize_transform = T.Normalize(
        mean=PIXEL_MEAN, std=PIXEL_STD
    )

    transform = T.Compose(
        [
            T.Resize(1000),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
            # T.ToPILImage()
        ]
    )
    return transform

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_images(image, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    if image_aspect_ratio=='pad': image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    glip_image=build_transform()(image)
    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    return image, glip_image


def gen_model():
    disable_torch_init()
    # model_path = os.path.expanduser('/home/haifeng/code/SVE-Math-Qwen2_5/checkpoints/Qwen2.5-VL-7B-unfre_geoglip_math360k_align2_sft+geodet')
    model_path = os.path.expanduser('/home/haifeng/code/SVE-Math-Qwen2_5/checkpoints/Qwen2.5-VL-7B-2mlp_trainglip+sft+geodet+deduction05')
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,4,  'cross_channel' )
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,num_of_kvs=4,
        merge_version='cross_channel')

    return model, tokenizer, image_processor

    

def eval_question(model, tokenizer, image_processor, image, question, conv_mode="qwen_2", temperature=0, save_dir=""):
    disable_torch_init()
    qs = question
    cur_prompt = qs
    # qs = f"{question}\n"
    # cur_prompt = prompt_choice['multimath']+qs
    if image is None:
        image = '/home/haifeng/code/web/chat_backend/blank.jpg'
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"{image}",
                "resized_width":model.config.image_resized_width,
                "resized_height":model.config.image_resized_height,
            },
            {"type": "text", "text": cur_prompt},
        ],
    }]
    image_inputs, video_inputs = process_vision_info(messages) 
    prompt = image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
    # prompt = copy.deepcopy(llava_to_openai(prompt['conversations']))
    inputs = image_processor(text=[prompt],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")
    inputs = inputs.to("cuda").to(torch.bfloat16)

    input_ids = inputs['input_ids']
    inputs['image_glip']=inputs['image_glip'].unsqueeze(0)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=2048,
            use_cache=True)
    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    input_token_len = input_ids.shape[1]
    stop_str = "<|im_end|>" 
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        output_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, output_ids)
        ]
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs
    # print(outputs)
    # print(is_bbox(outputs))
    # output_type = "text"
    # output_content = outputs
    # ret = {}
    # if is_bbox(outputs):
    #     output_type = "bbox"
    #     image_PIL = Image.open(image).convert("RGB")
    #     draw = ImageDraw.Draw(image_PIL)
    #     scale_bbox = ast.literal_eval(outputs)
    #     bbox = rescale_bbox(scale_bbox, image_PIL)
    #     draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=5)
    #     print(bbox)
    #     unique_filename = f"gen_{uuid.uuid4()}.jpg"
    #     # output_content = unique_filename
    #     file_path = Path(save_dir) / unique_filename
    #     image_PIL.save(str(file_path))
    #     ret["text"] = "The bounding box coordinate is" + str(bbox)
    #     ret["img_name"] = unique_filename
    # else:
    #     ret["text"] = outputs
    # ret["type"] = output_type
    # return ret

if __name__ == "__main__":
    model, tokenizer, image_processor = gen_model()
    # image = "/home/haifeng/code/SVE-Math/playground/Geo170K/images/test/0.png"
    # image = "/home/haifeng/code/tmp/tri2.png"
    # image = "/home/haifeng/code/SVE-Math/mathglance/benchmark/data_final/mathscope/easy/images/final_construction_sample_13.png"

    # question = "\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\nQuestion:As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, DE parallel  BC, then the size of angle CED is ()\nChoices:\nA:40\u00b0\nB:60\u00b0\nC:120\u00b0\nD:140\u00b0"
    # question = "Please provide the bounding box coordinate of the region this sentence describes: triangle ABC."
    # image = "/home/haifeng/code/web/chat_backend/tmp/ellipse1.jpg"
    # question = "Please provide the bounding box coordinate of the region this sentence describes: ellipse L."
    image = None
    question ='can you help me sovle the geo problems?'
    print("----------- 开始推理 -----------")
    eval_question(model, tokenizer, image_processor, image, question, temperature=0.2)
    