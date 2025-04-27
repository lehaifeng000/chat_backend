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
sys.path.append('/home/haifeng/code/SVE-Math')

from gllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from gllava.conversation import conv_templates, SeparatorStyle
from gllava.model.builder import load_pretrained_model
from gllava.utils import disable_torch_init
from gllava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image, ImageDraw
import math
from torchvision import transforms as T

prompt_choice = {
    "none": "",
    "wo": "\nPlease directly answer the question.\nQuestion: ",
    "cot": "\nPlease first conduct reasoning, and then answer the question.\nQuestion: "
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
    model_path = os.path.expanduser('/home/haifeng/code/SVE-Math/checkpoint/SVE-llava-deepseek-7B')
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,4,  'cross_channel' )

    return model, tokenizer, image_processor

    

def eval_question(model, tokenizer, image_processor, image, question, conv_mode="llava_deepsk", temperature=0, save_dir=""):
    disable_torch_init()
    # image_file = item["image"]
    # # 暂时使用拼接路径，后续img需要传入
    if image is None:
        # 创建一个空白图像
        image_PIL = Image.new("RGB", (480, 480), (0, 0, 0))
    else:
        image_PIL = Image.open(image).convert("RGB")
    image_tensor,image_glip = process_images(
        image_PIL,
        image_processor,
        model.config
    )
    # qs = question
    qs = f"{question}\n"
    qs = prompt_choice['cot']+qs
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
            image_glips=image_glip.unsqueeze(0).to(torch.bfloat16).cuda(),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    print(is_bbox(outputs))
    output_type = "text"
    output_content = outputs
    ret = {}
    if is_bbox(outputs):
        output_type = "bbox"
        draw = ImageDraw.Draw(image_PIL)
        scale_bbox = ast.literal_eval(outputs)
        bbox = rescale_bbox(scale_bbox, image_PIL)
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=5)
        print(bbox)
        unique_filename = f"gen_{uuid.uuid4()}.jpg"
        # output_content = unique_filename
        file_path = Path(save_dir) / unique_filename
        image_PIL.save(str(file_path))
        ret["text"] = "The bounding box coordinate is" + str(bbox)
        ret["img_name"] = unique_filename
    else:
        ret["text"] = outputs
    ret["type"] = output_type
    return ret

if __name__ == "__main__":
    model, tokenizer, image_processor = gen_model()
    # image = "/home/haifeng/code/SVE-Math/playground/Geo170K/images/test/0.png"
    # image = "/home/haifeng/code/tmp/tri2.png"
    image = "/home/haifeng/code/SVE-Math/mathglance/benchmark/data_final/mathscope/easy/images/final_construction_sample_13.png"

    # question = "\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\nQuestion:As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, DE parallel  BC, then the size of angle CED is ()\nChoices:\nA:40\u00b0\nB:60\u00b0\nC:120\u00b0\nD:140\u00b0"
    question = "Please provide the bounding box coordinate of the region this sentence describes: triangle ABC."
    print("----------- 开始推理 -----------")
    eval_question(model, tokenizer, image_processor, image, question)
    