from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Optional

import os
import uuid
import re
import ast
import Pyro5.api

from predict import gen_model, eval_question

# model, tokenizer, image_processor = gen_model()
# model, tokenizer, image_processor = None, None, None

app = FastAPI()

origins = ["*"]  # 允许来自所有源的请求

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# prefix_url = "http://127.0.1:8000"
prefix_url = "https://56b6f8e84781.ngrok.app"

UPLOAD_DIR = "static/uploads"
STATIC_URL = "/"+ UPLOAD_DIR
app.mount(STATIC_URL, StaticFiles(directory=UPLOAD_DIR), name="static")

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

@app.get("/")
def main():
    html_path = Path("res/html/index.html")
    with open(html_path, "r") as file:
        return HTMLResponse(content=file.read())

@app.get("/chat")
def chat():
    return "mllm return msg"

# model: default, math
@app.post("/chat/send_message")
async def send_message(text: str = Form(...), img_name: Optional[str] = Form(None), model: Optional[str] = Form("default")):
    print("模型: ", model)
    print("收到消息：", text)
    print("收到图片：", img_name)
    # img_url替换成路径
    img_path = None
    if img_name:
        img_path = Path(UPLOAD_DIR) / img_name
        img_path = str(img_path.resolve())
    print("lhf: 图片路径: ", img_path)
    
    uri = "PYRO:model_default@127.0.0.1:50000" # default model
    if model != "default":
        if model == "math":
            uri = "PYRO:model_math@127.0.0.1:50001"
    
    rpc_model = Pyro5.api.Proxy(uri)  # 获取远程模型代理
    outputs = rpc_model.eval(text, img_path)  # 调用远程方法

    # outputs = eval_question(model, tokenizer, image_processor, img_path, text, save_dir=UPLOAD_DIR)
    print(outputs)

    # 处理输出
    ret = {}
    if is_bbox(outputs):
        output_type = "bbox"
        image_PIL = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image_PIL)
        scale_bbox = ast.literal_eval(outputs)
        bbox = rescale_bbox(scale_bbox, image_PIL)
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=5)
        print(bbox)
        unique_filename = f"gen_{uuid.uuid4()}.jpg"
        # output_content = unique_filename
        file_path = Path(UPLOAD_DIR) / unique_filename
        image_PIL.save(str(file_path))
        ret["text"] = "The bounding box coordinate is" + str(bbox)
        ret['img_url'] = f"{prefix_url}{STATIC_URL}/{unique_filename}"
    else:
        ret["text"] = outputs

    return ret

@app.post("/chat/upload_img")
async def upload_img(img: UploadFile = File(...)):
    print("收到图片：", img.filename)
    try:
        image = await img.read()
        # 生成唯一文件名
        file_extension = os.path.splitext(img.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = Path(UPLOAD_DIR) / unique_filename
        with open(file_path, "wb") as f:
            f.write(image)
        image_url = f"{prefix_url}{STATIC_URL}/{unique_filename}"
        print(image_url)
        return {"image_url": image_url, "img_name":unique_filename}
    except Exception as e:
        print(e)