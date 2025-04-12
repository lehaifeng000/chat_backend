from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image

import os
import uuid

from predict import gen_model, eval_question

model, tokenizer, image_processor = gen_model()
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
prefix_url = "https://0329c0a18e86.ngrok.app"

UPLOAD_DIR = "static/uploads"
STATIC_URL = "/"+ UPLOAD_DIR
app.mount(STATIC_URL, StaticFiles(directory=UPLOAD_DIR), name="static")


@app.get("/")
def main():
    html_path = Path("res/html/index.html")
    with open(html_path, "r") as file:
        return HTMLResponse(content=file.read())

@app.get("/chat")
def chat():
    return "mllm return msg"

@app.post("/chat/send_message")
async def send_message(text: str = Form(...), img_name: str = Form(...)):
    print("收到消息：", text)
    print("收到图片：", img_name)
    # img_url替换成路径
    img_path = UPLOAD_DIR+"/"+img_name

    # 推理
    output = eval_question(model, tokenizer, image_processor, img_path, text)
    print("111111")
    print(output)

    # 这里可以添加处理逻辑
    # resp_text = "你的问题是:\""+ text+"\"\n,请稍后，正在处理..."
    return {"text": output}

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