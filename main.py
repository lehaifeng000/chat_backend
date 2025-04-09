from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import os
import uuid


app = FastAPI()

origins = ["*"]  # 允许来自所有源的请求

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

prefix_url = "http://127.0.1:8000"

STATIC_URL = "/static"
app.mount(STATIC_URL, StaticFiles(directory="static/uploads"), name="static")


@app.get("/")
def main():
    html_path = Path("res/html/index.html")
    with open(html_path, "r") as file:
        return HTMLResponse(content=file.read())

@app.get("/chat")
def chat():
    return "mllm return msg"

@app.get("/chat")
def chat():
    return "mllm return msg"

@app.post("/chat/send_message")
async def send_message(text: str = Form(...), img_url: str = Form(...)):
    print("收到消息：", text)
    print("收到图片：", img_url)
    # 这里可以添加处理逻辑
    return {"text": "你的问题是:\""+ text+"\"\n,请稍后，正在处理..."}

@app.post("/chat/upload_img")
async def upload_img(img: UploadFile = File(...)):
    print("收到图片：", img.filename)
    try:
        image = await img.read()
        # 生成唯一文件名
        file_extension = os.path.splitext(img.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = Path("static/uploads") / unique_filename
        with open(file_path, "wb") as f:
            f.write(image)
        image_url = f"{prefix_url}{STATIC_URL}/{unique_filename}"
        print(image_url)
        return {"image_url": image_url}
    except Exception as e:
        print(e)