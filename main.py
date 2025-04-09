from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path


app = FastAPI()

origins = ["*"]  # 允许来自所有源的请求

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def main():
    html_path = Path("res/html/index.html")
    with open(html_path, "r") as file:
        return HTMLResponse(content=file.read())

@app.get("/chat")
def chat():
    return "mllm return msg"

@app.post("/chat/send_message")
async def send_message(text: str = Form(...), img: UploadFile = File(...)):
    print("收到消息：",text)
    try:
        image = await img.read()
            # 你可以在这里保存文件到服务器
            # 例如:
        with open(f"uploads/{img.filename}", "wb") as f:
            f.write(image)
        print("receive img: ", img.filename)
    except Exception as e:
        print(e)
    return "yes, this is response"