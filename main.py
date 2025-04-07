from fastapi import FastAPI
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
    return "mllm return msg";