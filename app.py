"""
FastAPI app for uploading PDF files and embedding them into Qdrant.
"""

from typing import List
import os
from uuid import uuid4
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from rag import embed_pdf

app = FastAPI()

templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def read_form(request: Request):
    client = QdrantClient(url="http://localhost:6333")
    collection_list = client.get_collections()
    collections = [i.name for i in collection_list.collections]
    return templates.TemplateResponse(
        "upload.html", {"request": request, "collections": collections}
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...),
):

    error = None
    success_files = []
    for file in files:
        if allowed_file(file.filename):
            contents = await file.read()
            if len(contents) > MAX_CONTENT_LENGTH:
                error = f"檔案 {file.filename} 過大，不能超過20MB"
                return templates.TemplateResponse(
                    "upload.html", {"request": request, "error": error}
                )
            filename = file.filename
            unique_filename = f"{uuid4().hex}_{filename}"
            save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            with open(save_path, "wb") as f:
                f.write(contents)
            success_files.append(filename)

            embed_pdf(collection_name, save_path)
        else:
            error = f"檔案 {file.filename} 不允許的檔案類型"
            return templates.TemplateResponse(
                "upload.html", {"request": request, "error": error}
            )

    if success_files:
        return templates.TemplateResponse(
            "upload_success.html", {"request": request, "filenames": success_files}
        )
    else:
        error = "未選擇檔案或檔案類型不符合要求"
        return templates.TemplateResponse(
            "upload.html", {"request": request, "error": error}
        )


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat_embed.html", {"request": request})
