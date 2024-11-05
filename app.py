"""
FastAPI app for uploading PDF files and embedding them into Qdrant.
"""

from typing import List, Optional
import os
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from rag import QdrantRAGBot
from dotenv import dotenv_values

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
rag_bot = QdrantRAGBot()

templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    """set allowed file types"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def read_form(request: Request):
    """Display upload form"""
    datasets = [
        name
        for name in os.listdir(UPLOAD_FOLDER)
        if os.path.isdir(os.path.join(UPLOAD_FOLDER, name))
    ]
    return templates.TemplateResponse(
        "upload.html", {"request": request, "datasets": datasets, "error": None}
    )


@app.get("/delete-dataset/{dataset_name}/{file_name}")
async def delete_dataset(dataset_name: str, file_name: str):
    """Delete a dataset"""
    path = os.path.join(UPLOAD_FOLDER, dataset_name, file_name)

    if os.path.exists(path):
        os.remove(path)
    rag_bot.delete_dataset(dataset_name, path)
    if not os.listdir(os.path.join(UPLOAD_FOLDER, dataset_name)):
        os.rmdir(os.path.join(UPLOAD_FOLDER, dataset_name))

    return JSONResponse({"status": "success"})


@app.get("/dataset-files/{dataset_name}")
async def get_dataset_files(dataset_name: str):
    """Get list of files in a dataset"""
    folder = os.path.join(UPLOAD_FOLDER, dataset_name)
    if not os.path.exists(folder):
        return JSONResponse({"files": []})

    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return JSONResponse({"files": files})


@app.post("/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(...),
    overwrite: Optional[str] = Form(None),
):
    """Upload PDF files and embed them into Qdrant"""

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
            overwrite_flag = overwrite == "yes"
            folder = os.path.join(UPLOAD_FOLDER, dataset_name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = os.path.join(folder, filename)
            with open(save_path, "wb") as f:
                f.write(contents)
            success_files.append(filename)

            rag_bot.embed_pdf(
                dataset=dataset_name, pdf_path=save_path, overwrite=overwrite_flag
            )
        else:
            error = f"檔案 {file.filename} 不允許的檔案類型"
            return templates.TemplateResponse(
                "upload.html", {"request": request, "error": error}
            )

    if success_files:
        return templates.TemplateResponse(
            "upload_success.html", {"request": request, "filenames": success_files}
        )

    error = "未選擇檔案或檔案類型不符合要求"
    return templates.TemplateResponse(
        "upload.html", {"request": request, "error": error}
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    """Chat page"""
    config = dotenv_values(".env")
    return templates.TemplateResponse(
        "chat_embed.html", {"request": request, "url": config.get("BOT_URL")}
    )
