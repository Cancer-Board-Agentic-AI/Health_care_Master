from __future__ import annotations

import json
import shutil
import base64
import io
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from agents.chat_log import load_log as load_chat_eval_log
from agents.supervisor import answer, stream_answer
from backend.database import history, save_conversation
from config.settings import get_settings
from rag.eval_log import load_log as load_rag_eval_log
from rag.ingest import ingest_pdf

app = FastAPI(title="Local Medical AI", version="1.0.0")

IMAGE_SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=10000)
    model: str | None = None


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _validate_image(image: UploadFile) -> tuple[bytes, str]:
    if image.content_type not in IMAGE_SUFFIXES:
        raise HTTPException(400, "Supported image formats: JPEG, PNG, WEBP")
    data = await image.read()
    if not data or len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "Image must be between 1 byte and 20 MB")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Invalid image file") from exc
    return data, image.content_type


def _store_image(data: bytes, content_type: str) -> Path:
    destination = get_settings().path(get_settings().app["upload_dir"]) / f"{uuid4().hex}{IMAGE_SUFFIXES[content_type]}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local-only"}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    result = answer(request.question, request.model)
    save_conversation(request.question, result["report"])
    return {"answer": result["report"], "state": result}


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Server-sent events: each specialist agent's result as it finishes, then the summary token-by-token, then a final done event."""

    def generate():
        for event in stream_answer(request.question, request.model):
            if event["type"] == "done":
                save_conversation(request.question, event["answer"])
            yield _sse(event)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/ask-with-image")
async def ask_with_image(question: str = Form(..., min_length=3), model: str = Form("medgemma:27b"), vision_model: str = Form("medgemma:27b"), image: UploadFile = File(...)) -> dict:
    """Run local multimodal analysis using an image plus textual clinical context."""
    data, content_type = await _validate_image(image)
    destination = _store_image(data, content_type)
    result = answer(question, model, base64.b64encode(data).decode("ascii"), content_type, vision_model)
    save_conversation(question, result["report"])
    return {"answer": result["report"], "state": result, "image_id": destination.name}


@app.post("/ask-with-image/stream")
async def ask_with_image_stream(question: str = Form(..., min_length=3), model: str = Form("medgemma:27b"), vision_model: str = Form("medgemma:27b"), image: UploadFile = File(...)) -> StreamingResponse:
    data, content_type = await _validate_image(image)
    destination = _store_image(data, content_type)
    image_b64 = base64.b64encode(data).decode("ascii")

    def generate():
        for event in stream_answer(question, model, image_b64, content_type, vision_model):
            if event["type"] == "done":
                save_conversation(question, event["answer"])
                event = {**event, "image_id": destination.name}
            yield _sse(event)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, int | str]:
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(400, "Please upload a PDF")
    destination = get_settings().path(get_settings().app["upload_dir"]) / Path(file.filename).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return {"filename": destination.name, "chunks": ingest_pdf(destination)}


@app.get("/documents/samples")
def list_sample_documents() -> list[str]:
    """PDFs bundled under the curated sample directory, for users without a local PDF to upload."""
    sample_dir = get_settings().path(get_settings().app["sample_pdf_dir"])
    if not sample_dir.is_dir():
        return []
    return sorted(path.name for path in sample_dir.glob("*.pdf"))


@app.post("/documents/samples/{filename}")
def ingest_sample_document(filename: str) -> dict[str, int | str]:
    sample_dir = get_settings().path(get_settings().app["sample_pdf_dir"])
    source = sample_dir / Path(filename).name
    if source.suffix.lower() != ".pdf" or not source.is_file() or source.parent.resolve() != sample_dir.resolve():
        raise HTTPException(404, "Unknown sample PDF")
    return {"filename": source.name, "chunks": ingest_pdf(source)}


@app.get("/history")
def get_history() -> list[dict[str, str]]:
    return history()


@app.get("/rag-eval-log")
def get_rag_eval_log() -> list[dict]:
    """Every RAG turn logged so far: question, retrieved chunks with scores, and the generated answer."""
    return load_rag_eval_log()


@app.get("/chat-eval-log")
def get_chat_eval_log() -> list[dict]:
    """Every /ask turn logged so far: total latency, per-agent latency/success/error."""
    return load_chat_eval_log()
