from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from rag.vectordb import get_vectorstore


def ingest_pdf(path: Path) -> int:
    """Extract, chunk, and persist a local PDF. Returns number of stored chunks."""
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be ingested")
    if not path.is_file():
        raise FileNotFoundError(path)
    settings = get_settings()
    documents = PyPDFLoader(str(path)).load()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for document in documents:
        document.metadata.update({"source": path.name, "document_id": digest})
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag["chunk_size"], chunk_overlap=settings.rag["chunk_overlap"]
    ).split_documents(documents)
    if chunks:
        get_vectorstore().add_documents(chunks, ids=[f"{digest}:{index}" for index in range(len(chunks))])
    return len(chunks)
