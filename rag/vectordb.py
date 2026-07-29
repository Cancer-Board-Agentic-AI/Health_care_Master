from __future__ import annotations

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from config.settings import get_settings
from models.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    settings = get_settings()
    directory = settings.path(settings.rag["persist_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.rag["collection_name"],
        persist_directory=str(directory),
        embedding_function=get_embeddings(),
        # No usage telemetry leaves this local deployment.
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )
