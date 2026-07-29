from __future__ import annotations

from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load sentence-transformer embeddings locally; first run may download model weights."""
    model = get_settings().models["embedding"]
    try:
        # Resolve from the on-disk Hugging Face cache only. Passing a resolved
        # directory prevents SentenceTransformers from creating a fallback model.
        local_model = snapshot_download(repo_id=model, local_files_only=True)
        return HuggingFaceEmbeddings(
            model_name=local_model,
            model_kwargs={"device": "cuda" if _cuda_available() else "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except OSError as exc:
        raise RuntimeError(
            "Local embedding weights for BAAI/bge-large-en-v1.5 are not installed. "
            "Provision them before starting the offline service."
        ) from exc


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
