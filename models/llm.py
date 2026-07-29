from __future__ import annotations

import os
from typing import Any

from langchain_ollama import ChatOllama

from config.settings import get_settings


def get_llm(model_name: str | None = None, *, json_output: bool = True) -> ChatOllama:
    """Create an Ollama chat client. Ollama is contacted only on the local network."""
    settings = get_settings()
    options: dict[str, Any] = {
        "model": model_name or settings.models["primary"],
        "base_url": os.getenv("OLLAMA_BASE_URL", settings.models["ollama_base_url"]),
        "temperature": settings.models["temperature"],
        "client_kwargs": {"timeout": settings.models["request_timeout_s"]},
    }
    if json_output:
        options["format"] = "json"
    return ChatOllama(**options)
