from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)


def log_rag_result(question: str, model: str | None, retrieved: list[dict[str, Any]], answer: dict[str, Any]) -> None:
    """Append one JSON record per RAG query for later review of retrieval/answer quality."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "model": model,
        "retrieved": retrieved,
        "answer": answer,
    }
    try:
        path = get_settings().path(get_settings().rag["eval_log_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("Could not write RAG eval log entry", exc_info=True)


def load_log() -> list[dict[str, Any]]:
    """Read back all logged RAG turns, oldest first."""
    path = get_settings().path(get_settings().rag["eval_log_path"])
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
