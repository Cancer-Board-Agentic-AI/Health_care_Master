from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)


def log_chat_turn(question: str, model: str | None, total_latency_s: float, agent_timings: list[dict[str, Any]], error: str | None = None) -> None:
    """Append one JSON record per /ask turn: overall latency plus per-agent timing/outcome, for perf and error review."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "model": model,
        "total_latency_s": round(total_latency_s, 3),
        "agents": agent_timings,
        "error": error,
    }
    try:
        path = get_settings().path(get_settings().app["chat_log_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("Could not write chat eval log entry", exc_info=True)


def load_log() -> list[dict[str, Any]]:
    """Read back all logged chat turns, oldest first."""
    path = get_settings().path(get_settings().app["chat_log_path"])
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
