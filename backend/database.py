from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from config.settings import get_settings


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = get_settings().path(get_settings().app["database_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL)")
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_conversation(question: str, answer: str) -> None:
    with connection() as conn:
        conn.execute("INSERT INTO conversations(created_at, question, answer) VALUES (?, ?, ?)", (datetime.now(timezone.utc).isoformat(), question, answer))


def history(limit: int = 30) -> list[dict[str, str]]:
    with connection() as conn:
        rows = conn.execute("SELECT created_at, question, answer FROM conversations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"created_at": row[0], "question": row[1], "answer": row[2]} for row in reversed(rows)]
