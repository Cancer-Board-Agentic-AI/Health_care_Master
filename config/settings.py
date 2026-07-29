from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    """Validated application configuration loaded from the local YAML file."""

    app: dict[str, Any]
    models: dict[str, Any]
    rag: dict[str, Any]
    safety: dict[str, Any]

    def path(self, value: str) -> Path:
        return ROOT / value


@lru_cache
def get_settings() -> Settings:
    with (ROOT / "config" / "config.yaml").open("r", encoding="utf-8") as handle:
        return Settings.model_validate(yaml.safe_load(handle))
