from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from config.settings import get_settings

logger = logging.getLogger(__name__)

# One protocol file per specialty. Keys are the specialty ids used across the
# graph; values are the *_extraction.json filenames your authoring pipeline
# produces. Adjust filenames here if yours differ.
PROTOCOL_FILES: dict[str, str] = {
    "radiology": "radiology_extraction.json",
    "pathology": "pathology_extraction.json",
    "surgery": "surgery_extraction.json",
    "radiation": "radiation_extraction.json",
    "medical_oncology": "medical_oncology_extraction.json",
    "supportive": "supportive_extraction.json",
}


def _protocol_dir() -> Path:
    """Directory holding the *_extraction.json files.

    Reads settings.app['protocol_dir'] if present, else defaults to a
    'protocols' folder next to the app root. CONFIRM this path for your repo.
    """
    settings = get_settings()
    configured = settings.app.get("protocol_dir") if hasattr(settings, "app") else None
    # Files produced by tools/protocol_authoring/extract_protocol.py land here.
    return settings.path(configured) if configured else settings.path("tools/protocol_authoring/drafts")


@lru_cache(maxsize=None)
def load_protocol(specialty: str) -> dict:
    """Load and cache one specialty's extracted protocol. Empty dict if missing."""
    filename = PROTOCOL_FILES.get(specialty)
    if not filename:
        return {}
    path = _protocol_dir() / filename
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Protocol file not found for %s at %s", specialty, path)
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("Protocol file for %s is not valid JSON: %s", specialty, exc)
        return {}


def protocol_context(specialty: str, max_defs: int = 20, max_rules: int = 20) -> str:
    """Render a specialty's protocol into a compact text block for the LLM.

    Pulls definitions, clinical rules, and required inputs from the extraction
    JSON so the agent reasons against your curated guideline, not free memory.
    """
    data = load_protocol(specialty)
    if not data:
        return ""

    lines: list[str] = []

    definitions = data.get("extracted_definitions", [])[:max_defs]
    if definitions:
        lines.append("DEFINITIONS AND STAGING:")
        for item in definitions:
            term = item.get("term", "").strip()
            definition = item.get("definition", "").strip()
            values = item.get("valid_values") or []
            suffix = f" (valid: {', '.join(map(str, values))})" if values else ""
            if term:
                lines.append(f"- {term}: {definition}{suffix}")

    rules = data.get("extracted_rules", [])[:max_rules]
    if rules:
        lines.append("\nCLINICAL RULES:")
        for item in rules:
            description = item.get("description", "").strip()
            condition = item.get("condition_hint", "").strip()
            action = item.get("action_hint", "").strip()
            if description or action:
                lines.append(f"- {description}: IF {condition} THEN {action}".rstrip(": "))

    required = data.get("extracted_required_inputs", [])
    if required:
        lines.append("\nREQUIRED CASE INPUTS: " + ", ".join(map(str, required)))

    return "\n".join(lines).strip()


def required_inputs(specialty: str) -> list[str]:
    """Fields this specialty needs — used to raise missing-data flags (L1)."""
    return [str(x) for x in load_protocol(specialty).get("extracted_required_inputs", [])]