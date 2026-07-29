from __future__ import annotations

import logging
import os

import httpx

from agents.types import ClinicalState
from config.settings import get_settings

logger = logging.getLogger(__name__)


def vision(state: ClinicalState) -> dict:
    """Describe an uploaded medical image locally; never claim image-only diagnosis."""
    image = state.get("image_base64")
    if not image:
        return {"vision": {"reasoning": "No image was supplied.", "confidence": 0.0, "evidence": [], "citations": [], "recommendations": [], "uncertainty": "No visual evidence available."}}
    prompt = ("Examine this clinical image only for observable features. Describe image type, quality, "
              "visible findings, and important limitations. Do not diagnose, prescribe, or infer facts not visible. "
              f"User context: {state['question']}")
    try:
        settings = get_settings()
        endpoint = os.getenv("OLLAMA_BASE_URL", settings.models["ollama_base_url"]).rstrip("/") + "/api/chat"
        response = httpx.post(
            endpoint,
            json={
                "model": state.get("vision_model") or settings.models["vision"],
                "stream": False,
                "messages": [{"role": "system", "content": "You are a local medical-image observation assistant. Use cautious language."}, {"role": "user", "content": prompt, "images": [image]}],
            },
            timeout=600,
        )
        if response.is_error:
            raise RuntimeError(
                f"Ollama vision request failed ({response.status_code}): {response.text[:1000]}"
            )
        content = str(response.json().get("message", {}).get("content", "")).strip()
        return {"vision": {"reasoning": content or "No visual observations returned.", "confidence": 0.45, "evidence": [content] if content else [], "citations": [], "recommendations": ["Have a qualified clinician review the original image."], "uncertainty": "Image-only interpretation is limited and cannot establish a diagnosis."}}
    except Exception as exc:
        logger.exception("Vision agent failed")
        return {"vision": {"reasoning": "Vision agent unavailable.", "confidence": 0.0, "evidence": [], "citations": [], "recommendations": [], "uncertainty": f"{type(exc).__name__}: {exc}"}}
