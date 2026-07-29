from __future__ import annotations

from agents.common import run_clinical_agent
from agents.types import ClinicalState
from config.settings import get_settings


def triage(state: ClinicalState) -> dict:
    question = state["question"].lower()
    hits = [term for term in get_settings().safety["emergency_terms"] if term in question]
    if hits:
        return {"triage": {"reasoning": f"Emergency warning terms detected: {', '.join(hits)}.", "confidence": 0.95, "evidence": hits, "citations": [], "recommendations": ["Seek emergency medical care or call local emergency services now."], "uncertainty": "Keyword screening cannot determine severity."}}
    return {"triage": run_clinical_agent("Triage Agent", state["question"], state.get("model"))}
