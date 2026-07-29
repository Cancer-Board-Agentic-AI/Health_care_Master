from __future__ import annotations
from agents.common import run_clinical_agent
from agents.types import ClinicalState
def symptoms(state: ClinicalState) -> dict:
    return {"symptoms": run_clinical_agent("Symptom Analysis Agent", state["question"], state.get("model"))}
