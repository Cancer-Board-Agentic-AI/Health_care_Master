from __future__ import annotations
from agents.common import run_clinical_agent
from agents.types import ClinicalState
def diagnosis(state: ClinicalState) -> dict:
    return {"diagnosis": run_clinical_agent("Differential Diagnosis Agent", state["question"], state.get("model"))}
