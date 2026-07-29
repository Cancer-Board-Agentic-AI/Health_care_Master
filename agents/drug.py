from __future__ import annotations
from agents.common import run_clinical_agent
from agents.types import ClinicalState
def drug(state: ClinicalState) -> dict:
    return {"drug": run_clinical_agent("Drug Interaction Agent", state["question"], state.get("model"))}
