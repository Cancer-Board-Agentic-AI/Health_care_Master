from __future__ import annotations
from agents.common import run_clinical_agent
from agents.types import ClinicalState
def lab(state: ClinicalState) -> dict:
    return {"lab": run_clinical_agent("Laboratory Interpretation Agent", state["question"], state.get("model"))}
