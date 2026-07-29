from __future__ import annotations

from agents.common import run_specialty_agent
from agents.protocols import protocol_context, required_inputs
from agents.types import ClinicalState
from rag.retriever import retrieve

# specialty id -> display role shown in the board and report.
SPECIALISTS: dict[str, str] = {
    "radiology": "Radiology Agent",
    "pathology": "Pathology & Molecular Agent",
    "surgery": "Surgical Oncology Agent",
    "radiation": "Radiation Oncology Agent",
    "medical_oncology": "Medical Oncology Agent",
    "supportive": "Supportive Care Agent",
}


def _retrieve_context(question: str) -> tuple[str, list[str]]:
    """Shared L3 evidence retrieval; returns (context_block, citations)."""
    try:
        evidence = retrieve(question)
    except Exception:  # noqa: BLE001 - retrieval is best-effort, never fatal
        return "", []
    context = "\n\n".join(f"[{item.citation}] {item.content}" for item in evidence)
    return context, [item.citation for item in evidence]


def make_specialist(specialty: str):
    """Build one LangGraph node for a specialty. All six share this logic."""
    role = SPECIALISTS[specialty]

    def node(state: ClinicalState) -> dict:
        question = state["question"]
        protocol = protocol_context(specialty)
        context, retrieved_citations = _retrieve_context(question)
        result = run_specialty_agent(role, question, state.get("model"), protocol=protocol, context=context)
        # merge retrieved citations so evidence is traceable to source
        result["citations"] = list(dict.fromkeys([*result["citations"], *retrieved_citations]))
        # surface protocol-required inputs the case may lack (feeds L1 missing-data)
        missing = [field for field in required_inputs(specialty)
                   if field.lower() not in question.lower()]
        if missing:
            result["missing_information"] = list(dict.fromkeys([*result["missing_information"], *missing]))
        return {specialty: result}

    node.__name__ = f"{specialty}_node"
    return node


# Instantiate the six nodes once for import by the supervisor graph.
SPECIALIST_NODES = {specialty: make_specialist(specialty) for specialty in SPECIALISTS}