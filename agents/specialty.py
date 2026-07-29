from __future__ import annotations

import json

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


def peer_dossier(state: ClinicalState, reviewing_specialty: str) -> str:
    """Shared transcript supplied to one member during cross-specialty review."""
    dossier = {
        specialty: {
            "role": SPECIALISTS[specialty],
            "key_findings": [str(x)[:600] for x in result.get("key_findings", [])[:5]],
            "recommended_plan": [str(x)[:600] for x in result.get("recommended_plan", [])[:5]],
            "reasoning_summary": str(result.get("reasoning", ""))[:1200],
            "contraindications": [str(x)[:400] for x in result.get("contraindications", [])[:3]],
            "missing_information": [str(x)[:300] for x in result.get("missing_information", [])[:5]],
            "confidence": result.get("confidence", 0),
        }
        for specialty in SPECIALISTS
        if (result := state.get(specialty))
    }
    return (
        f"You are revising the {SPECIALISTS[reviewing_specialty]} opinion after reviewing "
        "the complete first-round board transcript below. Resolve conflicts within your scope; "
        "state any disagreement that remains clinically important.\n"
        + json.dumps(dossier, ensure_ascii=False)
    )


def make_deliberator(specialty: str):
    """Build a second-round node that revises one opinion using all six first-round views."""
    role = SPECIALISTS[specialty]

    def node(state: ClinicalState) -> dict:
        question = state["question"]
        protocol = protocol_context(specialty)
        context, retrieved_citations = _retrieve_context(question)
        prior = state.get(specialty) or {}
        result = run_specialty_agent(
            role,
            question,
            state.get("model"),
            protocol=protocol,
            context=context,
            peer_context=peer_dossier(state, specialty),
        )
        result["citations"] = list(dict.fromkeys([
            *prior.get("citations", []), *result["citations"], *retrieved_citations,
        ]))
        result["missing_information"] = list(dict.fromkeys([
            *prior.get("missing_information", []), *result["missing_information"],
        ]))
        return {specialty: result}

    node.__name__ = f"{specialty}_deliberation_node"
    return node


DELIBERATION_NODES = {specialty: make_deliberator(specialty) for specialty in SPECIALISTS}
