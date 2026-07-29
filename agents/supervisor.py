from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.common import run_specialty_agent
from agents.consensus import aggregate, compute_index, consensus, is_low_consensus
from agents.protocols import protocol_context
from agents.report import report
from agents.specialty import SPECIALIST_NODES, SPECIALISTS
from agents.types import ClinicalState
from config.settings import get_settings
from rag.retriever import retrieve

# Safety valve: never loop the evidence-retrieval step more than this many times.
MAX_RETRIEVAL_ROUNDS = 1


def chair(state: ClinicalState) -> dict:
    """Tumor Board Chair (L1 intake + safety gate).

    Runs the emergency-keyword screen kept from the old triage agent, then hands
    the case to every specialty agent. Emergency findings are recorded as a
    safety alert that the final report surfaces; they do not stop the board.
    """
    question = state["question"].lower()
    hits = [term for term in get_settings().safety["emergency_terms"] if term in question]
    if hits:
        return {"safety_alert": f"Emergency warning terms detected: {', '.join(hits)}. "
                                "Advise seeking urgent care in parallel with this review."}
    return {}


def evidence_retrieval(state: ClinicalState) -> dict:
    """L5: when consensus is low, retrieve focused evidence for the dissenting
    members and re-run only those agents, then let the board re-score."""
    _, _, dissenting = compute_index(state)
    updates: dict = {"retrieval_rounds": state.get("retrieval_rounds", 0) + 1}
    for specialty in dissenting:
        role = SPECIALISTS[specialty]
        prior = state.get(specialty, {})
        focus = " ".join(prior.get("key_findings", [])) or state["question"]
        query = f"{role} {focus}"
        try:
            evidence = retrieve(query)
        except Exception:  # noqa: BLE001 - retrieval is best-effort
            evidence = []
        context = "\n\n".join(f"[{item.citation}] {item.content}" for item in evidence)
        refreshed = run_specialty_agent(
            role, state["question"], state.get("model"),
            protocol=protocol_context(specialty), context=context,
        )
        refreshed["citations"] = list(dict.fromkeys(
            [*refreshed["citations"], *(item.citation for item in evidence)]
        ))
        updates[specialty] = refreshed
    return updates


def _route_after_consensus(state: ClinicalState) -> str:
    """Decide whether to retrieve more evidence or finalise the report."""
    if is_low_consensus(state) and state.get("retrieval_rounds", 0) < MAX_RETRIEVAL_ROUNDS:
        return "retrieve"
    return "report"


def build_graph():
    """Chair -> 6 specialists (parallel) -> aggregate -> consensus
    -> [low? retrieve -> re-aggregate] -> report."""
    graph = StateGraph(ClinicalState)

    graph.add_node("chair", chair)
    for specialty, node in SPECIALIST_NODES.items():
        graph.add_node(f"{specialty}_agent", node)
    graph.add_node("aggregate_node", aggregate)
    graph.add_node("consensus_node", consensus)
    graph.add_node("evidence_retrieval_node", evidence_retrieval)
    graph.add_node("report_node", report)

    # L1 intake first
    graph.add_edge(START, "chair")
    # fan-out to all six board members
    for specialty in SPECIALISTS:
        graph.add_edge("chair", f"{specialty}_agent")
        graph.add_edge(f"{specialty}_agent", "aggregate_node")
    # synthesise -> score
    graph.add_edge("aggregate_node", "consensus_node")
    # L4 -> decide: retrieve more evidence, or finish
    graph.add_conditional_edges(
        "consensus_node",
        _route_after_consensus,
        {"retrieve": "evidence_retrieval_node", "report": "report_node"},
    )
    # after retrieval, re-aggregate and re-score (loop guarded by MAX_RETRIEVAL_ROUNDS)
    graph.add_edge("evidence_retrieval_node", "aggregate_node")
    graph.add_edge("report_node", END)
    return graph.compile()


def _initial_state(question: str, model: str | None, image_base64: str,
                   image_mime_type: str, vision_model: str | None) -> dict:
    return {
        "question": question,
        "model": model or "",
        "image_base64": image_base64,
        "image_mime_type": image_mime_type,
        "vision_model": vision_model or "",
        "retrieval_rounds": 0,
    }


def answer(question: str, model: str | None = None, image_base64: str = "",
           image_mime_type: str = "", vision_model: str | None = None) -> ClinicalState:
    return build_graph().invoke(
        _initial_state(question, model, image_base64, image_mime_type, vision_model)
    )


def stream_answer(question: str, model: str | None = None, image_base64: str = "",
                  image_mime_type: str = "", vision_model: str | None = None):
    """Yield progress events as the tumor board runs, then the final report.

    Event shapes (consumed by the Streamlit UI):
      {"type":"agent","name":<specialty>,"result":{...}}  as each member finishes
      {"type":"done","answer":<report>,"state":<full state>}  at the end
      {"type":"error","message":<str>}  on failure
    """
    graph = build_graph()
    state = _initial_state(question, model, image_base64, image_mime_type, vision_model)
    final_state: dict = dict(state)
    try:
        # stream_mode="updates" yields {node_name: {returned channel updates}}
        for update in graph.stream(state):
            for _node, delta in update.items():
                if not isinstance(delta, dict):
                    continue
                final_state.update(delta)
                # emit one event per specialty as it reports
                for specialty in SPECIALISTS:
                    if specialty in delta:
                        yield {"type": "agent", "name": specialty, "result": delta[specialty]}
        yield {"type": "done", "answer": final_state.get("report", ""), "state": final_state}
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}