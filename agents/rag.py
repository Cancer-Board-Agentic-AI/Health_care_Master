from __future__ import annotations
from agents.common import run_clinical_agent
from agents.types import ClinicalState
from rag.eval_log import log_rag_result
from rag.retriever import retrieve
def rag(state: ClinicalState) -> dict:
    try:
        evidence = retrieve(state["question"])
    except Exception:
        evidence = []
    context = "\n\n".join(f"[{item.citation}] {item.content}" for item in evidence)
    result = run_clinical_agent("Medical RAG Agent", state["question"], state.get("model"), context)
    result["citations"] = list(dict.fromkeys([*result["citations"], *(item.citation for item in evidence)]))
    result["evidence"] = [item.content for item in evidence]
    log_rag_result(
        state["question"],
        state.get("model"),
        [{"citation": item.citation, "score": item.score, "content": item.content} for item in evidence],
        result,
    )
    return {"rag": result}
