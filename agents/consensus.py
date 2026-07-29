from __future__ import annotations

from statistics import mean, pstdev

from agents.specialty import SPECIALISTS
from agents.types import ClinicalState

# A board member is treated as "dissenting / uncertain" if its own confidence
# is this far (or more) below the board's mean confidence.
DISSENT_MARGIN = 0.20
# Overall board consensus below this is considered LOW -> triggers L5 retrieval.
LOW_CONSENSUS_THRESHOLD = 0.60


def aggregate(state: ClinicalState) -> dict:
    """Fan-in: collect the six specialty assessments into one readable block."""
    parts: list[str] = []
    for specialty, role in SPECIALISTS.items():
        value = state.get(specialty)
        if not value:
            continue
        plan = "; ".join(value.get("recommended_plan", [])) or value.get("reasoning", "")
        parts.append(f"{role} (confidence {value.get('confidence', 0):.0%}): {plan}")
    return {"aggregate": "\n".join(parts)}


def _contributing(state: ClinicalState) -> dict[str, dict]:
    """Specialty results that actually returned something usable."""
    return {
        specialty: state[specialty]
        for specialty in SPECIALISTS
        if state.get(specialty) and state[specialty].get("confidence", 0) > 0
    }


def compute_index(state: ClinicalState) -> tuple[float, str, list[str]]:
    """Return (consensus_index, human-readable notes, dissenting specialty ids).

    The index is deterministic and explainable (no extra LLM call), which is
    important for a reproducible, publishable metric. It blends three signals:

      coverage    - how many of the 6 members actually reported
      agreement   - how tightly their confidences cluster (low spread = agree)
      confidence  - the board's mean confidence

    and applies a small penalty when members flag missing information.
    """
    results = _contributing(state)
    if not results:
        return 0.0, "No specialty agent produced a usable assessment.", list(SPECIALISTS)

    confidences = [r["confidence"] for r in results.values()]
    coverage = len(results) / len(SPECIALISTS)
    mean_conf = mean(confidences)
    spread = pstdev(confidences) if len(confidences) > 1 else 0.0
    agreement = max(0.0, 1.0 - spread * 2)  # spread of 0.5 -> agreement 0

    missing_load = mean(
        1.0 if r.get("missing_information") else 0.0 for r in results.values()
    )
    penalty = 0.10 * missing_load

    index = max(0.0, min(1.0, 0.5 * agreement + 0.3 * mean_conf + 0.2 * coverage - penalty))

    # members sitting well below the board mean are the points of disagreement
    dissenting = [
        specialty for specialty, r in results.items()
        if r["confidence"] <= mean_conf - DISSENT_MARGIN
    ]

    notes = [
        f"Consensus Confidence Index: {index:.0%}",
        f"Coverage: {len(results)}/{len(SPECIALISTS)} specialty agents reported.",
        f"Mean confidence: {mean_conf:.0%} | Agreement (spread): {agreement:.0%}.",
    ]
    if dissenting:
        named = ", ".join(SPECIALISTS[s] for s in dissenting)
        notes.append(f"Lower-confidence / dissenting members: {named}.")
    else:
        notes.append("No member fell materially below the board mean.")
    flagged = sorted({field for r in results.values() for field in r.get("missing_information", [])})
    if flagged:
        notes.append("Missing information flagged: " + ", ".join(flagged) + ".")

    return index, "\n".join(notes), dissenting


def consensus(state: ClinicalState) -> dict:
    """Graph node: write the Consensus Confidence Index into state (L4)."""
    index, notes, _ = compute_index(state)
    return {"consensus_index": index, "consensus_notes": notes}


def is_low_consensus(state: ClinicalState) -> bool:
    """Used by the supervisor to decide whether to trigger L5 evidence retrieval."""
    return state.get("consensus_index", 0.0) < LOW_CONSENSUS_THRESHOLD