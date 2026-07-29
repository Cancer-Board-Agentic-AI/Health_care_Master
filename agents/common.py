from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.types import AgentResult
from models.llm import get_llm

# Richer schema matching the "common for all agents" block in the architecture:
# key findings, recommended plan, reasoning, required tests, contraindications,
# evidence, confidence, missing information.
SCHEMA = (
    '{"key_findings":["..."],"recommended_plan":["..."],"reasoning":"...",'
    '"required_tests":["..."],"contraindications":["..."],"evidence":["..."],'
    '"citations":["..."],"confidence":0.0,"missing_information":["..."]}'
)
logger = logging.getLogger(__name__)

_EMPTY: AgentResult = {
    "key_findings": [],
    "recommended_plan": [],
    "reasoning": "",
    "required_tests": [],
    "contraindications": [],
    "evidence": [],
    "citations": [],
    "confidence": 0.0,
    "missing_information": [],
    "uncertainty": "",
}


def run_specialty_agent(
    role: str, question: str, model: str | None = None, protocol: str = "",
    context: str = "", peer_context: str = "",
) -> AgentResult:
    """Run one virtual tumor-board member and return its structured assessment.

    The framing is deliberately non-deferring: the agent is a board member
    producing a specialty opinion FOR REVIEW BY THE TREATING CLINICIAN. The
    clinician — not the model — makes the decision, which is exactly why the
    agent must give a complete, specific, guideline-grounded assessment rather
    than refuse. This matches the study's non-autonomous design.
    """
    system = f"""You are the {role}, an independent member of a virtual multidisciplinary oncology tumor board.
Your task is to produce THIS SPECIALTY'S assessment of the case for review by the treating clinician.
The final treatment decision always rests with the clinician; your role is to give a complete, specific,
guideline-grounded specialty opinion, not to withhold one. Do NOT refuse or say the case is outside your
scope — that defeats the tumor board. State assumptions explicitly and list anything genuinely missing
under missing_information instead of declining.

Ground every recommendation in the SPECIALTY PROTOCOL below. Prefer its definitions, staging thresholds,
and rules over general knowledge, and reference the relevant rule in your reasoning. Do not invent citations.

SPECIALTY PROTOCOL:
{protocol or 'No curated protocol was loaded for this specialty; reason from standard oncology guidelines and flag this under missing_information.'}

RETRIEVED LOCAL EVIDENCE:
{context or 'None retrieved.'}

TUMOR-BOARD PEER REVIEW:
{peer_context or 'Initial independent assessment: peer opinions are not available yet.'}

{("This is the DELIBERATION ROUND. Review every peer opinion, identify agreements and conflicts "
  "relevant to your specialty, and return a complete revised assessment. In reasoning, briefly state "
  "which peer findings changed or confirmed your recommendation. Do not merely repeat your first opinion."
  if peer_context else
  "This is the INITIAL ROUND. Form an independent specialty opinion before seeing other members' views.")}

Return ONLY valid JSON exactly shaped as: {SCHEMA}. Confidence is a number from 0 to 1."""

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = get_llm(model).invoke([SystemMessage(content=system), HumanMessage(content=question)])
            raw = str(response.content).removeprefix("```json").removesuffix("```").strip()
            try:
                data: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("%s returned non-JSON output; using prose fallback", role)
                return {**_EMPTY, "reasoning": raw or "The local model returned no content.", "confidence": 0.35,
                        "uncertainty": "The model did not return the requested structured format."}
            return _coerce(data)
        except Exception as exc:  # noqa: BLE001 - surface any local model failure as low confidence
            last_error = exc
            logger.warning("%s failed (attempt %d/2): %s", role, attempt + 1, exc)
    return {**_EMPTY, "reasoning": "Agent unavailable.", "uncertainty": f"{type(last_error).__name__}: {last_error}"}


def _coerce(data: dict[str, Any]) -> AgentResult:
    """Normalise raw model JSON into a fully-populated AgentResult."""
    def as_list(key: str) -> list[str]:
        return [str(x) for x in data.get(key, []) or []]

    return {
        "key_findings": as_list("key_findings"),
        "recommended_plan": as_list("recommended_plan"),
        "reasoning": str(data.get("reasoning", "No reasoning returned.")),
        "required_tests": as_list("required_tests"),
        "contraindications": as_list("contraindications"),
        "evidence": as_list("evidence"),
        "citations": as_list("citations"),
        "confidence": max(0.0, min(1.0, float(data.get("confidence", 0) or 0))),
        "missing_information": as_list("missing_information"),
        "uncertainty": str(data.get("uncertainty", "")),
    }