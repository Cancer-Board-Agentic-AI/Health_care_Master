from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agents.specialty import SPECIALISTS
from agents.types import ClinicalState
from models.llm import get_llm

logger = logging.getLogger(__name__)


BOARD_SECTIONS = (
    ("pretreatment_confirmation", "1. Required Pretreatment Confirmation"),
    ("initial_treatment_decision", "2. Initial Treatment Decision"),
    ("breast_and_axillary_surgery", "3. Breast and Axillary Surgery"),
    ("postoperative_systemic_treatment", "4. Postoperative Systemic Treatment"),
    ("radiation", "5. Radiation"),
    ("supportive_and_access", "6. Supportive and Access Planning"),
)


def _board_evidence(state: ClinicalState) -> dict:
    """Compact, attributable input for the chair's adjudication call."""
    return {
        specialty: {
            "role": role,
            "key_findings": (state.get(specialty) or {}).get("key_findings", []),
            "recommended_plan": (state.get(specialty) or {}).get("recommended_plan", []),
            "required_tests": (state.get(specialty) or {}).get("required_tests", []),
            "contraindications": (state.get(specialty) or {}).get("contraindications", []),
            "missing_information": (state.get(specialty) or {}).get("missing_information", []),
            "confidence": (state.get(specialty) or {}).get("confidence", 0),
        }
        for specialty, role in SPECIALISTS.items()
        if state.get(specialty)
    }


def _empty_board_plan(state: ClinicalState, reason: str) -> dict:
    """Safe fallback: preserve sequencing without pretending conflicts were resolved."""
    required = list(dict.fromkeys(
        item
        for specialty in SPECIALISTS
        for item in (state.get(specialty) or {}).get("required_tests", [])
    ))
    missing = sorted({
        item
        for specialty in SPECIALISTS
        for item in (state.get(specialty) or {}).get("missing_information", [])
    })
    confirmation = [*required, *(f"Establish: {item}" for item in missing)]
    return {
        "case_summary": "Chair synthesis was unavailable; no preferred treatment pathway was inferred.",
        "pretreatment_confirmation": confirmation or [
            "Confirm diagnosis, clinical stage, biomarkers, and operative fitness before selecting treatment."
        ],
        "initial_treatment_decision": [
            "No preferred starting strategy can be declared until the treating board reviews the "
            "specialty opinions and the pretreatment data above."
        ],
        "breast_and_axillary_surgery": ["To be decided after the initial-treatment decision."],
        "postoperative_systemic_treatment": ["To be decided from final pathology and biomarkers."],
        "radiation": ["To be decided from operation performed and final pathologic risk."],
        "supportive_and_access": ["Assess patient goals, fitness, affordability, access, and treatment feasibility."],
        "decision_dependencies": missing,
        "unresolved_disagreements": [reason],
    }


def _synthesise_board_plan(state: ClinicalState) -> dict:
    """Have the chair adjudicate one sequenced plan, rather than concatenate opinions."""
    system = """You are the Chair of a multidisciplinary breast oncology tumor board.
Synthesize the members' opinions into ONE coherent, patient-specific decision pathway for clinician review.

Rules:
- Do not concatenate or vote-count recommendations.
- First state what must be confirmed before treatment.
- In initial_treatment_decision, name exactly one preferred starting strategy if the supplied facts support one.
- Upfront surgery, neoadjuvant endocrine therapy, neoadjuvant chemotherapy, and endocrine therapy alone are
  mutually exclusive starting strategies. Never present them as co-equal actions. Put a nonpreferred strategy
  only in an explicit IF/THEN contingency and state the finding that would activate it.
- Separate decisions that can be made now from decisions that require surgical pathology or response assessment.
- Specify breast and axillary surgery separately within the surgery section.
- Do not repeat endocrine therapy, chemotherapy, or radiation across sections.
- Surface disagreements that cannot safely be resolved; never invent patient facts, staging, biomarkers, or citations.
- Use only recommendations supported by the supplied specialty assessments.

Return ONLY valid JSON with this exact shape:
{"case_summary":"...",
"pretreatment_confirmation":["..."],
"initial_treatment_decision":["Preferred: ...","If ... then ..."],
"breast_and_axillary_surgery":["Breast: ...","Axilla: ..."],
"postoperative_systemic_treatment":["..."],
"radiation":["..."],
"supportive_and_access":["..."],
"decision_dependencies":["Decision X depends on finding Y"],
"unresolved_disagreements":["..."]}"""
    user = (
        f"CASE:\n{state.get('question', '')}\n\n"
        f"BOARD CONSENSUS NOTES:\n{state.get('consensus_notes', '')}\n\n"
        "SPECIALTY ASSESSMENTS:\n"
        + json.dumps(_board_evidence(state), ensure_ascii=False)
    )
    try:
        response = get_llm(state.get("model") or None).invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        raw = str(response.content).removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        plan = {
            "case_summary": str(data.get("case_summary", "")).strip(),
            **{
                key: [str(x).strip() for x in data.get(key, []) if str(x).strip()]
                for key, _ in BOARD_SECTIONS
            },
            "decision_dependencies": [
                str(x).strip() for x in data.get("decision_dependencies", []) if str(x).strip()
            ],
            "unresolved_disagreements": [
                str(x).strip() for x in data.get("unresolved_disagreements", []) if str(x).strip()
            ],
        }
        decisions = plan["initial_treatment_decision"]
        preferred = [item for item in decisions if item.lower().startswith("preferred:")]
        contingencies = [
            item for item in decisions
            if item.lower().startswith(("if ", "when "))
        ]
        if len(preferred) != 1 or len(preferred) + len(contingencies) != len(decisions):
            raise ValueError(
                "chair must return exactly one Preferred strategy and only explicit IF/WHEN contingencies"
            )
        return plan
    except Exception as exc:  # noqa: BLE001 - report must still be produced
        logger.warning("tumor-board synthesis failed: %s", exc)
        return _empty_board_plan(state, f"Chair synthesis unavailable: {type(exc).__name__}.")


def _format_board_plan(plan: dict) -> str:
    blocks: list[str] = []
    if plan.get("case_summary"):
        blocks.append(f"**Board conclusion:** {plan['case_summary']}")
    for key, heading in BOARD_SECTIONS:
        items = plan.get(key, [])
        blocks.append(f"#### {heading}\n" + (
            "\n".join(f"- {item}" for item in items) if items else "- No recommendation recorded."
        ))
    if plan.get("decision_dependencies"):
        blocks.append(
            "#### Decisions Pending Additional Results\n"
            + "\n".join(f"- {item}" for item in plan["decision_dependencies"])
        )
    if plan.get("unresolved_disagreements"):
        blocks.append(
            "#### Unresolved Board Disagreements\n"
            + "\n".join(f"- {item}" for item in plan["unresolved_disagreements"])
        )
    return "\n\n".join(blocks)


def _specialty_section(state: ClinicalState) -> str:
    blocks: list[str] = []
    for specialty, role in SPECIALISTS.items():
        r = state.get(specialty) or {}
        if not r:
            continue
        findings = "; ".join(r.get("key_findings", [])) or "—"
        plan = "; ".join(r.get("recommended_plan", [])) or r.get("reasoning", "—")
        contra = "; ".join(r.get("contraindications", []))
        block = (
            f"**{role} — {r.get('confidence', 0):.0%} confidence**\n\n"
            f"- Key findings: {findings}\n"
            f"- Recommendation: {plan}\n"
        )
        if contra:
            block += f"- Contraindications / risks: {contra}\n"
        blocks.append(block)
    return "\n".join(blocks)


def report(state: ClinicalState) -> dict:
    """L6: assemble the final structured tumor-board report for the clinician."""
    ideal = _synthesise_board_plan(state)

    citations = list(dict.fromkeys(
        c for specialty in SPECIALISTS
        for c in (state.get(specialty) or {}).get("citations", [])
    ))
    required_tests = list(dict.fromkeys(
        t for specialty in SPECIALISTS
        for t in (state.get(specialty) or {}).get("required_tests", [])
    ))

    parts: list[str] = ["# Virtual Tumor Board Report"]

    if state.get("safety_alert"):
        parts.append(f"> ⚠️ **Safety alert:** {state['safety_alert']}")

    parts.append(f"## Case\n{state.get('question', '')}")

    index = state.get("consensus_index")
    if index is None:
        compact_consensus = "Not computed."
    else:
        band = "High" if index >= 0.75 else "Moderate" if index >= 0.60 else "Low"
        compact_consensus = f"**{index:.0%} — {band} consensus.**"
        if state.get("retrieval_rounds"):
            compact_consensus += f" Evidence review: {state['retrieval_rounds']} round(s)."
    parts.append("## Consensus Confidence Index\n" + compact_consensus)

    parts.append(
        "## Final Recommendation\n\n"
        "### A. Ideal Guideline-Based Plan\n"
        + _format_board_plan(ideal)
    )

    if required_tests:
        parts.append("## Recommended Tests / Investigations\n" + "\n".join(f"- {t}" for t in required_tests))

    parts.append("## Evidence & References\n" + ("\n".join(f"- {c}" for c in citations) or "- No local citations retrieved."))

    parts.append(
        "## Limitations & Disclaimer\n"
        "This report is generated by an AI multidisciplinary decision-support system for "
        "**review by the treating clinician**. It does not make autonomous decisions, and the "
        "final treatment decision rests with the clinician."
    )

    return {"board_plan": ideal, "report": "\n\n".join(parts)}