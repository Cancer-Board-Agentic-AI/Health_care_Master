from __future__ import annotations

from typing import TypedDict


class AgentResult(TypedDict, total=False):
    key_findings: list[str]
    recommended_plan: list[str]
    reasoning: str
    required_tests: list[str]
    contraindications: list[str]
    evidence: list[str]
    citations: list[str]
    confidence: float
    missing_information: list[str]
    uncertainty: str  # kept for backward compatibility with existing UI/report


class ClinicalState(TypedDict, total=False):
    # inputs
    question: str
    model: str
    vision_model: str
    image_base64: str
    image_mime_type: str

    # optional image observation, feeds radiology / pathology
    vision: AgentResult

    # six virtual tumor-board members (L2)
    radiology: AgentResult
    pathology: AgentResult
    surgery: AgentResult
    radiation: AgentResult
    medical_oncology: AgentResult
    supportive: AgentResult

    # orchestration / synthesis
    aggregate: str
    consensus_index: float          # L4 Consensus Confidence Index, 0..1
    consensus_notes: str            # agreements / disagreements summary
    retrieval_rounds: int           # how many L5 evidence rounds ran
    board_plan: dict                 # chair-adjudicated, phased treatment pathway
    report: str                     # L6 dual-plan tumor-board report