"""
Domain layer: Protocol schema.

This module defines the canonical, disease-agnostic, specialty-agnostic
shape that every clinical protocol JSON file must conform to.

Design principles (see Claude_Project_Guidelines.md / 03_ENGINEERING_GUIDELINES.md):
- Protocols are executable specifications, not configuration.
- Clinical rules are DATA (condition trees), never Python if-statements.
- Confidence here is strictly agent-local (completeness / consistency /
  evidence). Inter-agent agreement is a Consensus-layer concern and must
  never appear in this schema.
- Unknown is a valid, first-class value everywhere — never inferred.

This file has zero framework dependencies beyond Pydantic. No I/O,
no LLM calls, no persistence concerns. Pure domain model.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProtocolStatus(str, Enum):
    DRAFT_EXTRACTED_PENDING_REVIEW = "draft_extracted_pending_clinician_review"
    DRAFT_CLINICIAN_REVIEW_REQUIRED = "draft_clinician_review_required"
    APPROVED = "approved"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class RuleType(str, Enum):
    CLINICAL = "clinical"
    SAFETY = "safety"


class SharedProtocolKind(str, Enum):
    """
    Shared protocols come in two flavors, per Shared_Clinical_Protocol_Architecture.md:
    - DEFINITIONAL: closed vocabularies / structural definitions (TNM staging
      categories, biomarker value sets, terminology standards). No rules.
    - BEHAVIORAL: cross-cutting rule logic that applies universally rather
      than to one specialty (validation, safety, confidence, missing-data
      handling). Uses the same Rule shape as specialty protocols.
    """
    DEFINITIONAL = "definitional"
    BEHAVIORAL = "behavioral"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BoundaryType(str, Enum):
    SELF_RESTRICTION = "self_restriction"
    CROSS_AGENT_BOUNDARY = "cross_agent_boundary"


class FieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    LIST = "list"
    UNKNOWN_OR_STRING = "string_or_unknown"
    UNKNOWN_OR_INTEGER = "integer_or_unknown"


# ---------------------------------------------------------------------------
# Condition tree (used by both clinical_rules and safety_rules)
# ---------------------------------------------------------------------------

class FieldCondition(BaseModel):
    """A single leaf condition against one input field."""
    field: str = Field(..., description="Name of the case-card field being evaluated")
    operator: Literal["equals", "not_equals", "in", "not_in", "present", "absent", "gt", "lt", "gte", "lte"]
    value: Any = Field(default=None, description="Comparison value; omitted for present/absent")


class BooleanCondition(BaseModel):
    """A nested boolean composition of conditions: AND / OR / NOT."""
    all_of: list["ConditionNode"] | None = Field(default=None, description="AND")
    any_of: list["ConditionNode"] | None = Field(default=None, description="OR")
    not_: Union["ConditionNode", None] = Field(default=None, alias="not", description="NOT")

    @field_validator("all_of", "any_of")
    @classmethod
    def _non_empty(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("boolean condition list must not be empty")
        return v


ConditionNode = Union[FieldCondition, BooleanCondition]
BooleanCondition.model_rebuild()


# ---------------------------------------------------------------------------
# Rules (unified shape for clinical + safety)
# ---------------------------------------------------------------------------

class RuleAction(BaseModel):
    """What happens when a rule's condition evaluates true."""
    required_action: list[str] = Field(default_factory=list)
    missing_data_action: str | None = Field(
        default=None,
        description="What to do if the condition cannot be evaluated because required data is absent",
    )


class Rule(BaseModel):
    rule_id: str
    rule_type: RuleType
    severity: Severity = Severity.INFO
    condition: ConditionNode
    action: RuleAction
    source_id: str = Field(..., description="Citation key into the evidence/guideline repository")
    source_excerpt: str | None = Field(
        default=None,
        description="Verbatim source sentence supporting this rule — required for LLM-extracted drafts",
    )


# ---------------------------------------------------------------------------
# Confidence logic — strictly agent-local
# ---------------------------------------------------------------------------

class ConfidenceLogic(BaseModel):
    """
    Defines how THIS agent computes its own local confidence score.

    Must never reference other agents' outputs or consensus/agreement —
    that is computed downstream by the Consensus domain model, not here.
    """
    method: Literal["weighted_completeness"] = "weighted_completeness"
    weights: dict[str, float]

    @field_validator("weights")
    @classmethod
    def _weights_sum_to_one(cls, v: dict[str, float]):
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"confidence weights must sum to 1.0, got {total}")
        return v

    @field_validator("weights")
    @classmethod
    def _no_inter_agent_signals(cls, v: dict[str, float]):
        forbidden = {"inter_agent_agreement", "consensus_score", "agreement_with_other_agents"}
        offending = forbidden.intersection(v.keys())
        if offending:
            raise ValueError(
                f"confidence_logic must be agent-local; forbidden keys found: {offending}. "
                "Inter-agent agreement belongs to the Consensus layer, not the protocol."
            )
        return v


# ---------------------------------------------------------------------------
# Output schema — real type/constraint metadata, not example values
# ---------------------------------------------------------------------------

class OutputField(BaseModel):
    type: FieldType
    enum_values: list[str] | None = Field(default=None, description="Required when type == 'enum'")
    min: float | None = None
    max: float | None = None
    item_type: FieldType | None = Field(default=None, description="Element type when type == 'list'")

    @field_validator("enum_values")
    @classmethod
    def _enum_requires_values(cls, v, info):
        if info.data.get("type") == FieldType.ENUM and not v:
            raise ValueError("type 'enum' requires enum_values")
        return v


# ---------------------------------------------------------------------------
# Prohibited actions — structured, machine-checkable
# ---------------------------------------------------------------------------

class ProhibitedAction(BaseModel):
    description: str
    boundary_type: BoundaryType = BoundaryType.SELF_RESTRICTION
    # When boundary_type == CROSS_AGENT_BOUNDARY, name the agent(s) whose
    # scope this protocol must not encroach on. Enables automated checks
    # like "pathology output must not contain a systemic-therapy recommendation".
    protects_scope_of: list[str] | None = Field(
        default=None,
        description="agent_role values this restriction protects, e.g. ['medical_oncology']",
    )


# ---------------------------------------------------------------------------
# Shared protocols (definitional or behavioral, cross-specialty)
# ---------------------------------------------------------------------------

class TermDefinition(BaseModel):
    """One entry in a shared vocabulary (a TNM category, a biomarker value, etc)."""
    term: str
    definition: str
    valid_values: list[str] | None = Field(
        default=None,
        description="Closed set of acceptable values, if this term is an enumerable field "
                    "(e.g. ER_status -> ['positive','negative','equivocal','unknown'])",
    )
    source_id: str
    source_excerpt: str | None = None


class SharedProtocolMetadata(BaseModel):
    protocol_id: str
    protocol_name: str
    version: str
    schema_version: str
    status: ProtocolStatus
    kind: SharedProtocolKind
    # Shared protocols are intentionally NOT cancer-type-scoped by default —
    # that's the point. A cancer_type field here would be a smell unless a
    # concept is genuinely cancer-specific but still cross-specialty
    # (e.g. breast-specific staging nuances shared by Radiology + Pathology).
    cancer_type: str | None = Field(
        default=None,
        description="Only set if this shared protocol is cancer-specific "
                    "but still cross-specialty. Leave null for fully generic "
                    "concepts (e.g. ECOG, generic validation rules).",
    )
    population_context: str | None = None


class SharedProtocol(BaseModel):
    """
    A reusable clinical concept module consumed by multiple specialty
    protocols. Contains no specialty-specific reasoning.

    DEFINITIONAL kind populates `definitions`.
    BEHAVIORAL kind populates `rules` (same Rule shape as AgentProtocol,
    but rule_type is still clinical/safety — the difference is scope,
    not shape).
    """
    metadata: SharedProtocolMetadata

    definitions: list[TermDefinition] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)

    @field_validator("definitions")
    @classmethod
    def _definitional_only(cls, v, info):
        kind = info.data.get("metadata").kind if info.data.get("metadata") else None
        if v and kind == SharedProtocolKind.BEHAVIORAL:
            raise ValueError("BEHAVIORAL shared protocols must not populate `definitions`; use `rules`")
        return v

    @field_validator("rules")
    @classmethod
    def _behavioral_only(cls, v, info):
        kind = info.data.get("metadata").kind if info.data.get("metadata") else None
        if v and kind == SharedProtocolKind.DEFINITIONAL:
            raise ValueError("DEFINITIONAL shared protocols must not populate `rules`; use `definitions`")
        return v


# ---------------------------------------------------------------------------
# Top-level specialty protocol
# ---------------------------------------------------------------------------

class ProtocolMetadata(BaseModel):
    protocol_id: str
    protocol_name: str
    version: str = Field(..., description="Clinical content version, owned by protocol authors")
    schema_version: str = Field(..., description="Version of THIS schema the file conforms to")
    status: ProtocolStatus
    cancer_type: str
    agent_role: str
    population_context: str | None = Field(
        default=None,
        description="e.g. 'india' — signals which guideline set (Plan A vs Plan B) is authoritative",
    )
    extends: list[str] = Field(
        default_factory=list,
        description="protocol_id values of SharedProtocol modules this specialty protocol "
                    "depends on and inherits definitions/rules from. Resolved at load time "
                    "by the ProtocolLoader/composer — never inlined into this file.",
    )


class AgentProtocol(BaseModel):
    metadata: ProtocolMetadata

    responsibilities: list[str]
    prohibited_actions: list[ProhibitedAction]

    required_inputs: list[str]
    conditional_inputs: list[str] = Field(default_factory=list)

    # Specialty-owned rules ONLY. Anything reusable across specialties
    # belongs in a SharedProtocol referenced via metadata.extends, not here.
    clinical_rules: list[Rule] = Field(default_factory=list)
    safety_rules: list[Rule] = Field(default_factory=list)

    confidence_logic: ConfidenceLogic

    output_schema: dict[str, OutputField]

    @field_validator("clinical_rules")
    @classmethod
    def _clinical_rules_typed_correctly(cls, v: list[Rule]):
        for r in v:
            if r.rule_type != RuleType.CLINICAL:
                raise ValueError(f"rule {r.rule_id} in clinical_rules must have rule_type=clinical")
        return v

    @field_validator("safety_rules")
    @classmethod
    def _safety_rules_typed_correctly(cls, v: list[Rule]):
        for r in v:
            if r.rule_type != RuleType.SAFETY:
                raise ValueError(f"rule {r.rule_id} in safety_rules must have rule_type=safety")
        return v
