import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents.report import BOARD_SECTIONS, _synthesise_board_plan, report


def _plan() -> dict:
    return {
        "case_summary": "Operable breast cancer; complete biomarker confirmation first.",
        "pretreatment_confirmation": ["Confirm core biopsy, ER/PR/HER2, and clinical nodal status."],
        "initial_treatment_decision": [
            "Preferred: upfront surgery after confirmation.",
            "If disease is not operable, then use subtype-directed neoadjuvant treatment.",
        ],
        "breast_and_axillary_surgery": [
            "Breast: breast-conserving surgery if clear margins are feasible.",
            "Axilla: sentinel-node staging if clinically node negative.",
        ],
        "postoperative_systemic_treatment": ["Select adjuvant therapy from final pathology and biomarkers."],
        "radiation": ["Base radiation fields on operation and final nodal/pathologic risk."],
        "supportive_and_access": ["Assess fitness, preferences, cost, travel, and adherence barriers."],
        "decision_dependencies": ["Chemotherapy depends on final pathologic risk."],
        "unresolved_disagreements": [],
    }


def test_chair_synthesis_returns_one_sequenced_pathway() -> None:
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content=json.dumps(_plan()))
    state = {
        "question": "Case details",
        "surgery": {"recommended_plan": ["Upfront surgery"], "confidence": 0.8},
        "medical_oncology": {
            "recommended_plan": ["Neoadjuvant endocrine therapy", "Adjuvant chemotherapy"],
            "confidence": 0.7,
        },
    }

    with patch("agents.report.get_llm", return_value=llm):
        result = _synthesise_board_plan(state)

    assert result["initial_treatment_decision"][0].startswith("Preferred:")
    assert all(key in result for key, _ in BOARD_SECTIONS)
    system_prompt = llm.invoke.call_args.args[0][0].content
    assert "mutually exclusive starting strategies" in system_prompt
    assert "Do not concatenate" in system_prompt


def test_invalid_coequal_starting_strategies_fail_safe() -> None:
    invalid = _plan()
    invalid["initial_treatment_decision"] = ["Upfront surgery", "Neoadjuvant endocrine therapy"]
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content=json.dumps(invalid))

    with patch("agents.report.get_llm", return_value=llm):
        result = _synthesise_board_plan({"question": "Case details"})

    assert result["case_summary"].startswith("Chair synthesis was unavailable")
    assert "No preferred starting strategy" in result["initial_treatment_decision"][0]


def test_report_uses_six_tumor_board_sections() -> None:
    ideal = _plan()
    with patch("agents.report._synthesise_board_plan", return_value=ideal):
        result = report({
            "question": "Case details",
            "consensus_index": 0.82,
            "consensus_notes": "A deliberately verbose internal consensus explanation.",
        })

    assert result["board_plan"] == ideal
    assert "**82% — High consensus.**" in result["report"]
    assert "deliberately verbose" not in result["report"]
    assert "## Specialty Assessments" not in result["report"]
    assert "### B. Resource-Adapted Plan" not in result["report"]
    assert "## Data Gaps / Missing Information" not in result["report"]
    for _, heading in BOARD_SECTIONS:
        assert heading in result["report"]
    assert "Decisions Pending Additional Results" in result["report"]
