from agents.triage import triage


def test_emergency_keyword_is_escalated() -> None:
    result = triage({"question": "I have chest pain and difficulty breathing"})["triage"]
    assert result["confidence"] >= 0.9
    assert "emergency" in result["recommendations"][0].lower()
