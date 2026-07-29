from backend.fastapi import AskRequest


def test_ask_request_accepts_ui_payload() -> None:
    request = AskRequest.model_validate({"question": "What are common cold symptoms?", "model": "meditron:7b"})
    assert request.model == "meditron:7b"
