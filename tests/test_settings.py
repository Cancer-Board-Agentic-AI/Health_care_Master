from config.settings import get_settings


def test_local_configuration_is_loaded() -> None:
    assert get_settings().models["primary"] == "qwen3:14b"
