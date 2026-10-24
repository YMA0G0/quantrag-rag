"""Tests for environment-based configuration loading."""

from __future__ import annotations

import pytest

from quantrag.config import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    ENV_API_KEY,
    ENV_MODEL,
    load_settings,
)
from quantrag.errors import ConfigurationError

ENV = {ENV_API_KEY: "secret-key", ENV_MODEL: "vendor/model"}


def test_loads_required_values() -> None:
    settings = load_settings(ENV, load_env_file=False)
    assert settings.api_key == "secret-key"
    assert settings.model == "vendor/model"
    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert settings.chat_completions_url == "https://openrouter.ai/api/v1/chat/completions"


def test_missing_api_key_raises() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings({ENV_MODEL: "vendor/model"}, load_env_file=False)
    assert ENV_API_KEY in str(excinfo.value)
    assert "secret-key" not in str(excinfo.value)


def test_missing_model_raises() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings({ENV_API_KEY: "secret-key"}, load_env_file=False)
    assert ENV_MODEL in str(excinfo.value)


def test_blank_values_are_treated_as_missing() -> None:
    with pytest.raises(ConfigurationError):
        load_settings({ENV_API_KEY: "   ", ENV_MODEL: "vendor/model"}, load_env_file=False)


def test_api_key_is_not_exposed_in_repr() -> None:
    settings = load_settings(ENV, load_env_file=False)
    assert "secret-key" not in repr(settings)
    assert "vendor/model" in repr(settings)


def test_overrides_are_applied() -> None:
    env = {
        **ENV,
        "OPENROUTER_BASE_URL": "https://example.invalid/v1/",
        "OPENROUTER_TIMEOUT_SECONDS": "12.5",
        "OPENROUTER_APP_NAME": "QuantRAG Dev",
        "OPENROUTER_APP_URL": "https://example.invalid",
    }
    settings = load_settings(env, load_env_file=False)
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.timeout_seconds == 12.5
    assert settings.app_name == "QuantRAG Dev"
    assert settings.app_url == "https://example.invalid"
    assert settings.chat_completions_url == "https://example.invalid/v1/chat/completions"


def test_invalid_timeout_raises() -> None:
    with pytest.raises(ConfigurationError):
        load_settings({**ENV, "OPENROUTER_TIMEOUT_SECONDS": "not-a-number"}, load_env_file=False)
    with pytest.raises(ConfigurationError):
        load_settings({**ENV, "OPENROUTER_TIMEOUT_SECONDS": "0"}, load_env_file=False)


def test_max_tokens_defaults_to_1024() -> None:
    settings = load_settings(ENV, load_env_file=False)
    assert settings.max_tokens == 1024


def test_max_tokens_override_is_applied() -> None:
    settings = load_settings({**ENV, "OPENROUTER_MAX_TOKENS": "64"}, load_env_file=False)
    assert settings.max_tokens == 64


def test_invalid_max_tokens_raises() -> None:
    with pytest.raises(ConfigurationError):
        load_settings({**ENV, "OPENROUTER_MAX_TOKENS": "many"}, load_env_file=False)
    with pytest.raises(ConfigurationError):
        load_settings({**ENV, "OPENROUTER_MAX_TOKENS": "0"}, load_env_file=False)
    with pytest.raises(ConfigurationError):
        load_settings({**ENV, "OPENROUTER_MAX_TOKENS": "-5"}, load_env_file=False)


def test_dotenv_file_is_loaded(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"{ENV_API_KEY}=dotenv-key\n{ENV_MODEL}=dotenv/model\n", encoding="utf-8")
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)

    settings = load_settings(dotenv_path=env_file)
    assert settings.api_key == "dotenv-key"
    assert settings.model == "dotenv/model"


def test_existing_environment_wins_over_dotenv(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"{ENV_API_KEY}=dotenv-key\n", encoding="utf-8")
    monkeypatch.setenv(ENV_API_KEY, "real-env-key")
    monkeypatch.setenv(ENV_MODEL, "vendor/model")

    settings = load_settings(dotenv_path=env_file)
    assert settings.api_key == "real-env-key"
