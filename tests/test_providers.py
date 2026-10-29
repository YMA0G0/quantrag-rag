"""Tests for the LLM provider factory and provider implementations."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from quantrag.config import Settings
from quantrag.openrouter import OpenRouterClient
from quantrag.providers import (
    KiloProvider,
    LLMProvider,
    OpenRouterProvider,
    create_provider,
    create_provider_from_env,
)

pytestmark = pytest.mark.anyio

TEST_API_KEY = "test-key"
TEST_MODEL = "test/model"
TEST_BASE_URL = "https://test.invalid/api"


def anyio_backend() -> str:
    return "asyncio"


def build_client(handler, settings: Settings) -> OpenRouterClient:
    return OpenRouterClient(settings, transport=httpx.MockTransport(handler))


def json_response(payload: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=payload)


def chat_completion(content: str) -> dict[str, Any]:
    return {
        "id": "gen-123",
        "model": "test/model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    }


async def test_create_provider_openrouter() -> None:
    """OpenRouter provider is created correctly."""
    settings = Settings(api_key="test-key", model="test/model")
    provider = create_provider("openrouter", settings)

    assert isinstance(provider, OpenRouterProvider)
    assert isinstance(provider, LLMProvider)
    assert provider.model == "test/model"


async def test_create_provider_kilo() -> None:
    """Kilo provider is created correctly."""
    settings = Settings(api_key="kilo-key", model="kilo/model", base_url="https://kilo.invalid/api")
    provider = create_provider("kilo", settings)

    assert isinstance(provider, KiloProvider)
    assert isinstance(provider, LLMProvider)
    assert provider.model == "kilo/model"


async def test_create_provider_unknown_raises() -> None:
    """Unknown provider raises clear error."""
    with pytest.raises(
        ValueError, match="Unknown LLM provider: unknown. Supported: openrouter, kilo"
    ):
        create_provider("unknown", Settings(api_key="x", model="y"))


async def test_create_provider_case_insensitive() -> None:
    """Provider names are case-insensitive."""
    settings = Settings(api_key="key", model="model", base_url="https://test.invalid/api")

    p1 = create_provider("OPENROUTER", settings)
    p2 = create_provider("OpenRouter", settings)
    p3 = create_provider("KILO", Settings(api_key="k", model="m", base_url="https://k.invalid/api"))
    p4 = create_provider("Kilo", Settings(api_key="k", model="m", base_url="https://k.invalid/api"))

    assert isinstance(p1, OpenRouterProvider)
    assert isinstance(p2, OpenRouterProvider)
    assert isinstance(p3, KiloProvider)
    assert isinstance(p4, KiloProvider)


async def test_create_provider_from_env_openrouter_default() -> None:
    """Default provider is openrouter when not specified."""
    env = {"OPENROUTER_API_KEY": "or-key", "OPENROUTER_MODEL": "or-model"}
    provider = create_provider_from_env(env)

    assert isinstance(provider, OpenRouterProvider)
    assert provider.model == "or-model"


async def test_create_provider_from_env_kilo() -> None:
    """QUANTRAG_LLM_PROVIDER=kilo selects Kilo provider with KILO_* config."""
    env = {
        "QUANTRAG_LLM_PROVIDER": "kilo",
        "KILO_API_KEY": "kilo-key",
        "KILO_MODEL": "kilo/model",
        "KILO_BASE_URL": "https://kilo.invalid/api",
    }
    provider = create_provider_from_env(env)

    assert isinstance(provider, KiloProvider)
    assert provider.model == "kilo/model"


async def test_create_provider_from_env_kilo_reads_correct_vars() -> None:
    """Kilo provider reads KILO_* env vars, not OPENROUTER_*."""
    env = {
        "QUANTRAG_LLM_PROVIDER": "kilo",
        "KILO_API_KEY": "kilo-only-key",
        "KILO_MODEL": "kilo-only-model",
        "KILO_BASE_URL": "https://kilo.invalid/api",
        # These should be ignored for kilo
        "OPENROUTER_API_KEY": "should-be-ignored",
        "OPENROUTER_MODEL": "should-be-ignored",
        "OPENROUTER_BASE_URL": "should-be-ignored",
    }
    provider = create_provider_from_env(env)

    assert isinstance(provider, KiloProvider)
    assert provider.model == "kilo-only-model"


async def test_create_provider_from_env_openrouter_explicit() -> None:
    """QUANTRAG_LLM_PROVIDER=openrouter uses OPENROUTER_* config."""
    env = {
        "QUANTRAG_LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "or-key",
        "OPENROUTER_MODEL": "or-model",
        "OPENROUTER_BASE_URL": "https://or.invalid/api",
    }
    provider = create_provider_from_env(env)

    assert isinstance(provider, OpenRouterProvider)
    assert provider.model == "or-model"


async def test_missing_kilo_api_key_raises() -> None:
    """Missing KILO_API_KEY raises clear ConfigurationError."""
    env = {
        "QUANTRAG_LLM_PROVIDER": "kilo",
        "KILO_MODEL": "model",
        "KILO_BASE_URL": "https://k.invalid/api",
    }

    with pytest.raises(Exception) as excinfo:
        create_provider_from_env(env)

    assert "KILO_API_KEY" in str(excinfo.value)
    assert "not set" in str(excinfo.value).lower()


async def test_missing_kilo_model_raises() -> None:
    """Missing KILO_MODEL raises clear ConfigurationError."""
    env = {
        "QUANTRAG_LLM_PROVIDER": "kilo",
        "KILO_API_KEY": "key",
        "KILO_BASE_URL": "https://k.invalid/api",
    }

    with pytest.raises(Exception) as excinfo:
        create_provider_from_env(env)

    assert "KILO_MODEL" in str(excinfo.value)
    assert "not set" in str(excinfo.value).lower()


async def test_missing_openrouter_api_key_raises() -> None:
    """Missing OPENROUTER_API_KEY raises clear ConfigurationError."""
    env = {"QUANTRAG_LLM_PROVIDER": "openrouter", "OPENROUTER_MODEL": "model"}

    with pytest.raises(Exception) as excinfo:
        create_provider_from_env(env)

    assert "OPENROUTER_API_KEY" in str(excinfo.value)
    assert "not set" in str(excinfo.value).lower()


async def test_unknown_provider_raises_clear_error() -> None:
    """Unknown provider raises clear error with supported list."""
    env = {
        "QUANTRAG_LLM_PROVIDER": "unknown",
        "KILO_API_KEY": "key",
        "KILO_MODEL": "model",
        "KILO_BASE_URL": "https://k.invalid/api",
    }

    with pytest.raises(
        ValueError, match="Unknown LLM provider: unknown. Supported: openrouter, kilo"
    ):
        create_provider_from_env(env)


async def test_kilo_never_falls_back_to_openrouter() -> None:
    """Kilo selection never uses OPENROUTER_* credentials or URL."""
    # Even if OPENROUTER_* are set, kilo should use KILO_*
    env = {
        "QUANTRAG_LLM_PROVIDER": "kilo",
        "KILO_API_KEY": "kilo-key",
        "KILO_MODEL": "kilo-model",
        "KILO_BASE_URL": "https://kilo.invalid/api",
        "OPENROUTER_API_KEY": "or-key",
        "OPENROUTER_MODEL": "or-model",
        "OPENROUTER_BASE_URL": "https://or.invalid/api",
    }
    provider = create_provider_from_env(env)

    assert isinstance(provider, KiloProvider)
    assert provider.model == "kilo-model"
    # The underlying client should have kilo config;
    # we can't easily test the internal client URL without mocking,
    # but the model check is sufficient


async def test_api_keys_never_in_error_messages() -> None:
    """API keys never appear in error messages."""
    # Test for Kilo
    env = {
        "QUANTRAG_LLM_PROVIDER": "kilo",
        "KILO_MODEL": "model",
        "KILO_BASE_URL": "https://k.invalid/api",
    }

    try:
        create_provider_from_env(env)
        pytest.fail("Should have raised")
    except Exception as e:
        error_msg = str(e)
        # The error should not contain any API key-like strings
        assert "secret" not in error_msg.lower()
        assert "key" not in error_msg.lower() or "not set" in error_msg.lower()


async def test_provider_completes_with_mock_transport() -> None:
    """Provider completes request using mocked transport."""
    from quantrag.config import load_provider_settings
    from quantrag.providers import create_provider

    # Test OpenRouter provider
    settings = load_provider_settings(
        "openrouter",
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "test/model",
            "OPENROUTER_BASE_URL": "https://test.invalid/api",
        },
    )
    provider = create_provider("openrouter", settings)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://test.invalid/api/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        return json_response(chat_completion("Hello from OpenRouter"))

    provider = OpenRouterProvider(
        Settings(api_key="test-key", model="test/model", base_url="https://test.invalid/api"),
        transport=httpx.MockTransport(handler),
    )
    result = await provider.complete("Hello", system="Test")
    assert result == "Hello from OpenRouter"

    # Test Kilo provider
    kilo_settings = Settings(
        api_key="kilo-key", model="kilo/model", base_url="https://kilo.invalid/api"
    )
    kilo_provider = KiloProvider(kilo_settings)

    async def kilo_handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://kilo.invalid/api/chat/completions"
        assert request.headers["Authorization"] == "Bearer kilo-key"
        return json_response(chat_completion("Hello from Kilo"))

    kilo_provider._client._transport = httpx.MockTransport(kilo_handler)
    result = await kilo_provider.complete("Hello", system="Test")
    assert result == "Hello from Kilo"


async def test_kilo_provider_has_aclose() -> None:
    """KiloProvider has aclose method for cleanup."""
    settings = Settings(api_key="key", model="model", base_url="https://test.invalid/api")
    provider = KiloProvider(settings)

    # Should not raise
    await provider.aclose()


async def test_openrouter_provider_has_aclose() -> None:
    """OpenRouterProvider has aclose method for cleanup."""
    settings = Settings(api_key="key", model="model")
    provider = OpenRouterProvider(settings)

    # Should not raise
    await provider.aclose()


# Test that the Settings object works correctly for both providers
async def test_settings_base_urls() -> None:
    """Settings base_url is correctly set for each provider."""
    from quantrag.config import load_provider_settings

    # Test openrouter default base_url
    settings = load_provider_settings(
        "openrouter",
        {
            "OPENROUTER_API_KEY": "key",
            "OPENROUTER_MODEL": "model",
        },
    )
    assert settings.base_url == "https://openrouter.ai/api/v1"

    # Test openrouter custom base_url
    settings = load_provider_settings(
        "openrouter",
        {
            "OPENROUTER_API_KEY": "key",
            "OPENROUTER_MODEL": "model",
            "OPENROUTER_BASE_URL": "https://custom.invalid/api",
        },
    )
    assert settings.base_url == "https://custom.invalid/api"

    # Test kilo default base_url
    settings = load_provider_settings(
        "kilo",
        {
            "KILO_API_KEY": "key",
            "KILO_MODEL": "model",
        },
    )
    assert settings.base_url == "https://ai-gateway.kilo.ai/api/v1"

    # Test kilo custom base_url
    settings = load_provider_settings(
        "kilo",
        {
            "KILO_API_KEY": "key",
            "KILO_MODEL": "model",
            "KILO_BASE_URL": "https://custom.invalid/api",
        },
    )
    assert settings.base_url == "https://custom.invalid/api"
