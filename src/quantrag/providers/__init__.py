"""LLM Provider abstraction for QuantRAG.

This module defines the provider interface and factory for creating
LLM clients. The application-side LLM (QuantRAG internal) is completely
decoupled from the agent-side LLM (Pi/MCP client).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from ..config import Settings
from ..openrouter import ChatCompleter

__all__ = [
    "ChatCompleter",
    "LLMProvider",
    "OpenRouterProvider",
    "KiloProvider",
    "create_provider",
    "create_provider_from_env",
]


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Each provider implements the ChatCompleter protocol and handles
    its own authentication, request/response formatting, and error mapping.
    """

    @abstractmethod
    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the assistant's text answer for ``prompt``."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """The configured model identifier."""
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Close any underlying connections."""
        ...


class OpenRouterProvider(LLMProvider):
    """OpenRouter Chat Completions provider."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Any = None,
        client: Any = None,
    ) -> None:
        from ..openrouter import OpenRouterClient

        self._client = OpenRouterClient(settings, transport=transport, client=client)

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        return await self._client.complete(prompt, system=system)

    @property
    def model(self) -> str:
        return self._client.model

    async def aclose(self) -> None:
        await self._client.aclose()


class KiloProvider(LLMProvider):
    """Kilo AI Gateway Chat Completions provider.

    Uses the same OpenAI-compatible API as OpenRouter but with
    Kilo-specific configuration and error handling.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Any = None,
        client: Any = None,
    ) -> None:
        from ..openrouter import OpenRouterClient  # Reuse OpenAI-compatible client

        self._client = OpenRouterClient(settings, transport=transport, client=client)

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        return await self._client.complete(prompt, system=system)

    @property
    def model(self) -> str:
        return self._client.model

    async def aclose(self) -> None:
        await self._client.aclose()


def create_provider(
    provider_name: str,
    settings: Settings,
    *,
    transport: Any = None,
    client: Any = None,
) -> LLMProvider:
    """Factory function to create an LLM provider by name.

    Args:
        provider_name: Provider identifier ("openrouter", "kilo", etc.)
        settings: Application settings containing API keys and model config
        transport: Optional custom HTTP transport
        client: Optional custom HTTP client

    Returns:
        An LLMProvider instance implementing ChatCompleter.

    Raises:
        ValueError: If provider_name is unknown.
        ConfigurationError: If required settings are missing.
    """
    provider_name = provider_name.lower()
    if provider_name == "openrouter":
        return OpenRouterProvider(settings, transport=transport, client=client)
    if provider_name == "kilo":
        return KiloProvider(settings, transport=transport, client=client)

    # Future providers can be added here:
    # if provider_name == "anthropic":
    #     return AnthropicProvider(settings, transport=transport, client=client)
    # if provider_name == "local":
    #     return LocalProvider(settings, transport=transport, client=client)

    raise ValueError(f"Unknown LLM provider: {provider_name}. Supported: openrouter, kilo")


def create_provider_from_env(
    env: Mapping[str, str] | None = None,
    *,
    transport: Any = None,
    client: Any = None,
) -> LLMProvider:
    """Create a provider from environment variables.

    Reads QUANTRAG_LLM_PROVIDER (default: openrouter) and corresponding
    API key / model settings from the environment.

    Args:
        env: Environment mapping (defaults to os.environ)
        transport: Optional custom HTTP transport
        client: Optional custom HTTP client

    Returns:
        An LLMProvider instance.

    Raises:
        ConfigurationError: If required configuration is missing.
        ValueError: If provider is unknown.
    """
    from ..config import load_provider_settings

    env = env or os.environ
    provider_name = env.get("QUANTRAG_LLM_PROVIDER", "openrouter").lower()
    settings = load_provider_settings(provider_name, env)
    return create_provider(provider_name, settings, transport=transport, client=client)
