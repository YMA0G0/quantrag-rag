"""OpenRouter Chat Completions client.

This module contains all HTTP logic for QuantRAG.  It is deliberately kept
independent of the MCP layer so that it can be unit tested with a mocked
transport and reused later by other transports (HTTP/streamable, cloud).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

import httpx

from .config import ENV_API_KEY, Settings
from .errors import (
    AuthenticationError,
    MalformedResponseError,
    RateLimitError,
    UpstreamConnectionError,
    UpstreamStatusError,
    UpstreamTimeoutError,
    redact,
)

logger = logging.getLogger(__name__)

MAX_ERROR_DETAIL_CHARS = 300


class ChatCompleter(Protocol):
    """Minimal interface used by the MCP tool handler."""

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the assistant's text answer for ``prompt``."""
        ...


class OpenRouterClient:
    """Thin async client for the OpenRouter Chat Completions API."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def model(self) -> str:
        """The configured model id (never hardcoded)."""
        return self._settings.model

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Send ``prompt`` to the configured model and return its text reply.

        Raises:
            UpstreamTimeoutError: The request timed out.
            UpstreamConnectionError: The provider could not be reached.
            AuthenticationError: The API key was rejected.
            RateLimitError: The provider rate limit was hit.
            UpstreamStatusError: The provider returned an error status.
            MalformedResponseError: The response body was not understood.
        """
        payload = self._build_payload(prompt, system=system)
        headers = self._build_headers()

        try:
            response = await self._post(payload, headers)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                f"Request to OpenRouter timed out after {self._settings.timeout_seconds:g}s. "
                "Retry with a shorter question or increase OPENROUTER_TIMEOUT_SECONDS."
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamConnectionError(
                f"Could not reach OpenRouter at {self._settings.base_url!r}: {type(exc).__name__}."
            ) from exc

        return self._parse_response(response)

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this instance owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _build_payload(self, prompt: str, *, system: str | None) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self._settings.model,
            "messages": messages,
            "max_tokens": self._settings.max_tokens,
            "stream": False,
        }

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Title": self._settings.app_name,
        }
        if self._settings.app_url:
            headers["HTTP-Referer"] = self._settings.app_url
        return headers

    async def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        url = self._settings.chat_completions_url
        if self._client is not None:
            return await self._client.post(url, json=payload, headers=headers)
        async with httpx.AsyncClient(
            timeout=self._settings.timeout_seconds,
            transport=self._transport,
        ) as client:
            return await client.post(url, json=payload, headers=headers)

    def _parse_response(self, response: httpx.Response) -> str:
        status = response.status_code
        if status in (401, 403):
            detail = self._error_detail(response)
            suffix = f": {detail}" if detail else "."
            raise AuthenticationError(
                f"OpenRouter rejected the API key (HTTP {status}){suffix} Check {ENV_API_KEY}.",
                status,
            )
        if status == 429:
            raise RateLimitError(
                "OpenRouter rate limit reached (HTTP 429). "
                f"Retry later{self._retry_after_hint(response)}.",
                status,
                retry_after=self._retry_after_seconds(response),
            )
        if status >= 400:
            detail = self._error_detail(response)
            suffix = f": {detail}" if detail else "."
            raise UpstreamStatusError(
                f"OpenRouter request failed with HTTP {status}{suffix}", status
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise MalformedResponseError(
                "OpenRouter returned a response that is not valid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise MalformedResponseError("OpenRouter returned an unexpected JSON payload.")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise MalformedResponseError("OpenRouter response did not contain any choices.")

        first = choices[0]
        if not isinstance(first, dict):
            raise MalformedResponseError("OpenRouter response contained a malformed choice entry.")

        message = first.get("message")
        if not isinstance(message, dict):
            raise MalformedResponseError("OpenRouter response choice did not contain a message.")

        content = message.get("content")
        if not isinstance(content, str):
            raise MalformedResponseError(
                "OpenRouter response did not contain a text answer "
                f"(content type: {type(content).__name__})."
            )
        return content.strip()

    def _error_detail(self, response: httpx.Response) -> str:
        """Extract a short, credential-free error message from a failed call."""
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = None
        detail = ""
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                raw = error.get("message")
            elif isinstance(error, str):
                raw = error
            else:
                raw = body.get("message")
            if isinstance(raw, str):
                detail = raw
        if not detail:
            detail = (response.text or "").strip()
        detail = redact(detail, self._settings.api_key).strip()
        if len(detail) > MAX_ERROR_DETAIL_CHARS:
            detail = detail[:MAX_ERROR_DETAIL_CHARS] + "..."
        return detail

    def _retry_after_seconds(self, response: httpx.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _retry_after_hint(self, response: httpx.Response) -> str:
        seconds = self._retry_after_seconds(response)
        return f" after about {seconds:g}s" if seconds is not None else ""
