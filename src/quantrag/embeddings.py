"""OpenRouter Embeddings client.

Turns text into vectors for the retriever.  Like :mod:`quantrag.openrouter`
it contains only HTTP logic and knows nothing about MCP or vector search, so
it can be tested with a mocked transport and swapped for another provider.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Protocol

import httpx

from .config import ENV_API_KEY, Settings
from .errors import (
    AuthenticationError,
    ConfigurationError,
    MalformedResponseError,
    RateLimitError,
    UpstreamConnectionError,
    UpstreamStatusError,
    UpstreamTimeoutError,
    redact,
)

logger = logging.getLogger(__name__)

ENV_EMBEDDING_MODEL = "OPENROUTER_EMBEDDING_MODEL"
MAX_ERROR_DETAIL_CHARS = 300


class Embedder(Protocol):
    """Minimal interface used by the retriever."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, in the same order."""
        ...


def load_embedding_model(env: Mapping[str, str]) -> str:
    """Read the embedding model id from the environment (never hardcoded)."""
    model = (env.get(ENV_EMBEDDING_MODEL) or "").strip()
    if not model:
        raise ConfigurationError(
            f"{ENV_EMBEDDING_MODEL} is not set. Set it to an OpenRouter embedding "
            "model id that is available on your account."
        )
    return model


class OpenRouterEmbedder:
    """Thin async client for the OpenRouter Embeddings API."""

    def __init__(
        self,
        settings: Settings,
        model: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` and return the vectors in input order.

        Raises:
            UpstreamTimeoutError / UpstreamConnectionError / AuthenticationError /
            RateLimitError / UpstreamStatusError / MalformedResponseError
        """
        if not texts:
            return []
        payload = {"model": self._model, "input": texts}
        try:
            async with httpx.AsyncClient(
                timeout=self._settings.timeout_seconds, transport=self._transport
            ) as client:
                response = await client.post(self._url, json=payload, headers=self._build_headers())
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                f"Embeddings request timed out after {self._settings.timeout_seconds:g}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamConnectionError(
                f"Could not reach OpenRouter at {self._settings.base_url!r}: {type(exc).__name__}."
            ) from exc
        return self._parse_response(response, expected=len(texts))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @property
    def _url(self) -> str:
        return f"{self._settings.base_url.rstrip('/')}/embeddings"

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

    def _parse_response(self, response: httpx.Response, *, expected: int) -> list[list[float]]:
        status = response.status_code
        if status in (401, 403):
            detail = self._error_detail(response)
            suffix = f": {detail}" if detail else "."
            raise AuthenticationError(
                f"OpenRouter rejected the API key (HTTP {status}){suffix} Check {ENV_API_KEY}.",
                status,
            )
        if status == 429:
            raise RateLimitError("OpenRouter rate limit reached (HTTP 429). Retry later.", status)
        if status >= 400:
            detail = self._error_detail(response)
            suffix = f": {detail}" if detail else "."
            raise UpstreamStatusError(
                f"OpenRouter embeddings request failed with HTTP {status}{suffix}", status
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise MalformedResponseError(
                "OpenRouter returned a response that is not valid JSON."
            ) from exc

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != expected:
            raise MalformedResponseError(
                f"OpenRouter embeddings response did not contain {expected} vectors."
            )

        vectors: list[list[float]] = []
        # ``index`` tells us which input each vector belongs to; sort defensively.
        for item in sorted(items, key=lambda i: i.get("index", 0) if isinstance(i, dict) else 0):
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not vector:
                raise MalformedResponseError("OpenRouter returned a malformed embedding entry.")
            try:
                vectors.append([float(x) for x in vector])
            except (TypeError, ValueError) as exc:
                raise MalformedResponseError("Embedding contained non-numeric values.") from exc
        return vectors

    def _error_detail(self, response: httpx.Response) -> str:
        """Extract a short, credential-free error message from a failed call."""
        try:
            body: Any = response.json()
        except (json.JSONDecodeError, ValueError):
            body = None
        detail = ""
        if isinstance(body, dict):
            error = body.get("error")
            raw = error.get("message") if isinstance(error, dict) else error
            if isinstance(raw, str):
                detail = raw
        if not detail:
            detail = (response.text or "").strip()
        detail = redact(detail, self._settings.api_key).strip()
        if len(detail) > MAX_ERROR_DETAIL_CHARS:
            detail = detail[:MAX_ERROR_DETAIL_CHARS] + "..."
        return detail
