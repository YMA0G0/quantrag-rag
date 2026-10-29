"""FastAPI application sharing the same RagService as the MCP server.

Exposes ``/health``, ``/ask`` and ``/search``.  All three endpoints use the
single :class:`quantrag.service.RagService` instance so retrieval and answering
are consistent across transports.

The MCP Streamable HTTP endpoint is mounted at ``/mcp`` and shares the same
RagService instance.  ``/health`` remains public; ``/ask``, ``/search`` and
``/mcp`` require a valid ``X-API-Key`` when ``QUANTRAG_API_KEYS`` is configured.
"""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, status
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field, field_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import RagSettings, load_auth_settings, load_rag_settings
from .errors import (
    ConfigurationError,
    RateLimitError,
    RetrievalError,
    UpstreamError,
    UpstreamStatusError,
    UpstreamTimeoutError,
)
from .server import create_server
from .service import RagService, build_service

logger = logging.getLogger("quantrag")

_SERVICE: RagService | None = None
_RAG_SETTINGS: RagSettings | None = None
# Populated by the lifespan even when the full RAG config fails to load, so
# that authentication never depends on retrieval settings being valid.
_API_KEYS: frozenset[str] = frozenset()
_REQUIRE_AUTH = False


async def get_service() -> RagService:
    """Lazy singleton: build the service on first use."""
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE
    _SERVICE = await build_service()
    return _SERVICE


def _get_rag_settings() -> RagSettings | None:
    """Return the loaded RagSettings (may be None before startup)."""
    return _RAG_SETTINGS


def _effective_api_keys() -> frozenset[str]:
    """Return the active API keys.

    Settings win when present; the auth-only load is the fallback so a broken
    retrieval configuration cannot turn ``/mcp`` into a public endpoint.
    """
    settings = _get_rag_settings()
    if settings is not None and settings.api_keys:
        return settings.api_keys
    return _API_KEYS


async def _validate_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    """Validate the X-API-Key header against registered keys.

    - If ``QUANTRAG_API_KEYS`` is configured, missing/invalid keys give 401.
    - If no keys are configured (local development), allow the request through
      so that /health and local use cases still work.
    - Keys are compared with secrets.compare_digest to avoid timing attacks.
    """
    api_keys = _effective_api_keys()
    if not api_keys:
        # No keys configured: allow through (local development mode)
        return

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    # Check if the provided key matches any of the registered keys
    # Use secrets.compare_digest for timing-safe comparison
    for registered_key in api_keys:
        if secrets.compare_digest(x_api_key, registered_key):
            return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid X-API-Key",
    )


class _MCPAuthMiddleware:
    """ASGI wrapper that enforces X-API-Key authentication for the MCP app.

    The wrapped Starlette app is created eagerly in ``__init__`` so that the
    MCP session manager exists before the FastAPI lifespan starts it.  The
    session task group is owned by the lifespan task (see
    :func:`_mcp_session_lifespan`), never by the HTTP request that happens to
    arrive first.

    When ``QUANTRAG_API_KEYS`` is not configured, requests pass through
    unauthenticated (local development mode).
    """

    def __init__(self, mcp_server) -> None:
        self.mcp_server = mcp_server
        # Creating the Starlette app also creates the session manager.
        self._app: ASGIApp = mcp_server.streamable_http_app()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        api_keys = _effective_api_keys()
        if api_keys:
            headers = dict(scope.get("headers", []))
            x_api_key = headers.get(b"x-api-key", b"").decode()
            if not x_api_key:
                await self._send_json(send, 401, {"detail": "Missing X-API-Key header"})
                return
            if not any(secrets.compare_digest(x_api_key, k) for k in api_keys):
                await self._send_json(send, 401, {"detail": "Invalid X-API-Key"})
                return

        await self._app(scope, receive, send)

    @staticmethod
    async def _send_json(send: Send, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})


@asynccontextmanager
async def _mcp_session_lifespan(mcp_server) -> AsyncIterator[None]:
    """Own the MCP session manager's task group for the app's lifetime.

    The SDK's ``StreamableHTTPSessionManager.run()`` can only be entered once
    per instance, which breaks repeated lifespan entry (tests, server
    reloads).  Entering an anyio task group from the *lifespan* task is the
    documented lifecycle; the only difference from ``run()`` is that a fresh
    task group is created on every entry.
    """
    manager = mcp_server.session_manager
    async with anyio.create_task_group() as task_group:
        manager._task_group = task_group
        try:
            yield
        finally:
            task_group.cancel_scope.cancel()
            manager._task_group = None
            manager._server_instances.clear()
            manager._session_owners.clear()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _API_KEYS, _RAG_SETTINGS, _REQUIRE_AUTH, _SERVICE

    # Authentication is loaded independently of the retrieval config: a broken
    # RAG setting must never silently disable access control.
    _API_KEYS, _REQUIRE_AUTH = load_auth_settings(load_env_file=True)
    if _REQUIRE_AUTH and not _API_KEYS:
        raise RuntimeError(
            "QUANTRAG_REQUIRE_AUTH is enabled but QUANTRAG_API_KEYS is empty. "
            "Refusing to start: /ask, /search and /mcp would be public."
        )

    try:
        _RAG_SETTINGS = load_rag_settings(load_env_file=True)
    except ConfigurationError as exc:
        logger.warning("QuantRAG is not fully configured: %s", exc)
        _RAG_SETTINGS = None

    async with _mcp_session_lifespan(_MCP_SERVER):
        yield

    _SERVICE = None
    _RAG_SETTINGS = None
    _API_KEYS = frozenset()
    _REQUIRE_AUTH = False


app = FastAPI(title="QuantRAG", lifespan=lifespan)

# ------------------------------------------------------------------
# Mount the MCP Streamable HTTP endpoint at /mcp
#
# streamable_http_path="/" so that the MCP route matches after FastAPI's
# mount strips the "/mcp" prefix (a request to /mcp/ arrives at the MCP
# app as path "/").
# ------------------------------------------------------------------
_MCP_SERVER = create_server(
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
app.mount("/mcp", _MCPAuthMiddleware(_MCP_SERVER))


def _non_empty(v: str) -> str:
    if not v or not v.strip():
        raise ValueError("must not be empty or whitespace")
    return v


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(None, ge=1, le=50)
    mode: str | None = Field(None, pattern="^(bm25|vector|hybrid)$")
    _validate_question = field_validator("question")(_non_empty)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int | None = Field(None, ge=1, le=50)
    mode: str | None = Field(None, pattern="^(bm25|vector|hybrid)$")
    min_score: float | None = Field(None, ge=0.0)
    _validate_query = field_validator("query")(_non_empty)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe - returns only status, no service details."""
    return {"status": "ok"}


@app.post("/ask", dependencies=[Depends(_validate_api_key)])
async def ask(payload: AskRequest) -> dict[str, Any]:
    """Answer a question using retrieval-augmented generation."""
    try:
        svc = await get_service()
        result = await svc.ask(payload.question, k=payload.k, mode=payload.mode)
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"QuantRAG is not configured: {exc}") from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UpstreamTimeoutError as exc:
        import logging

        logging.getLogger("quantrag").warning("Upstream timeout in /ask: %s", exc)
        raise HTTPException(status_code=504, detail="Upstream timeout.") from exc
    except RateLimitError as exc:
        import logging

        logging.getLogger("quantrag").warning("Upstream rate limit in /ask: %s", exc)
        raise HTTPException(status_code=429, detail="Rate limit exceeded.") from exc
    except UpstreamStatusError as exc:
        import logging

        logging.getLogger("quantrag").warning("Upstream status error in /ask: %s", exc)
        raise HTTPException(status_code=502, detail=f"Upstream error ({exc.status_code})") from exc
    except UpstreamError as exc:
        import logging

        logging.getLogger("quantrag").warning("Upstream error in /ask: %s", exc)
        raise HTTPException(status_code=502, detail="Upstream service error.") from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        import logging

        logging.getLogger("quantrag").exception("Unexpected failure in /ask")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    return result.to_dict()


@app.post("/search", dependencies=[Depends(_validate_api_key)])
async def search(payload: SearchRequest) -> dict[str, Any]:
    """Search the knowledge base and return ranked snippets."""
    try:
        svc = await get_service()
        hits = await svc.search(
            payload.query,
            k=payload.k,
            mode=payload.mode,
            min_score=payload.min_score,
        )
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"QuantRAG is not configured: {exc}") from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UpstreamTimeoutError as exc:
        import logging

        logging.getLogger("quantrag").warning("Upstream timeout in /search: %s", exc)
        raise HTTPException(status_code=504, detail="Upstream timeout.") from exc
    except RateLimitError as exc:
        import logging

        logging.getLogger("quantrag").warning("Upstream rate limit in /search: %s", exc)
        raise HTTPException(status_code=429, detail="Rate limit exceeded.") from exc
    except UpstreamStatusError as exc:
        import logging

        logging.getLogger("quantrag").warning("Upstream status error in /search: %s", exc)
        raise HTTPException(status_code=502, detail=f"Upstream error ({exc.status_code})") from exc
    except UpstreamError as exc:
        import logging

        logging.getLogger("quantrag").warning("Upstream error in /search: %s", exc)
        raise HTTPException(status_code=502, detail="Upstream service error.") from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        import logging

        logging.getLogger("quantrag").exception("Unexpected failure in /search")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return {
        "results": [hit.to_dict() for hit in hits],
        "count": len(hits),
        "mode": payload.mode or svc.settings.mode,
    }
