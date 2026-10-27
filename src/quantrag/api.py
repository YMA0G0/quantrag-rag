"""FastAPI application sharing the same RagService as the MCP server.

Exposes ``/health``, ``/ask`` and ``/search``.  All three endpoints use the
single :class:`quantrag.service.RagService` instance so retrieval and answering
are consistent across transports.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import load_rag_settings
from .errors import ConfigurationError, RetrievalError
from .service import RagService, build_service

_SERVICE: RagService | None = None


async def get_service() -> RagService:
    """Lazy singleton: build the service on first use."""
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE
    _SERVICE = await build_service()
    return _SERVICE


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_rag_settings(load_env_file=True)
    except ConfigurationError as exc:
        import logging

        logging.getLogger("quantrag").warning("QuantRAG is not fully configured: %s", exc)
    yield
    global _SERVICE
    _SERVICE = None


app = FastAPI(title="QuantRAG", lifespan=lifespan)


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
async def health() -> dict[str, Any]:
    """Liveness probe with a small service description."""
    svc = await get_service()
    return {"status": "ok", "service": svc.describe()}


@app.post("/ask")
async def ask(payload: AskRequest) -> dict[str, Any]:
    """Answer a question using retrieval-augmented generation."""
    try:
        svc = await get_service()
        result = await svc.ask(payload.question, k=payload.k, mode=payload.mode)
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"QuantRAG is not configured: {exc}") from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        import logging

        logging.getLogger("quantrag").exception("Unexpected failure in /ask")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    return result.to_dict()


@app.post("/search")
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
    except Exception as exc:  # pragma: no cover - defensive guard
        import logging

        logging.getLogger("quantrag").exception("Unexpected failure in /search")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return {
        "results": [hit.to_dict() for hit in hits],
        "count": len(hits),
        "mode": payload.mode or svc.settings.mode,
    }
