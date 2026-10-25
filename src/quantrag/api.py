"""FastAPI application (branch B contribution).

Exposes ``/health`` and ``/ask``.  This preliminary version calls the LLM
directly without retrieval and always returns ``sources=[]``; Phase 2 rewires
it to use :func:`quantrag.service.build_service` so ``/ask`` and ``/search``
share one retrieval path and one RagService instance.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from .config import load_settings
from .openrouter import OpenRouterClient

app = FastAPI(title="QuantRAG")


def _client() -> OpenRouterClient:
    """Build the OpenRouter chat client from the current environment."""
    return OpenRouterClient(load_settings())


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe: the service is up if this responds."""
    return {"status": "ok"}


@app.post("/ask")
async def ask(payload: dict[str, Any]) -> dict[str, object]:
    """Answer a question by calling the LLM directly (no retrieval yet).

    Returns the model's answer with an empty sources list.  Phase 2 replaces
    this with a retrieval-backed handler that returns real citations.
    """
    question = str((payload or {}).get("question", "") or "")
    if not question.strip():
        return {"answer": "", "grounded": False, "sources": [], "retrieved": []}
    answer = await _client().complete(question)
    return {"answer": answer, "grounded": True, "sources": [], "retrieved": []}
