"""Offline tests for the FastAPI endpoints (/ask, /search, /health)."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from quantrag.api import app, get_service
from quantrag.config import RagSettings, load_rag_settings
from quantrag.retriever import Chunk, load_chunks_jsonl
from quantrag.service import RagService, build_embedder

pytestmark = pytest.mark.anyio

FIXTURE = Path(__file__).parent / "fixtures" / "mini_chunks.jsonl"


def anyio_backend() -> str:
    return "asyncio"


class FakeChat:
    def __init__(self, reply: str = "The answer is [1].") -> None:
        self.reply = reply

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        return self.reply


@pytest.fixture
def settings() -> RagSettings:
    return load_rag_settings({}, load_env_file=False)


@pytest.fixture
def chunks() -> list[Chunk]:
    return load_chunks_jsonl(FIXTURE)


async def make_service(
    chunks: list[Chunk],
    settings: RagSettings,
    *,
    reply: str = "The answer is [1].",
) -> RagService:
    chat = FakeChat(reply)
    embedder = build_embedder("local", chunks=chunks)
    return await RagService.from_chunks(
        chunks,
        chat=chat,
        settings=settings,
        embedder=embedder,
        corpus_label="fixture",
    )


@pytest.fixture
async def client(chunks: list[Chunk], settings: RagSettings):
    service = await make_service(chunks, settings)
    # Override the service getter for tests
    from quantrag import api

    api._SERVICE = service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    api._SERVICE = None


async def test_health_endpoint(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert data["service"]["chunks"] == 30
    assert data["service"]["vector"] is True


async def test_ask_endpoint_returns_cited_answer(client: AsyncClient) -> None:
    response = await client.post("/ask", json={"question": "What is Brownian motion with drift?"})
    assert response.status_code == 200
    data = response.json()

    assert data["grounded"] is True
    assert data["answer"] == "The answer is [1]."
    assert data["sources"]
    assert data["sources"][0]["cited"] is True
    assert data["retrieved"]
    assert data["retrieved"][0]["chunk_id"] == "3460fe14fd-0000"


async def test_ask_endpoint_not_found(client: AsyncClient) -> None:
    from quantrag import api
    from quantrag.rag import NOT_FOUND
    from quantrag.service import RagService

    # Replace service with one that returns NOT_FOUND
    original = api._SERVICE
    from quantrag.config import load_rag_settings
    from quantrag.retriever import load_chunks_jsonl
    from quantrag.service import build_embedder

    settings = load_rag_settings({}, load_env_file=False)
    chunks = load_chunks_jsonl(FIXTURE)
    chat = FakeChat(NOT_FOUND)
    embedder = build_embedder("local", chunks=chunks)
    api._SERVICE = await RagService.from_chunks(
        chunks, chat=chat, settings=settings, embedder=embedder, corpus_label="fixture"
    )

    response = await client.post("/ask", json={"question": "What is the meaning of life?"})
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["answer"] == NOT_FOUND
    assert data["sources"] == []

    api._SERVICE = original


async def test_ask_endpoint_rejects_empty_question(client: AsyncClient) -> None:
    response = await client.post("/ask", json={"question": "   "})
    assert response.status_code == 422  # Pydantic validation


async def test_ask_endpoint_accepts_k_and_mode(client: AsyncClient) -> None:
    response = await client.post(
        "/ask",
        json={"question": "Brownian motion", "k": 2, "mode": "bm25"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    # Only 2 results requested
    assert len(data["retrieved"]) <= 2


async def test_search_endpoint_returns_ranked_results(client: AsyncClient) -> None:
    response = await client.post("/search", json={"query": "Brownian motion with drift", "k": 3})
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 3
    assert data["mode"] == "hybrid"  # default
    assert len(data["results"]) == 3
    assert all("score" in hit for hit in data["results"])
    assert all("chunk_id" in hit for hit in data["results"])
    assert all("snippet" in hit for hit in data["results"])
    assert data["results"][0]["score"] >= data["results"][1]["score"]


async def test_search_endpoint_respects_mode(client: AsyncClient) -> None:
    bm25 = await client.post("/search", json={"query": "Brownian motion", "k": 2, "mode": "bm25"})
    assert bm25.status_code == 200
    bm25_data = bm25.json()
    assert bm25_data["mode"] == "bm25"
    assert all(r["source"] == "bm25" for r in bm25_data["results"])

    vector = await client.post(
        "/search", json={"query": "Brownian motion", "k": 2, "mode": "vector"}
    )
    assert vector.status_code == 200
    vector_data = vector.json()
    assert vector_data["mode"] == "vector"
    assert all(r["source"] == "vector" for r in vector_data["results"])

    hybrid = await client.post(
        "/search", json={"query": "Brownian motion", "k": 2, "mode": "hybrid"}
    )
    assert hybrid.status_code == 200
    hybrid_data = hybrid.json()
    assert hybrid_data["mode"] == "hybrid"
    assert all(r["source"] == "hybrid" for r in hybrid_data["results"])


async def test_search_endpoint_min_score_filter(client: AsyncClient) -> None:
    unfiltered = await client.post(
        "/search", json={"query": "Brownian motion", "k": 5, "mode": "bm25"}
    )
    assert unfiltered.status_code == 200
    assert len(unfiltered.json()["results"]) > 0

    filtered = await client.post(
        "/search", json={"query": "Brownian motion", "k": 5, "mode": "bm25", "min_score": 1e9}
    )
    assert filtered.status_code == 200
    filtered_data = filtered.json()
    assert filtered_data["count"] == 0
    assert filtered_data["results"] == []


async def test_search_endpoint_rejects_empty_query(client: AsyncClient) -> None:
    response = await client.post("/search", json={"query": "   "})
    assert response.status_code == 422


async def test_both_endpoints_share_the_same_service(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    """The /ask and /search endpoints must share one RagService instance."""
    from quantrag import api

    api._SERVICE = await make_service(chunks, settings)

    # First call populates the singleton
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac1:
        await ac1.get("/health")
        svc1 = await get_service()

    # Second call reuses it
    async with AsyncClient(transport=transport, base_url="http://test") as ac2:
        await ac2.get("/health")
        svc2 = await get_service()

    assert svc1 is svc2
    api._SERVICE = None


async def test_service_unavailable_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing configuration yields 503 instead of crashing."""
    from quantrag import api

    api._SERVICE = None

    def failing_build(*args, **kwargs):
        from quantrag.errors import ConfigurationError

        raise ConfigurationError("No index, no corpus")

    monkeypatch.setattr(api, "build_service", failing_build)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/ask", json={"question": "test"})
        assert response.status_code == 503
        assert "QuantRAG is not configured" in response.json()["detail"]

        response = await ac.post("/search", json={"query": "test"})
        assert response.status_code == 503
        assert "QuantRAG is not configured" in response.json()["detail"]

    api._SERVICE = None


async def test_retrieval_error_returns_400(client: AsyncClient) -> None:
    """Bad mode parameter yields 400."""
    response = await client.post("/search", json={"query": "test", "mode": "semantic"})
    assert response.status_code == 422  # Pydantic validation

    response = await client.post("/ask", json={"question": "test", "mode": "semantic"})
    assert response.status_code == 422
