"""Offline tests for :class:`quantrag.service.RagService` (fixtures + fake chat)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantrag.config import RagSettings, load_rag_settings
from quantrag.errors import ConfigurationError, RetrievalError
from quantrag.rag import NOT_FOUND, RAG_SYSTEM_PROMPT
from quantrag.retriever import Chunk, load_chunks_jsonl
from quantrag.service import SNIPPET_CHARS, RagService, build_embedder

pytestmark = pytest.mark.anyio

FIXTURE = Path(__file__).parent / "fixtures" / "mini_chunks.jsonl"


def anyio_backend() -> str:
    return "asyncio"


class FakeChat:
    """Chat stub: records the prompt/system pair and replays a canned answer."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
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
    reply: str = "Answer [1].",
    with_vectors: bool = True,
) -> tuple[RagService, FakeChat]:
    chat = FakeChat(reply)
    embedder = build_embedder("local", chunks=chunks) if with_vectors else None
    service = await RagService.from_chunks(
        chunks,
        chat=chat,
        settings=settings,
        embedder=embedder,
        corpus_label="fixture",
    )
    return service, chat


async def test_from_chunks_indexes_the_fixture(chunks: list[Chunk], settings: RagSettings) -> None:
    service, _chat = await make_service(chunks, settings)

    assert service.size == len(chunks) == 30
    assert service.has_vectors is True
    assert service.corpus_label == "fixture"

    described = service.describe()
    assert described["chunks"] == 30
    assert described["vector"] is True
    assert described["embedder"] == "TfidfEmbedder"
    assert described["mode"] == "hybrid"
    assert described["min_score"] is None


async def test_bm25_search_returns_ranked_hits(chunks: list[Chunk], settings: RagSettings) -> None:
    service, _chat = await make_service(chunks, settings)

    hits = await service.search("Brownian motion with drift", k=3, mode="bm25")

    assert len(hits) == 3
    assert all(hit.source == "bm25" for hit in hits)
    assert all(hit.score > 0 for hit in hits)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


async def test_vector_search_returns_cosine_scores(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    service, _chat = await make_service(chunks, settings)

    hits = await service.search("Brownian motion with drift", k=2, mode="vector")

    assert len(hits) == 2
    assert all(hit.source == "vector" for hit in hits)
    assert hits[0].score >= hits[1].score
    assert -1.0 <= hits[1].score <= 1.0


async def test_hybrid_search_fuses_into_unit_scores(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    service, _chat = await make_service(chunks, settings)

    hits = await service.search("Brownian motion with drift", k=4, mode="hybrid")

    assert hits
    assert all(hit.source == "hybrid" for hit in hits)
    assert all(0.0 < hit.score <= 1.0 for hit in hits)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({hit.chunk.id for hit in hits}) == len(hits)


async def test_search_rejects_bad_input(chunks: list[Chunk], settings: RagSettings) -> None:
    service, _chat = await make_service(chunks, settings)

    with pytest.raises(RetrievalError):
        await service.search("   ")
    with pytest.raises(RetrievalError):
        await service.search("Brownian", mode="semantic")
    with pytest.raises(RetrievalError):
        await service.search("Brownian", k=0)


async def test_vector_mode_requires_an_embedding_index(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    service, _chat = await make_service(chunks, settings, with_vectors=False)

    assert service.has_vectors is False
    with pytest.raises(RetrievalError):
        await service.search("Brownian motion", mode="vector")

    # Hybrid degrades to BM25 instead of failing the whole request.
    hits = await service.search("Brownian motion", mode="hybrid", k=2)
    assert hits and all(hit.source == "bm25" for hit in hits)


async def test_min_score_override_filters_hits(chunks: list[Chunk], settings: RagSettings) -> None:
    service, _chat = await make_service(chunks, settings)

    unfiltered = await service.search("Brownian motion", k=5, mode="bm25")
    assert unfiltered

    assert await service.search("Brownian motion", k=5, mode="bm25", min_score=1e9) == []
    assert len(await service.search("Brownian motion", k=5, mode="bm25", min_score=0.0)) == len(
        unfiltered
    )


async def test_ask_returns_cited_sources(chunks: list[Chunk], settings: RagSettings) -> None:
    reply = "Brownian motion with drift has E[dX] = mu dt [1]."
    service, chat = await make_service(chunks, settings, reply=reply)

    result = await service.ask("What is Brownian motion with drift?")

    assert result.grounded is True
    assert result.answer == reply
    assert result.retrieved
    assert chat.systems == [RAG_SYSTEM_PROMPT]
    assert "Question:" in chat.prompts[0]

    first = result.sources[0]
    assert first["n"] == 1
    assert first["cited"] is True
    assert first["url"].startswith("https://")
    assert first["title"].startswith("18.642 Lecture 19_1")
    assert result.sources[0]["chunk_id"] == result.retrieved[0].chunk.id

    payload = result.to_dict()
    assert payload["grounded"] is True
    first_hit = payload["retrieved"][0]
    assert first_hit["chunk_id"] == result.retrieved[0].chunk.id
    # Fixture passages are longer than the snippet budget, so they get trimmed.
    assert first_hit["truncated"] is True
    assert first_hit["snippet"].endswith("\u2026")
    assert len(first_hit["snippet"]) <= SNIPPET_CHARS + 1


async def test_ask_marks_uncited_sources(chunks: list[Chunk], settings: RagSettings) -> None:
    service, _chat = await make_service(chunks, settings, reply="No citation in this text.")

    result = await service.ask("What is Brownian motion with drift?")

    assert result.grounded is True
    assert result.sources
    assert all(source["cited"] is False for source in result.sources)


@pytest.mark.parametrize(
    "reply",
    [NOT_FOUND, f"  {NOT_FOUND.lower()}  ", f'"{NOT_FOUND}."'],
)
async def test_ask_reports_not_found_without_sources(
    chunks: list[Chunk], settings: RagSettings, reply: str
) -> None:
    service, _chat = await make_service(chunks, settings, reply=reply)

    result = await service.ask("What is Brownian motion with drift?")

    assert result.grounded is False
    assert result.answer == NOT_FOUND
    assert result.sources == []
    assert result.retrieved


async def test_ask_without_hits_skips_the_model(chunks: list[Chunk], settings: RagSettings) -> None:
    service, chat = await make_service(chunks, settings)
    service._bm25 = type(service._bm25)(chunks=[])  # simulate an empty corpus index

    result = await service.ask("What is Brownian motion?", mode="bm25")

    assert result.grounded is False
    assert result.answer == NOT_FOUND
    assert result.sources == []
    assert chat.prompts == []


async def test_ask_rejects_empty_question(chunks: list[Chunk], settings: RagSettings) -> None:
    service, _chat = await make_service(chunks, settings)

    with pytest.raises(RetrievalError):
        await service.ask("  ")


async def test_from_chunks_rejects_empty_input(settings: RagSettings) -> None:
    with pytest.raises(ConfigurationError):
        await RagService.from_chunks([], chat=FakeChat("x"), settings=settings)
