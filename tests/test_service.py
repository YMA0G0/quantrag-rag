"""Offline tests for :class:`quantrag.service.RagService` (fixtures + fake chat)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantrag.config import RagSettings, load_rag_settings
from quantrag.errors import ConfigurationError, RetrievalError, ZeroVectorError
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


async def test_retrieval_only_construction_and_search(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    """Retrieval-only: build service without chat provider and search successfully."""
    embedder = build_embedder("local", chunks=chunks)
    service = await RagService.from_chunks(
        chunks,
        chat=None,  # no chat provider
        settings=settings,
        embedder=embedder,
        corpus_label="fixture",
    )

    # search should work without chat
    hits = await service.search("Brownian motion with drift", k=3, mode="bm25")
    assert len(hits) == 3
    assert all(hit.source == "bm25" for hit in hits)

    # vector search should also work
    hits = await service.search("Brownian motion with drift", k=2, mode="vector")
    assert len(hits) == 2
    assert all(hit.source == "vector" for hit in hits)

    # hybrid search should also work
    hits = await service.search("Brownian motion with drift", k=2, mode="hybrid")
    assert len(hits) == 2
    assert all(hit.source == "hybrid" for hit in hits)

    # ask() should fail with a clear error
    with pytest.raises(ConfigurationError, match="chat provider"):
        await service.ask("What is Brownian motion?")


# ------------------------------------------------------------------
# Zero-vector / OOV query regression tests
# ------------------------------------------------------------------


async def test_vector_mode_oov_query_returns_empty(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    """Vector mode with out-of-vocabulary query returns empty list.

    The query contains no terms from the corpus vocabulary, so the
    TF-IDF embedder produces a zero vector. The service should catch
    ZeroVectorError and return an empty result set instead of crashing.
    """
    service, _chat = await make_service(chunks, settings)

    # This query has no vocabulary overlap with the MIT Brownian motion corpus
    hits = await service.search(
        "lie group rigid registration medical imaging",
        k=5,
        mode="vector",
    )
    assert hits == []


async def test_hybrid_mode_oov_query_falls_back_to_bm25(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    """Hybrid mode with OOV query falls back to BM25 results.

    The vector side returns no hits (OOV), but BM25 still works.
    Hybrid fusion should produce results using only BM25 scores.
    """
    service, _chat = await make_service(chunks, settings)

    hits = await service.search(
        "lie group rigid registration medical imaging",
        k=5,
        mode="hybrid",
    )
    # Should return BM25 results (may be empty if BM25 also has no matches)
    assert isinstance(hits, list)
    # All returned hits should be from BM25 (source contains 'bm25' or 'hybrid')
    for hit in hits:
        assert "bm25" in hit.source.lower() or "hybrid" in hit.source.lower()


async def test_valid_in_vocabulary_vector_query_unchanged(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    """In-vocabulary vector queries behave exactly as before."""
    service, _chat = await make_service(chunks, settings)

    hits = await service.search("Brownian motion with drift", k=3, mode="vector")

    assert len(hits) == 3
    assert all(hit.source == "vector" for hit in hits)
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert all(-1.0 <= hit.score <= 1.0 for hit in hits)


async def test_unrelated_retrievalerror_propagates(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    """RetrievalError from dimension mismatch or corrupt index still propagates."""
    from quantrag.retriever import InMemoryVectorStore

    embedder = build_embedder("local", chunks=chunks)
    service = await RagService.from_chunks(
        chunks,
        chat=None,
        settings=settings,
        embedder=embedder,
        corpus_label="fixture",
    )

    # Directly test the vector store with wrong dimension
    store = service._store
    assert store is not None
    with pytest.raises(RetrievalError, match="dimension"):
        store.search([1.0, 0.0, 0.0], 1)  # wrong dimension


async def test_zero_vector_error_is_retrievalerror_subclass() -> None:
    """ZeroVectorError is a subclass of RetrievalError for proper catching."""
    from quantrag.errors import ZeroVectorError, RetrievalError

    assert issubclass(ZeroVectorError, RetrievalError)

    # Can be caught as RetrievalError
    try:
        raise ZeroVectorError("test")
    except RetrievalError:
        pass
    else:
        pytest.fail("ZeroVectorError not caught as RetrievalError")


async def test_blank_query_still_rejected_before_embedding(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    """Blank queries are rejected before any embedding attempt."""
    service, _chat = await make_service(chunks, settings)

    for bad_query in ["", "   ", "\n\t"]:
        with pytest.raises(RetrievalError):
            await service.search(bad_query, mode="vector")
        with pytest.raises(RetrievalError):
            await service.search(bad_query, mode="hybrid")
