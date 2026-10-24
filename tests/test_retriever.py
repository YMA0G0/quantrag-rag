"""Tests for chunk loading, the vector store and the retriever.

A deterministic bag-of-words ``FakeEmbedder`` stands in for the real embedding
model, so relevance behaviour can be tested offline without any credentials.
"""

from __future__ import annotations

import json
import re

import pytest

from quantrag.errors import RetrievalError
from quantrag.retriever import Chunk, InMemoryVectorStore, Retriever, load_chunks_jsonl

pytestmark = pytest.mark.anyio

VOCAB = [
    "black",
    "scholes",
    "option",
    "pricing",
    "brownian",
    "motion",
    "ito",
    "lemma",
    "bond",
    "yield",
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeEmbedder:
    """Counts vocabulary words: similar texts get similar vectors."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            out.append([float(words.count(w)) + 0.001 for w in VOCAB])
        return out


def chunk(id_: str, text: str) -> Chunk:
    return Chunk(id=id_, text=text, title=f"Title {id_}", url=f"https://example.invalid/{id_}")


CHUNKS = [
    chunk("bs", "Black Scholes option pricing formula for a European option"),
    chunk("ito", "Ito lemma and Brownian motion in stochastic calculus"),
    chunk("bond", "Bond yield and duration"),
]


async def test_most_relevant_chunk_comes_first() -> None:
    retriever = Retriever(FakeEmbedder())
    await retriever.index(CHUNKS)

    results = await retriever.retrieve("How does Black Scholes option pricing work?", k=3)

    assert [r.chunk.id for r in results][0] == "bs"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_result_carries_citation_fields() -> None:
    retriever = Retriever(FakeEmbedder())
    await retriever.index(CHUNKS)

    [top] = await retriever.retrieve("Brownian motion", k=1)

    assert top.chunk.id == "ito"
    assert top.chunk.title == "Title ito"
    assert top.chunk.url == "https://example.invalid/ito"


async def test_k_limits_results() -> None:
    retriever = Retriever(FakeEmbedder())
    await retriever.index(CHUNKS)
    assert len(await retriever.retrieve("option", k=2)) == 2
    assert len(await retriever.retrieve("option", k=99)) == 3


async def test_min_score_filters_weak_matches() -> None:
    retriever = Retriever(FakeEmbedder())
    await retriever.index(CHUNKS)

    results = await retriever.retrieve("Black Scholes option pricing", k=3, min_score=0.5)

    assert [r.chunk.id for r in results] == ["bs"]


async def test_empty_index_returns_nothing() -> None:
    assert await Retriever(FakeEmbedder()).retrieve("anything") == []


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t"])
async def test_blank_query_is_rejected_before_embedding(bad_query: str) -> None:
    embedder = FakeEmbedder()
    retriever = Retriever(embedder)
    with pytest.raises(RetrievalError):
        await retriever.retrieve(bad_query)
    assert embedder.calls == []


async def test_invalid_k_is_rejected() -> None:
    with pytest.raises(RetrievalError):
        await Retriever(FakeEmbedder()).retrieve("option", k=0)


async def test_indexing_is_batched() -> None:
    embedder = FakeEmbedder()
    retriever = Retriever(embedder)

    added = await retriever.index(CHUNKS, batch_size=2)

    assert added == 3
    assert [len(call) for call in embedder.calls] == [2, 1]
    assert len(retriever.store) == 3


async def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(RetrievalError):
        await Retriever(FakeEmbedder()).index(CHUNKS, batch_size=0)


def test_store_rejects_mismatched_input() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(RetrievalError):
        store.add(CHUNKS, [[1.0, 0.0]])  # 3 chunks, 1 vector
    with pytest.raises(RetrievalError):
        store.add(CHUNKS[:1], [[0.0, 0.0]])  # zero vector
    with pytest.raises(RetrievalError):
        store.add(CHUNKS[:2], [[1.0, 0.0], [1.0, 0.0, 0.0]])  # mixed dimensions


def test_store_rejects_wrong_query_dimension() -> None:
    store = InMemoryVectorStore()
    store.add(CHUNKS[:1], [[1.0, 0.0]])
    with pytest.raises(RetrievalError):
        store.search([1.0, 0.0, 0.0], 1)


async def test_save_and_load_roundtrip(tmp_path) -> None:
    retriever = Retriever(FakeEmbedder())
    await retriever.index(CHUNKS)
    path = tmp_path / "index.json"
    retriever.store.save(path)

    restored = Retriever(FakeEmbedder(), InMemoryVectorStore.load(path))
    results = await restored.retrieve("Black Scholes option pricing", k=1)

    assert [r.chunk for r in results] == [CHUNKS[0]]


def test_load_of_missing_or_corrupt_index_fails_cleanly(tmp_path) -> None:
    with pytest.raises(RetrievalError):
        InMemoryVectorStore.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(RetrievalError):
        InMemoryVectorStore.load(bad)


def test_load_chunks_jsonl(tmp_path) -> None:
    path = tmp_path / "chunks.jsonl"
    rows = [
        {"id": "a", "text": "hello", "title": "T", "url": "https://x.invalid", "category": "C"},
        {"id": "b", "text": "world", "title": "T2", "url": "https://y.invalid"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")

    chunks = load_chunks_jsonl(path)

    assert [c.id for c in chunks] == ["a", "b"]
    assert chunks[0].category == "C"
    assert chunks[1].category == ""


def test_load_chunks_jsonl_reports_the_bad_line(tmp_path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text('{"id": "a", "text": "x"}\n', encoding="utf-8")  # missing title/url
    with pytest.raises(RetrievalError) as excinfo:
        load_chunks_jsonl(path)
    assert "line 1" in str(excinfo.value)
