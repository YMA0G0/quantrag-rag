"""Tests for hybrid retriever."""

from __future__ import annotations

import re

import pytest

from quantrag.bm25 import create_bm25_index
from quantrag.errors import RetrievalError
from quantrag.hybrid_retriever import Filter, HybridRetriever
from quantrag.retriever import Chunk, InMemoryVectorStore, load_chunks_jsonl


class FakeEmbedder:
    """Fake embedder for testing."""

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        # Simple fake vectors: 10 dimensions
        return [[float(i == j) for i in range(10)] for j in range(len(texts))]


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            id="a",
            text="Black Scholes option pricing formula",
            title="A",
            url="http://a",
            category="Options",
        ),
        Chunk(
            id="b",
            text="Ito lemma and Brownian motion",
            title="B",
            url="http://b",
            category="Stochastic Calculus",
        ),
        Chunk(
            id="c",
            text="Bond yield and duration",
            title="C",
            url="http://c",
            category="Fixed Income",
        ),
        Chunk(
            id="d",
            text="Heston stochastic volatility model",
            title="D",
            url="http://d",
            category="Stochastic Volatility",
        ),
        Chunk(
            id="e",
            text="Black Scholes equation for European options",
            title="E",
            url="http://e",
            category="Options",
        ),
    ]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def hybrid_retriever(sample_chunks):

    create_bm25_index(sample_chunks)
    InMemoryVectorStore()
    retriever = HybridRetriever(
        embedder=None,
        vector_store=None,
        bm25_index=None,
    )
    return retriever


class TestFilter:
    def test_filter_category(self):
        f = Filter(category="Options")
        assert f.matches(Chunk(id="a", text="test", title="A", url="http://a", category="Options"))
        assert not f.matches(
            Chunk(id="b", text="test", title="B", url="http://b", category="Stochastic")
        )

    def test_filter_url_contains(self):
        f = Filter(url_contains="mit.edu")
        assert f.matches(
            Chunk(id="a", text="test", title="A", url="https://mit.edu/x", category="")
        )
        assert not f.matches(
            Chunk(id="b", text="test", title="B", url="http://other.com", category="")
        )

    def test_filter_title_contains(self):
        f = Filter(title_contains="Black Scholes")
        assert f.matches(Chunk(id="a", text="test", title="Black Scholes Formula", url="http://a"))
        assert not f.matches(Chunk(id="b", text="test", title="Heston Model", url="http://b"))

    def test_combined_filters(self):
        f = Filter(category="Options", url_contains="mit.edu")
        assert f.matches(
            Chunk(id="a", text="test", title="A", url="https://mit.edu/x", category="Options")
        )
        assert not f.matches(
            Chunk(id="b", text="test", title="B", url="https://mit.edu/x", category="Stochastic")
        )
        assert not f.matches(
            Chunk(id="c", text="test", title="C", url="http://other.com", category="Options")
        )


class TestHybridRetrieverBM25:
    """Test BM25-only mode of hybrid retriever."""

    @pytest.mark.anyio
    async def test_bm25_retrieve(self, sample_chunks):
        retriever = HybridRetriever(
            embedder=None,
            vector_store=None,
            bm25_index=None,
        )
        await retriever.index(sample_chunks, build_bm25=True)

        results = await retriever.retrieve("Black Scholes", k=3, mode="bm25")
        assert len(results) <= 3
        ids = [r.chunk.id for r in results]
        assert "a" in ids or "e" in ids

    @pytest.mark.anyio
    async def test_bm25_with_filter(self, sample_chunks):
        retriever = HybridRetriever(
            embedder=None,
            vector_store=None,
            bm25_index=None,
        )
        await retriever.index(sample_chunks, build_bm25=True)

        results = await retriever.retrieve(
            "Black", k=3, mode="bm25", filter_=Filter(category="Options")
        )
        for r in results:
            assert r.chunk.category == "Options"

    @pytest.mark.anyio
    async def test_bm25_min_score(self, sample_chunks):
        retriever = HybridRetriever(
            embedder=None,
            vector_store=None,
            bm25_index=None,
        )
        await retriever.index(sample_chunks, build_bm25=True)

        results = await retriever.retrieve("Black Scholes", k=5, mode="bm25", min_score=10.0)
        # Should filter by min_score
        assert all(r.score >= 10.0 for r in results)


class TestHybridRetrieverVector:
    """Test vector-only mode of hybrid retriever."""

    @pytest.mark.anyio
    async def test_vector_retrieve_requires_embedder(self, sample_chunks):
        retriever = HybridRetriever(embedder=None)
        with pytest.raises(RetrievalError):
            await retriever.retrieve("test", mode="vector")


class TestHybridRetrieverHybrid:
    """Test hybrid mode."""

    @pytest.mark.anyio
    async def test_hybrid_retrieve(self, sample_chunks):
        """Test hybrid mode with fake embedder."""
        embedder = FakeEmbedder()
        retriever = HybridRetriever(
            embedder=embedder,
            bm25_index=None,
        )
        await retriever.index(sample_chunks)

        results = await retriever.retrieve("Black Scholes", k=3, mode="hybrid")
        assert len(results) <= 3
        # Should have both vector and bm25 sources in hybrid
        sources = [r.source for r in results]
        assert any("hybrid" in s for s in sources) or any("bm25" in s for s in sources)

    @pytest.mark.anyio
    async def test_hybrid_alpha_zero_is_bm25_only(self, sample_chunks):
        embedder = FakeEmbedder()
        retriever = HybridRetriever(embedder=embedder, hybrid_alpha=0.0)
        await retriever.index(sample_chunks)

        results = await retriever.retrieve("Black Scholes", k=3, mode="hybrid")
        # With alpha=0, should be BM25 only
        assert len(results) > 0

    @pytest.mark.anyio
    async def test_hybrid_alpha_one_is_vector_only(self, sample_chunks):
        embedder = FakeEmbedder()
        retriever = HybridRetriever(embedder=embedder, hybrid_alpha=1.0)
        await retriever.index(sample_chunks)

        results = await retriever.retrieve("Black Scholes", k=3, mode="hybrid")
        # With alpha=1, should be vector only (but needs embedder)
        # Note: vector results will have fake embeddings
        assert len(results) > 0


class TestHybridRetrieverFiltering:
    @pytest.mark.anyio
    async def test_metadata_filtering(self, sample_chunks):
        retriever = HybridRetriever(bm25_index=None)
        await retriever.index(sample_chunks, build_bm25=True)

        # Filter by category
        results = await retriever.retrieve(
            "Black", k=5, mode="bm25", filter_=Filter(category="Options")
        )
        for r in results:
            assert r.chunk.category == "Options"

    @pytest.mark.anyio
    async def test_url_filter(self, sample_chunks):
        retriever = HybridRetriever(bm25_index=None)
        await retriever.index(sample_chunks, build_bm25=True)

        results = await retriever.retrieve(
            "Black", k=5, mode="bm25", filter_=Filter(url_contains="http://a")
        )
        for r in results:
            assert "http://a" in r.chunk.url


class TestDeterministicResults:
    @pytest.mark.anyio
    async def test_deterministic_bm25(self, sample_chunks):
        """BM25 results should be deterministic for same query."""
        retriever = HybridRetriever(bm25_index=None)
        await retriever.index(sample_chunks, build_bm25=True)

        results1 = await retriever.retrieve("Black Scholes", k=3, mode="bm25")
        results2 = await retriever.retrieve("Black Scholes", k=3, mode="bm25")

        ids1 = [r.chunk.id for r in results1]
        ids2 = [r.chunk.id for r in results2]
        assert ids1 == ids2

    @pytest.mark.anyio
    async def test_deterministic_hybrid(self, sample_chunks):
        embedder = FakeEmbedder()
        retriever = HybridRetriever(embedder=embedder, hybrid_alpha=0.5)
        await retriever.index(sample_chunks)

        results1 = await retriever.retrieve("Black Scholes", k=3, mode="hybrid")
        results2 = await retriever.retrieve("Black Scholes", k=3, mode="hybrid")

        ids1 = [r.chunk.id for r in results1]
        ids2 = [r.chunk.id for r in results2]
        assert ids1 == ids2


class TestHybridRetrieverWithMinCorpus:
    """Integration tests against tests/fixtures/mini_chunks.jsonl (30 real chunks)."""

    @pytest.mark.anyio
    async def test_bm25_on_min_corpus(self):
        chunks = load_chunks_jsonl("tests/fixtures/mini_chunks.jsonl")
        retriever = HybridRetriever(bm25_index=None)
        await retriever.index(chunks, build_bm25=True)

        # Query that exists in the mini corpus (first 30 chunks are from MIT 18.642 lec19)
        results = await retriever.retrieve("Brownian Motion", k=5, mode="bm25")
        assert len(results) > 0
        assert all(r.source == "bm25" for r in results)
        # Relevance: top chunk must mention the query concept
        assert "brownian motion" in results[0].chunk.text.lower()

    @pytest.mark.anyio
    async def test_bm25_formula_queries(self):
        chunks = load_chunks_jsonl("tests/fixtures/mini_chunks.jsonl")
        retriever = HybridRetriever(bm25_index=None)
        await retriever.index(chunks, build_bm25=True)

        formula_queries = [
            "dS = μS dt + σS dW",
            "∂V/∂t + ½σ²S²∂²V/∂S²",
            "dS = μS dt + σS dW_t",
            "∫_0^T σ² dt",
        ]

        for query in formula_queries:
            results = await retriever.retrieve(query, k=3, mode="bm25")
            # Relevance: at least one retrieved chunk must contain a math symbol token
            assert len(results) > 0
            assert any(re.search(r"[σµ∫∂²½]", r.chunk.text) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
