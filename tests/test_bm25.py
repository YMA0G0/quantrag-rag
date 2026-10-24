"""Tests for BM25 lexical retrieval."""

from __future__ import annotations

import pytest

from quantrag.bm25 import BM25Index, BM25Result, create_bm25_index
from quantrag.errors import RetrievalError
from quantrag.retriever import Chunk


@pytest.fixture
def sample_chunks():
    return [
        Chunk(id="a", text="Black Scholes option pricing formula", title="A", url="http://a"),
        Chunk(id="b", text="Ito lemma and Brownian motion", title="B", url="http://b"),
        Chunk(id="c", text="Bond yield and duration", title="C", url="http://c"),
        Chunk(id="d", text="Heston stochastic volatility model", title="D", url="http://d"),
        Chunk(
            id="e", text="Black Scholes equation for European options", title="E", url="http://e"
        ),
    ]


@pytest.fixture
def bm25_index(sample_chunks):
    return create_bm25_index(sample_chunks)


class TestBM25IndexCreation:
    def test_creates_index(self, sample_chunks):
        index = create_bm25_index(sample_chunks)
        assert len(index.chunks) == 5
        assert index.total_docs == 5
        assert index.avg_doc_length > 0

    def test_empty_index(self):
        index = create_bm25_index([])
        assert index.total_docs == 0
        assert index.avg_doc_length == 0.0

    def test_doc_freqs(self, bm25_index):
        assert "black" in bm25_index.doc_freqs
        assert "scholes" in bm25_index.doc_freqs
        assert bm25_index.doc_freqs["black"] == 2  # appears in 2 docs
        assert bm25_index.doc_freqs["scholes"] == 2


class TestBM25Search:
    def test_basic_search(self, bm25_index):
        results = bm25_index.search("Black Scholes", k=3)
        assert len(results) <= 3
        assert all(isinstance(r, BM25Result) for r in results)
        # Should find both Black Scholes docs
        ids = [r.chunk.id for r in results]
        assert "a" in ids
        assert "e" in ids

    def test_single_term_search(self, bm25_index):
        results = bm25_index.search("Ito", k=3)
        ids = [r.chunk.id for r in results]
        assert "b" in ids

    def test_no_results(self, bm25_index):
        results = bm25_index.search("nonexistentterm", k=3)
        assert len(results) == 0

    def test_k_limits_results(self, bm25_index):
        assert len(bm25_index.search("Black Scholes", k=2)) <= 2
        assert len(bm25_index.search("Black Scholes", k=99)) <= 5

    def test_min_score(self, bm25_index):
        results = bm25_index.search("Black Scholes", k=5, min_score=10.0)
        # High min_score should filter out all
        assert all(r.score >= 10.0 for r in results)

    def test_scores_sorted_descending(self, bm25_index):
        results = bm25_index.search("Black Scholes", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self, bm25_index):
        results = bm25_index.search("", k=5)
        assert len(results) == 0

    def test_invalid_k_raises(self, bm25_index):
        with pytest.raises(RetrievalError):
            bm25_index.search("test", k=0)


class TestBM25Scoring:
    def test_term_frequency(self, sample_chunks):
        index = create_bm25_index(sample_chunks)
        # "black" appears twice in chunk "a" and twice in chunk "e"
        # Should have higher score for docs with more occurrences
        results = index.search("black", k=5)
        # Both docs with "black" should be ranked higher than others
        top_ids = [r.chunk.id for r in results[:2]]
        assert "a" in top_ids
        assert "e" in top_ids


class TestBM25IndexSaveLoad:
    def test_save_and_load_roundtrip(self, sample_chunks, tmp_path):
        index = create_bm25_index(sample_chunks)
        path = tmp_path / "bm25_index.json"
        index.save(path)

        # Load back
        loaded = BM25Index.load(path)
        assert loaded.total_docs == index.total_docs
        assert len(loaded.chunks) == len(index.chunks)
        assert loaded.doc_freqs == index.doc_freqs


class TestFastTokenize:
    def test_tokenize_basic(self):
        from quantrag.formula import fast_tokenize

        tokens = fast_tokenize("Black Scholes option pricing")
        assert "black" in tokens
        assert "scholes" in tokens

    def test_tokenize_math(self):
        from quantrag.formula import fast_tokenize

        tokens = fast_tokenize("dS = μS dt + σS dW")
        assert "dt" in tokens
        assert "mu" in tokens
        assert "sigma" in tokens

    def test_greek_letters(self):
        from quantrag.formula import fast_tokenize

        tokens = fast_tokenize("α β γ")
        assert "alpha" in tokens
        assert "beta" in tokens
        assert "gamma" in tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
