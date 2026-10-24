"""BM25 lexical retrieval with formula-aware tokenization.

Provides a lightweight BM25 implementation for deterministic lexical retrieval
over the chunk corpus, with formula-aware tokenization for mathematical content.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .formula import compute_term_frequencies, fast_tokenize
from .retriever import Chunk, RetrievalError

# BM25 parameters (standard defaults)
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True, slots=True)
class BM25Result:
    """A chunk returned by BM25 search with its score."""

    chunk: Chunk
    score: float
    source: str = "bm25"


@dataclass
class BM25Index:
    """
    BM25 index over a corpus of chunks.

    Builds an inverted index with document frequencies for BM25 scoring.
    """

    chunks: list[Chunk]
    doc_freqs: dict[str, int] = field(default_factory=dict)
    doc_lengths: list[int] = field(default_factory=list)
    avg_doc_length: float = 0.0
    term_freqs: list[dict[str, int]] = field(default_factory=list)  # per-doc term freqs
    total_docs: int = 0

    # Tokenized representations for each chunk
    chunk_tokens: list[list[str]] = field(default_factory=list)

    # Mapping from token to set of document indices containing it
    inverted_index: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))

    def __post_init__(self):
        if self.chunks and not self.doc_freqs:
            self._build()

    def _build(self) -> None:
        """Build the BM25 index from chunks."""
        self.total_docs = len(self.chunks)
        if self.total_docs == 0:
            return

        # Tokenize all chunks
        self.chunk_tokens = []
        total_len = 0

        for i, chunk in enumerate(self.chunks):
            tokens = fast_tokenize(chunk.text)
            self.chunk_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            # Compute term frequencies for this document
            tf = compute_term_frequencies(tokens)
            self.term_freqs.append(tf)

            # Update document frequencies and inverted index
            for term in tf:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                self.inverted_index[term].add(i)

        self.avg_doc_length = total_len / self.total_docs if self.total_docs > 0 else 0.0

    def search(self, query: str, k: int = 10, min_score: float = 0.0) -> list[BM25Result]:
        """
        Search the index using BM25.

        Args:
            query: Query string
            k: Number of results to return
            min_score: Minimum score threshold

        Returns:
            List of BM25Result sorted by score descending
        """
        if not self.chunks:
            return []
        if k <= 0:
            raise RetrievalError("k must be greater than 0")

        query_tokens = fast_tokenize(query)
        if not query_tokens:
            return []

        # Compute BM25 scores for documents containing at least one query term
        candidate_docs: set[int] = set()
        for token in query_tokens:
            candidate_docs.update(self.inverted_index.get(token, set()))

        if not candidate_docs:
            return []

        scores: list[BM25Result] = []

        for doc_idx in candidate_docs:
            score = self._score_doc(query_tokens, doc_idx)
            if score >= min_score:
                scores.append(BM25Result(chunk=self.chunks[doc_idx], score=score))

        # Sort by score descending
        scores.sort(key=lambda r: r.score, reverse=True)
        return scores[:k]

    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a document given query tokens."""
        if doc_idx >= self.total_docs:
            return 0.0

        tf_dict = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term not in tf_dict:
                continue

            tf = tf_dict[term]
            df = self.doc_freqs.get(term, 0)

            if df == 0:
                continue

            # IDF component (using standard BM25 formula)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

            # TF component with saturation
            numerator = tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self.avg_doc_length)

            score += idf * (numerator / denominator)

        return score

    def save(self, path: str) -> None:
        """Persist the BM25 index to disk."""
        import json

        data = {
            "chunks": [
                {"id": c.id, "text": c.text, "title": c.title, "url": c.url, "category": c.category}
                for c in self.chunks
            ],
            "doc_freqs": self.doc_freqs,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "term_freqs": self.term_freqs,
            "total_docs": self.total_docs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> BM25Index:
        """Load a BM25 index from disk."""
        import json

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            chunks = [Chunk(**c) for c in data["chunks"]]
            index = cls(chunks=chunks)
            index.doc_freqs = data["doc_freqs"]
            index.doc_lengths = data["doc_lengths"]
            index.avg_doc_length = data["avg_doc_length"]
            index.term_freqs = data["term_freqs"]
            index.total_docs = data["total_docs"]

            # Rebuild inverted index
            for _term, _df in index.doc_freqs.items():
                # We need to rebuild from term_freqs
                pass

            # Rebuild inverted index from term_freqs
            for doc_idx, tf_dict in enumerate(index.term_freqs):
                for term in tf_dict:
                    index.inverted_index[term].add(doc_idx)

            return index
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RetrievalError(f"Could not load BM25 index from {path}: {exc!r}") from exc


def create_bm25_index(chunks: list[Chunk]) -> BM25Index:
    """Factory function to create a BM25 index from chunks."""
    return BM25Index(chunks=chunks)
