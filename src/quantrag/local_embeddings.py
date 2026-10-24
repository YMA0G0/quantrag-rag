"""Deterministic, offline TF-IDF embedder.

This is a *real* vector model (sparse TF-IDF, L2-normalised) that runs
without network access.  It is **not** a neural embedding model and it is
never selected implicitly: callers must ask for it by name.

The evaluation CLI requires an explicit ``--embedder`` choice so that a
missing OpenRouter key can never silently degrade the reported numbers.
"""

from __future__ import annotations

import math
from collections import Counter

from .formula import fast_tokenize


class TfidfEmbedder:
    """Corpus-fitted TF-IDF embedder implementing the ``Embedder`` protocol."""

    is_neural = False

    def __init__(self, *, min_df: int = 1, sublinear_tf: bool = True) -> None:
        if min_df < 1:
            raise ValueError("min_df must be >= 1")
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []
        self._fitted = False
        self.corpus_size = 0
        self.document_frequencies: dict[str, int] = {}

    # -- fitting ---------------------------------------------------------
    def fit(self, texts: list[str]) -> TfidfEmbedder:
        docs = [fast_tokenize(t) for t in texts]
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        vocab = sorted(tok for tok, count in df.items() if count >= self.min_df)
        if not vocab:
            raise ValueError("Empty vocabulary: cannot fit a TF-IDF embedder.")
        self.corpus_size = len(docs)
        self.document_frequencies = dict(df)
        self._vocab = {tok: i for i, tok in enumerate(vocab)}
        n = len(docs)
        self._idf = [math.log((1 + n) / (1 + df[tok])) + 1.0 for tok in vocab]
        self._fitted = True
        return self

    # -- embedding -------------------------------------------------------
    def _vector(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed().")
        counts = Counter(tok for tok in fast_tokenize(text) if tok in self._vocab)
        vector = [0.0] * len(self._vocab)
        for token, count in counts.items():
            index = self._vocab[token]
            weight = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            vector[index] = weight * self._idf[index]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    # -- introspection ---------------------------------------------------
    @property
    def model_name(self) -> str:
        return (
            f"local-tfidf(min_df={self.min_df},sublinear={self.sublinear_tf},"
            f"vocab={len(self._vocab)},corpus={self.corpus_size})"
        )

    @property
    def dimension(self) -> int:
        return len(self._vocab)
