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
from collections.abc import Mapping
from typing import Any

from .errors import ZeroVectorError
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
            raise ZeroVectorError(
                "Query has no vocabulary overlap with the corpus."
            )
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

    # -- persistence -----------------------------------------------------
    def state(self) -> dict[str, Any]:
        """Serialise the fitted model so a query embedder can be rebuilt exactly.

        The vocabulary insertion order is preserved, so the saved IDF list
        still lines up with the vector dimensions stored in the index.
        """
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before state().")
        return {
            "format": 1,
            "min_df": self.min_df,
            "sublinear_tf": self.sublinear_tf,
            "vocab": dict(self._vocab),
            "idf": list(self._idf),
            "corpus_size": self.corpus_size,
            "document_frequencies": dict(self.document_frequencies),
            "model_name": self.model_name,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> TfidfEmbedder:
        """Rebuild a fitted model from :meth:`state` (needs no corpus)."""
        try:
            vocab = {str(token): int(index) for token, index in dict(state["vocab"]).items()}
            idf = [float(value) for value in state["idf"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid TfidfEmbedder state: {exc!r}") from exc
        if not vocab or len(idf) != len(vocab):
            raise ValueError("Invalid TfidfEmbedder state: vocabulary and idf disagree.")
        embedder = cls(
            min_df=int(state.get("min_df", 1)),
            sublinear_tf=bool(state.get("sublinear_tf", True)),
        )
        embedder._vocab = vocab
        embedder._idf = idf
        embedder._fitted = True
        embedder.corpus_size = int(state.get("corpus_size", 0))
        embedder.document_frequencies = {
            str(token): int(count)
            for token, count in dict(state.get("document_frequencies", {})).items()
        }
        return embedder
