"""Vector and lexical search over knowledge-base chunks.

The retriever is the bridge between the knowledge base (Data Engineer) and the
RAG API (LLM/RAG Engineer): it takes a question and returns the most relevant
chunks, each carrying the source information needed for citations.

Supports:
- Vector search (cosine similarity) via embeddings
- Lexical search (BM25) with formula-aware tokenization
- Hybrid search combining both approaches
- Metadata filtering (category, source URL, title)
- Deterministic results for same query/config
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .bm25 import BM25Index, BM25Result
from .errors import RetrievalError

DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class Chunk:
    """One passage of a source document, ready to be embedded and cited."""

    id: str
    text: str
    title: str
    url: str
    category: str = ""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by a search, with its score and source."""

    chunk: Chunk
    score: float
    source: str  # "vector", "bm25", or "hybrid"


@dataclass(frozen=True, slots=True)
class Filter:
    """Metadata filter for retrieval."""

    category: str | None = None
    url_contains: str | None = None
    title_contains: str | None = None

    def matches(self, chunk: Chunk) -> bool:
        if self.category and chunk.category != self.category:
            return False
        if self.url_contains and self.url_contains not in chunk.url:
            return False
        return not (self.title_contains and self.title_contains.lower() not in chunk.title.lower())


def load_chunks_jsonl(path: str | Path) -> list[Chunk]:
    """Read chunks from a JSON-Lines file (one JSON object per line).

    Required keys per line: ``id``, ``text``, ``title``, ``url``.
    Optional: ``category``.  This is the hand-off format from the Data Engineer.
    """
    import json

    chunks: list[Chunk] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                chunks.append(
                    Chunk(
                        id=str(row["id"]),
                        text=str(row["text"]),
                        title=str(row["title"]),
                        url=str(row["url"]),
                        category=str(row.get("category", "")),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise RetrievalError(f"{path}: invalid chunk on line {line_no}: {exc!r}") from exc
    return chunks


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise RetrievalError("Cannot index a zero vector.")
    return [x / norm for x in vector]


class InMemoryVectorStore:
    """Stores chunks with unit-length vectors; search is a cosine top-k scan."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise RetrievalError("Number of chunks and vectors must match.")
        normalized = [_normalize(v) for v in vectors]
        if self._vectors and any(len(v) != len(self._vectors[0]) for v in normalized):
            raise RetrievalError("All vectors must have the same dimension.")
        if normalized and any(len(v) != len(normalized[0]) for v in normalized):
            raise RetrievalError("All vectors must have the same dimension.")
        self._chunks.extend(chunks)
        self._vectors.extend(normalized)

    def search(
        self, vector: list[float], k: int, filter_fn: Callable[[Chunk], bool] | None = None
    ) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        query = _normalize(vector)
        if len(query) != len(self._vectors[0]):
            raise RetrievalError("Query vector dimension does not match the index.")

        scored: list[RetrievedChunk] = []
        for chunk, vec in zip(self._chunks, self._vectors, strict=True):
            if filter_fn and not filter_fn(chunk):
                continue
            score = sum(a * b for a, b in zip(query, vec, strict=True))
            scored.append(RetrievedChunk(chunk=chunk, score=score, source="vector"))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> None:
        """Persist the index so embeddings are not recomputed on every start."""
        data = {
            "chunks": [asdict(c) for c in self._chunks],
            "vectors": self._vectors,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InMemoryVectorStore:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in data["chunks"]]
            vectors = data["vectors"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RetrievalError(f"Could not load index from {path}: {exc!r}") from exc
        store = cls()
        store.add(chunks, vectors)
        return store

    def get_all_chunks(self) -> list[Chunk]:
        """Return all chunks in the store."""
        return list(self._chunks)


class HybridRetriever:
    """
    Hybrid retriever combining vector (embedding) and lexical (BM25) search.

    Supports:
    - Vector search (cosine similarity) via embeddings
    - Lexical search (BM25) with formula-aware tokenization
    - Hybrid search with configurable weighting
    - Metadata filtering (category, source URL, title)
    - Deterministic results for same query/config
    """

    def __init__(
        self,
        embedder: object | None = None,  # Embedder protocol
        vector_store: object | None = None,
        bm25_index: object | None = None,
        *,
        hybrid_alpha: float = 0.5,  # weight for vector (1-alpha for BM25)
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            embedder: Embedder protocol implementation (for vector search)
            vector_store: InMemoryVectorStore instance
            bm25_index: BM25Index instance
            hybrid_alpha: Weight for vector scores (0=BM25 only, 1=vector only)
        """
        from .bm25 import BM25Index
        from .retriever import InMemoryVectorStore as VectorStore

        self._embedder = embedder
        self._vector_store: VectorStore = (
            vector_store if vector_store is not None else InMemoryVectorStore()
        )
        self._bm25_index: BM25Index = bm25_index if bm25_index is not None else BM25Index(chunks=[])
        self._hybrid_alpha = max(0.0, min(1.0, hybrid_alpha))

    @property
    def vector_store(self) -> object:
        return self._vector_store

    @property
    def bm25_index(self) -> object:
        return self._bm25_index

    def _make_filter_fn(self, filter_: object | None) -> Callable[[object], bool] | None:
        """Create a filter function from a Filter object."""
        if filter_ is None:
            return None
        return lambda chunk: filter_.matches(chunk)

    async def index(
        self,
        chunks: Iterable[object],
        *,
        batch_size: int = 32,
        build_bm25: bool = True,
    ) -> int:
        """
        Embed and store chunks in vector store, optionally build BM25 index.

        Args:
            chunks: Chunks to index
            batch_size: Batch size for embedding
            build_bm25: Whether to build BM25 index (default True)

        Returns:
            Number of chunks indexed
        """
        if batch_size <= 0:
            raise RetrievalError("batch_size must be greater than 0.")

        items = list(chunks)

        # Index in vector store if embedder available
        if self._embedder is not None:
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                vectors = await self._embedder.embed([c.text for c in batch])
                self._vector_store.add(batch, vectors)

        # Build BM25 index
        if build_bm25:
            self._bm25_index = BM25Index(chunks=list(items))

        return len(items)

    def add_chunks(self, chunks: Iterable[object], *, build_bm25: bool = True) -> int:
        """Add chunks to existing index (without re-embedding existing)."""
        items = list(chunks)
        # Note: This requires embedder to be available for vector store
        # For BM25, we rebuild the index (could be optimized for incremental)
        if build_bm25:
            all_chunks = list(self._bm25_index.chunks) + items
            self._bm25_index = BM25Index(chunks=all_chunks)
        return len(items)

    def _make_filter_fn(self, filter_: object | None) -> Callable[[object], bool] | None:
        if filter_ is None:
            return None
        return lambda chunk: filter_.matches(chunk)

    def _wrap_bm25_results(self, results: list[BM25Result]) -> list:
        """Wrap BM25Result objects in a common result wrapper."""
        wrapped = []
        for r in results:
            wrapped.append(_WrappedResult(r.chunk, r.score, "bm25"))
        return wrapped

    def _wrap_vector_results(self, results: list) -> list:
        """Wrap vector results with source info."""
        wrapped = []
        for r in results:
            wrapped.append(_WrappedResult(r.chunk, r.score, "vector"))
        return wrapped

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float | None = None,
        filter_: object | None = None,
        mode: str = "hybrid",  # "vector", "bm25", "hybrid"
        vector_weight: float | None = None,
    ) -> list[object]:
        """
        Retrieve top-k chunks for a query.

        Args:
            query: Query string
            k: Number of results to return
            min_score: Minimum score threshold
            filter_: Optional metadata filter
            mode: Search mode - "vector", "bm25", or "hybrid"
            vector_weight: Override hybrid_alpha for this query

        Returns:
            List of RetrievedChunk sorted by score descending
        """

        text = query.strip() if isinstance(query, str) else ""
        if not text:
            raise RetrievalError("The query must be a non-empty string.")
        if k <= 0:
            raise RetrievalError("k must be greater than 0.")

        alpha = vector_weight if vector_weight is not None else self._hybrid_alpha
        filter_fn = self._make_filter_fn(filter_)

        if mode == "vector":
            if self._embedder is None:
                raise RetrievalError("Vector search requires an embedder")
            [vector] = await self._embedder.embed([text])
            results = self._vector_store.search(vector, k, filter_fn=filter_fn)
            for r in results:
                r.source = "vector"
            if min_score is not None:
                results = [r for r in results if r.score >= min_score]
            return results[:k]

        elif mode == "bm25":
            results = self._bm25_index.search(text, k=k, min_score=min_score if min_score else 0.0)
            # Apply filter
            if filter_:

                def filter_fn(c):
                    return filter_.matches(c)

                results = [r for r in results if filter_fn(r.chunk)]
            # Wrap BM25 results
            wrapped = []
            for r in results:
                wrapped.append(_WrappedResult(r.chunk, r.score, "bm25"))
            return wrapped[:k]

        elif mode == "hybrid":
            # Get results from both (ask for more to have pool for fusion)
            pool_k = min(k * 3, len(self._bm25_index.chunks) if self._bm25_index.chunks else k * 3)

            vector_results = []
            bm25_results = []

            if self._embedder is not None:
                [vector] = await self._embedder.embed([text])
                vector_results = self._vector_store.search(vector, pool_k, filter_fn=filter_)

            bm25_results = self._bm25_index.search(text, k=pool_k, min_score=0.0)
            if filter_:

                def filter_fn(c):
                    return filter_.matches(c)

                bm25_results = [r for r in bm25_results if filter_fn(r.chunk)]

            # Normalize scores to [0, 1] range for fair fusion
            def normalize_scores(results: list, max_possible: float = 1.0) -> list:
                if not results:
                    return []
                max_score = max(r.score for r in results)
                min_score = min(r.score for r in results)
                if max_score == min_score:
                    return [(r.chunk, 0.5) for r in results]
                return [(r.chunk, (r.score - min_score) / (max_score - min_score)) for r in results]

            vec_norm = normalize_scores(vector_results)
            bm25_norm = normalize_scores(self._wrap_bm25_results(bm25_results))

            # Merge by chunk id
            merged: dict[str, tuple[object, float, str]] = {}

            for chunk, score in vec_norm:
                merged[chunk.id] = (chunk, score, "vector")

            for chunk, score in bm25_norm:
                if chunk.id in merged:
                    old_chunk, old_score, old_src = merged[chunk.id]
                    new_score = alpha * old_score + (1 - alpha) * score
                    # Prefer the source that contributed more
                    if alpha * old_score >= (1 - alpha) * score:
                        merged[chunk.id] = (chunk, new_score, "hybrid(vector-dominant)")
                    else:
                        merged[chunk.id] = (chunk, new_score, "hybrid(bm25-dominant)")
                else:
                    merged[chunk.id] = (chunk, score, "hybrid")

            # Convert back to RetrievedChunk and sort
            final_results = [
                _WrappedResult(chunk, score, src) for chunk, score, src in merged.values()
            ]

            final_results.sort(key=lambda r: r.score, reverse=True)

            if min_score is not None:
                final_results = [r for r in final_results if r.score >= min_score]

            return final_results[:k]

        else:
            raise RetrievalError(f"Unknown search mode: {mode}")

    def retrieve_bm25(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float = 0.0,
        filter_: object | None = None,
    ) -> list[object]:
        """Convenience method for BM25-only search."""
        results = self._bm25_index.search(query, k=k, min_score=min_score)
        if filter_:

            def filter_fn(c):
                return filter_.matches(c)

            results = [r for r in results if filter_fn(r.chunk)]
        wrapped = []
        for r in results:
            wrapped.append(_WrappedResult(r.chunk, r.score, "bm25"))
        return wrapped[:k]

    def retrieve_vector(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float | None = None,
        filter_: object | None = None,
    ) -> list[object]:
        """Convenience method for vector-only search."""
        if self._embedder is None:
            raise RetrievalError("Vector search requires an embedder")
        raise NotImplementedError("Use retrieve(mode='vector') async method")

    def set_hybrid_alpha(self, alpha: float) -> None:
        """Set the hybrid weighting parameter (0=BM25, 1=vector)."""
        self._hybrid_alpha = max(0.0, min(1.0, alpha))

    def get_all_chunks(self) -> list[object]:
        """Return all chunks in the vector store."""
        return self._vector_store.get_all_chunks()

    def get_bm25_chunks(self) -> list[object]:
        """Return all chunks in the BM25 index."""
        return self._bm25_index.chunks


@dataclass(frozen=True, slots=True)
class _WrappedResult:
    """Internal wrapper for unified result handling."""

    chunk: object
    score: float
    source: str
