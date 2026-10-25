"""Vector search over knowledge-base chunks.

The retriever is the bridge between the knowledge base (Data Engineer) and the
RAG API (LLM/RAG Engineer): it takes a question and returns the most relevant
chunks, each carrying the source information needed for citations.

It is deliberately small: an in-memory store with cosine similarity is plenty
for a corpus of a few thousand chunks, and the ``Embedder`` protocol means the
embedding provider can be swapped without touching this module.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .context import add_context_header
from .embeddings import Embedder
from .errors import RetrievalError

DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class Chunk:
    """One passage of a source document, ready to be embedded and cited.

    The ``text`` field contains the *body* only (no title/section header).
    The header is added at index time via ``add_context_header``.
    The ``section`` field records the logical section/heading within the document.
    The ``kind`` field classifies the chunk type for filtering.
    """

    id: str
    text: str
    title: str
    url: str
    category: str = ""
    section: str = ""
    kind: str = "prose"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by a search, with its cosine similarity (-1..1)."""

    chunk: Chunk
    score: float


def _normalise_kind(raw: object) -> str:
    """Map legacy/default kind values onto the public vocabulary.

    Older pipeline builds wrote ``"text"`` for ordinary prose (or omitted
    ``kind`` entirely).  Both mean ``"prose"`` for indexing purposes.
    """
    kind = str(raw or "").strip()
    return "prose" if kind in {"", "text"} else kind


def load_chunks_jsonl(path: str | Path) -> list[Chunk]:
    """Read chunks from a JSON-Lines file (one JSON object per line).

    Required keys per line: ``id``, ``text``, ``title``, ``url``.
    Optional: ``category``, ``section``, ``kind``.
    This is the hand-off format from the Data Engineer.
    """
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
                        section=str(row.get("section", "")),
                        kind=_normalise_kind(row.get("kind")),
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

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        query = _normalize(vector)
        if len(query) != len(self._vectors[0]):
            raise RetrievalError("Query vector dimension does not match the index.")
        scored = [
            RetrievedChunk(chunk, sum(a * b for a, b in zip(query, vec, strict=True)))
            for chunk, vec in zip(self._chunks, self._vectors, strict=True)
        ]
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


class Retriever:
    """Embeds chunks into a store and answers top-k similarity queries."""

    def __init__(self, embedder: Embedder, store: InMemoryVectorStore | None = None) -> None:
        self._embedder = embedder
        self._store = store if store is not None else InMemoryVectorStore()

    @property
    def store(self) -> InMemoryVectorStore:
        return self._store

    async def index(
        self,
        chunks: Iterable[Chunk],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        include_non_prose: bool = False,
    ) -> int:
        """Embed and store ``chunks`` (in batches); returns how many were added.

        Non-prose chunks (``kind != "prose"``) are skipped by default.  The
        header (title + section) is prepended to the body text at index time so
        that both the vector and the stored text include the citation context.
        """
        if batch_size <= 0:
            raise RetrievalError("batch_size must be greater than 0.")
        items = list(chunks)
        if not include_non_prose:
            items = [c for c in items if c.kind == "prose"]
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            texts_with_header = [add_context_header(c.text, c.title, c.section) for c in batch]
            vectors = await self._embedder.embed(texts_with_header)
            self._store.add(batch, vectors)
        return len(items)

    async def retrieve(
        self, query: str, *, k: int = DEFAULT_TOP_K, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks most similar to ``query``, best first.

        ``min_score`` drops weak matches so the RAG layer can answer "I don't
        know" instead of citing irrelevant text.
        """
        text = query.strip() if isinstance(query, str) else ""
        if not text:
            raise RetrievalError("The query must be a non-empty string.")
        if k <= 0:
            raise RetrievalError("k must be greater than 0.")
        [vector] = await self._embedder.embed([text])
        results = self._store.search(vector, k)
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results
