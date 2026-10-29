"""RagService -- one retrieval + answering pipeline shared by every transport.

The MCP server and the FastAPI app both call this class, so ``/ask``,
``/search``, ``ask_quant_question`` and ``search_quant_kb`` cannot drift apart.

Design notes
------------
* Retrieval modes are ``bm25``, ``vector`` and ``hybrid`` (``alpha`` weights
  the vector side).  Hybrid fusion min-max normalises the two candidate pools
  and combines them as ``alpha * vector + (1 - alpha) * bm25`` -- the same
  definition ``scripts/evaluate_retrieval.py`` scores, so the placeholder
  ``alpha=0.5`` can be replaced by the evaluated winner without touching this
  module's public API.
* ``QUANTRAG_MIN_SCORE`` is unset by default.  When set it filters *final*
  scores (cosine for ``vector``, raw BM25 for ``bm25``, the fused 0..1 score
  for ``hybrid``); see the README for the calibration procedure.
* Nothing here opens a socket directly: the embedder and the chat client are
  injected, which is what keeps the tests offline.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .config import RagSettings, load_rag_settings, load_settings
from .embeddings import Embedder, OpenRouterEmbedder, load_embedding_model
from .errors import ConfigurationError, MalformedResponseError, RetrievalError, ZeroVectorError
from .local_embeddings import TfidfEmbedder
from .openrouter import ChatCompleter
from .providers import create_provider_from_env
from .rag import (
    NOT_FOUND,
    RAG_SYSTEM_PROMPT,
    build_rag_prompt,
    cited_numbers,
    is_not_found,
)
from .retriever import Chunk, InMemoryVectorStore, Retriever, load_chunks_jsonl

logger = logging.getLogger(__name__)

INDEX_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
CHUNKS_NAME = "chunks.jsonl"
VECTORS_NAME = "vectors.json"
EMBEDDER_STATE_NAME = "embedder.json"

DEFAULT_BATCH_SIZE = 32
HYBRID_POOL_MULTIPLIER = 3
SNIPPET_CHARS = 400

_CHUNK_FIELDS = tuple(field.name for field in fields(Chunk))


class _Unset:
    """Sentinel type: "argument not supplied" (distinct from ``None``)."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A chunk returned by a search, with its score and the mode that produced it."""

    chunk: Chunk
    score: float
    source: str

    def to_dict(self, *, snippet_chars: int | None = SNIPPET_CHARS) -> dict[str, Any]:
        text = self.chunk.text
        truncated = False
        if snippet_chars is not None and len(text) > snippet_chars:
            text = text[:snippet_chars].rstrip() + "\u2026"
            truncated = True
        return {
            "chunk_id": self.chunk.id,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "url": self.chunk.url,
            "category": self.chunk.category,
            "kind": self.chunk.kind,
            "score": round(self.score, 6),
            "source": self.source,
            "snippet": text,
            "truncated": truncated,
        }


@dataclass(frozen=True, slots=True)
class AskResult:
    """A grounded (or explicitly not-found) answer plus its citations."""

    answer: str
    grounded: bool
    sources: list[dict[str, Any]]
    retrieved: list[RetrievalHit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "grounded": self.grounded,
            "sources": self.sources,
            "retrieved": [hit.to_dict() for hit in self.retrieved],
        }


class RagService:
    """Retrieval + grounded answering over one loaded knowledge base."""

    def __init__(
        self,
        *,
        chat: ChatCompleter | None = None,
        settings: RagSettings,
        chunks: list[Chunk],
        vector_store: InMemoryVectorStore | None = None,
        embedder: Embedder | None = None,
        corpus_label: str = "<in-memory>",
    ) -> None:
        self._chat = chat
        self._settings = settings
        self._chunks = list(chunks)
        self._store = vector_store
        self._embedder = embedder
        self._corpus_label = corpus_label
        self._bm25 = BM25Index(chunks=list(chunks))

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    async def from_chunks(
        cls,
        chunks: list[Chunk],
        *,
        chat: ChatCompleter | None = None,
        settings: RagSettings,
        embedder: Embedder | None = None,
        corpus_label: str = "<in-memory>",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> RagService:
        """Index ``chunks`` in memory (no artifacts written).

        ``Retriever.index`` applies the indexable-kind filter itself, so the
        BM25 index is built over exactly the chunks the vector store kept.
        """
        items = list(chunks)
        if not items:
            raise ConfigurationError("Cannot build a RagService from an empty chunk list.")
        if embedder is None:
            return cls(chat=chat, settings=settings, chunks=items, corpus_label=corpus_label)

        retriever = Retriever(embedder)
        await retriever.index(items, batch_size=batch_size)
        indexed_chunks, vectors = _round_trip_store(retriever.store)
        store: InMemoryVectorStore | None = None
        if indexed_chunks:
            store = InMemoryVectorStore()
            store.add(indexed_chunks, vectors)
        return cls(
            chat=chat,
            settings=settings,
            chunks=indexed_chunks,
            vector_store=store,
            embedder=embedder,
            corpus_label=corpus_label,
        )

    @classmethod
    def from_index(
        cls,
        index_dir: str | Path,
        *,
        chat: ChatCompleter | None = None,
        settings: RagSettings,
        embedder: Embedder | None = None,
    ) -> RagService:
        """Load a built index (``manifest.json`` + ``chunks.jsonl`` + vectors)."""
        directory = Path(index_dir)
        load_index_manifest(directory)

        chunks_path = directory / CHUNKS_NAME
        if not chunks_path.is_file():
            raise RetrievalError(f"Index {directory} is missing {CHUNKS_NAME}; rebuild it.")
        chunks = load_chunks_jsonl(chunks_path)

        store: InMemoryVectorStore | None = None
        vectors_path = directory / VECTORS_NAME
        if vectors_path.is_file():
            stored_chunks, vectors = read_vectors_artifact(vectors_path)
            if [chunk.id for chunk in stored_chunks] != [chunk.id for chunk in chunks]:
                raise RetrievalError(
                    f"{vectors_path} does not match {CHUNKS_NAME} (chunk order differs); "
                    "rebuild the index."
                )
            if chunks:
                store = InMemoryVectorStore()
                store.add(chunks, vectors)

        return cls(
            chat=chat,
            settings=settings,
            chunks=chunks,
            vector_store=store,
            embedder=embedder if store is not None else None,
            corpus_label=str(directory),
        )

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def settings(self) -> RagSettings:
        return self._settings

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def corpus_label(self) -> str:
        return self._corpus_label

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def has_vectors(self) -> bool:
        return self._store is not None and self._embedder is not None and len(self._store) > 0

    def describe(self) -> dict[str, Any]:
        """Small, secret-free description used by ``/health`` and the logs."""
        return {
            "corpus": self._corpus_label,
            "chunks": self.size,
            "vector": self.has_vectors,
            "embedder": type(self._embedder).__name__ if self._embedder is not None else None,
            "mode": self._settings.mode,
            "alpha": self._settings.alpha,
            "top_k": self._settings.top_k,
            "min_score": self._settings.min_score,
            "max_context_chars": self._settings.max_context_chars,
        }

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        *,
        k: int | None = None,
        mode: str | None = None,
        min_score: float | None | _Unset = _UNSET,
    ) -> list[RetrievalHit]:
        """Return up to ``k`` chunks for ``query``, best first."""
        text = query.strip() if isinstance(query, str) else ""
        if not text:
            raise RetrievalError("The query must be a non-empty string.")

        top_k = self._settings.top_k if k is None else _coerce_k(k)
        chosen = str(mode if mode is not None else self._settings.mode).strip().lower()
        if chosen not in {"bm25", "vector", "hybrid"}:
            raise RetrievalError(
                f"Unknown retrieval mode {chosen!r}; expected one of bm25, vector, hybrid."
            )

        threshold = self._settings.min_score if isinstance(min_score, _Unset) else min_score

        if chosen == "hybrid" and not self.has_vectors:
            logger.info("Hybrid search requested without a vector index; using bm25 only.")
            chosen = "bm25"

        if chosen == "vector":
            hits = await self._vector_search(text, top_k)
        elif chosen == "bm25":
            hits = self._bm25_search(text, top_k)
        else:
            hits = await self._hybrid_search(text, top_k)

        if threshold is not None:
            cutoff = float(threshold)
            hits = [hit for hit in hits if hit.score >= cutoff]
        return hits

    async def _vector_search(self, text: str, k: int) -> list[RetrievalHit]:
        if not self.has_vectors:
            raise RetrievalError(
                "Vector search is unavailable: no embedding index is loaded. "
                "Build the index or use mode='bm25'."
            )
        assert self._store is not None  # narrowed by has_vectors
        assert self._embedder is not None
        try:
            [vector] = await self._embedder.embed([text])
            results = self._store.search(vector, k)
        except ZeroVectorError:
            # Query has no vocabulary overlap; return empty list so hybrid
            # can fall back to BM25 and vector callers get an empty result.
            return []
        return [
            RetrievalHit(chunk=result.chunk, score=result.score, source="vector")
            for result in results
        ]

    def _bm25_search(self, text: str, k: int) -> list[RetrievalHit]:
        return [
            RetrievalHit(chunk=result.chunk, score=result.score, source="bm25")
            for result in self._bm25.search(text, k=k, min_score=0.0)
        ]

    async def _hybrid_search(self, text: str, k: int) -> list[RetrievalHit]:
        pool = max(k * HYBRID_POOL_MULTIPLIER, k)
        vector_hits = await self._vector_search(text, pool)
        bm25_hits = self._bm25_search(text, pool)

        chunks: dict[str, Chunk] = {}
        vector_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        for hit in vector_hits:
            chunks.setdefault(hit.chunk.id, hit.chunk)
            vector_scores[hit.chunk.id] = hit.score
        for hit in bm25_hits:
            chunks.setdefault(hit.chunk.id, hit.chunk)
            bm25_scores[hit.chunk.id] = hit.score

        vector_norm = _minmax(vector_scores)
        bm25_norm = _minmax(bm25_scores)
        alpha = min(max(self._settings.alpha, 0.0), 1.0)
        fused = {
            chunk_id: alpha * vector_norm.get(chunk_id, 0.0)
            + (1.0 - alpha) * bm25_norm.get(chunk_id, 0.0)
            for chunk_id in chunks
        }
        # A chunk with a zero fused score was retrieved with no positive
        # signal from either retriever; it would only add noise.
        ranked = sorted(
            (item for item in fused.items() if item[1] > 0.0),
            key=lambda item: (-item[1], item[0]),
        )[:k]
        return [
            RetrievalHit(chunk=chunks[chunk_id], score=score, source="hybrid")
            for chunk_id, score in ranked
        ]

    # ------------------------------------------------------------------
    # answering
    # ------------------------------------------------------------------
    async def ask(
        self,
        question: str,
        *,
        k: int | None = None,
        mode: str | None = None,
    ) -> AskResult:
        """Retrieve, then answer strictly from the retrieved chunks."""
        if self._chat is None:
            raise ConfigurationError(
                "RagService was constructed without a chat provider. "
                "Provide a ChatCompleter to use ask()."
            )
        text = question.strip() if isinstance(question, str) else ""
        if not text:
            raise RetrievalError("The question must be a non-empty string.")

        hits = await self.search(text, k=k, mode=mode)
        if not hits:
            return AskResult(answer=NOT_FOUND, grounded=False, sources=[], retrieved=[])

        prompt, prompt_sources = build_rag_prompt(
            text,
            hits,
            max_context_chars=self._settings.max_context_chars,
        )
        if not prompt_sources:
            return AskResult(answer=NOT_FOUND, grounded=False, sources=[], retrieved=hits)

        answer = await self._chat.complete(prompt, system=RAG_SYSTEM_PROMPT)
        answer = (answer or "").strip()
        if not answer:
            raise MalformedResponseError("The model returned an empty answer.")
        if is_not_found(answer):
            return AskResult(answer=NOT_FOUND, grounded=False, sources=[], retrieved=hits)

        cited = cited_numbers(answer)
        sources = [source.to_citation(cited=source.number in cited) for source in prompt_sources]
        return AskResult(answer=answer, grounded=True, sources=sources, retrieved=hits)


# ---------------------------------------------------------------------------
# index artifacts
# ---------------------------------------------------------------------------
def read_vectors_artifact(path: str | Path) -> tuple[list[Chunk], list[list[float]]]:
    """Read the ``vectors.json`` artifact written by ``InMemoryVectorStore.save``."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = data["chunks"]
        vectors = [[float(value) for value in vector] for vector in data["vectors"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RetrievalError(f"Invalid vector index artifact {path}: {exc!r}") from exc

    if not isinstance(rows, list) or len(rows) != len(vectors):
        raise RetrievalError(f"Vector index artifact {path} is inconsistent.")

    chunks: list[Chunk] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RetrievalError(f"Vector index artifact {path} has a malformed chunk.")
        try:
            chunks.append(Chunk(**{key: row[key] for key in _CHUNK_FIELDS if key in row}))
        except TypeError as exc:
            raise RetrievalError(
                f"Vector index artifact {path} has a malformed chunk: {exc}"
            ) from exc
    return chunks, vectors


def load_index_manifest(index_dir: str | Path) -> dict[str, Any]:
    """Read and validate ``manifest.json`` for ``index_dir``."""
    directory = Path(index_dir)
    path = directory / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"No QuantRAG index at {str(directory)!r} (missing {MANIFEST_NAME}). "
            "Build one with: uv run python -m quantrag.index_build "
            "--chunks <chunks.jsonl> --out <index-dir> --embedder local"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Cannot read {path}: {exc!r}") from exc

    if not isinstance(data, dict) or data.get("format") != INDEX_FORMAT_VERSION:
        raise RetrievalError(
            f"{path} is not a version-{INDEX_FORMAT_VERSION} QuantRAG index; rebuild it."
        )
    return data


def _round_trip_store(store: InMemoryVectorStore) -> tuple[list[Chunk], list[list[float]]]:
    """Persist then re-read a store to learn which chunks it actually indexed.

    ``Retriever.index`` owns the indexable-kind filter, so reading the
    serialised store is the only supported way to recover that subset without
    duplicating the rule in this module.
    """
    with tempfile.TemporaryDirectory(prefix="quantrag-index-") as tmp:
        path = Path(tmp) / VECTORS_NAME
        store.save(path)
        return read_vectors_artifact(path)


# ---------------------------------------------------------------------------
# embedder / service factories
# ---------------------------------------------------------------------------
def build_embedder(
    kind: str,
    *,
    chunks: list[Chunk],
    env: Mapping[str, str] | None = None,
) -> Embedder:
    """Create an embedder fitted to ``chunks`` (``local``) or backed by OpenRouter."""
    if kind == "local":
        try:
            return TfidfEmbedder().fit([chunk.text for chunk in chunks])
        except ValueError as exc:
            raise ConfigurationError(f"Cannot fit the local TF-IDF embedder: {exc}") from exc
    if kind == "openrouter":
        source = os.environ if env is None else env
        return OpenRouterEmbedder(
            load_settings(env, load_env_file=False), load_embedding_model(source)
        )
    raise ConfigurationError(f"Unknown embedder kind {kind!r}; expected 'openrouter' or 'local'.")


def load_index_embedder(
    index_dir: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> Embedder:
    """Recreate the *query* embedder that matches a built index."""
    directory = Path(index_dir)
    manifest = load_index_manifest(directory)
    meta = manifest.get("embedder")
    meta = meta if isinstance(meta, dict) else {}
    kind = str(meta.get("kind") or "openrouter")

    if kind == "local":
        path = directory / EMBEDDER_STATE_NAME
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrievalError(f"Cannot read {path}: {exc!r}") from exc
        try:
            return TfidfEmbedder.from_state(state)
        except (TypeError, ValueError) as exc:
            raise RetrievalError(f"Invalid TF-IDF state in {path}: {exc}") from exc

    source = os.environ if env is None else env
    model = str(meta.get("model") or "").strip() or load_embedding_model(source)
    return OpenRouterEmbedder(load_settings(env, load_env_file=False), model)


async def build_service(
    *,
    chat: ChatCompleter | None = None,
    settings: RagSettings | None = None,
    env: Mapping[str, str] | None = None,
    embedder: Embedder | None = None,
    llm_provider: str | None = None,
) -> RagService:
    """Build the one service instance used by the MCP server and the HTTP app.

    If ``chat`` is not provided, attempts to build an LLM provider from
    ``env`` (or the process environment) using the provider specified by
    ``llm_provider`` (or QUANTRAG_LLM_PROVIDER env var, default: openrouter).
    If the API key is not configured, the service is created without a chat
    provider -- retrieval works, but ``ask()`` will raise a clear configuration
    error.
    """
    rag_settings = settings or load_rag_settings(env, load_env_file=False)

    # Build chat provider if not explicitly provided and credentials are available
    if chat is None:
        try:
            # Use provider factory to create the appropriate LLM provider
            provider_env = env if env is not None else os.environ
            chat = create_provider_from_env(provider_env)
        except ConfigurationError:
            # No API key configured; retrieval-only mode
            chat = None

    if (rag_settings.index_dir / MANIFEST_NAME).is_file():
        return RagService.from_index(
            rag_settings.index_dir,
            chat=chat,
            settings=rag_settings,
            embedder=embedder or load_index_embedder(rag_settings.index_dir, env=env),
        )

    corpus = rag_settings.corpus_path
    if corpus.is_file():
        chunks = load_chunks_jsonl(corpus)
        return await RagService.from_chunks(
            chunks,
            chat=chat,
            settings=rag_settings,
            embedder=embedder or build_embedder(rag_settings.embedder, chunks=chunks, env=env),
            corpus_label=str(corpus),
        )

    raise ConfigurationError(
        "No knowledge base is available: "
        f"{rag_settings.index_dir} has no {MANIFEST_NAME} and the corpus {corpus} "
        "does not exist. Build the index with: uv run python -m quantrag.index_build "
        "--chunks <chunks.jsonl> --out <index-dir> --embedder local"
    )


def _coerce_k(k: int) -> int:
    try:
        value = int(k)
    except (TypeError, ValueError) as exc:
        raise RetrievalError(f"k must be an integer, got {k!r}.") from exc
    if value <= 0:
        raise RetrievalError("k must be greater than 0.")
    return value


def _minmax(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalise to 0..1; a single distinct value maps to 1.0."""
    if not values:
        return {}
    lowest = min(values.values())
    highest = max(values.values())
    if highest == lowest:
        return dict.fromkeys(values, 1.0)
    span = highest - lowest
    return {key: (value - lowest) / span for key, value in values.items()}
