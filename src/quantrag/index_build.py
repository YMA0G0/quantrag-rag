"""Build a portable QuantRAG index from a ``chunks.jsonl`` file.

Artifacts written to ``--out``::

    manifest.json   format version, counters, embedder identity, timestamp
    chunks.jsonl    the chunks that were actually indexed (bodies verbatim)
    vectors.json    chunk order + embedding vectors (InMemoryVectorStore format)
    embedder.json   fitted TF-IDF vocabulary/IDF (local embedder only)

``quantrag.service.build_service`` reads ``manifest.json`` first; vector and
hybrid search additionally need ``vectors.json``.  ``embedder.json`` is what
lets a *query* be embedded with the same TF-IDF state used at index time --
without it, query and document vectors would live in different spaces.

The index is built offline with ``--embedder local``; the ``openrouter`` kind
is there for the full corpus and needs ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import load_rag_settings, load_settings
from .embeddings import Embedder, OpenRouterEmbedder, load_embedding_model
from .errors import ConfigurationError, QuantRagError, RetrievalError
from .local_embeddings import TfidfEmbedder
from .retriever import Chunk, Retriever, load_chunks_jsonl
from .service import (
    CHUNKS_NAME,
    EMBEDDER_STATE_NAME,
    INDEX_FORMAT_VERSION,
    MANIFEST_NAME,
    VECTORS_NAME,
    read_vectors_artifact,
)

DEFAULT_BATCH_SIZE = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quantrag.index_build",
        description="Embed chunks.jsonl and write a QuantRAG index directory.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Input chunks.jsonl (default: $QUANTRAG_CORPUS_PATH or data/chunks.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output index directory (default: $QUANTRAG_INDEX_DIR or data/index)",
    )
    parser.add_argument(
        "--embedder",
        choices=("local", "openrouter"),
        default=None,
        help="Embedding backend (default: $QUANTRAG_EMBEDDER or openrouter)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the embedding model id (default: $QUANTRAG_EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a non-empty output directory",
    )
    return parser


def _chunk_row(chunk: Chunk) -> dict[str, Any]:
    return asdict(chunk)


def _make_embedder(
    kind: str,
    *,
    chunks: list[Chunk],
    model: str | None,
    env: Mapping[str, str] | None,
) -> Embedder:
    if kind == "local":
        try:
            return TfidfEmbedder().fit([chunk.text for chunk in chunks])
        except ValueError as exc:
            raise ConfigurationError(f"Cannot fit the local TF-IDF embedder: {exc}") from exc
    if kind == "openrouter":
        source = os.environ if env is None else env
        settings = load_settings(env, load_env_file=False)
        return OpenRouterEmbedder(settings, model or load_embedding_model(source))
    raise ConfigurationError(f"Unknown embedder kind {kind!r}; expected 'openrouter' or 'local'.")


def _prepare_output(out_dir: Path, *, force: bool) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise ConfigurationError(f"Output path {out_dir} exists and is not a directory.")
    if out_dir.is_dir() and any(out_dir.iterdir()) and not force:
        raise ConfigurationError(
            f"Output directory {out_dir} is not empty; pass --force to overwrite it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)


async def build_index(
    *,
    chunks_path: str | Path,
    out_dir: str | Path,
    embedder_kind: str = "local",
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Index ``chunks_path`` into ``out_dir`` and return the written manifest."""
    source = Path(chunks_path)
    if not source.is_file():
        raise ConfigurationError(f"Chunks file {source} does not exist.")
    if batch_size <= 0:
        raise ConfigurationError(f"batch_size must be greater than 0, got {batch_size}.")

    chunks = load_chunks_jsonl(source)
    if not chunks:
        raise ConfigurationError(f"No chunks found in {source}.")

    out = Path(out_dir)
    _prepare_output(out, force=force)

    embedder = _make_embedder(embedder_kind, chunks=chunks, model=model, env=env)
    retriever = Retriever(embedder)
    await retriever.index(chunks, batch_size=batch_size)

    # ``Retriever.index`` owns the indexable-kind filter, so the artifact is
    # first written to a scratch directory to learn which chunks survived.
    with tempfile.TemporaryDirectory(prefix="quantrag-build-") as tmp:
        scratch = Path(tmp) / VECTORS_NAME
        retriever.store.save(scratch)
        indexed_chunks, _vectors = read_vectors_artifact(scratch)

    if not indexed_chunks:
        raise RetrievalError(
            f"None of the {len(chunks)} chunks in {source} are indexable; "
            "refusing to write an empty index."
        )

    with (out / CHUNKS_NAME).open("w", encoding="utf-8") as handle:
        for chunk in indexed_chunks:
            handle.write(json.dumps(_chunk_row(chunk), ensure_ascii=False) + "\n")

    retriever.store.save(out / VECTORS_NAME)

    if isinstance(embedder, TfidfEmbedder):
        (out / EMBEDDER_STATE_NAME).write_text(
            json.dumps(embedder.state(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest: dict[str, Any] = {
        "format": INDEX_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chunks_file": CHUNKS_NAME,
        "vectors_file": VECTORS_NAME,
        "embedder": {
            "kind": embedder_kind,
            "model": getattr(embedder, "model_name", None) or getattr(embedder, "model", None),
            "dimension": getattr(embedder, "dimension", None),
        },
        "chunks_total": len(chunks),
        "chunks_indexed": len(indexed_chunks),
        "batch_size": batch_size,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        settings = load_rag_settings(load_env_file=True)
        manifest = asyncio.run(
            build_index(
                chunks_path=args.chunks or settings.corpus_path,
                out_dir=args.out or settings.index_dir,
                embedder_kind=args.embedder or settings.embedder,
                model=args.model,
                batch_size=args.batch_size,
                force=args.force,
            )
        )
    except QuantRagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"indexed {manifest['chunks_indexed']}/{manifest['chunks_total']} chunks "
        f"with embedder {manifest['embedder']['kind']} "
        f"-> {args.out or settings.index_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
