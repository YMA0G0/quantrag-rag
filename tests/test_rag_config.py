"""Offline tests for the retrieval/RAG settings loaded from the environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantrag.config import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RAG_TOP_K,
    DEFAULT_RETRIEVAL_MODE,
    RagSettings,
    load_rag_settings,
)
from quantrag.errors import ConfigurationError


def test_defaults_are_conservative() -> None:
    settings = load_rag_settings({}, load_env_file=False)

    assert isinstance(settings, RagSettings)
    assert settings.mode == DEFAULT_RETRIEVAL_MODE == "hybrid"
    assert settings.alpha == DEFAULT_HYBRID_ALPHA == 0.5
    assert settings.top_k == DEFAULT_RAG_TOP_K == 5
    assert settings.embedder == "openrouter"
    assert settings.max_context_chars == 6000
    # The threshold must stay unset until it is calibrated (see README).
    assert settings.min_score is None
    assert settings.index_dir == Path.cwd() / "data" / "index"
    assert settings.corpus_path == Path.cwd() / "data" / "chunks.jsonl"


def test_environment_overrides() -> None:
    settings = load_rag_settings(
        {
            "QUANTRAG_RETRIEVAL_MODE": "BM25",
            "QUANTRAG_HYBRID_ALPHA": "0.25",
            "QUANTRAG_TOP_K": "8",
            "QUANTRAG_MIN_SCORE": "0.31",
            "QUANTRAG_MAX_CONTEXT_CHARS": "1200",
            "QUANTRAG_EMBEDDER": "local",
            "QUANTRAG_INDEX_DIR": "build/index",
            "QUANTRAG_CORPUS_PATH": "build/chunks.jsonl",
        },
        root="/tmp/root",
        load_env_file=False,
    )

    assert settings.mode == "bm25"
    assert settings.alpha == pytest.approx(0.25)
    assert settings.top_k == 8
    assert settings.min_score == pytest.approx(0.31)
    assert settings.max_context_chars == 1200
    assert settings.embedder == "local"
    assert settings.index_dir == Path("/tmp/root/build/index")
    assert settings.corpus_path == Path("/tmp/root/build/chunks.jsonl")


def test_absolute_paths_are_kept() -> None:
    settings = load_rag_settings(
        {"QUANTRAG_INDEX_DIR": "/var/quantrag/index"},
        root="/tmp/root",
        load_env_file=False,
    )

    assert settings.index_dir == Path("/var/quantrag/index")


def test_blank_values_fall_back_to_defaults() -> None:
    settings = load_rag_settings(
        {
            "QUANTRAG_RETRIEVAL_MODE": "   ",
            "QUANTRAG_HYBRID_ALPHA": "",
            "QUANTRAG_TOP_K": "  ",
            "QUANTRAG_MIN_SCORE": "",
            "QUANTRAG_EMBEDDER": "",
        },
        load_env_file=False,
    )

    assert settings.mode == "hybrid"
    assert settings.alpha == 0.5
    assert settings.top_k == 5
    assert settings.min_score is None
    assert settings.embedder == "openrouter"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"QUANTRAG_RETRIEVAL_MODE": "semantic"}, "QUANTRAG_RETRIEVAL_MODE"),
        ({"QUANTRAG_EMBEDDER": "faiss"}, "QUANTRAG_EMBEDDER"),
        ({"QUANTRAG_HYBRID_ALPHA": "1.5"}, "QUANTRAG_HYBRID_ALPHA"),
        ({"QUANTRAG_HYBRID_ALPHA": "half"}, "QUANTRAG_HYBRID_ALPHA"),
        ({"QUANTRAG_TOP_K": "0"}, "QUANTRAG_TOP_K"),
        ({"QUANTRAG_TOP_K": "many"}, "QUANTRAG_TOP_K"),
        ({"QUANTRAG_MAX_CONTEXT_CHARS": "-5"}, "QUANTRAG_MAX_CONTEXT_CHARS"),
        ({"QUANTRAG_MIN_SCORE": "high"}, "QUANTRAG_MIN_SCORE"),
    ],
)
def test_invalid_values_raise(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        load_rag_settings(env, load_env_file=False)

    assert message in str(excinfo.value)
