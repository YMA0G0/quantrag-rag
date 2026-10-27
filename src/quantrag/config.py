"""Environment-driven configuration for QuantRAG.

Local development uses a ``.env`` file (see ``.env.example``); in production
the same values are expected to be provided as real environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigurationError

ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_MODEL = "OPENROUTER_MODEL"
ENV_BASE_URL = "OPENROUTER_BASE_URL"
ENV_TIMEOUT = "OPENROUTER_TIMEOUT_SECONDS"
ENV_MAX_TOKENS = "OPENROUTER_MAX_TOKENS"
ENV_APP_NAME = "OPENROUTER_APP_NAME"
ENV_APP_URL = "OPENROUTER_APP_URL"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_APP_NAME = "QuantRAG"


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime configuration.

    The API key is excluded from ``repr`` so that it cannot leak into logs or
    tracebacks.
    """

    api_key: str = field(repr=False)
    model: str
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    app_name: str = DEFAULT_APP_NAME
    app_url: str | None = None

    @property
    def chat_completions_url(self) -> str:
        """Absolute URL of the OpenRouter Chat Completions endpoint."""
        return f"{self.base_url.rstrip('/')}/chat/completions"


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: str | Path | None = None,
    load_env_file: bool = True,
) -> Settings:
    """Build :class:`Settings` from the environment.

    Args:
        env: Mapping to read from. Defaults to ``os.environ``.
        dotenv_path: Explicit path to a ``.env`` file. ``None`` searches the
            working directory upwards (the python-dotenv default).
        load_env_file: Set to ``False`` to skip reading ``.env`` entirely.

    Raises:
        ConfigurationError: If a required variable is missing or invalid.
    """
    if load_env_file:
        # python-dotenv never overrides variables that are already exported,
        # so real environment variables always win over the .env file.
        load_dotenv(dotenv_path)

    source: Mapping[str, str] = os.environ if env is None else env

    api_key = (source.get(ENV_API_KEY) or "").strip()
    if not api_key:
        raise ConfigurationError(
            f"{ENV_API_KEY} is not set. Export it or add it to a .env file "
            "(see .env.example). The key is never logged."
        )

    model = (source.get(ENV_MODEL) or "").strip()
    if not model:
        raise ConfigurationError(
            f"{ENV_MODEL} is not set. Set it to an OpenRouter model id that is "
            "available on your account (for example: openai/gpt-4o-mini)."
        )

    base_url = (source.get(ENV_BASE_URL) or "").strip().rstrip("/") or DEFAULT_BASE_URL
    if not base_url.startswith(("http://", "https://")):
        raise ConfigurationError(f"{ENV_BASE_URL} must be an http(s) URL, got {base_url!r}.")

    timeout = _parse_timeout(source.get(ENV_TIMEOUT))
    max_tokens = _parse_max_tokens(source.get(ENV_MAX_TOKENS))
    app_name = (source.get(ENV_APP_NAME) or "").strip() or DEFAULT_APP_NAME
    app_url = (source.get(ENV_APP_URL) or "").strip() or None

    return Settings(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout,
        max_tokens=max_tokens,
        app_name=app_name,
        app_url=app_url,
    )


def _parse_timeout(raw: str | None) -> float:
    if raw is None or not raw.strip():
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{ENV_TIMEOUT} must be a number, got {raw!r}.") from exc
    if value <= 0:
        raise ConfigurationError(f"{ENV_TIMEOUT} must be greater than 0, got {value}.")
    return value


def _parse_max_tokens(raw: str | None) -> int:
    """Upper bound on the completion length (caps cost per request)."""
    if raw is None or not raw.strip():
        return DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{ENV_MAX_TOKENS} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ConfigurationError(f"{ENV_MAX_TOKENS} must be greater than 0, got {value}.")
    return value


# ---------------------------------------------------------------------------
# Retrieval / RAG configuration
# ---------------------------------------------------------------------------
ENV_RAG_INDEX_DIR = "QUANTRAG_INDEX_DIR"
ENV_RAG_CORPUS = "QUANTRAG_CORPUS_PATH"
ENV_RAG_MODE = "QUANTRAG_RETRIEVAL_MODE"
ENV_RAG_ALPHA = "QUANTRAG_HYBRID_ALPHA"
ENV_RAG_TOP_K = "QUANTRAG_TOP_K"
ENV_RAG_MIN_SCORE = "QUANTRAG_MIN_SCORE"
ENV_RAG_MAX_CONTEXT_CHARS = "QUANTRAG_MAX_CONTEXT_CHARS"
ENV_RAG_EMBEDDER = "QUANTRAG_EMBEDDER"

DEFAULT_RAG_INDEX_DIR = "data/index"
DEFAULT_RAG_CORPUS_PATH = "data/chunks.jsonl"
DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_HYBRID_ALPHA = 0.5
DEFAULT_RAG_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 6000
DEFAULT_RAG_EMBEDDER = "openrouter"
RETRIEVAL_MODES = ("bm25", "vector", "hybrid")
EMBEDDER_KINDS = ("openrouter", "local")


@dataclass(frozen=True, slots=True)
class RagSettings:
    """Retrieval and RAG-answering configuration.

    ``min_score`` is deliberately ``None`` by default: the threshold has to be
    calibrated against the evaluation set (see README) and must never be
    invented.  ``mode`` and ``alpha`` are placeholders until the evaluation
    picks the winner.
    """

    index_dir: Path = Path(DEFAULT_RAG_INDEX_DIR)
    corpus_path: Path = Path(DEFAULT_RAG_CORPUS_PATH)
    mode: str = DEFAULT_RETRIEVAL_MODE
    alpha: float = DEFAULT_HYBRID_ALPHA
    top_k: int = DEFAULT_RAG_TOP_K
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    embedder: str = DEFAULT_RAG_EMBEDDER
    min_score: float | None = None


def load_rag_settings(
    env: Mapping[str, str] | None = None,
    *,
    root: str | Path | None = None,
    load_env_file: bool = True,
) -> RagSettings:
    """Build :class:`RagSettings` from the environment.

    Relative paths are resolved against ``root`` (default: the working
    directory) so the same values work from any launch directory.
    """
    if load_env_file:
        load_dotenv()

    source: Mapping[str, str] = os.environ if env is None else env
    base = Path.cwd() if root is None else Path(root)

    mode = (source.get(ENV_RAG_MODE) or "").strip().lower() or DEFAULT_RETRIEVAL_MODE
    if mode not in RETRIEVAL_MODES:
        raise ConfigurationError(
            f"{ENV_RAG_MODE} must be one of {', '.join(RETRIEVAL_MODES)}, got {mode!r}."
        )

    embedder = (source.get(ENV_RAG_EMBEDDER) or "").strip().lower() or DEFAULT_RAG_EMBEDDER
    if embedder not in EMBEDDER_KINDS:
        raise ConfigurationError(
            f"{ENV_RAG_EMBEDDER} must be one of {', '.join(EMBEDDER_KINDS)}, got {embedder!r}."
        )

    return RagSettings(
        index_dir=_resolve_path(source.get(ENV_RAG_INDEX_DIR), base, DEFAULT_RAG_INDEX_DIR),
        corpus_path=_resolve_path(source.get(ENV_RAG_CORPUS), base, DEFAULT_RAG_CORPUS_PATH),
        mode=mode,
        alpha=_parse_alpha(source.get(ENV_RAG_ALPHA)),
        top_k=_parse_positive_int(source.get(ENV_RAG_TOP_K), ENV_RAG_TOP_K, DEFAULT_RAG_TOP_K),
        max_context_chars=_parse_positive_int(
            source.get(ENV_RAG_MAX_CONTEXT_CHARS),
            ENV_RAG_MAX_CONTEXT_CHARS,
            DEFAULT_MAX_CONTEXT_CHARS,
        ),
        embedder=embedder,
        min_score=_parse_optional_float(source.get(ENV_RAG_MIN_SCORE), ENV_RAG_MIN_SCORE),
    )


def _resolve_path(raw: str | None, base: Path, default: str) -> Path:
    text = (raw or "").strip() or default
    path = Path(text).expanduser()
    return path if path.is_absolute() else (base / path)


def _parse_alpha(raw: str | None) -> float:
    if raw is None or not raw.strip():
        return DEFAULT_HYBRID_ALPHA
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{ENV_RAG_ALPHA} must be a number, got {raw!r}.") from exc
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{ENV_RAG_ALPHA} must be between 0 and 1, got {value}.")
    return value


def _parse_positive_int(raw: str | None, name: str, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than 0, got {value}.")
    return value


def _parse_optional_float(raw: str | None, name: str) -> float | None:
    """Parse an optional threshold; blank/unset stays ``None`` (no filtering)."""
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from exc
