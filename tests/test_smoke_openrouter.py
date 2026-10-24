"""Optional smoke test against the real OpenRouter API.

This is the only test in the suite that touches the network, and it is
disabled by default so that a plain ``uv run pytest`` stays offline, free of
charges and fully deterministic.

Enable it explicitly with real credentials:

    QUANTRAG_SMOKE=1 uv run pytest -m smoke -q

It is skipped (not failed) when the flag is missing or the credentials are
not configured.
"""

from __future__ import annotations

import os

import pytest

from quantrag.config import Settings, load_settings
from quantrag.errors import ConfigurationError
from quantrag.openrouter import OpenRouterClient

_ENABLED = os.environ.get("QUANTRAG_SMOKE", "").strip().lower() in {"1", "true", "yes", "on"}

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.smoke,
    pytest.mark.skipif(not _ENABLED, reason="set QUANTRAG_SMOKE=1 to run the live OpenRouter test"),
]


def _settings_or_skip() -> Settings:
    """Load real configuration, skipping the test when credentials are absent."""
    try:
        return load_settings()
    except ConfigurationError as exc:
        pytest.skip(f"smoke test enabled but configuration is incomplete: {exc}")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_real_openrouter_roundtrip() -> None:
    """Send one tiny request to the configured model and check the answer."""
    settings = _settings_or_skip()
    client = OpenRouterClient(settings)
    try:
        answer = await client.complete("Reply with exactly the word: pong")
    finally:
        await client.aclose()

    assert answer, "OpenRouter returned an empty answer"
    assert "pong" in answer.lower()
