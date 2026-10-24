"""Shared pytest fixtures and helpers for the QuantRAG test suite.

Every test runs fully offline: HTTP is mocked with ``httpx.MockTransport`` and
no real OpenRouter credentials are used.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from quantrag.config import Settings

TEST_API_KEY = "test-key-not-a-real-secret"
TEST_MODEL = "test/model-v1"


@pytest.fixture
def settings() -> Settings:
    """Settings with dummy values; the API key is never a real credential."""
    return Settings(api_key=TEST_API_KEY, model=TEST_MODEL)


def make_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def json_response(payload: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=payload)


def chat_completion(content: str, model: str = TEST_MODEL) -> dict[str, Any]:
    return {
        "id": "gen-123",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    }


def error_payload(message: str) -> dict[str, Any]:
    return {"error": {"message": message, "code": 400}}


def dumps(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")
