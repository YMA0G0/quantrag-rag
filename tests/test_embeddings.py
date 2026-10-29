"""Tests for the OpenRouter embeddings client (HTTP fully mocked)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from quantrag.config import Settings
from quantrag.embeddings import ENV_EMBEDDING_MODEL, OpenRouterEmbedder, load_embedding_model
from quantrag.errors import (
    AuthenticationError,
    ConfigurationError,
    MalformedResponseError,
    RateLimitError,
    UpstreamConnectionError,
    UpstreamStatusError,
    UpstreamTimeoutError,
)

from .conftest import TEST_API_KEY, error_payload, json_response, make_transport

pytestmark = pytest.mark.anyio

EMBED_MODEL = "test/embed-v1"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def embedding_payload(vectors: list[list[float]]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
        ],
    }


def build(handler, settings: Settings) -> OpenRouterEmbedder:
    return OpenRouterEmbedder(settings, EMBED_MODEL, transport=make_transport(handler))


def test_model_is_read_from_environment() -> None:
    assert load_embedding_model({ENV_EMBEDDING_MODEL: " vendor/embed "}) == "vendor/embed"


def test_missing_model_raises() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        load_embedding_model({})
    assert ENV_EMBEDDING_MODEL in str(excinfo.value)


async def test_returns_vectors_in_input_order(settings: Settings) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        # answer out of order on purpose: the client must sort by ``index``
        body = embedding_payload([[1.0, 0.0], [0.0, 1.0]])
        body["data"].reverse()
        return json_response(body)

    vectors = await build(handler, settings).embed(["a", "b"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert captured["auth"] == f"Bearer {TEST_API_KEY}"
    assert captured["payload"] == {"model": EMBED_MODEL, "input": ["a", "b"]}


async def test_empty_input_makes_no_request(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert await build(handler, settings).embed([]) == []


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, AuthenticationError), (403, AuthenticationError), (429, RateLimitError)],
)
async def test_auth_and_rate_limit_errors(
    settings: Settings, status: int, error_type: type[Exception]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(error_payload("nope"), status_code=status)

    with pytest.raises(error_type):
        await build(handler, settings).embed(["a"])


async def test_authentication_error_401_includes_provider_detail(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(error_payload("Invalid API key: quota exceeded"), status_code=401)

    with pytest.raises(AuthenticationError) as excinfo:
        await build(handler, settings).embed(["a"])
    assert excinfo.value.status_code == 401
    assert "quota exceeded" in str(excinfo.value)
    assert TEST_API_KEY not in str(excinfo.value)


async def test_authentication_error_403_includes_provider_detail(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(error_payload("Forbidden: model access denied"), status_code=403)

    with pytest.raises(AuthenticationError) as excinfo:
        await build(handler, settings).embed(["a"])
    assert excinfo.value.status_code == 403
    assert "model access denied" in str(excinfo.value)
    assert TEST_API_KEY not in str(excinfo.value)


async def test_server_error_never_contains_the_api_key(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(error_payload(f"key {TEST_API_KEY} bad"), status_code=500)

    with pytest.raises(UpstreamStatusError) as excinfo:
        await build(handler, settings).embed(["a"])
    assert excinfo.value.status_code == 500
    assert TEST_API_KEY not in str(excinfo.value)
    assert "redacted" in str(excinfo.value)


async def test_timeout_and_connection_failures_are_wrapped(settings: Settings) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(UpstreamTimeoutError):
        await build(timeout, settings).embed(["a"])
    with pytest.raises(UpstreamConnectionError):
        await build(refused, settings).embed(["a"])


@pytest.mark.parametrize(
    "body",
    [
        {"data": []},  # wrong number of vectors
        {"data": [{"index": 0}]},  # missing embedding
        {"data": [{"index": 0, "embedding": ["x"]}]},  # non-numeric
        ["not", "an", "object"],
    ],
    ids=["count", "missing", "non-numeric", "not-object"],
)
async def test_malformed_responses_are_reported(settings: Settings, body: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(body)

    with pytest.raises(MalformedResponseError):
        await build(handler, settings).embed(["a"])


async def test_non_json_response_is_reported(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, content=b"<html>oops</html>")

    with pytest.raises(MalformedResponseError):
        await build(handler, settings).embed(["a"])
