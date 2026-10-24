"""Tests for the OpenRouter HTTP client.

All HTTP traffic is mocked through ``httpx.MockTransport``: the suite never
touches the network and never needs real credentials.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from quantrag.config import Settings
from quantrag.errors import (
    AuthenticationError,
    MalformedResponseError,
    RateLimitError,
    UpstreamConnectionError,
    UpstreamError,
    UpstreamStatusError,
    UpstreamTimeoutError,
)
from quantrag.openrouter import OpenRouterClient

from .conftest import (
    TEST_API_KEY,
    TEST_MODEL,
    chat_completion,
    error_payload,
    json_response,
    make_transport,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def build_client(handler, settings: Settings) -> OpenRouterClient:
    return OpenRouterClient(settings, transport=make_transport(handler))


async def test_returns_model_text(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_completion("The Sharpe ratio is 1.4."))

    client = build_client(handler, settings)
    answer = await client.complete("What is the Sharpe ratio?")
    assert answer == "The Sharpe ratio is 1.4."


async def test_request_uses_configured_model_and_key(settings: Settings) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["title"] = request.headers.get("X-Title")
        captured["payload"] = json.loads(request.content)
        return json_response(chat_completion("ok"))

    client = build_client(handler, settings)
    await client.complete("hello", system="be terse")

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["auth"] == f"Bearer {TEST_API_KEY}"
    assert captured["title"] == "QuantRAG"
    assert captured["payload"]["model"] == TEST_MODEL
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["messages"][-1] == {"role": "user", "content": "hello"}


async def test_max_tokens_is_sent_in_payload(settings: Settings) -> None:
    """The completion length is bounded so upstream credit reservation is capped."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return json_response(chat_completion("ok"))

    client = OpenRouterClient(
        Settings(api_key=TEST_API_KEY, model=TEST_MODEL, max_tokens=64),
        transport=make_transport(handler),
    )
    await client.complete("hi")

    assert captured["payload"]["max_tokens"] == 64
    # default settings also bound the request
    client_default = build_client(handler, settings)
    await client_default.complete("hi")
    assert captured["payload"]["max_tokens"] == 1024


async def test_system_prompt_is_included_when_requested(settings: Settings) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return json_response(chat_completion("ok"))

    client = build_client(handler, settings)
    await client.complete("hello", system="You are QuantRAG.")
    messages = captured["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "You are QuantRAG."}
    assert len(messages) == 2


async def test_custom_base_url_is_used() -> None:
    settings = Settings(
        api_key=TEST_API_KEY,
        model=TEST_MODEL,
        base_url="https://proxy.invalid/openrouter/v1",
    )
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return json_response(chat_completion("ok"))

    client = build_client(handler, settings)
    await client.complete("hi")
    assert captured["url"] == "https://proxy.invalid/openrouter/v1/chat/completions"


async def test_authentication_error_on_401(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(error_payload("Invalid API key"), status_code=401)

    client = build_client(handler, settings)
    with pytest.raises(AuthenticationError) as excinfo:
        await client.complete("hi")
    assert excinfo.value.status_code == 401
    assert TEST_API_KEY not in str(excinfo.value)


async def test_authentication_error_on_403(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(error_payload("Forbidden"), status_code=403)

    client = build_client(handler, settings)
    with pytest.raises(AuthenticationError) as excinfo:
        await client.complete("hi")
    assert excinfo.value.status_code == 403


async def test_rate_limit_error_on_429(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            json=error_payload("Rate limit exceeded"),
            headers={"Retry-After": "30"},
        )

    client = build_client(handler, settings)
    with pytest.raises(RateLimitError) as excinfo:
        await client.complete("hi")
    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 30
    assert "30" in str(excinfo.value)


async def test_server_error_is_reported(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(error_payload("upstream exploded"), status_code=503)

    client = build_client(handler, settings)
    with pytest.raises(UpstreamStatusError) as excinfo:
        await client.complete("hi")
    assert excinfo.value.status_code == 503
    assert "upstream exploded" in str(excinfo.value)


async def test_error_detail_never_contains_the_api_key(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(error_payload(f"key {TEST_API_KEY} rejected"), status_code=400)

    client = build_client(handler, settings)
    with pytest.raises(UpstreamStatusError) as excinfo:
        await client.complete("hi")
    assert TEST_API_KEY not in str(excinfo.value)
    assert "redacted" in str(excinfo.value)


async def test_timeout_is_wrapped(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = build_client(handler, settings)
    with pytest.raises(UpstreamTimeoutError):
        await client.complete("hi")


async def test_connection_failure_is_wrapped(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = build_client(handler, settings)
    with pytest.raises(UpstreamConnectionError):
        await client.complete("hi")


async def test_malformed_json_is_reported(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, content=b"<html>not json</html>")

    client = build_client(handler, settings)
    with pytest.raises(MalformedResponseError):
        await client.complete("hi")


async def test_missing_choices_is_reported(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"id": "gen-1", "choices": []})

    client = build_client(handler, settings)
    with pytest.raises(MalformedResponseError):
        await client.complete("hi")


async def test_missing_content_is_reported(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"choices": [{"message": {"role": "assistant"}}]})

    client = build_client(handler, settings)
    with pytest.raises(MalformedResponseError):
        await client.complete("hi")


async def test_non_object_json_body_is_reported(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(["not", "an", "object"])

    client = build_client(handler, settings)
    with pytest.raises(MalformedResponseError):
        await client.complete("hi")


async def test_non_numeric_retry_after_is_tolerated(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            json=error_payload("Rate limit exceeded"),
            headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
        )

    client = build_client(handler, settings)
    with pytest.raises(RateLimitError) as excinfo:
        await client.complete("hi")
    assert excinfo.value.retry_after is None


async def test_retry_after_absent_is_tolerated(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=429, json=error_payload("Rate limit exceeded"))

    client = build_client(handler, settings)
    with pytest.raises(RateLimitError) as excinfo:
        await client.complete("hi")
    assert excinfo.value.retry_after is None


async def test_api_key_is_never_written_to_logs(settings: Settings, caplog) -> None:
    """Even when the upstream echoes the key back, logs stay credential-free."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(error_payload(f"rejected key {TEST_API_KEY}"), status_code=401)

    client = build_client(handler, settings)
    with caplog.at_level(logging.DEBUG), pytest.raises(UpstreamError):
        await client.complete("hi")

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert TEST_API_KEY not in joined


async def test_httpx_error_does_not_include_credentials(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset")

    client = build_client(handler, settings)
    with pytest.raises(UpstreamConnectionError) as excinfo:
        await client.complete("hi")

    assert TEST_API_KEY not in str(excinfo.value)
