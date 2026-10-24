"""Tests for the MCP server layer (tool discovery, error mapping, wiring)."""

from __future__ import annotations

import json
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
from quantrag.server import TOOL_NAME, create_server

from .conftest import TEST_API_KEY, chat_completion, json_response, make_transport

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class StubClient:
    """Minimal ChatCompleter implementation used to drive the tool handler."""

    def __init__(self, answer: str = "stub answer", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        if self.error is not None:
            raise self.error
        return self.answer


def extract_text(result: Any) -> str:
    """Pull the plain text out of a FastMCP ``call_tool`` result."""
    content = result[0] if isinstance(result, tuple) else result
    return "".join(block.text for block in content if getattr(block, "type", None) == "text")


async def test_tool_is_discoverable() -> None:
    server = create_server(StubClient())
    tools = await server.list_tools()

    names = [tool.name for tool in tools]
    assert TOOL_NAME in names

    tool = next(tool for tool in tools if tool.name == TOOL_NAME)
    assert tool.description
    schema = tool.inputSchema
    assert schema["properties"]["question"]["type"] == "string"
    assert schema["required"] == ["question"]


async def test_tool_returns_model_answer() -> None:
    client = StubClient(answer="VaR at 95% is 1.2%.")
    server = create_server(client)

    result = await server.call_tool(TOOL_NAME, {"question": "What is VaR at 95%?"})

    text = extract_text(result)
    assert text == "VaR at 95% is 1.2%."
    assert client.prompts == ["What is VaR at 95%?"]
    assert client.systems and client.systems[0]


@pytest.mark.parametrize("bad_question", ["", "   ", "\t", "\n  \n", " \u00a0 "])
async def test_tool_rejects_blank_questions(bad_question: str) -> None:
    """Empty and whitespace-only questions fail validation before the backend."""
    client = StubClient()
    server = create_server(client)

    with pytest.raises(Exception) as excinfo:
        await server.call_tool(TOOL_NAME, {"question": bad_question})

    assert "non-empty" in str(excinfo.value)
    assert client.prompts == [], "the backend must not be called for blank input"


async def test_upstream_failure_is_mapped_to_tool_error() -> None:
    client = StubClient(error=UpstreamStatusError("OpenRouter request failed with HTTP 503.", 503))
    server = create_server(client)

    with pytest.raises(Exception) as excinfo:
        await server.call_tool(TOOL_NAME, {"question": "hello"})

    message = str(excinfo.value)
    assert "503" in message
    assert TOOL_NAME in message


async def test_missing_configuration_is_reported_cleanly(tmp_path, monkeypatch) -> None:
    """No credentials anywhere => a clean, actionable tool error.

    The dotenv read is stubbed out so the test stays hermetic even when a real
    ``.env`` exists in the project (load_dotenv resolves it from the source
    file, not the working directory).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("quantrag.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    server = create_server()
    with pytest.raises(Exception) as excinfo:
        await server.call_tool(TOOL_NAME, {"question": "hello"})

    message = str(excinfo.value)
    assert "OPENROUTER_API_KEY" in message
    assert "not configured" in message


async def test_tool_talks_to_openrouter_end_to_end(settings: Settings) -> None:
    """The real client is wired to a mocked transport: no network, no key."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return json_response(chat_completion("A Sharpe ratio of 1.4 is good."))

    client = OpenRouterClient(settings, transport=make_transport(handler))
    server = create_server(client)

    result = await server.call_tool(TOOL_NAME, {"question": "Is a Sharpe ratio of 1.4 good?"})

    text = extract_text(result)
    assert text == "A Sharpe ratio of 1.4 is good."
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["payload"]["model"] == settings.model


async def test_api_key_never_leaks_through_tool_errors(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": {"message": f"bad key {TEST_API_KEY}"}}, status_code=401)

    client = OpenRouterClient(settings, transport=make_transport(handler))
    server = create_server(client)

    with pytest.raises(Exception) as excinfo:
        await server.call_tool(TOOL_NAME, {"question": "hello"})

    assert TEST_API_KEY not in str(excinfo.value)


async def test_unexpected_error_is_wrapped(settings: Settings) -> None:
    class Boom:
        async def complete(self, prompt: str, *, system: str | None = None) -> str:
            raise RuntimeError("kaboom")

    server = create_server(Boom())
    with pytest.raises(Exception) as excinfo:
        await server.call_tool(TOOL_NAME, {"question": "hello"})

    assert "RuntimeError" in str(excinfo.value)


def test_upstream_error_is_a_quantrag_error() -> None:
    assert issubclass(UpstreamError, Exception)


async def test_question_is_stripped_before_reaching_backend() -> None:
    """Surrounding whitespace is removed before the prompt is sent."""
    client = StubClient()
    server = create_server(client)

    await server.call_tool(TOOL_NAME, {"question": "  What is a bond future?  "})

    assert client.prompts == ["What is a bond future?"]


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("rate limited, retry later", 429, retry_after=30),
        AuthenticationError("rejected the API key", 401),
        UpstreamStatusError("OpenRouter request failed with HTTP 503.", 503),
        UpstreamTimeoutError("timed out after 60s"),
        UpstreamConnectionError("could not reach OpenRouter"),
        MalformedResponseError("response is not valid JSON"),
    ],
    ids=["rate-limit", "auth", "status", "timeout", "connection", "malformed"],
)
async def test_every_upstream_error_maps_to_the_same_tool_error(error: UpstreamError) -> None:
    """All upstream failures surface as tool errors with a consistent prefix."""
    server = create_server(StubClient(error=error))

    with pytest.raises(Exception) as excinfo:
        await server.call_tool(TOOL_NAME, {"question": "hello"})

    message = str(excinfo.value)
    assert "QuantRAG could not reach the model" in message
    assert str(error) in message


async def test_empty_backend_answer_is_rejected() -> None:
    """A backend that answers with nothing is an error, not a silent success.

    The real client strips whitespace, so an all-blank upstream payload arrives
    here as an empty string.
    """
    server = create_server(StubClient(answer=""))

    with pytest.raises(Exception) as excinfo:
        await server.call_tool(TOOL_NAME, {"question": "hello"})

    assert "empty answer" in str(excinfo.value)
