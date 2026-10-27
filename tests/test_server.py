"""Tests for the MCP server layer (tool discovery, error mapping, wiring).

These tests use a fake RagService backed by the fixture corpus so the server
logic is exercised without network calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from quantrag.config import load_rag_settings
from quantrag.errors import ConfigurationError
from quantrag.retriever import load_chunks_jsonl
from quantrag.server import ASK_TOOL, SEARCH_TOOL, create_server
from quantrag.service import RagService, build_embedder

pytestmark = pytest.mark.anyio

FIXTURE = "tests/fixtures/mini_chunks.jsonl"


def anyio_backend() -> str:
    return "asyncio"


class FakeChat:
    """Minimal ChatCompleter implementation used to drive the service."""

    def __init__(self, answer: str = "stub answer", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        if self.error is not None:
            raise self.error
        return self.answer


class FailingService:
    """Mock service that raises ConfigurationError on ask."""

    async def ask(self, *a, **kw):
        raise ConfigurationError("No index, no corpus")


async def make_service(
    *,
    reply: str = "The answer is [1].",
    error: Exception | None = None,
) -> RagService:
    settings = load_rag_settings({}, load_env_file=False)
    chunks = load_chunks_jsonl(FIXTURE)
    chat = FakeChat(reply, error)
    embedder = build_embedder("local", chunks=chunks)
    return await RagService.from_chunks(
        chunks,
        chat=chat,
        settings=settings,
        embedder=embedder,
        corpus_label="fixture",
    )


def extract_text(result: Any) -> str:
    """Pull the plain text out of a FastMCP ``call_tool`` result."""
    content = result[0] if isinstance(result, tuple) else result
    return "".join(block.text for block in content if getattr(block, "type", None) == "text")


@pytest.fixture
def reset_server_service() -> None:
    """Reset the global service before each test that needs it."""
    import quantrag.server as server_module

    server_module._SERVICE = None


async def test_tools_are_discoverable() -> None:
    service = await make_service()
    server = create_server(service=service)
    tools = await server.list_tools()

    names = [tool.name for tool in tools]
    assert ASK_TOOL in names
    assert SEARCH_TOOL in names

    ask_tool = next(tool for tool in tools if tool.name == ASK_TOOL)
    assert ask_tool.description
    schema = ask_tool.inputSchema
    assert schema["properties"]["question"]["type"] == "string"
    assert schema["required"] == ["question"]

    search_tool = next(tool for tool in tools if tool.name == SEARCH_TOOL)
    assert search_tool.description
    schema = search_tool.inputSchema
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["required"] == ["query"]


async def test_ask_quant_question_returns_grounded_answer() -> None:
    import json

    service = await make_service(reply="Brownian motion has drift [1].")
    server = create_server(service=service)

    result = await server.call_tool(ASK_TOOL, {"question": "What is Brownian motion with drift?"})

    text = extract_text(result)
    data = json.loads(text)  # tool returns JSON string
    assert data["grounded"] is True
    assert data["answer"] == "Brownian motion has drift [1]."
    assert data["sources"]
    assert data["sources"][0]["cited"] is True


@pytest.mark.parametrize("bad_question", ["", "   ", "\t", "\n  \n", " \u00a0 "])
async def test_ask_quant_question_rejects_blank_questions(bad_question: str) -> None:
    """Empty and whitespace-only questions fail validation before the service."""
    service = await make_service()
    server = create_server(service=service)

    with pytest.raises(Exception) as excinfo:
        await server.call_tool(ASK_TOOL, {"question": bad_question})

    assert "non-empty" in str(excinfo.value)


async def test_ask_quant_question_not_found() -> None:
    import json

    from quantrag.rag import NOT_FOUND

    service = await make_service(reply=NOT_FOUND)
    server = create_server(service=service)

    result = await server.call_tool(ASK_TOOL, {"question": "What is the meaning of life?"})

    text = extract_text(result)
    data = json.loads(text)
    assert data["grounded"] is False
    assert data["answer"] == NOT_FOUND
    assert data["sources"] == []


async def test_search_quant_kb_returns_ranked_results() -> None:
    import json

    service = await make_service()
    server = create_server(service=service)

    result = await server.call_tool(
        SEARCH_TOOL, {"query": "Brownian motion with drift", "k": 3, "mode": "bm25"}
    )

    text = extract_text(result)
    data = json.loads(text)
    assert data["count"] == 3
    assert data["mode"] == "bm25"
    assert len(data["results"]) == 3
    assert all("score" in hit for hit in data["results"])
    assert all("chunk_id" in hit for hit in data["results"])


async def test_search_quant_kb_rejects_blank_query() -> None:
    service = await make_service()
    server = create_server(service=service)

    with pytest.raises(Exception) as excinfo:
        await server.call_tool(SEARCH_TOOL, {"query": "   "})

    assert "non-empty" in str(excinfo.value)


async def test_missing_configuration_is_reported_cleanly(reset_server_service) -> None:
    """A service that raises ConfigurationError on ask yields a clean error."""
    server = create_server(service=FailingService())
    with pytest.raises(Exception) as excinfo:
        await server.call_tool(ASK_TOOL, {"question": "hello"})

    message = str(excinfo.value)
    assert "not configured" in message


async def test_streamable_http_transport_option(monkeypatch) -> None:
    """The --transport streamable-http flag is accepted."""
    import quantrag.server as server_module

    class MockServer:
        def run(self, transport: str) -> None:
            assert transport == "streamable-http"

    monkeypatch.setattr(server_module, "create_server", lambda: MockServer())
    monkeypatch.setattr(server_module, "configure_logging", lambda: None)
    monkeypatch.setattr(server_module, "load_rag_settings", lambda **_: None)

    assert server_module.main(["--transport", "streamable-http"]) == 0
