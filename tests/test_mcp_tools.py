"""Offline tests for the MCP tools (ask_quant_question, search_quant_kb)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantrag.config import RagSettings, load_rag_settings
from quantrag.errors import ConfigurationError
from quantrag.retriever import Chunk, load_chunks_jsonl
from quantrag.server import create_server
from quantrag.service import RagService, build_embedder

pytestmark = pytest.mark.anyio

FIXTURE = Path(__file__).parent / "fixtures" / "mini_chunks.jsonl"


def anyio_backend() -> str:
    return "asyncio"


class FakeChat:
    def __init__(self, reply: str = "The answer is [1].") -> None:
        self.reply = reply

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        return self.reply


@pytest.fixture
def settings() -> RagSettings:
    return load_rag_settings({}, load_env_file=False)


@pytest.fixture
def chunks() -> list[Chunk]:
    return load_chunks_jsonl(FIXTURE)


async def make_service(
    chunks: list[Chunk],
    settings: RagSettings,
    *,
    reply: str = "The answer is [1].",
) -> RagService:
    chat = FakeChat(reply)
    embedder = build_embedder("local", chunks=chunks)
    return await RagService.from_chunks(
        chunks,
        chat=chat,
        settings=settings,
        embedder=embedder,
        corpus_label="fixture",
    )


async def test_ask_quant_question_returns_cited_answer(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    service = await make_service(chunks, settings, reply="Brownian motion has drift [1].")
    server = create_server(service=service)

    # Find the ask_quant_question tool
    tool = next(t for t in server._tool_manager._tools.values() if t.name == "ask_quant_question")
    result = await tool.fn(question="What is Brownian motion with drift?")

    assert result["grounded"] is True
    assert result["answer"] == "Brownian motion has drift [1]."
    assert result["sources"]
    assert result["sources"][0]["cited"] is True
    assert result["retrieved"]
    assert result["retrieved"][0]["chunk_id"] == "3460fe14fd-0000"


async def test_ask_quant_question_not_found(chunks: list[Chunk], settings: RagSettings) -> None:
    from quantrag.rag import NOT_FOUND

    service = await make_service(chunks, settings, reply=NOT_FOUND)
    server = create_server(service=service)

    tool = next(t for t in server._tool_manager._tools.values() if t.name == "ask_quant_question")
    result = await tool.fn(question="What is the meaning of life?")

    assert result["grounded"] is False
    assert result["answer"] == NOT_FOUND
    assert result["sources"] == []


async def test_ask_quant_question_rejects_empty_question(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    service = await make_service(chunks, settings)
    server = create_server(service=service)

    tool = next(t for t in server._tool_manager._tools.values() if t.name == "ask_quant_question")
    with pytest.raises(Exception, match="non-empty string"):
        await tool.fn(question="   ")


async def test_search_quant_kb_returns_ranked_results(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    service = await make_service(chunks, settings)
    server = create_server(service=service)

    tool = next(t for t in server._tool_manager._tools.values() if t.name == "search_quant_kb")
    result = await tool.fn(query="Brownian motion with drift", k=3, mode="bm25")

    assert result["count"] == 3
    assert result["mode"] == "bm25"
    assert len(result["results"]) == 3
    assert all("score" in hit for hit in result["results"])
    assert all("chunk_id" in hit for hit in result["results"])
    assert all("snippet" in hit for hit in result["results"])
    assert result["results"][0]["score"] >= result["results"][1]["score"]


async def test_search_quant_kb_respects_mode(chunks: list[Chunk], settings: RagSettings) -> None:
    service = await make_service(chunks, settings)
    server = create_server(service=service)

    tool = next(t for t in server._tool_manager._tools.values() if t.name == "search_quant_kb")

    bm25 = await tool.fn(query="Brownian motion", k=2, mode="bm25")
    assert bm25["mode"] == "bm25"
    assert all(r["source"] == "bm25" for r in bm25["results"])

    vector = await tool.fn(query="Brownian motion", k=2, mode="vector")
    assert vector["mode"] == "vector"
    assert all(r["source"] == "vector" for r in vector["results"])

    hybrid = await tool.fn(query="Brownian motion", k=2, mode="hybrid")
    assert hybrid["mode"] == "hybrid"
    assert all(r["source"] == "hybrid" for r in hybrid["results"])


async def test_search_quant_kb_min_score_filter(chunks: list[Chunk], settings: RagSettings) -> None:
    service = await make_service(chunks, settings)
    server = create_server(service=service)

    tool = next(t for t in server._tool_manager._tools.values() if t.name == "search_quant_kb")

    unfiltered = await tool.fn(query="Brownian motion", k=5, mode="bm25")
    filtered = await tool.fn(query="Brownian motion", k=5, mode="bm25", min_score=1e9)

    assert len(filtered["results"]) == 0
    assert filtered["count"] == 0
    assert len(unfiltered["results"]) > 0


async def test_search_quant_kb_rejects_empty_query(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    service = await make_service(chunks, settings)
    server = create_server(service=service)

    tool = next(t for t in server._tool_manager._tools.values() if t.name == "search_quant_kb")
    with pytest.raises(Exception, match="non-empty string"):
        await tool.fn(query="   ")


async def test_both_tools_share_the_same_service(
    chunks: list[Chunk], settings: RagSettings
) -> None:
    """The two tools must share one RagService instance."""
    service = await make_service(chunks, settings)
    server = create_server(service=service)

    ask_tool = next(
        t for t in server._tool_manager._tools.values() if t.name == "ask_quant_question"
    )
    search_tool = next(
        t for t in server._tool_manager._tools.values() if t.name == "search_quant_kb"
    )

    # Call both tools with a query that matches the fixture
    await ask_tool.fn(question="Brownian motion with drift")
    await search_tool.fn(query="Brownian motion with drift")

    import quantrag.server as server_module

    assert server_module._SERVICE is service


async def test_server_without_configured_settings_fails_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server with no service and no env config returns clean config errors."""
    # Reset the global service and force build_service to fail
    import quantrag.server as server_module

    server_module._SERVICE = None

    async def failing_build(*a, **kw):
        raise ConfigurationError("No index, no corpus")

    monkeypatch.setattr(server_module, "build_service", failing_build)

    server = create_server(service=None)
    ask_tool = next(
        t for t in server._tool_manager._tools.values() if t.name == "ask_quant_question"
    )
    search_tool = next(
        t for t in server._tool_manager._tools.values() if t.name == "search_quant_kb"
    )

    with pytest.raises(Exception, match="QuantRAG is not configured"):
        await ask_tool.fn(question="test")

    with pytest.raises(Exception, match="QuantRAG is not configured"):
        await search_tool.fn(query="test")


async def test_streamable_http_transport_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The --transport streamable-http flag is accepted (smoke test)."""
    # FastMCP.run with streamable-http tries to bind a port; we mock it to just
    # verify the argument parsing path.
    import quantrag.server as server_module
    from quantrag.server import main

    class MockServer:
        def run(self, transport: str) -> None:
            assert transport == "streamable-http"

    monkeypatch.setattr(server_module, "create_server", lambda: MockServer())
    monkeypatch.setattr(server_module, "configure_logging", lambda: None)
    monkeypatch.setattr(server_module, "load_rag_settings", lambda **_: None)

    assert main(["--transport", "streamable-http"]) == 0
