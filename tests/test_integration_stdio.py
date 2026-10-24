"""MCP protocol tests over a real STDIO connection.

The server runs as a real subprocess and is driven by the official Python MCP
client API (``mcp.client.stdio``). The LLM backend is a local HTTP mock that
speaks the OpenRouter Chat Completions protocol, so these tests need no API
key and no network access.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


MOCK_MODEL = "test/model"
MOCK_ANSWER = "Mocked quant answer."
FAKE_API_KEY = "test-key-not-a-real-secret"
TOOL_NAME = "ask_quant_question"


class _MockOpenRouterHandler(BaseHTTPRequestHandler):
    """Minimal OpenRouter stand-in: records requests, returns one canned reply."""

    received: list[dict[str, Any]] = []
    status_code: int = 200
    content: str | None = MOCK_ANSWER

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).received.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "payload": json.loads(body) if body else {},
            }
        )
        if type(self).content is None:
            payload: Any = {"error": {"message": "upstream exploded"}}
        else:
            payload = {
                "id": "gen-mock",
                "model": MOCK_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": type(self).content},
                    }
                ],
            }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(type(self).status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: Any) -> None:  # silence the default stderr logging
        return


def _start_mock_server(
    status_code: int = 200, content: str | None = MOCK_ANSWER
) -> tuple[ThreadingHTTPServer, str]:
    handler = type(
        "Handler",
        (_MockOpenRouterHandler,),
        {"received": [], "status_code": status_code, "content": content},
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    return httpd, f"http://{host}:{port}/v1"


def _stdio_env(base_url: str) -> dict[str, str]:
    return {
        **os.environ,
        "OPENROUTER_API_KEY": FAKE_API_KEY,
        "OPENROUTER_MODEL": MOCK_MODEL,
        "OPENROUTER_BASE_URL": base_url,
        "PYTHONUNBUFFERED": "1",
    }


def _stdio_params(tmp_path, base_url: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "quantrag"],
        env=_stdio_env(base_url),
        cwd=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Protocol tests driven by the official MCP client
# ---------------------------------------------------------------------------


async def test_initialize_succeeds(tmp_path) -> None:
    httpd, base_url = _start_mock_server()
    try:
        params = _stdio_params(tmp_path, base_url)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            init = await session.initialize()

            assert init.serverInfo.name == "quantrag"
            assert init.protocolVersion
    finally:
        httpd.shutdown()
        httpd.server_close()


async def test_tools_list_exposes_ask_quant_question(tmp_path) -> None:
    httpd, base_url = _start_mock_server()
    try:
        params = _stdio_params(tmp_path, base_url)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = [tool.name for tool in tools.tools]
            assert names == [TOOL_NAME]

            schema = tools.tools[0].inputSchema
            assert schema["properties"]["question"]["type"] == "string"
            assert schema["required"] == ["question"]
    finally:
        httpd.shutdown()
        httpd.server_close()


async def test_tools_call_returns_answer_and_reaches_openrouter(tmp_path) -> None:
    httpd, base_url = _start_mock_server()
    try:
        params = _stdio_params(tmp_path, base_url)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(TOOL_NAME, {"question": "What is VaR?"})

            assert result.isError is False
            assert getattr(result.content[0], "text", "") == MOCK_ANSWER

        received = httpd.RequestHandlerClass.received
        assert len(received) == 1
        assert received[0]["path"] == "/v1/chat/completions"
        assert received[0]["authorization"] == f"Bearer {FAKE_API_KEY}"
        assert received[0]["payload"]["model"] == MOCK_MODEL
        assert received[0]["payload"]["messages"][-1] == {"role": "user", "content": "What is VaR?"}
    finally:
        httpd.shutdown()
        httpd.server_close()


async def test_upstream_failure_is_reported_as_tool_error(tmp_path) -> None:
    httpd, base_url = _start_mock_server(status_code=503, content=None)
    try:
        params = _stdio_params(tmp_path, base_url)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(TOOL_NAME, {"question": "What is VaR?"})

            assert result.isError is True
            text = getattr(result.content[0], "text", "")
            assert "503" in text
            assert FAKE_API_KEY not in text
    finally:
        httpd.shutdown()
        httpd.server_close()


async def test_blank_question_is_reported_as_tool_error(tmp_path) -> None:
    httpd, base_url = _start_mock_server()
    try:
        params = _stdio_params(tmp_path, base_url)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(TOOL_NAME, {"question": "   "})

            assert result.isError is True
            assert "non-empty" in getattr(result.content[0], "text", "")
            assert httpd.RequestHandlerClass.received == [], "no request should reach the backend"
    finally:
        httpd.shutdown()
        httpd.server_close()


async def test_missing_argument_does_not_kill_the_server(tmp_path) -> None:
    httpd, base_url = _start_mock_server()
    try:
        params = _stdio_params(tmp_path, base_url)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()

            try:
                result = await session.call_tool(TOOL_NAME, {})
                assert result.isError is True
            except Exception:
                pass  # a protocol-level error is equally acceptable

            # the server must still be alive and serving
            tools = await session.list_tools()
            assert [tool.name for tool in tools.tools] == [TOOL_NAME]
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Stdout purity: the raw byte stream must contain JSON-RPC only
# ---------------------------------------------------------------------------


async def _read_jsonrpc_responses(
    proc: asyncio.subprocess.Process, wanted_id: int, collected: list[str]
) -> dict[str, Any]:
    """Read stdout until the response with ``wanted_id`` arrives."""
    deadline = asyncio.get_running_loop().time() + 20
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"no response for id {wanted_id}")
        line = await asyncio.wait_for(proc.stdout.readline(), remaining)  # type: ignore[union-attr]
        if not line:
            raise EOFError("server closed stdout before responding")
        text = line.decode().strip()
        if not text:
            continue
        collected.append(text)
        message = json.loads(text)
        if message.get("id") == wanted_id:
            return message


async def test_stdout_carries_only_jsonrpc_messages(tmp_path) -> None:
    """A raw look at stdout: every line must be a valid JSON-RPC message."""
    httpd, base_url = _start_mock_server()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "quantrag",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_stdio_env(base_url),
        cwd=str(tmp_path),
    )
    collected: list[str] = []

    def _write(payload: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(payload) + "\n").encode())

    try:
        _write(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "stdout-purity", "version": "0"},
                },
            }
        )
        assert proc.stdin is not None
        await proc.stdin.drain()
        init_response = await _read_jsonrpc_responses(proc, 1, collected)
        assert init_response["result"]["serverInfo"]["name"] == "quantrag"

        _write({"jsonrpc": "2.0", "method": "notifications/initialized"})
        await proc.stdin.drain()

        _write({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        await proc.stdin.drain()
        tools_response = await _read_jsonrpc_responses(proc, 2, collected)
        names = [tool["name"] for tool in tools_response["result"]["tools"]]
        assert names == [TOOL_NAME]

        proc.stdin.close()
        remaining_stdout, stderr_bytes = await asyncio.wait_for(proc.communicate(), 20)
        collected.extend(
            line.decode().strip() for line in remaining_stdout.splitlines() if line.strip()
        )
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        httpd.shutdown()
        httpd.server_close()

    assert collected, "no stdout was captured"
    for line in collected:
        message = json.loads(line)  # must not raise: no stray debug output
        assert message["jsonrpc"] == "2.0"
        assert "id" in message or "method" in message

    # secrets must never reach stderr either
    assert FAKE_API_KEY not in stderr_bytes.decode()
