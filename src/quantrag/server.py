"""QuantRAG MCP server.

Exposes a single tool, ``ask_quant_question``, that forwards a question to the
LLM configured through OpenRouter and returns the model's answer.

The server speaks MCP over stdio; stdout is reserved for the protocol, so all
diagnostics go to stderr.
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .config import Settings, load_settings
from .errors import ConfigurationError, UpstreamError
from .openrouter import ChatCompleter, OpenRouterClient

logger = logging.getLogger("quantrag")

SERVER_NAME = "quantrag"
TOOL_NAME = "ask_quant_question"

TOOL_DESCRIPTION = (
    "Answer a quantitative-finance question (markets, derivatives, portfolio "
    "construction, risk metrics, time series, backtesting) using the "
    "configured OpenRouter model."
)

SYSTEM_PROMPT = (
    "You are QuantRAG, a quantitative-finance assistant. Answer the user's "
    "question accurately and concisely. State assumptions explicitly and say "
    "when something is unknown or opinion rather than fact."
)


def _build_client() -> ChatCompleter:
    """Create the OpenRouter client from the current environment."""
    settings: Settings = load_settings()
    logger.info("Using OpenRouter model %s (base url %s)", settings.model, settings.base_url)
    return OpenRouterClient(settings)


class _ClientProvider:
    """Lazily creates the chat client and caches configuration errors."""

    def __init__(self, client: ChatCompleter | None = None) -> None:
        self._client = client
        self._error: str | None = None

    def get(self) -> ChatCompleter:
        if self._client is not None:
            return self._client
        if self._error is not None:
            raise ConfigurationError(self._error)
        try:
            self._client = _build_client()
        except ConfigurationError as exc:
            self._error = str(exc)
            raise
        return self._client


def create_server(client: ChatCompleter | None = None) -> FastMCP:
    """Build the FastMCP server, optionally with an injected client (tests)."""
    server = FastMCP(
        SERVER_NAME,
        instructions=(
            "Quantitative-finance question answering backed by an OpenRouter model. "
            f"Use the {TOOL_NAME} tool to answer questions."
        ),
    )
    provider = _ClientProvider(client)

    @server.tool(name=TOOL_NAME, description=TOOL_DESCRIPTION)
    async def ask_quant_question(question: str) -> str:
        """Answer a quantitative-finance question with the configured model."""
        prompt = question.strip() if isinstance(question, str) else ""
        if not prompt:
            raise ToolError("The 'question' argument must be a non-empty string.")

        try:
            chat = provider.get()
            answer = await chat.complete(prompt, system=SYSTEM_PROMPT)
        except ToolError:
            raise
        except ConfigurationError as exc:
            raise ToolError(f"QuantRAG is not configured: {exc}") from exc
        except UpstreamError as exc:
            raise ToolError(f"QuantRAG could not reach the model: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected failure while answering a question")
            raise ToolError(
                f"QuantRAG hit an unexpected error ({type(exc).__name__}). "
                "See the server logs for details."
            ) from exc

        if not answer:
            raise ToolError("The model returned an empty answer.")
        return answer

    return server


def configure_logging(level: int = logging.INFO) -> None:
    """Send logs to stderr so stdout stays free for the MCP protocol."""
    root = logging.getLogger()
    if not any(getattr(h, "_quantrag", False) for h in root.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler._quantrag = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the MCP server over stdio."""
    configure_logging()
    try:
        load_settings()
    except ConfigurationError as exc:
        # Not fatal: the server still starts so the tool can be discovered,
        # and calling it returns a clean configuration error.
        logger.warning("QuantRAG is not fully configured: %s", exc)
    server = create_server()
    try:
        server.run(transport="stdio")
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.info("Shutting down.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
