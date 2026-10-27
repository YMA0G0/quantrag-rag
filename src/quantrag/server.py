"""QuantRAG MCP server.

Exposes two tools, ``ask_quant_question`` and ``search_quant_kb``, that share
a single :class:`quantrag.service.RagService` instance so retrieval and
answering are consistent across transports.

The server speaks MCP over stdio by default; ``--transport streamable-http``
starts an HTTP server on ``QUANTRAG_HTTP_HOST`` / ``QUANTRAG_HTTP_PORT``.
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .config import load_rag_settings
from .errors import ConfigurationError, RetrievalError
from .service import RagService, build_service

logger = logging.getLogger("quantrag")

SERVER_NAME = "quantrag"
ASK_TOOL = "ask_quant_question"
SEARCH_TOOL = "search_quant_kb"

ASK_DESCRIPTION = (
    "Answer a quantitative-finance question (markets, derivatives, portfolio "
    "construction, risk metrics, time series, backtesting) using the "
    "retrieval-augmented knowledge base."
)

SEARCH_DESCRIPTION = (
    "Search the quantitative-finance knowledge base for relevant passages. "
    "Returns ranked snippets with metadata for citations."
)


async def _get_service() -> RagService:
    """Lazy singleton: build the service on first use."""
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE
    _SERVICE = await build_service()
    return _SERVICE


_SERVICE: RagService | None = None


def create_server(*, service: RagService | None = None) -> FastMCP:
    """Build the FastMCP server, optionally with an injected service (tests)."""
    global _SERVICE
    if service is not None:
        _SERVICE = service

    async def get_svc() -> RagService:
        return await _get_service()

    server = FastMCP(
        SERVER_NAME,
        instructions=(
            "Quantitative-finance retrieval-augmented answering. "
            f"Use {ASK_TOOL} for grounded answers and {SEARCH_TOOL} for raw search."
        ),
    )

    @server.tool(name=ASK_TOOL, description=ASK_DESCRIPTION)
    async def ask_quant_question(
        question: str,
        *,
        k: int | None = None,
        mode: str | None = None,
    ) -> dict[str, object]:
        """Answer a quantitative-finance question with grounded citations."""
        text = question.strip() if isinstance(question, str) else ""
        if not text:
            raise ToolError("The 'question' argument must be a non-empty string.")

        try:
            svc = await get_svc()
            result = await svc.ask(text, k=k, mode=mode)
        except ConfigurationError as exc:
            raise ToolError(f"QuantRAG is not configured: {exc}") from exc
        except RetrievalError as exc:
            raise ToolError(f"QuantRAG retrieval failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected failure while answering a question")
            raise ToolError(
                f"QuantRAG hit an unexpected error ({type(exc).__name__}). "
                "See the server logs for details."
            ) from exc

        return result.to_dict()

    @server.tool(name=SEARCH_TOOL, description=SEARCH_DESCRIPTION)
    async def search_quant_kb(
        query: str,
        *,
        k: int | None = None,
        mode: str | None = None,
        min_score: float | None = None,
    ) -> dict[str, object]:
        """Search the knowledge base and return ranked snippets."""
        text = query.strip() if isinstance(query, str) else ""
        if not text:
            raise ToolError("The 'query' argument must be a non-empty string.")

        try:
            svc = await get_svc()
            hits = await svc.search(text, k=k, mode=mode, min_score=min_score)
        except ConfigurationError as exc:
            raise ToolError(f"QuantRAG is not configured: {exc}") from exc
        except RetrievalError as exc:
            raise ToolError(f"QuantRAG retrieval failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected failure while searching the knowledge base")
            raise ToolError(
                f"QuantRAG hit an unexpected error ({type(exc).__name__}). "
                "See the server logs for details."
            ) from exc

        return {
            "results": [hit.to_dict() for hit in hits],
            "count": len(hits),
            "mode": mode or (await get_svc()).settings.mode,
        }

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
    """Entry point: run the MCP server over stdio or streamable-http."""
    configure_logging()

    transport = "stdio"
    if argv and "--transport" in argv:
        try:
            idx = argv.index("--transport")
            transport = argv[idx + 1]
        except (IndexError, ValueError):
            pass

    try:
        load_rag_settings(load_env_file=True)
    except ConfigurationError as exc:
        logger.warning("QuantRAG is not fully configured: %s", exc)

    server = create_server()
    try:
        server.run(transport=transport)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.info("Shutting down.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
