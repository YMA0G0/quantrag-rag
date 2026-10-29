"""Exception hierarchy for QuantRAG.

All errors are raised as plain exceptions so that they can be mapped onto
MCP tool errors without leaking implementation details.  Messages must never
contain API keys or other credentials -- use :func:`redact` when passing
upstream payloads through to the user.
"""

from __future__ import annotations

REDACTION = "***redacted***"


class QuantRagError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(QuantRagError):
    """Required configuration is missing or invalid."""


class RetrievalError(QuantRagError):
    """The retriever received invalid input or an unusable index."""


class ZeroVectorError(RetrievalError):
    """Raised when a query vector has zero magnitude (no vocabulary overlap).

    This is a *recoverable* error: hybrid retrieval can fall back to BM25,
    and vector-only callers may choose to return an empty result set.
    """


class UpstreamError(QuantRagError):
    """Base class for failures while talking to an upstream LLM provider."""


class UpstreamTimeoutError(UpstreamError):
    """The upstream request exceeded the configured timeout."""


class UpstreamConnectionError(UpstreamError):
    """The upstream service could not be reached."""


class UpstreamStatusError(UpstreamError):
    """The upstream service answered with an error status code."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(UpstreamStatusError):
    """The upstream rejected the credentials."""


class RateLimitError(UpstreamStatusError):
    """The upstream rate limit was hit."""

    def __init__(
        self, message: str, status_code: int = 429, retry_after: float | None = None
    ) -> None:
        super().__init__(message, status_code)
        self.retry_after = retry_after


class MalformedResponseError(UpstreamError):
    """The upstream response could not be understood."""


def redact(text: str, *secrets: str | None) -> str:
    """Replace every occurrence of ``secrets`` in ``text`` with a placeholder.

    Used defensively: even though the code never puts the API key into an
    error message, an upstream error payload could echo it back.
    """
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTION)
    return text
