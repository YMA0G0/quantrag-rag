"""Tests for the error helpers.

``redact`` is the last line of defence against credential leakage, so it is
tested directly rather than only through the HTTP client.
"""

from __future__ import annotations

from quantrag.errors import (
    REDACTION,
    AuthenticationError,
    ConfigurationError,
    MalformedResponseError,
    QuantRagError,
    RateLimitError,
    UpstreamConnectionError,
    UpstreamError,
    UpstreamStatusError,
    UpstreamTimeoutError,
    redact,
)


def test_redact_removes_every_occurrence() -> None:
    secret = "sk-or-v1-supersecret"
    payload = f"bad key {secret} and again {secret} end"
    cleaned = redact(payload, secret)
    assert secret not in cleaned
    assert cleaned.count(REDACTION) == 2


def test_redact_ignores_empty_and_none_secrets() -> None:
    assert redact("unchanged", "", None) == "unchanged"


def test_redact_handles_no_secret_match() -> None:
    assert redact("harmless", "not-present") == "harmless"


def test_error_hierarchy() -> None:
    for exc_type in (
        AuthenticationError,
        RateLimitError,
        MalformedResponseError,
        UpstreamTimeoutError,
        UpstreamConnectionError,
        UpstreamStatusError,
    ):
        assert issubclass(exc_type, UpstreamError)
        assert issubclass(exc_type, QuantRagError)

    assert issubclass(UpstreamError, QuantRagError)
    assert issubclass(ConfigurationError, QuantRagError)


def test_status_code_is_preserved() -> None:
    err = UpstreamStatusError("boom", 503)
    assert err.status_code == 503
    assert str(err) == "boom"

    auth = AuthenticationError("nope", 401)
    assert auth.status_code == 401

    rate = RateLimitError("slow down", 429, retry_after=12.5)
    assert rate.status_code == 429
    assert rate.retry_after == 12.5
    assert RateLimitError("slow down").retry_after is None
