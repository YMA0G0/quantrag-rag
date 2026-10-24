"""Offline fetch tests: mock transport, detection, one-hop follow, arXiv."""

import httpx
import pytest

from quantrag.ingest import fetch


@pytest.fixture(autouse=True)
def _no_robots(monkeypatch):
    monkeypatch.setattr(fetch, "_robots_allowed", lambda url: True)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_detect_format_prefers_magic_bytes():
    assert fetch.detect_format(b"%PDF-1.7\n...", "text/html") == "pdf"
    assert fetch.detect_format(b"<!doctype html>\n<html></html>", "application/pdf") == "html"
    assert fetch.detect_format(b"# Heading\n\nText", "text/plain") == "text"


def test_fetch_retry_and_failure_isolation(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(
            200, content=b"%PDF-1.4 test", headers={"content-type": "application/pdf"}
        )

    client = _client(handler)
    res = fetch.fetch(
        "http://example.com/test",
        cache_dir=tmp_path,
        rate_limiter=fetch.HostRateLimiter(0),
        client=client,
    )
    assert res.ok and res.detected_format == "pdf"
    client.close()


def test_html_served_for_pdf_url_is_detected_as_html(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<!doctype html><html><body>wrapper</body></html>",
            headers={"content-type": "application/pdf"},
        )

    client = _client(handler)
    res = fetch.fetch(
        "http://example.com/paper.pdf",
        cache_dir=tmp_path,
        rate_limiter=fetch.HostRateLimiter(0),
        client=client,
        follow_pdf_hop=False,
    )
    assert res.ok and res.detected_format == "html"
    client.close()


def test_one_hop_pdf_follow_keeps_original_url(tmp_path):
    page = b'<!doctype html><html><body><a href="/files/paper.pdf">Download file</a></body></html>'
    pdf = b"%PDF-1.4 fake pdf body"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/resources/lec1/":
            return httpx.Response(200, content=page, headers={"content-type": "text/html"})
        if request.url.path == "/files/paper.pdf":
            return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})
        return httpx.Response(404)

    original = "http://example.com/resources/lec1/"
    client = _client(handler)
    res = fetch.fetch(
        original, cache_dir=tmp_path, rate_limiter=fetch.HostRateLimiter(0), client=client
    )
    assert res.ok and res.detected_format == "pdf"
    assert res.original_url == original
    assert res.used_url.endswith("/files/paper.pdf")
    assert res.followed_from == original
    client.close()


def test_arxiv_abs_rewritten_to_pdf(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})

    client = _client(handler)
    res = fetch.fetch(
        "https://arxiv.org/abs/2502.07766v2",
        cache_dir=tmp_path,
        rate_limiter=fetch.HostRateLimiter(0),
        client=client,
    )
    assert res.ok
    assert seen["url"] == "https://arxiv.org/pdf/2502.07766v2"
    client.close()
