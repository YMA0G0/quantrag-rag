"""HTTP fetching with caching, retries, rate limiting and robots.txt respect.

Format is decided from the response ``Content-Type`` **and** magic bytes, never
from the CSV hint: many publishers serve an HTML wrapper page for a PDF.  Such
pages are followed one hop to the linked PDF while the caller keeps the
original URL for citations.

A failing source never crashes the run: callers receive a ``FetchResult``
with ``ok=False`` and an error code.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "QuantRAG-Ingest/0.1 (+https://github.com/quantrag)"
MAX_BYTES = 25 * 1024 * 1024
TIMEOUT = 30.0
MAX_RETRIES = 3
MIN_HOST_INTERVAL = 1.0

_ARXIV_ABS = re.compile(r"^(?P<base>https?://(?:www\.)?arxiv\.org)/abs/(?P<id>[^?#]+?)/?$")
_PDF_HREF = re.compile(r"""href\s*=\s*["']([^"']+\.pdf[^"']*)["']""", re.IGNORECASE)
_DOWNLOAD_HREF = re.compile(
    r"""<a[^>]+href\s*=\s*["']([^"']+)["'][^>]*>[^<]*download[^<]*</a>""",
    re.IGNORECASE,
)


@dataclass
class FetchResult:
    url: str
    original_url: str = ""
    ok: bool = False
    status_code: int | None = None
    content: bytes = b""
    content_type: str = ""
    final_url: str = ""
    error: str = ""
    from_cache: bool = False
    followed_from: str = ""
    used_url: str = ""
    detected_format: str = ""


class HostRateLimiter:
    def __init__(self, min_interval: float = MIN_HOST_INTERVAL) -> None:
        self._min = min_interval
        self._last: dict[str, float] = {}

    def wait(self, host: str) -> None:
        now = time.monotonic()
        last = self._last.get(host)
        if last is not None:
            delta = now - last
            if delta < self._min:
                time.sleep(self._min - delta)
        self._last[host] = time.monotonic()


def arxiv_pdf_url(url: str) -> str:
    """Rewrite an arXiv abstract URL to its PDF URL (leave everything else)."""
    match = _ARXIV_ABS.match(url.strip())
    if not match:
        return url
    return f"{match.group('base')}/pdf/{match.group('id')}"


def detect_format(content: bytes, content_type: str = "") -> str:
    """Classify payload as pdf/html/text/binary using magic bytes first."""
    head = content[:1024].lstrip()
    if content[:5] == b"%PDF-":
        return "pdf"
    lowered_ct = (content_type or "").lower()
    head_lower = head[:512].lower()
    if head_lower.startswith(b"<!doctype html") or head_lower.startswith(b"<html"):
        return "html"
    if "application/pdf" in lowered_ct:
        return "pdf"
    if "text/html" in lowered_ct or "application/xhtml" in lowered_ct:
        return "html"
    if "text/plain" in lowered_ct or "application/json" in lowered_ct:
        return "text"
    if head_lower.startswith(b"<"):
        return "html"
    return "binary"


def find_pdf_link(content: bytes, base_url: str) -> str | None:
    """Return the first plausible PDF link on a wrapper page, if any."""
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        return None
    for match in _DOWNLOAD_HREF.finditer(text):
        candidate = urljoin(base_url, match.group(1))
        if candidate.lower().split("?")[0].endswith(".pdf"):
            return candidate
    for match in _PDF_HREF.finditer(text):
        href = match.group(1)
        # Skip obvious related-file links that are not the main download.
        if href.lower().split("?")[0].endswith(".pdf"):
            return urljoin(base_url, href)
    return None


def cache_path(cache_dir: Path, url: str, ext: str = ".bin") -> Path:
    return cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()}{ext}"


def _find_cached(cache_dir: Path, url: str) -> Path | None:
    digest = hashlib.sha1(url.encode()).hexdigest()
    for candidate in cache_dir.glob(digest + ".*"):
        return candidate
    return None


_EXT = {"pdf": ".pdf", "html": ".html", "text": ".txt", "binary": ".bin"}


def _ext_for(fmt: str) -> str:
    return _EXT.get(fmt, ".bin")


_robots_cache: dict[str, RobotFileParser | None] = {}


def _robots_allowed(url: str) -> bool:
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}"
    if base not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        _robots_cache[base] = rp
    rp = _robots_cache[base]
    if rp is None:
        return True
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def _get_once(
    url: str, client: httpx.Client, limiter: HostRateLimiter
) -> tuple[httpx.Response | None, str]:
    """GET with retries/backoff. Returns (response, error_code)."""
    host = urlsplit(url).hostname or ""
    limiter.wait(host)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url, headers={"User-Agent": USER_AGENT})
        except httpx.HTTPError as exc:
            if attempt == MAX_RETRIES - 1:
                return None, f"connection_error: {type(exc).__name__}"
            time.sleep(2**attempt)
            continue
        if resp.status_code >= 500 or resp.status_code == 429:
            if attempt == MAX_RETRIES - 1:
                return resp, f"http_{resp.status_code}"
            delay = 2**attempt
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                with contextlib.suppress(ValueError):
                    delay = max(delay, float(retry_after))
            time.sleep(delay)
            continue
        if resp.status_code >= 400:
            return resp, f"http_{resp.status_code}"
        return resp, ""
    return None, "unknown"


def fetch(
    url: str,
    *,
    cache_dir: Path,
    rate_limiter: HostRateLimiter | None = None,
    client: httpx.Client | None = None,
    follow_pdf_hop: bool = True,
) -> FetchResult:
    """Fetch ``url`` (or read from cache). Never raises.

    ``url`` is rewritten for arXiv abstract links, then the response format is
    detected.  HTML wrapper pages are followed one hop to a linked PDF when
    ``follow_pdf_hop`` is set; the original URL is preserved in
    ``FetchResult.original_url`` for citation.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = arxiv_pdf_url(url)
    limiter = rate_limiter or HostRateLimiter()

    cached = _find_cached(cache_dir, wanted)
    if cached is not None:
        try:
            data = cached.read_bytes()
            fmt = detect_format(data)
            if fmt == "html" and follow_pdf_hop:
                pdf_link = find_pdf_link(data, wanted)
                if pdf_link:
                    nested = fetch(
                        pdf_link,
                        cache_dir=cache_dir,
                        rate_limiter=limiter,
                        client=client,
                        follow_pdf_hop=False,
                    )
                    if nested.ok and nested.detected_format == "pdf":
                        nested.original_url = url
                        nested.followed_from = wanted
                        return nested
            return FetchResult(
                url=wanted,
                original_url=url,
                ok=True,
                content=data,
                from_cache=True,
                final_url=wanted,
                used_url=wanted,
                detected_format=fmt,
            )
        except OSError as exc:
            return FetchResult(url=wanted, original_url=url, error=f"cache_read_error: {exc}")

    if not _robots_allowed(wanted):
        return FetchResult(url=wanted, original_url=url, error="robots_disallowed")

    own_client = client is None
    if own_client:
        client = httpx.Client(
            timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
    try:
        resp, err = _get_once(wanted, client, limiter)
        if resp is None or err:
            return FetchResult(
                url=wanted,
                original_url=url,
                ok=False,
                status_code=resp.status_code if resp is not None else None,
                error=err,
            )
        content = resp.content
        if len(content) > MAX_BYTES:
            return FetchResult(
                url=wanted,
                original_url=url,
                status_code=resp.status_code,
                error="too_large",
                content_type=resp.headers.get("content-type", ""),
            )
        content_type = resp.headers.get("content-type", "")
        fmt = detect_format(content, content_type)
        used_url = wanted
        followed_from = ""

        if fmt == "html" and follow_pdf_hop:
            pdf_link = find_pdf_link(content, str(resp.url))
            if pdf_link and pdf_link != wanted:
                # one hop only
                nested = fetch(
                    pdf_link,
                    cache_dir=cache_dir,
                    rate_limiter=limiter,
                    client=client,
                    follow_pdf_hop=False,
                )
                if nested.ok and nested.detected_format == "pdf":
                    nested.original_url = url
                    nested.followed_from = wanted
                    return nested
                # fall through and keep the HTML page if the hop failed
                followed_from = pdf_link
            elif pdf_link:
                followed_from = pdf_link

        cpath = cache_path(cache_dir, used_url, _ext_for(fmt))
        with contextlib.suppress(OSError):
            cpath.write_bytes(content)
        return FetchResult(
            url=used_url,
            original_url=url,
            ok=True,
            status_code=resp.status_code,
            content=content,
            content_type=content_type,
            final_url=str(resp.url),
            followed_from=followed_from,
            used_url=used_url,
            detected_format=fmt,
        )
    finally:
        if own_client and client is not None:
            client.close()
