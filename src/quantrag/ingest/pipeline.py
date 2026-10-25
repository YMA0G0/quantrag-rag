"""Orchestrate ingestion: CSV + manual files -> fetch -> extract -> chunk -> output."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .chunking import (
    MAX_TOKENS_HARD,
    TARGET_TOKENS,
    add_context_header,
    chunk_text,
    count_tokens,
    make_chunk_id,
)
from .extract import extract_html, extract_pdf
from .fetch import FetchResult, HostRateLimiter, detect_format, fetch
from .sources import Source, load_manual_sources, select_sources

THIN_PAGE_CHARS = 1500


@dataclass
class _Doc:
    text: str
    pages: int
    fmt: str
    used_url: str
    followed_from: str


def _doc_from_bytes(source: Source, data: bytes, used_url: str, followed_from: str) -> _Doc:
    fmt = detect_format(data)
    if fmt == "pdf":
        extracted = extract_pdf(data)
        return _Doc(extracted.text, extracted.pages, "pdf", used_url, followed_from)
    if fmt == "html":
        extracted = extract_html(data)
        return _Doc(extracted.text, extracted.pages, "html", used_url, followed_from)
    # plain text
    text = data.decode("utf-8", errors="replace")
    return _Doc(text, 1, "text", used_url, followed_from)


def _get_doc(source: Source, cache_dir: Path, limiter: HostRateLimiter) -> tuple[_Doc | None, str]:
    """Return (doc, error_reason)."""
    if source.manual_file:
        try:
            data = Path(source.manual_file).read_bytes()
        except OSError as exc:
            return None, f"manual_read_error: {type(exc).__name__}"
        try:
            return _doc_from_bytes(source, data, source.url, ""), ""
        except Exception as exc:  # noqa: BLE001
            return None, f"extract_error: {type(exc).__name__}"

    result: FetchResult = fetch(source.url, cache_dir=cache_dir, rate_limiter=limiter)
    if not result.ok:
        return None, result.error
    try:
        doc = _doc_from_bytes(
            source, result.content, result.used_url or result.url, result.followed_from
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"extract_error: {type(exc).__name__}"
    doc.pages = doc.pages
    return doc, ""


def run_pipeline(
    csv_paths: list[str],
    *,
    out_chunks: Path,
    out_report_json: Path,
    out_report_csv: Path,
    raw_dir: Path,
    cache_dir: Path | None = None,
    include_unreviewed: bool = False,
    max_tier: int | None = None,
    only_category: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    profile: str = "core",
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS_HARD,
    overlap_ratio: float = 0.12,
    min_chars: int = 200,
    manual_csv: str | None = None,
    manual_dir: str | None = None,
) -> dict:
    cache_dir = cache_dir or raw_dir
    selected, report = select_sources(
        csv_paths,
        include_unreviewed=include_unreviewed,
        max_tier=max_tier,
        only_category=only_category,
        limit=limit,
        profile=profile,
    )
    if manual_csv and manual_dir:
        manual_selected, manual_report = load_manual_sources(manual_csv, manual_dir)
        selected.extend(manual_selected)
        report.extend(manual_report)

    report_by_url = {r["url"]: r for r in report}

    if dry_run:
        _write_reports(report, {}, out_report_json, out_report_csv)
        return {
            "selected": len(selected),
            "ingested": 0,
            "skipped": sum(1 for r in report if r["status"] == "skipped"),
            "failed": 0,
            "chunks": 0,
        }

    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    limiter = HostRateLimiter()
    all_chunks: list[dict] = []
    chunk_token_counts: list[int] = []
    seen_chunk_hashes: set[str] = set()
    chunk_drafts: list = []

    ingested = 0
    failed = 0
    skipped = sum(1 for r in report if r["status"] == "skipped")

    for source in selected:
        rrow = report_by_url.get(source.url)
        doc, err = _get_doc(source, cache_dir, limiter)

        if doc is None:
            # Distinguish fetch failures from extraction failures.
            if err.endswith(("robots_disallowed",)) or err.startswith(
                ("http_", "connection_error", "too_large", "manual_read_error")
            ):
                failed += 1
                if rrow is not None:
                    rrow["status"] = "failed"
                    rrow["reason"] = err
            else:
                failed += 1
                if rrow is not None:
                    rrow["status"] = "failed"
                    rrow["reason"] = err
            continue

        if rrow is not None:
            rrow["used_url"] = doc.used_url
            rrow["followed_from"] = doc.followed_from
            rrow["detected_format"] = doc.fmt

        if len(doc.text.strip()) < 500:
            skipped += 1
            if rrow is not None:
                rrow["status"] = "skipped"
                rrow["reason"] = "empty_extraction"
                rrow["characters"] = str(len(doc.text))
                rrow["pages"] = str(doc.pages)
            continue

        # Non-PDF pages that are mostly navigation are not worth indexing.
        if doc.fmt != "pdf" and len(doc.text.strip()) < THIN_PAGE_CHARS:
            skipped += 1
            if rrow is not None:
                rrow["status"] = "skipped"
                rrow["reason"] = "thin_page"
                rrow["characters"] = str(len(doc.text))
                rrow["pages"] = str(doc.pages)
            continue

        drafts = chunk_text(
            doc.text,
            title=source.title,
            url=source.url,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_ratio=overlap_ratio,
            min_chars=min_chars,
        )

        chunk_count = 0
        formula_dump_count = 0
        # chunk_drafts already initialized at the outer scope
        for idx, draft in enumerate(drafts):
            if draft.kind == "formula_dump":
                formula_dump_count += 1
                if rrow is not None:
                    rrow["status"] = "skipped"
                    rrow["reason"] = "formula_dump"
                chunk_drafts.append(draft)
                continue
            norm = " ".join(draft.text.split()).lower()
            chash = hashlib.sha1(norm.encode()).hexdigest()
            if chash in seen_chunk_hashes:
                continue
            seen_chunk_hashes.add(chash)
            cid = make_chunk_id(source.url, idx)
            embedded_text = add_context_header(draft.text, source.title, draft.section)
            record = {
                "id": cid,
                "text": embedded_text,
                "title": source.title,
                "url": source.url,
                "category": source.category,
                "kind": draft.kind,
            }
            all_chunks.append(record)
            chunk_token_counts.append(count_tokens(embedded_text))
            chunk_count += 1
            chunk_drafts.append(draft)

        ingested += 1
        if rrow is not None:
            rrow["status"] = "ingested"
    out_chunks.parent.mkdir(parents=True, exist_ok=True)
    with out_chunks.open("w", encoding="utf-8") as handle:
        for record in all_chunks:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if chunk_token_counts:
        avg_tokens = statistics.mean(chunk_token_counts)
        median_tokens = statistics.median(chunk_token_counts)
        max_tokens_actual = max(chunk_token_counts)
    else:
        avg_tokens = median_tokens = max_tokens_actual = 0

    totals = {
        "sources_selected": len(selected),
        "sources_ingested": ingested,
        "sources_skipped": skipped,
        "sources_failed": failed,
        "formula_dump_skipped": sum(1 for d in chunk_drafts if d.kind == "formula_dump"),
        "total_chunks": len(all_chunks),
        "avg_tokens_per_chunk": round(avg_tokens, 2),
        "median_tokens_per_chunk": round(median_tokens, 2),
        "max_tokens_per_chunk": max_tokens_actual,
        "chunks_over_800_tokens": sum(1 for c in chunk_token_counts if c > 800),
        "chunks_over_1000_tokens": sum(1 for c in chunk_token_counts if c > 1000),
    }

    _write_reports(report, totals, out_report_json, out_report_csv)
    return totals


def _write_reports(report, totals_obj, json_path, csv_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"totals": totals_obj, "sources": report}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "url",
        "used_url",
        "followed_from",
        "title",
        "origin",
        "status",
        "reason",
        "detected_format",
        "http_status",
        "pages",
        "characters",
        "chunks",
        "formula_dump_chunks",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
