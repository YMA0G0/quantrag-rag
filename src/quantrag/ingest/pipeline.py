"""Orchestrate ingestion: CSV + manual files -> fetch -> extract -> chunk -> output."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .chunking import (
    MAX_TOKENS_HARD,
    TARGET_TOKENS,
    chunk_text,
    count_tokens,
    make_chunk_id,
)
from .extract import (
    ArxivSourceUnavailable,
    PandocNotFoundError,
    arxiv_id_from_url,
    extract_arxiv_source,
    extract_html,
    extract_pdf,
)
from .fetch import (
    FetchResult,
    HostRateLimiter,
    detect_format,
    fetch,
    find_all_pdf_links,
    find_resource_links,
)
from .sources import Source, load_manual_sources, resolve_category, select_sources

THIN_PAGE_CHARS = 1500


@dataclass
class _Doc:
    text: str
    pages: int
    fmt: str
    used_url: str
    followed_from: str
    extractor: str = ""
    quality_score: float = float("inf")
    category: str = ""
    source_format: str = ""


def _doc_from_bytes(
    source: Source, data: bytes, used_url: str, followed_from: str, category: str = ""
) -> _Doc:
    fmt = detect_format(data)
    if fmt == "pdf":
        extracted = extract_pdf(data)
        return _Doc(
            extracted.text,
            extracted.pages,
            "pdf",
            used_url,
            followed_from,
            extractor=extracted.extractor,
            quality_score=extracted.quality_score,
            category=category or source.category,
            source_format=extracted.source_format,
        )
    if fmt == "html":
        extracted = extract_html(data)
        return _Doc(
            extracted.text,
            extracted.pages,
            "html",
            used_url,
            followed_from,
            extractor=extracted.extractor,
            quality_score=extracted.quality_score,
            category=category or source.category,
            source_format=extracted.source_format,
        )
    # plain text
    text = data.decode("utf-8", errors="replace")
    return _Doc(
        text,
        1,
        "text",
        used_url,
        followed_from,
        category=category or source.category,
        source_format="text",
    )


def _try_arxiv_source(source: Source, cache_dir: Path) -> tuple[_Doc | None, str]:
    """Try the LaTeX e-print route; return (doc, error_reason)."""
    arxiv_id = arxiv_id_from_url(source.url)
    if not arxiv_id:
        return None, ""
    try:
        extracted = extract_arxiv_source(source.url, cache_dir=cache_dir)
    except PandocNotFoundError as exc:
        return None, str(exc)
    except ArxivSourceUnavailable as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - fall back to PDF/HTML
        return None, f"{type(exc).__name__}: {exc}"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    return (
        _Doc(
            extracted.text,
            0,
            "latex",
            pdf_url,
            "",
            extractor=extracted.extractor,
            quality_score=extracted.quality_score,
            category=source.category,
            source_format=extracted.source_format,
        ),
        "",
    )


def _get_docs(
    source: Source,
    cache_dir: Path,
    limiter: HostRateLimiter,
    category_resolver: Callable[[str], str | None] | None = None,
) -> tuple[list[_Doc] | None, str, dict[str, object] | None]:
    """Return (docs, error_reason, image_profile).

    For HTML pages with PDF links, follows up to 15 PDF hops (one hop each),
    returning one _Doc per followed PDF plus the HTML doc itself if no PDF
    links were found.

    ``category_resolver`` resolves the category for hop-found child URLs
    (from CSV or CATEGORY_MAPPING).  If None, children inherit the parent's
    category.
    """
    if source.manual_file:
        try:
            data = Path(source.manual_file).read_bytes()
        except OSError as exc:
            return None, f"manual_read_error: {type(exc).__name__}", None
        try:
            if category_resolver:
                cat = category_resolver(source.url, source.category)
            else:
                cat = source.category
            doc = _doc_from_bytes(source, data, source.url, "", category=cat)
        except Exception:  # noqa: BLE001
            return None, "extract_error", None
        profile: dict[str, object] | None = None
        if Path(source.manual_file).suffix.lower() == ".pdf":
            from .extract import pdf_image_profile

            try:
                profile = pdf_image_profile(data)
            except Exception:  # noqa: BLE001 - corrupt PDF shouldn't crash the pipeline
                profile = None
        return [doc], "", profile

    # Fetch without following PDF hops first; we handle hops ourselves.
    result: FetchResult = fetch(
        source.url,
        cache_dir=cache_dir,
        rate_limiter=limiter,
        follow_pdf_hop=False,
    )
    if not result.ok:
        return None, result.error, None

    # arXiv LaTeX route: one e-print request per paper, cached in data/raw.
    # If the e-print is a PDF (no LaTeX) or pandoc is unavailable, fall back
    # to the normal PDF/HTML extraction below.
    arxiv_doc, _arxiv_error = _try_arxiv_source(source, cache_dir)
    if arxiv_doc is not None:
        return [arxiv_doc], "", None

    fmt = result.detected_format
    docs: list[_Doc] = []

    if fmt == "pdf":
        try:
            if category_resolver:
                cat = category_resolver(result.used_url or result.url, source.category)
            else:
                cat = source.category
            doc = _doc_from_bytes(
                source,
                result.content,
                result.used_url or result.url,
                result.followed_from,
                category=cat,
            )
        except Exception:  # noqa: BLE001
            return None, "extract_error", None
        docs.append(doc)
        return docs, "", None

    if fmt == "html":
        # Find all PDF links on the HTML page (one-hop follow, cap 15)
        pdf_links = find_all_pdf_links(result.content, result.used_url or result.url, max_links=15)

        if pdf_links:
            # Follow each PDF link (up to 15)
            for pdf_url in pdf_links:
                nested = fetch(
                    pdf_url,
                    cache_dir=cache_dir,
                    rate_limiter=limiter,
                    follow_pdf_hop=False,
                )
                if nested.ok and nested.detected_format == "pdf":
                    try:
                        cat = (
                            category_resolver(nested.used_url or pdf_url, source.category)
                            if category_resolver
                            else source.category
                        )
                        doc = _doc_from_bytes(
                            source,
                            nested.content,
                            nested.used_url or nested.url,
                            source.url,  # followed_from = original HTML source
                            category=cat,
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    docs.append(doc)
            return docs, "", None
        else:
            # Second hop for landing pages: follow same-host "/resources/"
            # links, then look for PDFs on those resource pages.
            resource_links = find_resource_links(
                result.content, result.used_url or result.url, max_links=15
            )
            if resource_links:
                for res_url in resource_links:
                    nested = fetch(
                        res_url,
                        cache_dir=cache_dir,
                        rate_limiter=limiter,
                        follow_pdf_hop=False,
                    )
                    if not nested.ok or nested.detected_format != "html":
                        continue
                    res_pdf_links = find_all_pdf_links(
                        nested.content, nested.used_url or res_url, max_links=15
                    )
                    for pdf_url in res_pdf_links:
                        pdf_nested = fetch(
                            pdf_url,
                            cache_dir=cache_dir,
                            rate_limiter=limiter,
                            follow_pdf_hop=False,
                        )
                        if not pdf_nested.ok or pdf_nested.detected_format != "pdf":
                            continue
                        try:
                            cat = (
                                category_resolver(pdf_nested.used_url or pdf_url, source.category)
                                if category_resolver
                                else source.category
                            )
                            doc = _doc_from_bytes(
                                source,
                                pdf_nested.content,
                                pdf_nested.used_url or pdf_url,
                                source.url,  # followed_from = original landing page
                                category=cat,
                            )
                        except Exception:  # noqa: BLE001
                            continue
                        docs.append(doc)
                if docs:
                    return docs, "", None

            # No PDF links: process the HTML page itself
            try:
                cat = (
                    category_resolver(result.used_url or result.url, source.category)
                    if category_resolver
                    else source.category
                )
                doc = _doc_from_bytes(
                    source,
                    result.content,
                    result.used_url or result.url,
                    result.followed_from,
                    category=cat,
                )
            except Exception:  # noqa: BLE001
                return None, "extract_error", None
            docs.append(doc)
            return docs, "", None

    # plain text or binary
    try:
        cat = (
            category_resolver(result.used_url or result.url, source.category)
            if category_resolver
            else source.category
        )
        doc = _doc_from_bytes(
            source,
            result.content,
            result.used_url or result.url,
            result.followed_from,
            category=cat,
        )
    except Exception:  # noqa: BLE001
        return None, "extract_error", None
    docs.append(doc)
    return docs, "", None


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
    per_doc_cap: float | None = None,
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
        manual_selected, manual_report = load_manual_sources(
            manual_csv, manual_dir, profile=profile
        )
        selected.extend(manual_selected)
        report.extend(manual_report)

    # Key by (url, origin) so two sources that share a URL but come from
    # different inputs (e.g. a CSV row and a manual row) never collapse onto a
    # single report row.
    report_by_url = {(r["url"], r["origin"]): r for r in report}

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

    # Build a category resolver: hop-found children get their category from
    # CATEGORY_MAPPING or inherit the parent's category (if in allow-list).
    def _make_resolver() -> Callable[[str, str], str | None]:
        def resolver(url: str, fallback: str) -> str | None:
            return resolve_category(url, fallback)

        return resolver

    category_resolver = _make_resolver()

    all_chunks: list[dict] = []
    chunk_token_counts: list[int] = []
    seen_chunk_hashes: set[str] = set()
    chunk_drafts: list = []

    ingested = 0
    failed = 0
    skipped = sum(1 for r in report if r["status"] == "skipped")

    for source in selected:
        rrow = report_by_url.get((source.url, source.origin))
        docs, err, image_profile = _get_docs(source, cache_dir, limiter, category_resolver)

        if docs is None:
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

        # Aggregate metrics across all docs from this source
        total_chars = 0
        total_pages = 0
        total_chunk_count = 0
        kind_counts: dict[str, int] = {}
        source_ingested = False
        source_extractor: str | None = None
        source_quality_score: float | None = None
        source_format: str | None = None

        for doc_idx, doc in enumerate(docs):
            if rrow is not None:
                rrow["used_url"] = doc.used_url
                rrow["followed_from"] = doc.followed_from
                rrow["detected_format"] = doc.fmt
                if source_extractor is None:
                    source_extractor = doc.extractor
                    source_quality_score = doc.quality_score
                    source_format = doc.source_format

            # Image-locked slide deck check
            if image_profile and image_profile.get("image_locked"):
                skipped += 1
                if rrow is not None:
                    rrow["status"] = "skipped"
                    rrow["reason"] = "needs_ocr"
                    rrow["pages"] = str(image_profile.get("pages", ""))
                    rrow["characters"] = str(image_profile.get("total_chars", ""))
                break  # Skip entire source if any doc is image-locked

            total_chars += len(doc.text.strip())
            total_pages += doc.pages

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

            # Strip document title from extracted body if present at start.
            body = doc.text
            if body.startswith(source.title):
                body = body[len(source.title) :].lstrip("\n ")

            record_url = doc.used_url if doc.source_format == "arxiv_latex" else source.url

            drafts = chunk_text(
                body,
                title=source.title,
                url=record_url,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                overlap_ratio=overlap_ratio,
                min_chars=min_chars,
            )

            for idx, draft in enumerate(drafts):
                kind_counts[draft.kind] = kind_counts.get(draft.kind, 0) + 1
                if draft.kind not in {"prose", "formula_block", "exercises"}:
                    chunk_drafts.append(draft)
                    continue
                norm = f"{draft.kind}: " + " ".join(draft.text.split()).lower()
                chash = hashlib.sha1(norm.encode()).hexdigest()
                if chash in seen_chunk_hashes:
                    continue
                seen_chunk_hashes.add(chash)
                cid = make_chunk_id(record_url, doc_idx * 10000 + idx)
                # Store body-only text in JSONL; header added at index time
                record = {
                    "id": cid,
                    "text": draft.text,
                    "title": source.title,
                    "url": record_url,
                    "category": source.category,
                    "section": draft.section,
                    "kind": draft.kind,
                    "source_format": doc.source_format,
                }
                all_chunks.append(record)
                chunk_token_counts.append(count_tokens(draft.text))
                total_chunk_count += 1
                chunk_drafts.append(draft)

            source_ingested = True

        if rrow is not None and source_ingested:
            rrow["status"] = "ingested"
            rrow["chunks"] = str(total_chunk_count)
            rrow["characters"] = str(total_chars)
            rrow["pages"] = str(total_pages)
            rrow["extractor"] = source_extractor or ""
            rrow["source_format"] = source_format or ""
            if source_quality_score is not None:
                rrow["quality_score"] = str(round(source_quality_score, 2))
            else:
                rrow["quality_score"] = ""
            for kind_name in (
                "formula_dump",
                "toc",
                "references",
                "boilerplate",
                "formula_block",
                "exercises",
            ):
                rrow[f"{kind_name}_chunks"] = str(kind_counts.get(kind_name, 0))
        if source_ingested:
            ingested += 1

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Per-document cap: no single source may exceed ``per_doc_cap`` of the
    # corpus.  Opt-in: defaults to off (None).  Deterministic: keeps the
    # first ``cap`` chunks per over-cap source URL.  Iterates until no
    # source exceeds the cap.  Records how many chunks were dropped per
    # document and which sections were dropped.
    # ------------------------------------------------------------------
    dropped_per_doc: dict[str, int] = {}
    dropped_sections: dict[str, list[str]] = {}
    if per_doc_cap is not None and all_chunks:
        from collections import Counter as _Counter

        changed = True
        while changed:
            total = len(all_chunks)
            cap = max(1, int(total * per_doc_cap))
            url_counts = _Counter(r["url"] for r in all_chunks)
            over_cap_urls = {u for u, c in url_counts.items() if c > cap}
            if not over_cap_urls:
                changed = False
                continue
            capped_chunks: list[dict] = []
            url_seen: _Counter = _Counter()
            for r in all_chunks:
                u = r["url"]
                if u in over_cap_urls:
                    url_seen[u] += 1
                    if url_seen[u] > cap:
                        dropped_per_doc[u] = dropped_per_doc.get(u, 0) + 1
                        dropped_sections.setdefault(u, []).append(r.get("section", ""))
                        continue
                    capped_chunks.append(r)
                else:
                    capped_chunks.append(r)
            all_chunks = capped_chunks
            chunk_token_counts = [count_tokens(r["text"]) for r in all_chunks]
        # Update report chunk counts for capped sources
        final_url_counts = _Counter(r["url"] for r in all_chunks)
        for rrow in report:
            if rrow.get("url") in final_url_counts:
                rrow["chunks"] = str(final_url_counts[rrow["url"]])

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
        "toc_skipped": sum(1 for d in chunk_drafts if d.kind == "toc"),
        "references_skipped": sum(1 for d in chunk_drafts if d.kind == "references"),
        "boilerplate_skipped": sum(1 for d in chunk_drafts if d.kind == "boilerplate"),
        "formula_block_chunks": sum(1 for d in chunk_drafts if d.kind == "formula_block"),
        "exercises_chunks": sum(1 for d in chunk_drafts if d.kind == "exercises"),
        "total_chunks": len(all_chunks),
        "dropped_per_doc": dropped_per_doc,
        "dropped_sections": dropped_sections,
        "per_doc_cap": per_doc_cap,
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
        "source_format",
        "http_status",
        "pages",
        "characters",
        "chunks",
        "extractor",
        "quality_score",
        "formula_dump_chunks",
        "formula_block_chunks",
        "exercises_chunks",
        "toc_chunks",
        "references_chunks",
        "boilerplate_chunks",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
