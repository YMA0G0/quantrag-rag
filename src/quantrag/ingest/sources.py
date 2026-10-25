"""Source selection from the Data Engineer's CSV inputs.

Reads one or more source CSVs, normalises columns, de-duplicates by normalised
URL and applies the inclusion/exclusion rules from the project spec.  Every
skipped row is recorded with an explicit reason code so the ingest report can
explain what happened to each source.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

TEXT_DOC_TYPES = frozenset(
    {
        "Research Paper",
        "PDF / Lecture Notes",
        "Documentation",
        "Book",
        "Course",
        "Website",
        "Official Report",
    }
)
ALLOWED_CONTENT_TYPES = frozenset({"pdf", "html_docs"})
SKIP_CHECK_STATUSES_ALWAYS = frozenset({"NOT_FOUND"})
SKIP_CHECK_STATUSES_DEFAULT = frozenset({"REVIEW_BLOCKED", "NEW_UNCHECKED"})

CORE_CATEGORIES = frozenset(
    {
        "Black–Scholes and Option Pricing",
        "Black-Scholes and Option Pricing",
        "Option Greeks and Implied Volatility",
        "Stochastic Calculus and Itô's Lemma",
        "Stochastic Calculus and Ito's Lemma",
        "Stochastic Processes",
        "Stochastic Differential Equations (SDEs)",
        "Heston and SABR Models",
        "Binomial and Monte Carlo Pricing",
        "Financial Derivatives and Exotic Options",
        "Computational Finance and Numerical Methods",
        "Fixed Income and Interest Rate Models",
        "University Courses and Lecture Notes",
    }
)
CORE_MAX_TIER = 2

_PRESS_MARKERS = (
    "news-insights",
    "media-release",
    "media-releases",
    "press-release",
    "press-releases",
    "/news/",
    "/press/",
)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def normalize_url(url: str) -> str:
    """Normalise a URL for de-duplication: strip fragment, trailing slash, lowercase host."""
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw.lower()
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    if parts.port:
        host = f"{host}:{parts.port}"
    path = parts.path.rstrip("/")
    # keep query (some sources differ by query), drop fragment
    return urlunsplit((scheme, host, path, parts.query, ""))


def _valid_url(url: str) -> bool:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False
    return bool(parts.scheme and parts.hostname)


def _row_get(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            val = str(row[name]).strip()
            if val:
                return val
    return ""


def _normalise_row(row: dict[str, str], origin: str) -> dict[str, str]:
    """Map the two CSV schemas onto a common set of fields."""
    title = _row_get(row, "title")
    url = _row_get(row, "url")
    final_url = _row_get(row, "final_url")
    category = _row_get(row, "category", "topic")
    resource_type = _row_get(row, "resource_type")
    # Map variations
    if resource_type == "Lecture Notes":
        resource_type = "PDF / Lecture Notes"
    check_status = _row_get(row, "check_status", "link_status")
    is_extractable = _row_get(row, "is_extractable")
    content_ready = _row_get(row, "content_ready")
    access_type = _row_get(row, "access_type")
    content_type_preferred = _row_get(row, "content_type_preferred")
    needs_manual_review = _row_get(row, "needs_manual_review")
    recommended_action = _row_get(row, "recommended_action")
    priority_tier_raw = _row_get(row, "priority_tier")
    source_quality = _row_get(row, "source_quality")
    chosen_url = final_url if (final_url and _valid_url(final_url)) else url
    # Default priority_tier for new_sources = 3
    default_tier = "3" if origin == "quant_finance_50_new_sources.csv" else "1"
    return {
        "origin": origin,
        "title": title,
        "url": url,
        "final_url": final_url,
        "chosen_url": chosen_url,
        "category": category,
        "resource_type": resource_type,
        "check_status": check_status,
        "is_extractable": is_extractable,
        "content_ready": content_ready,
        "access_type": access_type,
        "content_type_preferred": content_type_preferred,
        "needs_manual_review": needs_manual_review,
        "recommended_action": recommended_action,
        "priority_tier": priority_tier_raw or default_tier,
        "source_quality": source_quality,
    }


@dataclass(frozen=True, slots=True)
class Source:
    title: str
    url: str  # chosen (final) url
    original_url: str
    category: str
    resource_type: str
    check_status: str
    access_type: str
    content_type_preferred: str
    priority_tier: int
    source_quality: str
    origin: str
    manual_file: str = ""


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(r) for r in reader]


def _marketing_reason(row: dict[str, str]) -> str:
    """Return a skip reason for landing/marketing pages, else ''."""
    url = (row["chosen_url"] or row["url"]).lower()
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    host = parts.hostname or ""
    title = row["title"].lower()
    if not path and not parts.query:
        return "landing_or_marketing"
    if host.startswith("pypi.org"):
        return "landing_or_marketing"
    if host.startswith("huggingface.co"):
        segments = [s for s in path.split("/") if s]
        if segments and (len(segments) <= 2 or "model" in title):
            return "landing_or_marketing"
    if any(marker in url for marker in _PRESS_MARKERS):
        return "landing_or_marketing"
    if (
        row["resource_type"] == "Course"
        and not url.split("?")[0].endswith(".pdf")
        and row["content_type_preferred"] != "pdf"
    ):
        return "landing_or_marketing"
    return ""


def _decide(
    row: dict[str, str], include_unreviewed: bool, profile: str = "core"
) -> tuple[bool, str]:
    """Return (included, reason_code). reason_code is 'included' when included."""
    if not _truthy(row["is_extractable"]):
        return False, "not_extractable"
    if row["access_type"] == "blocked":
        return False, "access_blocked"
    if row["resource_type"] not in TEXT_DOC_TYPES:
        return False, "unsupported_resource_type"
    if row["content_type_preferred"] not in ALLOWED_CONTENT_TYPES:
        return False, "unsupported_content_type"
    if row["check_status"] in SKIP_CHECK_STATUSES_ALWAYS:
        return False, "not_found" if row["check_status"] == "NOT_FOUND" else "check_status_skip"
    if row["check_status"] in SKIP_CHECK_STATUSES_DEFAULT and not include_unreviewed:
        # Do not skip VERIFIED_LIVE as unreviewed
        if row["check_status"] == "VERIFIED_LIVE":
            pass
        else:
            return False, "needs_review"
    if _truthy(row["needs_manual_review"]) and not include_unreviewed:
        return False, "needs_manual_review"
    marketing = _marketing_reason(row)
    if marketing:
        # tier<=2 allow-listed landing pages are fetched so their same-host
        # PDF links can be followed (cap 15, robots respected, rate limited).
        if profile == "core":
            try:
                tier = int(row["priority_tier"])
            except (ValueError, TypeError):
                tier = 1
            if tier <= CORE_MAX_TIER and row["category"] in CORE_CATEGORIES:
                pass  # allow tier<=2 allow-listed landing pages through
            else:
                return False, marketing
        else:
            return False, marketing
    if profile == "core" and row["category"] not in CORE_CATEGORIES:
        # Strict category allow-list: tier only ORDERS sources, it never
        # admits a source whose category is off-list.
        return False, "outside_core_profile"
    return True, "included"


def select_sources(
    csv_paths: Iterable[str | Path],
    *,
    include_unreviewed: bool = False,
    max_tier: int | None = None,
    only_category: str | None = None,
    limit: int | None = None,
    profile: str = "core",
) -> tuple[list[Source], list[dict[str, str]]]:
    """Select sources.

    Returns (selected_sources, report_rows) where report_rows contains one entry
    per input row (after de-dup) with keys: url, title, status ('selected'/'skipped'),
    reason, origin.  Selected rows have status 'selected' and reason 'included'.
    """
    # Load + normalise
    normalised: list[dict[str, str]] = []
    for path in csv_paths:
        origin = Path(path).name
        for raw in load_csv(path):
            normalised.append(_normalise_row(raw, origin))

    # De-dup by normalised chosen URL, prefer first occurrence
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in normalised:
        key = normalize_url(row["chosen_url"]) or normalize_url(row["url"])
        if not key:
            deduped.append(row)
            continue
        if key in seen:
            # record as duplicate skipped? We'll skip silently-ish but report
            row = dict(row)
            row["_duplicate"] = "true"
            deduped.append(row)
            continue
        seen.add(key)
        deduped.append(row)

    # Remove duplicates from selection but keep report
    report: list[dict[str, str]] = []
    selected: list[Source] = []
    for row in deduped:
        if row.get("_duplicate") == "true":
            report.append(
                {
                    "url": row["chosen_url"] or row["url"],
                    "title": row["title"],
                    "origin": row["origin"],
                    "status": "skipped",
                    "reason": "duplicate_url",
                    "http_status": "",
                    "pages": "",
                    "characters": "",
                    "chunks": "",
                }
            )
            continue
        included, reason = _decide(row, include_unreviewed, profile)
        if included:
            try:
                tier = int(row["priority_tier"])
            except (ValueError, TypeError):
                tier = 1
            if max_tier is not None and tier > max_tier:
                included = False
                reason = "tier_filtered"
            if included and only_category and row["category"].lower() != only_category.lower():
                included = False
                reason = "category_filtered"
        if not included:
            report.append(
                {
                    "url": row["chosen_url"] or row["url"],
                    "title": row["title"],
                    "origin": row["origin"],
                    "status": "skipped",
                    "reason": reason,
                    "http_status": "",
                    "pages": "",
                    "characters": "",
                    "chunks": "",
                }
            )
            continue
        try:
            tier = int(row["priority_tier"])
        except (ValueError, TypeError):
            tier = 1
        src = Source(
            title=row["title"],
            url=row["chosen_url"] or row["url"],
            original_url=row["url"],
            category=row["category"],
            resource_type=row["resource_type"],
            check_status=row["check_status"],
            access_type=row["access_type"],
            content_type_preferred=row["content_type_preferred"],
            priority_tier=tier,
            source_quality=row["source_quality"],
            origin=row["origin"],
        )
        selected.append(src)
        report.append(
            {
                "url": src.url,
                "title": src.title,
                "origin": src.origin,
                "status": "selected",
                "reason": "included",
                "http_status": "",
                "pages": "",
                "characters": "",
                "chunks": "",
            }
        )

    if limit is not None and limit >= 0:
        # keep first `limit` selected; mark the rest skipped
        kept_urls = {s.url for s in selected[:limit]}
        new_selected = selected[:limit]
        for row in report:
            if row["status"] == "selected" and row["url"] not in kept_urls:
                row["status"] = "skipped"
                row["reason"] = "limit"
        selected = new_selected

    return selected, report


def sha1_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def load_manual_sources(
    csv_path: str | Path, manual_dir: str | Path, profile: str = "core"
) -> tuple[list[Source], list[dict[str, str]]]:
    """Load manually downloaded files listed in ``manual_sources.csv``.

    Columns: ``title,url,category,file,priority_tier``.  ``file`` is resolved
    relative to ``manual_dir``.  Manual sources bypass the network (and thus
    robots.txt) but go through the same extract/chunk path and are cited by
    their original ``url``.  When ``profile == "core"``, the strict category
    allow-list applies: tier never admits an off-list category.
    """
    path = Path(csv_path)
    if not path.exists():
        return [], []
    selected: list[Source] = []
    report: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for raw in load_csv(path):
        title = _row_get(raw, "title")
        url = _row_get(raw, "url")
        category = _row_get(raw, "category")
        filename = _row_get(raw, "file")
        try:
            tier = int(_row_get(raw, "priority_tier") or "1")
        except ValueError:
            tier = 1
        # Strict category allow-list in core profile: tier never admits
        # an off-list category, not even for manual sources.
        if profile == "core" and category not in CORE_CATEGORIES:
            report.append(
                {
                    "url": url,
                    "title": title,
                    "origin": path.name,
                    "status": "skipped",
                    "reason": "outside_core_profile",
                    "http_status": "",
                    "pages": "",
                    "characters": "",
                    "chunks": "",
                    "used_url": "",
                }
            )
            continue
        file_path = Path(manual_dir) / filename
        if not filename or not file_path.is_file():
            report.append(
                {
                    "url": url,
                    "title": title,
                    "origin": path.name,
                    "status": "skipped",
                    "reason": "manual_file_missing",
                    "http_status": "",
                    "pages": "",
                    "characters": "",
                    "chunks": "",
                    "used_url": "",
                }
            )
            continue
        # Fail fast on duplicate manual rows sharing a normalised URL: keep the
        # first occurrence and mark later ones ``duplicate_url`` so the caller
        # never ingests the same source twice.
        norm_url = normalize_url(url) or url.lower()
        if norm_url in seen_urls:
            report.append(
                {
                    "url": url,
                    "title": title,
                    "origin": path.name,
                    "status": "skipped",
                    "reason": "duplicate_url",
                    "http_status": "",
                    "pages": "",
                    "characters": "",
                    "chunks": "",
                    "used_url": "",
                }
            )
            continue
        seen_urls.add(norm_url)
        suffix = file_path.suffix.lower()
        ctype = "pdf" if suffix == ".pdf" else "html_docs"
        src = Source(
            title=title,
            url=url,
            original_url=url,
            category=category,
            resource_type="Manual",
            check_status="MANUAL",
            access_type="open",
            content_type_preferred=ctype,
            priority_tier=tier,
            source_quality="",
            origin=path.name,
            manual_file=str(file_path),
        )
        selected.append(src)
        report.append(
            {
                "url": url,
                "title": title,
                "origin": path.name,
                "status": "selected",
                "reason": "included",
                "http_status": "",
                "pages": "",
                "characters": "",
                "chunks": "",
                "used_url": "",
            }
        )
    return selected, report
