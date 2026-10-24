"""Offline tests for source selection rules."""

from __future__ import annotations

import csv
from pathlib import Path

from quantrag.ingest import sources


def _write_csv(tmp_path: Path, name: str, rows: list[dict[str, str]]) -> Path:
    p = tmp_path / name
    if not rows:
        p.write_text("")
        return p
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def test_normalize_url():
    assert sources.normalize_url("http://EXAMPLE.COM/Page/") == "http://example.com/Page"
    assert sources.normalize_url("http://example.com/page#frag") == "http://example.com/page"


def test_url_dedup(tmp_path: Path):
    rows = [
        {
            "title": "A",
            "url": "http://example.com/doc/",
            "category": "Derivatives",
            "resource_type": "Research Paper",
            "check_status": "LIVE",
            "http_status": "200",
            "final_url": "http://example.com/doc",
            "content_type_preferred": "html_docs",
            "is_extractable": "True",
            "content_ready": "yes",
            "access_type": "open",
            "recommended_action": "keep",
            "priority_tier": "1",
            "source_quality": "High",
            "needs_manual_review": "False",
        },
        {
            "title": "A2",
            "url": "http://example.com/doc",
            "category": "Derivatives",
            "resource_type": "Research Paper",
            "check_status": "LIVE",
            "http_status": "200",
            "final_url": "http://example.com/doc",
            "content_type_preferred": "html_docs",
            "is_extractable": "True",
            "content_ready": "yes",
            "access_type": "open",
            "recommended_action": "keep",
            "priority_tier": "1",
            "source_quality": "High",
            "needs_manual_review": "False",
        },
    ]
    p = _write_csv(tmp_path, "quant_finance_kb_cleaned.csv", rows)
    selected, report = sources.select_sources([p])
    assert len(selected) == 1
    reasons = [r["reason"] for r in report]
    assert "duplicate_url" in reasons


def test_selection_reasons(tmp_path: Path):
    base = {
        "title": "T",
        "url": "http://example.com/x",
        "category": "Cat",
        "resource_type": "Research Paper",
        "check_status": "LIVE",
        "http_status": "200",
        "final_url": "",
        "content_type_preferred": "html_docs",
        "is_extractable": "True",
        "content_ready": "yes",
        "access_type": "open",
        "recommended_action": "keep",
        "priority_tier": "1",
        "source_quality": "High",
        "needs_manual_review": "False",
    }
    rows = []
    # not extractable
    r = dict(base)
    r["url"] = "http://example.com/a"
    r["is_extractable"] = "False"
    rows.append(r)
    # blocked
    r = dict(base)
    r["url"] = "http://example.com/b"
    r["access_type"] = "blocked"
    rows.append(r)
    # github
    r = dict(base)
    r["url"] = "http://example.com/c"
    r["resource_type"] = "GitHub Repository"
    rows.append(r)
    # dataset
    r = dict(base)
    r["url"] = "http://example.com/d"
    r["resource_type"] = "Dataset"
    rows.append(r)
    # api
    r = dict(base)
    r["url"] = "http://example.com/e"
    r["resource_type"] = "API"
    rows.append(r)
    # video? not in set; use Website but content_type other
    r = dict(base)
    r["url"] = "http://example.com/f"
    r["content_type_preferred"] = "other"
    rows.append(r)
    # not found
    r = dict(base)
    r["url"] = "http://example.com/g"
    r["check_status"] = "NOT_FOUND"
    rows.append(r)
    # review blocked default
    r = dict(base)
    r["url"] = "http://example.com/h"
    r["check_status"] = "REVIEW_BLOCKED"
    rows.append(r)
    # unchecked default
    r = dict(base)
    r["url"] = "http://example.com/i"
    r["check_status"] = "NEW_UNCHECKED"
    rows.append(r)
    # needs manual review default
    r = dict(base)
    r["url"] = "http://example.com/j"
    r["needs_manual_review"] = "True"
    rows.append(r)
    p = _write_csv(tmp_path, "quant_finance_kb_cleaned.csv", rows)
    selected, report = sources.select_sources([p])
    assert len(selected) == 0
    reasons = {r["reason"] for r in report}
    for need in [
        "not_extractable",
        "access_blocked",
        "unsupported_resource_type",
        "unsupported_content_type",
        "not_found",
        "needs_review",
        "needs_manual_review",
    ]:
        assert need in reasons


def _row(**kwargs):
    base = {
        "title": "T",
        "url": "http://example.com/x",
        "category": "Cat",
        "resource_type": "Research Paper",
        "check_status": "LIVE",
        "http_status": "200",
        "final_url": "",
        "content_type_preferred": "html_docs",
        "is_extractable": "True",
        "content_ready": "yes",
        "access_type": "open",
        "recommended_action": "keep",
        "priority_tier": "1",
        "source_quality": "High",
        "needs_manual_review": "False",
    }
    base.update(kwargs)
    return base


def test_core_profile_filters_tier_3_non_core(tmp_path):
    rows = [
        _row(url="http://example.com/a", priority_tier="3", category="Credit Risk"),
        _row(url="http://example.com/b", priority_tier="3", category="Heston and SABR Models"),
        _row(url="http://example.com/c", priority_tier="2", category="Credit Risk"),
    ]
    p = _write_csv(tmp_path, "quant_finance_kb_cleaned.csv", [dict(r) for r in rows])
    selected, report = sources.select_sources([p], profile="core")
    urls = {s.url for s in selected}
    assert "http://example.com/a" not in urls
    assert "http://example.com/b" in urls
    assert "http://example.com/c" in urls
    assert any(r["reason"] == "outside_core_profile" for r in report)

    selected_all, _ = sources.select_sources([p], profile="all")
    assert len(selected_all) == 3


def test_landing_and_marketing_skipped(tmp_path):
    rows = [
        _row(url="https://example.com/"),
        _row(url="https://pypi.org/project/arch/"),
        _row(url="https://huggingface.co/ProsusAI/finbert"),
        _row(url="https://example.com/news/2020/launch"),
        _row(
            url="https://example.com/course",
            resource_type="Course",
            content_type_preferred="html_docs",
        ),
    ]
    p = _write_csv(tmp_path, "quant_finance_kb_cleaned.csv", [dict(r) for r in rows])
    selected, report = sources.select_sources([p], profile="all")
    assert selected == []
    assert all(r["reason"] == "landing_or_marketing" for r in report)
