"""Offline pipeline tests: manual sources, thin pages, report totals."""

import csv
import json
from pathlib import Path

from quantrag.ingest import pipeline
from quantrag.retriever import load_chunks_jsonl


def _write_manual(tmp_path: Path, rows: list[dict[str, str]], files: dict[str, bytes]):
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    for name, data in files.items():
        (manual_dir / name).write_bytes(data)
    csv_path = tmp_path / "manual_sources.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["title", "url", "category", "file", "priority_tier"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, manual_dir


def _run(tmp_path: Path, manual_csv: Path, manual_dir: Path) -> dict:
    return pipeline.run_pipeline(
        [],
        out_chunks=tmp_path / "chunks.jsonl",
        out_report_json=tmp_path / "report.json",
        out_report_csv=tmp_path / "report.csv",
        raw_dir=tmp_path / "raw",
        manual_csv=str(manual_csv),
        manual_dir=str(manual_dir),
    )


def _build_pdf(text: str) -> bytes:
    """Build a minimal, valid single-page PDF whose content stream shows ``text``.

    Hand-rolled (no reportlab) so the test stays dependency-light; pypdf can
    round-trip it and extract the embedded literal string.
    """
    esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    one_line = esc.replace("\n", " ")
    body = f"BT /F1 12 Tf 72 712 Td ({one_line}) Tj ET".encode("latin-1")

    segs: list[bytes] = []

    def add(seg: bytes) -> int:
        off = sum(len(s) for s in segs)
        segs.append(seg)
        return off

    add(b"%PDF-1.4\n")
    o1 = add(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")  # catalog
    o2 = add(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")  # pages
    o3 = add(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )  # page
    o4 = add(
        b"4 0 obj\n<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream\nendobj\n"
    )  # contents
    o5 = add(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")  # font
    xref_start = sum(len(s) for s in segs)
    xref = b"xref\n0 6\n0000000000 65535 f \n" + b"".join(
        f"{o:010d} 00000 n \n".encode() for o in [o1, o2, o3, o4, o5]
    )
    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_start).encode() + b"\n%%EOF\n"
    )
    return b"".join(segs) + xref + trailer


def test_manual_html_source_ingested_and_roundtrips(tmp_path):
    body = "<h1>Options</h1><p>" + ("Black-Scholes formula and Greeks. " * 120) + "</p>"
    csv_path, manual_dir = _write_manual(
        tmp_path,
        [
            {
                "title": "Manual Notes",
                "url": "https://ex.com/notes",
                "category": "Options",
                "file": "notes.html",
                "priority_tier": "1",
            }
        ],
        {"notes.html": body.encode()},
    )
    totals = _run(tmp_path, csv_path, manual_dir)
    assert totals["sources_ingested"] == 1
    chunks = load_chunks_jsonl(tmp_path / "chunks.jsonl")
    assert chunks
    assert all(c.title == "Manual Notes" for c in chunks)
    assert all(c.url == "https://ex.com/notes" for c in chunks)
    assert "url" not in json.loads((tmp_path / "report.json").read_text())["sources"][0].get(
        "unused", {}
    )
    # cited by original url, not the local file
    assert str(manual_dir) not in (tmp_path / "chunks.jsonl").read_text()


def test_thin_page_skipped(tmp_path):
    csv_path, manual_dir = _write_manual(
        tmp_path,
        [
            {
                "title": "Thin",
                "url": "https://ex.com/thin",
                "category": "Options",
                "file": "thin.html",
                "priority_tier": "1",
            }
        ],
        {
            "thin.html": b"<html><body><nav>Menu</nav><p>Short page. "
            + b"word " * 120
            + b"</p></body></html>"
        },
    )
    totals = _run(tmp_path, csv_path, manual_dir)
    assert totals["sources_skipped"] == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["sources"][0]["reason"] == "thin_page"


def test_manual_file_missing_reported(tmp_path):
    csv_path, manual_dir = _write_manual(
        tmp_path,
        [
            {
                "title": "Missing",
                "url": "https://ex.com/m",
                "category": "Options",
                "file": "nope.pdf",
                "priority_tier": "1",
            }
        ],
        {},
    )
    totals = _run(tmp_path, csv_path, manual_dir)
    assert totals["sources_selected"] == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["sources"][0]["reason"] == "manual_file_missing"


def test_report_totals_have_token_counts(tmp_path):
    body = "<h1>H</h1><p>" + ("word " * 2000) + "</p>"
    csv_path, manual_dir = _write_manual(
        tmp_path,
        [
            {
                "title": "Big",
                "url": "https://ex.com/big",
                "category": "Options",
                "file": "big.html",
                "priority_tier": "1",
            }
        ],
        {"big.html": body.encode()},
    )
    totals = _run(tmp_path, csv_path, manual_dir)
    assert totals["total_chunks"] >= 1
    assert totals["chunks_over_1000_tokens"] == 0
    assert totals["max_tokens_per_chunk"] <= 900


def test_dry_run_does_not_clobber_real_report(tmp_path, monkeypatch):
    csv_path = tmp_path / "sources.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "url",
                "category",
                "resource_type",
                "check_status",
                "http_status",
                "final_url",
                "content_type_preferred",
                "is_extractable",
                "content_ready",
                "access_type",
                "recommended_action",
                "priority_tier",
                "source_quality",
                "needs_manual_review",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Dry Run Notes",
                "url": "https://example.com/notes",
                "category": "Derivatives",
                "resource_type": "Lecture Notes",
                "check_status": "LIVE",
                "http_status": "200",
                "final_url": "https://example.com/notes",
                "content_type_preferred": "html_docs",
                "is_extractable": "True",
                "content_ready": "yes",
                "access_type": "open",
                "recommended_action": "keep",
                "priority_tier": "1",
                "source_quality": "High",
                "needs_manual_review": "False",
            }
        )

    from quantrag.ingest.__main__ import main

    monkeypatch.chdir(tmp_path)
    rc = main(
        [
            "--csv",
            str(csv_path),
            "--out-chunks",
            "chunks.jsonl",
            "--out-report-json",
            "report.json",
            "--out-report-csv",
            "report.csv",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not (tmp_path / "report.json").exists()
    assert (tmp_path / "report.dryrun.json").exists()
    assert (tmp_path / "report.dryrun.csv").exists()
    assert not (tmp_path / "chunks.jsonl").exists()


def test_manual_pdf_source_read_from_manual_dir_cited_by_url(tmp_path):
    """A manual PDF is read from data/manual/, cited by its original url, and
    never touches the network (so robots.txt cannot apply)."""
    body = "Delta Gamma Vega Theta hedging passage for options pricing. " * 30
    pdf_bytes = _build_pdf(body)
    # Sanity: pypdf round-trips the fixture and extracts enough text to ingest.
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted = "\n".join(p.extract_text() or "" for p in reader.pages)
    assert len(extracted) > 500

    csv_path, manual_dir = _write_manual(
        tmp_path,
        [
            {
                "title": "Greek Notes",
                "url": "https://ex.com/greeks",
                "category": "Option Greeks and Implied Volatility",
                "file": "greeks.pdf",
                "priority_tier": "1",
            }
        ],
        {"greeks.pdf": pdf_bytes},
    )
    totals = _run(tmp_path, csv_path, manual_dir)

    # Ingested (manual bypasses the network -> not failed / not robots_disallowed).
    assert totals["sources_selected"] == 1
    assert totals["sources_ingested"] == 1
    assert totals["sources_failed"] == 0
    assert totals["sources_skipped"] == 0

    report = json.loads((tmp_path / "report.json").read_text())
    src = report["sources"][0]
    assert src["status"] == "ingested"
    assert src["reason"] == "included"
    assert "robot" not in src.get("reason", "").lower()
    assert "manual" in src["origin"].lower() or src["origin"] == "sources.csv"

    # Chunks are cited by the ORIGINAL url, never by the local file path.
    chunks = load_chunks_jsonl(tmp_path / "chunks.jsonl")
    assert chunks
    assert all(c.title == "Greek Notes" for c in chunks)
    assert all(c.url == "https://ex.com/greeks" for c in chunks)
    assert str(manual_dir) not in (tmp_path / "chunks.jsonl").read_text()
    assert "greeks.pdf" not in (tmp_path / "chunks.jsonl").read_text()
