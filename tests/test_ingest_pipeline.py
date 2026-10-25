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
                "category": "Option Greeks and Implied Volatility",
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
                "category": "Option Greeks and Implied Volatility",
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
                "category": "Black–Scholes and Option Pricing",
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
                "category": "Computational Finance and Numerical Methods",
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
                "category": "Financial Derivatives and Exotic Options",
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


def _build_image_pdf(pages_spec: list[tuple[str, bool]]) -> bytes:
    """Build a minimal valid PDF where each spec entry is ``(text, has_image)``.

    Pages with ``has_image=True`` carry a 1x1 grayscale image XObject and almost
    no text, simulating image-locked slides that no extractor can read without
    OCR.  Designed to trigger ``pdf_image_profile``'s ``image_locked`` flag.
    """
    N = len(pages_spec)
    any_image = any(has_image for _, has_image in pages_spec)
    font_num = 3 + 2 * N
    img_num = 3 + 2 * N + 1 if any_image else None
    max_obj = img_num if any_image else font_num
    img_stream = b"\x00"
    img_xobj = (
        (
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
            b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length %d >>\nstream\n" % len(img_stream)
        )
        + img_stream
        + b"\nendstream\n"
    )

    segs: list[bytes] = []
    off: dict[int, int] = {}

    def add_obj(num: int, body: bytes) -> None:
        full = (b"%d 0 obj\n" % num) + body + b"endobj\n"
        off[num] = sum(len(s) for s in segs)
        segs.append(full)

    segs.append(b"%PDF-1.4\n")
    add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    page_refs = " ".join(f"{3 + i} 0 R" for i in range(N))
    add_obj(2, (f"<< /Type /Pages /Kids [{page_refs}] /Count {N} >>").encode())
    for i in range(N):
        text, has_image = pages_spec[i]
        page_num = 3 + i
        content_num = 3 + N + i
        content = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode("latin-1")
        if has_image:
            resources = b"<< /Font << /F1 %d 0 R >> /XObject << /Im1 %d 0 R >> >>" % (
                font_num,
                img_num,
            )
        else:
            resources = b"<< /Font << /F1 %d 0 R >> >>" % (font_num,)
        page_body = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources %s >>"
        ) % (content_num, resources)
        add_obj(page_num, page_body)
        cbody = (b"<< /Length %d >>\nstream\n" % len(content)) + content + b"\nendstream"
        add_obj(content_num, cbody)
    add_obj(font_num, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    if any_image:
        add_obj(img_num, img_xobj)
    xref_start = sum(len(s) for s in segs)
    xref = b"xref\n0 %d\n0000000000 65535 f \n" % (max_obj + 1)
    xref += b"".join(f"{off[i]:010d} 00000 n \n".encode() for i in range(1, max_obj + 1))
    trailer = (
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n" % (max_obj + 1)
        + str(xref_start).encode()
        + b"\n%%EOF\n"
    )
    return b"".join(segs) + xref + trailer


def test_manual_duplicate_url_skipped_once(tmp_path: Path) -> None:
    """Two manual rows sharing a normalised URL: first kept, later skipped
    with ``duplicate_url``; corpus has ONE set of chunks (no text-hash dupes)."""
    body = "Black-Scholes formula and Greeks passage. " * 40
    pdf_bytes = _build_pdf(body)
    csv_path, manual_dir = _write_manual(
        tmp_path,
        [
            {
                "title": "First Copy",
                "url": "https://ex.com/dup",
                "category": "Black–Scholes and Option Pricing",
                "file": "a.pdf",
                "priority_tier": "1",
            },
            {
                "title": "Second Copy",
                "url": "https://ex.com/dup/",
                "category": "Black–Scholes and Option Pricing",
                "file": "b.pdf",
                "priority_tier": "1",
            },
        ],
        {"a.pdf": pdf_bytes, "b.pdf": pdf_bytes},
    )
    totals = _run(tmp_path, csv_path, manual_dir)

    assert totals["sources_selected"] == 1
    assert totals["sources_ingested"] == 1
    assert totals["sources_skipped"] == 1
    assert totals["total_chunks"] >= 1

    report = json.loads((tmp_path / "report.json").read_text())
    statuses = {r["title"]: r["reason"] for r in report["sources"]}
    assert statuses["First Copy"] == "included"
    assert statuses["Second Copy"] == "duplicate_url"

    # No duplicate chunks by text hash.
    chunks = load_chunks_jsonl(tmp_path / "chunks.jsonl")
    hashes = {
        __import__("hashlib").sha1(" ".join(c.text.split()).lower().encode()).hexdigest()
        for c in chunks
    }
    assert len(hashes) == len(chunks)


def test_manual_pdf_image_locked_needs_ocr(tmp_path: Path) -> None:
    """An image-locked slide deck (many pages, images, little text) is marked
    ``needs_ocr`` and excluded from the corpus rather than partially ingested."""
    # 8 pages: 4 image-dominated (tiny text + image), 4 short text -> image_locked
    pages_spec = [("X", True)] * 4 + [("short text", False)] * 4
    pdf_bytes = _build_image_pdf(pages_spec)

    csv_path, manual_dir = _write_manual(
        tmp_path,
        [
            {
                "title": "Locked Slides",
                "url": "https://ex.com/locked",
                "category": "Option Greeks and Implied Volatility",
                "file": "locked.pdf",
                "priority_tier": "1",
            }
        ],
        {"locked.pdf": pdf_bytes},
    )
    totals = _run(tmp_path, csv_path, manual_dir)

    assert totals["sources_selected"] == 1
    assert totals["sources_ingested"] == 0
    assert totals["sources_skipped"] == 1
    assert totals["total_chunks"] == 0

    report = json.loads((tmp_path / "report.json").read_text())
    src = report["sources"][0]
    assert src["status"] == "skipped"
    assert src["reason"] == "needs_ocr"
    assert src["pages"] == "8"

    # Nothing written to the corpus.
    chunks_path = tmp_path / "chunks.jsonl"
    assert not chunks_path.exists() or chunks_path.read_text() == ""


def test_manual_sources_included_under_core_profile(tmp_path: Path) -> None:
    """Under ``--profile core`` both manual sources are selected and carry the
    exact allow-list category names (never filtered out by the core profile)."""
    from quantrag.ingest.sources import CORE_CATEGORIES

    body = "Black-Scholes formula and Greeks passage. " * 40
    pdf_bytes = _build_pdf(body)
    csv_path, manual_dir = _write_manual(
        tmp_path,
        [
            {
                "title": "BS Notes",
                "url": "https://ex.com/bs",
                "category": "Black–Scholes and Option Pricing",
                "file": "bs.pdf",
                "priority_tier": "1",
            },
            {
                "title": "Greeks Notes",
                "url": "https://ex.com/greeks",
                "category": "Option Greeks and Implied Volatility",
                "file": "greeks.pdf",
                "priority_tier": "1",
            },
        ],
        {"bs.pdf": pdf_bytes, "greeks.pdf": pdf_bytes},
    )
    for profile in ("core", "all"):
        totals = _run(tmp_path, csv_path, manual_dir)
        assert totals["sources_selected"] == 2, profile
        assert totals["sources_ingested"] == 2, profile
        assert totals["sources_skipped"] == 0, profile

        report = json.loads((tmp_path / "report.json").read_text())
        urls = {r["url"] for r in report["sources"]}
        assert urls == {"https://ex.com/bs", "https://ex.com/greeks"}, (profile, report["sources"])

        # Categories match the allow-list exactly.
        assert "Black–Scholes and Option Pricing" in CORE_CATEGORIES
        assert "Option Greeks and Implied Volatility" in CORE_CATEGORIES


def test_report_does_not_collapse_same_url_rows(tmp_path: Path) -> None:
    """A CSV row and a manual row sharing a URL but different origins both
    appear in the report (keyed by (url, origin), not URL alone)."""
    body = "Black-Scholes passage. " * 40
    pdf_bytes = _build_pdf(body)
    csv_path, manual_dir = _write_manual(
        tmp_path,
        [
            {
                "title": "Manual Copy",
                "url": "https://ex.com/shared",
                "category": "Black–Scholes and Option Pricing",
                "file": "m.pdf",
                "priority_tier": "1",
            }
        ],
        {"m.pdf": pdf_bytes},
    )
    # Add a CSV source with the same URL but a different origin.
    import csv

    csv_path2 = tmp_path / "kb.csv"
    with csv_path2.open("w", newline="", encoding="utf-8") as handle:
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
                "title": "CSV Copy",
                "url": "https://ex.com/shared",
                "category": "Financial Derivatives and Exotic Options",
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
        )

    from quantrag.ingest import pipeline

    totals = pipeline.run_pipeline(
        [str(csv_path2)],
        out_chunks=tmp_path / "chunks.jsonl",
        out_report_json=tmp_path / "report.json",
        out_report_csv=tmp_path / "report.csv",
        raw_dir=tmp_path / "raw",
        manual_csv=str(csv_path),
        manual_dir=str(manual_dir),
    )
    assert totals["sources_selected"] == 2
    report = json.loads((tmp_path / "report.json").read_text())
    urls = [(r["url"], r["origin"]) for r in report["sources"]]
    assert ("https://ex.com/shared", "kb.csv") in urls
    assert ("https://ex.com/shared", "manual_sources.csv") in urls
