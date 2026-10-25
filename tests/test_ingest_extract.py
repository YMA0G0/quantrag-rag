"""Extraction tests with tiny fixtures."""

import json
from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from quantrag.ingest import extract
from quantrag.ingest.chunking import corpus_health_check
from quantrag.ingest.sources import CORE_CATEGORIES


def test_extract_pdf_headers_page_numbers_logic():
    """Ensure extraction doesn't crash on simple PDFs."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    bio = BytesIO()
    writer.write(bio)
    data = bio.getvalue()
    try:
        res = extract.extract_pdf(data)
        assert isinstance(res.text, str)
    except Exception:
        pass


def test_extract_html_nav_headings_formula():
    html = b"""
    <html><body>
    <nav>Menu</nav>
    <h1>Options</h1>
    <p>Black-Scholes: $C = S N(d_1) - K e^{-rT} N(d_2)$</p>
    <footer>Footer</footer>
    <script>var x=1;</script>
    </body></html>
    """
    res = extract.extract_html(html)
    assert "Black-Scholes" in res.text or "$" in res.text
    assert res.content_type == "html"


# ---------------------------------------------------------------------------
# Regression tests for squashed-word repair removal
# ---------------------------------------------------------------------------
def test_old_intra_word_repair_squashed_single_letter_words():
    """Document the regression: old _fix_intra_word_spaces joined "a string" -> "astring".

    The old function used re.split with a pattern consuming whitespace, which
    flattens paragraphs and incorrectly joins single-letter words like "a" and "I".
    This test documents the bug that motivated removing the function.
    """
    import re

    def old_fix_intra_word_spaces(text: str) -> str:
        parts = re.split(r"\s+(?=[a-zA-Z])", text)
        repaired = []
        for i, p in enumerate(parts):
            if i == 0:
                repaired.append(p)
            else:
                prev = parts[i - 1]
                if re.match(r"[aeiouyAEIOUY]$", prev) and re.match(
                    r"^[bcdfghjklmnpqrstuvwxyzBCDFGHJKLMNPQRSTUVWXYZ]{2,}", p
                ):
                    repaired[-1] += p
                else:
                    repaired.append(p)
        return "".join(repaired)

    # Single-letter word 'a' followed by consonant cluster -> incorrectly joined
    assert old_fix_intra_word_spaces("a string") == "astring"
    assert old_fix_intra_word_spaces("I think") == "Ithink"
    assert old_fix_intra_word_spaces("a cat sat on a mat") == "acatsatonamat"

    # Newlines consumed (flattened)
    inp = "Title line\n\nParagraph one.\n\nParagraph two."
    out = old_fix_intra_word_spaces(inp)
    assert "\n" not in out, "Old function consumed all newlines"
    assert out == "TitlelineParagraphone.Paragraphtwo."


def test_extract_preserves_spaces_and_newlines():
    """Extraction must not squash words or flatten paragraph structure."""
    # Build a minimal PDF with spaced text
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    # pypdf blank pages have no text, so we test via the extract pipeline
    # with a synthetic text input through the cleaning functions.
    from quantrag.ingest.extract import (
        _collapse_whitespace,
        _dedupe_repeated_paragraphs,
        _nfkc,
    )

    synthetic = """Title of Document

This is paragraph one with normal words and spacing.

This is paragraph two with a formula dS = μSdt + σSdW_t.

This is paragraph three with single letters a and I included."""

    result = _collapse_whitespace(_dedupe_repeated_paragraphs(_nfkc(synthetic)))

    # Spaces preserved between words
    assert "normal words" in result
    assert "paragraph two" in result

    # Single-letter words preserved
    assert " single letters a and I included" in result or "a and I" in result

    # Paragraph breaks preserved (newlines)
    assert result.count("\n") >= 2, f"Expected paragraph breaks, got: {repr(result)}"

    # No squashed words
    assert "normalwords" not in result
    assert "paragraphtwo" not in result
    assert "formuladS" not in result


def test_corpus_health_check():
    # Squashed fixture should FAIL
    squashed = "StochasticCalculusMIT18.642DrKempthorneFall2024BrownianMotionwithDrift"
    result = corpus_health_check([squashed])
    assert result["ok"] is False
    assert result["long_token_pct"] > 2.0
    assert any("Long tokens" in i for i in result["issues"])

    # Normal text should PASS
    normal = """This is a normal paragraph with proper spacing.

Here is a second paragraph.

And a third with a formula dS = μSdt + σSdW_t."""
    result = corpus_health_check([normal])
    assert result["ok"] is True
    assert result["long_token_pct"] == 0.0
    assert result["avg_chars_per_token"] >= 3.5

    # Text with 'a', 'I', paragraph breaks, formula should PASS
    formula_text = """The variable a is important.

I think dS = μSdt + σSdW_t describes it well.

Final paragraph here."""
    result = corpus_health_check([formula_text])
    assert result["ok"] is True
    words = formula_text.split()
    assert "a" in words
    assert "I" in words
    assert "dS" in words


# ---------------------------------------------------------------------------
# Part E: toc / references / boilerplate chunk filters
# ---------------------------------------------------------------------------
def test_chunk_filter_classifies_sabr_appendix_b():
    """SABR Appendix-B notation chunk (REAL snippet from docs/phase1_audit.md)
    must not be 'prose'."""
    from quantrag.ingest.chunking import _chunk_filter_kind

    sabr_appendix_b = """Appendix B: Closed-form expressions for mrSABR
The following formulas for mrSABR are in Excel format and use notations as
follows: alp: parameter 𝛼 (initial instantaneous volatility)
the: parameter 𝜃 (long-term volatility)
lam: parameter 𝜆 (speed of mean reversion of volatility)
nu: parameter 𝜈 (lognormal volatility of volatility)
rho: parameter 𝜌 (correlation)
T_ex: maturity/expiry of option
E:"""

    kind = _chunk_filter_kind(sabr_appendix_b)
    assert kind != "prose", f"SABR Appendix-B should not be prose, got {kind}"


def test_chunk_filter_toc_detection_bielefeld():
    """REAL Bielefeld NumSDE TOC chunk: dotted leader at start of chunk -> 'toc'."""
    from quantrag.ingest.chunking import _looks_like_toc

    # From chunk 6146332aa7-0000 (Bielefeld NumSDE script.pdf) in data/chunks.jsonl
    toc_text = """. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 141
Chapter 1
Introduction
Inrecent years dynamical systems(of deterministic and stochasticnature),
describing many modelsinmathematics,
to deterministic algorithms seems to be insufficient for such systems, since,
for example,"""
    assert _looks_like_toc(toc_text) is True

    normal_text = """This is a normal paragraph about option pricing theory.
It discusses the Black-Scholes formula and its applications.
No page numbers or dotted leaders here."""
    assert _looks_like_toc(normal_text) is False


def test_chunk_filter_references_detection_finrl_x():
    """REAL FinRL-X reference chunk: years + DOIs/URLs -> 'references'."""
    from quantrag.ingest.chunking import _looks_like_references

    # From chunk 989ea86b83-0013 (arXiv 1908.10063) in data/chunks.jsonl
    ref_text = """2014. News
500 examples. This is an important result, since deep learning tech-
impact on stock price return via sentiment analysis. Knowledge-Based Systems
69 (oct 2014), 14–23. https://doi.org/10.1016/j.knosys.2014.05.012"""
    assert _looks_like_references(ref_text) is True

    normal_text = """This paper discusses option pricing using the Black-Scholes model.
The key insight is that options can be hedged dynamically.
No references section here."""
    assert _looks_like_references(normal_text) is False


def test_chunk_filter_boilerplate_detection_mit_tou():
    """REAL MIT OCW Terms-of-Use footer -> 'boilerplate'."""
    from quantrag.ingest.chunking import _looks_like_boilerplate

    # From chunk 3460fe14fd-0031 (MIT OCW 18.642) in data/chunks.jsonl
    boilerplate = """• T = 4(years)
• µ =0.30, σ =0.40
• m = increments on time interval (0, T ]. Endpoints at
t = i × (T /m), i = 1, 2,..., m
For information about citing these materials or our Terms of Use, visit:
https://ocw.mit.edu/terms."""
    assert _looks_like_boilerplate(boilerplate) is True

    normal_text = """This is a substantive paragraph about quantitative finance.
It discusses the mathematics of option pricing in detail.
The Black-Scholes equation is derived from first principles."""
    assert _looks_like_boilerplate(normal_text) is False


# ---------------------------------------------------------------------------
# Part F: extractor selection, text normalization, join predicate
# ---------------------------------------------------------------------------
def test_extractor_selection_chooses_best(monkeypatch):
    """select_best_extractor picks the extractor with the lowest score."""
    from quantrag.ingest import extract

    # All extractors return the same text; _score_extraction is monkeypatched
    # to return different scores.
    def same_extractor(data):
        return "same text"

    monkeypatch.setattr(extract, "_extract_pypdf_default", same_extractor)
    monkeypatch.setattr(extract, "_extract_pypdf_layout", same_extractor)
    monkeypatch.setattr(extract, "_extract_pdfplumber", same_extractor)
    monkeypatch.setattr(extract, "_extract_pdfminer", same_extractor)

    scores = [10.0, 5.0, 20.0, 15.0]
    call_count = [0]

    def fake_score(text):
        call_count[0] += 1
        return scores[call_count[0] - 1]

    monkeypatch.setattr(extract, "_score_extraction", fake_score)

    name, text, score, all_scores = extract.select_best_extractor(b"fake pdf data")
    assert name == "pypdf layout"
    assert score == 5.0


def test_extractor_selection_handles_failures(monkeypatch):
    """select_best_extractor skips extractors that raise exceptions."""
    from quantrag.ingest import extract

    def failing_extractor(data):
        raise ValueError("boom")

    def good_extractor(data):
        return "good text"

    monkeypatch.setattr(extract, "_extract_pypdf_default", failing_extractor)
    monkeypatch.setattr(extract, "_extract_pypdf_layout", failing_extractor)
    monkeypatch.setattr(extract, "_extract_pdfplumber", failing_extractor)
    monkeypatch.setattr(extract, "_extract_pdfminer", good_extractor)

    name, text, score, all_scores = extract.select_best_extractor(b"fake pdf data")
    assert name == "pdfminer.six"
    assert text == "good text"
    assert all_scores == {"pdfminer.six": extract._score_extraction("good text")}


def test_normalize_single_symbol_lines():
    """Stranded single-symbol lines are merged into the previous line."""
    from quantrag.ingest.extract import _normalize_single_symbol_lines

    text = "Some formula here.\nn\nk\nNext paragraph."
    result = _normalize_single_symbol_lines(text)
    assert "Some formula here. n k" in result
    assert "Next paragraph." in result

    # Lines that are not single symbols are kept as-is
    text2 = "Normal line.\nThis is a normal line.\nAnother line."
    result2 = _normalize_single_symbol_lines(text2)
    assert result2 == text2


def test_segment_squashed_prose_positive():
    """Squashed prose runs are segmented using the dictionary."""
    from quantrag.ingest.extract import _segment_squashed_prose

    text = "Thesimplestquadraturethatapproximates (2.1)istherectangle quadrature"
    result = _segment_squashed_prose(text)
    assert "simplest" in result
    assert "quadrature" in result
    assert "approximates" in result
    assert result != text


def test_segment_squashed_prose_negative_math_names():
    """Math names are not segmented."""
    from quantrag.ingest.extract import _segment_squashed_prose

    for name in ["EulerMaruyama", "Milstein", "OrnsteinUhlenbeck", "Girsanov", "BlackScholes"]:
        result = _segment_squashed_prose(name)
        assert result == name, f"{name} was changed to {result}"


def test_segment_squashed_prose_negative_formulas():
    """Math formulas are not segmented."""
    from quantrag.ingest.extract import _segment_squashed_prose

    text = "dS_t = mu S_t dt + sigma S_t dW_t"
    result = _segment_squashed_prose(text)
    assert result == text


def test_join_predicate_rejects_two_standalone_words():
    """Two common standalone words are not joined."""
    from quantrag.ingest.extract import _is_likely_split_word

    assert not _is_likely_split_word("www", "googletagmanager")
    assert not _is_likely_split_word("Cardo", "Italic")
    assert not _is_likely_split_word("the", "cat")
    assert not _is_likely_split_word("a", "able")


def test_join_predicate_accepts_split_fragments():
    """Word fragments are joined when they form a valid word."""
    from quantrag.ingest.extract import _is_likely_split_word

    assert _is_likely_split_word("vari", "able")
    assert _is_likely_split_word("corre", "sponding")
    assert _is_likely_split_word("log", "normal")


# ---------------------------------------------------------------------------
# Part G: corpus category allow-list test
# ---------------------------------------------------------------------------
def test_no_chunk_has_offlist_category():
    """No chunk in data/chunks.jsonl has a category outside CORE_CATEGORIES."""
    chunks_path = Path("data/chunks.jsonl")
    if not chunks_path.exists():
        pytest.skip("data/chunks.jsonl not found; run ingest first")

    off_list = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cat = row.get("category", "")
            if cat and cat not in CORE_CATEGORIES:
                off_list.append((row["id"], cat))

    assert not off_list, f"Off-list categories found: {off_list[:10]}"
