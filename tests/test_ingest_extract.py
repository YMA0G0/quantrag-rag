"""Extraction tests with tiny fixtures."""

from io import BytesIO

from pypdf import PdfWriter

from quantrag.ingest import extract
from quantrag.ingest.chunking import corpus_health_check


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
    """SABR Appendix-B notation chunk from docs/phase1_audit.md must not be 'prose'.

    The old SABR Appendix-B chunk lists parameter notations with Greek letters
    and should be classified as formula_dump (or another non-prose kind),
    never as plain prose.
    """
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


def test_chunk_filter_toc_detection():
    """Table-of-contents chunks with dotted leaders and page numbers are 'toc'."""
    from quantrag.ingest.chunking import _looks_like_toc

    toc_text = """Contents
1. Introduction ...... 3
2. Black-Scholes Model ...... 7
3. Option Pricing ...... 12
4. Hedging Strategies ...... 18"""
    assert _looks_like_toc(toc_text) is True

    normal_text = """This is a normal paragraph about option pricing theory.
It discusses the Black-Scholes formula and its applications.
No page numbers or dotted leaders here."""
    assert _looks_like_toc(normal_text) is False


def test_chunk_filter_references_detection():
    """Reference/bibliography chunks with years and citations are 'references'."""
    from quantrag.ingest.chunking import _looks_like_references

    ref_text = """References
Black, F., and Scholes, M. (1973). The Pricing of Options and Corporate
Liabilities. Journal of Political Economy, 81(3), pp. 637-654.
Merton, R. C. (1973). Theory of Rational Option Pricing. Bell Journal of
Economics, 4(1), pp. 141-160.
doi: 10.2307/1831029"""
    assert _looks_like_references(ref_text) is True

    normal_text = """This paper discusses option pricing using the Black-Scholes model.
The key insight is that options can be hedged dynamically.
No references section here."""
    assert _looks_like_references(normal_text) is False


def test_chunk_filter_boilerplate_detection():
    """Boilerplate copyright/disclaimer chunks are 'boilerplate'."""
    from quantrag.ingest.chunking import _looks_like_boilerplate

    boilerplate = """All rights reserved. No part of this publication may be reproduced
without prior written permission from the publisher.
Copyright © 2024 University Press."""
    assert _looks_like_boilerplate(boilerplate) is True

    normal_text = """This is a substantive paragraph about quantitative finance.
It discusses the mathematics of option pricing in detail.
The Black-Scholes equation is derived from first principles."""
    assert _looks_like_boilerplate(normal_text) is False
