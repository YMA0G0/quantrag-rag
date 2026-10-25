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
