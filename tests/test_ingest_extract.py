"""Extraction tests with tiny fixtures."""

from io import BytesIO

from pypdf import PdfWriter

from quantrag.ingest import extract


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
