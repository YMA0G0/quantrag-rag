"""Text extraction and cleaning for PDF and HTML sources.

Heavy dependencies (pypdf, trafilatura/bs4) are imported lazily so the module
imports cleanly when only the core runtime is installed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class Extracted:
    text: str
    pages: int
    content_type: str  # 'pdf' or 'html'


def _nfkc(text: str) -> str:
    """NFKC normalisation to fuse ligatures (e.g. ﬁ → fi)."""
    return unicodedata.normalize("NFKC", text)


def _dedupe_repeated_paragraphs(text: str) -> str:
    """Remove duplicate consecutive paragraphs within a chunk.

    If the same paragraph text appears on consecutive lines/blocks, keep only
    the first occurrence.  This is a best-effort measure – the pipeline already
    has de-duplication at the chunk‐hash level, but removing intra-chunk
    repetitions improves retrieval quality.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen: set[str] = set()
    unique_paras: list[str] = []
    for para in paragraphs:
        norm = para.strip().lower()
        if norm not in seen:
            seen.add(norm)
            unique_paras.append(para)
    return "\n\n".join(unique_paras)


def _formula_dump_ratio(text: str) -> float:
    """Return the fraction of characters in *text* that are math symbols or digits.

    If > 60 % the chunk is considered a formula dump and excluded.
    """
    if not text:
        return 0.0
    # Math symbols from Unicode's "Mathematical Operators" block and digits.
    math_range = sum(
        1
        for ch in text
        if unicodedata.category(ch) in {"Sk", "Nd", "Nl"}  # math symbols / numbers
        or ch in "0123456789"
        or ch in "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        or ch in "σµ∂∫∂½√∞"
    )
    return math_range / len(text)


def _strip_title(text: str, title: str) -> str:
    """Remove the prepended document title from chunk text, if present.

    The ingest layer prepends the document title to every chunk; stripping it
    avoids marking every chunk of a document as relevant.
    """
    if title and text.startswith(title):
        return text[len(title) :]
    return text


def extract_pdf(data: bytes) -> Extracted:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("pypdf is required for PDF extraction (install extra 'ingest')") from exc
    import io

    reader = PdfReader(io.BytesIO(data))
    pages_text: list[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")

    # remove repeated headers/footers: lines appearing on many pages
    line_counts: dict[str, int] = {}
    page_lines: list[list[str]] = []
    for pt in pages_text:
        lines = [ln.rstrip() for ln in pt.splitlines()]
        page_lines.append(lines)
        seen_on_page: set[str] = set()
        for ln in lines:
            s = ln.strip()
            if s and s not in seen_on_page:
                line_counts[s] = line_counts.get(s, 0) + 1
                seen_on_page.add(s)

    threshold = max(2, len(pages_text) // 2) if pages_text else 2
    boilerplate = {ln for ln, c in line_counts.items() if c >= threshold}

    def _is_page_furniture(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        # "12", "12 / 26", "Page 12 of 26", "- 12 -"
        if re.fullmatch(r"\d{1,4}", s):
            return True
        if re.fullmatch(r"(?:page\s*)?\d{1,4}\s*(?:/|of)\s*\d{1,4}", s, re.IGNORECASE):
            return True
        if re.fullmatch(r"[-–—]\s*\d{1,4}\s*[-–—]", s):
            return True
        # encoding artefacts: only replacement chars / stray marks, no letters/digits
        return bool(not any(ch.isalnum() for ch in s) and len(s) <= 8)

    cleaned_pages: list[str] = []
    for lines in page_lines:
        out: list[str] = []
        for ln in lines:
            s = ln.strip()
            if s in boilerplate:
                continue
            if _is_page_furniture(s):
                continue
            out.append(ln)
        cleaned_pages.append("\n".join(out))

    text = "\n\n".join(cleaned_pages)
    # de-hyphenate line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # ---- NEW cleaning steps ----
    # 1. NFKC normalisation (fuses ligatures such as ﬁ → fi)
    text = _nfkc(text)
    # 4. Dedupe repeated paragraphs inside a chunk
    text = _dedupe_repeated_paragraphs(text)
    # 5. Exclude chunks that are >60% math symbols/digits as kind=formula_dump
    if _formula_dump_ratio(text) > 0.6:
        # Return a minimal Extracted so the pipeline can record the reason
        # and the chunk will be dropped later in chunk_text via the ratio check.
        # We set content_type so the caller knows why it was excluded.
        return Extracted(text=text, pages=len(pages_text), content_type="pdf")

    text = _collapse_whitespace(text)
    return Extracted(text=text, pages=len(pages_text), content_type="pdf")


def _collapse_whitespace(text: str) -> str:
    # collapse runs of spaces/tabs but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_html(data: bytes) -> Extracted:
    # try trafilatura first
    try:
        import trafilatura

        html = data.decode("utf-8", errors="replace")
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            output_format="txt",
            with_metadata=False,
        )
        if extracted and len(extracted.strip()) >= 500:
            text = _collapse_whitespace(extracted)
            # headings: trafilatura already returns plain text; keep as is
            return Extracted(text=text, pages=1, content_type="html")
    except ImportError:
        pass
    except Exception:
        pass

    # fallback bs4
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("beautifulsoup4 is required for HTML extraction") from exc
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    # headings kept with level prefix
    parts: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code"]):
        name = el.name
        if name and name.startswith("h") and len(name) == 2:
            level = name[1]
            parts.append(f"{'#' * int(level)} {el.get_text(' ', strip=True)}")
        else:
            parts.append(el.get_text(" ", strip=True))
    text = "\n".join(p for p in parts if p.strip())
    text = _collapse_whitespace(text)
    return Extracted(text=text, pages=1, content_type="html")
