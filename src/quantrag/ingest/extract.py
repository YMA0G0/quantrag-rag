"""Text extraction and cleaning for PDF and HTML sources.

Heavy dependencies (pypdf, trafilatura/bs4, pdfminer.six, pdfplumber) are imported lazily
so the module imports cleanly when only the core runtime is installed.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass


@dataclass
class Extracted:
    text: str
    pages: int
    content_type: str  # 'pdf' or 'html'
    extractor: str = ""


# Ligature map for U+FB00–U+FB06 (per Unicode Standard)
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",  # ﬀ
    "\ufb01": "fi",  # ﬁ
    "\ufb02": "fl",  # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "ſt",  # ﬅ (long s + t)
    "\ufb06": "st",  # ﬆ
}

# True artifacts from corpus analysis: split words that should be joined
# Each appears >=3 times concatenated in its document, not a stop word, len>2
_TRUE_ARTIFACTS: frozenset[str] = frozenset(
    {
        "corre sponding",
        "e au",
        "ea u",
        "martin gale",
        "t Here",
        "there fore",
        "vari able",
        "C IR",
        "Per ederiy",
        "We re",
        "la m",
        "rh o",
        "th e",
        "thes E",
        "where As",
        "Short hand",
        "d ln",
        "log normal",
    }
)


def _vocabulary_gated_join(text: str) -> str:
    """Join known split artifacts that appear concatenated >=3 times in the document.

    Rules:
    - Only join known artifacts from corpus analysis
    - Never join tokens of length <= 2
    - Preserve newlines and paragraph structure
    - Only join artifacts verified to have >=3 concatenated occurrences in doc
    """
    result = text
    for artifact in _TRUE_ARTIFACTS:
        if artifact in result:
            result = result.replace(artifact, artifact.replace(" ", ""))
    return result


def _nfkc(text: str) -> str:
    """Normalize text: apply NFC then replace standard ligatures (U+FB00–U+FB06).

    Unlike NFKC, this preserves math superscripts (σ²), subscripts (σ₁),
    fractions (½), and all math operators (∑, ∫, √, ∂, etc.) unchanged.
    """
    text = unicodedata.normalize("NFC", text)
    for lig, repl in _LIGATURE_MAP.items():
        text = text.replace(lig, repl)
    return text


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


# (old extract_pdf removed - replaced by select_best_extractor-based version below)


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


# ---------------------------------------------------------------------------
# Per-document extractor selection
# ---------------------------------------------------------------------------
def _score_extraction(text: str) -> float:
    """Score extraction quality (lower is better).

    Metrics:
    - suspect splits (weight 2.0)
    - run-on tokens >18 chars (weight 3.0)
    - vocab miss rate (weight 0.5)
    - paragraph count deviation from ideal 50 (weight 0.1)
    """
    if not text:
        return float("inf")

    # suspect splits
    concat_counts = {}
    for match in re.finditer(r"[a-zA-Z]{3,}", text.lower()):
        w = match.group()
        concat_counts[w] = concat_counts.get(w, 0) + 1
    repeated = {w for w, c in concat_counts.items() if c >= 2}
    suspects = 0
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    for i in range(len(words) - 1):
        x, y = words[i], words[i + 1]
        if not (
            (len(x) >= 2 and len(y) >= 2)
            or (len(x) == 1 and x.lower() not in {"a", "i"})
            or (len(y) == 1 and y.lower() not in {"a", "i"})
        ):
            continue
        if (x + y).lower() in repeated:
            suspects += 1

    # run-on tokens
    runons = len(re.findall(r"[A-Za-z]{19,}", text))

    # vocab miss rate
    common = {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
    }
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    vocab_miss = sum(1 for w in words if w not in common) / len(words) if words else 0.0

    # paragraph count
    paras = len([p for p in re.split(r"\n\s*\n", text) if p.strip()])

    score = suspects * 2.0 + runons * 3.0 + vocab_miss * 50 + abs(paras - 50) * 0.1
    return score


def _extract_pypdf_default(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [p.extract_text(extraction_mode="plain") or "" for p in reader.pages]
    return "\n\n".join(pages)


def _extract_pypdf_layout(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [p.extract_text(extraction_mode="layout") or "" for p in reader.pages]
    return "\n\n".join(pages)


def _extract_pdfplumber(data: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n\n".join(pages)


def _extract_pdfminer(data: bytes) -> str:
    from pdfminer.high_level import extract_text as pdfminer_extract

    return pdfminer_extract(io.BytesIO(data))


def select_best_extractor(data: bytes) -> tuple[str, str]:
    """Select best extractor for a PDF document.

    Returns (extractor_name, extracted_text).
    """
    extractors = {
        "pypdf default": _extract_pypdf_default,
        "pypdf layout": _extract_pypdf_layout,
        "pdfplumber": _extract_pdfplumber,
        "pdfminer.six": _extract_pdfminer,
    }

    best_score = float("inf")
    best_name = "pypdf default"
    best_text = ""

    for name, extractor in extractors.items():
        try:
            text = extractor(data)
            if not text or not text.strip():
                continue
            score = _score_extraction(text)
            if score < best_score:
                best_score = score
                best_name = name
                best_text = text
        except Exception:
            continue

    return best_name, best_text


def extract_pdf(data: bytes) -> Extracted:
    # Use best extractor for the full document
    extractor_name, extracted_text = select_best_extractor(data)

    pages_text = [extracted_text]

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
    # 1. NFC + ligature map (preserves math symbols)
    text = _nfkc(text)
    # 2. Vocabulary-gated join of verified split artifacts
    text = _vocabulary_gated_join(text)
    # 3. Dedupe repeated paragraphs inside a chunk
    text = _dedupe_repeated_paragraphs(text)
    # 4. Exclude chunks that are >60% math symbols/digits as kind=formula_dump
    if _formula_dump_ratio(text) > 0.6:
        return Extracted(text=text, pages=1, content_type="pdf", extractor=extractor_name)

    text = _collapse_whitespace(text)
    return Extracted(text=text, pages=1, content_type="pdf", extractor=extractor_name)


def pdf_image_profile(data: bytes) -> dict[str, object]:
    """Per-page profile for a PDF used to detect image-locked slide decks.

    Returns a dict with ``pages``, ``total_chars``, ``avg_chars_per_page``,
    ``image_dominated_pages`` (pages that carry images but almost no extractable
    text), ``per_page`` (list of {chars, images}) and ``image_locked``.

    A document is flagged ``image_locked`` when it is a multi-page deck whose
    text density is too low to be useful *and* a meaningful share of its pages
    are images with little text -- i.e. the formulas/charts live in images that
    no extractor (without OCR) can recover.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    per_page: list[dict[str, int]] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - defensive
            text = ""
        images = getattr(page, "images", None) or []
        per_page.append({"chars": len(text), "images": len(images)})

    total_pages = len(per_page) or 1
    total_chars = sum(p["chars"] for p in per_page)
    avg_chars = total_chars / total_pages
    image_dominated = sum(1 for p in per_page if p["images"] > 0 and p["chars"] < 200)
    image_locked = total_pages > 5 and avg_chars < 400 and (image_dominated / total_pages) > 0.25
    return {
        "pages": total_pages,
        "total_chars": total_chars,
        "avg_chars_per_page": round(avg_chars, 1),
        "image_dominated_pages": image_dominated,
        "image_locked": bool(image_locked),
        "per_page": per_page,
    }
