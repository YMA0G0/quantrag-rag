"""Structure-aware chunking of extracted document text.

Constants at the top are overridable by CLI flags.  A hard token cap is
enforced: oversized paragraphs/formula blocks are split by sentence, then by
line, then by a token window as a last resort.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

TARGET_TOKENS = 600
MAX_TOKENS_HARD = 900
OVERLAP_RATIO = 0.12
MIN_CHARS = 200

_ENC = None
_HAS_TIKTOKEN = False
try:  # optional dependency
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover - optional dep
    _ENC = None


def count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def _decode_tokens(tokens: list[int]) -> str:
    if _ENC is not None:
        return _ENC.decode(tokens)
    # unreachable in practice; keeps the signature honest
    return " ".join(str(t) for t in tokens)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    return [p for p in parts if p.strip()]


def _token_window(text: str, max_tokens: int) -> list[str]:
    """Last-resort split: fixed token windows (character windows without tiktoken)."""
    if _ENC is None:
        step = max(1, max_tokens * 4)
        return [text[i : i + step] for i in range(0, len(text), step)]
    tokens = _ENC.encode(text)
    out: list[str] = []
    for start in range(0, len(tokens), max_tokens):
        out.append(_decode_tokens(tokens[start : start + max_tokens]))
    return [o for o in out if o.strip()]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Split ``text`` so every piece is <= ``max_tokens``.

    Order: sentence boundaries, then line boundaries, then token windows.
    Formula/derivation lines are kept together as long as they fit.
    """
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    # 1. sentences
    sentences = _split_sentences(text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sentence in sentences:
        st = count_tokens(sentence)
        if st > max_tokens:
            if buf:
                pieces.append(" ".join(buf))
                buf = []
                buf_tokens = 0
            # 2. lines
            lines = sentence.splitlines() or [sentence]
            lbuf: list[str] = []
            lb_tokens = 0
            for line in lines:
                lt = count_tokens(line)
                if lt > max_tokens:
                    if lbuf:
                        pieces.append("\n".join(lbuf))
                        lbuf = []
                        lb_tokens = 0
                    # 3. token window
                    pieces.extend(_token_window(line, max_tokens))
                    continue
                if lbuf and lb_tokens + lt > max_tokens:
                    pieces.append("\n".join(lbuf))
                    lbuf = []
                    lb_tokens = 0
                lbuf.append(line)
                lb_tokens += lt
            if lbuf:
                pieces.append("\n".join(lbuf))
            continue
        if buf and buf_tokens + st > max_tokens:
            pieces.append(" ".join(buf))
            buf = []
            buf_tokens = 0
        buf.append(sentence)
        buf_tokens += st
    if buf:
        pieces.append(" ".join(buf))
    return [p for p in pieces if p.strip()]


@dataclass
class ChunkDraft:
    text: str
    section: str = ""
    kind: str = "prose"


def _formula_dump_ratio(text: str) -> float:
    """Return the fraction of characters in *text* that are math symbols or digits.

    If > 60 % the chunk is considered a formula dump and excluded.
    Includes Mathematical Alphanumeric Symbols (U+1D400–U+1D7FF) such as
    𝛼, 𝜃, 𝜆, 𝜈, 𝜌 used in formula notation.
    """
    if not text:
        return 0.0
    math_chars = sum(
        1
        for ch in text
        if unicodedata.category(ch) in {"Sk", "Nd", "Nl"}
        or ch in "0123456789"
        or ch in "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        or ch in "σµ∂∫∂½√∞"
        or "\u1d400" <= ch <= "\u1d7ff"  # Mathematical Alphanumeric Symbols
    )
    return math_chars / len(text)


# ---------------------------------------------------------------------------
# Additional chunk filters: toc, references, boilerplate
# ---------------------------------------------------------------------------
_TOC_PATTERNS = [
    re.compile(r"\.\s*\.\s*\.\s*\.\s*\.\s*\.\s*\.\s*\.\s*\.\s*\.\s*\d+\s*$"),  # dotted TOC
    re.compile(
        r"^\s*\d+(\.\d+)*\s+[A-Z][^\n]{0,80}\s*\.\s*\.\s*\.\s*\.\s*\.\s*\d+\s*$",
        re.MULTILINE,
    ),  # numbered TOC
    re.compile(r"^\s*(contents|table of contents)\s*$", re.IGNORECASE | re.MULTILINE),
]

_REFERENCES_PATTERNS = [
    re.compile(r"\b(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b\d{4}\b.*\b(doi|http|journal|vol\.|no\.|pp\.)", re.IGNORECASE),
    re.compile(r"\b(et al\.)", re.IGNORECASE),
]

_BOILERPLATE_PATTERNS = [
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"copyright\s*©?\s*\d{4}", re.IGNORECASE),
    re.compile(r"this document may not be reproduced", re.IGNORECASE),
    re.compile(r"no part of this publication", re.IGNORECASE),
    re.compile(r"published by", re.IGNORECASE),
]

# Notation/appendix: "Appendix B:" headings + parameter definitions with
# math alphanumeric symbols (e.g. "alp: parameter 𝛼 ...")
_APPENDIX_HEADING = re.compile(r"^appendix\s+[A-Z0-9]", re.IGNORECASE | re.MULTILINE)


def _has_math_alpha(text: str) -> bool:
    """True if *text* contains Mathematical Alphanumeric Symbols (U+1D400–U+1D7FF)."""
    return any(0x1D400 <= ord(ch) <= 0x1D7FF for ch in text)


def _looks_like_appendix(text: str) -> bool:
    """Detect appendix/notation chunks: Appendix heading + parameter defs."""
    if not _APPENDIX_HEADING.search(text):
        return False
    return bool(re.search(r":\s*parameter\s+", text, re.IGNORECASE)) and _has_math_alpha(text)


def _looks_like_toc(text: str) -> bool:
    """Detect table-of-contents chunks: dotted leaders, page numbers, section lists."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    toc_lines = sum(1 for pat in _TOC_PATTERNS for line in lines if pat.search(line))
    short_page_lines = sum(
        1
        for line in lines
        if len(line.strip()) < 100 and re.search(r"\d+\s*$", line.strip()) and len(line.strip()) > 3
    )
    return toc_lines >= 2 or short_page_lines >= len(lines) * 0.7


def _looks_like_references(text: str) -> bool:
    """Detect reference/bibliography chunks."""
    if re.search(r"\b(references|bibliography)\s*$", text, re.IGNORECASE | re.MULTILINE):
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        citation_lines = sum(
            1
            for line in lines
            if re.search(r"\b\d{4}\b", line)
            and re.search(r"(doi|http|journal|vol|no|pp|et al)", line, re.IGNORECASE)
        )
        if citation_lines >= len(lines) * 0.5:
            return True
    return False


def _looks_like_boilerplate(text: str) -> bool:
    """Detect boilerplate: copyright, disclaimers, publisher notices."""
    return any(pat.search(text) for pat in _BOILERPLATE_PATTERNS) and len(text.strip()) < 2000


def _chunk_filter_kind(text: str) -> str:
    """Classify a chunk's kind: prose, formula_dump, toc, references, boilerplate."""
    if _formula_dump_ratio(text) > 0.6:
        return "formula_dump"
    if _looks_like_toc(text):
        return "toc"
    if _looks_like_references(text):
        return "references"
    if _looks_like_boilerplate(text):
        return "boilerplate"
    if _looks_like_appendix(text):
        return "formula_dump"
    return "prose"


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines or not sections:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def chunk_text(
    text: str,
    *,
    title: str,
    url: str,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS_HARD,
    overlap_ratio: float = OVERLAP_RATIO,
    min_chars: int = MIN_CHARS,
) -> list[ChunkDraft]:
    """Return structure-aware chunks (text without context header).

    Guarantees every returned chunk is <= ``max_tokens`` by token count.
    """
    if not text or not text.strip():
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    drafts: list[ChunkDraft] = []
    for heading, body in _sections(text):
        if not body:
            continue
        blocks = _split_paragraphs(body)
        buf: list[str] = []
        buf_tokens = 0
        for block in blocks:
            bt = count_tokens(block)
            if bt > max_tokens:
                if buf:
                    drafts.append(ChunkDraft("\n\n".join(buf), heading))
                    buf, buf_tokens = [], 0
                for piece in _hard_split(block, max_tokens):
                    drafts.append(ChunkDraft(piece, heading))
                continue
            if buf and buf_tokens + bt > target_tokens:
                drafts.append(ChunkDraft("\n\n".join(buf), heading))
                # carry overlap from the tail of the previous chunk
                carry: list[str] = []
                carry_tokens = 0
                overlap_budget = int(target_tokens * overlap_ratio)
                for prev in reversed(buf):
                    pt = count_tokens(prev)
                    if carry_tokens + pt <= overlap_budget:
                        carry.insert(0, prev)
                        carry_tokens += pt
                    else:
                        break
                buf, buf_tokens = carry, carry_tokens
            buf.append(block)
            buf_tokens += bt
        if buf:
            drafts.append(ChunkDraft("\n\n".join(buf), heading))

    # Safety net: never emit a chunk above the hard cap, counting the context
    # header that will be prepended before embedding.
    capped: list[ChunkDraft] = []
    for draft in drafts:
        overhead = count_tokens(add_context_header("", title, draft.section))
        # small margin: BPE merges across the header/body boundary are not additive
        budget = max(1, max_tokens - overhead - 10)
        if count_tokens(draft.text) <= budget:
            capped.append(draft)
        else:
            for piece in _hard_split(draft.text, budget):
                # _hard_split can return pieces slightly above budget because
                # joining sentences with spaces can create new BPE merges that
                # increase the token count vs the sum of individual counts.
                # Belt-and-suspenders: split any remaining oversized pieces.
                if count_tokens(piece) <= budget:
                    capped.append(ChunkDraft(piece, draft.section))
                else:
                    for subpiece in _token_window(piece, budget):
                        capped.append(ChunkDraft(subpiece, draft.section))
    drafts = capped

    if len(drafts) > 1:
        drafts = [d for d in drafts if len(d.text.strip()) >= min_chars]

    # Classify each chunk: formula_dump, toc, references, boilerplate, or prose
    kept: list[ChunkDraft] = []
    kind_counts: dict[str, int] = {}
    for draft in drafts:
        draft.kind = _chunk_filter_kind(draft.text)
        kind_counts[draft.kind] = kind_counts.get(draft.kind, 0) + 1
        if draft.kind != "prose":
            continue
        kept.append(draft)
    drafts = kept

    seen: set[str] = set()
    unique: list[ChunkDraft] = []
    for draft in drafts:
        norm = re.sub(r"\s+", " ", draft.text.strip()).lower()
        key = hashlib.sha1(norm.encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(draft)
    return unique


def make_chunk_id(url: str, index: int) -> str:
    return f"{hashlib.sha1(url.encode()).hexdigest()[:10]}-{index:04d}"


def add_context_header(text: str, title: str, section: str) -> str:
    header = title.strip()
    if section.strip():
        header = f"{header} — {section.strip()}"
    return f"{header}\n\n{text.strip()}"


# ---------------------------------------------------------------------------
# Corpus health check
# ---------------------------------------------------------------------------
def corpus_health_check(texts: list[str]) -> dict[str, object]:
    """Check corpus health.

    Fails if >2% of tokens are >25 letters OR average chars per token < 3.5.
    Returns a dict with ``ok`` flag and metrics.
    """
    all_tokens: list[str] = []
    total_chars = 0

    for text in texts:
        toks = text.split()
        all_tokens.extend(toks)
        total_chars += sum(len(t) for t in toks)

    total_tokens = len(all_tokens)
    if total_tokens == 0:
        return {
            "ok": True,
            "tokens": 0,
            "long_tokens": 0,
            "long_token_pct": 0.0,
            "avg_chars_per_token": 0.0,
            "issues": [],
        }

    long_tokens = [t for t in all_tokens if len(t) > 25]
    long_token_pct = len(long_tokens) / total_tokens * 100
    avg_chars_per_token = total_chars / total_tokens

    issues: list[str] = []
    if long_token_pct > 2.0:
        issues.append(f"Long tokens (>25 letters): {long_token_pct:.2f}% > 2%")
    if avg_chars_per_token < 3.5:
        issues.append(f"Avg chars/token: {avg_chars_per_token:.2f} < 3.5")

    return {
        "ok": len(issues) == 0,
        "tokens": total_tokens,
        "long_tokens": len(long_tokens),
        "long_token_pct": round(long_token_pct, 2),
        "avg_chars_per_token": round(avg_chars_per_token, 2),
        "issues": issues,
    }
