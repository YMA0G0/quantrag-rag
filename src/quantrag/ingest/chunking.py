"""Structure-aware chunking of extracted document text.

Constants at the top are overridable by CLI flags.  A hard token cap is
enforced: oversized paragraphs/formula blocks are split by sentence, then by
line, then by a token window as a last resort.
"""

from __future__ import annotations

import hashlib
import re
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
                capped.append(ChunkDraft(piece, draft.section))
    drafts = capped

    if len(drafts) > 1:
        drafts = [d for d in drafts if len(d.text.strip()) >= min_chars]

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
