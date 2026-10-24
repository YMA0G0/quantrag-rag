"""Evaluation primitives for retrieval.

This module owns the *definition* of relevance (a content rule per query),
the derivation of ground-truth chunk ids from that rule, and the ranking
metrics.  Nothing here may consult retriever output: the ground truth is a
property of the corpus plus the query rule only.

Rule semantics (see ``tests/eval_queries.json``):

    relevant(chunk) := chunk.url == source_url
                       AND every regex in ``must_contain`` matches the body
                       AND (if ``must_contain_any`` is non-empty) at least one
                           regex in ``must_contain_any`` matches the body

``body`` is the chunk text with the leading title prefix removed, after
NFKC normalisation, diacritic stripping and case-folding.  Removing the title
matters because the ingest layer prepends the document title to every chunk;
without this step a title word would mark every chunk of a document as
relevant.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .retriever import Chunk


# ---------------------------------------------------------------------------
# Text normalisation and rule matching
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalise text for rule matching.

    NFKC first (fuses ligatures such as ``ﬁ`` -> ``fi``), then NFKD with
    combining marks removed (``Itô`` -> ``Ito``), then case-folding.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def chunk_body(text: str, title: str) -> str:
    """Strip the prepended document title from a chunk, if present."""
    if title and text.startswith(title):
        return text[len(title) :]
    return text


@dataclass(frozen=True)
class RelevanceRule:
    """A content-based relevance rule for one query."""

    source_url: str | None = None
    must_contain: tuple[str, ...] = ()
    must_contain_any: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source_url is None and (self.must_contain or self.must_contain_any):
            raise ValueError("A rule with content patterns must also name a source_url.")
        for pattern in (*self.must_contain, *self.must_contain_any):
            re.compile(normalize_text(pattern))  # fail fast on a bad regex

    def matches(self, chunk: Chunk) -> bool:
        if self.source_url is None:
            return False
        if chunk.url != self.source_url:
            return False
        return self._content_matches(chunk)

    def _content_matches(self, chunk: Chunk) -> bool:
        body = normalize_text(chunk_body(chunk.text, chunk.title))
        if any(re.search(normalize_text(p), body) is None for p in self.must_contain):
            return False
        return not (
            self.must_contain_any
            and not any(re.search(normalize_text(p), body) for p in self.must_contain_any)
        )

    def matches_content(self, chunk: Chunk) -> bool:
        """Content part of the rule only (used to report leakage across sources)."""
        if not self.must_contain and not self.must_contain_any:
            return False
        body = normalize_text(chunk_body(chunk.text, chunk.title))
        if any(re.search(normalize_text(p), body) is None for p in self.must_contain):
            return False
        return not (
            self.must_contain_any
            and not any(re.search(normalize_text(p), body) for p in self.must_contain_any)
        )


def rule_from_query(query: dict[str, Any]) -> RelevanceRule:
    return RelevanceRule(
        source_url=query.get("source_url"),
        must_contain=tuple(query.get("must_contain") or ()),
        must_contain_any=tuple(query.get("must_contain_any") or ()),
    )


# ---------------------------------------------------------------------------
# Ground-truth derivation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DerivedTruth:
    query_id: str
    ids: tuple[str, ...]
    leak_urls: tuple[tuple[str, int], ...]

    @property
    def count(self) -> int:
        return len(self.ids)


def derive_ground_truth(
    queries: Sequence[dict[str, Any]], chunks: Sequence[Chunk]
) -> dict[str, DerivedTruth]:
    """Derive relevant chunk ids for every query purely from the corpus.

    Raises ``ValueError`` when an answerable query matches no chunk: silently
    producing an empty ground truth is exactly the failure mode this replaces.
    """
    out: dict[str, DerivedTruth] = {}
    for query in queries:
        qid = str(query["id"])
        rule = rule_from_query(query)
        expected = query.get("expected", "chunks")
        ids = tuple(sorted(c.id for c in chunks if rule.matches(c)))

        leaks: dict[str, int] = {}
        if rule.source_url is not None:
            for chunk in chunks:
                if chunk.url != rule.source_url and rule.matches_content(chunk):
                    leaks[chunk.url] = leaks.get(chunk.url, 0) + 1

        if expected == "chunks" and not ids:
            raise ValueError(
                f"{qid}: rule matched no chunk in {rule.source_url!r}. "
                f"must_contain={rule.must_contain} must_contain_any={rule.must_contain_any}"
            )
        if expected == "none" and ids:
            raise ValueError(f"{qid}: marked expected=none but the rule matched {len(ids)} chunks.")

        out[qid] = DerivedTruth(
            query_id=qid,
            ids=ids,
            leak_urls=tuple(sorted(leaks.items())),
        )
    return out


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def hit_at_k(ranked_ids: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant id appears in the top ``k``."""
    rel = set(relevant)
    return 1.0 if any(doc_id in rel for doc_id in ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Recall@k with the standard ``min(k, |relevant|)`` denominator.

    This is the fraction of the *retrievable* relevant chunks that made it into
    the top ``k``.  It cannot exceed 1.0 and it is not diluted by relevant
    chunks that are impossible to retrieve because ``k`` is smaller than the
    number of relevant chunks.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in rel)
    return hits / min(k, len(rel))


def reciprocal_rank(ranked_ids: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant id (0.0 if none is retrieved)."""
    rel = set(relevant)
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in rel:
            return 1.0 / position
    return 0.0


# ---------------------------------------------------------------------------
# Random-ranker baselines (exact expectations, no sampling)
# ---------------------------------------------------------------------------
def random_hit_at_k(num_chunks: int, num_relevant: int, k: int) -> float:
    """Probability that a uniformly random top-k contains a relevant chunk."""
    if num_relevant <= 0 or k <= 0:
        return 0.0
    k = min(k, num_chunks)
    if num_chunks - num_relevant < k:
        return 1.0
    misses = math.comb(num_chunks - num_relevant, k) / math.comb(num_chunks, k)
    return 1.0 - misses


def random_recall_at_k(num_chunks: int, num_relevant: int, k: int) -> float:
    """Expected Recall@k of a uniformly random ranking."""
    if num_relevant <= 0 or k <= 0:
        return 0.0
    k = min(k, num_chunks)
    expected_hits = k * num_relevant / num_chunks
    return expected_hits / min(k, num_relevant)


def random_reciprocal_rank(num_chunks: int, num_relevant: int) -> float:
    """Expected reciprocal rank of a uniformly random ranking.

    P(first relevant at position i) =
        C(N-R, i-1) / C(N, i-1) * R / (N - i + 1)
    where the first factor is the probability of drawing i-1 non-relevant
    chunks in the first i-1 positions (without replacement).
    """
    if num_relevant <= 0:
        return 0.0
    if num_relevant >= num_chunks:
        return 1.0
    n = num_chunks
    r = num_relevant
    expected = 0.0
    for position in range(1, n + 1):
        if position - 1 > n - r:
            break
        no_relevant_before = math.comb(n - r, position - 1) / math.comb(n, position - 1)
        p_relevant_here = r / (n - position + 1)
        expected += no_relevant_before * p_relevant_here / position
    return expected
