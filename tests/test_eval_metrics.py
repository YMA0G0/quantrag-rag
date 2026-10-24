"""Unit tests for retrieval metrics with hand‑computed expected values.

These tests validate the exact numeric behaviour of the ranking metrics and the
random‑ranker baselines without any reliance on the real corpus.  Hand‑computed
values avoid accidental drift and guarantee deterministic regression checks.
"""

from __future__ import annotations

from quantrag.evaluation import (
    chunk_body,
    derive_ground_truth,
    hit_at_k,
    normalize_text,
    random_hit_at_k,
    random_recall_at_k,
    random_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from quantrag.retriever import Chunk

# ---------------------------------------------------------------------------
# Metric tests with a tiny, controlled universe
# ---------------------------------------------------------------------------
# Four dummy chunks: ids A, B, C, D.
# The relevant set consists of A and C only.
# Ranked order for the test cases: B, A, D, C.
RANKED = ["B", "A", "D", "C"]
RELEVANT = {"A", "C"}


class TestHitAtK:
    """hit@k must be 1 iff any relevant appears in the top k."""

    def test_hit1_misses(self):
        assert hit_at_k(RANKED, RELEVANT, 1) == 0.0

    def test_hit2_hits(self):
        assert hit_at_k(RANKED, RELEVANT, 2) == 1.0

    def test_hit5_saturated(self):
        assert hit_at_k(RANKED, RELEVANT, 5) == 1.0


class TestRecallAtK:
    """Recall@k = #relevant in top‑k / min(k, |relevant|)."""

    def test_recall1(self):
        assert recall_at_k(RANKED, RELEVANT, 1) == 0.0

    def test_recall2(self):
        # Top‑2 contain only A → 1 / min(2,2) = 1/2 = 0.5
        assert recall_at_k(RANKED, RELEVANT, 2) == 0.5

    def test_recall3(self):
        # Top‑3 contain A only → 1 / min(3,2) = 1/2 = 0.5
        assert recall_at_k(RANKED, RELEVANT, 3) == 0.5

    def test_recall4(self):
        # Top‑4 contain A and C → 2 / min(4,2) = 2/2 = 1.0
        assert recall_at_k(RANKED, RELEVANT, 4) == 1.0


class TestReciprocalRank:
    """Mean Reciprocal Rank – 1 / rank of first relevant chunk."""

    def test_rr_first_at_position2(self):
        assert reciprocal_rank(RANKED, RELEVANT) == 0.5

    def test_rr_no_relevant(self):
        assert reciprocal_rank(RANKED, set()) == 0.0


class TestRandomRankerBaselines:
    """Exact expectations for a uniformly random ranking of N chunks."""

    N = 4  # total chunks
    R = 2  # relevant chunks

    def test_random_hit1(self):
        # Probability that a random top‑1 contains a relevant item = R / N
        assert random_hit_at_k(self.N, self.R, 1) == 0.5

    def test_random_hit2(self):
        # 1 - C(N‑R, k) / C(N, k) = 1 - C(2,2) / C(4,2) = 1 - 1/6 = 5/6
        assert abs(random_hit_at_k(self.N, self.R, 2) - 5 / 6) < 1e-9

    def test_random_hit3(self):
        # k > N‑R => all relevant must be in top‑k → probability = 1
        assert random_hit_at_k(self.N, self.R, 3) == 1.0

    def test_random_recall1(self):
        # (k * R) / N / min(k, R) = (1*2)/4 / 1 = 0.5
        assert random_recall_at_k(self.N, self.R, 1) == 0.5

    def test_random_recall2(self):
        # (2*2)/4 / 2 = 1/2 = 0.5
        assert random_recall_at_k(self.N, self.R, 2) == 0.5

    def test_random_recall3(self):
        # (3*2)/4 / 2 = 1.5/2 = 0.75
        assert random_recall_at_k(self.N, self.R, 3) == 0.75

    def test_random_mrr(self):
        # Hand‑computed: sum over i of P(first relevant at i) / i = 0.72222...
        expected = (
            0.5 * 1.0
            + ((2 / 4) * (2 / 3)) * (1 / 2)
            + ((2 / 4) * (1 / 3) * (2 / 2)) * (1 / 3)
            + 0.0  # position 4 impossible
        )
        assert abs(random_reciprocal_rank(self.N, self.R) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------
class TestContentHelpers:
    """Test normalisation and title‑stripping helpers."""

    def test_chunk_body_strips_title_when_present(self):
        c = Chunk(
            id="a",
            text="TITLE\n\nBody text here",
            title="TITLE",
            url="http://example.com",
            category="test",
        )
        # chunk_body removes the title prefix; the remaining "\n\n" separators
        # and body are preserved verbatim.
        assert chunk_body(c.text, c.title) == "\n\nBody text here"
        # normalize_text preserves leading whitespace; patterns use re.search.
        assert normalize_text(chunk_body(c.text, c.title)) == "\n\nbody text here"

    def test_chunk_body_no_strip_when_title_missing(self):
        c = Chunk(id="b", text="plain text", title="", url="", category="")
        assert chunk_body(c.text, c.title) == "plain text"

    def test_normalize_text_lowercases_and_strips_accents(self):
        # NFKC + NFKD + combining-mark removal + casefold:
        #   "Itô's Lemma" -> "ito's lemma"
        #   (ô decomposed, combining stripped; ASCII apostrophe kept)
        assert normalize_text("Itô's Lemma") == "ito's lemma"


# ---------------------------------------------------------------------------
# Relevance rule derivation (content‑based ground truth)
# ---------------------------------------------------------------------------
class TestRuleDeriveGroundTruth:
    """Validate derive_ground_truth with hand‑crafted corpus and queries."""

    def test_rule_matches_with_title_stripped(self):
        # One chunk contains the query concept; the title must be stripped.
        chunks = [
            Chunk(
                id="a",
                text="TITLE\n\nBlack Scholes formula",
                title="TITLE",
                url="http://source.com",
                category="",
            ),
            Chunk(
                id="b",
                text="TITLE\n\nother stuff",
                title="TITLE",
                url="http://source.com",
                category="",
            ),
        ]
        query = {
            "id": "q1",
            "query": "black scholes",
            "source_url": "http://source.com",
            "must_contain": ["black scholes"],
        }
        truth = derive_ground_truth([query], chunks)
        assert truth["q1"].ids == ("a",)

    def test_rule_no_match_when_unanswerable(self):
        # Query has expected="none" and rule matches nothing.
        chunks = [
            Chunk(
                id="a",
                text="TITLE\n\nBlack Scholes",
                title="TITLE",
                url="http://source.com",
                category="",
            )
        ]
        query = {
            "id": "q2",
            "query": "some obscure term",
            "source_url": "http://source.com",
            "must_contain": ["some obscure term"],
            "expected": "none",
        }
        truth = derive_ground_truth([query], chunks)
        assert truth["q2"].ids == ()
