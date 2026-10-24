"""Offline tests for chunking: bounds, oversize splitting, determinism."""

from quantrag.ingest import chunking


def test_chunk_size_bounds_and_overlap():
    text = "\n\n".join(["Paragraph " + str(i) + " " + ("word " * 80) for i in range(20)])
    drafts = chunking.chunk_text(text, title="T", url="http://x", target_tokens=200, max_tokens=400)
    for d in drafts:
        assert chunking.count_tokens(d.text) <= 400
    assert len(drafts) >= 2


def test_oversize_single_paragraph_is_split_below_hard_cap():
    # ~3000 tokens in one paragraph, no paragraph breaks
    text = " ".join(["alpha beta gamma delta epsilon zeta eta theta"] * 400)
    assert chunking.count_tokens(text) > 3000
    drafts = chunking.chunk_text(text, title="T", url="http://x", max_tokens=900)
    assert drafts
    for d in drafts:
        assert chunking.count_tokens(d.text) <= 900


def test_oversize_formula_block_is_split_below_hard_cap():
    formula_line = (
        "$dS_t = \\mu S_t dt + \\sigma S_t dW_t \\quad "
        "\\int_0^T f(t,\\omega) dt = \\sum_{i=1}^{n} \\alpha_i \\beta_i$"
    )
    block = "\n".join([formula_line] * 40)
    assert chunking.count_tokens(block) > 900
    drafts = chunking.chunk_text(block, title="T", url="http://x", max_tokens=900)
    assert drafts
    for d in drafts:
        assert chunking.count_tokens(d.text) <= 900


def test_formula_kept_intact_when_it_fits():
    formula = "$C = S N(d_1) - K e^{-rT} N(d_2)$"
    text = ("Intro text. " * 50) + "\n\n" + formula + "\n\n" + ("Outro text. " * 50)
    drafts = chunking.chunk_text(text, title="T", url="http://x", target_tokens=600, max_tokens=900)
    assert any(formula in d.text for d in drafts)


def test_deterministic_ids():
    assert chunking.make_chunk_id("http://example.com/a", 0) == chunking.make_chunk_id(
        "http://example.com/a", 0
    )
    cid = chunking.make_chunk_id("http://example.com/a", 0)
    assert len(cid.split("-")[0]) == 10


def test_dedup_within_document():
    text = ("Repeated sentence. " * 40) + "\n\n" + ("Repeated sentence. " * 40)
    drafts = chunking.chunk_text(text, title="T", url="http://x", max_tokens=900)
    norms = [" ".join(d.text.split()) for d in drafts]
    assert len(norms) == len(set(norms))
