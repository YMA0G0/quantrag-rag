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


# ---------------------------------------------------------------------------
# Atomic display math, formula_block, and exercise classification
# ---------------------------------------------------------------------------
def test_display_math_is_not_split_between_chunks():
    formula = "$$\n\\sum_{i=1}^{100} x_i + \\frac{\\mu}{\\sigma} = \\int_0^T f(t) dt\n$$"
    prose = " ".join(["Long prose sentence about the Euler scheme."] * 80)
    text = prose + "\n\n" + formula + "\n\n" + prose
    drafts = chunking.chunk_text(text, title="T", url="http://x", target_tokens=120, max_tokens=900)
    joined = "\n".join(d.text for d in drafts)
    assert formula in joined
    # Formula stays INSIDE prose chunk (atomic, not split)
    assert all(d.kind == "prose" for d in drafts)
    assert any(formula in d.text for d in drafts)


def test_formula_block_keeps_balanced_display_delimiters_when_oversized():
    rows = ["\\sum_{i=1}^{100} x_i + \\frac{\\mu}{\\sigma} \\\\" for _ in range(80)]
    formula = "$$\n\\begin{eqnarray}\n" + "\n".join(rows) + "\n\\end{eqnarray}\n$$"
    assert chunking.count_tokens(formula) > 900
    drafts = chunking.chunk_text(formula, title="T", url="http://x", max_tokens=900)
    assert len(drafts) >= 2
    for draft in drafts:
        assert draft.kind == "formula_block"
        assert chunking.count_tokens(draft.text) <= 900
        assert draft.text.strip().startswith("$$")
        assert draft.text.strip().endswith("$$")
        assert "\\begin{eqnarray}" in draft.text
        assert "\\end{eqnarray}" in draft.text


def test_math_fence_stays_in_prose():
    formula = "```math\n" + "\\sum_{i=1}^{100} x_i = \\frac{\\mu}{\\sigma}\n" + "```"
    # With realistic target_tokens, the formula stays inside the prose chunk
    drafts = chunking.chunk_text(
        "Intro.\n\n" + formula + "\n\nOutro.", title="T", url="http://x", target_tokens=600
    )
    # Formula stays INSIDE prose chunk (atomic, not split)
    assert len(drafts) == 1
    assert drafts[0].kind == "prose"
    assert formula in drafts[0].text


def test_exercise_section_gets_exercises_kind():
    text = (
        "## Exercises\n\n"
        "Exercise 1. Show that the Euler scheme converges for "
        + ("a Lipschitz drift and diffusion coefficient. " * 8)
        + "\n\n## Theory\n\n"
        + ("The Euler scheme is a stochastic Taylor method. " * 8)
    )
    drafts = chunking.chunk_text(text, title="T", url="http://x", target_tokens=100)
    assert any(d.kind == "exercises" for d in drafts)
    assert any(d.kind == "prose" for d in drafts)
    assert any(d.section == "Exercises" for d in drafts if d.kind == "exercises")


def test_token_count_includes_latex_markup():
    plain = "frac mu sigma"
    latex = r"$\frac{\mu}{\sigma}$"
    assert chunking.count_tokens(latex) > chunking.count_tokens(plain)
