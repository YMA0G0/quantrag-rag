"""Offline tests for prompt rendering, grounding detection and citations."""

from __future__ import annotations

import pytest

from quantrag.rag import (
    NOT_FOUND,
    build_prompt,
    build_rag_prompt,
    cited_numbers,
    is_not_found,
    source_header,
    strip_leading_title,
)
from quantrag.retriever import Chunk
from quantrag.service import RetrievalHit


def hit(
    *,
    chunk_id: str = "c1",
    text: str = "A parabolic partial differential equation for option prices.",
    title: str = "Black-Scholes",
    section: str = "PDE",
    url: str = "https://example.test/bs",
    kind: str = "prose",
    score: float = 0.9,
) -> RetrievalHit:
    chunk = Chunk(id=chunk_id, text=text, title=title, url=url, section=section, kind=kind)
    return RetrievalHit(chunk=chunk, score=score, source="hybrid")


def test_source_header_uses_title_and_section() -> None:
    assert source_header(hit().chunk) == "Black-Scholes \u2014 PDE"
    assert source_header(hit(section="").chunk) == "Black-Scholes"


def test_prompt_shows_header_once_and_keeps_body() -> None:
    prompt, sources = build_rag_prompt("What is the PDE?", [hit()])

    assert "[1] Black-Scholes \u2014 PDE" in prompt
    assert prompt.count("Black-Scholes") == 1
    assert "A parabolic partial differential equation for option prices." in prompt
    assert "<<<SOURCE 1>>>" in prompt and "<<<END SOURCE 1>>>" in prompt
    assert sources[0].url == "https://example.test/bs"
    assert sources[0].kind == "prose"
    assert sources[0].truncated is False
    assert sources[0].number == 1


def test_prompt_keeps_latex_verbatim() -> None:
    formula = (
        "$$ \\frac{\\partial V}{\\partial t} + \\frac{1}{2}\\sigma^2 S^2 "
        "\\frac{\\partial^2 V}{\\partial S^2} + rS\\frac{\\partial V}{\\partial S} "
        "- rV = 0 $$ and inline $\\alpha$ with $S_t$."
    )
    prompt, sources = build_rag_prompt("Give me the PDE.", [hit(text=formula)])

    assert formula in prompt
    assert sources[0].text == formula
    assert "\\$" not in prompt  # never escaped for the model


def test_prompt_drops_duplicated_leading_title() -> None:
    text = "Black-Scholes\n\nA parabolic PDE for option prices."
    prompt, sources = build_rag_prompt("What is it?", [hit(text=text)])

    assert prompt.count("Black-Scholes") == 1
    assert sources[0].text == "A parabolic PDE for option prices."


def test_strip_leading_title_is_conservative() -> None:
    assert strip_leading_title("Other heading\n\nbody", "Black-Scholes") == (
        "Other heading\n\nbody",
        False,
    )
    assert strip_leading_title("Black-Scholes", "Black-Scholes") == ("Black-Scholes", False)
    body, dropped = strip_leading_title("  black-scholes \nbody", "Black-Scholes")
    assert dropped is True and body == "body"


def test_prompt_skips_empty_bodies() -> None:
    prompt, sources = build_rag_prompt("Q?", [hit(text="   ")])

    assert sources == []
    assert "<<<SOURCE 1>>>" not in prompt
    assert prompt.startswith("Question:\nQ?")


def test_prompt_truncates_to_the_context_budget() -> None:
    long_text = "alpha beta " * 400
    prompt, sources = build_rag_prompt(
        "Q?",
        [hit(text=long_text), hit(chunk_id="c2", text=long_text)],
        max_context_chars=300,
    )

    assert len(sources) == 1
    assert sources[0].truncated is True
    assert "truncated to fit the context budget" in prompt
    assert len(prompt) < len(long_text)


def test_prompt_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError):
        build_rag_prompt("Q?", [hit()], max_context_chars=0)


def test_is_not_found_matches_the_canonical_sentence() -> None:
    assert is_not_found(NOT_FOUND)
    assert is_not_found(f"  {NOT_FOUND.lower()}  ")
    assert is_not_found(f'"{NOT_FOUND}."')
    assert not is_not_found("The answer is 42.")
    assert not is_not_found("")


def test_cited_numbers() -> None:
    assert cited_numbers("A [1], B [2][3].") == {1, 2, 3}
    assert cited_numbers("No citations here.") == set()


def test_citation_payload_shape() -> None:
    _prompt, sources = build_rag_prompt("Q?", [hit()])
    payload = sources[0].to_citation(cited=True)

    assert payload == {
        "n": 1,
        "chunk_id": "c1",
        "title": "Black-Scholes",
        "section": "PDE",
        "url": "https://example.test/bs",
        "kind": "prose",
        "score": 0.9,
        "source": "hybrid",
        "cited": True,
    }


def test_legacy_build_prompt_numbers_sources() -> None:
    prompt = build_prompt("Q?", ["first", "second"])

    assert "[1] first" in prompt
    assert "[2] second" in prompt
