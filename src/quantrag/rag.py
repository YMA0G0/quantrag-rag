"""RAG answerer over retrieved chunks (branch B contribution).

This is the preliminary RAG layer contributed by the RAG engineer.  It builds
a numbered prompt from a question plus retrieved chunks and asks the
configured chat model to answer.  It knows nothing about retrieval itself --
the caller supplies the chunks -- which keeps it independent of the vector /
BM25 layer.

Phase 1 replaces this module's role with :class:`quantrag.service.RagService`,
which adds citation mapping, prompt-injection delimiting, context capping and
a not-found short-circuit.  Until then this is the RAG engineer's baseline.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

RAG_SYSTEM_PROMPT = (
    "You are QuantRAG, a quantitative-finance assistant. "
    "Answer using only the provided numbered sources. Cite every claim with "
    "[n]. If the sources do not contain the answer, say exactly: "
    "I could not find the answer in the provided sources."
)

# Sentence the model (and the service) must emit when there is nothing to
# ground the answer in.  Kept here so rag.py and service.py can share it.
NOT_FOUND = "I could not find the answer in the provided sources."


class ChatCompleter(Protocol):
    """Minimal chat interface: complete a prompt, optionally with a system prompt."""

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the assistant's text answer for ``prompt``."""
        ...


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """The model's answer plus the sources it was allowed to cite."""

    answer: str
    sources: list[str]


def build_prompt(question: str, sources: Iterable[str]) -> str:
    """Number the sources and embed them in the user prompt.

    Each source block is prefixed with ``[n]`` so the model can cite it.  The
    caller is responsible for truncation; this helper does not cap context.
    """
    lines: list[str] = [
        f"Question: {question}",
        "",
        "Sources:",
    ]
    for index, text in enumerate(sources, start=1):
        lines.append(f"[{index}] {text}")
    lines.extend(["", "Answer using only the sources above, citing [n]."])
    return "\n".join(lines)


class RagAnswerer:
    """Turns a question + retrieved chunks into a cited answer via the chat model."""

    def __init__(self, chat: ChatCompleter) -> None:
        self._chat = chat

    async def answer(self, question: str, sources: list[str]) -> RagAnswer:
        """Answer ``question`` using ``sources``; returns the answer and the sources."""
        prompt = build_prompt(question, sources)
        answer = await self._chat.complete(prompt, system=RAG_SYSTEM_PROMPT)
        return RagAnswer(answer=answer, sources=list(sources))
