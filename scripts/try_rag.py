"""Quick smoke for the RAG answerer (branch B contribution, moved to scripts/).

Demonstrates :class:`quantrag.rag.RagAnswerer` with a stub chat backend so it
runs fully offline and needs no OpenRouter credentials.  Intended for local
developer sanity checks only.
"""

from __future__ import annotations

import asyncio

from quantrag.rag import RagAnswer, RagAnswerer, build_prompt


class StubChat:
    """Stub chat completer: echoes the question and the number of sources."""

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        count = prompt.count("[")
        return f"Stub answer for {count} cited sources (system={bool(system)})."


async def main() -> None:
    answerer = RagAnswerer(StubChat())
    sources = [
        "Brownian motion has independent increments.",
        "Ito's lemma gives d f(X_t) = f'(X_t) dX_t + 1/2 f''(X_t) (dX_t)^2.",
    ]
    answer: RagAnswer = await answerer.answer("What is Ito's lemma?", sources)
    print(answer.answer)
    print("sources:", answer.sources)
    print("---- prompt ----")
    print(build_prompt("What is Ito's lemma?", sources))


if __name__ == "__main__":
    asyncio.run(main())
