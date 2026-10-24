# Retriever module — hand-off note (role 2: Retrieval Engineer)

## What's included
| File | Purpose |
|---|---|
| `src/quantrag/embeddings.py` | OpenRouter `/embeddings` client (text -> vectors) |
| `src/quantrag/retriever.py` | `Chunk`, `Retriever`, `InMemoryVectorStore`, `load_chunks_jsonl` |
| `src/quantrag/errors.py` | Same as the current file + one new class `RetrievalError` |
| `tests/test_embeddings.py`, `tests/test_retriever.py` | Offline tests (mocked HTTP, fake embedder) |

Full suite with the existing tests: 96 passed, 1 skipped.

## Config
Add to `.env` / `.env.example`:
`OPENROUTER_EMBEDDING_MODEL=<an embedding model id available on our OpenRouter account>`

## For the Data Engineer
Please produce `chunks.jsonl`, one JSON object per line:
`{"id": "...", "text": "...", "title": "...", "url": "...", "category": "..."}`
(`category` optional). `title` and `url` are used for citations.

## For the RAG Engineer
```python
from quantrag.config import load_settings
from quantrag.embeddings import OpenRouterEmbedder, load_embedding_model
from quantrag.retriever import Retriever, load_chunks_jsonl
import os

settings = load_settings()
embedder = OpenRouterEmbedder(settings, load_embedding_model(os.environ))
retriever = Retriever(embedder)
await retriever.index(load_chunks_jsonl("chunks.jsonl"))

results = await retriever.retrieve("What is Black-Scholes?", k=5)
for r in results:
    print(r.score, r.chunk.title, r.chunk.url, r.chunk.text)
```
- Results are sorted best-first. Optional `min_score=` drops weak matches (so the RAG layer can say "I don't know"). It is off by default: a good value depends on the embedding model and the real chunks, so it should be tuned once real data exists (look at the scores of good vs. irrelevant results and pick a cut-off). `k` defaults to 5.
- `retriever.store.save("index.json")` / `InMemoryVectorStore.load("index.json")` avoid re-embedding on every start.

## Note
The project needs `mcp<2` pinned in `pyproject.toml` (v2 renamed `FastMCP`).
