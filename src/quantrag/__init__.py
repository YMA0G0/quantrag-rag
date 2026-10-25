"""QuantRAG -- Quantitative-finance retrieval and Q&A.

Phase 1: Ingestion pipeline (chunks.jsonl)
Phase 2: Retrieval layer (BM25 + embeddings, hybrid search, formula-aware)
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Chunk",
    "RetrievedChunk",
    "Filter",
    "InMemoryVectorStore",
    "BM25Index",
    "BM25Result",
    "HybridRetriever",
    "load_chunks_jsonl",
    "RetrievalError",
    "RagAnswer",
    "RagAnswerer",
    "build_prompt",
    "RAG_SYSTEM_PROMPT",
]

# Re-export core types
from .errors import RetrievalError
from .hybrid_retriever import BM25Index, BM25Result, Filter, HybridRetriever
from .rag import RAG_SYSTEM_PROMPT, RagAnswer, RagAnswerer, build_prompt
from .retriever import Chunk, InMemoryVectorStore, RetrievedChunk, load_chunks_jsonl
