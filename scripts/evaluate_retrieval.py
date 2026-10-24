#!/usr/bin/env python3
"""Retrieval evaluation with content-derived ground truth (Phase 2).

Metrics
-------
* ``hit@k``     - fraction of queries with at least one relevant chunk in the top k.
* ``Recall@k``  - |relevant ∩ top-k| / min(k, |relevant|).  The ``min`` denominator
                  is what makes Recall@10 attainable when a query has 16
                  relevant chunks and only 10 slots.
* ``MRR``       - mean of 1/rank of the first relevant chunk.
* ``random@k``  - exact expectation for a uniformly random ranking of the same
                  corpus, shown next to every metric so numbers are readable.

Modes
-----
``bm25``, ``vector`` and ``hybrid`` (alpha 0.3/0.5/0.7) are all really executed.
An embedding backend must be named explicitly with ``--embedder``; there is no
implicit fallback.  ``--embedder openrouter`` requires OPENROUTER_API_KEY and
OPENROUTER_EMBEDDING_MODEL and exits non-zero if either is missing.

Usage
-----
    python scripts/evaluate_retrieval.py --embedder tfidf
    python scripts/evaluate_retrieval.py --embedder openrouter
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantrag.bm25 import create_bm25_index  # noqa: E402
from quantrag.config import Settings  # noqa: E402
from quantrag.embeddings import (  # noqa: E402
    OpenRouterEmbedder,
    load_embedding_model,
)
from quantrag.errors import ConfigurationError  # noqa: E402
from quantrag.evaluation import (  # noqa: E402
    derive_ground_truth,
    hit_at_k,
    random_hit_at_k,
    random_recall_at_k,
    random_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    rule_from_query,
)
from quantrag.local_embeddings import TfidfEmbedder  # noqa: E402
from quantrag.retriever import Chunk, load_chunks_jsonl  # noqa: E402

K_VALUES = (1, 3, 5, 10)
HYBRID_ALPHAS = (0.3, 0.5, 0.7)
CONFIGS = [("bm25", None)] + [("vector", None)] + [("hybrid", alpha) for alpha in HYBRID_ALPHAS]


def build_embedder(kind: str, texts: list[str]) -> Any:
    """Return (embedder, model_name, is_neural) or raise ConfigurationError."""
    if kind == "tfidf":
        embedder = TfidfEmbedder().fit(texts)
        return embedder, embedder.model_name, False
    if kind == "openrouter":
        api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            raise ConfigurationError(
                "--embedder openrouter requires OPENROUTER_API_KEY. "
                "Refusing to silently substitute a local embedder."
            )
        model = load_embedding_model(os.environ)
        base_url = (os.environ.get("OPENROUTER_BASE_URL") or "").strip()
        settings = Settings(
            api_key=api_key,
            model=os.environ.get("OPENROUTER_MODEL", ""),
            base_url=base_url or "https://openrouter.ai/api/v1",
        )
        return OpenRouterEmbedder(settings, model), model, True
    raise ConfigurationError(f"Unknown embedder {kind!r}")


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def rank_all(scores: list[float], chunk_ids: list[str]) -> list[str]:
    """Deterministic ranking: score desc, then chunk id asc."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], chunk_ids[i]))
    return [chunk_ids[i] for i in order]


def cosine(query: list[float], vectors: list[list[float]]) -> list[float]:
    out: list[float] = []
    for vector in vectors:
        out.append(sum(a * b for a, b in zip(query, vector, strict=True)))
    return out


def score_configs(
    queries: list[dict[str, Any]],
    chunks: list[Chunk],
    emb: Any,
    is_neural: bool,
    k_values: tuple[int, ...],
) -> tuple[dict[tuple[str, float | None], dict[str, dict[str, Any]]], list[list[float]]]:
    """Return per-config per-query metrics plus the corpus vectors."""
    chunk_ids = [chunk.id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]

    import asyncio

    corpus_vectors = asyncio.run(emb.embed(texts))
    bm25 = create_bm25_index(chunks)

    results: dict[tuple[str, float | None], dict[str, dict[str, Any]]] = {}
    for config in CONFIGS:
        results[config] = {}

    for query in queries:
        qid = str(query["id"])
        text = str(query["query"])
        relevant = set(query.get("ground_truth") or [])
        expected = query.get("expected", "chunks")

        bm25_hits = bm25.search(text, k=len(chunks))
        bm25_map = {hit.chunk.id: hit.score for hit in bm25_hits}
        bm25_scores = [bm25_map.get(cid, 0.0) for cid in chunk_ids]

        [query_vector] = asyncio.run(emb.embed([text]))
        vector_scores = cosine(query_vector, corpus_vectors)

        config_scores: dict[tuple[str, float | None], list[float]] = {
            ("bm25", None): bm25_scores,
            ("vector", None): vector_scores,
        }
        bm25_norm = minmax(bm25_scores)
        vector_norm = minmax(vector_scores)
        for alpha in HYBRID_ALPHAS:
            config_scores[("hybrid", alpha)] = [
                alpha * v + (1.0 - alpha) * b for v, b in zip(vector_norm, bm25_norm, strict=True)
            ]

        for config, scores in config_scores.items():
            ranked = rank_all(scores, chunk_ids)
            entry: dict[str, Any] = {
                "expected": expected,
                "num_relevant": len(relevant),
                "top1_score": scores[chunk_ids.index(ranked[0])] if ranked else 0.0,
                "top1_id": ranked[0] if ranked else None,
                "first_relevant_rank": None,
                "hit": {},
                "recall": {},
                "mrr": 0.0,
            }
            if relevant:
                for k in k_values:
                    entry["hit"][k] = hit_at_k(ranked, relevant, k)
                    entry["recall"][k] = recall_at_k(ranked, relevant, k)
                entry["mrr"] = reciprocal_rank(ranked, relevant)
                for position, doc_id in enumerate(ranked, start=1):
                    if doc_id in relevant:
                        entry["first_relevant_rank"] = position
                        break
            results[config][qid] = entry

    return results, corpus_vectors


def fmt_metrics(
    entries: list[dict[str, Any]],
    k_values: tuple[int, ...],
    num_chunks: int,
) -> dict[str, float]:
    n = len(entries)
    out: dict[str, float] = {}
    for k in k_values:
        out[f"hit@{k}"] = sum(e["hit"][k] for e in entries) / n
        out[f"recall@{k}"] = sum(e["recall"][k] for e in entries) / n
        out[f"random_hit@{k}"] = (
            sum(random_hit_at_k(num_chunks, e["num_relevant"], k) for e in entries) / n
        )
        out[f"random_recall@{k}"] = (
            sum(random_recall_at_k(num_chunks, e["num_relevant"], k) for e in entries) / n
        )
    out["mrr"] = sum(e["mrr"] for e in entries) / n
    out["random_mrr"] = (
        sum(random_reciprocal_rank(num_chunks, e["num_relevant"]) for e in entries) / n
    )
    return out


def describe(scores: list[float]) -> str:
    if not scores:
        return "n=0"
    ordered = sorted(scores)
    median = statistics.median(ordered)
    return f"n={len(ordered)} min={ordered[0]:.4f} median={median:.4f} max={ordered[-1]:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="tests/eval_dataset.json")
    parser.add_argument("--corpus", default="data/chunks.jsonl")
    parser.add_argument("--embedder", choices=["tfidf", "openrouter"], required=True)
    parser.add_argument("--k", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--output", help="write detailed JSON results here")
    args = parser.parse_args()

    k_values = tuple(sorted(set(args.k)))
    dataset_path = Path(args.dataset)
    corpus_path = Path(args.corpus)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    queries = dataset["queries"]
    chunks = sorted(load_chunks_jsonl(corpus_path), key=lambda c: c.id)

    # --- ground-truth consistency: recompute from rules, must match the file ---
    recomputed = derive_ground_truth(queries, chunks)
    mismatches = []
    for query in queries:
        qid = str(query["id"])
        recorded = tuple(sorted(query.get("ground_truth") or []))
        if recorded != recomputed[qid].ids:
            mismatches.append(qid)
    if mismatches:
        print(
            "ERROR: ground truth in the dataset does not match the rules for: "
            + ", ".join(mismatches)
            + "\nRe-run scripts/build_ground_truth.py.",
            file=sys.stderr,
        )
        return 2

    print("=" * 100)
    print("CORPUS AND EMBEDDING BACKEND")
    print("=" * 100)
    print(f"corpus      : {corpus_path}")
    print(f"chunks      : {len(chunks)}")
    print(f"dataset     : {dataset_path}")
    print(f"queries     : {len(queries)}")
    print(f"k values    : {list(k_values)}")

    try:
        embedder, model_name, is_neural = build_embedder(args.embedder, [c.text for c in chunks])
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"embedder    : {model_name}")
    print(f"neural      : {is_neural}")
    if not is_neural:
        print("WARNING     : local TF-IDF vectors, not neural embeddings.")
    print()

    # --- rules per query ---
    print("=" * 100)
    print("RELEVANCE RULE PER QUERY (chunk.url == source AND body matches rule; title stripped)")
    print("=" * 100)
    for query in queries:
        rule = rule_from_query(query)
        parts = []
        if rule.must_contain:
            parts.append("ALL: " + " & ".join(rule.must_contain))
        if rule.must_contain_any:
            parts.append("ANY: " + " | ".join(rule.must_contain_any))
        rule_text = "  ".join(parts) if parts else "(expected: none)"
        print(
            f"{query['id']:5} R={query.get('num_relevant', 0):<3} "
            f"{str(query.get('expected', 'chunks')):6} {query['query']!r}"
        )
        print(f"      source: {rule.source_url or '-'}")
        print(f"      rule  : {rule_text}")
    print()

    results, _ = score_configs(queries, chunks, embedder, is_neural, k_values)

    positives = [q for q in queries if q.get("expected", "chunks") == "chunks"]
    negatives = [q for q in queries if q.get("expected") == "none"]

    # --- metrics table ---
    print("=" * 100)
    print("METRICS OVER ANSWERABLE QUERIES (+ exact random-ranker baseline)")
    print("=" * 100)
    header = f"{'mode':16} " + " ".join(f"{'hit@' + str(k):>8}" for k in k_values)
    header += " " + " ".join(f"{'R@' + str(k):>8}" for k in k_values) + f" {'MRR':>8}"
    print(header)
    print("-" * len(header))

    table: dict[str, dict[str, float]] = {}
    for config in CONFIGS:
        name = config[0] if config[1] is None else f"hybrid(a={config[1]:.1f})"
        entries = [results[config][str(q["id"])] for q in positives]
        metrics = fmt_metrics(entries, k_values, len(chunks))
        table[name] = metrics
        row = f"{name:16} " + " ".join(f"{metrics['hit@' + str(k)]:8.3f}" for k in k_values)
        row += " " + " ".join(f"{metrics['recall@' + str(k)]:8.3f}" for k in k_values)
        row += f" {metrics['mrr']:8.3f}"
        print(row)

    entries = [results[("bm25", None)][str(q["id"])] for q in positives]
    random_metrics = fmt_metrics(entries, k_values, len(chunks))
    row = f"{'random':16} " + " ".join(
        f"{random_metrics['random_hit@' + str(k)]:8.3f}" for k in k_values
    )
    row += " " + " ".join(f"{random_metrics['random_recall@' + str(k)]:8.3f}" for k in k_values)
    row += f" {random_metrics['random_mrr']:8.3f}"
    print(row)
    print()

    # --- per-query detail ---
    print("=" * 100)
    print("PER-QUERY DETAIL (rank of first relevant chunk / top-1 score)")
    print("=" * 100)
    for query in positives:
        qid = str(query["id"])
        row = f"{qid:5} R={query['num_relevant']:<3}"
        for config in CONFIGS:
            entry = results[config][qid]
            rank = entry["first_relevant_rank"] or "-"
            alpha = "" if config[1] is None else str(config[1])
            row += f" | {config[0][:4]}{alpha:5} rank={str(rank):>4} s={entry['top1_score']:.3f}"
        print(row)
    print()

    # --- negatives / threshold analysis ---
    print("=" * 100)
    print("TOP-1 SCORE DISTRIBUTION: NEGATIVES vs POSITIVES (for choosing min_score)")
    print("=" * 100)
    for config in CONFIGS:
        name = config[0] if config[1] is None else f"hybrid(a={config[1]:.1f})"
        neg_scores = [results[config][str(q["id"])]["top1_score"] for q in negatives]
        pos_scores = [results[config][str(q["id"])]["top1_score"] for q in positives]
        print(f"{name}")
        print(f"  negatives: {describe(neg_scores)}")
        print(f"  positives: {describe(pos_scores)}")
        threshold = max(neg_scores) if neg_scores else float("nan")
        pos_min = min(pos_scores) if pos_scores else float("nan")
        if threshold < pos_min:
            print(
                f"  separable : yes, threshold in ({threshold:.4f}, {pos_min:.4f}]; "
                f"suggested min_score={threshold + (pos_min - threshold) / 2:.4f}"
            )
        else:
            print(f"  separable : NO (max negative {threshold:.4f} >= min positive {pos_min:.4f})")
        print()

    print("=" * 100)
    print("NEGATIVE QUERY TOP-1 SCORES")
    print("=" * 100)
    for query in negatives:
        qid = str(query["id"])
        row = f"{qid:5} {query['query']!r:45}"
        for config in CONFIGS:
            entry = results[config][qid]
            row += f" {entry['top1_score']:8.3f}"
        print(row)
    print()

    if args.output:
        payload = {
            "metadata": dataset.get("metadata", {}),
            "embedder": model_name,
            "neural": is_neural,
            "k_values": list(k_values),
            "modes": [
                {
                    "mode": c[0],
                    "alpha": c[1],
                    "metrics": table.get(c[0] if c[1] is None else f"hybrid(a={c[1]:.1f})"),
                }
                for c in CONFIGS
            ],
            "random_baseline": random_metrics,
            "per_query": {
                (c[0] if c[1] is None else f"hybrid(a={c[1]:.1f})"): results[c] for c in CONFIGS
            },
        }
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
