#!/usr/bin/env python3
"""Derive retrieval ground truth from content rules (never from retriever output).

Reads the rule file (``tests/eval_queries.json``) and the corpus, then writes
``tests/eval_dataset.json`` with the *derived* relevant chunk ids, the corpus
hash, and the leakage report.

Usage:
    python scripts/build_ground_truth.py \
        --queries tests/eval_queries.json \
        --corpus data/chunks.jsonl \
        --out tests/eval_dataset.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantrag.evaluation import derive_ground_truth, rule_from_query  # noqa: E402
from quantrag.retriever import load_chunks_jsonl  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def short_url(url: str | None) -> str:
    if not url:
        return "-"
    return url.rstrip("/").rsplit("/", 1)[-1] or url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="tests/eval_queries.json")
    parser.add_argument("--corpus", default="data/chunks.jsonl")
    parser.add_argument("--out", default="tests/eval_dataset.json")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    corpus_path = Path(args.corpus)
    out_path = Path(args.out)

    rules_doc = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = rules_doc["queries"]
    chunks = load_chunks_jsonl(corpus_path)

    print(f"corpus: {corpus_path}  chunks={len(chunks)}  sha256={sha256_file(corpus_path)}")
    print(f"rules:  {queries_path}  queries={len(queries)}")
    print()

    try:
        derived = derive_ground_truth(queries, chunks)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    header = f"{'id':5} {'exp':9} {'R':>4}  {'source':32} rule"
    print(header)
    print("-" * len(header))
    enriched = []
    for query in queries:
        qid = str(query["id"])
        truth = derived[qid]
        rule = rule_from_query(query)
        all_of = " AND ".join(rule.must_contain) or "-"
        any_of = " OR ".join(rule.must_contain_any)
        rule_text = all_of + (f"  [any: {any_of}]" if any_of else "")
        print(
            f"{qid:5} {query.get('expected', 'chunks'):9} {truth.count:4d}  "
            f"{short_url(rule.source_url)[:32]:32} {rule_text}"
        )
        if truth.leak_urls:
            leaks = ", ".join(f"{short_url(u)}={n}" for u, n in truth.leak_urls)
            print(f"      note: same content also appears outside source_url: {leaks}")

        item = dict(query)
        item["ground_truth"] = list(truth.ids)
        item["num_relevant"] = truth.count
        item["leak_outside_source"] = {u: n for u, n in truth.leak_urls}
        enriched.append(item)

    positive = [q for q in enriched if q.get("expected", "chunks") == "chunks"]
    negatives = [q for q in enriched if q.get("expected") == "none"]
    print()
    print(
        f"summary: {len(positive)} answerable queries "
        f"({sum(q['num_relevant'] for q in positive)} relevant chunk links), "
        f"{len(negatives)} expected=none queries"
    )
    if positive:
        counts = sorted(q["num_relevant"] for q in positive)
        print(
            f"relevant-per-query: min={counts[0]} median={counts[len(counts) // 2]} "
            f"max={counts[-1]}"
        )
        max_recall10 = sum(min(10, q["num_relevant"]) for q in positive) / (len(positive) * 10)
        print(
            "max attainable R@10 under this ground truth (all relevant retrieved): "
            f"{max_recall10:.3f}"
        )

    dataset = {
        "metadata": {
            "corpus": str(corpus_path),
            "corpus_hash": sha256_file(corpus_path),
            "corpus_size": len(chunks),
            "num_queries": len(enriched),
            "num_answerable": len(positive),
            "num_expected_none": len(negatives),
            "rules_file": str(queries_path),
            "matching": rules_doc.get("metadata", {}).get("matching", ""),
            "derived_by": "scripts/build_ground_truth.py",
            "description": (
                "Ground truth derived from content rules: chunk.url == source_url AND "
                "must_contain (all) AND must_contain_any (>=1), matched on the chunk body "
                "with the prepended title removed, NFKC + diacritics stripped + casefolded."
            ),
        },
        "queries": enriched,
    }
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
