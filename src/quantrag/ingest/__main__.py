"""CLI entry point: ``python -m quantrag.ingest``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chunking import MAX_TOKENS_HARD, TARGET_TOKENS
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantrag.ingest")
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[
            "data/quant_finance_kb_cleaned.csv",
            "data/quant_finance_50_new_sources.csv",
        ],
    )
    parser.add_argument("--out-chunks", default="data/chunks.jsonl")
    parser.add_argument("--out-report-json", default="data/ingest_report.json")
    parser.add_argument("--out-report-csv", default="data/ingest_report.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--manual-csv", default="data/manual_sources.csv")
    parser.add_argument("--manual-dir", default="data/manual")
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument(
        "--profile",
        choices=["core", "all"],
        default="core",
        help="'core' (default) keeps tier<=2 or core finance categories; 'all' keeps everything.",
    )
    parser.add_argument("--max-tier", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-category", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_HARD)
    parser.add_argument(
        "--per-doc-cap",
        type=float,
        default=None,
        help=(
            "If set, no single source may exceed this fraction of the corpus "
            "(e.g. 0.25). Off by default."
        ),
    )
    args = parser.parse_args(argv)

    root = Path.cwd()
    csv_paths = [str((root / p).resolve()) if not Path(p).is_absolute() else p for p in args.csv]
    csv_paths = [p for p in csv_paths if Path(p).exists()]
    if not csv_paths:
        print("No CSV inputs found.", file=sys.stderr)
        return 2

    manual_csv = args.manual_csv
    if manual_csv and not Path(manual_csv).exists():
        manual_csv = None

    report_json = Path(args.out_report_json)
    report_csv = Path(args.out_report_csv)
    if args.dry_run:
        # never clobber the artifacts produced by a real run
        report_json = report_json.with_name(f"{report_json.stem}.dryrun{report_json.suffix}")
        report_csv = report_csv.with_name(f"{report_csv.stem}.dryrun{report_csv.suffix}")

    totals = run_pipeline(
        csv_paths,
        out_chunks=Path(args.out_chunks),
        out_report_json=report_json,
        out_report_csv=report_csv,
        raw_dir=Path(args.raw_dir),
        include_unreviewed=args.include_unreviewed,
        max_tier=args.max_tier,
        only_category=args.only_category,
        limit=args.limit,
        dry_run=args.dry_run,
        profile=args.profile,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        manual_csv=manual_csv,
        manual_dir=args.manual_dir,
        per_doc_cap=args.per_doc_cap,
    )
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
