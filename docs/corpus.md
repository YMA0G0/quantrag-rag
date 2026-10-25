# Corpus — `data/chunks.jsonl`

`data/chunks.jsonl` is **git-ignored on purpose** (third-party text, licences unchecked).
It is a build artefact, not source. Rebuild it deterministically with one command:

```bash
# install the optional heavy PDF/HTML extraction deps once
uv sync --extra ingest

# rebuild from the two CSVs + manual PDFs (respects robots.txt, caches under data/raw/)
uv run python -m quantrag.ingest \
  --csv data/quant_finance_kb_cleaned.csv data/quant_finance_50_new_sources.csv \
  --manual-csv data/manual_sources.csv --manual-dir data/manual \
  --profile core --out-chunks data/chunks.jsonl \
  --out-report-json data/ingest_report.json --out-report-csv data/ingest_report.csv
```

The same command with `--dry-run` previews source selection (writes
`data/ingest_report.dryrun.json/.csv`) and never clobbers `data/chunks.jsonl`.

## Current snapshot (committed pipeline, do not treat as authoritative)

| Field | Value |
|---|---|
| File | `data/chunks.jsonl` (git-ignored) |
| sha256 | `44af2e4e7e622cd33bed7a3bfad3e691c3a32e26d9843ac6d81935a4dd9b1104` |
| Chunks | 582 |
| Source documents | 18 |
| Pipeline commit | `6c6a679` (feat(ingest): multi-doc hops, kind filters, header-aware hard cap) |
| Format per line | `{"id","text","title","url","category"}` + optional `section`, `kind` (`prose`\|`formula_dump`\|`toc`\|`references`\|`boilerplate`) |

`chunk.text` is body only; the header `"{title} — {section}"` is added at index
time by `src/quantrag/retriever.py` (`Retriever.index`) and is counted toward
the 900-token hard cap. `chunk.url` is the citation URL. Loader accepts old
files without `section`/`kind` (legacy `kind` values normalize to `prose`).

## Known defects (to be fixed in Gate 1)

This snapshot reflects the half-finished ingestion and is **not** the target
state. Verified problems:

- **Off-list categories present** (the `core` profile still admits them via its
  tier clause): "Python Quantitative Finance Libraries" 156 (QuantLib docs),
  "Financial NLP and Sentiment Analysis" 14, "Reinforcement Learning for
  Portfolio Management" 9 = 179 chunks (31%) that are not in the core
  category allow-list.
- **Imbalance**: Monte Carlo arXiv 2208.05531 = 159 chunks and Bielefeld NumSDE
  = 74 (~40% of corpus). "Black–Scholes and Option Pricing" has only 10
  chunks, Stochastic Processes 7, SDEs 4, Fixed Income 9, Heston/SABR 24,
  Greeks 58, Stochastic Calculus 57.
- **Filters never fired** (toc/references/boilerplate/formula_dump counters 0)
  although the corpus has dot-leader chunks, MIT "Terms of Use"/"All rights
  reserved" chunks and reference-list chunks.
- 38 chunks contain a run of 26+ letters (URLs / formulas / squashed words).
- median chunk 832 tokens; 365 chunks exceed 800 (target 500–800, hard cap 900).

## Licence caveat

Corpus text comes from third-party documents (MIT OCW, Columbia, Bielefeld,
arXiv, QuantLib, etc.). Licences have **not** been cleared. `data/manual/`
PDFs are large and licence-unclear, so they are git-ignored except for
`data/manual/README.md`. Do not redistribute `data/chunks.jsonl` or the
underlying PDFs without reviewing each source's licence.

## Lineage

- `b0a5abd` baseline: MCP + retriever + ingest + eval
- `6c3f64d`, `d93c07d`: manual PDF sources (Columbia, NYU DSLecture6)
- `283d027`: retriever `section`/`kind` + index-time header + kind filter
- `764f937`: pin `mcp>=1.30`, declare pdfplumber/pdfminer.six
- `6c6a679`: ingest multi-doc hops, kind filters, header-aware hard cap
- `222daf4`: toc/references/boilerplate filter tests
