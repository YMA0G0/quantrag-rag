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
| sha256 | `d4330b5e39f1ed0ee36a71ac0deaabfcdf006a955959ebde923bf0317070f0e6` |
| Chunks | 559 |
| Source documents | 16 |
| Pipeline commit | `733368a` (feat(ingest): make per-document cap opt-in, default off) |
| Format per line | `{"id","text","title","url","category"}` + optional `section`, `kind` (`prose`\|`formula_dump`\|`toc`\|`references`\|`boilerplate`) |

`chunk.text` is body only; the header `"{title} — {section}"` is added at index
time by `src/quantrag/retriever.py` (`Retriever.index`) and is counted toward
the 900-token hard cap. `chunk.url` is the citation URL. Loader accepts old
files without `section`/`kind` (legacy `kind` values normalize to `prose`).

## Known defects (to be fixed in Gate 1)

This snapshot reflects the half-finished ingestion and is **not** the target
state. Verified problems:

- **Off-list categories removed**: QuantLib (156), Financial NLP (14), RL for
  Portfolio (9) were excluded by the strict category allow-list. "University
  Courses and Lecture Notes" added to allow-list to admit tier-1 lecture notes
  (Kohn, Goodman, MIT 15.450 index).
- **Imbalance**: Monte Carlo arXiv 2208.05531 = 220 chunks (39%) and Bielefeld
  NumSDE = 102 (18%). Per-document cap (25%) available via `--per-doc-cap 0.25`
  (opt-in, default off). "Black–Scholes and Option Pricing" has 9 chunks,
  Stochastic Processes 9, SDEs 5, Heston/SABR 17, Greeks 55, Stochastic
  Calculus 70.
- **Filters now fire**: toc/references/boilerplate detectors catch real snippets.
  TOC (Bielefeld), boilerplate (MIT Terms of Use), references (FinRL-X) all
  detected.
- **26+ letter runs**: 1 remaining ("strongoptimalapproximation" in Monte Carlo).
- **Chunk size**: median 572 tokens, max 663, 0 over 800, 94.1% in 400-800 range.
- **Short lines <=2 chars**: all documents under 5% (max 1.5%).
- Columbia IEOR 6712 notes ingested manually (6 chunks); NYU Avellaneda
  DSLecture6.pdf is image-locked (~194 chars/page) and correctly marked
  needs_ocr and excluded.
- Tier-1 landing pages (NYU Kohn, Goodman) blocked by robots.txt; MIT 15.450
  lecture-notes index accessible via second hop (58 chunks).
- CSU Stanislaus "High Finance" page: no PDF links found, not ingested.

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
- `57bbaca`: strict category allow-list; tier never admits off-list
- `3c785d4`: chunk filters now catch TOC/boilerplate/references
- `7154c88`: CamelCase split + rule-based vocabulary-gated join
- `3ae5a63`: ruff format chunking.py
- `733368a`: per-document cap opt-in (default off); extractor selection in report
- `be3ba1d`: landing pages fetched for PDF hops; second hop for MIT 15.450
