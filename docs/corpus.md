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
| sha256 | `71ff2361052224c244141d75be55fa7c4c098b23d1767d870fb819c9bd656dbd` |
| Chunks | 501 |
| Source documents | 14 |
| Pipeline commit | `822771c` (fix(ingest): extractor selection, single-symbol lines, squashed prose, join audit fix, landing page second hop) |
| Format per line | `{"id","text","title","url","category"}` + optional `section`, `kind` (`prose`\|`formula_dump`\|`toc`\|`references`\|`boilerplate`) |

`chunk.text` is body only; the header `"{title} — {section}"` is added at index
time by `src/quantrag/retriever.py` (`Retriever.index`) and is counted toward
the 900-token hard cap. `chunk.url` is the citation URL. Loader accepts old
files without `section`/`kind` (legacy `kind` values normalize to `prose`).

## Known defects (to be fixed in Gate 1)

This snapshot reflects the half-finished ingestion and is **not** the target
state. Verified problems:

- **Off-list categories excluded**: "University Courses and Lecture Notes" was
  removed from `CORE_CATEGORIES`; tier never admits off-list categories.  The
  MIT 15.450 lecture-notes index page (off-list category) is not fetched, so its
  58 chunks are absent from the corpus. Individual MIT 15.450 lecture resources
  (categories in the allow-list) are admitted directly as separate CSV entries.
- **Imbalance**: Monte Carlo arXiv 2208.05531 = 220 chunks (44%) and Bielefeld
  NumSDE = 102 (20%). Per-document cap (25%) available via `--per-doc-cap 0.25`
  (opt-in, default off). "Black–Scholes and Option Pricing" has 9 chunks,
  Stochastic Processes 9, SDEs 5, Heston/SABR 17, Greeks 55, Stochastic
  Calculus 70.
- **Filters now fire**: toc/references/boilerplate detectors catch real snippets.
  TOC (Bielefeld), boilerplate (MIT Terms of Use), references (FinRL-X) all
  detected.
- **26+ letter runs**: 1 remaining ("strongoptimalapproximation" in Monte Carlo).
- **Chunk size**: median 570 tokens, max 663, 0 over 800, 93.8% in 400-800 range.
- **Short lines <=2 chars**: all documents under 5% (max 1.5%).
- **Topic coverage (chunk text)**: "Put–call parity" (2 chunks), "Milstein"
  (1 chunk), "Hull–White" (3 chunks) are thin (< 5 chunks). "Variance reduction"
  (8 chunks) and "Heston"/"SABR" (9 chunks each) are also limited.
- Columbia IEOR 6712 notes ingested manually (6 chunks); NYU Avellaneda
  DSLecture6.pdf is image-locked (~194 chars/page) and correctly marked
  needs_ocr and excluded.
- Tier-1 landing pages (NYU Kohn, Goodman) blocked by robots.txt; MIT 15.450
  lecture-notes index blocked (off-list category); CSU Stanislaus "High Finance"
  page: no PDF links found, not ingested.

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
- `822771c`: extractor selection, single-symbol lines, squashed prose, join audit fix, landing page second hop

## Corpus coverage (chunk text, case-insensitive)

| Topic | Chunks | Share | Status |
|---|---|---|---|
| Black–Scholes | 47 | 9.4% | |
| Put–call parity | 2 | 0.4% | THIN (< 5 chunks) |
| Delta/Gamma/Vega | 42 | 8.4% | |
| Itô | 92 | 18.4% | |
| GBM | 11 | 2.2% | |
| SDE | 61 | 12.2% | |
| Euler–Maruyama | 14 | 2.8% | |
| Milstein | 1 | 0.2% | THIN (< 5 chunks) |
| Monte Carlo | 50 | 10.0% | |
| Variance reduction | 8 | 1.6% | |
| Heston | 9 | 1.8% | |
| SABR | 9 | 1.8% | |
| Vasicek | 7 | 1.4% | |
| Hull–White | 3 | 0.6% | THIN (< 5 chunks) |
| CIR | 10 | 2.0% | |

Topics with < 5 chunks are known gaps: Put–call parity, Milstein, Hull–White.
