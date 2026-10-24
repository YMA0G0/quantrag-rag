# QuantRAG

**Milestone 1** — a minimal, working MCP server that exposes an LLM (hosted on
OpenRouter) to any MCP-compatible client.

The server exposes exactly one tool, `ask_quant_question`, which forwards a
question to the configured OpenRouter model and returns the model's answer.
This is the foundation for the future cloud-hosted Quantitative Finance RAG
service; RAG, embeddings, vector storage, auth and deployment are **not** part
of this milestone.

---

## Architecture

```
MCP client  ──stdio──▶  quantrag.server  ──▶  quantrag.openrouter  ──HTTPS──▶  OpenRouter
 (tools/list,            (FastMCP tool           (httpx, Chat
  tools/call)             ask_quant_question)      Completions API)
```

| Module | Responsibility |
|---|---|
| `src/quantrag/config.py` | Reads `OPENROUTER_*` environment variables (plus `.env` for local dev) into a validated, frozen `Settings` object. |
| `src/quantrag/openrouter.py` | All HTTP logic: request building, auth header, error/timeout/rate-limit/malformed-response handling. Knows nothing about MCP. |
| `src/quantrag/errors.py` | Exception hierarchy (`ConfigurationError`, `Upstream*Error`, …) plus a `redact()` helper so credentials never reach an error message. |
| `src/quantrag/server.py` | FastMCP server, the `ask_quant_question` tool, logging setup and the stdio entry point. |

The MCP layer only depends on the small `ChatCompleter` protocol
(`complete(prompt, *, system=None) -> str`), so the HTTP client can be swapped
or mocked without touching the tool handler.

---

## Knowledge-base ingestion (offline)

The ingestion pipeline is separate from the server runtime (`server.py` never
imports it). It turns the source CSVs into `data/chunks.jsonl`.

```bash
# install the optional heavy dependencies (pypdf, trafilatura, tiktoken, ...)
uv sync --extra ingest

# preview selection without any network access
uv run python -m quantrag.ingest --dry-run --profile core
uv run python -m quantrag.ingest --dry-run --profile all

# real run (respects robots.txt, caches downloads under data/raw/)
uv run python -m quantrag.ingest --profile core --max-tier 1 --limit 10
```

Key flags: `--profile {core,all}` (default `core`: tier <= 2 or a core
finance category), `--max-tier N`, `--limit N`, `--only-category TEXT`,
`--include-unreviewed`, `--target-tokens`, `--max-tokens` (hard cap, default
900), `--manual-csv`, `--manual-dir`.

Outputs: `data/chunks.jsonl`, `data/ingest_report.json`,
`data/ingest_report.csv`. A `--dry-run` writes
`data/ingest_report.dryrun.json` / `.dryrun.csv` instead, so previewing a
profile never overwrites the artifacts of the last real run. The report records a reason code for every skipped
or failed source (`not_extractable`, `access_blocked`,
`unsupported_resource_type`, `unsupported_content_type`, `needs_review`,
`needs_manual_review`, `landing_or_marketing`, `outside_core_profile`,
`duplicate_url`, `not_found`, `thin_page`, `empty_extraction`, `http_*`,
`robots_disallowed`, `extract_error: ...`, `manual_file_missing`).

### Manually downloaded sources

Some sources block crawlers via `robots.txt`. Download those by hand and list
them in `data/manual_sources.csv`:

```csv
title,url,category,file,priority_tier
Black-Scholes Notes,https://www.columbia.edu/~ks20/6712-14/6712-14-Notes-B-Scholes.pdf,Black–Scholes and Option Pricing,6712-14-Notes-B-Scholes.pdf,1
```

`file` is resolved relative to `data/manual/`. Manual files go through the
same extract/chunk path and are cited by their original `url`; no network
request is made for them. The manual directory is git-ignored except for its
README.

---

## Requirements

* Python >= 3.12
* [uv](https://docs.astral.sh/uv/) (recommended; a plain `pip` install works too)
* An OpenRouter API key and a model id that is available on your account

---

## Installation

```bash
cd /data/projects/proj26/Cloud_project_clean

# with uv (creates .venv and installs the project in editable mode)
uv sync --extra dev

# or, with plain pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Configuration

Configuration comes from environment variables. For local development, put them
in a `.env` file (git-ignored) — copy the template:

```bash
cp .env.example .env
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENROUTER_API_KEY` | yes | — | OpenRouter API key, sent as `Authorization: Bearer …` |
| `OPENROUTER_MODEL` | yes | — | Model id, e.g. `openai/gpt-4o-mini` (never hardcoded) |
| `OPENROUTER_BASE_URL` | no | `https://openrouter.ai/api/v1` | API base URL (proxy/testing) |
| `OPENROUTER_TIMEOUT_SECONDS` | no | `60` | Per-request timeout |
| `OPENROUTER_MAX_TOKENS` | no | `1024` | Cap on the completion length; bounding it caps the credit reserved per request |
| `OPENROUTER_APP_NAME` | no | `QuantRAG` | Sent as `X-Title` |
| `OPENROUTER_APP_URL` | no | — | Sent as `HTTP-Referer` |

Real environment variables always take precedence over the `.env` file.

> `.env` is excluded by `.gitignore`; only `.env.example` (placeholders) is
> committed. API keys are never written to logs or error messages — upstream
> payloads are passed through `redact()` before being surfaced.

---

## Running the server

The server uses **stdio** transport, so it is normally launched by an MCP
client rather than by hand:

```bash
uv run quantrag            # console script
# or
uv run python -m quantrag  # module entry point
```

For a quick manual smoke test (the server waits for MCP messages on stdin):

```bash
OPENROUTER_API_KEY=sk-or-... OPENROUTER_MODEL=openai/gpt-4o-mini uv run quantrag
```

`stdout` is reserved for the MCP protocol; all diagnostics are written to
`stderr`.

---

## Connecting an MCP client

### Claude Desktop / Claude Code (stdio)

Add to your client configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "quantrag": {
      "command": "uv",
      "args": [
        "--directory", "/data/projects/proj26/Cloud_project_clean",
        "run", "quantrag"
      ],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-...",
        "OPENROUTER_MODEL": "openai/gpt-4o-mini"
      }
    }
  }
}
```

### Any MCP client (generic JSON)

```json
{
  "mcpServers": {
    "quantrag": {
      "command": "/data/projects/proj26/Cloud_project_clean/.venv/bin/python",
      "args": ["-m", "quantrag"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-...",
        "OPENROUTER_MODEL": "openai/gpt-4o-mini"
      }
    }
  }
}
```

After connecting, the client can:

1. discover the tool — `tools/list` returns `ask_quant_question` with schema
   `{ "question": { "type": "string" } }`;
2. call it — `tools/call` with `{"question": "How do I compute a rolling Sharpe ratio?"}`
   returns the model's text answer.

`tests/test_integration_stdio.py` performs exactly this handshake against a
local mock OpenRouter, so the flow is verified offline.

---

## Tests, linting and type checks

The whole suite is offline and credential-free by default:

```bash
uv run pytest -q            # full suite (includes an end-to-end stdio MCP test)
uv run ruff check .         # lint
uv run ruff format --check .
uvx pyright                 # type check (pyright is run through uvx, not installed in .venv)
```

Optional live smoke test against the real OpenRouter API — never run by
default, and skipped unless explicitly enabled *and* credentials are present:

```bash
QUANTRAG_SMOKE=1 uv run pytest -m smoke -q
```

It reads the same `.env` as the server. Two account-side conditions make it
skip or fail even though the code is fine:

* **Free models** (`…:free`) are limited to 50 requests/day on a free-tier
  account; once exhausted the API answers `429` until the daily reset at
  00:00 UTC.
* **Paid models** reserve credit against `max_tokens`. With a near-empty
  balance OpenRouter answers `402`; keep `OPENROUTER_MAX_TOKENS` low.

What each layer covers:

| Area | File |
|---|---|
| Environment / `.env` configuration | `tests/test_config.py` |
| Error classes and credential redaction | `tests/test_errors.py` |
| OpenRouter HTTP client (mocked) | `tests/test_openrouter.py` |
| MCP tool handler (mocked backend) | `tests/test_server.py` |
| MCP protocol over stdio (real subprocess, mock backend) | `tests/test_integration_stdio.py` |
| Live OpenRouter round-trip (opt-in) | `tests/test_smoke_openrouter.py` |

The suite covers configuration validation, request construction (model id,
auth header, payload), and every failure mode: 401/403, 429 (with
`Retry-After`, including a non-numeric header), 4xx/5xx, timeouts, connection
errors, malformed JSON, missing `choices`, missing `content`, and credential
redaction in both exceptions and log records. The stdio tests additionally
assert that every line the server writes to stdout is a valid JSON-RPC
message and that the API key never appears on stdout or stderr.

---

## Project layout

```
Cloud_project_clean/
├── .env.example           # placeholder configuration
├── .gitignore
├── pyproject.toml
├── README.md
├── src/quantrag/
│   ├── __init__.py
│   ├── __main__.py        # python -m quantrag
│   ├── config.py          # environment/.env configuration
│   ├── errors.py          # error types + redaction
│   ├── openrouter.py      # httpx Chat Completions client
│   └── server.py          # FastMCP server + ask_quant_question
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_errors.py
    ├── test_openrouter.py
    ├── test_server.py
    ├── test_integration_stdio.py
    └── test_smoke_openrouter.py   # opt-in live test (QUANTRAG_SMOKE=1)
```

---

## Limitations (intentional for this milestone)

* No RAG yet: no document ingestion, embeddings, vector database or citations.
* No authentication/authorization and no cloud deployment; stdio + local `.env` only.
* No streaming, tool-calling or multi-turn conversation state in the upstream request.
* The model id must exist on your OpenRouter account — this project does not
  provision or assume any particular model.
* The upstream request has a single system prompt; prompt versioning is not
  implemented yet.
