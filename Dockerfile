# Multi-stage Dockerfile for QuantRAG private REST + MCP deployment.
# The image contains only the prebuilt index (data/index), not the
# raw corpus or manual data.

# -------------------------------------------------------------------
# Stage 1: build environment (install dependencies from uv.lock)
# -------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy lockfile and project metadata for dependency resolution
COPY uv.lock pyproject.toml ./

# Install dependencies (production only, no dev/test extras needed)
RUN uv sync --frozen --no-dev --extra api --no-install-project

# -------------------------------------------------------------------
# Stage 2: runtime image (non-root user, minimal footprint)
# -------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Create non-root user for container execution
RUN groupadd -r quantrag && useradd -r -g quantrag -d /app -s /sbin/nologin quantrag

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /build/.venv/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages/
COPY --from=builder /build/.venv/bin /usr/local/bin/
COPY --from=builder /usr/local/bin /usr/local/bin/

# Copy source code (only src/, not tests or caches)
COPY src/ ./src/

# Copy prebuilt index (required for container startup without rebuild)
# The index directory must exist; it is NOT rebuilt at runtime.
COPY data/index/ ./data/index/

# Copy configuration examples (no secrets embedded)
COPY .env.example .env.example

# Set runtime environment defaults (operator must supply real secrets)
ENV PORT=8080
ENV QUANTRAG_API_KEYS=""
# Fail-safe: the container refuses to start without a key, so a misconfigured
# deployment can never expose /ask, /search or /mcp publicly.
ENV QUANTRAG_REQUIRE_AUTH=1
ENV QUANTRAG_API_KEY=""
ENV QUANTRAG_API_URL=""
ENV OPENROUTER_API_KEY=""
ENV OPENROUTER_MODEL="openai/gpt-4o-mini"
ENV QUANTRAG_INDEX_DIR="data/index"
ENV PYTHONPATH="src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Use non-root user
USER quantrag

# Expose port (operator can override via PORT env at runtime)
EXPOSE 8080

# Health-check endpoint (used by container orchestrator)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8080}/health')" || exit 1

# Start the FastAPI REST server (stateless, uses prebuilt index)
CMD python -m uvicorn quantrag.api:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
