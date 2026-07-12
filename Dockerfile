# ── Stage 1: dependency installer ────────────────────────────────
# Installs runtime deps into an isolated virtualenv so the final runtime
# image has no build toolchain (pip, uv, git, etc.).
# Layer-cached: rebuilds only when pyproject.toml or uv.lock change.
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dep specs only — source not needed here so this layer is cache-stable
COPY pyproject.toml uv.lock ./

# Create venv and install only runtime dependencies (not dev extras, not the project itself)
# tomllib is stdlib in Python 3.11 — no extra dep needed
RUN uv venv .venv && \
    python3 -c "
import tomllib, subprocess, sys
with open('pyproject.toml', 'rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
result = subprocess.run(
    ['uv', 'pip', 'install', '--python', '.venv/bin/python', '--no-cache'] + deps,
    capture_output=False
)
sys.exit(result.returncode)
"

# ── Stage 2: runtime image ────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="FinSight"
LABEL org.opencontainers.image.description="Regulatory-grade financial intelligence"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# curl is only needed for the HEALTHCHECK probe
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtualenv from the builder — no pip/uv in the runtime image
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application source (respects .dockerignore — excludes .env, qdrant_storage, tests, etc.)
COPY . .

# Writable runtime directories (usually overridden by volume mounts in compose)
RUN mkdir -p audit_logs data/filings

# Run as a non-root user
RUN useradd --no-create-home --shell /bin/false finsight && \
    chown -R finsight:finsight /app
USER finsight

EXPOSE 8000

# Liveness probe: /health returns {"status":"ok"} when the app is up.
# --start-period gives model-download time on first cold start.
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# --workers 1: single process preserves in-process state (SK kernel singleton, metrics store).
# --log-level warning: uvicorn's own logger is silenced; structlog handles structured output.
# --access-log: HTTP request lines go to stdout alongside structlog events.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log", \
     "--log-level", "warning"]
