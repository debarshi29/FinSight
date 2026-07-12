# ── Stage 1: dependency installer ────────────────────────────────
# Installs runtime deps into an isolated virtualenv so the final image
# carries no build toolchain (pip, uv, gcc, etc.).
# Layer-cached: the uv pip install layer rebuilds only when pyproject.toml
# or uv.lock change, not when application source changes.
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy dep specs first — keeps the install layer cache-stable
COPY pyproject.toml uv.lock ./

# 1. Extract runtime deps from pyproject.toml into a plain requirements file.
#    tomllib is stdlib in Python 3.11 — no extra package needed.
# 2. Create virtualenv and install all runtime deps into it.
RUN python3 -c \
    "import tomllib; deps=tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; open('/tmp/reqs.txt','w').write('\n'.join(deps))" && \
    uv venv .venv && \
    uv pip install --python .venv/bin/python --no-cache -r /tmp/reqs.txt

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

# Writable runtime directories (overridden by volume mounts in compose)
RUN mkdir -p audit_logs data/filings

# Run as a non-root user
RUN useradd --no-create-home --shell /bin/false finsight && \
    chown -R finsight:finsight /app
USER finsight

EXPOSE 8000

# Liveness probe — /health returns {"status":"ok"} when the app is ready.
# --start-period gives time for HuggingFace model downloads on first cold start.
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# --workers 1 preserves in-process state (SK kernel singleton, metrics store).
# --log-level warning silences uvicorn's own logger; structlog handles structured output.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log", \
     "--log-level", "warning"]
