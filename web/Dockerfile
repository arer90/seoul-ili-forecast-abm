# Frame D dev image — Next.js + MCP bridge + Python MCP server.
#
# Only used for local E2E (docker compose). Vercel deploy doesn't use
# this Dockerfile — it serves the Next.js app directly and talks to
# Turso / MCP bridge over HTTPS.

FROM node:20-bookworm-slim

# System packages:
#   - python3 / pip  : to run `python -m simulation mcp-server`
#   - build-essential: numpy / scipy wheels often need headers on slim
#   - git            : occasionally needed by pip during dev
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node deps first so the build cache survives source edits.
COPY web/package.json web/package-lock.json* ./web/
RUN cd web && npm ci --no-audit --no-fund

# Install the Python side into a venv at /app/.venv.
COPY pyproject.toml requirements*.txt ./
RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --upgrade pip \
    && /app/.venv/bin/pip install -e . || true

# Copy the rest of the repo. We keep the layer order so that edits to
# simulation/ don't invalidate the node_modules layer.
COPY simulation ./simulation
COPY web ./web

ENV PORT=3000 \
    NODE_ENV=development \
    PYTHONUNBUFFERED=1 \
    MCP_PYTHON=/app/.venv/bin/python \
    MCP_BRIDGE_HOST=0.0.0.0 \
    MCP_BRIDGE_PORT=8787 \
    MCP_BRIDGE_URL=http://localhost:8787

EXPOSE 3000 8787

# Run the MCP bridge + Next.js dev server together.
CMD ["/bin/sh", "-c", "cd /app/web && npx tsx scripts/mcp-bridge.ts & cd /app/web && npm run dev"]
