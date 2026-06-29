# V2T-Search query service — scale-to-zero CPU host (Railway / Render / Fly).
# Query-time loads only the SigLIP TEXT encoder; YOLO / SigLIP-image / Redis are
# ingestion-only and are NOT installed here.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    SIGLIP_MODEL=google/siglip2-base-patch16-256

WORKDIR /app

# System libs needed by torch/pillow at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU torch from the CPU wheel index (keeps the image small), then the rest.
COPY deploy/requirements-serve.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 torchvision==0.27.0 \
    && pip install -r requirements-serve.txt

# Pre-bake the SigLIP text encoder so cold starts don't pay a model download
# (build brief Phase 2 gate: cold start < ~10s). Set BAKE_MODEL=0 to skip.
ARG BAKE_MODEL=1
RUN if [ "$BAKE_MODEL" = "1" ]; then \
      python -c "from transformers import AutoProcessor, AutoModel; \
m='google/siglip2-base-patch16-256'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m)"; \
    fi

# App code (apps/, services/, packages/). Keeps the package import paths intact.
COPY apps ./apps
COPY services ./services
COPY packages ./packages

EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8002}/health || exit 1

# Honor the platform's $PORT (Railway/Render inject it); default 8002.
CMD ["sh", "-c", "uvicorn apps.api.search_api:app --host 0.0.0.0 --port ${PORT:-8002}"]
