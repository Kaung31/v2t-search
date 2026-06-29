# Deploy — V2T-Search (Phase 2)

Scale-to-zero **CPU** host (Railway or Render). Query-time loads only the SigLIP **text** encoder;
YOLO / SigLIP-image / Redis are ingestion-only and do not run here.

Planned artifacts (added in Phase 2):

- `Dockerfile` — CPU image: FastAPI + SigLIP text encoder + Postgres/Qdrant clients.
- `railway.json` / `render.yaml` — service config, health check, scale-to-zero.
- Env contract: [`../../../infra/env/v2t-search.env.example`](../../../infra/env/v2t-search.env.example).

Gate (brief §8, Phase 2): a live query from the deployed site returns ranked results with per-layer
"why" data; the semantic-vs-fused toggle reproduces the 0.04→0.99 gap; cold start < ~10 s surfaced
gracefully; the backend key never reaches the browser.

`TODO(owner): provision Neon + Qdrant Cloud + the chosen host; fill exact build/run commands.`
