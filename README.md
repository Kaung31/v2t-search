# V2T-Search

Retrieval system for multi-camera driving video that answers compound behavioral queries — like *"a driver on the phone while turning"* — by combining an in-cabin camera with a road camera, something no single frame embedding can do on its own.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![License: TODO](https://img.shields.io/badge/license-TODO-lightgrey)

<!-- TODO: add a screenshot or short GIF of a search — type a query, show the ranked results, then the quad-camera player syncing all four views -->

## Overview

Standard video search embeds frames with a vision-language model and finds the nearest match to a text query. That works for appearance ("a red truck") but fails for compound behavioral queries, because driver emotion lives on the in-cabin camera and road context lives on the forward camera — no single embedding captures both. This project splits retrieval into three purpose-built layers (semantic, structured events, cross-camera correspondence), fuses them, and measures what each layer actually contributes. Built on the [AIDE dataset](https://github.com/ydk122024/AIDE) of synchronized 4-camera driving clips.

**Headline result:** on compound queries, semantic-only search gets Recall@10 = 0.04. The fused system gets 0.99. Overall Recall@10 across all query types goes from 0.26 to 0.95.

## Features

- Search driving clips by natural language, including compound queries that mix driver state and road context ("anxious driver changing lanes")
- Cross-camera correspondence layer that finds clips where an in-cabin condition and a road condition co-occur — the piece that makes compound queries work at all
- Dual-mode query planner: deterministic rule-based matching (for reproducible eval) or a Claude-powered planner (for open-ended phrasing)
- Reciprocal Rank Fusion combines the semantic and structured layers into one ranked list, weighted per layer
- Synchronized quad-camera player (front/left/right/incar) with a "why matched" panel showing which layer and which event matched
- Semantic-vs-fused mode toggle to compare retrieval quality with and without the structured layer
- Idempotent, resumable ingestion pipeline (content-hash dedup, safe to re-run or restart mid-way)
- Ablation harness that scores three retrieval configs against a 48-query hand-verified eval set and renders the results as a chart

## Tech stack

| Tech | Used for |
|---|---|
| Python 3.12 | Backend language |
| FastAPI | Search API, video streaming, static UI hosting |
| PostgreSQL 16 | Structured driver/road events, cross-camera join |
| Qdrant | Vector search over object-crop embeddings |
| Redis Streams | Idempotent, checkpointed ingestion queue |
| SigLIP 2 | Text/image embeddings for appearance search |
| YOLO26 (ultralytics) | Object detection on frames before embedding crops |
| Anthropic Claude API | Optional LLM query planner |
| Docker Compose | Local Postgres + Redis + Qdrant |

## Getting started

### Prerequisites

- Python 3.12
- Docker (for Postgres, Redis, Qdrant)
- The [AIDE dataset](https://github.com/ydk122024/AIDE) extracted to `data/aide/AIDE_Dataset/`

### Installation

```bash
# infrastructure
docker compose up -d

# python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Environment variables

| Variable | Purpose | Where to get it |
|---|---|---|
| `POSTGRES_USER` | Postgres username | set your own (default: `v2t`) |
| `POSTGRES_PASSWORD` | Postgres password | set your own |
| `POSTGRES_DB` | Postgres database name | set your own (default: `v2t`) |
| `POSTGRES_HOST` | Postgres host | `postgres` for docker-compose, `localhost` for local Python |
| `POSTGRES_PORT` | Postgres port | `5432` |
| `REDIS_HOST` | Redis host | `redis` for docker-compose, `localhost` for local Python |
| `REDIS_PORT` | Redis port | `6379` |
| `QDRANT_HOST` | Qdrant host | `qdrant` for docker-compose, `localhost` for local Python |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `ANTHROPIC_API_KEY` | Enables the LLM query planner (`use_llm=true`) | console.anthropic.com — optional, rule-based planner works without it |

Cloud-only overrides (`DATABASE_URL`, `QDRANT_URL`, `QDRANT_API_KEY`, `MEDIA_BASE_URL`) exist for deploying against Neon/Qdrant Cloud — see [packages/core/config.py](packages/core/config.py). Not needed for local dev.

### Ingest the data

Run once, in order, after the AIDE dataset is in place:

```bash
python services/worker/aide_loader.py    # builds data/aide/manifest.csv
python -m scripts.ingest_labels           # driver/road events into Postgres
python -m scripts.ingest_objects          # YOLO + SigLIP object embeddings into Qdrant (slow)
```

### Run it

```bash
uvicorn apps.api.search_api:app --port 8002
```

Open `http://localhost:8002`.

## Usage

Search from the UI, or hit the API directly:

```bash
curl -X POST http://localhost:8002/search \
  -H "Content-Type: application/json" \
  -d '{"query": "driver on the phone while turning", "mode": "fused", "limit": 10}'
```

```json
{
  "query": "driver on the phone while turning",
  "plan": {"driver_states": ["making_phone"], "road_states": ["turning"], "planner": "rule"},
  "mode": "fused",
  "result_count": 10,
  "results": [
    {"sample_id": "001842", "fused_score": 0.03252, "matched_by": ["correspondence", "semantic"], "details": {"...": "..."}}
  ]
}
```

Set `"mode": "semantic"` or `"mode": "correspondence"` to see either layer in isolation — that's what the ablation compares. Set `"use_llm": true` to route the query through Claude instead of the rule-based planner.

## Project structure

```
apps/
  api/            # FastAPI app: search_api.py (main entry), planner.py, retrieval/ (semantic, crosscam, fusion), static/ (UI)
  web/            # early Next.js prototype, superseded by apps/api/static — not actively used
services/
  worker/         # ingestion: AIDE loader, YOLO detection, SigLIP embedding, event extraction, Redis consumer
  ingest_api/     # small FastAPI service that queues ingestion jobs onto Redis
packages/
  core/           # shared settings (packages/core/config.py)
infra/
  init.sql        # Postgres schema: assets, clips, events, feedback, eval_runs
scripts/          # ingestion scripts, eval-set builder, ablation harness, chart renderer
eval/             # generated eval queries, ablation results, headline chart
tests/            # planner injection tests + infra smoke tests
deploy/           # Dockerfile + Railway/Render config for the scale-to-zero query API
```

## How it works

A query goes through a planner that maps free text onto a fixed vocabulary of driver states, road states, and object classes ([apps/api/planner.py](apps/api/planner.py)) — either deterministically or via Claude. The correspondence layer runs a SQL self-join in Postgres to find clips where an in-cabin state and a road state co-occur ([apps/api/retrieval/crosscam.py](apps/api/retrieval/crosscam.py)). The semantic layer embeds the query with SigLIP and searches object-crop vectors in Qdrant ([apps/api/retrieval/semantic.py](apps/api/retrieval/semantic.py)). Both ranked lists get merged with Reciprocal Rank Fusion ([apps/api/retrieval/fusion.py](apps/api/retrieval/fusion.py)), and the top results are returned with per-layer match details so the UI can show why each clip matched.

```
AIDE dataset → Redis Streams → workers (YOLO26 + SigLIP 2 + label→event extraction)
            → Postgres (assets, clips, events) + Qdrant (objects)
            → FastAPI: planner → retrieval layers → RRF fusion → /search
```

## Roadmap / known limitations

- Cross-camera correspondence is same-clip co-occurrence, not true temporal sequencing — AIDE only provides clip-level labels, not sub-clip timestamps
- Dataset support is AIDE-specific: folder layout, label keys, and the event vocabulary are hard-coded. A `DatasetAdapter` interface to support other datasets is planned but not built
- Fusion weights (`correspondence: 1.5, semantic: 1.0`) are hand-picked, not learned or tuned against the eval set
- No authentication on any endpoint, and no rate limiting on the LLM planner path — fine for a local demo, not for a public deployment
- No CI pipeline; tests are run manually (`pytest tests/test_planner_injection.py`, `make test` for infra smoke tests)
- Keypoint-derived blink/gaze detection was built and measured, then disabled — AIDE's pre-extracted keypoints had mean confidence under 0.05 (code kept, commented out, in [services/worker/keypoints.py](services/worker/keypoints.py))

## License

<!-- TODO: no LICENSE file exists yet. MIT is the common default for a solo portfolio project like this — add a LICENSE file if you want one. -->

## Contact

<!-- TODO: add your name, GitHub profile link, and LinkedIn -->
