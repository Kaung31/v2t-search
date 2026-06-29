"""POST a manifest path -> enqueue one job per sample to a Redis Stream."""
import csv
import hashlib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis

from packages.core.config import settings

app = FastAPI()
r = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
STREAM = "ingest:jobs"


class IngestRequest(BaseModel):
    manifest_path: str
    limit: int | None = None


@app.post("/ingest")
def ingest(req: IngestRequest):
    path = Path(req.manifest_path)
    if not path.is_file():
        raise HTTPException(404, f"Manifest not found: {req.manifest_path}")
    queued = 0
    with path.open() as f:
        for i, row in enumerate(csv.DictReader(f)):
            if req.limit and i >= req.limit:
                break
            content_hash = hashlib.md5(
                Path(row["annotation_path"]).read_bytes(), usedforsecurity=False
            ).hexdigest()[:12]
            r.xadd(STREAM, {
                "content_hash":    content_hash,
                "sample_id":       row["sample_id"],
                "split":           row["split"],
                "sample_dir":      row["sample_dir"],
                "annotation_path": row["annotation_path"],
            })
            queued += 1
    return {"queued": queued, "stream": STREAM}
