"""Unified search API + static frontend + video streaming."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apps.api.planner import make_plan
from apps.api.retrieval.crosscam import find_correspondences
from apps.api.retrieval.semantic import semantic_object_search
from apps.api.retrieval.fusion import reciprocal_rank_fusion

app = FastAPI(title="V2T-Search")

DATASET = Path("data/aide/AIDE_Dataset")
VIEWS = {"front", "left", "right", "incar"}
STATIC_DIR = Path("apps/api/static")


class SearchRequest(BaseModel):
    query: str
    use_llm: bool = False
    limit: int = 20
    # mode powers the semantic-vs-fused comparison toggle (contract §6.1).
    # "semantic" | "correspondence" | "fused"
    mode: str = "fused"


def run_search(
    query: str, use_llm: bool = False, limit: int = 20, mode: str = "fused"
) -> dict:
    plan = make_plan(query, use_llm=use_llm)
    ranked_lists: dict[str, list[str]] = {}
    details: dict[str, dict] = {}

    want_corr = mode in ("fused", "correspondence")
    want_sem = mode in ("fused", "semantic")

    if want_corr and (plan.driver_states or plan.road_states):
        corr = find_correspondences(
            driver_states=plan.driver_states or None,
            road_states=plan.road_states or None,
            limit=200,
        )
        ranked_lists["correspondence"] = [h.sample_id for h in corr]
        for h in corr:
            details.setdefault(h.sample_id, {})["correspondence"] = h.matched_events

    if want_sem:
        sem = semantic_object_search(plan.semantic_query or query, limit=200)
        ranked_lists["semantic"] = [h.sample_id for h in sem]
        for h in sem:
            details.setdefault(h.sample_id, {})["semantic"] = {
                "best_class": h.best_class, "view": h.view, "score": round(h.score, 3),
            }

    fused, provenance = reciprocal_rank_fusion(
        ranked_lists, weights={"correspondence": 1.5, "semantic": 1.0},
    )
    results = []
    for sid, score in fused[:limit]:
        results.append({
            "sample_id": sid,
            "fused_score": round(score, 5),
            "matched_by": provenance[sid],
            "details": details.get(sid, {}),
        })
    return {"query": query, "plan": plan.to_dict(), "mode": mode,
            "result_count": len(results), "results": results}


@app.post("/search")
def search(req: SearchRequest):
    return run_search(req.query, use_llm=req.use_llm, limit=req.limit, mode=req.mode)


@app.get("/health")
def health():
    """Liveness probe for the scale-to-zero host (cheap; no model load)."""
    return {"status": "ok"}


@app.get("/video/{sample_id}/{view}")
def video(sample_id: str, view: str):
    if view not in VIEWS or not sample_id.isdigit():
        raise HTTPException(400, "invalid sample or view")
    path = DATASET / sample_id / f"{view}.mp4"
    if not path.is_file():
        raise HTTPException(404, f"video not found: {sample_id}/{view}")
    # Starlette's FileResponse supports HTTP range requests, so seeking works.
    return FileResponse(str(path), media_type="video/mp4")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# (mount static last so it doesn't shadow the routes above)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
