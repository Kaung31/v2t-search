"""Unified search: plan -> retrieve across layers -> RRF fuse -> ranked results."""
from fastapi import FastAPI
from pydantic import BaseModel

from apps.api.planner import make_plan
from apps.api.retrieval.crosscam import find_correspondences
from apps.api.retrieval.semantic import semantic_object_search
from apps.api.retrieval.fusion import reciprocal_rank_fusion

app = FastAPI(title="V2T-Search")


class SearchRequest(BaseModel):
    query: str
    use_llm: bool = False
    limit: int = 20


def run_search(query: str, use_llm: bool = False, limit: int = 20) -> dict:
    plan = make_plan(query, use_llm=use_llm)

    ranked_lists: dict[str, list[str]] = {}
    details: dict[str, dict] = {}

    # Layer 1 — cross-camera correspondence (structured)
    if plan.driver_states or plan.road_states:
        corr = find_correspondences(
            driver_states=plan.driver_states or None,
            road_states=plan.road_states or None,
            limit=200,
        )
        ranked_lists["correspondence"] = [h.sample_id for h in corr]
        for h in corr:
            details.setdefault(h.sample_id, {})["correspondence"] = h.matched_events

    # Layer 2 — semantic objects
    sem = semantic_object_search(plan.semantic_query or query, limit=200)
    ranked_lists["semantic"] = [h.sample_id for h in sem]
    for h in sem:
        details.setdefault(h.sample_id, {})["semantic"] = {
            "best_class": h.best_class, "view": h.view, "score": round(h.score, 3),
        }

    # Fuse. Weight correspondence a bit higher — it's the precise structured signal.
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

    return {
        "query": query,
        "plan": plan.to_dict(),
        "result_count": len(results),
        "results": results,
    }


@app.post("/search")
def search(req: SearchRequest):
    return run_search(req.query, use_llm=req.use_llm, limit=req.limit)
