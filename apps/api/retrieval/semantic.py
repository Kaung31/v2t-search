"""Semantic retrieval over the Qdrant objects collection, aggregated to sample level."""
from dataclasses import dataclass
from qdrant_client import QdrantClient

from packages.core.config import settings
from services.worker.embed import embed_text


@dataclass
class SemanticHit:
    sample_id: str
    score: float
    best_class: str
    view: str


_qdrant = None
def _client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _qdrant


def semantic_object_search(query_text: str, limit: int = 100,
                           candidate_multiplier: int = 8) -> list[SemanticHit]:
    """
    Embed the query, search the objects collection, then collapse to the best
    crop per sample. Returns samples ranked by their best-matching crop score.
    """
    emb = embed_text(query_text)
    raw = _client().query_points(
        collection_name="objects",
        query=emb,
        limit=limit * candidate_multiplier,   # over-fetch since many crops per sample
    ).points

    best: dict[str, SemanticHit] = {}
    for h in raw:
        sid = h.payload["sample_id"]
        if sid not in best or h.score > best[sid].score:
            best[sid] = SemanticHit(
                sample_id=sid, score=float(h.score),
                best_class=h.payload.get("class", "?"),
                view=h.payload.get("view", "?"),
            )
    ranked = sorted(best.values(), key=lambda x: x.score, reverse=True)
    return ranked[:limit]
