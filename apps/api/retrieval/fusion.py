"""Reciprocal Rank Fusion: combine multiple ranked lists into one."""
from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> tuple[list[tuple[str, float]], dict[str, list[str]]]:
    """
    ranked_lists: {layer_name: [sample_id, ...] in rank order (best first)}
    Returns:
      fused: [(sample_id, fused_score), ...] sorted best first
      provenance: {sample_id: [layer_names that contributed]}
    """
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    provenance: dict[str, list[str]] = defaultdict(list)

    for layer, sample_ids in ranked_lists.items():
        w = weights.get(layer, 1.0)
        for rank, sid in enumerate(sample_ids, start=1):
            scores[sid] += w / (k + rank)
            provenance[sid].append(layer)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused, provenance
