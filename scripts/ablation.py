"""Run the system in 3 configs against the eval set, compute metrics, save results + chart."""
import json
from collections import defaultdict

from apps.api.planner import make_plan
from apps.api.retrieval.crosscam import find_correspondences
from apps.api.retrieval.semantic import semantic_object_search
from apps.api.retrieval.fusion import reciprocal_rank_fusion

EVAL = "eval/eval_queries.json"
OUT_JSON = "eval/ablation_results.json"
K = 10


# ---- retrieval configurations ----
def retrieve_semantic_only(plan, limit=50):
    return [h.sample_id for h in semantic_object_search(plan.semantic_query or plan.query, limit=limit)]

def retrieve_correspondence_only(plan, limit=50):
    if not (plan.driver_states or plan.road_states):
        return []
    return [h.sample_id for h in find_correspondences(
        driver_states=plan.driver_states or None,
        road_states=plan.road_states or None, limit=limit)]

def retrieve_fused(plan, limit=50):
    lists = {}
    corr = retrieve_correspondence_only(plan, limit=200)
    if corr:
        lists["correspondence"] = corr
    lists["semantic"] = retrieve_semantic_only(plan, limit=200)
    fused, _ = reciprocal_rank_fusion(lists, weights={"correspondence": 1.5, "semantic": 1.0})
    return [sid for sid, _ in fused[:limit]]

CONFIGS = {
    "semantic_only":        retrieve_semantic_only,
    "correspondence_only":  retrieve_correspondence_only,
    "fused":                retrieve_fused,
}


# ---- metrics ----
def recall_at_k(retrieved, relevant, k=K):
    rel = set(relevant)
    hits = sum(1 for sid in retrieved[:k] if sid in rel)
    return hits / min(len(rel), k) if rel else 0.0   # capped: 1.0 = perfect within top-k

def precision_at_k(retrieved, relevant, k=K):
    rel = set(relevant)
    if not retrieved[:k]:
        return 0.0
    return sum(1 for sid in retrieved[:k] if sid in rel) / min(k, len(retrieved))

def mrr(retrieved, relevant):
    rel = set(relevant)
    for i, sid in enumerate(retrieved, start=1):
        if sid in rel:
            return 1.0 / i
    return 0.0


def main():
    with open(EVAL) as f:
        queries = json.load(f)

    # results[config][type] = list of per-query metric dicts
    results = defaultdict(lambda: defaultdict(list))

    for q in queries:
        plan = make_plan(q["query"], use_llm=False)   # deterministic planner for reproducibility
        for cfg_name, fn in CONFIGS.items():
            retrieved = fn(plan)
            m = {
                "recall@10": recall_at_k(retrieved, q["relevant"]),
                "precision@10": precision_at_k(retrieved, q["relevant"]),
                "mrr": mrr(retrieved, q["relevant"]),
            }
            results[cfg_name][q["type"]].append(m)

    # aggregate (mean per config per type, and overall)
    summary = {}
    for cfg, by_type in results.items():
        summary[cfg] = {}
        all_metrics = []
        for t, rows in by_type.items():
            agg = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}
            agg["n"] = len(rows)
            summary[cfg][t] = agg
            all_metrics.extend(rows)
        summary[cfg]["overall"] = {
            k: round(sum(r[k] for r in all_metrics) / len(all_metrics), 4)
            for k in all_metrics[0]
        }
        summary[cfg]["overall"]["n"] = len(all_metrics)

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # print table
    print("\n" + "=" * 64)
    print(f"{'CONFIG':22s} {'TYPE':14s} {'R@10':>6s} {'P@10':>6s} {'MRR':>6s} {'n':>4s}")
    print("=" * 64)
    for cfg in CONFIGS:
        for t in ["appearance", "single_state", "compound", "overall"]:
            if t in summary[cfg]:
                s = summary[cfg][t]
                print(f"{cfg:22s} {t:14s} {s['recall@10']:6.3f} "
                      f"{s['precision@10']:6.3f} {s['mrr']:6.3f} {s['n']:4d}")
        print("-" * 64)
    print(f"\n✅ Results saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
