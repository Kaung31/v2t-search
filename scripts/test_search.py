"""End-to-end search test from the command line."""
from apps.api.search_api import run_search

QUERIES = [
    "distracted driver during a lane change",
    "driver on the phone while turning",
    "anxious driver changing lanes",
    "a truck on the road",
]

for q in QUERIES:
    print(f"\n{'='*70}\nQUERY: {q}\n{'='*70}")
    out = run_search(q, use_llm=False, limit=8)
    p = out["plan"]
    print(f"  plan: driver={p['driver_states']} road={p['road_states']} objects={p['object_classes']}")
    print(f"  {out['result_count']} results:")
    for r in out["results"]:
        layers = "+".join(r["matched_by"])
        print(f"    {r['sample_id']}  score={r['fused_score']:.4f}  [{layers}]")
