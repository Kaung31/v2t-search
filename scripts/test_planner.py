"""Test both planner modes on natural-language queries."""
from apps.api.planner import make_plan

QUERIES = [
    "show me distracted drivers during lane changes",
    "driver on the phone while turning",
    "anxious driver in heavy traffic",
    "someone dozing off at a red light",
    "a red truck on the road",
    "driver looking around when a pedestrian is nearby",
]

for q in QUERIES:
    print(f"\n=== {q!r} ===")
    p = make_plan(q, use_llm=False)
    print(f"  [rule] driver={p.driver_states}  road={p.road_states}  objects={p.object_classes}")
