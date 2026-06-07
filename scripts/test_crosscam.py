"""Test the cross-camera correspondence join with real query examples."""
from apps.api.retrieval.crosscam import find_correspondences

EXAMPLES = [
    ("Distracted during a lane change", ["looking_around"], ["changing_lane"]),
    ("Phone use while turning",          ["making_phone"],   ["turning"]),
    ("Anxious driver changing lanes",    ["anxiety"],        ["changing_lane"]),
    ("Drowsy in a traffic jam",          ["dozing_off", "weariness"], ["traffic_jam"]),
    ("Any distraction during any maneuver",
        ["looking_around", "making_phone", "smoking", "dozing_off"],
        ["changing_lane", "turning"]),
]

for name, driver, road in EXAMPLES:
    hits = find_correspondences(driver_states=driver, road_states=road, limit=100)
    print(f"\n=== {name} ===")
    print(f"    driver={driver}  road={road}")
    print(f"    {len(hits)} matching clips")
    for h in hits[:5]:
        print(f"      sample {h.sample_id}: {h.driver_state} + {h.road_state} (score {h.score:.2f})")
