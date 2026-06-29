"""Print a few queries and a sample of their ground-truth clips, to verify by eye."""
import json
import random

with open("eval/eval_queries.json") as f:
    queries = json.load(f)

# pick 2 of each type
by_type = {}
for q in queries:
    by_type.setdefault(q["type"], []).append(q)

for t, qs in by_type.items():
    for q in random.sample(qs, min(2, len(qs))):
        sample = random.sample(q["relevant"], min(3, len(q["relevant"])))
        print(f"\n[{t}] \"{q['query']}\"  ({len(q['relevant'])} relevant)")
        print("  spot-check these clips:")
        for sid in sample:
            print(f"    open data/aide/AIDE_Dataset/{sid}/incar.mp4 "
                  f"data/aide/AIDE_Dataset/{sid}/front.mp4")
