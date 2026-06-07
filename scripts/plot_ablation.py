"""Render the headline ablation chart: Recall@10 by query type, per config."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("eval/ablation_results.json") as f:
    summary = json.load(f)

types = ["appearance", "single_state", "compound", "overall"]
type_labels = ["Appearance", "Single-state", "Compound", "Overall"]
configs = ["semantic_only", "correspondence_only", "fused"]
config_labels = ["Semantic only", "Correspondence only", "Fused (full system)"]
colors = ["#45d3e6", "#ff9e3d", "#58d68d"]

x = np.arange(len(types))
width = 0.26

fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0a0d12")
ax.set_facecolor("#0a0d12")

for i, (cfg, label, color) in enumerate(zip(configs, config_labels, colors)):
    vals = [summary[cfg].get(t, {}).get("recall@10", 0) for t in types]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, edgecolor="#0a0d12")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", color="#e6eaf2", fontsize=9, family="monospace")

ax.set_ylabel("Recall@10", color="#e6eaf2", fontsize=12)
ax.set_title("V2T-Search — Retrieval ablation by query type",
             color="#e6eaf2", fontsize=15, fontweight="bold", pad=18)
ax.set_xticks(x)
ax.set_xticklabels(type_labels, color="#e6eaf2", fontsize=11)
ax.set_ylim(0, 1.1)
ax.tick_params(colors="#788396")
ax.legend(facecolor="#11161f", edgecolor="#232d3d", labelcolor="#e6eaf2", fontsize=10)
ax.grid(axis="y", color="#232d3d", linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color("#232d3d")

plt.tight_layout()
plt.savefig("eval/ablation_chart.png", dpi=160, facecolor="#0a0d12")
print("✅ Saved eval/ablation_chart.png")
