"""Generate research figures directly from research/benchmark_results.json."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R = ROOT / "research"
D = json.loads((R / "benchmark_results.json").read_text(encoding="utf-8"))

patterns = ["uniform", "hotspot", "elephant_mice", "stride", "llm_training", "mixed"]
strategies = ["ecmp", "reactive", "persistence", "gnn", "guarded_gnn"]
labels = {"ecmp":"ECMP", "reactive":"Reactive", "persistence":"Persistence", "gnn":"GNN", "guarded_gnn":"Guarded GNN"}

# 1) Predictor metrics with separate scales for probability/ranking metrics.
metric_names = ["Average precision", "ROC AUC", "Brier score"]
gnn = D["prediction"]["gnn"]
pers = D["prediction"]["persistence"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
for ax, name, vals in [
    (axes[0], "Ranking quality", [(gnn["average_precision"], pers["average_precision"]), (gnn["roc_auc"], pers["roc_auc"])]),
    (axes[1], "Probability error", [(gnn["brier"], pers["brier"])])]:
    x = np.arange(len(vals))
    width = 0.34
    a = [v[0] for v in vals]
    b = [v[1] for v in vals]
    cats = ["Avg. Precision", "ROC AUC"] if len(vals) == 2 else ["Brier score"]
    ax.bar(x - width/2, a, width, label="GCN-LSTM")
    ax.bar(x + width/2, b, width, label="Persistence")
    ax.set_xticks(x, cats)
    ax.set_title(name)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False)
fig.suptitle("Held-out congestion forecast quality")
fig.savefig(R / "metrics_chart.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 2) Packet-drop reduction versus ECMP for each workload family.
reductions = []
for pattern in patterns:
    e = np.mean([x["total_drops"] for x in D["runs"] if x["pattern"] == pattern and x["strategy"] == "ecmp"])
    g = np.mean([x["total_drops"] for x in D["runs"] if x["pattern"] == pattern and x["strategy"] == "guarded_gnn"])
    reductions.append(100 * (1 - g/e) if e else 0.0)
fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
y = np.arange(len(patterns))
ax.barh(y, reductions)
ax.axvline(0, linewidth=1)
ax.set_yticks(y, [p.replace("_", " ").title() for p in patterns])
ax.set_xlabel("Mean packet-drop reduction vs ECMP (%)")
ax.set_title("Guarded GNN effect across six traffic families")
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)
for yy, v in zip(y, reductions):
    ax.text(v + 1.0, yy, f"{v:.1f}%", va="center", fontsize=9)
fig.savefig(R / "drop_reduction.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 3) Mean reroute mutations.
rvals = [np.mean([x["reroutes"] for x in D["runs"] if x["strategy"] == s]) for s in strategies]
fig, ax = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
x = np.arange(len(strategies))
bars = ax.bar(x, rvals)
ax.set_xticks(x, [labels[s] for s in strategies], rotation=18)
ax.set_ylabel("Mean reroute mutations / run")
ax.set_title("Routing control-plane churn")
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
for b, v in zip(bars, rvals):
    ax.text(b.get_x()+b.get_width()/2, v + max(rvals)*0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
fig.savefig(R / "reroutes_chart.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 4) Delivered packet delay (means of per-run statistics).
mean_lat = [np.mean([x["delivered_mean_latency_us"] for x in D["runs"] if x["strategy"] == s]) / 1000.0 for s in strategies]
p99_lat = [np.mean([x["delivered_p99_latency_us"] for x in D["runs"] if x["strategy"] == s]) / 1000.0 for s in strategies]
fig, ax = plt.subplots(figsize=(8.1, 4.6), constrained_layout=True)
x = np.arange(len(strategies)); width = 0.36
ax.bar(x-width/2, mean_lat, width, label="Mean delivered-packet delay")
ax.bar(x+width/2, p99_lat, width, label="Mean per-run p99 delivered delay")
ax.set_xticks(x, [labels[s] for s in strategies], rotation=18)
ax.set_ylabel("Delay (ms)")
ax.set_title("Delivered-packet delay on the synthetic simulator")
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8)
fig.savefig(R / "latency_chart.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 5) Benchmark drops grouped by family and strategy.
vals = np.array([[np.mean([x["total_drops"] for x in D["runs"] if x["pattern"] == p and x["strategy"] == s]) for s in strategies] for p in patterns])
fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
x = np.arange(len(patterns)); width = 0.15
for i, s in enumerate(strategies):
    ax.bar(x + (i-2)*width, vals[:,i], width, label=labels[s])
ax.set_xticks(x, [p.replace("_", " ").title() for p in patterns], rotation=18)
ax.set_ylabel("Mean total packet drops / run")
ax.set_title("Routing outcomes across matched traffic seeds")
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
ax.legend(ncol=3, frameon=False, fontsize=8)
fig.savefig(R / "benchmark_drops.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Research figures regenerated from benchmark_results.json")
