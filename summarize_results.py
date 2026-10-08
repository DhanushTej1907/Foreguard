"""Render a clean Markdown summary from the saved benchmark JSON without rerunning it."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).parent / "research"
d = json.loads((root / "benchmark_results.json").read_text(encoding="utf-8"))
strategies = ["ecmp", "reactive", "persistence", "gnn", "guarded_gnn"]
patterns = list(dict.fromkeys(r["pattern"] for r in d["runs"]))

lines = [
    "# NeuralFabric - Recorded Pilot Experiment Results",
    "",
    "Generated from `benchmark_results.json`. This is the recorded pilot benchmark, not a claim of publication-level statistical significance.",
    "",
    f"Configuration: {d['settings']['epochs']} training epochs; {d['settings']['ticks']} ticks/run; {d['settings']['seeds']} routing seeds/family; k=4; 1 ms tick.",
    "",
    "## Benchmark configuration",
    "",
    "| Setting | Value |",
    "|---|---:|",
    f"| Data scenarios | {len(d['scenarios'])} |",
    f"| Train / val / test scenarios | {len(d['splits']['train'])} / {len(d['splits']['val'])} / {len(d['splits']['test'])} |",
    f"| Workload families | {len(patterns)} |",
    f"| Routing strategies | {len(strategies)} |",
    f"| Routing runs | {len(d['runs'])} |",
    "| Topology | k=4 Fat-Tree |",
    "| Simulation tick | 1 ms |",
    "| Link capacity | 500 packets/tick |",
    "| Buffer | 2000 packets |",
    "",
    "## Forecast quality on held-out scenario windows",
    "",
    "| Predictor | F1 @ 0.5 | Average precision | ROC AUC | Brier score |",
    "|---|---:|---:|---:|---:|",
]
for name, m in d["prediction"].items():
    lines.append(f"| {name} | {m['f1']:.4f} | {m['average_precision']:.4f} | {m['roc_auc']:.4f} | {m['brier']:.4f} |")

lines += [
    "",
    f"Positive-label prevalence: {d['prediction']['gnn']['positive_fraction']:.4f}. Windows within a held-out scenario overlap and are not independent statistical trials.",
    "Persistence holds the last observed congestion label constant. The GNN has stronger AP, ROC AUC, and Brier score but lower F1 at the fixed 0.5 cutoff.",
    "",
    "## Mean packet drops per run",
    "",
    "Lower is better; each family mean is over matched routing seeds.",
    "",
    "| Traffic family | ECMP | Reactive | Persistence | GNN | Guarded GNN |",
    "|---|---:|---:|---:|---:|---:|",
]
for pattern in patterns:
    row = [np.mean([r["total_drops"] for r in d["runs"] if r["pattern"] == pattern and r["strategy"] == s]) for s in strategies]
    lines.append("| " + pattern + " | " + " | ".join(f"{v:,.0f}" for v in row) + " |")

lines += [
    "",
    "## Guarded GNN vs ECMP by family",
    "",
    "| Family | Mean drop reduction |",
    "|---|---:|",
]
for pattern in patterns:
    e = np.mean([r["total_drops"] for r in d["runs"] if r["pattern"] == pattern and r["strategy"] == "ecmp"])
    g = np.mean([r["total_drops"] for r in d["runs"] if r["pattern"] == pattern and r["strategy"] == "guarded_gnn"])
    lines.append(f"| {pattern} | {(1-g/e)*100:.2f}% |")

lines += [
    "",
    "## Overall routing outcomes",
    "",
    "| Strategy | Drops/run | Delivered pkt/tick | Mean delivered delay (ms) | Mean run p99 delay (ms) | Reroutes/run |",
    "|---|---:|---:|---:|---:|---:|",
]
for s in strategies:
    m = d["overall_summary"][s]
    lines.append(f"| {s} | {m['mean_total_drops']:,.1f} | {m['mean_delivered_packets_per_tick']:,.2f} | {m['mean_delivered_latency_us']/1000:.3f} | {m['mean_run_p99_latency_us']/1000:.3f} | {m['mean_reroutes_per_run']:,.2f} |")

lines += [
    "",
    "### Guarded GNN vs ECMP matched-trial summary",
    "",
    f"{d['guarded_vs_ecmp']['wins']} wins, {d['guarded_vs_ecmp']['ties']} ties, {d['guarded_vs_ecmp']['losses']} losses across {d['guarded_vs_ecmp']['matched_trials']} matched trials.",
    f"Mean packet-drop reduction: {d['guarded_vs_ecmp']['mean_drop_reduction_percent']:.2f}%. Mean difference: {d['guarded_vs_ecmp']['mean_drop_difference']:,.1f} packets/run.",
    f"Reroute mutation reduction vs reactive: {d['guarded_vs_ecmp']['reroute_mutation_reduction_vs_reactive_percent']:.2f}%.",
    "",
    "## Paired descriptive intervals",
    "",
    "Bootstrap resampling uses only three paired seeds per family; intervals are descriptive rather than definitive significance findings.",
    "",
    "| Family | Guarded GNN mean difference vs ECMP | 95% paired bootstrap interval |",
    "|---|---:|---|",
]
for row in d["paired_summary"]:
    if row["strategy"] == "guarded_gnn":
        lo, hi = row["paired_bootstrap_95_percent_interval"]
        lines.append(f"| {row['pattern']} | {row['mean_drop_difference']:,.0f} | [{lo:,.0f}, {hi:,.0f}] |")

timings = [r["inference_ms_p99"] for r in d["runs"] if r["inference_ms_p99"] is not None]
lines += [
    "",
    "## Interpretation and limits",
    "",
    f"All {len(d['runs'])} routing runs satisfy `offered = dropped + delivered + outstanding`.",
    f"Per-inference p99 spans {min(timings):.3f}-{max(timings):.3f} ms on the benchmark host. Timings exclude telemetry export, path computation, and route installation, and the simulator does not wait for inference.",
    "A 5-tick forecast is 5 ms at the configured resolution. The experiment does not establish microsecond response, PFC-deadlock prevention, RDMA completion-time benefit, or production deployment feasibility.",
    "The LLM-training traffic is synthetic collective-inspired traffic, not real NCCL/GPU trace replay.",
    "The benchmark uses one recorded training seed, one topology size, and three routing seeds per family. More independent seeds, held-out topologies, delay sweeps, ablations, and an RDMA-aware simulator are required for stronger claims.",
    "",
    "![Measured packet drops](benchmark_drops.png)",
]
(root / "EXPERIMENT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Wrote research/EXPERIMENT_RESULTS.md")
