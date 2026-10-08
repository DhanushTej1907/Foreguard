# NeuralFabric — context for Claude Code

GCN-LSTM link-congestion forecaster + capacity-guarded rerouting controller on a synthetic
k=4 Fat-Tree simulator (1 ms ticks, 500 pkts/tick links, 2000-pkt buffers). Academic CN
project + paper (`computer_networks.tex`). Windows, Python 3.12, torch 2.14 CPU, PyG installed.

## Layout
- `simulation/` topology, traffic_generator, network_engine (causal batch sim), telemetry (features/labels)
- `model/` gnn_model (GCN-LSTM, 17,985 params), train, dataset, splits (by scenario_id), evaluate
- `routing/` rerouter.py (reactive + unguarded proactive), guarded.py (GuardedRerouter), comparator.py
- `research_benchmark.py` → `research/benchmark_results.json`; `summarize_results.py` → `research/EXPERIMENT_RESULTS.md`
- `dashboard/` FastAPI + WebSocket server, static D3/Chart.js frontend
- `tests/` unittest suite: `python -m unittest discover -s tests -v`

## Known findings (verified by independent rerun)
- Simulator is deterministic: ECMP drops reproduce the recorded JSON exactly.
- **Guard-only ablation:** GuardedRerouter fed last-tick persistence labels gives 6.03% drop
  reduction vs ECMP, fed current utilization 6.10%; with GNN predictions 5.94%. The guard, not the
  GNN, produces the routing gain. See `tools/guard_ablation_check.py`.
- GNN F1 0.794 < persistence 0.812. Continuous persistence (current util as score) AP 0.825 /
  AUC 0.951 vs GNN 0.869 / 0.958. Recorded "persistence AP" uses a 0/1 score, which is unfair.
- Inference p99 6–13.5 ms > 5 ms forecast horizon.
- Onsets (not congested now → congested at t+5) are ~19% of test positives; no onset metric exists.

## Working rules
- Ask before large refactors; keep each phase a separate git commit.
- Never overwrite `research/benchmark_results.json` or `checkpoints/best_model.pt` without being
  told; write new results to new filenames (e.g. `benchmark_results_v4.json`).
- Any change to simulator/features/labels invalidates the dataset and checkpoint: bump the
  checkpoint `schema_version` and say so.
- Add or update a unit test for every behavioural change; run the full test suite before finishing.
- Keep claims honest: the paper must not say the GNN helps unless the benchmark shows it beats
  the guard-only baselines.