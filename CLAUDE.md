# NeuralFabric — context for Claude Code

GCN-LSTM link-congestion forecaster + capacity-guarded rerouting controller on a synthetic
k=4 Fat-Tree simulator (1 ms ticks, 500 pkts/tick links, 2000-pkt buffers). Academic CN
project + paper (`computer_networks.tex`). Windows, Python 3.14, torch 2.14 CPU, PyG installed.
On 3.14, the `torch.jit.script` FutureWarning and PyG `typing._eval_type` DeprecationWarning are
expected and harmless.

## Layout
- `simulation/` topology, traffic_generator, network_engine (causal batch sim, full-duplex: one
  FIFO per direction, `link_states` keyed by directed `(u, v)`), telemetry (features/labels; per
  physical link = max over both directions, still 32 links)
- `config.py`: `CHECKPOINT_SCHEMA_VERSION` (currently 3; checked by `model.gnn_model.check_checkpoint_schema`),
  `MODEL["label_mode"]` = `"offered"` (default) | `"served_or_queued"`
- `engine.metrics()` exports `host_link_drops + fabric_link_drops + unroutable_drops == total_drops` (`drop_split_ok`)
- `model/` gnn_model (GCN-LSTM, 17,985 params), train, dataset, splits (by scenario_id), evaluate
- `routing/` rerouter.py (reactive + unguarded proactive), guarded.py (GuardedRerouter), comparator.py
- `research_benchmark.py` → `research/benchmark_results.json`; `summarize_results.py` → `research/EXPERIMENT_RESULTS.md`
- `dashboard/` FastAPI + WebSocket server, static D3/Chart.js frontend
- `tests/` unittest suite: `python -m unittest discover -s tests -v`

## Known findings — schema 2 (pre-Phase-1 shared-FIFO simulator)
These were measured on the old simulator and recorded JSON; after Phase 1 (full-duplex) they must
be re-verified. See `reports/PHASE1_REPORT.md` for the post-Phase-1 numbers (no retrained GNN yet;
`checkpoints/best_model.pt` is schema 2 and is rejected).
- Simulator is deterministic: ECMP drops reproduce the recorded JSON exactly (v2 simulator only).
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