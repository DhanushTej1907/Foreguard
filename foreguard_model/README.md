# ForeGuard

**Capacity-guarded predictive routing on a synthetic datacenter fabric**

ForeGuard is a reproducible computer-networks research prototype. It combines a GCN-LSTM congestion forecaster with a deterministic rerouting controller that admits route changes only when predicted risk improves and the alternative path passes explicit capacity and stability checks.

## Repository layout

| Folder | Contents |
|---|---|
| project root (`simulation/`, `model/`, `routing/`, `dashboard/`, `research/`, ...) | The ForeGuard project: simulator, GCN-LSTM forecaster, guarded rerouting controller, dashboard, stored benchmark |
| `revision/` | Every experiment added during the paper's review rounds: scripts, raw results, analysis summaries and figures. See `revision/README.md` for which script backs which section and table of the paper |
| `ForeGuard_fully_resolved.docx` | The final revised paper resolving all teacher review concerns |
| `paper/changelogs/` | Edit-by-edit changelogs of the review rounds |

The numbers in the final paper come from `revision/results/`; the "Recorded research result"
table below is the original pilot benchmark. To rerun the revision experiments from the repo root:
`bash revision/scripts/run_all.sh` (about 1.5–2 h with two workers; finished tasks are skipped).

## What is actually implemented

The project is intentionally scoped as a synthetic simulator, not a production RoCEv2/PFC implementation.

- k=4 Fat-Tree: 20 switches, 16 hosts, 32 switch-to-switch links, 48 physical links.
- 1 ms simulation tick; 500 packets/tick link capacity; 2000-packet FIFO buffer.
- One shared FIFO per physical link across both directions.
- Stable SHA-256 flow hashing for ECMP across equal-cost shortest paths.
- Causal packet forwarding: a packet advances at most one link per tick.
- GCN-LSTM predictor: two GCN layers + one LSTM layer; 17,985 trainable parameters.
- Ten-tick input window; congestion target five ticks after the last observed feature.
- Guarded rerouting: 0.7 alert threshold, 0.15 minimum risk improvement, 0.90 headroom, at most 4 route changes/tick, 5-tick cooldown, maximum prediction age 1 tick, at most 3 candidate paths, and at most +2 hops.
- Packet-conservation assertion on every benchmark run.
- Local dashboard for demonstrating telemetry, predictions, traffic injection, and guarded routing.

## Recorded research result

The benchmark contains 30 data scenarios and 90 routing runs (6 workload families x 3 paired seeds x 5 strategies).

| Finding | Recorded value |
|---|---:|
| GCN-LSTM F1 @ 0.5 | 0.7940 |
| Persistence F1 @ 0.5 | 0.8118 |
| GCN-LSTM AP | 0.8692 |
| GCN-LSTM ROC AUC | 0.9580 |
| Mean guarded-GNN drops/run | 283,899.0 |
| Mean ECMP drops/run | 301,821.5 |
| Guarded vs ECMP mean drop reduction | 5.94% |
| Guarded vs ECMP matched trials | 15 wins / 3 ties / 0 losses |
| Guarded GNN stride drop reduction | 86.4% |
| Reactive reroutes/run | 622.83 |
| Guarded GNN reroutes/run | 14.72 |
| Reroute mutation reduction | 97.6% |

The GNN does **not** beat persistence on F1. The benefit is therefore attributed to the combination of forecasting and conservative admission control, not to an unconditional claim that the neural model is superior.

## Important limitations

The simulator does not implement PFC, ECN, DCQCN, RoCEv2 transport, RDMA packet ordering, retransmissions, NCCL/GPU execution, switch programming, or a measured real-time control loop. The 1 ms simulation tick cannot establish microsecond reaction time.

The recorded inference p99 is 6.185--13.516 ms on the benchmark host, excluding telemetry export, path computation, and route installation. Some inference tails exceed the 5 ms forecast horizon.

Three routing seeds per workload family and one recorded training seed make this a **pilot experiment**, not a final statistical validation.

## Installation

Use the tested environment for the closest reproduction.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-tested.txt
```

The tested dependency file includes PyTorch Geometric. The code also contains a bundled manual-GCN fallback and a compatible checkpoint loader for environments where PyG is unavailable.

## Run the tests

```powershell
python -m unittest discover -s tests -v
```

Expected: **13 tests, all passing**.

## Command-line evaluation

```powershell
python main.py --evaluate
```

This loads the stored checkpoint, evaluates the held-out predictor, generates the standard plots under `plots/`, and runs a small routing demonstration.

## Retrain / reproduce the research benchmark

```powershell
python research_benchmark.py --epochs 20 --ticks 150 --seeds 3
python summarize_results.py
```

`research_benchmark.py` regenerates the dataset, trains a model, and writes a new checkpoint and `research/benchmark_results.json`. It is intentionally separate from the compact demo workflow in `main.py`.

The recorded benchmark already shipped with this project should be treated as the authoritative pilot artifact unless the full benchmark is deliberately rerun in the tested environment.

## Dashboard

```powershell
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. Run the server as a single local worker so the shared simulation clock remains deterministic.

## Project structure

```text
simulation/      topology, traffic, causal FIFO network engine, telemetry
model/           GCN-LSTM, datasets, grouped splits, training, evaluation
routing/         reactive, unguarded GNN, and capacity-guarded rerouting
dashboard/       local FastAPI/WebSocket demonstration
research/        benchmark data, plots, research review, experiment notes
checkpoints/     trained checkpoint + training history
tests/           simulator, dataset, controller, dashboard regressions
```

## Submission files

- `computer_networks.pdf` - revised IEEE-style research paper.
- `computer_networks.tex` - paper source.
- `explanation.pdf` - technical project explanation / viva preparation.
- `research/EXPERIMENT_RESULTS.md` - measured pilot results and limits.
- `research/RESEARCH_AND_PATENT_REVIEW.md` - current technical/prior-art screen.
- `MODEL_AUDIT.md` - implementation and evidence audit.
- `SUBMISSION_CHECKLIST.md` - final submission and demonstration checklist.

## Research position

The defensible claim is narrow: **capacity and stability guards make forecast-driven rerouting more controlled in the synthetic benchmark.** The project does not establish state-of-the-art performance, production readiness, microsecond control latency, PFC-deadlock prevention, or patentability.
