# NeuralFabric - Start Here

## 1. What this project is

NeuralFabric is a GCN-LSTM forecasting + capacity-guarded routing experiment on a causal synthetic k=4 Fat-Tree simulator.

The correct way to describe it in a viva is:

> NeuralFabric forecasts which switch-to-switch links are likely to become congested and allows a routing controller to move flows only when the candidate path passes explicit risk, capacity, freshness, cooldown, and route-change constraints.

## 2. First command: verify the project

```powershell
python -m unittest discover -s tests -v
```

Expected: 13/13 tests pass.

## 3. Run the stored model

```powershell
python main.py --evaluate
```

Expected behavior: the checkpoint loads, the held-out classification report is printed, standard plots are refreshed, and the small routing comparison completes.

## 4. Launch the dashboard

```powershell
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## 5. Research benchmark

The supplied `research/benchmark_results.json` is the recorded 90-run pilot: six workload families, three paired seeds per family, and five routing strategies.

To regenerate it from scratch:

```powershell
python research_benchmark.py --epochs 20 --ticks 150 --seeds 3
python summarize_results.py
```

Use `requirements-tested.txt` for the closest reproduction.

## 6. What the evidence supports

- Guarded GNN vs ECMP: 5.94% mean drop reduction over 18 matched trials.
- Guarded GNN vs ECMP: 15 wins, 3 ties, 0 losses.
- Stride workload: 86.4% drop reduction.
- Reactive reroutes/run: 622.83.
- Guarded GNN reroutes/run: 14.72 (97.6% fewer mutations).
- GCN-LSTM F1: 0.7940.
- Persistence F1: 0.8118.

The persistence F1 result matters: do not claim the neural forecaster is universally better.

## 7. What must not be claimed

Do not say that the current project:

- reacts in microseconds;
- prevents PFC deadlocks;
- implements real RoCEv2/RDMA behavior;
- executes real LLM/NCCL training traffic;
- is production-ready; or
- is proven novel/patentable.

The simulator uses a 1 ms tick and does not model those mechanisms.

## 8. Files for the final submission

- `computer_networks.pdf`
- `computer_networks.tex`
- `explanation.pdf`
- `research/EXPERIMENT_RESULTS.md`
- `research/RESEARCH_AND_PATENT_REVIEW.md`
- `MODEL_AUDIT.md`
- `SUBMISSION_CHECKLIST.md`
- the source directories and stored checkpoint inside the final ZIP
