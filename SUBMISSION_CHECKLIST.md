# NeuralFabric - Submission Checklist

## Before uploading the project

- [x] Research paper rebuilt as `computer_networks.pdf`.
- [x] Paper source updated in `computer_networks.tex`.
- [x] Technical explanation rebuilt as `explanation.pdf`.
- [x] 13/13 regression/integration tests passing.
- [x] Stored checkpoint loads through the portable compatibility loader.
- [x] `main.py --evaluate` completes.
- [x] Benchmark data preserved in `research/benchmark_results.json`.
- [x] Figures regenerated from benchmark data.
- [x] Research/prior-art review updated.
- [x] Limitations are explicitly stated.
- [x] Old overclaims about microseconds, PFC deadlocks, state-of-the-art performance, and patentability removed.

## Commands to demonstrate

```powershell
python -m unittest discover -s tests -v
python main.py --evaluate
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8000
```

## What to say in the viva

**Problem:** bursty traffic can create localized queue pressure under static multipath routing.

**Model:** a GCN captures spatial topology relationships and an LSTM captures temporal evolution in link telemetry.

**Controller:** the forecast is only a trigger; the capacity guard decides whether a route change is admissible.

**Best result:** stride workload drops fell from about 59.8k under ECMP to 8.1k under Guarded GNN (86.4% reduction).

**System-level result:** across 18 matched routing trials, Guarded GNN had 5.94% fewer mean drops than ECMP and used 97.6% fewer route mutations than reactive routing.

**Important negative result:** persistence has higher F1 than GCN-LSTM at the fixed 0.5 threshold, so the project does not prove that the neural predictor is universally better.

**Limitation:** this is a 1 ms-tick synthetic simulator with no PFC/RDMA/transport ordering or real switch programming delay.

## Submission discipline

Do not change author names, emails, measured values, or benchmark settings at the last minute without rerunning and rechecking the artifacts. Do not describe synthetic LLM-training traffic as real NCCL/GPU traces.
