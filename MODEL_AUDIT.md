# NeuralFabric - Model and System Audit

## Executive assessment

The project is technically coherent as a synthetic research prototype, but the original paper overclaimed what the implementation proves. The revised version narrows the scientific claims, fixes checkpoint portability, and aligns the paper with the measured benchmark.

## Model audit

| Item | Finding | Action |
|---|---|---|
| Architecture | 2-layer GCN + 1-layer LSTM + MLP | Kept; appropriate for current experiment scope |
| Parameters | 17,985 trainable parameters | Corrected in paper/docs |
| Input | 10 ticks of node/link telemetry | Kept |
| Forecast target | Congestion at 5 ticks after last observed feature | Explicitly documented |
| Label | Raw offered arrivals / capacity > 0.85 | Explicitly documented |
| Loss | Weighted BCE, positive weight 3 | Kept for benchmark compatibility |
| Optimizer | Adam, 1e-3 LR, 1e-4 weight decay | Kept |
| Early stopping | Validation F1 | Kept |
| Comparator | Persistence baseline | Added to interpretation and paper |
| Probability calibration | Not implemented | Paper now calls scores uncalibrated |
| Seed count | One recorded training seed | Reported as a limitation |

## Data audit

The dataset is split by complete scenario IDs rather than contiguous windows, preventing overlapping windows from crossing train/validation/test partitions. The recorded split is 21/4/5 scenarios. Held-out forecast windows still overlap within a scenario and therefore are not independent statistical trials.

The routing benchmark uses fresh seeds 3000--3002 for six workload families and five strategies, giving 90 routing runs.

## Simulator audit

The simulator is causal: packets are enqueued at the first hop and can advance at most one link per tick. Packet conservation is asserted for every benchmark run.

Important abstractions remain: one shared FIFO per physical link across both directions, packet-batch rather than packet-event simulation, 1 ms ticks, no PFC/ECN/RDMA, and no rule-installation delay.

## Controller audit

The guarded controller now rejects invalid and stale predictions, orders candidate flows by path risk, checks bounded alternate paths, enforces a minimum risk improvement, checks headroom on added links, sequentially updates reservations, and enforces cooldown and a per-tick change budget.

The capacity formula is a conservative admission proxy, not a physical bandwidth model.

## Checkpoint portability audit

The project includes `load_compatible_state_dict()` to translate checkpoint keys between PyTorch Geometric GCNConv weights and the bundled manual-GCN representation, including removal of the `_orig_mod.` prefix produced by `torch.compile`.

`torch.compile` is now opt-in through `NEURALFABRIC_TORCH_COMPILE=1`, preventing it from changing the default checkpoint layout.

## Verification performed

- 13/13 regression/integration tests pass in the current environment.
- `main.py --evaluate` loads the supplied checkpoint through the compatibility loader and completes the evaluation/demo workflow.
- The original recorded benchmark JSON and checkpoint were preserved rather than overwritten by an incompatible local environment.
- The full 90-run benchmark was **not rerun after the source edits in the current environment** because the environment lacks the tested PyTorch Geometric stack. The recorded benchmark remains the authoritative numerical artifact.
- The paper PDF was rebuilt and rendered for visual inspection.

## Priority improvements for the next research iteration

1. Multiple training seeds and at least 10--20 traffic seeds per workload family.
2. Held-out topology sizes, not only new traffic seeds.
3. Threshold selection on validation data.
4. Probability calibration and uncertainty-aware decisions.
5. Controller-delay and inference-tail sweeps.
6. Ablations for the capacity guard, cooldown, route-change budget, and improvement gate.
7. Packet-level/RDMA-aware simulation with full-duplex links, packet ordering, and lossless congestion mechanisms.
