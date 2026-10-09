# Phase 3 report: full benchmark (seeds, calibration, routing)

Branch `fix/phase3`. Code commit `246810e`; results in `research/benchmark_results_v4.json`.
Date: 2026-10-09. Checkpoint schema 3, `label_mode = served_or_queued`; results-JSON schema 4.

## Verdict

1. **Forecasting:**
   - **The learned models beat persistence in ranking quality.** GNN overall AP is
     0.947 ± 0.003, against 0.880 for continuous persistence: Δ +0.067, 95% scenario-bootstrap
     CI [+0.040, +0.131].
   - **The GCN adds nothing.** GNN − LSTM-only is +0.001 [−0.003, +0.014] overall and
     −0.007 [−0.022, +0.024] on onsets. LSTM-only has a slightly higher F1 (0.900 vs 0.893) and
     a better Brier score (0.041 vs 0.049).
2. **Onsets:**
   - **Every predictor is still weak on onsets.** Onset AP is 0.18–0.19 for the learned models
     and 0.163 for continuous persistence.
   - **The GNN's onset edge over persistence is marginal:** +0.021, CI [+0.001, +0.040], barely
     above zero.
   - At its calibrated threshold the GNN catches 7% of onsets at 33% precision.
3. **Routing:**
   - **The GNN does not beat the guard-only baselines.** guarded_gnn cuts total drops 5.27%
     vs ECMP. The model-free guarded_score cuts 5.72%, guarded_lstm 5.37%, guarded_persistence
     4.86%.
   - No paired difference between guarded_gnn and a guard-only baseline or guarded_lstm is
     significant: every 95% CI includes 0. Against guarded_score, the point estimate favours the
     baseline.
   - **The gain comes from the guard, not the forecaster.** Per CLAUDE.md, the paper must not
     claim the GNN improves routing.
4. **Inference is within the 5 ms horizon on this laptop:**
   - GNN p50 2.0 ms, p99 4.1 ms, 0.45% of ticks over 5 ms.
   - LSTM-only p99 1.5 ms.
   - These cover inference only (see Limitations).

What the paper can honestly claim:
- A capacity guard turns a simple congestion signal into a 5–6% total and about 20–25% fabric
  drop reduction vs ECMP, with about 10× fewer route changes than reactive rerouting.
- Learned per-link temporal models rank congestion better than persistence.

It cannot claim that the graph neural network specifically helps.

---

## Setup

```
python research_benchmark.py --epochs 50 --train-seeds 3 --seeds 10 --ticks 150
```
Wall clock: 31.5 min (forecast 13.3 min, routing 18.2 min), against the 4 h limit, so nothing
was reduced.

**Data and split:**
- 30 scenarios (6 families × 5, seeds 42–71, 150 ticks).
- **Stratified split, 3/1/1 per family** (18/6/6).
  - Train scenarios: 0–5, 7, 9, 14–16, 18, 23–26, 28, 29.
  - Validation: 6, 8, 10, 13, 17, 21.
  - Test: 11, 12, 19, 20, 22, 27.

**Training:**
- Max 50 epochs. Early stopping and best epoch on validation AP (patience 10); LR scheduler
  patience 5.
- 3 training seeds (42, 43, 44) for the GNN and LSTM-only.
- Temperature scaling fitted on validation; the decision threshold is then re-tuned (max F1) on
  the calibrated validation probabilities.

**Routing:**
- 6 families × 10 traffic seeds (3000–3009), 150 ticks.
- **Model strategies** (`gnn`, `guarded_gnn`, `guarded_lstm`) **ran once per training seed with
  that seed's models**, so all 3 seeds and not only seed 0 (your amendment). Each used its own
  calibrated threshold.
- **Non-model strategies ran once:** `ecmp`, `reactive`, `guarded_persistence` (last label) and
  `guarded_score` (continuous score, at continuous persistence's validation threshold 0.851).
- **Seed handling:** the ± values in the routing table are std across training seeds of each
  seed's mean drops. Paired bootstraps use **seed-averaged** drops per (family, traffic seed)
  pair, 60 pairs.
- That makes 780 routing runs: 60 × (4 non-model + 3 model strategies × 3 seeds). All of them
  passed the conservation and drop-split assertions.

Every ± is a sample std (ddof = 1) across the 3 training seeds. Logistic regression and both
persistence baselines are deterministic and shown as single values.

## Forecasting

**Overall** (test set: 26,112 link-windows, 6,094 positives, rate 0.233)

| Predictor | Threshold | F1 | Precision | Recall | AP | AUC | Brier (calibrated) | Brier (raw) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GNN (GCN-LSTM) | 0.796 ± 0.047 | 0.893 ± 0.004 | 0.909 ± 0.008 | 0.877 ± 0.007 | **0.947 ± 0.003** | **0.976 ± 0.001** | 0.049 ± 0.002 | 0.049 ± 0.003 |
| LSTM-only | 0.767 ± 0.019 | **0.900 ± 0.001** | **0.929 ± 0.002** | 0.873 ± 0.001 | 0.946 ± 0.000 | 0.972 ± 0.000 | **0.041 ± 0.001** | 0.042 ± 0.001 |
| Per-link LogReg | 0.282 | 0.865 | 0.838 | 0.894 | 0.938 | 0.968 | 0.046 | — |
| Persistence (binary) | 0.500 | 0.887 | 0.887 | 0.886 | 0.813 | 0.926 | 0.053 | — |
| Persistence (continuous) | 0.851 | 0.887 | 0.887 | 0.886 | 0.880 | 0.961 | 0.130* | — |

\* Computed on the clipped score, which is not a calibrated probability. Continuous persistence's
validation threshold landed at 0.851, so its decisions equal binary persistence's.

**Onset-only** (link-windows not congested at the last observed tick: 20,021 windows, 692
positives, rate 0.035)

| Predictor | F1 | Precision | Recall | AP | AUC | Brier (calibrated) | Brier (raw) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GNN (GCN-LSTM) | 0.114 ± 0.014 | 0.331 ± 0.007 | 0.069 ± 0.010 | 0.184 ± 0.000 | **0.846 ± 0.002** | 0.040 ± 0.001 | 0.041 ± 0.002 |
| LSTM-only | 0.070 ± 0.004 | **0.386 ± 0.004** | 0.039 ± 0.002 | **0.192 ± 0.003** | 0.807 ± 0.004 | 0.033 ± 0.000 | 0.034 ± 0.000 |
| Per-link LogReg | **0.243** | 0.246 | **0.240** | 0.180 | 0.791 | **0.032** | — |
| Persistence (binary) | 0.000 | — | 0.000 | 0.035 | 0.500 | 0.035 | — |
| Persistence (continuous) | 0.000 | — | 0.000 | 0.163 | 0.789 | 0.137 | — |

Logistic regression's higher onset F1 comes from its much lower decision threshold (0.282). At
that threshold it trades overall precision (0.838) for onset recall.

**Scenario bootstrap**
- Seed-averaged AP, 6 test scenarios resampled with replacement, 2,000 draws, all valid.

| Comparison | Overall Δ AP [95% CI] | Onset Δ AP [95% CI] |
|---|---|---|
| GNN − continuous persistence | **+0.067 [+0.040, +0.131]** | **+0.021 [+0.001, +0.040]** |
| GNN − LSTM-only | +0.001 [−0.003, +0.014] | −0.007 [−0.022, +0.024] |

With 6 test scenarios, these intervals describe scenario-to-scenario variation. They are not a
population-level significance test.

**Training and calibration per seed**

| Model | Seed | Best epoch / epochs run | Best val AP | Temperature | Val Brier raw → calibrated | Threshold | Train (s) |
|---|---:|---:|---:|---:|---|---:|---:|
| GNN | 42 | 15 / 25 | 0.9342 | 0.840 | 0.0579 → 0.0573 | 0.788 | 104.7 |
| GNN | 43 | 17 / 27 | 0.9328 | 0.883 | 0.0577 → 0.0570 | 0.846 | 133.8 |
| GNN | 44 | 37 / 47 | 0.9347 | 0.980 | 0.0536 → 0.0535 | 0.754 | 234.5 |
| LSTM-only | 42 | 48 / 50 | 0.9326 | 0.808 | 0.0456 → 0.0444 | 0.781 | 64.4 |
| LSTM-only | 43 | 49 / 50 | 0.9319 | 0.786 | 0.0447 → 0.0431 | 0.745 | 64.9 |
| LSTM-only | 44 | 50 / 50 | 0.9329 | 0.809 | 0.0445 → 0.0432 | 0.774 | 72.6 |

**Calibration had almost no effect.**
- Every temperature is below 1 (0.79–0.98), meaning the models were slightly *under*-confident.
- Brier improves by at most 0.0016 on validation and about 0.001 on test.
- Temperature scaling only rescales logits, so it cannot remove the upward shift that
  `pos_weight = 3` puts into the probabilities. The GNN's remaining Brier gap to LSTM-only
  (0.049 vs 0.041) is not a calibration-scale problem.
- AP and AUC are unchanged by construction.

## Routing

Mean per run over 6 families × 10 traffic seeds.

| Strategy | Total drops | Fabric drops | Host drops | Total vs ECMP | Fabric vs ECMP | W/T/L vs ECMP | Reroutes/run |
|---|---:|---:|---:|---:|---:|---|---:|
| ecmp | 218,550 | 67,581 | 150,969 | — | — | — | 0 |
| reactive | 209,470 | 56,056 | 153,414 | −4.15% | −17.05% | 41/7/12 | 547.5 |
| gnn (unguarded) | 206,497 ± 202 | **49,697 ± 462** | 156,800 ± 260 | −5.51% | **−26.46%** | 43/4/13 | 585.6 |
| guarded_gnn | 207,040 ± 676 | 52,050 ± 791 | 154,990 ± 231 | −5.27% | −22.98% | 47/10/3 | 58.9 |
| guarded_lstm | 206,816 ± 247 | 51,687 ± 196 | 155,128 ± 117 | −5.37% | −23.52% | 46/10/4 | 47.8 |
| guarded_persistence | 207,929 | 54,210 | 153,718 | −4.86% | −19.79% | 45/10/5 | 47.8 |
| guarded_score | **206,049** | 50,790 | 155,259 | **−5.72%** | −24.85% | **48/10/2** | **37.9** |

The ± values are std across the 3 training seeds' mean drops. W/T/L counts the 60 seed-averaged
(family, traffic seed) pairs against ECMP.

**Paired bootstrap: guarded_gnn − baseline**
- Seed-averaged drops, 60 pairs, 2,000 draws.
- A negative difference means guarded_gnn dropped fewer packets.

| Baseline | Metric | Mean difference [95% CI] | Relative | W/T/L for guarded_gnn |
|---|---|---|---:|---|
| guarded_persistence | total drops | −889 [−2,863, +878] | −0.43% | 19/25/16 |
| guarded_persistence | fabric drops | −2,160 [−4,748, +73] | −3.99% | 20/30/10 |
| guarded_score | total drops | +990 [−463, +2,500] | +0.48% | 14/24/22 |
| guarded_score | fabric drops | +1,259 [−559, +3,106] | +2.48% | 15/28/17 |
| guarded_lstm | total drops | +224 [−751, +1,205] | +0.11% | 14/23/23 |
| guarded_lstm | fabric drops | +362 [−898, +1,833] | +0.70% | 17/29/14 |
| ecmp | total drops | **−11,510 [−15,086, −8,483]** | −5.27% | 47/10/3 |
| ecmp | fabric drops | **−15,531 [−20,217, −11,008]** | −22.98% | 46/10/4 |

**Per-family mean total drops** (seed-averaged)

| Family | ecmp | reactive | gnn | guarded_gnn | guarded_lstm | guarded_persistence | guarded_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 33,890 | 22,477 | 24,739 | 22,792 | 22,719 | 22,211 | **22,144** |
| hotspot | 278,899 | 269,573 | 269,469 | 263,726 | 264,105 | 266,308 | **261,996** |
| elephant_mice | 76,963 | **53,416** | 53,901 | 55,565 | 56,456 | 56,007 | 55,061 |
| stride | 8,916 | **0** | 753 | **0** | **0** | **0** | **0** |
| llm_training | **24,684** | 24,725 | 24,712 | **24,684** | **24,684** | **24,684** | **24,684** |
| mixed | 887,949 | 886,633 | **865,410** | 875,471 | 872,929 | 878,360 | 872,411 |

### What the routing numbers say

- **The guard does the work.** Every guarded variant lands at −4.9% to −5.7% total and −20% to
  −25% fabric drops. Which signal it is fed matters less than the guard itself. The model-free
  continuous score is at least as good as the GNN.
- **Rerouting moves loss from fabric links to host links.** Host-link drops rise from 150,969
  (ECMP) to 153–157k for every rerouting strategy. Traffic that is no longer dropped in the
  fabric reaches the receiver's edge→host link and is dropped there. So fabric reductions of
  20–26% become total reductions of only 4–6%.
- **The unguarded GNN removes the most fabric drops (−26%)**, but it needs about 10× more route
  changes (586 per run vs 59). It also loses to ECMP in 13 of 60 pairs, against 3 for
  guarded_gnn.
- **In elephant_mice the more aggressive strategies win; in mixed, only the unguarded GNN does.**
  In elephant_mice, reactive (53,416) and the unguarded GNN (53,901) beat every guarded variant
  (55,061–56,456 drops per run). In mixed, the unguarded GNN (865,410) beats every guarded variant
  (872,411–878,360), but reactive (886,633) does not. The guard's per-tick change budget
  (`max_changes = 4`) appears to limit it on these workloads. That limit is the guard's, not the
  forecaster's.
  - *Phase 4 correction:* an earlier version said reactive also beat every guarded variant in
    mixed; the per-family table above shows it does not.
  - *Phase 4 correction:* the sensitivity sweep (`research/guard_sweep_v4.json`, guarded_score,
    new traffic seeds 4000–4009) does not support the `max_changes` explanation. Raising it from 4 to
    8 or 16 moves elephant_mice from −7.68% to −7.68% and mixed from −1.57% to −1.59% total drops
    vs ECMP; no family moves by more than 0.512 points, and 8 and 16 give identical results. The limit
    is elsewhere in the guard (e.g. headroom, minimum improvement, cooldown), not its change budget.
- **llm_training is unaffected by any strategy**, and stride is solved completely by every
  guarded strategy.

## Inference timing

Per-tick model inference, pooled over all model routing runs.

| Model | Calls | p50 (ms) | p99 (ms) | Mean (ms) | Max (ms) | Share > 5 ms horizon |
|---|---:|---:|---:|---:|---:|---:|
| GNN (GCN-LSTM) | 50,760 | 2.00 | 4.06 | 2.12 | 64.7 | 0.45% |
| LSTM-only | 25,380 | 0.73 | 1.54 | 0.79 | 16.2 | 0.02% |

- **Measured on:** this laptop's CPU, `torch.set_num_threads(2)`, batch size 1.
- **Not included:** telemetry export, path computation and route installation. The simulator
  does not wait for inference.
- **Not comparable with the old figure:** the schema-2 "p99 6–13.5 ms" was measured under a
  different setup, and the timings above are not an end-to-end control-loop latency.

## Limitations

- **Small test set.** 1 test scenario per family (6 in total), so the forecast CIs describe
  variation across scenarios only.
- **LSTM-only may be under-trained.** It reached its best epoch at 48–50 of the 50-epoch cap on
  all seeds. It might improve slightly with a higher cap. The GNN early-stopped on its own
  (best epochs 15–37).
- **GNN seed variance is visible.** Seed 44 trained much longer (47 epochs) than seeds 42/43
  (25–27), and the per-seed thresholds range from 0.754 to 0.846.
- **One topology** (k = 4), one simulator, synthetic traffic, 150-tick runs.
- **No PFC, ECN or RDMA**, and no route-installation delay.
- **The guard's settings were not tuned.** `max_changes = 4`, `headroom = 0.9` and
  `min_improvement = 0.15` may favour or penalize particular signals.

## Artifacts

- **Committed:** `research/benchmark_results_v4.json` and this report. The code is in `246810e`.
- **Committed in `5273b9c` and still tracked:**
  - `checkpoints/best_model_v3.pt` (GNN, seed 42; the dashboard default);
  - `best_model_v3_lstm_only.pt`;
  - `best_model_v3_seed{1,2}.pt`, `best_model_v3_seed{1,2}_lstm_only.pt`;
  - the `*.history.json` files.
- **Committed in `5273b9c`, untracked in Phase 4 (`1cf90a0`)**, so they exist only in git history
  and as local files (all regenerable from fixed seeds):
  - `checkpoints/pilot_v3*.pt`;
  - `data/datasets_v3.pkl`, `data/pilot_datasets_v3.pkl`.

*(Corrected in Phase 4: an earlier version of this section said the `best_model_v3*.pt` checkpoints
were not committed.)*
- Tests: 44 / 44 pass, including the dashboard test, which now loads `best_model_v3.pt` and its
  calibrated threshold.

## Suggested paper framing (for the deadline)

- **Primary contribution:** the capacity-guarded admission controller. It cuts total drops
  5–6% and fabric drops 20–25% vs ECMP with about 10× fewer route changes than reactive
  rerouting, and it works with a model-free congestion score.
- **Secondary finding:** learned per-link temporal models (LSTM-only, GNN) improve congestion
  ranking over persistence (AP +0.067). A GCN adds no measurable benefit over an LSTM on
  per-link features, and onsets remain largely unpredicted.
- **Negative result to report honestly:** feeding GNN forecasts to the guard does not reduce
  drops beyond guard-only baselines.
