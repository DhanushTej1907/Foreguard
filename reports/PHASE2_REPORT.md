# Phase 2 report: honest forecasting evaluation

Branch `fix/phase2`, one commit on top of `fix/phase1` (`68aa43b`).
Date: 2026-10-08. Checkpoint schema 3; results-JSON schema 4.

## Summary and verdict

This phase makes the forecasting comparison fair:
- A label that reflects what the link actually experienced.
- A continuous persistence baseline built on the same score as the label.
- Thresholds tuned on validation for every predictor.
- Onset-only metrics.
- Two learned baselines without a graph.
- Held-out traffic families.
- A scenario-level bootstrap.

It then ran a forecast-only pilot: 20 epochs, 150 ticks.

What the pilot shows:

1. **The GNN ranks congestion better than continuous persistence overall.** Overall AP is higher
   on the standard split (+0.042, 95% CI [+0.030, +0.049]) and on all 6 held-out families. Every
   CI excludes 0.
2. **The graph part of the GNN adds nothing measurable.** An LSTM-only model without the GCN
   matches the GNN on every split: standard AP 0.947 vs 0.946, and within ±0.009 on every
   held-out family. A per-link logistic regression comes within 0.005 on the standard split.
   The ranking gain comes from learning on per-link time series, not from the GCN.
3. **Binary persistence still has the best F1.** On the standard split it scores 0.895 vs the
   GNN's 0.875, and it is at least as good as the GNN in every held-out family.
4. **Onsets remain essentially unpredicted.** Onsets make up 2.9% of non-congested link-windows.
   - GNN onset AP is 0.168 vs 0.146 for continuous persistence; the CI [−0.019, +0.070]
     includes 0.
   - The difference is significant in only 1 of 7 splits (llm_training), and there every
     predictor's AP is below 0.04.
   - At the validation-tuned threshold the GNN catches 11% of onsets at 19% precision.
5. **No routing claim can be made yet.** Routing was not run in this pilot. Guard-only baselines
   still give 5.5–6.1% drop reduction vs ECMP (see the guard ablation below).

Under CLAUDE.md's honesty rule, the paper can say:
- the learned forecasters (GNN, LSTM-only, logistic regression) beat continuous persistence in
  overall AP on this simulator.

It cannot say:
- that the graph structure helps;
- that the GNN predicts congestion onsets;
- that the GNN improves routing.

---

## Changes by item

### 0. Output names and data provenance
- **New defaults in `config.py`:**
  - `CHECKPOINT_OUT = checkpoints/best_model_v3.pt`
  - `RESULTS_OUT = research/benchmark_results_v4.json`
  - `EVALUATE_RESULTS_OUT = research/evaluate_results_v4.json` (for `main.py`, so it never
    shares a file with the benchmark)
  - `DATA_OUT = data/datasets_v3.pkl`
  - `PLOTS_OUT_DIR = plots/v3/`
- **Overwrite guard:** `check_output_path` refuses `best_model.pt`, `training_history.json`,
  `benchmark_results.json`, `datasets.pkl` and anything directly in `plots/`.
- **Command-line options:**
  - `model/train.py`: `train_model(checkpoint_path=...)`. The history is written next to the
    checkpoint as `<name>.history.json`.
  - `main.py`: `--checkpoint-out`, `--data-out`, `--results-out`. Evaluate now writes a JSON with
    forecast metrics and the comparison.
  - `research_benchmark.py`: `--checkpoint-out`, `--results-out`, `--data-out`.
- **Dataset provenance:** datasets store `schema_version` and `label_mode`, and `load_datasets`
  rejects a missing or mismatched value. The old `datasets.pkl` is therefore rejected.
- **Checkpoint provenance:** `check_checkpoint_schema` also rejects a checkpoint whose
  `label_mode` differs from config.

### 1. Default label and continuous score
- `MODEL["label_mode"] = "served_or_queued"`.
- New `congestion_score`:
  - served_or_queued: `s = clip(max(packets_out/cap, 0.85·queue/(0.1·buffer)), 0, 1)`.
  - offered: `s = clip(packets_in/cap)`.
- **The label is defined as `s > 0.85`**, so the two agree by construction. A test checks this
  against the original rule ("served > 0.85 or queue > 0.1·buffer") over a grid of 101 × 201
  integer (packets_out, queue) pairs.
- Every window carries `last_labels` and `last_scores` from the last observed tick:
  - binary persistence = `last_labels`;
  - continuous persistence = `last_scores`.
- Both persistence baselines now follow `label_mode` in `research_benchmark.py`,
  `guard_ablation_check.py` and evaluate.

### 2. Edge feature 3
- Latency/100 is replaced by **served utilization** (`packets_out/cap`), as agreed.
- Normalized queue delay would have been an exact copy of feature 1: queue delay is
  `queue/cap·tick`, so divided by its maximum it equals `queue/buffer`.
- Still 4 features; the architecture and the 17,985 parameters are unchanged.
- A test checks that every node feature, edge feature and score is in [0, 1] over hotspot and
  mixed runs.

### 3. Forecast metrics (`model/metrics.py`)
- `forecast_metrics` reports these fields **overall** and **onset-only** (link-windows with
  `last_label == 0`): n, positives, rate, precision, recall, F1, AP, AUC and Brier.
- Undefined values are `null`, not 0:
  - AUC with one class;
  - precision with no positive predictions;
  - AP with no positives.
- The decision rule everywhere is `score > threshold`, the same comparison the guarded
  controller uses.

### 4. Threshold chosen on validation
- `tune_threshold` picks the max-F1 cut on validation. It returns a midpoint between distinct
  scores, so ties are never split.
- `train_model`:
  - stores the threshold in the checkpoint as `alert_threshold`;
  - also stores `model_class`, `val_ap` and `label_mode`.
- Test metrics for every scored predictor use its own validation-tuned threshold. Binary
  persistence stays at 0.5.
- The guarded controller uses the checkpoint threshold in the benchmark and in the dashboard
  (where it also sets the slider's starting value).
- **As you requested, the best epoch is chosen by validation AP**, not F1 at 0.5, for both the
  GNN and LSTM-only. The LR scheduler and early stopping also track validation AP.

### 5. Learned baselines (`model/baselines.py`)
- **LSTM-only** (9,153 parameters): the GNN's temporal path without the GCN. A projection, LSTM
  and head are shared across links and see only each link's own 4 edge features. It uses the
  same training loop, loss, split and epoch selection as the GNN.
- **Per-link logistic regression:** 32 standardized, unweighted models on each link's flattened
  10 × 4 window. A link with only one class in training falls back to that class's constant
  prior.

### 6. Routing summary
- `research_benchmark.py` is restructured into functions: data, forecast, routing, summary.
- `summarize_routing` is a pure, unit-tested function. For each strategy it reports mean total,
  host, fabric and unroutable drops. For each non-ECMP strategy it reports **total, fabric and
  host drop reduction vs ECMP**, wins/ties/losses, and a paired bootstrap by family.
- **Guard-only baselines are now routing strategies in the benchmark itself:**
  `guarded_persistence` (last label) and `guarded_score` (continuous score at continuous
  persistence's validation threshold), next to `guarded_gnn`.
- Each routing run asserts conservation and the drop split.
- Not run in this phase; see the follow-ups.

### 7 and 9. Forecast-only mode and family holdout
- `--forecast-only` skips routing.
- `--holdout-family <family>|all`:
  - Requires `--forecast-only`.
  - Tests on all 5 scenarios of the family.
  - Validates on 1 random scenario from each of the other 5 families and trains on the
    remaining 20.
  - `all` runs the 6 rotations on one shared set of simulated scenarios.
- **Bootstrap (your request):** `bootstrap_ap_difference` resamples test scenarios with
  replacement (2,000 draws, seed 42). It reports AP(GNN) − AP(continuous persistence) with a
  95% percentile CI, overall and onset-only, for the main split and every rotation. Draws
  without positives are skipped and counted in `valid_draws`; none were skipped here.
- **Training wall-clock seconds** per model (GNN, LSTM-only, logistic regression) are recorded
  under `training_seconds` in the JSON.

### 8. Tests
- 34 tests, all passing: 20 existing, 14 new in `tests/test_phase2.py`. The new tests cover:
  - the legacy-path guard;
  - dataset and checkpoint schema and label_mode rejection;
  - score/label exactness;
  - features in [0, 1];
  - `forecast_metrics` including the onset subset and null cases;
  - max-F1 threshold tuning;
  - the scenario bootstrap;
  - LSTM-only and logistic regression;
  - the holdout split;
  - best-epoch-by-val-AP with a stored threshold;
  - a `run_forecast` smoke test (all 5 predictors, JSON-serializable);
  - `summarize_routing` arithmetic.
- Existing tests adjusted:
  - `test_horizon`: adds score history and checks `last_labels` / `last_scores`.
  - `test_schema_version`: adds `label_mode`.
  - The dashboard test now uses `best_model_v3.pt` and checks schema plus label mode.

---

## Pilot results

### How the pilot was run
```
python research_benchmark.py --forecast-only --epochs 20 --ticks 150 \
  --results-out research/pilot_forecast_v3.json --checkpoint-out checkpoints/pilot_v3.pt \
  --data-out data/pilot_datasets_v3.pkl
python research_benchmark.py --forecast-only --holdout-family all --epochs 20 --ticks 150 \
  --results-out research/pilot_holdout_v3.json --checkpoint-out checkpoints/pilot_v3.pt
```
- 30 scenarios: 6 families × 5, seeds 42–71, 150 ticks.
- `label_mode = served_or_queued`; training seed 42.
- Both runs were started at the same time on one CPU with `torch.set_num_threads(2)` each.

### Standard split (test scenarios 0 uniform, 3 stride, 8 and 20 elephant_mice, 23 mixed)

**Overall**

| Predictor | Threshold | F1 | Precision | Recall | AP | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| GNN (GCN-LSTM) | 0.819 | 0.875 | 0.874 | 0.876 | 0.946 | 0.981 | 0.059 |
| LSTM-only | 0.761 | 0.873 | 0.866 | 0.880 | **0.947** | 0.980 | 0.050 |
| Per-link LogReg | 0.494 | 0.876 | 0.913 | 0.842 | 0.941 | 0.976 | **0.042** |
| Persistence (binary) | 0.500 | **0.895** | 0.901 | 0.890 | 0.825 | 0.932 | 0.043 |
| Persistence (continuous) | 0.877 | 0.853 | 0.908 | 0.805 | 0.904 | 0.974 | 0.149* |

\* Computed on the clipped score, which is not a calibrated probability.

Test set: 21,760 link-windows, 4,535 positives (rate 0.2084). Onset subset: 17,283 link-windows,
500 positives (rate 0.0289). These match the Phase 1 label-mode measurement on the same
scenarios (0.2084, 500 onsets).

**Onset-only** (link-windows not congested at the last observed tick)

| Predictor | F1 | Precision | Recall | AP | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| GNN (GCN-LSTM) | 0.140 | 0.186 | 0.112 | 0.168 | **0.895** | 0.054 |
| LSTM-only | **0.146** | 0.178 | 0.124 | **0.168** | 0.878 | 0.043 |
| Per-link LogReg | 0.085 | 0.239 | 0.052 | 0.159 | 0.855 | 0.030 |
| Persistence (binary) | 0.000 | — | 0.000 | 0.029 | 0.500 | 0.029 |
| Persistence (continuous) | 0.000 | — | 0.000 | 0.146 | 0.860 | 0.163 |

Both persistence baselines predict no onsets by construction:
- On onset windows the binary score is 0.
- The continuous score is ≤ 0.85, below its 0.877 threshold.

**Training**

| Model | Parameters | Best epoch (val AP) | Best val AP | Train time (s) |
|---|---:|---:|---:|---:|
| GNN (GCN-LSTM) | 17,985 | 20 of 20 | 0.847 | 156.2 |
| LSTM-only | 9,153 | 18 of 20 | 0.833 | 41.2 |
| Per-link LogReg | 32 models | — | — | 0.8 |

### Bootstrap: AP(GNN) − AP(continuous persistence)
Scenario-level, 2,000 draws, 5 test scenarios per split.

| Split | Overall Δ AP [95% CI] | Onset Δ AP [95% CI] |
|---|---|---|
| Standard | **+0.042 [+0.030, +0.049]** | +0.022 [−0.019, +0.070] |
| Holdout: uniform | **+0.049 [+0.036, +0.061]** | +0.051 [−0.033, +0.095] |
| Holdout: hotspot | **+0.052 [+0.036, +0.064]** | +0.003 [−0.013, +0.017] |
| Holdout: elephant_mice | **+0.037 [+0.024, +0.051]** | −0.008 [−0.027, +0.022] |
| Holdout: stride | **+0.020 [+0.011, +0.028]** | +0.016 [−0.017, +0.044] |
| Holdout: llm_training | **+0.004 [+0.002, +0.013]** | **+0.006 [+0.003, +0.018]** |
| Holdout: mixed | **+0.039 [+0.032, +0.047]** | +0.001 [−0.016, +0.014] |

All 2,000 draws were valid in every row. With only 5 scenarios per split, the intervals describe
scenario-to-scenario variation. They are not a population-level significance test.

### Held-out family generalization (train on 5 families, test on the 6th)

**Overall AP and F1**

| Held-out family | Pos. rate | GNN AP | Pers-bin AP | Pers-cont AP | LSTM-only AP | LogReg AP | GNN F1 | Pers-bin F1 | Pers-cont F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 0.106 | 0.952 | 0.843 | 0.903 | 0.951 | 0.934 | 0.874 | **0.913** | 0.877 |
| hotspot | 0.336 | 0.933 | 0.823 | 0.882 | **0.937** | 0.899 | 0.881 | 0.884 | 0.884 |
| elephant_mice | 0.122 | 0.928 | 0.804 | 0.892 | **0.934** | 0.927 | 0.847 | **0.889** | **0.889** |
| stride | 0.130 | **0.957** | 0.913 | 0.937 | 0.948 | 0.898 | 0.902 | **0.952** | **0.952** |
| llm_training | 0.028 | 0.041 | 0.035 | 0.038 | 0.035 | 0.034 | 0.103 | 0.103 | 0.103 |
| mixed | 0.537 | 0.963 | 0.865 | 0.924 | **0.966** | 0.937 | 0.890 | 0.897 | 0.897 |

**Onset-only AP and F1**

| Held-out family | Onset positives | GNN AP | Pers-cont AP | LSTM-only AP | LogReg AP | GNN F1 | Pers-cont F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 202 | **0.139** | 0.088 | 0.101 | 0.114 | 0.183 | 0.000 |
| hotspot | 903 | 0.141 | 0.138 | **0.161** | 0.133 | 0.067 | 0.000 |
| elephant_mice | 305 | 0.093 | 0.100 | 0.089 | **0.117** | 0.091 | 0.000 |
| stride | 160 | **0.042** | 0.025 | 0.024 | 0.014 | 0.000 | 0.000 |
| llm_training | 533 | **0.033** | 0.027 | 0.027 | 0.022 | 0.000 | 0.000 |
| mixed | 1,299 | 0.264 | 0.264 | **0.298** | 0.232 | 0.281 | 0.000 |

Binary persistence's onset AP equals the onset prevalence in every family, so it is omitted.

**Training per rotation**

| Held-out family | GNN train (s) | LSTM-only train (s) | GNN best epoch | LSTM-only best epoch |
|---|---:|---:|---:|---:|
| uniform | 149.9 | 38.7 | 18 | 19 |
| hotspot | 100.7 | 28.5 | 16 | 18 |
| elephant_mice | 104.7 | 27.4 | 15 | 20 |
| stride | 106.2 | 34.2 | 19 | 20 |
| llm_training | 118.6 | 41.0 | 16 | 18 |
| mixed | 115.0 | 31.4 | 17 | 18 |

### Guard-only routing baselines under the new default label
From `tools/guard_ablation_check.py`: 18 runs, mean drops, compared with ECMP on the current
simulator.

| | Mean drops | vs ECMP |
|---|---:|---:|
| ECMP | 213,467.4 | — |
| Guarded + persistence labels | 201,778.3 | −5.48% |
| Guarded + continuous score (controller default threshold 0.7) | 200,444.6 | −6.10% |

The ablation's forecasting section matches the benchmark exactly through an independent code
path:
- binary persistence F1 0.8955 / AP 0.8249 / AUC 0.9320;
- continuous persistence AP 0.9039 / AUC 0.9742.

---

## Interpretation

- **The GCN is not earning its keep.** On every split the LSTM-only model, which sees only each
  link's own history, is within 0.009 AP of the GNN and often ahead. Per-link logistic
  regression trails the GNN by 0.005 AP on the standard split. In the held-out families its gap
  is larger (up to 0.06 on stride), though it still beats continuous persistence in 4 of 6.
  Whatever helps comes from learning on per-link time series, not from the graph.
- **F1 favours binary persistence**, because most positives are continuing congestion. Ranking
  metrics (AP, AUC) are where the learned models gain.
- **Onsets are the real forecasting problem, and nothing solves it yet.** Onset prevalence is
  2.9%, and the best onset AP on the standard split is 0.168. In llm_training, 533 of the
  roughly 610 positives are onsets: synchronized collective bursts. Every predictor scores below
  0.04 AP there, including models trained on the other five families.
- **Calibration:** the GNN's Brier score (0.059) is the worst of the learned models and worse
  than binary persistence (0.043). Training uses `pos_weight = 3`, which pushes probabilities up.
  The validation-tuned threshold (0.819) compensates for decisions, but the raw probabilities are
  not calibrated.

## Limitations

- **Pilot scale.** One training seed, 20 epochs, 150 ticks, 5 test scenarios per split.
- **The standard-split GNN may be under-trained.** Its best epoch was the last one. Val AP over
  epochs 17–20 was 0.843, 0.845, 0.837, 0.847: noisy but not flat. The LSTM-only model had
  plateaued at about 0.833.
  Holdout GNNs peaked at epochs 15–19.
- **Unbalanced standard split.** Validation has 2 of its 4 scenarios from llm_training (ids 4
  and 28), which pulls validation AP down (0.847 vs test 0.946) and shifts threshold tuning. The
  test set contains no hotspot or llm_training scenario. The holdout rotations partly cover this.
- **Training times are not clean.** The two pilot runs ran concurrently on one CPU, so the
  standard split and the first rotation (uniform) are inflated: about 150–156 s against
  100–119 s for the later rotations.
- **No routing results with a schema-3 GNN.** `guarded_gnn` vs the guard-only baselines must
  come from a full (non-forecast-only) benchmark run.
- **Inference latency** was not re-measured; the schema-2 p99 was 6–13.5 ms, above the 5 ms
  horizon.

## Artifacts

- **Committed:**
  - `research/pilot_forecast_v3.json`
  - `research/pilot_holdout_v3.json`
  - this report
- **Not committed** (regenerable from fixed seeds):
  - `checkpoints/pilot_v3*.pt` and `*.history.json` (14 checkpoints);
  - `data/pilot_datasets_v3.pkl`.
- **Untouched:** `best_model_v3.pt`, `benchmark_results_v4.json` and `datasets_v3.pkl` are
  reserved for the full run and don't exist yet. No legacy file was modified.

## Follow-ups for Phase 3

1. Full benchmark run (routing included) to get `guarded_gnn` vs `guarded_persistence` /
   `guarded_score` with fabric-drop reductions. Use more epochs, and add LSTM-only as a routing
   predictor too.
2. Multiple training seeds, so model-to-model differences can be separated from seed noise.
3. An onset-focused objective or sampling, if forecasting onsets is to remain the paper's claim.
4. Calibration: temperature scaling on validation, or drop `pos_weight`.
5. A stratified standard split, so validation and test each cover all families.
6. Port `summarize_results.py` and `plot_extra_graphs.py` to the v4 JSON. Update the paper and
   docs: shared-FIFO wording, the label definition, feature 3.

## Diff stat (excluding this report)
```
 CLAUDE.md                       |   24 +-
 config.py                       |   27 +-
 dashboard/server.py             |   12 +-
 main.py                         |   68 +-
 model/baselines.py              |   66 +
 model/dataset.py                |   49 +-
 model/evaluate.py               |   26 +-
 model/gnn_model.py              |    5 +-
 model/metrics.py                |   89 ++
 model/splits.py                 |   18 +
 model/train.py                  |   72 +-
 research/pilot_forecast_v3.json |  568 +++++++++
 research/pilot_holdout_v3.json  | 2525 +++++++++++++++++++++++++++++++++++++++
 research_benchmark.py           |  322 +++--
 simulation/telemetry.py         |   53 +-
 tests/test_dashboard.py         |   16 +-
 tests/test_phase2.py            |  216 ++++
 tests/test_research.py          |    6 +-
 tools/guard_ablation_check.py   |   16 +-
 19 files changed, 3961 insertions(+), 217 deletions(-)
```
