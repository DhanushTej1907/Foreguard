# Concern 3 — mixed-load retraining: pre-registered protocol

Written and committed before any Concern 3 run. It is not edited after results exist; any deviation is reported in REPORT.md.

## Question
Concern 3 has two separate findings:
- **(a) Forecaster generalization (tested here).** The GNN was trained at load 1.0 only and does not beat the model-free score
  signal across the offered-load sweep (`rev3_load.py`). Does training on all sweep loads improve its forecasts, and through them
  the guarded controller, at those loads?
- **(b) Guard effectiveness (existing result, not a target).** The guard lowers drops at loads ≥0.5 and is idle or slightly
  harmful below that. Retraining the forecaster neither resolves nor invalidates this; it is reported in its own section.

The aim is to test, not to find a benefit. Positive, null and negative results are reported as they come out.

## Models (architecture, features, target, hyperparameters unchanged: `model/gnn_model.py`, `model/train.py`)
| | data | seeds | environment |
|---|---|---|---|
| M0 | stored `checkpoints/best_model_v3[_seed1/_seed2].pt` (load 1.0) | 42, 43, 44 | original |
| M0′ | load 1.0: 18 train / 6 val base scenarios (2,448 / 816 windows) | 42, 43, 44 | current venv (Py 3.14.3, torch 2.14.1+cpu, 2 threads) |
| M1 | the same base scenarios at all 8 loads (19,584 / 6,528 windows) | 42, 43, 44 | current venv |

Loads: 0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0. Every flow rate is multiplied by the load exactly as in `rev3_load.py`
(`c3_common.scaled_flows`). Load copies keep the base `scenario_id`, so the stratified split (Random(42) per family) puts every
copy of a base scenario in one partition, equal to the benchmark split (train [0–5,7,9,14,15,16,18,23–26,28,29], val
[6,8,10,13,17,21], test [11,12,19,20,22,27]). Training follows `research_benchmark.run_forecast`: `seed_everything(42+k)` before
building the model, `train_model` with ≤50 epochs, early stop 10, batch 32, checkpoint selection on pooled val AP, then temperature
and alert threshold fitted on the group's own validation set. **Known difference:** M1 has 8× the windows and so ~8× the gradient
steps per epoch. This is reported, not controlled (no matched-count run in this round).

## Calibration / threshold protocols (no test data used)
- **P1.0:** temperature T + alert threshold τ fitted (`fit_temperature`, `tune_threshold` max-F1) on the 6 val base scenarios at load 1.0.
- **Pmix:** the same fit on the 6 val base scenarios pooled over the 8 loads.
- Score: P1.0 τ = 0.851 (the value every revision script uses; `tune_threshold` on val `last_scores` = 0.8510000109672546);
  Pmix τ = `tune_threshold` on pooled val `last_scores`.
- Every signal gets exactly one fit per protocol, on the same data with the same function. Natural protocols: M0/P1.0, M0′/P1.0, M1/Pmix.

## Hypotheses and primary statistics (two-sided; no direction assumed)
**HF — forecast (core test of (a), threshold-free).** On open-loop ECMP telemetry of the unseen traffic (seeds 4000–4009 ×
6 families = 60 base scenarios) at each load: AP(M1) − AP(M0′), AP averaged over the 3 training seeds, with a 95% scenario
bootstrap CI (2,000 draws, seed 42, base scenarios resampled; `model.metrics.bootstrap_ap_difference` semantics). Also reported:
M1 − continuous persistence, M0′ − persistence, M0′ − M0, M1 − M0, onset AP, Brier under both protocols; the 6 benchmark test
base scenarios as a secondary set; all loads pooled (base scenarios resampled with all their loads).
**HR — routing (primary routing comparison).** Per load: M1/Pmix − score/Pmix total drops, in % of the mean ECMP drops, paired over
the 60 (family, seed) cells (model runs averaged over training seeds), with pair and family-clustered bootstrap 95% CIs
(`rev5_stats.compare`). Also: all loads aggregated (drops summed over loads per base scenario).

**Decision rule.** A difference is called *detected* only if its 95% CI excludes 0. For routing, both the pair and the clustered CI
must exclude 0. Otherwise it is reported as *not detected* (CI includes 0), with the interval. AP is **undefined** at a load with 0
test positives. It is **unstable** when the load has < 100 positive link-windows or < 5 base scenarios with positives; unstable
estimates are reported but not interpreted. The paper may say the GNN helps only where HR shows M1 detected better than score/Pmix.

## Secondary comparisons
Required (tier 1): M1/Pmix − M0′/P1.0 (mixed-training effect, same environment); M0′/P1.0 − score/0.851 (same-environment replicate
of the existing result); M0/P1.0 − score/0.851 (original checkpoints); M1/Pmix − M0/P1.0; score/Pmix − score/0.851 (threshold effect).
Optional (tier 2): M1/P1.0, M0/Pmix, M0′/Pmix, giving the model effect at a fixed protocol and the threshold effect at a fixed model.

## Guard effectiveness (finding (b), separate section)
Per load: score/0.851 − ECMP and score/Pmix − ECMP (plus score without rules and the random signal), with the same CIs.

## Passive oracle (diagnostic, separate from HR), loads 0.2 and 0.25 only
`run_case_x` signals `oracle_label` (τ 0.5) and `oracle_score` (score Pmix τ; also τ 0.851 if they differ). Each controller tick
copies the simulator, advances it 5 ticks with routing frozen, and gives the guard the true t+5 label / score of every link.
Comparisons: oracle − score (same τ), oracle − ECMP, oracle − M1/Pmix, paired with the same CIs.
Purpose: to tell apart weak forecasts from limited value of better information under this routing policy, should M1 be null at the
knee loads.
**Limitations:** it uses future labels and scores, including future flow arrivals, so it is not deployable. It is passive: its
look-ahead does not model how rerouting changes later traffic and outcomes. It is not a guaranteed upper bound on the value of
forecasting. A poor oracle result does not prove that prediction quality is irrelevant; it holds only for this guard (τ gating,
δ = 0.15, capacity check α 0.9, ≤ 4 changes per tick, cooldown 5, stretch 2). Loads 0.1–0.15 and 0.5–1.0 are not run for the oracle.

## Routing runs
Unseen seeds 4000–4009 × 6 families × 8 loads, 150 ticks, `instrument=True` (reordering, per-flow loss); same flows, ids and
`run_case_x` arguments as `rev3_load.py`. Tier 1 per cell: ECMP, score/0.851, score/Pmix, M0/P1.0 ×3, M0′/P1.0 ×3, M1/Pmix ×3,
score without rules, random/0.851 (14 runs; 6,720 total). Tier 2: M1/P1.0 ×3, M0/Pmix ×3, M0′/Pmix ×3 (4,320). Oracle: 6 families
× 10 seeds × 2 loads × 2–3 variants.

**Budget rule (6 h compute in total).** After the pilots: if tier 1 + full oracle + tier 2 fit, run all. If only tier 1 + full oracle
fit, tier 2 is "not run". Otherwise drop, in this order, until it fits: (1) tier 2; (2) the full oracle, cut to families mixed +
hotspot; (3) score-without-rules / random reruns; (4) M0 reruns (the existing rev3_load rows then serve as the M0 reference).
Never dropped: ECMP, score/0.851, score/Pmix, M1/Pmix ×3, M0′/P1.0 ×3, the forecast evaluation. Dropped items are labelled
"not run", never "tested".

## Gates
Blockers (stop on failure): B1 baseline tests; B2 copied-file verification; B3 = G1 dataset determinism (bitwise vs local
`data/datasets_v3.pkl`); B4 = G2 fidelity to the stored checkpoints and benchmark JSON; B5 mixed-load data integrity (load-1.0
slice bitwise equal to the original, one partition per base scenario equal to the benchmark split at every load, window counts);
B6 pipeline tests + protected-file manifest; B7 training sanity (schema, finite T/τ, val AP > 0). Sweep blockers: S1 model-free
reruns = `rev3_load.jsonl`; S2 instrumented = uninstrumented; S3 no errors, packet conservation, drop split; S4 oracle reruns at load
1.0 = `rev4_c1.jsonl`. Informational (never stop the run): I1 = G3 M0′ vs stored checkpoints epoch by epoch; I2 M0 GNN reruns vs
`rev3_load.jsonl`.

## Fixed seeds and rules
Training-data seeds 42–71 (scenario i ↔ seed 42+i); training seeds 42/43/44; traffic 4000–4009; bootstrap seed 42; random signal
rng 1000·family_index + seed. No test data selects models, calibrations, thresholds or variants. No reruns with changed settings
after seeing results. Nothing is tuned to make M1 beat the score. Outputs: `revision/results/c3/`, `checkpoints/c3/`,
`data/datasets_c3_mixedload.pkl`. Protected (never written): `checkpoints/best_model_v3*`, `data/datasets_v3.pkl`, `research/*`,
existing `revision/results/*`.
