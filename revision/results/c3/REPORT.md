# Concern 3 — mixed-load retraining: report

Protocol: `PROTOCOL.md` (committed in `40036fd` before any run). Generated tables: `tables.md`; numbers: `summary.json`,
`forecast.json`, `train_*.json`, `calibration.json`, `gate.json`, `data.json`, `runtime.json`.

## Summary
- **(a) Forecaster.** Training on all 8 loads (M1) gives a small but detected forecast gain over the same-environment load-1.0
  model (M0′) at every load from 0.15 to 1.0 on unseen traffic: AP +0.020 to +0.027 at loads 0.15–0.25, +0.003 at 1.0, all
  95% CIs above 0. Brier is lower at every load. Both GNNs beat continuous persistence by 0.07–0.12 AP at every defined load.
  At 0.1 there are no positives (AP undefined); at 0.125 the estimate is unstable.
- **Routing (primary, HR):** M1/Pmix − score/Pmix is **not detected at any load** (all CIs include 0). All loads aggregated:
  +0.13 points [−0.37, +0.59] pair / [−0.66, +1.01] clustered. M1 vs M0′ in routing: point estimates are slightly negative
  (fewer drops) at every load with drops, but not detected anywhere (aggregate −0.16 [−0.47, +0.10] / [−0.63, +0.12]).
  The better forecasts do not translate into measurably fewer drops.
- **Passive oracle at 0.2 / 0.25:** a perfect passive t+5 label or score also *raises* drops vs ECMP (+1.6% to +3.7%) and does not
  differ detectably from the score signal. Under this guard, better passive information does not remove the knee-load harm,
  so a null M1 result there is consistent with limited value of information under the tested routing policy, not only with weak
  forecasts. This holds within the oracle's assumptions (see §4).
- **(b) Guard effectiveness**, reproduced exactly from `rev3_load.jsonl`: +0.34 / +1.56 / +2.97% drops vs ECMP at 0.15 / 0.2 / 0.25,
  and −5.57 / −4.44 / −4.28% at 0.5 / 0.75 / 1.0.
- Consequence for the paper: it may report the forecast gain from mixed-load training, but it must not claim the GNN improves
  routing. HR shows no detected benefit over the score signal at any load.

## 1. Gates
| gate | result | evidence |
|---|---|---|
| B1 baseline test suite | pass | 72 tests OK before any change; 75 → 86 tests OK before each commit |
| B2 copied-file verification | pass | `rev5_cap_analyze.py`, `rev5_fb_compare.py`, `rev5_external_check.py` rerun in a scratch copy: parsed-JSON identical; byte-identical after CRLF→LF (raw hashes differ only by line endings); external check 780/780 runs identical. `rev5_cap_diag.json` values equal the `rev5_capterms` audit aggregates (10/10 checks) |
| B3 = G1 dataset determinism | pass | rebuilt load-1.0 train / val / test bitwise equal to local `data/datasets_v3.pkl` |
| B4 = G2 fidelity | pass | split lists equal the benchmark; all 6 checkpoints: val AP and test AP exactly equal to the stored / benchmark values, T and τ within ~1e-11; score τ = 0.8510000109672546 |
| B5 mixed-load data | pass | load-1.0 slice bitwise equal to the original; one partition per base scenario, equal to the benchmark split at every load; 32,640 windows (19,584 / 6,528 / 6,528) |
| B6 pipeline tests + manifest | pass | `tests/test_c3.py` (11), `tests/test_load_models.py` (3); 88 protected files unchanged at every check |
| B7 training sanity | pass | 6 runs: schema 3, label_mode, finite T/τ, val AP > 0 |
| S1 model-free reruns = rev3_load | pass | 1,920 / 1,920 rows identical on all recorded fields |
| S2 instrumented = plain | pass | 60 / 60 cells identical (drops, delivered, delay, reroutes, decision / signal digests) |
| S3 no errors, conservation | pass | 11,280 rows, 0 errors, packet conservation and drop split hold everywhere |
| S4 oracle repro = rev4_c1 | pass | 4 / 4 rows identical (load 1.0, mixed, seeds 4000–4001) |
| **I1 = G3** (informational) | **exact** | M0′ seeds 42/43/44 reproduce the stored checkpoints epoch by epoch (identical val-AP curves, best epochs 15/17/37) with **bit-identical weights**. In this environment M0′ ≡ M0 |
| **I2** (informational) | **exact** | 1,440 / 1,440 M0 GNN routing reruns identical to `rev3_load.jsonl` (recorded on Linux, Python 3.13) |

Because M0′ ≡ M0, every M0′ row equals the matching M0 row (identical decisions in 1,440 / 1,440 cells). "M1 − M0′" is therefore
also "M1 − original checkpoints".

## 2. Forecast (finding a): open-loop ECMP telemetry, AP mean ± sd over 3 training seeds
Unseen traffic (seeds 4000–4009 × 6 families, 60 base scenarios; primary). CIs: scenario bootstrap, 2,000 draws, seed 42.
AP is threshold-free. Brier uses each model's natural protocol (M0′/P1.0, M1/Pmix).

| load | positives | scen. w/ pos | AP M0 = M0′ | AP M1 | AP persistence | M1 − M0′ AP [95% CI] | M1 − persistence [95% CI] | onset AP M0′ / M1 / pers. | onset M1 − M0′ [95% CI] | Brier M0′ / M1 | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 0/60 | undef. | undef. | undef. | — | — | — | — | 0.0021 / 0.0000 | AP undefined |
| 0.125 | 44 | 5/60 | 0.375 ± 0.010 | 0.406 ± 0.072 | 0.354 | +0.031 [−0.088, +0.088] | +0.052 [−0.029, +0.166] | 0.033 / 0.012 / 0.009 | −0.021 [−0.326, +0.001] | 0.0023 / 0.0003 | unstable |
| 0.15 | 237 | 11/60 | 0.554 ± 0.004 | 0.581 ± 0.007 | 0.464 | **+0.027 [+0.004, +0.055]** | **+0.117 [+0.068, +0.147]** | 0.043 / 0.055 / 0.027 | +0.011 [+0.000, +0.020] | 0.0029 / 0.0009 | ok; onset unstable |
| 0.2 | 1,569 | 25/60 | 0.709 ± 0.005 | 0.734 ± 0.002 | 0.645 | **+0.025 [+0.014, +0.034]** | **+0.089 [+0.059, +0.106]** | 0.041 / 0.054 / 0.050 | **+0.012 [+0.002, +0.022]** | 0.0061 / 0.0039 | ok |
| 0.25 | 4,200 | 26/60 | 0.798 ± 0.005 | 0.818 ± 0.001 | 0.733 | **+0.020 [+0.014, +0.026]** | **+0.086 [+0.069, +0.100]** | 0.068 / 0.086 / 0.081 | **+0.018 [+0.010, +0.024]** | 0.0100 / 0.0074 | ok |
| 0.5 | 21,791 | 35/60 | 0.900 ± 0.002 | 0.910 ± 0.000 | 0.819 | **+0.010 [+0.007, +0.014]** | **+0.091 [+0.079, +0.110]** | 0.197 / 0.222 / 0.173 | **+0.025 [+0.014, +0.037]** | 0.0317 / 0.0258 | ok |
| 0.75 | 39,680 | 55/60 | 0.930 ± 0.002 | 0.937 ± 0.002 | 0.862 | **+0.008 [+0.005, +0.011]** | **+0.076 [+0.065, +0.092]** | 0.175 / 0.208 / 0.151 | **+0.032 [+0.021, +0.044]** | 0.0440 / 0.0357 | ok |
| 1.0 | 55,916 | 60/60 | 0.929 ± 0.002 | 0.933 ± 0.002 | 0.859 | **+0.003 [+0.002, +0.005]** | **+0.074 [+0.060, +0.092]** | 0.147 / 0.157 / 0.117 | **+0.010 [+0.004, +0.016]** | 0.0581 / 0.0502 | ok |
| all loads pooled | 123,437 | 60/60 | 0.912 ± 0.004 | 0.923 ± 0.002 | 0.843 | **+0.011 [+0.008, +0.013]** | **+0.080 [+0.069, +0.096]** | 0.147 / 0.173 / 0.126 | **+0.026 [+0.020, +0.031]** | 0.0196 / 0.0155 | ok |

Bold = detected (CI excludes 0). "Unstable" = < 100 positives or < 5 base scenarios with positives (pre-registered); unstable
estimates are listed but not interpreted. The secondary set (6 benchmark test base scenarios; full table in `tables.md`) shows the
same direction. It is unstable at every load ≤ 0.5, because only 1–4 base scenarios contain positives, so it adds little here.
At load 1.0 M1 − M0′ = +0.0006 [−0.010, +0.004]; all loads pooled +0.009 [+0.002, +0.012].
Training data had the same scarcity: 3 / 8 / 63 / 281 / 778 training positives at loads 0.1 / 0.125 / 0.15 / 0.2 / 0.25 (`data.json`).

## 3. Routing: total drops vs ECMP, route changes per run (rc), reordered packets (% of delivered)
Unseen seeds 4000–4009 × 6 families, 150 ticks, instrumented; model variants averaged over 3 training seeds per cell.

| load | ECMP loss % | score/0.851 | score/Pmix | M0/P1.0 | M0′/P1.0 | M1/Pmix |
|---|---|---|---|---|---|---|
| 0.1 | 0.00 | n/a; 0.0 rc; 0.000% | n/a; 0.0 rc; 0.000% | n/a; 0.0 rc; 0.000% | n/a; 0.0 rc; 0.000% | n/a; 0.0 rc; 0.000% |
| 0.125 | 0.14 | +0.00%; 1.7 rc; 0.000% | +0.00%; 1.7 rc; 0.000% | +0.00%; 2.5 rc; 0.001% | +0.00%; 2.5 rc; 0.001% | −0.46%; 2.0 rc; 0.000% |
| 0.15 | 0.32 | +0.34%; 3.8 rc; 0.005% | +0.34%; 3.8 rc; 0.005% | +0.96%; 4.7 rc; 0.005% | +0.96%; 4.7 rc; 0.005% | +0.76%; 4.6 rc; 0.005% |
| 0.2 | 1.79 | +1.56%; 14.6 rc; 0.036% | +1.56%; 14.6 rc; 0.036% | +1.62%; 17.7 rc; 0.051% | +1.62%; 17.7 rc; 0.051% | +1.07%; 16.0 rc; 0.047% |
| 0.25 | 4.55 | +2.97%; 24.4 rc; 0.059% | +2.97%; 24.4 rc; 0.059% | +3.28%; 28.5 rc; 0.085% | +3.28%; 28.5 rc; 0.085% | +3.06%; 26.6 rc; 0.088% |
| 0.5 | 15.39 | −5.57%; 33.1 rc; 0.191% | −5.57%; 33.1 rc; 0.191% | −5.49%; 38.5 rc; 0.279% | −5.49%; 38.5 rc; 0.279% | −5.67%; 38.5 rc; 0.283% |
| 0.75 | 26.17 | −4.44%; 29.0 rc; 0.208% | −4.44%; 29.0 rc; 0.208% | −3.86%; 42.3 rc; 0.272% | −3.86%; 42.3 rc; 0.272% | −4.06%; 36.3 rc; 0.267% |
| 1.0 | 35.51 | −4.28%; 33.9 rc; 0.223% | −4.28%; 33.9 rc; 0.223% | −4.10%; 55.3 rc; 0.268% | −4.10%; 55.3 rc; 0.268% | −4.23%; 43.5 rc; 0.299% |

ECMP reorders nothing (0.000% at every load). Score/Pmix makes the same decisions as score/0.851 in 480 / 480 cells: its Pmix τ
(0.8510000109672546) is float32(0.851), and no score value lies between the two.

**Paired differences** in points of the mean ECMP drops (packets at load 0.1, where ECMP drops 0); mean [pair 95% CI] / [family-clustered 95% CI]:

| load | **PRIMARY M1/Pmix − score/Pmix** | M1/Pmix − M0′/P1.0 | M0′/P1.0 − score/0.851 | M0/P1.0 − score/0.851 |
|---|---|---|---|---|
| 0.1 | +0.0 pkts [0, 0] / [0, 0] | +0.0 pkts [0, 0] / [0, 0] | +0.0 pkts | +0.0 pkts |
| 0.125 | −0.46 [−1.37, +0.00] / [−2.28, +0.00] | −0.46 [−1.37, +0.00] / [−2.28, +0.00] | +0.00 [0, 0] | +0.00 [0, 0] |
| 0.15 | +0.41 [+0.00, +1.24] / [+0.00, +1.66] | −0.21 [−0.62, +0.00] / [−0.83, +0.00] | +0.62 [+0.00, +1.86] / [+0.00, +2.48] | same as M0′ |
| 0.2 | −0.49 [−2.59, +1.12] / [−3.13, +1.02] | −0.54 [−1.52, +0.06] / [−1.96, +0.04] | +0.05 [−2.14, +2.35] / [−2.38, +2.37] | same |
| 0.25 | +0.09 [−0.56, +1.12] / [−0.60, +1.23] | −0.22 [−1.05, +0.43] / [−1.12, +0.40] | +0.31 [−0.85, +2.10] / [−0.84, +2.23] | same |
| 0.5 | −0.09 [−0.67, +0.53] / [−0.71, +0.59] | −0.18 [−0.77, +0.46] / [−0.98, +0.36] | +0.09 [−0.57, +0.82] / [−0.54, +1.06] | same |
| 0.75 | +0.38 [−0.28, +0.92] / [−0.67, +1.74] | −0.20 [−0.67, +0.20] / [−0.76, +0.26] | +0.58 [−0.11, +1.27] / [−0.61, +2.03] | same |
| 1.0 | +0.05 [−0.58, +0.66] / [−0.88, +1.00] | −0.13 [−0.47, +0.18] / [−0.70, +0.30] | +0.18 [−0.55, +0.93] / [−0.95, +1.53] | same |
| all loads | +0.13 [−0.37, +0.59] / [−0.66, +1.01] | −0.16 [−0.47, +0.10] / [−0.63, +0.12] | +0.29 [−0.31, +0.92] / [−0.66, +1.52] | same |

None of these differences is detected: every interval includes 0. Intervals with an endpoint of exactly 0.00 occur at low loads,
where most cells have identical outcomes; the 2.5% or 97.5% bootstrap quantile is then 0, so the interval includes 0.
Tier 2 (fixed-protocol breakdown, in `tables.md`) shows no detected model effect at a fixed protocol at any load or in aggregate:
M1/P1.0 − M0′/P1.0 −0.03 [−0.40, +0.31]; M1/Pmix − M0′/Pmix −0.15 [−0.42, +0.11]. No detected threshold effect for M0 / M0′ either.
The one boundary case: M1/Pmix − M1/P1.0 at load 1.0 is −0.26, pair CI [−0.55, −0.05] but clustered CI [−0.69, +0.03], so it is
not detected under the two-CI rule.

## 4. Passive oracle (diagnostic, loads 0.2 and 0.25, all 6 families × 10 seeds)
| load | variant | vs ECMP | route changes | reordered % |
|---|---|---|---|---|
| 0.2 | oracle label (τ 0.5) | +2.28% | 10.9 | 0.030 |
| 0.2 | oracle score (τ Pmix) | +1.57% | 8.6 | 0.031 |
| 0.25 | oracle label (τ 0.5) | +3.69% | 20.4 | 0.045 |
| 0.25 | oracle score (τ Pmix) | +3.70% | 15.0 | 0.022 |

| comparison (points of ECMP drops) | 0.2 | 0.25 |
|---|---|---|
| oracle label − score/Pmix | +0.71 [−1.11, +3.17] / [−2.13, +4.38] | +0.72 [−0.56, +2.12] / [−0.49, +2.53] |
| oracle score − score/Pmix | +0.01 [−1.53, +1.50] / [−2.13, +2.04] | +0.72 [−0.82, +2.45] / [−0.84, +3.18] |
| oracle label − ECMP | +2.28 [−6.30, +10.87] / [−12.08, +16.91] | +3.69 [+1.28, +6.43] / [+0.00, +9.19] |
| oracle score − ECMP | +1.57 [−6.78, +9.11] / [−12.08, +14.98] | +3.70 [+1.01, +6.71] / [−0.07, +9.62] |
| oracle label − M1/Pmix | +1.20 [−1.25, +4.84] / [−2.22, +6.63] | +0.63 [−0.55, +1.84] / [−0.72, +2.34] |
| oracle score − M1/Pmix | +0.50 [−1.31, +2.83] / [−2.23, +3.93] | +0.64 [−0.82, +2.29] / [−0.82, +2.92] |

Reading: at the knee loads, perfect passive 5-tick information is no better than the score signal or M1 (all CIs include 0), and
it raises drops vs ECMP. At 0.25 the pair CI excludes 0 but the clustered CI does not, so under the two-CI rule this is not
detected. Under this guard, then, better information does not remove the knee-load harm. A null M1 routing result at these loads
is therefore not by itself evidence that the forecasts are too weak.
**Limitations:** the oracle uses future labels and scores, including future flow arrivals, so it is not deployable. It is passive:
its look-ahead freezes routing, so it doesn't model how its own reroutes change later traffic and outcomes. It is not a guaranteed
upper bound on the value of forecasting, and a poor oracle result doesn't prove prediction quality irrelevant. The result holds only
for this guard (τ gating, δ 0.15, capacity check α 0.9, ≤ 4 changes per tick, cooldown 5, stretch 2). The oracle was not run at
loads 0.1–0.15 or 0.5–1.0. `oracle score/0.851` was not run because it would be decision-identical to `oracle score/Pmix`.

## 5. Guard effectiveness (finding b, existing result, reproduced exactly)
Points of mean ECMP drops; [pair] / [clustered] 95% CI. The rows equal `rev3_load.jsonl` exactly (S1).

| load | score/0.851 − ECMP | score, no rules − ECMP | random/0.851 − ECMP |
|---|---|---|---|
| 0.1 | 0 (no drops) | 0 | 0 |
| 0.125 | +0.00 [0, 0] | −11.27 [−26.43, +0.00] / [−35.47, +0.00] | −1.70 [−10.51, +4.01] |
| 0.15 | +0.34 [+0.00, +1.02] / [+0.00, +1.37] | −7.90 [−18.17, −1.18] / [−26.45, +0.00] | −0.76 [−4.78, +2.55] |
| 0.2 | +1.56 [−5.99, +8.75] / [−9.95, +13.97] | −3.64 [−10.78, +1.44] | +0.33 [−8.04, +7.67] |
| 0.25 | +2.97 [+0.52, +5.56] / [+0.00, +7.93] | −2.31 [−5.27, −0.00] / [−7.21, +0.41] | +2.23 [−0.10, +4.71] |
| 0.5 | −5.57 [−9.66, −2.40] / [−17.05, +0.04] | −7.76 [−13.24, −3.41] / [−23.80, +0.05] | −5.03 [−8.95, −1.79] / [−15.63, −0.03] |
| 0.75 | **−4.44 [−6.82, −2.29] / [−9.53, −0.74]** | −9.02 [−13.23, −5.16] / [−22.23, −1.11] | −4.98 [−7.24, −2.87] / [−11.00, −0.40] |
| 1.0 | **−4.28 [−5.87, −2.78] / [−6.73, −1.93]** | −9.76 [−12.78, −6.97] / [−19.08, −3.25] | −4.59 [−6.08, −3.22] / [−7.53, −2.09] |
| all loads | **−4.32 [−6.10, −2.70] / [−7.92, −1.51]** | −9.13 [−12.65, −5.92] / [−20.07, −2.19] | −4.61 [−6.24, −3.05] / [−8.91, −1.35] |

The guard lowers drops at 0.75 and 1.0 (both CIs exclude 0). At 0.5 the pair CI excludes 0 and the clustered CI just reaches +0.04.
At 0.15–0.25 it is idle or raises drops (+0.34 to +2.97 points; at 0.25 the pair CI excludes 0, the clustered CI touches 0).
Retraining the forecaster leaves this unchanged. Score without rules removes more drops but makes 12–2,492 route changes per run
and reorders up to 18.8% of delivered packets (vs 0.22% for score/0.851 at load 1.0).

## 6. Tier, runtime, not run, deviations
**Tier:** tier 1 + full oracle + tier 2, all run. Decision at 22:45 after the pilots: 2.86 h used; measured per-run costs projected
~5.2 h in total, within the 6 h budget. **Measured compute: 4.24 h** (`runtime.json`): M1 training 89 min (1,805 / 1,255 / 2,250 s);
M0′ 8 min; forecast evaluation 64 min (the pooled unseen-set bootstrap alone took 63 min; estimate was 10–25 min); tier 1 sweep 47 min
(6,552 runs); tier 2 32 min (4,212 runs); oracle 2 min + 1 min pilot; pilots and checks 6 min; gates, data, calibration, analysis
and tests ≈ 7 min. Pilot per-run means (load 0.25 / 1.0, one worker): ECMP 0.23 / 0.18 s, score 0.89 / 2.79 s, GNN variants
1.2–1.4 / 3.2–3.7 s, oracle 4.3–5.0 s; 3 workers gave a parallel efficiency of 2.6.

**Training budget difference (reported, not controlled):** M1 trained on 19,584 windows (8×) and took 15,300 / 12,240 / 20,196
gradient steps (best epochs 15 / 10 / 23 of 25 / 20 / 33). M0′ took 1,925 / 2,079 / 3,619 steps (best epochs 15 / 17 / 37), so M1
had 6.3× more steps in total. M1's pooled mixed val AP (0.927–0.930) and M0′'s load-1.0 val AP (0.933–0.935) come from different
validation sets and are not comparable.
**Calibration:** M0 = M0′ P1.0 (T, τ) = (0.840, 0.788), (0.883, 0.846), (0.980, 0.754); Pmix = (0.583, 0.901), (0.583, 0.923),
(0.757, 0.855). M1 Pmix = (0.898, 0.691), (1.073, 0.780), (0.903, 0.750); P1.0 = (1.029, 0.612), (1.243, 0.740), (1.045, 0.714).
Score: 0.851 under both protocols.

**Not run:** `oracle score/0.851` (decision-identical to oracle score/Pmix, see §3); the oracle at loads 0.1–0.15 and 0.5–1.0
(by design); the FCT / closed-loop sweep (deferred by design); a matched-sample-count training run (not in this round by design).
None of these hypotheses was tested or rejected.

**Deviations from the plan:**
1. The oracle task runner was folded into `c3_load.py` (`oracle-pilot`, `oracle`, `oracle-repro` subcommands) instead of a separate
   `c3_oracle.py`. Same code path as the routing runs.
2. `c3_analyze.py` was run once on partial data (after tier 1) to debug it. That run exposed a normalisation bug in the oracle
   section: with the pilot covering one family, it divided by the ECMP mean of all 60 cells. Oracle differences are now normalised by
   the ECMP mean over the compared cells. With the full oracle (all 60 cells) the two give the same values. No setting, variant or
   hypothesis was changed, and nothing was selected from the partial run.
3. Pilot rows (276 routing, 40 oracle) were kept as part of the full sweep; they ran with identical settings.
4. M1's test AP is printed in `train_mixed.json` as information only, as in the benchmark; it was not used for any choice.
5. The forecast evaluation took 64 min against an estimate of 10–25 min; the total still fit the budget.

## 7. Commits
| commit | hash | content |
|---|---|---|
| (a) | `ad6e740` | local Concern 1 build (rev5_capterms*, rev5_stats, rev5_external_check, nf_exp switches, test_capacity_terms, make_dataset, README hunk 1, `*.lnk` ignore) |
| (b) | `78770cc` | copied Concern 1–2 files (rev5_cap*, rev5_fb_compare, external check + mapping, diag with provenance note, both docx, README rows moved into the table, run_all.sh) |
| (c) | `73719fe` | `nf_exp.load_models` stem / tag / calibration + `tests/test_load_models.py` |
| (d1) | `40036fd` | Concern 3 protocol, code and tests, before any run |
| (d2) | the commit that adds this report | Concern 3 results, `checkpoints/c3/*.pt` + histories, README row, analysis fix |

Staged diffs of every commit: `commit_logs/<commit>.txt`. Left untracked as instructed: `plots/v3/*.png`, `research/evaluate_results_v4.json`.
The mixed-load dataset `data/datasets_c3_mixedload.pkl` (285.5 MB) is gitignored and regenerable with `c3_data.py`.
