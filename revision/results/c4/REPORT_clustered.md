# Concern 4 follow-up — family-clustered CIs for the closed-loop FCT results (Tables A6 and A10)

Analysis only: no new simulation runs. Input: `runs.jsonl` (2,160 runs: 9 settings × 8 strategies × 30 cells). Script: `revision/scripts/c4_clustered.py`. Output: `clustered.json` (every interval, verdict, per-family number), `tables_clustered.md` (all contrasts, verdict changes, per-family tables). Test: `tests/test_c4_clustered.py`.

**Method.** The FCT extraction, the censoring of unfinished flows and the pooled mean / p99 are copied unchanged from `rev4_c4_analyze.py`, and the interval and verdict functions from `c4_analyze.py`. Neither module is imported. Both bootstraps use 2,000 draws with seed 42, the same sampled cells for every strategy in a draw, and the pooled statistic over the concatenated flows of the sampled cells. **Cell** (existing): the 30 (family, seed) cells are resampled. **Clustered** (new): the 6 families are resampled with replacement, then the 5 seeds of each drawn family are resampled with replacement. A family drawn twice gets independent seed draws. The random-number calls are made in the same order as `summarize_all.boot(cluster=True)`. Contrast = FCT(strategy) − FCT(reference) in ticks and % (positive = strategy slower). The verdict uses the tick CI as in `c4_analyze.py`: **+** entire CI > 0, **−** entire CI < 0, **0** includes 0. The % CI gives the same verdict as the tick CI in every one of the comparisons, under both bootstraps, so the % CIs are shown below. In 98.0% of clustered draws at least one family appears more than once, and a draw contains 3.99 distinct families on average.

## Summary
- **Gate: pass.** The single-stage path reproduces 2,020 saved values exactly: `c4/summary.json` (points, tick and % CIs, rounded CIs, verdicts; 9 settings × all contrasts), `rev4_c4_summary.json` (Table A6, both blocks) and `rev4_fb_summary.json` `closed` (FlowBender-style rows).
- **Mean FCT: the guard advantage holds under clustering.** Against score without rules, FlowBender-style and ECMP, the clustered CI is entirely above 0 in all 9 settings (27 of 27). The clustered intervals are 1.8–2.7× as wide for these 27 contrasts (1.0–4.3× over all contrasts, median 2.3×) and their lower ends move toward 0. The smallest are FlowBender-style with no penalty, [+0.4, +7.3]%, and ECMP with no penalty, [+0.6, +9.1]%.
- **52 verdicts change** (14 vs guarded score, 38 vs ECMP). **Every change goes from detected (+ or −) to not detected (0).** No verdict flips sign, and no comparison that was not detected becomes detected.
- **p99 FCT loses 12 of its 22 detections vs guarded score.** The ECMP p99 detections at W0, W4, `rtt4` and dup3 are lost, as are FlowBender-style at dup3 and reactive at 6 of 9 settings. **Score without rules p99 stays detected at W0 and dup3**, the only settings where the cell CI detected it, so the crossover bracket (0, 1] is unchanged.
- **Guarded score vs ECMP:** the mean-FCT reduction stays detected in every setting. The p99 reductions detected by the cell CI (W0, W4, `rtt4`, dup3) are **not detected** under clustering, and this includes the Table A6 full-penalty value −4.4%.
- **Many vs-ECMP detections are lost**: reactive and unguarded GNN mean FCT at most settings, FlowBender-style mean at W0, W1 and dup3, FlowBender-style p99 (cell: better than ECMP) at off, W4, W8, W16, and score without rules mean at W2.
- **Two families drive the pooled mean contrasts: mixed** (43.5% of all flows, 47–70% of the pooled contrast vs guarded score for the three primary comparators) **and hotspot** (22–49% for score without rules and FlowBender-style). Stride and LLM-training contribute ≤ 3%. No single family, when left out, changes the sign of any primary pooled mean contrast.

## 1. Gate
| check | result |
|---|---|
| single-stage path vs `c4/summary.json`, `rev4_c4_summary.json`, `rev4_fb_summary.json` closed | **pass**: 2,020 values equal exactly, 0 mismatches |
| unit tests (`tests/test_c4_clustered.py`, 7) | pass: cell draws = `c4_analyze` sequence; clustered draws = `summarize_all.boot(cluster=True)` (500 draws, |Δ| ≤ 1e-12); constant shift → degenerate CI; per-family contributions sum to the pooled Δ; import writes nothing |

## 2. Results per setting
Values are % of the reference with 95% CIs. **v** = cell verdict → clustered verdict; **≠** marks a change.

### 2a. Primary comparators vs guarded score (positive = longer FCT than guarded score)

**off (no penalty)**

| comparator | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| score without rules | +5.3% | [+3.6, +7.3] | [+1.9, +9.8] | + → + | −0.8% | [−4.4, +3.5] | [−6.4, +11.3] | 0 → 0 |
| FlowBender-style | +3.4% | [+1.8, +5.3] | [+0.4, +7.3] | + → + | −2.5% | [−8.0, +1.4] | [−8.6, +4.8] | 0 → 0 |
| ECMP | +4.0% | [+2.4, +5.8] | [+0.6, +9.1] | + → + | +6.7% | [−1.0, +10.3] | [−10.3, +15.4] | 0 → 0 |

**W0 (full penalty)**

| comparator | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| score without rules | +23.1% | [+21.0, +25.5] | [+17.0, +26.3] | + → + | +11.0% | [+4.6, +19.9] | [+5.0, +28.0] | + → + |
| FlowBender-style | +12.4% | [+9.8, +14.5] | [+3.0, +15.1] | + → + | +1.4% | [−0.6, +10.0] | [−1.6, +13.5] | 0 → 0 |
| ECMP | +3.6% | [+2.2, +5.3] | [+0.8, +8.5] | + → + | +4.6% | [+0.4, +8.7] | [−8.6, +14.6] | + → 0 ≠ |

**W1**

| comparator | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| score without rules | +12.8% | [+10.7, +15.2] | [+9.6, +17.5] | + → + | +5.9% | [−1.2, +12.8] | [−1.1, +19.0] | 0 → 0 |
| FlowBender-style | +8.9% | [+6.9, +10.6] | [+1.7, +11.3] | + → + | −0.5% | [−3.5, +6.9] | [−4.7, +10.2] | 0 → 0 |
| ECMP | +3.7% | [+2.3, +5.4] | [+0.9, +8.8] | + → + | +4.8% | [−0.2, +8.3] | [−8.7, +15.3] | 0 → 0 |

**W2**

| comparator | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| score without rules | +9.8% | [+8.2, +11.8] | [+6.3, +13.7] | + → + | +3.7% | [+0.0, +10.7] | [−0.6, +13.4] | 0 → 0 |
| FlowBender-style | +5.2% | [+3.6, +6.8] | [+0.7, +7.9] | + → + | −1.6% | [−3.6, +5.5] | [−4.7, +7.6] | 0 → 0 |
| ECMP | +4.1% | [+2.6, +5.8] | [+0.6, +9.0] | + → + | +7.0% | [+0.0, +11.2] | [−9.5, +16.0] | 0 → 0 |

**W4**

| comparator | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| score without rules | +6.1% | [+4.0, +8.7] | [+2.9, +11.4] | + → + | +2.2% | [−2.3, +7.3] | [−3.3, +13.2] | 0 → 0 |
| FlowBender-style | +4.1% | [+2.3, +6.0] | [+0.3, +7.9] | + → + | +0.6% | [−4.7, +5.6] | [−6.1, +8.8] | 0 → 0 |
| ECMP | +4.1% | [+2.8, +5.7] | [+1.1, +9.2] | + → + | +8.7% | [+0.2, +13.0] | [−6.9, +15.4] | + → 0 ≠ |

**W8**

| comparator | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| score without rules | +5.2% | [+3.4, +7.4] | [+1.8, +10.2] | + → + | +0.4% | [−4.6, +3.9] | [−5.4, +12.8] | 0 → 0 |
| FlowBender-style | +3.5% | [+1.8, +5.4] | [+0.4, +7.6] | + → + | −2.5% | [−8.0, +1.7] | [−8.6, +7.1] | 0 → 0 |
| ECMP | +4.0% | [+2.5, +5.9] | [+0.7, +9.3] | + → + | +6.7% | [−1.0, +10.7] | [−9.7, +16.0] | 0 → 0 |

**W16**

| comparator | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| score without rules | +5.3% | [+3.6, +7.3] | [+1.9, +9.8] | + → + | −0.8% | [−4.4, +3.5] | [−6.4, +11.3] | 0 → 0 |
| FlowBender-style | +3.4% | [+1.8, +5.3] | [+0.4, +7.3] | + → + | −2.5% | [−8.0, +1.4] | [−8.6, +4.8] | 0 → 0 |
| ECMP | +4.0% | [+2.4, +5.8] | [+0.6, +9.1] | + → + | +6.7% | [−1.0, +10.3] | [−10.3, +15.4] | 0 → 0 |

**rtt4 (RTT/4)**

| comparator | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| score without rules | +6.6% | [+4.5, +8.8] | [+2.8, +11.0] | + → + | −1.2% | [−3.7, +8.2] | [−4.5, +12.9] | 0 → 0 |
| FlowBender-style | +4.5% | [+2.6, +6.6] | [+0.5, +8.4] | + → + | −2.1% | [−3.6, +7.1] | [−4.9, +10.6] | 0 → 0 |
| ECMP | +4.1% | [+2.8, +5.8] | [+1.3, +9.2] | + → + | +6.3% | [+1.2, +11.2] | [−7.4, +15.4] | + → 0 ≠ |

**dup3 (3 dupACKs)**

| comparator | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| score without rules | +22.7% | [+20.7, +25.0] | [+17.1, +25.7] | + → + | +11.1% | [+4.0, +19.8] | [+4.1, +27.4] | + → + |
| FlowBender-style | +12.1% | [+9.6, +14.0] | [+2.9, +14.6] | + → + | +2.8% | [+0.9, +10.3] | [−1.8, +13.8] | + → 0 ≠ |
| ECMP | +3.6% | [+2.1, +5.3] | [+0.6, +8.4] | + → + | +4.6% | [+0.4, +8.5] | [−8.5, +14.6] | + → 0 ≠ |

### 2b. Remaining strategies vs guarded score (compact: Δ% [clustered CI] cell-verdict clustered-verdict)

| setting | reactive mean | reactive p99 | unguarded GNN mean | unguarded GNN p99 | guarded GNN mean | guarded GNN p99 | guarded persistence mean | guarded persistence p99 |
|---|---|---|---|---|---|---|---|---|
| off | +14.6 [+1.9, +17.8] ++ | +4.9 [−3.8, +13.0] +0 | +10.0 [+3.2, +12.5] ++ | +4.6 [−2.5, +13.5] 00 | +0.2 [−1.2, +1.5] 00 | +0.9 [−8.7, +6.3] 00 | +0.2 [−0.7, +1.0] 00 | −2.8 [−9.1, +5.5] 00 |
| W0 | +31.2 [+8.7, +37.1] ++ | +16.6 [+3.7, +25.4] ++ | +27.4 [+15.3, +30.0] ++ | +15.1 [+5.3, +27.8] ++ | +0.4 [−0.4, +1.6] 00 | +1.2 [−7.3, +8.3] 00 | +0.5 [−0.2, +1.0] +0 | −1.1 [−3.9, +9.9] 00 |
| W1 | +29.1 [+6.1, +35.5] ++ | +15.3 [+1.4, +22.7] ++ | +24.5 [+11.9, +27.1] ++ | +11.7 [+0.6, +21.3] ++ | −0.1 [−0.6, +1.8] 00 | −1.2 [−6.4, +5.3] 00 | +0.3 [−1.0, +1.1] 00 | −1.3 [−10.5, +6.0] 00 |
| W2 | +27.0 [+3.8, +33.9] ++ | +15.6 [−0.8, +22.5] +0 | +20.4 [+9.0, +23.4] ++ | +14.4 [−0.4, +22.6] +0 | +0.5 [−0.7, +1.7] 00 | +4.5 [−8.8, +9.6] 00 | +0.9 [+0.0, +1.5] ++ | −0.5 [−6.0, +5.2] 00 |
| W4 | +23.2 [+2.7, +29.4] ++ | +17.1 [−1.3, +22.4] +0 | +13.7 [+5.3, +16.3] ++ | +10.0 [+0.7, +15.9] ++ | +0.1 [−0.7, +1.5] 00 | +1.0 [−5.6, +6.3] 00 | +0.3 [−0.3, +1.6] 00 | +1.0 [−5.6, +9.1] 00 |
| W8 | +17.7 [+2.1, +22.1] ++ | +8.8 [−3.3, +15.2] +0 | +10.5 [+3.5, +13.6] ++ | +7.9 [−2.8, +15.5] 00 | +0.1 [−1.0, +1.4] 00 | +0.1 [−7.9, +8.6] 00 | +0.4 [−0.6, +1.2] 00 | +0.0 [−6.5, +6.0] 00 |
| W16 | +15.7 [+1.9, +19.3] ++ | +7.9 [−3.7, +13.4] +0 | +9.9 [+3.2, +12.5] ++ | +4.7 [−2.5, +13.3] 00 | +0.2 [−1.2, +1.5] 00 | +0.9 [−8.7, +6.3] 00 | +0.2 [−0.7, +1.0] 00 | −2.8 [−9.1, +5.5] 00 |
| rtt4 | +23.0 [+2.5, +29.0] ++ | +11.6 [−0.3, +17.7] +0 | +15.4 [+5.5, +18.5] ++ | +9.3 [+0.4, +20.9] ++ | +0.2 [−0.5, +1.5] 00 | −3.5 [−4.9, +6.6] 00 | +0.4 [−0.2, +1.3] 00 | −0.3 [−7.8, +7.5] 00 |
| dup3 | +31.1 [+8.7, +36.8] ++ | +15.3 [+2.6, +24.2] ++ | +27.1 [+15.5, +29.6] ++ | +13.2 [+2.5, +25.9] ++ | +0.3 [−0.6, +1.6] 00 | +1.2 [−7.3, +8.0] 00 | +0.5 [−0.4, +1.0] +0 | −0.7 [−3.3, +8.0] 00 |

### 2c. Every strategy vs ECMP (compact: Δ% [clustered CI] cell-verdict clustered-verdict; negative = faster than ECMP)

**mean FCT**

| setting | reactive | unguarded GNN | score without rules | FlowBender-style | guarded GNN | guarded persistence | guarded score |
|---|---|---|---|---|---|---|---|
| off | +10.2 [−5.7, +13.9] +0 | +5.8 [−3.5, +10.4] +0 | +1.3 [−5.2, +8.3] 00 | −0.6 [−7.0, +6.1] 00 | −3.7 [−7.2, −1.7] −− | −3.6 [−7.9, −0.9] −− | −3.8 [−8.4, −0.6] −− |
| W0 | +26.6 [+2.2, +32.5] ++ | +22.9 [+8.2, +26.3] ++ | +18.8 [+9.2, +24.4] ++ | +8.5 [−3.6, +12.5] +0 | −3.1 [−6.5, −0.5] −− | −3.0 [−7.2, −0.7] −− | −3.5 [−7.8, −0.8] −− |
| W1 | +24.6 [−0.6, +30.9] +0 | +20.0 [+4.9, +23.5] ++ | +8.8 [+2.2, +15.5] ++ | +5.0 [−5.3, +9.5] +0 | −3.6 [−6.5, −0.9] −− | −3.3 [−7.3, −1.0] −− | −3.5 [−8.1, −0.9] −− |
| W2 | +22.0 [−3.4, +28.4] +0 | +15.7 [+2.2, +19.5] ++ | +5.5 [−0.9, +12.3] +0 | +1.0 [−6.8, +6.7] 00 | −3.4 [−6.8, −1.2] −− | −3.1 [−7.6, +0.2] −0 | −3.9 [−8.3, −0.6] −− |
| W4 | +18.3 [−4.5, +24.4] +0 | +9.3 [−1.6, +13.9] +0 | +1.9 [−4.2, +9.3] 00 | +0.0 [−7.1, +6.1] 00 | −3.8 [−7.1, −1.4] −− | −3.7 [−7.8, −0.3] −− | −3.9 [−8.4, −1.1] −− |
| W8 | +13.1 [−5.3, +17.4] +0 | +6.2 [−3.3, +11.6] +0 | +1.1 [−5.3, +8.5] 00 | −0.6 [−7.0, +6.1] 00 | −3.8 [−7.2, −1.6] −− | −3.5 [−7.8, −0.9] −− | −3.9 [−8.5, −0.7] −− |
| W16 | +11.3 [−5.7, +15.1] +0 | +5.7 [−3.5, +10.5] +0 | +1.3 [−5.2, +8.3] 00 | −0.6 [−7.0, +6.1] 00 | −3.7 [−7.2, −1.7] −− | −3.6 [−7.9, −0.9] −− | −3.8 [−8.4, −0.6] −− |
| rtt4 | +18.2 [−4.8, +24.1] +0 | +10.9 [−1.4, +15.2] +0 | +2.4 [−4.4, +8.8] 00 | +0.4 [−7.0, +6.6] 00 | −3.7 [−7.1, −1.3] −− | −3.6 [−7.8, −0.9] −− | −3.9 [−8.4, −1.3] −− |
| dup3 | +26.6 [+2.3, +32.2] ++ | +22.7 [+8.6, +26.3] ++ | +18.4 [+9.2, +24.1] ++ | +8.2 [−3.6, +12.3] +0 | −3.2 [−6.5, −0.5] −− | −3.0 [−7.2, −0.7] −− | −3.5 [−7.8, −0.6] −− |

**p99 FCT**

| setting | reactive | unguarded GNN | score without rules | FlowBender-style | guarded GNN | guarded persistence | guarded score |
|---|---|---|---|---|---|---|---|
| off | −1.7 [−12.2, +15.9] 00 | −2.0 [−13.5, +16.8] 00 | −7.0 [−11.5, +18.7] 00 | −8.6 [−14.7, +13.4] −0 | −5.5 [−10.6, +5.2] 00 | −8.9 [−12.1, +8.6] −0 | −6.3 [−13.4, +11.5] 00 |
| W0 | +11.5 [−0.1, +21.9] +0 | +10.1 [+3.1, +24.1] ++ | +6.1 [+1.9, +28.8] ++ | −3.0 [−10.1, +18.2] 00 | −3.3 [−12.2, +10.4] 00 | −5.4 [−10.5, +13.9] 00 | −4.4 [−12.7, +9.4] −0 |
| W1 | +10.0 [−3.4, +19.7] +0 | +6.6 [+0.0, +20.8] ++ | +1.0 [−3.1, +24.2] 00 | −5.1 [−11.9, +16.0] 00 | −5.8 [−11.9, +7.1] 00 | −5.8 [−10.9, +7.1] 00 | −4.6 [−13.3, +9.5] 00 |
| W2 | +8.1 [−7.9, +17.8] +0 | +7.0 [−4.9, +19.8] +0 | −3.0 [−6.7, +19.0] 00 | −8.0 [−12.9, +16.9] 00 | −2.3 [−10.7, +10.2] 00 | −7.0 [−12.1, +11.8] −0 | −6.5 [−13.8, +10.5] 00 |
| W4 | +7.8 [−9.8, +16.4] +0 | +1.2 [−10.6, +20.5] 00 | −6.0 [−11.7, +18.3] 00 | −7.4 [−15.0, +13.7] −0 | −7.0 [−10.5, +7.8] −0 | −7.0 [−12.0, +11.4] 00 | −8.0 [−13.4, +7.5] −0 |
| W8 | +2.0 [−11.9, +16.3] 00 | +1.1 [−12.4, +16.7] 00 | −5.9 [−12.3, +19.2] 00 | −8.6 [−14.7, +13.4] −0 | −6.2 [−10.5, +7.9] 00 | −6.2 [−12.1, +8.6] −0 | −6.3 [−13.8, +10.7] 00 |
| W16 | +1.1 [−12.2, +16.0] 00 | −1.9 [−13.5, +16.9] 00 | −7.0 [−11.5, +18.7] 00 | −8.6 [−14.7, +13.4] −0 | −5.5 [−10.6, +5.2] 00 | −8.9 [−12.1, +8.6] −0 | −6.3 [−13.4, +11.5] 00 |
| rtt4 | +5.0 [−10.4, +16.1] +0 | +2.8 [−8.9, +17.9] 00 | −7.0 [−11.9, +18.1] 00 | −7.9 [−14.4, +16.7] 00 | −9.2 [−10.9, +7.8] −0 | −6.2 [−12.0, +6.7] 00 | −5.9 [−13.4, +7.9] −0 |
| dup3 | +10.3 [−0.5, +20.5] +0 | +8.2 [+1.7, +24.6] ++ | +6.2 [+1.8, +26.2] ++ | −1.7 [−10.1, +18.2] 00 | −3.3 [−12.2, +10.4] 00 | −5.1 [−10.5, +13.7] 00 | −4.4 [−12.7, +9.3] −0 |

Cell CIs for 2b/2c are in `tables_clustered.md` and `clustered.json`.

## 3. Verdict changes (cell → clustered): 52

Every change is to **not detected**. Grouped:

| reference | strategy | metric | cell verdict | settings where the clustered CI includes 0 |
|---|---|---|---|---|
| guarded score | ECMP | p99 | + | W0, W4, rtt4, dup3 |
| guarded score | reactive | p99 | + | off, W2, W4, W8, W16, rtt4 |
| guarded score | unguarded GNN | p99 | + | W2 |
| guarded score | FlowBender-style | p99 | + | dup3 |
| guarded score | guarded persistence | mean | + | W0, dup3 |
| ecmp | reactive | mean | + | off, W1, W2, W4, W8, W16, rtt4 |
| ecmp | reactive | p99 | + | W0, W1, W2, W4, rtt4, dup3 |
| ecmp | unguarded GNN | mean | + | off, W4, W8, W16, rtt4 |
| ecmp | unguarded GNN | p99 | + | W2 |
| ecmp | score without rules | mean | + | W2 |
| ecmp | FlowBender-style | mean | + | W0, W1, dup3 |
| ecmp | FlowBender-style | p99 | − | off, W4, W8, W16 |
| ecmp | guarded GNN | p99 | − | W4, rtt4 |
| ecmp | guarded persistence | mean | − | W2 |
| ecmp | guarded persistence | p99 | − | off, W2, W8, W16 |
| ecmp | guarded score | p99 | − | W0, W4, rtt4, dup3 |

Unchanged detections that matter for the paper: the mean-FCT advantage of guarded score over score without rules, FlowBender-style and ECMP in all 9 settings; the p99 advantage over score without rules at W0 and dup3; reactive and unguarded GNN mean FCT vs guarded score at every setting; score without rules worse than ECMP on mean FCT at W0, W1, dup3 and on p99 at W0, dup3; guarded score faster than ECMP on mean FCT at every setting.

## 4. Paper-ready values

### 4a. Table A6: each strategy vs ECMP, % with cell CI and clustered CI

**No penalty (off)**

| strategy | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| reactive | +10.2% | [+5.6, +12.8] | [−5.7, +13.9] | + → 0 ≠ | −1.7% | [−4.5, +4.8] | [−12.2, +15.9] | 0 → 0 |
| unguarded GNN | +5.8% | [+2.8, +8.1] | [−3.5, +10.4] | + → 0 ≠ | −2.0% | [−7.1, +6.4] | [−13.5, +16.8] | 0 → 0 |
| score without rules | +1.3% | [−1.1, +4.1] | [−5.2, +8.3] | 0 → 0 | −7.0% | [−9.3, +1.2] | [−11.5, +18.7] | 0 → 0 |
| FlowBender-style | −0.6% | [−2.9, +2.1] | [−7.0, +6.1] | 0 → 0 | −8.6% | [−11.5, −0.4] | [−14.7, +13.4] | − → 0 ≠ |
| guarded GNN | −3.7% | [−5.0, −2.6] | [−7.2, −1.7] | − → − | −5.5% | [−9.5, +0.0] | [−10.6, +5.2] | 0 → 0 |
| guarded persistence | −3.6% | [−5.0, −2.5] | [−7.9, −0.9] | − → − | −8.9% | [−11.2, −0.6] | [−12.1, +8.6] | − → 0 ≠ |
| guarded score | −3.8% | [−5.5, −2.3] | [−8.4, −0.6] | − → − | −6.3% | [−9.3, +1.0] | [−13.4, +11.5] | 0 → 0 |

**Full penalty (W0)**

| strategy | mean Δ% | cell CI | clustered CI | v | p99 Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|---|---|---|---|
| reactive | +26.6% | [+19.1, +30.5] | [+2.2, +32.5] | + → + | +11.5% | [+5.8, +17.2] | [−0.1, +21.9] | + → 0 ≠ |
| unguarded GNN | +22.9% | [+19.1, +25.2] | [+8.2, +26.3] | + → + | +10.1% | [+4.8, +21.1] | [+3.1, +24.1] | + → + |
| score without rules | +18.8% | [+15.6, +21.8] | [+9.2, +24.4] | + → + | +6.1% | [+1.7, +14.2] | [+1.9, +28.8] | + → + |
| FlowBender-style | +8.5% | [+4.9, +11.0] | [−3.6, +12.5] | + → 0 ≠ | −3.0% | [−5.6, +4.2] | [−10.1, +18.2] | 0 → 0 |
| guarded GNN | −3.1% | [−4.5, −1.9] | [−6.5, −0.5] | − → − | −3.3% | [−8.7, +0.0] | [−12.2, +10.4] | 0 → 0 |
| guarded persistence | −3.0% | [−4.5, −1.7] | [−7.2, −0.7] | − → − | −5.4% | [−7.2, +2.3] | [−10.5, +13.9] | 0 → 0 |
| guarded score | −3.5% | [−5.1, −2.1] | [−7.8, −0.8] | − → − | −4.4% | [−8.0, −0.4] | [−12.7, +9.4] | − → 0 ≠ |

Table A6 also lists contrasts vs guarded score; for off and W0 they are in 2a/2b above.

### 4b. Table A10 columns (reordering-model sweep, contrasts vs guarded score)

The repo copies of the paper (`paper/*.docx`) predate Table A10. This assumes A10 is the reordering-model crossover table (`c4/REPORT.md` §4: verdicts along W0 W1 W2 W4 W8 W16 off, plus `rtt4` and dup3). Verdict strings in the order **W0 W1 W2 W4 W8 W16 off rtt4 dup3**:

| comparator | metric | cell | clustered |
|---|---|---|---|
| ECMP | mean | `+ + + + + + + + +` | `+ + + + + + + + +` |
| ECMP | p99 | `+ 0 0 + 0 0 0 + +` | `0 0 0 0 0 0 0 0 0` ≠ |
| reactive | mean | `+ + + + + + + + +` | `+ + + + + + + + +` |
| reactive | p99 | `+ + + + + + + + +` | `+ + 0 0 0 0 0 0 +` ≠ |
| unguarded GNN | mean | `+ + + + + + + + +` | `+ + + + + + + + +` |
| unguarded GNN | p99 | `+ + + + 0 0 0 + +` | `+ + 0 + 0 0 0 + +` ≠ |
| score without rules | mean | `+ + + + + + + + +` | `+ + + + + + + + +` |
| score without rules | p99 | `+ 0 0 0 0 0 0 0 +` | `+ 0 0 0 0 0 0 0 +` |
| FlowBender-style | mean | `+ + + + + + + + +` | `+ + + + + + + + +` |
| FlowBender-style | p99 | `0 0 0 0 0 0 0 0 +` | `0 0 0 0 0 0 0 0 0` ≠ |
| guarded GNN | mean | `0 0 0 0 0 0 0 0 0` | `0 0 0 0 0 0 0 0 0` |
| guarded GNN | p99 | `0 0 0 0 0 0 0 0 0` | `0 0 0 0 0 0 0 0 0` |
| guarded persistence | mean | `+ 0 + 0 0 0 0 0 +` | `0 0 + 0 0 0 0 0 0` ≠ |
| guarded persistence | p99 | `0 0 0 0 0 0 0 0 0` | `0 0 0 0 0 0 0 0 0` |

The Δ% with both CIs for each column is in 2a (primary comparators) and 2b (the others).

### 4c. Numbers quoted in V-G

| quantity | setting | Δ% | Δ ticks | cell CI (%) | clustered CI (%) | clustered CI (ticks) | v |
|---|---|---|---|---|---|---|---|
| no-rules vs guard, mean, no penalty ("5.3% [+3.6, +7.3]") | off | +5.3% | +6.5 | [+3.6, +7.3] | [+1.9, +9.8] | [+1.0, +11.9] | + → + |
| no-rules vs guard, mean, full penalty ("23.1%") | W0 | +23.1% | +28.3 | [+21.0, +25.5] | [+17.0, +26.3] | [+7.7, +40.5] | + → + |
| FlowBender-style vs guard, mean, full penalty ("12.4%") | W0 | +12.4% | +15.1 | [+9.8, +14.5] | [+3.0, +15.1] | [+1.5, +23.4] | + → + |
| FlowBender-style vs guard, mean, no penalty ("3.4%") | off | +3.4% | +4.1 | [+1.8, +5.3] | [+0.4, +7.3] | [+0.2, +8.8] | + → + |
| ECMP vs guard, mean, no penalty | off | +4.0% | +4.8 | [+2.4, +5.8] | [+0.6, +9.1] | [+0.5, +7.8] | + → + |
| ECMP vs guard, mean, full penalty | W0 | +3.6% | +4.4 | [+2.2, +5.3] | [+0.8, +8.5] | [+0.6, +7.3] | + → + |
| tail: no-rules vs guard, p99, full penalty | W0 | +11.0% | +52.6 | [+4.6, +19.9] | [+5.0, +28.0] | [+22.4, +90.0] | + → + |
| tail: no-rules vs guard, p99, dup3 | dup3 | +11.1% | +53.2 | [+4.0, +19.8] | [+4.1, +27.4] | [+20.3, +87.0] | + → + |
| tail: no-rules vs guard, p99, W1 (first W not detected) | W1 | +5.9% | +28.3 | [−1.2, +12.8] | [−1.1, +19.0] | [−5.8, +64.9] | 0 → 0 |
| tail: no-rules vs guard, p99, no penalty | off | −0.8% | −3.7 | [−4.4, +3.5] | [−6.4, +11.3] | [−23.1, +31.0] | 0 → 0 |
| tail: FlowBender-style vs guard, p99, full penalty | W0 | +1.4% | +6.8 | [−0.6, +10.0] | [−1.6, +13.5] | [−5.2, +50.8] | 0 → 0 |
| tail: FlowBender-style vs guard, p99, dup3 | dup3 | +2.8% | +13.6 | [+0.9, +10.3] | [−1.8, +13.8] | [−6.0, +50.0] | + → 0 ≠ |
| tail: ECMP vs guard, p99, full penalty | W0 | +4.6% | +22.0 | [+0.4, +8.7] | [−8.6, +14.6] | [−31.6, +39.5] | + → 0 ≠ |
| quarter-RTT: no-rules vs guard, mean | rtt4 | +6.6% | +8.0 | [+4.5, +8.8] | [+2.8, +11.0] | [+1.5, +13.9] | + → + |
| quarter-RTT: no-rules vs guard, p99 | rtt4 | −1.2% | −5.7 | [−3.7, +8.2] | [−4.5, +12.9] | [−20.0, +49.3] | 0 → 0 |
| quarter-RTT: FlowBender-style vs guard, mean | rtt4 | +4.5% | +5.5 | [+2.6, +6.6] | [+0.5, +8.4] | [+0.3, +10.7] | + → + |
| quarter-RTT: FlowBender-style vs guard, p99 | rtt4 | −2.1% | −10.0 | [−3.6, +7.1] | [−4.9, +10.6] | [−18.0, +40.3] | 0 → 0 |
| quarter-RTT: ECMP vs guard, mean | rtt4 | +4.1% | +5.0 | [+2.8, +5.8] | [+1.3, +9.2] | [+1.1, +7.6] | + → + |
| quarter-RTT: ECMP vs guard, p99 | rtt4 | +6.3% | +29.7 | [+1.2, +11.2] | [−7.4, +15.4] | [−26.1, +46.7] | + → 0 ≠ |

V-G / Table A6 text values vs ECMP (from `NeuralFabric_C4_concern2.docx`, "Reordering penalty" paragraph):

| quantity | Δ% | cell CI | clustered CI | v |
|---|---|---|---|---|
| guarded score mean, full penalty ("3.5% below") | −3.5% | [−5.1, −2.1] | [−7.8, −0.8] | − → − |
| guarded score p99, full penalty ("4.4% below") | −4.4% | [−8.0, −0.4] | [−12.7, +9.4] | − → 0 ≠ |
| score without rules mean, full penalty ("18.8% above") | +18.8% | [+15.6, +21.8] | [+9.2, +24.4] | + → + |
| score without rules p99, full penalty ("+6.1%") | +6.1% | [+1.7, +14.2] | [+1.9, +28.8] | + → + |

Suggested wording, if the paper adds the clustered interval: on mean FCT, guarded score's advantage over score without rules, FlowBender-style and ECMP is detected in every reordering setting with both the cell and the family-clustered bootstrap. On p99 FCT, among these three comparators, the family-clustered bootstrap detects only the advantage over score without rules under the full penalty (W0) and dupACK 3. Guarded score's p99 reduction vs ECMP is not detected with family clustering in any setting.

## 5. Per-family breakdown (mean FCT vs guarded score)

Pooled Δ = Σ_f (flows_f / flows) · Δ_f exactly, because every strategy has the same flows in a cell. "Share" = a family's flow-weighted contribution as a fraction of the pooled Δ. Full per-setting tables (flows, Δ ticks, Δ%, seeds with Δ > 0, contribution, pooled Δ% without the family) are in `tables_clustered.md`; per-seed Δ and per-family p99 Δ are in `clustered.json`.

| family | flows (share) | score without rules: Δ% range / share range | FlowBender-style: Δ% range / share range | ECMP: Δ% range / share range |
|---|---|---|---|---|
| mixed | 838 (43.5%) | +4.1…+23.0% / 49…61% | +2.6…+14.1% / 47…70% | +3.7…+4.4% / 62…67% |
| hotspot | 250 (13.0%) | +11.4…+28.5% / 22…42% | +8.9…+16.3% / 24…49% | −0.2…+0.9% / −1…4% |
| elephant_mice | 240 (12.4%) | +5.1…+20.4% / 5…6% | +2.1…+6.3% / 3…4% | +14.5…+15.5% / 22…24% |
| uniform | 160 (8.3%) | +0.9…+22.3% / 1…8% | −0.0…+4.0% / −0…3% | +3.8…+4.6% / 9…10% |
| stride | 80 (4.1%) | +0.3…+16.1% / 0…3% | +0.0…+0.5% / 0…0% | +3.0…+3.1% / 3…3% |
| llm_training | 360 (18.7%) | +5.2…+5.2% / 0…2% | +0.7…+0.7% / 0…0% | −0.0…−0.0% / −0…−0% |

**off (no penalty): Δ ticks (Δ%) per family, and pooled Δ% without the family**

| comparator | mixed | hotspot | elephant_mice | uniform | stride | llm_training | pooled |
|---|---|---|---|---|---|---|---|
| score without rules | +7.59 (+4.4%); w/o +6.8% | +19.61 (+11.4%); w/o +3.9% | +3.14 (+5.4%); w/o +5.3% | +1.13 (+0.9%); w/o +5.7% | +0.30 (+0.3%); w/o +5.5% | +0.68 (+5.2%); w/o +5.3% | +6.46 (+5.3%) |
| FlowBender-style | +4.50 (+2.6%); w/o +4.7% | +15.29 (+8.9%); w/o +2.2% | +1.49 (+2.6%); w/o +3.5% | −0.01 (−0.0%); w/o +3.7% | +0.00 (+0.0%); w/o +3.5% | +0.10 (+0.7%); w/o +3.5% | +4.14 (+3.4%) |
| ECMP | +7.24 (+4.2%); w/o +3.6% | −0.20 (−0.1%); w/o +4.9% | +8.95 (+15.5%); w/o +3.3% | +5.57 (+4.6%); w/o +3.9% | +3.52 (+3.1%); w/o +4.0% | −0.00 (−0.0%); w/o +4.1% | +4.84 (+4.0%) |

**W0 (full penalty): Δ ticks (Δ%) per family, and pooled Δ% without the family**

| comparator | mixed | hotspot | elephant_mice | uniform | stride | llm_training | pooled |
|---|---|---|---|---|---|---|---|
| score without rules | +39.89 (+23.0%); w/o +23.3% | +48.90 (+28.5%); w/o +21.9% | +11.46 (+19.6%); w/o +23.4% | +27.25 (+22.3%); w/o +23.2% | +18.51 (+16.1%); w/o +23.4% | +0.67 (+5.2%); w/o +23.5% | +28.26 (+23.1%) |
| FlowBender-style | +24.39 (+14.1%); w/o +9.7% | +27.96 (+16.3%); w/o +11.5% | +3.69 (+6.3%); w/o +12.8% | +4.95 (+4.0%); w/o +13.2% | +0.54 (+0.5%); w/o +12.9% | +0.10 (+0.7%); w/o +12.6% | +15.14 (+12.4%) |
| ECMP | +6.48 (+3.7%); w/o +3.5% | +0.25 (+0.1%); w/o +4.4% | +8.47 (+14.5%); w/o +2.9% | +4.69 (+3.8%); w/o +3.6% | +3.47 (+3.0%); w/o +3.7% | −0.00 (−0.0%); w/o +3.7% | +4.43 (+3.6%) |

Reading:
- **Mixed drives every primary result.** It has the most flows (130–202 per seed vs 16–84 elsewhere), and Δ > 0 in 5/5 seeds for all three comparators in every setting. It accounts for 47–70% of each pooled mean contrast. Because the pooled statistic weights families by flow count, a clustered draw with mixed 0 times or 2+ times moves the pooled contrast a lot, which is a main source of the wider clustered CIs.
- **Hotspot is the second driver for score without rules and FlowBender-style** (22–49% of the pooled Δ, +8.9 to +28.5%), but contributes almost nothing vs ECMP (−1 to +4%: on hotspot, guarded score and ECMP have about the same mean FCT, and the seeds go both ways). Vs ECMP, **elephant-mice** is the second driver (22–24%), dominated by seeds 3000 and 3004.
- **Leave-one-family-out:** the smallest pooled mean contrasts left are score without rules +3.7% (without hotspot, W8), FlowBender-style +2.2% (without hotspot, W16), ECMP +2.9% (without elephant_mice, dup3). All stay positive, so no single family produces a primary result by itself.
- **Stride contributes almost no information** (4.1% of flows; 0–3% of any pooled contrast). FlowBender-style and guarded score have identical stride FCT on all 5 seeds in every setting except W0 and dup3 (Δ = 0). Correction to the brief: stride seeds share the same 16 (source, destination, flow ID) triples and therefore the same ECMP paths, but not the same flows. Sizes, rates and start ticks differ by seed, and so do ECMP fabric drops (4,832, 543, 2,070, 826, 5,314 on seeds 3000–3004) and FCTs. Under the penalty, stride is not empty for score without rules (W0: +16.1%, 5/5 seeds), but its weight is small.
- **LLM-training contributes almost no information** (18.7% of flows but ≤ 2% of any pooled contrast). ECMP has 0 fabric drops on every seed, guarded score never needs to act, and guarded score = ECMP exactly on 4 of 5 seeds (the fifth differs by 0.01 ticks). Score without rules reroutes anyway and is +5.2% slower there, but on ~13-tick flows that is only +0.68 ticks.
- **Uniform** carries 0–10% of each contrast; at off, W4, W8, W16 and `rtt4`, FlowBender-style ≈ guarded score there (|Δ| ≤ 0.5 ticks).

So the clustered bootstrap effectively has 2–3 informative clusters (mixed, hotspot, plus elephant-mice vs ECMP) out of 6. Percentile intervals from 6 clusters are approximate. The clustered CIs should be read as a sensitivity check on between-family heterogeneity, not as exact 95% coverage.

## 6. Runtime, deviations, commit
- Analysis runtime: 110 s (9 settings × 8 strategies × 2 bootstraps × 2,000 draws, one process). Full test suite: 105 tests, OK (67 s test time, 108 s wall).
- No protected file changed: no file in `revision/results/` (other than the three new outputs), `checkpoints/`, `data/` or `research/` was written. The paper was not touched.
- Deviations: none in method. Two additions beyond the brief: the gate also covers `rev4_fb_summary.json` (FlowBender-style rows of Table A6), and the per-family section includes leave-one-family-out pooled contrasts and exact contribution shares. The stride premise in the brief ("identical flows") is corrected in §5. Table A10 is not in the repo copies of the paper, so §4b states the assumed column layout.
- Commit: the commit that adds this report (script, test, three result files, README row).
