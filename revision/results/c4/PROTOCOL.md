# Concern 4 (4th review) — a middle reordering model for the closed-loop FCT comparison: protocol

Written and committed with the code before any Concern 4 run. It is not edited after results exist; any deviation is
reported in `REPORT.md`.

## Question
Table A6 evaluates the closed-loop transport (`nf_exp.ClosedLoopEngine`) under two reordering models: **no penalty** and
the **full penalty** (every out-of-order batch is treated as a loss: half an RTT later the batch is retransmitted
spuriously and the rate is halved, at most once per RTT). The reviewer asks for a middle case that tolerates some
reordering, to show where the crossover lies. The aim is to measure the crossover, not to favour the guard; every result
is reported as it comes out.

## Reordering, RTT and the penalty (code as it stands, `revision/scripts/nf_exp.py`)
- A delivered, non-spurious **batch** (one `(count, payload)` departure from the last-hop link; FIFO service can split an
  injected batch into pieces with the same birth tick) is **reordered** iff `born < max_born`, the newest birth tick
  already delivered for that flow. Displacement `d = max_born − born ≥ 1` ticks. Spurious copies never update `max_born`.
- RTT of a path = `max(2, 2·hops)` ticks, `hops = len(path) − 1`. Host–host paths on the k=4 Fat-Tree always have an even
  hop count (hosts hang off one edge switch; links join adjacent layers only): 2 / 4 / 6 hops on shortest paths
  (RTT 4 / 8 / 12); rerouted paths can be longer (GuardX stretch ≤ 2, score-without-rules unbounded, reactive / unguarded
  weighted shortest paths), RTT 16+.
- Penalty mechanism (unchanged): a penalised batch of `count` packets, if its flow has not completed, is queued for
  spurious retransmission after half the RTT of the batch's path (`max(1, hops)` ticks); its maturing counts as a loss
  event for the multiplicative decrease. Only **which** reordered batches are penalised changes between settings.

## Settings (all closed loop, all else equal)
| setting | rule (a reordered batch of an incomplete flow is penalised iff …) | parameters |
|---|---|---|
| `off` | never | `reorder_penalty=False` (legacy default) |
| `W0` | always (`d > 0`) = today's full penalty | `reorder_penalty=True, reorder_window=0` |
| `W1`, `W2`, `W4`, `W8`, `W16` | `d > W` | `reorder_window=W` |
| `rtt4` | `d > RTT_b / 4`, tested as the integer `4·d > RTT_b` | `reorder_window='rtt/4'` |
| `dup3` | dupACK count ≥ 3 | `dupack_threshold=3` |

**`rtt/4`:** per delivered batch, `RTT_b` is the RTT of that batch's own path (the convention of the existing half-RTT
penalty delay and loss detection), not of the flow's current path. Because hop counts are even, `RTT_b / 4 = hops / 2` is
an exact integer: 1 / 2 / 3 ticks on shortest paths, ≥ 4 on stretched paths. No rounding is involved; the realised windows
are reported.

**dupACK count** of a reordered batch with birth tick `born`: the number of **packets** (not batches) of the **same flow**
that were **delivered before** this batch and have a **strictly greater** birth tick:
`Σ_{b = born+1 … max_born} born_hist[b]`, where `born_hist` (new per-flow state; `max_born` alone is not sufficient)
counts the non-spurious packets of the flow delivered so far by birth tick. Excluded: other flows, equal birth ticks, future
deliveries (not yet in `born_hist`), and spurious copies (consistent with the existing `max_born` handling: they are
duplicates of data already delivered). Every non-spurious delivery is added to `born_hist` right after its own decision.

**Same-tick ordering:** all deliveries of a flow leave through its single last-hop link (edge switch → destination host),
whose FIFO departures are processed in order; the existing code updates `max_born` immediately. Earlier deliveries in the
same tick therefore count as "delivered before", for both the window rule (`max_born`, hence `d`) and the dupACK rule.

**Parameters and precedence:** `reorder_window` / `dupack_threshold` require `reorder_penalty=True` and at most one of
them may be given; otherwise `ValueError`. No combination is silently ignored. `reorder_penalty=True` alone is the legacy
full penalty and `reorder_penalty=False` (default) the legacy no penalty; existing callers (`run_exps.py`, `rev2_exps.py`,
`rev4_c4.py`, `rev4_fb.py`) are unchanged. `run_case_x` rejects these parameters outside the closed loop.

## Design
- Strategies (8): ECMP; reactive; unguarded GNN; score without rules (score signal, threshold 0.851, all admission rules
  off); FlowBender-style (`k_frac=0.1, n_consec=2, rng_seed = 1000·family_index + seed`, as `rev4_fb.py`); guarded GNN;
  guarded persistence (0.5); guarded score (0.851). GNN = training seed 0 (`best_model_v3.pt`), 1 torch thread,
  checkpoint threshold for guarded GNN (as `rev4_c4.py`). Other strategies `rng_seed = 0`.
- Cells: benchmark traffic seeds 3000–3004 × 6 families (uniform, hotspot, elephant_mice, stride, llm_training, mixed);
  150-tick scenarios, closed loop runs until all flows complete or 600 ticks.
- 9 settings × 8 strategies × 30 cells = **2,160 runs** (`c4_sweep.py` → `runs.jsonl`). Each run asserts packet
  conservation and the drop split.
- Diagnostics (never read by the dynamics): reordered batches / packets (all and of incomplete flows), penalised batches /
  packets, histograms of reordered-batch packet count, displacement `d`, dupACK count and `RTT_b`.

## Gates (blockers; checked on `off` and `W0` before the other settings are run)
- **G1** `off` rows equal the saved no-penalty rows field for field (incl. `per_flow`): `closedloop.jsonl` (6 strategies),
  `rev2_noguard.jsonl` (`score_no_rules_cl`), `rev4_fb.jsonl` (closed loop, penalty off).
- **G2** `W0` rows equal `rev4_c4.jsonl` (7 strategies) and `rev4_fb.jsonl` (closed loop, penalty on).
- **G3** the `rev4_c4_analyze.py` / `rev4_fb_analyze.py` statistics recomputed from the new `off` and `W0` rows equal
  `rev4_c4_summary.json` and the `closed` block of `rev4_fb_summary.json` exactly (means, p99s, retransmission / spurious %,
  reroutes, % contrasts and their CIs).
- **G4** every run present once, no errors, packet conservation and drop split hold in all runs.
A gate failure stops the study.

## Metrics
Per strategy and setting (pooled over the 30 cells, as `rev4_c4_analyze.py`): mean and p99 FCT over flows with size > 0,
unfinished flows censored at the horizon (FCT = makespan − start, makespan = 600 when not all flows finish); spurious
retransmissions as % of offered packets (`offered_unique`); penalised share of reordered batches and packets (over all
reordered and over those of incomplete flows); the reordered-batch packet-count distribution.

## Bootstrap
- Resampling unit: the **(family, traffic seed) cell** (30), the paired cell of `rev4_c4_analyze.compare`.
- Each draw resamples 30 cells with replacement; both strategies of a contrast use the **same** sampled cells; pooled
  per-flow censored FCTs over the sampled cells (repeats included) give the draw's mean and p99.
- 2,000 draws, `numpy.random.default_rng(42)`, the identical draw sequence for every contrast and setting (as `compare`).
- 95% percentile interval (`np.percentile`, linear), unrounded for verdicts.

## Contrasts and verdicts
- **vs guarded score:** Δ = FCT(comparator) − FCT(guarded score), in ticks (primary) and as (ratio − 1)·100 % (Table A6
  form; same draws). Positive favours guarded score. Comparators: score without rules, FlowBender-style, ECMP (primary),
  and every other strategy.
- **vs ECMP:** FCT(strategy) − FCT(ECMP), same form.
- Separately for mean and p99 FCT, every setting:
  **+** = entire tick CI > 0 (guard advantage detected); **−** = entire CI < 0 (comparator detectably better than
  guarded score); **0** = CI includes 0 (not detected). No inference from point estimates. Any setting where the % CI gives
  a different verdict from the tick CI is flagged.

## Crossover rule
Ordered axis W = 0, 1, 2, 4, 8, 16, ∞ (`off`). For each comparator and metric, report independently
(a) the **largest W at which the guard's advantage is detected** and (b) the **smallest W at which it is not detected**;
they need not be adjacent. A single crossover bracket (a, b) is reported only if the verdicts are monotone in W (detected on
a prefix of the axis, not detected on the rest) **and** a and b are adjacent on the tested axis. Otherwise the non-monotone
pattern is stated and no single crossover is implied. If the advantage is detected at every W, or at none, the report says
no crossover can be established in the tested range. Settings where the comparator is detectably better (−) are listed.
`rtt4` and `dup3` are reported next to the axis with their realised windows / penalised shares; they are not placed on it.

## Outputs
`results/c4/`: `runs.jsonl`, `gate.json`, `summary.json`, `tables.md`, `runtime.json`, `REPORT.md`.
Scripts: `revision/scripts/c4_sweep.py`, `c4_analyze.py`. Tests: `tests/test_c4.py`.
Nothing in `checkpoints/`, `data/`, `research/` or existing `revision/results/*` files is written.
