# NeuralFabric – revision experiments

Code, raw results and figures behind every number that was added to the paper in response to the
review. Everything runs on top of the unchanged NeuralFabric project (simulator, models, checkpoints
and `research/benchmark_results_v4.json`); nothing in the project folder is modified.

## Setup

1. This folder sits inside the NeuralFabric repo, and `NF_ROOT` defaults to the repo root. Set it
   only if the project lives somewhere else:
   ```
   export NF_ROOT=/path/to/NeuralFabric      # folder with config.py, simulation/, checkpoints/
   ```
2. Install the project requirements plus SciPy: `pip install -r $NF_ROOT/requirements.txt scipy`.
   The results here were produced with Python 3.13, PyTorch 2.14.1 (CPU), NumPy 2.5.3,
   SciPy 1.18.1, scikit-learn 1.9.1, NetworkX 3.6.1 and Matplotlib 3.11.2 on a 2-vCPU machine.

## Run

```
bash revision/scripts/run_all.sh   # from the repo root; everything, in dependency order (about 1.5-2 h with 2 workers)
python3 revision/scripts/run_exps.py delay  # or one experiment group (see the docstring of run_exps.py)
```

Experiment groups append to `results/<group>.jsonl` and are resumable: tasks already in the file are
skipped, so delete a file to recompute that group. The routing runs are deterministic; before the new
experiments, 156 runs of the stored benchmark (every strategy, seeds 3000-3001) were reproduced
exactly (`verify_modelfree.py`, `results/verify_model.jsonl`). Timing results depend on the machine.

## Where each part of the paper comes from

| Paper | Script | Result file(s) |
|---|---|---|
| V-A calibration: ECE, reliability diagram (Fig. 3), Platt scaling | `forecast_analysis.py`, `platt_params.py`, `fig_reliability.py` | `forecast_analysis.json`, `platt_params.json` |
| V-A/V-B test-scenario overlap (leakage) and per-scenario AP / sign test | `leakage_check.py`, `forecast_analysis.py` | `leakage_check.json`, `forecast_analysis.json` |
| V-C/V-D per-family results, TOST equivalence, family-clustered bootstrap on the stored benchmark (Tables VI, VII) | `analyze_existing.py`, `equiv.py` | `existing_analysis.json`, `equiv.json` |
| V-D fresh benchmark on unseen traffic seeds 4000-4009 | `run_exps.py replicate`, `equiv.py` | `replicate.jsonl`, `equiv.json` |
| V-E leave-one-rule-out ablation (Fig. 6) | `run_exps.py ablation`, `fig_new.py` | `ablation.jsonl` |
| V-E sensitivity to alpha, delta, kappa (Table VIII; the M rows come from the project's `research/guard_sweep_v4.json`) | `run_exps.py sweep` | `sweep.jsonl` |
| V-E routing with Platt-recalibrated probabilities | `run_exps.py platt` | `platt.jsonl` |
| V-F telemetry and actuation delay (Fig. 8), congestion episode lengths | `run_exps.py delay`, `episodes.py`, `fig_new.py` | `delay.jsonl`, `episodes.json` |
| V-G packet reordering and per-flow loss | `run_exps.py instrument` | `instrument.jsonl` |
| V-G closed-loop transport, FCT and slowdown (Table IX) | `run_exps.py closedloop` | `closedloop.jsonl` |
| V-H controller and inference time, coarser forecast interval | `run_exps.py timing`, `timing_original.py`, `run_exps.py interval` | `timing.jsonl`, `timing_original.json`, `interval.jsonl` |
| All of the above, aggregated with paired-bootstrap CIs | `summarize_all.py` | `summary.json` |
| Round 2: add-one-in rule ablation (V-E) | `rev2_exps.py addone`, `rev2_analyze.py` | `rev2_addone.jsonl`, `rev2_summary.json` |
| Round 2: score without rules in Tables V and IX and the reordering study | `rev2_exps.py noguard`, `rev2_analyze.py` | `rev2_noguard.jsonl`, `rev2_summary.json` |
| Round 2: random / static / oracle risk signals (V-F, "Control signals") | `rev2_exps.py controls`, `rev2_analyze.py` | `rev2_controls.jsonl`, `rev2_summary.json` |
| Round 2: FCT with unfinished flows censored at 600 ticks (V-G) | `rev2_analyze.py` | `rev2_summary.json` (`table9`) |
| Round 3/4: offered-load sweep, load factors 0.1–1.0 (V-F, Table A5) | `rev3_load.py`, `rev3_analyze.py` | `rev3_load.jsonl`, `rev3_load_summary.json` |
| Round 4: noisy / rate-blind / telemetry rates, signal-free guard, controls with reordering (Tables A3, A4) | `rev4_c1.py`, `rev4_c2.py`, `rev4_c1_analyze.py` | `rev4_c1.jsonl`, `rev4_c1_summary.json` |
| Round 4: closed loop with reordering penalty (Table A6) | `rev4_c4.py`, `rev4_c4_analyze.py` | `rev4_c4.jsonl`, `rev4_c4_summary.json` |
| Round 4: FlowBender-style baseline (V-I, Table V) | `rev4_fb.py`, `rev4_fb_analyze.py` | `rev4_fb.jsonl`, `rev4_fb_summary.json` |

Statistics: paired bootstrap with seed 42 and 2,000 resamples over (family, traffic seed) pairs;
the clustered variant resamples families first and seeds within families; TOST uses a ±1% of ECMP
total-drop margin with 90% intervals.

## Files

- `scripts/nf_exp.py` – experiment harness: guard with switchable admission rules, telemetry /
  actuation delay, forecast interval, reordering instrumentation, closed-loop transport (retransmission
  with AIMD), Platt-scaled forecaster.
- `scripts/run_exps.py` – experiment groups and the resumable process pool.
- `scripts/paths.py` – folder locations (`NF_ROOT`, optional `NF_WORK`).
- `results/log_*.txt` – console logs of the original runs.
- `figs/` – the three new figures (PNG at 300 dpi and PDF).
