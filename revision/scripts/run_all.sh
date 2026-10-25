#!/usr/bin/env bash
# Re-runs every analysis and experiment added in the revision, in dependency order.
# Usage (from the repo root):  bash revision/scripts/run_all.sh
# NF_ROOT defaults to the repo root; set it only if the project lives elsewhere.
# Experiment groups are resumable: finished tasks already in results/<group>.jsonl are skipped,
# so delete a group's .jsonl file to recompute it from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NF_ROOT:=$(cd .. && pwd)}"
export NF_ROOT
W=${WORKERS:-2}

# 1. Exactness checks against the stored benchmark (research/benchmark_results_v4.json)
python3 scripts/verify_modelfree.py 3000 3001
python3 scripts/run_exps.py verify_model --workers "$W"

# 2. Re-analysis of the stored benchmark and of the forecasting data
python3 scripts/analyze_existing.py "$NF_ROOT/research/benchmark_results_v4.json" results/existing_analysis.json
python3 scripts/leakage_check.py "$NF_ROOT" results/leakage_check.json
python3 scripts/forecast_analysis.py results/forecast_analysis.json
python3 scripts/platt_params.py
python3 scripts/episodes.py

# 3. New routing experiments (about 1.5-2 hours with two workers)
for g in replicate ablation sweep delay interval instrument closedloop platt; do
  python3 scripts/run_exps.py "$g" --workers "$W"
done
python3 scripts/run_exps.py timing --workers 1 --threads 2
python3 scripts/timing_original.py

# 4. Statistics, tables and figures
python3 scripts/equiv.py
python3 scripts/summarize_all.py
python3 scripts/fig_reliability.py
python3 scripts/fig_new.py
# 5. Second review round
for g in addone noguard controls; do python3 scripts/rev2_exps.py "$g" "$W"; done
python3 scripts/rev2_analyze.py
python3 scripts/rev3_load.py "$W"
python3 scripts/rev3_analyze.py
# 6. Third review round
python3 scripts/rev4_c1.py "$W"
python3 scripts/rev4_c2.py "$W"
python3 scripts/rev4_c1_analyze.py
python3 scripts/rev4_c4.py "$W"
python3 scripts/rev4_c4_analyze.py
python3 scripts/rev4_fb.py "$W"
python3 scripts/rev4_fb_analyze.py
python3 scripts/rev5_cap.py "$W"
python3 scripts/rev5_cap_analyze.py
python3 scripts/rev5_fb_compare.py
echo "done: see results/summary.json, results/rev*_summary.json and figs/"
