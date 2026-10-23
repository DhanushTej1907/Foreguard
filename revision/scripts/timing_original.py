"""Per-tick computation time of the ORIGINAL controller code (routing/guarded.py) and of model inference,
single process, torch 2 threads, on the benchmark seeds 3000-3001 for every family."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os, sys, time, copy
import numpy as np
ROOT = NF_ROOT
sys.path.insert(0, ROOT); sys.path.insert(0, SCRIPTS)
import torch
torch.set_num_threads(2)
import research_benchmark as RB
import nf_exp as X
from config import MODEL
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from routing.guarded import GuardedRerouter

X.load_models(kinds=('gnn', 'lstm_only'), seeds=(0,), threads=2)
g = build_fat_tree()
out = {}
for name, sig in (('guarded_gnn', 'gnn'), ('guarded_lstm', 'lstm_only'), ('guarded_score', 'score')):
    ctrl, inf, per_family = [], [], {}
    for fam in X.PATTERNS:
        fc = []
        for seed in (3000, 3001):
            flows = TrafficGenerator(g, seed=seed).generate_scenario(total_ticks=150, pattern=fam)
            engine = NetworkEngine(g); flows = copy.deepcopy(flows)
            for f in flows: f.current_path = []; f.active = False
            engine.set_flows(flows); c = TelemetryCollector(g, engine)
            th = 0.851 if sig == 'score' else X._MODELS[(sig, 0)]['threshold']
            ctl = GuardedRerouter(g, c.edge_to_idx, threshold=th)
            for _ in range(150):
                c.collect_tick(engine.step())
                if engine.tick >= MODEL['window_size']:
                    if sig == 'score':
                        p = c.edge_scores_history[-1]
                    else:
                        s = time.perf_counter(); p = RB.predict(X._MODELS[(sig, 0)]['model'], c)
                        inf.append((time.perf_counter() - s) * 1000)
                    s = time.perf_counter(); ctl.evaluate_and_reroute(flows, p, engine, engine.tick)
                    d = (time.perf_counter() - s) * 1000
                    ctrl.append(d); fc.append(d)
        per_family[fam] = dict(p50=float(np.median(fc)), p99=float(np.percentile(fc, 99)), mean=float(np.mean(fc)))
    ctrl, inf = np.array(ctrl), np.array(inf)
    out[name] = dict(ctrl_p50=float(np.median(ctrl)), ctrl_p99=float(np.percentile(ctrl, 99)), ctrl_mean=float(ctrl.mean()),
                     ctrl_share_over_1ms=float((ctrl > 1).mean()), ctrl_share_over_5ms=float((ctrl > 5).mean()),
                     inf_p50=float(np.median(inf)) if inf.size else None, inf_p99=float(np.percentile(inf, 99)) if inf.size else None,
                     inf_mean=float(inf.mean()) if inf.size else None, per_family=per_family, ticks=int(ctrl.size))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items() if k != 'per_family'})
    print('   per family', {f: round(v['p50'], 2) for f, v in per_family.items()}, {f: round(v['p99'], 1) for f, v in per_family.items()})
json.dump(out, open(RESULTS + '/timing_original.json', 'w'), indent=1)
