from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, sys, time
sys.path.insert(0, SCRIPTS)
import nf_exp as X
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
bench = json.load(open(X.ROOT + '/research/benchmark_results_v4.json'))
th = {n: bench['forecast']['metrics'][n]['per_seed'][0]['threshold'] for n in ('persistence_binary', 'persistence_continuous')}
stored = {(r['pattern'], r['seed'], r['strategy']): r for r in bench['runs'] if r['train_seed'] is None}
g = build_fat_tree()
seeds = [int(a) for a in sys.argv[1:]] or [3000, 3001]
mism = 0; t0 = time.time(); n = 0
for pat in X.PATTERNS:
    for s in seeds:
        flows = TrafficGenerator(g, seed=s).generate_scenario(total_ticks=150, pattern=pat)
        for name, kw in [('ecmp', dict(strategy='ecmp')), ('reactive', dict(strategy='reactive')),
                         ('guarded_score', dict(strategy='guarded', threshold=th['persistence_continuous'], guard=dict(signal='score'))),
                         ('guarded_persistence', dict(strategy='guarded', threshold=th['persistence_binary'], guard=dict(signal='persistence')))]:
            r = X.run_case_x(g, flows, 150, **kw); n += 1
            ref = stored[(pat, s, name)]
            ok = all(r[k] == ref[k] for k in ('total_drops', 'fabric_link_drops', 'host_link_drops', 'reroutes', 'total_delivered'))
            if not ok:
                mism += 1; print('MISMATCH', pat, s, name, {k: (r[k], ref[k]) for k in ('total_drops', 'reroutes')})
print('runs', n, 'mismatches', mism, 'sec/run', round((time.time() - t0) / n, 2))
