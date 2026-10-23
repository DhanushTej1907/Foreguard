"""Concern 4: closed-loop transport with a reordering penalty (benchmark seeds 3000-3004, six families).
Results -> results/rev4_c4.jsonl."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os, sys, time, traceback
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results', 'rev4_c4.jsonl')
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
NONE = dict(headroom=None, max_changes=None, stretch=None, min_improvement=0.0, cooldown=0)
VARS = [('ecmp', 'ecmp', None, None, {}), ('reactive', 'reactive', None, None, {}),
        ('unguarded_gnn', 'unguarded', 'gnn', None, {}), ('guarded_gnn', 'guarded', 'gnn', None, {}),
        ('guarded_persistence', 'guarded', 'persistence', 0.5, {}), ('guarded_score', 'guarded', 'score', 0.851, {}),
        ('score_no_rules', 'guarded', 'score', 0.851, NONE)]
KEEP = ['total_drops', 'reroutes', 'completed', 'incomplete', 'fct_mean', 'fct_p99', 'retx_packets', 'offered_unique',
        'spurious_packets', 'cl_reordered_packets', 'makespan', 'per_flow']

def tasks():
    return [dict(id=f'{v}|{f}|{s}', variant=v, strategy=st, signal=sig, threshold=th, guard=g, family=f, seed=s)
            for v, st, sig, th, g in VARS for f in FAM for s in range(3000, 3005)]

_W = {}
def _init():
    sys.path.insert(0, HERE)
    import nf_exp
    from simulation.topology import build_fat_tree
    _W['X'], _W['g'] = nf_exp, build_fat_tree()

def _run(t):
    try:
        X, g = _W['X'], _W['g']
        from simulation.traffic_generator import TrafficGenerator
        flows = TrafficGenerator(g, seed=t['seed']).generate_scenario(total_ticks=150, pattern=t['family'])
        model, th, guard = None, t['threshold'], dict(t['guard'])
        if t['signal'] == 'gnn':
            if not X._MODELS:
                X.load_models(kinds=('gnn',), seeds=(0,), threads=1)
            model = X._MODELS[('gnn', 0)]['model']
            th = X._MODELS[('gnn', 0)]['threshold'] if t['strategy'] == 'guarded' else None
        if t['strategy'] == 'guarded':
            guard['signal'] = 'model' if t['signal'] == 'gnn' else t['signal']
        r = X.run_case_x(g, flows, 150, t['strategy'], threshold=th, model=model, guard=guard,
                         closed_loop=True, reorder_penalty=True)
        out = {k: t[k] for k in ('id', 'variant', 'family', 'seed')}
        out.update({k: r[k] for k in KEEP})
        return out
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())

if __name__ == '__main__':
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    done = {json.loads(l)['id'] for l in open(OUT)} if os.path.exists(OUT) else set()
    T = [t for t in tasks() if t['id'] not in done]
    print(len(T), 'tasks', flush=True)
    with mp.Pool(2, initializer=_init) as pool, open(OUT, 'a') as fh:
        for i, r in enumerate(pool.imap_unordered(_run, T), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if i % 30 == 0: print(i, flush=True)
    print('done')
