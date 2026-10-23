"""Offered-load sweep: every flow rate of the unseen traffic (seeds 4000-4009, six families) is multiplied by
a load factor (1.0 = the traffic used everywhere else in the paper). Results -> results/rev3_load.jsonl (resumable).
Strategies: ECMP, guarded score, guarded GNN (three training seeds), score without admission rules, and the
guarded controller driven by a random risk signal."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import copy, json, os, sys, time, traceback
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results', 'rev3_load.jsonl')
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
SEEDS = list(range(4000, 4010))
LOADS = [0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
NONE = dict(headroom=None, max_changes=None, stretch=None, min_improvement=0.0, cooldown=0)
KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'total_offered', 'total_delivered',
        'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes']


def tasks():
    T = []
    for lf in LOADS:
        for fi, fam in enumerate(FAM):
            for s in SEEDS:
                base = dict(load=lf, family=fam, seed=s)
                T.append(dict(base, variant='ecmp', strategy='ecmp'))
                T.append(dict(base, variant='guarded_score', strategy='guarded', signal='score', threshold=0.851, guard={}))
                for m in (0, 1, 2):
                    T.append(dict(base, variant='guarded_gnn', strategy='guarded', signal='gnn', mseed=m, guard={}))
                T.append(dict(base, variant='score_no_rules', strategy='guarded', signal='score', threshold=0.851, guard=dict(NONE)))
                T.append(dict(base, variant='random', strategy='guarded', signal='random', threshold=0.851, guard={},
                              rng_seed=1000 * fi + s))
    for t in T:
        t['id'] = '|'.join(str(t.get(k)) for k in ('load', 'variant', 'mseed', 'family', 'seed'))
    return T


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
        flows = copy.deepcopy(TrafficGenerator(g, seed=t['seed']).generate_scenario(total_ticks=150, pattern=t['family']))
        for f in flows:
            f.rate *= t['load']
        model, th, guard = None, t.get('threshold'), dict(t.get('guard', {}))
        if t.get('signal') == 'gnn':
            if not X._MODELS:
                X.load_models(kinds=('gnn',), threads=1)
            mm = X._MODELS[('gnn', t['mseed'])]
            model, th = mm['model'], mm['threshold']
            guard['signal'] = 'model'
        elif t['strategy'] == 'guarded':
            guard['signal'] = t['signal']
        r = X.run_case_x(g, flows, 150, t['strategy'], threshold=th, model=model, guard=guard, rng_seed=t.get('rng_seed', 0))
        out = {k: t[k] for k in ('id', 'load', 'variant', 'family', 'seed')}
        out['mseed'] = t.get('mseed')
        out.update({k: r[k] for k in KEEP})
        return out
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())


if __name__ == '__main__':
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    done = {json.loads(l)['id'] for l in open(OUT)} if os.path.exists(OUT) else set()
    T = [t for t in tasks() if t['id'] not in done]
    print(len(T), 'tasks', flush=True)
    t0 = time.time()
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2, initializer=_init) as pool, open(OUT, 'a') as fh:
        for i, r in enumerate(pool.imap_unordered(_run, T), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if i % 50 == 0 or i == len(T):
                el = (time.time() - t0) / 60
                print(f'  {i}/{len(T)} {el:.1f} min, eta {el / i * (len(T) - i):.1f} min', flush=True)
