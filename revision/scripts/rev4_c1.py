"""Concern 1 (third review): does the capacity check need exact flow rates, and what does a signal-free
guard do? Unseen seeds 4000-4009, six families. Results -> results/rev4_c1.jsonl (resumable).
  rate modes: exact (default), log-normal noise sigma 0.25 / 0.5 / 1.0 (mean-preserving, fresh each tick),
              blind (every active flow assumed to send the mean active rate)
  signals:    score, GNN (three training seeds), random; plus a signal-free guard (constant risk: every flow
              eligible, plain shortest-path candidates, capacity check alone decides)"""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import copy, json, os, sys, time, traceback
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results', 'rev4_c1.jsonl')
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
SEEDS = list(range(4000, 4010))
MODES = {'exact': {}, 'noise0.25': dict(rate_noise=0.25), 'noise0.5': dict(rate_noise=0.5),
         'noise1.0': dict(rate_noise=1.0), 'blind': dict(rate_mode='blind'),
         'telemetry': dict(rate_mode='telemetry')}
KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'total_offered', 'total_delivered',
        'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes', 'reordered_packets',
        'reordered_fraction', 'flows_with_reordering', 'n_flows']


def tasks():
    T = []
    for fi, fam in enumerate(FAM):
        for s in SEEDS:
            base = dict(family=fam, seed=s, noise_seed=1000 * fi + s)
            for mode, g in MODES.items():
                if mode != 'exact':          # exact score / GNN / random already exist; rerun only for the reference cells
                    pass
                T.append(dict(base, mode=mode, variant='score', signal='score', threshold=0.851, guard=dict(g)))
                for m in (0, 1, 2):
                    T.append(dict(base, mode=mode, variant='gnn', signal='gnn', mseed=m, guard=dict(g)))
                T.append(dict(base, mode=mode, variant='random', signal='random', threshold=0.851, guard=dict(g)))
            for mode in ('exact', 'noise0.5', 'blind', 'telemetry'):
                T.append(dict(base, mode=mode, variant='signal_free', signal='constant', threshold=0.5,
                              guard=dict(MODES[mode], min_improvement=0.0)))
    for t in T:
        t['id'] = '|'.join(str(t.get(k)) for k in ('mode', 'variant', 'mseed', 'family', 'seed'))
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
        flows = TrafficGenerator(g, seed=t['seed']).generate_scenario(total_ticks=150, pattern=t['family'])
        guard = dict(t['guard'], noise_seed=t['noise_seed'])
        model, th = None, t.get('threshold')
        if t['signal'] == 'gnn':
            if not X._MODELS:
                X.load_models(kinds=('gnn',), threads=1)
            mm = X._MODELS[('gnn', t['mseed'])]
            model, th = mm['model'], mm['threshold']
            guard['signal'] = 'model'
        else:
            guard['signal'] = t['signal']
        r = X.run_case_x(g, flows, 150, 'guarded', threshold=th, model=model, guard=guard,
                         rng_seed=t['noise_seed'], instrument=True)
        out = {k: t[k] for k in ('id', 'mode', 'variant', 'family', 'seed')}
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
            if i % 100 == 0 or i == len(T):
                el = (time.time() - t0) / 60
                print(f'  {i}/{len(T)} {el:.1f} min, eta {el / i * (len(T) - i):.1f} min', flush=True)
