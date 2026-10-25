"""Concern 1 (fourth review): which terms of the capacity check matter?  Unseen seeds 4000-4009, six families,
score and random risk signals, everything else at the defaults.  Results -> results/rev5_cap.jsonl (resumable).

Capacity check of the guarded controller (paper Eq. 21), per directed link e on the candidate path:
    D(e) + r_f + Q(e)/dt <= alpha * C(e),  with  D(e) = max(R(e), A(e))
  R(e) = sum of the sending rates of the active flows whose current path uses e   [packets/tick]
  A(e) = packets that arrived at e in the last tick (observed arrivals)          [packets/tick]
  Q(e) = queue of e in packets, dt = 1 tick;  C(e) in packets/tick;  alpha = 0.90
Variants (only these terms change):
  A   D = max(R, A), queue term kept         (the paper's check)
  B   D = max(R, A), queue term removed
  Cp  D = R,         queue term kept         (observed arrivals removed)
  D   D = R,         queue term removed      (demand-only check)
  E   as D, but every flow's rate is replaced by the mean active rate (existing rate-blind estimator);
      supplementary: the mean is still computed from the flows' sending rates
  nocap  no capacity check at all (alpha = None), the leave-one-rule-out reference
plus ECMP.  The random signal and its seeds (1000 * family index + seed) are those of rev4_c1."""
from paths import NF_ROOT  # folder locations (see paths.py)
import json, os, sys, time, traceback
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results', 'rev5_cap.jsonl')
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
SEEDS = list(range(4000, 4010))
# switch names follow the local nf_exp.py (cap_arrivals / cap_queue, defaults True = paper's check)
VARIANTS = {'A': {}, 'B': dict(cap_queue=False), 'Cp': dict(cap_arrivals=False),
            'D': dict(cap_arrivals=False, cap_queue=False),
            'E': dict(cap_arrivals=False, cap_queue=False, rate_mode='blind'),
            'nocap': dict(headroom=None)}
KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'total_offered', 'total_delivered',
        'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes', 'reordered_packets',
        'reordered_fraction', 'flows_with_reordering', 'n_flows', 'packet_conservation', 'drop_split_ok']


def tasks():
    T = []
    for fi, fam in enumerate(FAM):
        for s in SEEDS:
            base = dict(family=fam, seed=s, noise_seed=1000 * fi + s)
            T.append(dict(base, variant='ecmp', signal=None))
            for v, g in VARIANTS.items():
                for sig in ('score', 'random'):
                    T.append(dict(base, variant=v, signal=sig, threshold=0.851, guard=dict(g)))
    for t in T:
        t['id'] = '|'.join(str(t.get(k)) for k in ('variant', 'signal', 'family', 'seed'))
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
        if t['variant'] == 'ecmp':
            r = X.run_case_x(g, flows, 150, 'ecmp', instrument=True)
        else:
            guard = dict(t['guard'], noise_seed=t['noise_seed'], signal=t['signal'])
            r = X.run_case_x(g, flows, 150, 'guarded', threshold=t['threshold'], guard=guard,
                             rng_seed=t['noise_seed'], instrument=True)
        out = {k: t.get(k) for k in ('id', 'variant', 'signal', 'family', 'seed')}
        out.update({k: r[k] for k in KEEP})
        out['delivered_per_tick'] = r['total_delivered'] / 150
        return out
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())


if __name__ == '__main__':
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    done = {json.loads(l)['id'] for l in open(OUT)} if os.path.exists(OUT) else set()
    T = [t for t in tasks() if t['id'] not in done]
    if len(sys.argv) > 2 and sys.argv[2] == 'smoke':
        T = [t for t in T if t['family'] == 'hotspot' and t['seed'] == 4000]
    print(len(T), 'tasks', flush=True)
    t0 = time.time()
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2, initializer=_init) as pool, open(OUT, 'a') as fh:
        for i, r in enumerate(pool.imap_unordered(_run, T), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if i % 100 == 0 or i == len(T):
                el = (time.time() - t0) / 60
                print(f'  {i}/{len(T)} {el:.1f} min, eta {el / i * (len(T) - i):.1f} min', flush=True)
