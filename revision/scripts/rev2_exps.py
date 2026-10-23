"""Experiments for the second review round (results/rev2_<group>.jsonl, resumable).

  addone      guarded score on unseen seeds 4000-4009: no admission rules, then each rule added alone
  noguard     score signal with all admission rules removed on benchmark seeds 3000-3009 (instrumented:
              Table V metrics + reordering); closed-loop transport on seeds 3000-3004 (Table IX)
  controls    guarded controller (default rules) on unseen seeds driven by a random risk signal
              (i.i.d. uniform per link and tick), a static random risk (drawn once per run), and an
              oracle with perfect 5-tick look-ahead under the current routing (future score, future label)
"""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os, sys, time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results')
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
BENCH, UNSEEN = list(range(3000, 3010)), list(range(4000, 4010))
TH_SCORE, TH_PERS = 0.851, 0.5
NONE = dict(headroom=None, max_changes=None, stretch=None, min_improvement=0.0, cooldown=0)
ADD = {'none': {}, '+capacity': dict(headroom=0.9), '+budget': dict(max_changes=4), '+stretch': dict(stretch=2),
       '+min_improvement': dict(min_improvement=0.15), '+cooldown': dict(cooldown=5)}
KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'unroutable_drops', 'total_offered', 'total_delivered',
        'avg_throughput', 'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes', 'ticks_run',
        'reordered_packets', 'reordered_fraction', 'flows_with_reordering', 'n_flows', 'completed', 'incomplete',
        'fct_mean', 'fct_p50', 'fct_p99', 'slowdown_mean', 'slowdown_p50', 'slowdown_p99', 'makespan',
        'retx_packets', 'offered_unique', 'per_flow']


def tasks(group):
    T = []
    for fi, fam in enumerate(FAM):
        if group == 'addone':
            for s in UNSEEN:
                for name, g in ADD.items():
                    T.append(dict(id=f'addone|{name}|{fam}|{s}', variant=name, family=fam, seed=s, signal='score',
                                  threshold=TH_SCORE, guard=dict(NONE, **g)))
        elif group == 'noguard':
            for s in BENCH:
                T.append(dict(id=f'noguard|open|{fam}|{s}', variant='score_no_rules', family=fam, seed=s, signal='score',
                              threshold=TH_SCORE, guard=dict(NONE), instrument=True))
            for s in BENCH[:5]:
                T.append(dict(id=f'noguard|closed|{fam}|{s}', variant='score_no_rules_cl', family=fam, seed=s,
                              signal='score', threshold=TH_SCORE, guard=dict(NONE), closed_loop=True))
        elif group == 'controls':
            for s in UNSEEN:
                for sig, th in (('random', TH_SCORE), ('static', TH_SCORE), ('oracle_score', TH_SCORE),
                                ('oracle_label', TH_PERS)):
                    T.append(dict(id=f'controls|{sig}|{fam}|{s}', variant=sig, family=fam, seed=s, signal=sig,
                                  threshold=th, guard={}, rng_seed=1000 * fi + s))
    return T


_W = {}


def _init():
    sys.path.insert(0, HERE)
    import nf_exp
    from simulation.topology import build_fat_tree
    _W['X'], _W['g'] = nf_exp, build_fat_tree()


def _run(t):
    X, g = _W['X'], _W['g']
    from simulation.traffic_generator import TrafficGenerator
    flows = TrafficGenerator(g, seed=t['seed']).generate_scenario(total_ticks=150, pattern=t['family'])
    s = time.time()
    r = X.run_case_x(g, flows, 150, 'guarded', threshold=t['threshold'], guard=dict(t['guard'], signal=t['signal']),
                     instrument=t.get('instrument', False), closed_loop=t.get('closed_loop', False),
                     rng_seed=t.get('rng_seed', 0))
    out = {k: r[k] for k in KEEP if k in r}
    out.update({k: t[k] for k in ('id', 'variant', 'family', 'seed', 'signal')}, sec=time.time() - s)
    return out


def main():
    group, workers = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
    path = os.path.join(OUT, f'rev2_{group}.jsonl')
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)['id'] for l in open(path)}
    T = [t for t in tasks(group) if t['id'] not in done]
    print(group, len(T), 'tasks', flush=True)
    t0 = time.time()
    with mp.Pool(workers, initializer=_init) as pool, open(path, 'a') as fh:
        for i, r in enumerate(pool.imap_unordered(_run, T), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if i % 20 == 0 or i == len(T):
                el = (time.time() - t0) / 60
                print(f'  {i}/{len(T)} {el:.1f} min, eta {el / i * (len(T) - i):.1f} min', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    main()
