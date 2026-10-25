"""Concern 2 (fourth review): paired comparisons of the FlowBender-style baseline with score-without-rules and with
the guarded schemes on the benchmark seeds 3000-3009 (60 runs each; existing results, no new runs).
Inputs: rev4_fb.jsonl (fb_default, open loop), rev2_noguard.jsonl (score_no_rules), instrument.jsonl (ECMP and guarded;
guarded GNN / LSTM there use one training seed). Differences: FlowBender minus other; drops in points of the mean ECMP
total drops; paired bootstrap, 2,000 draws, seed 42. -> results/rev5_fb_compare.json"""
import json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
L = lambda f: [json.loads(l) for l in open(os.path.join(R, f))]
fb = {(r['family'], r['seed']): r for r in L('rev4_fb.jsonl') if r['id'].startswith('open|fb_default|')}
nr = {(r['family'], r['seed']): r for r in L('rev2_noguard.jsonl') if r['variant'] == 'score_no_rules'}
ins = L('instrument.jsonl')
g = {v: {(r['family'], r['seed']): r for r in ins if r['variant'] == v}
     for v in ('ecmp', 'guarded_score', 'guarded_persistence', 'guarded_gnn', 'guarded_lstm')}
assert len(fb) == len(nr) == 60 and all(len(v) == 60 for v in g.values())
E = np.mean([r['total_drops'] for r in g['ecmp'].values()])


def boot(d):
    v = np.array([d[k] for k in sorted(d)]); rng = np.random.default_rng(42)
    return np.percentile(v[rng.integers(0, len(v), (2000, len(v)))].mean(1), [2.5, 97.5]).tolist()


def cmp(a, b, f):
    d = {k: f(a[k]) - f(b[k]) for k in a}
    return dict(mean=float(np.mean(list(d.values()))), ci95=boot(d),
                fb_lower=int(sum(x < 0 for x in d.values())), ties=int(sum(x == 0 for x in d.values())))


M = {'total_drops_pp': lambda r: r['total_drops'] * 100 / E, 'delivered_per_tick': lambda r: r['total_delivered'] / 150,
     'mean_ms': lambda r: r['delivered_mean_latency_us'] / 1000, 'p99_ms': lambda r: r['delivered_p99_latency_us'] / 1000,
     'reroutes': lambda r: r['reroutes']}
S = dict(ecmp_mean_drops=float(E), fb_vs={})
for name, other in [('score_no_rules', nr)] + [(v, g[v]) for v in ('guarded_score', 'guarded_persistence', 'guarded_gnn', 'guarded_lstm')]:
    S['fb_vs'][name] = {m: cmp(fb, other, f) for m, f in M.items()}
json.dump(S, open(os.path.join(R, 'rev5_fb_compare.json'), 'w'), indent=1)
for n, o in S['fb_vs'].items():
    print(n, {m: (round(x['mean'], 2), [round(c, 2) for c in x['ci95']]) for m, x in o.items()})
