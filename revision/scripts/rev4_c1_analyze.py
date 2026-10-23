"""Summary for concern 1 (rate knowledge, signal-free guard) -> results/rev4_c1_summary.json"""
import json, os
from collections import defaultdict
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
rows = [json.loads(l) for l in open(os.path.join(R, 'rev4_c1.jsonl'))]
assert not [r for r in rows if 'error' in r]
rep = [json.loads(l) for l in open(os.path.join(R, 'replicate.jsonl'))]
ecmp = {(r['family'], r['seed']): r for r in rep if r['variant'] == 'ecmp'}
E = np.mean([r['total_drops'] for r in ecmp.values()])
Ep99 = np.mean([r['delivered_p99_latency_us'] for r in ecmp.values()]) / 1000


def boot(d, draws=2000, seed=42):
    v = np.array([d[k] for k in sorted(d)])
    rng = np.random.default_rng(seed)
    return np.percentile(v[rng.integers(0, len(v), size=(draws, len(v)))].mean(1), [2.5, 97.5]).round(3).tolist()


cells = defaultdict(lambda: defaultdict(list))
for r in rows:
    cells[(r['mode'], r['variant'])][(r['family'], r['seed'])].append(r)
avg = {k: {c: {m: float(np.mean([x[m] for x in v])) for m in ('total_drops', 'reroutes', 'delivered_p99_latency_us',
                                                                   'reordered_packets', 'total_delivered', 'flows_with_reordering', 'n_flows')}
           for c, v in d.items()} for k, d in cells.items()}
S = dict(ecmp_drops=float(E), ecmp_p99_ms=float(Ep99), cells={})
for (mode, var), a in sorted(avg.items()):
    o = dict(total_pct=float((np.mean([x['total_drops'] for x in a.values()]) / E - 1) * 100),
             reroutes=float(np.mean([x['reroutes'] for x in a.values()])),
             p99_ms=float(np.mean([x['delivered_p99_latency_us'] for x in a.values()]) / 1000),
             reordered_pct=float(np.sum([x['reordered_packets'] for x in a.values()]) / np.sum([x['total_delivered'] for x in a.values()]) * 100),
             flows_reordered_pct=float(np.mean([x['flows_with_reordering'] / x['n_flows'] for x in a.values()]) * 100))
    ref = avg.get(('exact', var))
    if mode != 'exact' and ref:
        d = {c: (a[c]['total_drops'] - ref[c]['total_drops']) / E * 100 for c in a}
        o['vs_exact_pp'] = float(np.mean(list(d.values()))); o['vs_exact_ci95'] = boot(d)
    sc = avg.get((mode, 'score'))
    if var != 'score' and sc:
        d = {c: (a[c]['total_drops'] - sc[c]['total_drops']) / E * 100 for c in a}
        o['vs_score_pp'] = float(np.mean(list(d.values()))); o['vs_score_ci95'] = boot(d)
    S['cells'][f'{mode}|{var}'] = o
json.dump(S, open(os.path.join(R, 'rev4_c1_summary.json'), 'w'), indent=1)
print('ECMP', round(E), 'p99', round(Ep99, 2))
for k, o in S['cells'].items():
    print(f"{k:22s} {o['total_pct']:7.2f}%  rr {o['reroutes']:7.1f}  p99 {o['p99_ms']:.2f}  reord {o['reordered_pct']:.2f}% flows {o['flows_reordered_pct']:.1f}%",
          (f"vs exact {o['vs_exact_pp']:+.2f} {o['vs_exact_ci95']}" if 'vs_exact_pp' in o else ''),
          (f"vs score {o['vs_score_pp']:+.2f} {o['vs_score_ci95']}" if 'vs_score_pp' in o else ''))
