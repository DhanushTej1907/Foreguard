"""Summary of the offered-load sweep -> results/rev3_load_summary.json"""
import json, os
from collections import defaultdict
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
rows = [json.loads(l) for l in open(os.path.join(R, 'rev3_load.jsonl'))]
assert not [r for r in rows if 'error' in r], 'failed runs'
cells = defaultdict(lambda: defaultdict(list))        # (load, variant) -> (family, seed) -> [rows]
for r in rows:
    cells[(r['load'], r['variant'])][(r['family'], r['seed'])].append(r)
avg = {k: {c: {m: float(np.mean([x[m] for x in v])) for m in ('total_drops', 'total_offered', 'reroutes', 'delivered_p99_latency_us')}
           for c, v in d.items()} for k, d in cells.items()}


def boot(d, draws=2000, seed=42):
    v = np.array([d[k] for k in sorted(d)])
    rng = np.random.default_rng(seed)
    return np.percentile(v[rng.integers(0, len(v), size=(draws, len(v)))].mean(1), [2.5, 97.5]).round(3).tolist()


S = {}
for lf in sorted({r['load'] for r in rows}):
    e = avg[(lf, 'ecmp')]
    E = np.mean([v['total_drops'] for v in e.values()])
    out = dict(ecmp_loss_pct=float(np.sum([v['total_drops'] for v in e.values()]) / np.sum([v['total_offered'] for v in e.values()]) * 100),
               ecmp_drops=float(E), ecmp_p99_ms=float(np.mean([v['delivered_p99_latency_us'] for v in e.values()]) / 1000))
    for v in ('guarded_score', 'guarded_gnn', 'score_no_rules', 'random'):
        a = avg[(lf, v)]
        D = np.mean([x['total_drops'] for x in a.values()])
        out[v] = dict(total_pct=float((D / E - 1) * 100) if E > 0 else None, drops=float(D),
                      loss_pct=float(np.sum([x['total_drops'] for x in a.values()]) / np.sum([x['total_offered'] for x in a.values()]) * 100),
                      reroutes=float(np.mean([x['reroutes'] for x in a.values()])),
                      p99_ms=float(np.mean([x['delivered_p99_latency_us'] for x in a.values()]) / 1000))
    sc = avg[(lf, 'guarded_score')]
    for v in ('guarded_gnn', 'score_no_rules', 'random'):
        dabs = {c: (avg[(lf, v)][c]['total_drops'] - sc[c]['total_drops']) for c in sc}
        out[v]['minus_score_pkts'] = float(np.mean(list(dabs.values())))
        out[v]['minus_score_pkts_ci95'] = boot(dabs)
        if E > 0:
            d = {c: x / E * 100 for c, x in dabs.items()}
            out[v]['minus_score_pp'] = float(np.mean(list(d.values())))
            out[v]['minus_score_ci95'] = boot(d)
    S[str(lf)] = out
json.dump(S, open(os.path.join(R, 'rev3_load_summary.json'), 'w'), indent=1)
for lf, o in S.items():
    print(f"load {lf}: ECMP loss {o['ecmp_loss_pct']:.1f}%  drops {o['ecmp_drops']:.0f}  p99 {o['ecmp_p99_ms']:.2f}")
    for v in ('guarded_score', 'guarded_gnn', 'score_no_rules', 'random'):
        x = o[v]
        tp = f"{x['total_pct']:7.2f}%" if x['total_pct'] is not None else '    n/a '
        print(f"   {v:15s} {tp} drops {x['drops']:9.1f} loss {x['loss_pct']:.2f}%  rr {x['reroutes']:8.1f}  p99 {x['p99_ms']:.2f}  "
              + (f"minus score {x.get('minus_score_pp', float('nan')):+.2f} {x.get('minus_score_ci95')} pkts {x['minus_score_pkts']:+.0f} {x['minus_score_pkts_ci95']}" if 'minus_score_pkts' in x else ''))
