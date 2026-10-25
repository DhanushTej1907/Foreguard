"""Summary of the capacity-check ablation (rev5_cap.jsonl) -> results/rev5_cap_summary.json
Differences are paired over the 60 (family, seed) cells, in points of the mean ECMP total drops (the paper's
convention), with the paired bootstrap (2,000 draws, seed 42) and the family-clustered bootstrap."""
import json, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
rows = [json.loads(l) for l in open(os.path.join(R, 'rev5_cap.jsonl'))]
assert not [r for r in rows if 'error' in r], 'failed runs'
assert all(r['packet_conservation'] and r['drop_split_ok'] for r in rows), 'invariant violated'
cell = {}
for r in rows:
    cell.setdefault((r['variant'], r['signal']), {})[(r['family'], r['seed'])] = r
assert all(len(v) == 60 for v in cell.values()), {k: len(v) for k, v in cell.items()}
ecmp = cell[('ecmp', None)]
E = np.mean([x['total_drops'] for x in ecmp.values()])


def boot(d, draws=2000, seed=42, cluster=False):
    rng = np.random.default_rng(seed)
    if not cluster:
        v = np.array([d[k] for k in sorted(d)])
        out = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(1)
    else:
        fams = sorted({f for f, _ in d})
        byf = {f: np.array([d[k] for k in sorted(d) if k[0] == f]) for f in fams}
        out = np.empty(draws)
        for i in range(draws):
            pick = rng.integers(0, len(fams), size=len(fams))
            out[i] = np.mean([byf[fams[j]][rng.integers(0, len(byf[fams[j]]), size=len(byf[fams[j]]))].mean() for j in pick])
    return np.percentile(out, [2.5, 97.5]).round(2).tolist()


def diff(a, b, key='total_drops', scale=None):
    """paired a - b; drops in points of mean ECMP drops, other metrics in their own units"""
    sc = (100 / E) if scale is None else scale
    d = {c: (a[c][key] - b[c][key]) * sc for c in a}
    return dict(mean=round(float(np.mean(list(d.values()))), 2), ci95=boot(d), ci95_cluster=boot(d, cluster=True))


S = dict(ecmp_mean_drops=float(E), n_cells=60, cells={}, effects={})
for (v, sig), a in sorted(cell.items(), key=lambda kv: str(kv[0])):
    m = lambda k: float(np.mean([x[k] for x in a.values()]))
    o = dict(total_pct=(m('total_drops') / E - 1) * 100,
             fabric_pct=(m('fabric_link_drops') / np.mean([x['fabric_link_drops'] for x in ecmp.values()]) - 1) * 100,
             host_pct=(m('host_link_drops') / np.mean([x['host_link_drops'] for x in ecmp.values()]) - 1) * 100,
             delivered_per_tick=m('delivered_per_tick'), mean_ms=m('delivered_mean_latency_us') / 1000,
             p99_ms=m('delivered_p99_latency_us') / 1000, reroutes=m('reroutes'),
             reordered_pct=float(np.sum([x['reordered_packets'] for x in a.values()]) /
                                 np.sum([x['total_delivered'] for x in a.values()]) * 100))
    if v != 'ecmp':
        o['vs_ecmp'] = diff(a, ecmp)
    S['cells'][f'{v}|{sig}'] = {k: (float(x) if isinstance(x, float) else x) for k, x in o.items()}   # unrounded: round once, when formatting

C = lambda v, s: cell[(v, s)]
for sig in ('score', 'random'):
    S['effects'][sig] = {
        'B-A (remove queue term)': diff(C('B', sig), C('A', sig)),
        'Cp-A (remove arrivals)': diff(C('Cp', sig), C('A', sig)),
        'D-A (remove both)': diff(C('D', sig), C('A', sig)),
        'D-B (remove arrivals, no queue)': diff(C('D', sig), C('B', sig)),
        'D-Cp (remove queue, no arrivals)': diff(C('D', sig), C('Cp', sig)),
        'E-D (rate-blind vs exact rates)': diff(C('E', sig), C('D', sig)),
        'E-A (rate-blind demand only)': diff(C('E', sig), C('A', sig)),
        'nocap-A (remove the check)': diff(C('nocap', sig), C('A', sig)),
        'nocap-D (remove demand-only check)': diff(C('nocap', sig), C('D', sig)),
        'reroutes D-A': diff(C('D', sig), C('A', sig), 'reroutes', 1),
        'p99_ms D-A': diff(C('D', sig), C('A', sig), 'delivered_p99_latency_us', 1e-3),
    }
S['effects']['random-score'] = {v: diff(C(v, 'random'), C(v, 'score')) for v in ('A', 'B', 'Cp', 'D', 'E', 'nocap')}
S['effects']['random-score reroutes'] = {v: diff(C(v, 'random'), C(v, 'score'), 'reroutes', 1) for v in ('A', 'B', 'Cp', 'D', 'E', 'nocap')}
# interaction: does the effect of removing the queue term / arrivals differ between the signals?
for name, (x, y) in {'queue': ('B', 'A'), 'arrivals': ('Cp', 'A'), 'both': ('D', 'A')}.items():
    a, b = C(x, 'random'), C(y, 'random')
    c, d = C(x, 'score'), C(y, 'score')
    dd = {k: ((a[k]['total_drops'] - b[k]['total_drops']) - (c[k]['total_drops'] - d[k]['total_drops'])) * 100 / E for k in a}
    S['effects'][f'interaction {name} (random minus score)'] = dict(mean=round(float(np.mean(list(dd.values()))), 2),
                                                                  ci95=boot(dd), ci95_cluster=boot(dd, cluster=True))
json.dump(S, open(os.path.join(R, 'rev5_cap_summary.json'), 'w'), indent=1)
print(f'ECMP mean drops {E:.0f}')
for k, o in S['cells'].items():
    print(f"{k:14s} total {o['total_pct']:6.2f}% fab {o['fabric_pct']:6.2f}% host {o['host_pct']:5.2f}% del/t {o['delivered_per_tick']:7.1f} "
          f"mean {o['mean_ms']:.2f} p99 {o['p99_ms']:.2f} rr {o['reroutes']:7.1f} reord {o['reordered_pct']:.2f}%  {o.get('vs_ecmp', '')}")
for g, e in S['effects'].items():
    if isinstance(e, dict) and 'mean' in e:
        print(g, e); continue
    print('--', g)
    for k, v in e.items():
        print(f'   {k:36s} {v}')
