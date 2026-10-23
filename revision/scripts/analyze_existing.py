"""Re-analyse research/benchmark_results_v4.json: per-family tables, TOST equivalence,
family-clustered bootstrap. Reads JSON only."""
import json, sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
runs = d['runs']
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
STRATS = ['ecmp', 'reactive', 'gnn', 'guarded_gnn', 'guarded_lstm', 'guarded_persistence', 'guarded_score']
KEYS = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'avg_throughput', 'total_offered',
        'total_delivered', 'reroutes', 'delivered_mean_latency_us', 'delivered_p99_latency_us']

# seed-average model strategies over training seeds
acc = {}
for r in runs:
    acc.setdefault((r['pattern'], r['seed'], r['strategy']), []).append(r)
avg = {k: {key: float(np.mean([x[key] for x in v])) for key in KEYS} for k, v in acc.items()}
seeds = sorted({s for _, s, _ in avg})

res = {'per_family': {}, 'overall': {}, 'tost': {}, 'cluster_bootstrap': {}}
# per family absolute means
for fam in FAM + ['ALL']:
    fams = FAM if fam == 'ALL' else [fam]
    row = {}
    for st in STRATS:
        vals = {key: float(np.mean([avg[(f, s, st)][key] for f in fams for s in seeds])) for key in KEYS}
        row[st] = vals
    e = row['ecmp']
    for st in STRATS:
        v = row[st]
        v['total_change_pct'] = (v['total_drops'] / e['total_drops'] - 1) * 100 if e['total_drops'] else None
        v['fabric_change_pct'] = (v['fabric_link_drops'] / e['fabric_link_drops'] - 1) * 100 if e['fabric_link_drops'] else None
        v['host_change_pct'] = (v['host_link_drops'] / e['host_link_drops'] - 1) * 100 if e['host_link_drops'] else None
        v['delivered_change_pct'] = (v['avg_throughput'] / e['avg_throughput'] - 1) * 100
        v['loss_rate_pct'] = v['total_drops'] / v['total_offered'] * 100
    res['per_family'][fam] = row

def diffs_for(a, b, key='total_drops'):
    return {(f, s): avg[(f, s, a)][key] - avg[(f, s, b)][key] for f in FAM for s in seeds}

def pair_boot(dd, draws=2000, seed=42):
    v = np.array(list(dd.values()))
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(1)
    return m

def cluster_boot(dd, draws=2000, seed=42):
    """Two-stage bootstrap: resample families, then traffic seeds within each family."""
    rng = np.random.default_rng(seed)
    by_f = {f: np.array([dd[(f, s)] for s in seeds]) for f in FAM}
    out = np.empty(draws)
    for i in range(draws):
        fs = rng.integers(0, len(FAM), size=len(FAM))
        vals = []
        for fi in fs:
            x = by_f[FAM[fi]]
            vals.append(x[rng.integers(0, len(x), size=len(x))].mean())
        out[i] = np.mean(vals)
    return out

ecmp_mean = res['per_family']['ALL']['ecmp']['total_drops']
margin = 0.01 * ecmp_mean
res['tost_margin_packets'] = margin
for b in ['guarded_score', 'guarded_lstm', 'guarded_persistence']:
    dd = diffs_for('guarded_gnn', b)
    mean = float(np.mean(list(dd.values())))
    pb, cb = pair_boot(dd), cluster_boot(dd)
    entry = dict(mean=mean, mean_pct_of_ecmp=mean / ecmp_mean * 100)
    for name, bs in (('pair', pb), ('cluster', cb)):
        ci95 = np.percentile(bs, [2.5, 97.5]); ci90 = np.percentile(bs, [5, 95])
        entry[name] = dict(ci95=ci95.tolist(), ci90=ci90.tolist(),
                           ci95_pct=(ci95 / ecmp_mean * 100).tolist(), ci90_pct=(ci90 / ecmp_mean * 100).tolist(),
                           equivalent_at_1pct=bool(ci90[0] > -margin and ci90[1] < margin),
                           smallest_equiv_margin_pct=float(max(abs(ci90[0]), abs(ci90[1])) / ecmp_mean * 100))
    # per family difference
    entry['per_family'] = {}
    for f in FAM:
        x = np.array([dd[(f, s)] for s in seeds])
        rng = np.random.default_rng(7)
        bs = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(1)
        fe = res['per_family'][f]['ecmp']['total_drops']
        entry['per_family'][f] = dict(mean=float(x.mean()), ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                                      mean_pct_of_family_ecmp=float(x.mean() / fe * 100) if fe else None)
    res['tost'][b] = entry
# each strategy vs ECMP, cluster bootstrap on total drops and delivered
for st in STRATS[1:]:
    e = {}
    for key in ('total_drops', 'avg_throughput', 'fabric_link_drops', 'host_link_drops'):
        dd = diffs_for(st, 'ecmp', key)
        base = res['per_family']['ALL']['ecmp'][key]
        pb, cb = pair_boot(dd), cluster_boot(dd)
        e[key] = dict(mean=float(np.mean(list(dd.values()))),
                      pair_ci95_pct=(np.percentile(pb, [2.5, 97.5]) / base * 100).tolist(),
                      cluster_ci95_pct=(np.percentile(cb, [2.5, 97.5]) / base * 100).tolist())
    res['cluster_bootstrap'][st] = e
json.dump(res, open(out, 'w'), indent=1)

# print compact tables
print('ECMP absolute per family (per run): total / fabric / host / delivered pkt/tick / loss rate')
for f in FAM + ['ALL']:
    e = res['per_family'][f]['ecmp']
    print(f'{f:14s} {e["total_drops"]:10.0f} {e["fabric_link_drops"]:9.0f} {e["host_link_drops"]:9.0f} {e["avg_throughput"]:8.1f} {e["loss_rate_pct"]:6.2f}%')
print('\nchanges vs ECMP: total% fabric% host% delivered%')
for st in STRATS[1:]:
    print(st)
    for f in FAM + ['ALL']:
        v = res['per_family'][f][st]
        fmt = lambda x: '   n/a' if x is None else f'{x:+6.2f}'
        print(f'   {f:14s} tot {fmt(v["total_change_pct"])} fab {fmt(v["fabric_change_pct"])} host {fmt(v["host_change_pct"])} deliv {fmt(v["delivered_change_pct"])} rr {v["reroutes"]:7.1f}')
print('\nTOST margin (1% ECMP total drops):', round(margin))
for b, e in res['tost'].items():
    print(b, round(e['mean']), f"{e['mean_pct_of_ecmp']:+.2f}%")
    for name in ('pair', 'cluster'):
        x = e[name]
        print('  ', name, 'ci95', [round(v) for v in x['ci95']], 'ci90%', [round(v, 2) for v in x['ci90_pct']],
              'equiv@1%', x['equivalent_at_1pct'], 'smallest margin %', round(x['smallest_equiv_margin_pct'], 2))
    for f, v in e['per_family'].items():
        print('     ', f, round(v['mean']), [round(c) for c in v['ci95']], None if v['mean_pct_of_family_ecmp'] is None else round(v['mean_pct_of_family_ecmp'], 2))
print('\nvs ECMP, pair vs cluster CI (% of ECMP mean)')
for st, e in res['cluster_bootstrap'].items():
    print(st, {k: (round(v['mean']), [round(x, 2) for x in v['pair_ci95_pct']], [round(x, 2) for x in v['cluster_ci95_pct']]) for k, v in e.items()})
