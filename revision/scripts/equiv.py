"""Equivalence (TOST) numbers for guarded GNN vs alternatives, same bootstrap as the paper
(2000 paired draws over (family, seed), seed 42) plus a two-stage family-clustered bootstrap."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, sys
from collections import defaultdict
import numpy as np
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']

def seed_avg(rows, name_key, fam_key, seed_key):
    acc = defaultdict(list)
    for r in rows:
        acc[(r[fam_key], r[seed_key], r[name_key])].append(r['total_drops'])
    return {k: float(np.mean(v)) for k, v in acc.items()}

def analyse(avg, seeds, names):
    ecmp = np.mean([avg[(f, s, names['ecmp'])] for f in FAM for s in seeds])
    margin = 0.01 * ecmp
    out = dict(ecmp_mean=ecmp, margin=margin, rows={})
    for alt in ('guarded_persistence', 'guarded_score', 'guarded_lstm'):
        pairs = sorted((f, s) for f in FAM for s in seeds)
        d = np.array([avg[(f, s, names['guarded_gnn'])] - avg[(f, s, names[alt])] for f, s in pairs])
        rng = np.random.default_rng(42)
        m = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
        rng2 = np.random.default_rng(42); cm = np.empty(2000)
        byf = {f: np.array([avg[(f, s, names['guarded_gnn'])] - avg[(f, s, names[alt])] for s in seeds]) for f in FAM}
        for i in range(2000):
            fs = rng2.integers(0, 6, size=6)
            cm[i] = np.mean([byf[FAM[j]][rng2.integers(0, len(seeds), size=len(seeds))].mean() for j in fs])
        ci95 = np.percentile(m, [2.5, 97.5]); ci90 = np.percentile(m, [5, 95]); cc90 = np.percentile(cm, [5, 95])
        out['rows'][alt] = dict(mean=float(d.mean()), ci95=ci95.tolist(), mean_pct=float(d.mean() / ecmp * 100),
                                ci90_pct=(ci90 / ecmp * 100).tolist(), ci90_cluster_pct=(cc90 / ecmp * 100).tolist(),
                                equiv_pair=bool(ci90[0] > -margin and ci90[1] < margin),
                                equiv_cluster=bool(cc90[0] > -margin and cc90[1] < margin))
    return out

res = {}
b = json.load(open(NF_ROOT + '/research/benchmark_results_v4.json'))
avg = seed_avg(b['runs'], 'strategy', 'pattern', 'seed')
res['benchmark'] = analyse(avg, list(range(3000, 3010)), {k: k for k in ('ecmp', 'guarded_gnn', 'guarded_persistence', 'guarded_score', 'guarded_lstm')})
try:
    rows = [json.loads(l) for l in open(RESULTS + '/replicate.jsonl')]
    rows = [r for r in rows if 'error' not in r]
    avg2 = seed_avg(rows, 'variant', 'family', 'seed')
    if len({k for k in avg2}) >= 13 * 60 // 13 * 7 - 1 and all((f, s, 'guarded_gnn') in avg2 for f in FAM for s in range(4000, 4010)):
        res['unseen'] = analyse(avg2, list(range(4000, 4010)), {k: k for k in ('ecmp', 'guarded_gnn', 'guarded_persistence', 'guarded_score', 'guarded_lstm')})
except FileNotFoundError:
    pass
json.dump(res, open(RESULTS + '/equiv.json', 'w'), indent=1)
for k, v in res.items():
    print(k, 'ECMP mean', round(v['ecmp_mean']), 'margin', round(v['margin']))
    for alt, r in v['rows'].items():
        print('  ', alt, round(r['mean']), [round(x) for x in r['ci95']], f"{r['mean_pct']:+.2f}%", [round(x, 2) for x in r['ci90_pct']], [round(x, 2) for x in r['ci90_cluster_pct']], r['equiv_pair'], r['equiv_cluster'])
