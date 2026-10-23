"""Concern 4 summary: closed-loop FCT with and without the reordering penalty, with cell-bootstrap CIs."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os, sys
from collections import defaultdict
import numpy as np
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
sys.path.insert(0, os.environ.get('NF_ROOT', NF_ROOT))
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
g = build_fat_tree(); starts = {}
def start(f, s, fid):
    if (f, s) not in starts:
        starts[(f, s)] = {x.flow_id: x.start_tick for x in TrafficGenerator(g, seed=s).generate_scenario(total_ticks=150, pattern=f)}
    return starts[(f, s)][fid]
load = lambda n: [json.loads(l) for l in open(os.path.join(R, n))]
base = [r for r in load('closedloop.jsonl') if r['variant'] != 'guarded_lstm'] + \
       [dict(r, variant='score_no_rules') for r in load('rev2_noguard.jsonl') if r['variant'] == 'score_no_rules_cl']
pen = load('rev4_c4.jsonl')
def cells(rows):
    C = defaultdict(dict)
    for r in rows:
        fl = [(x[4] if x[4] is not None else r['makespan'] - start(r['family'], r['seed'], x[0])) for x in r['per_flow'] if x[2] > 0]
        comp = [x[4] for x in r['per_flow'] if x[2] > 0 and x[4] is not None]
        C[r['variant']][(r['family'], r['seed'])] = dict(cens=np.array(fl), comp=np.array(comp), retx=r['retx_packets'],
                                                        spur=r.get('spurious_packets', 0), off=r['offered_unique'], rr=r['reroutes'])
    return C
def stats(d, keys):
    cens = np.concatenate([d[k]['cens'] for k in keys]); comp = np.concatenate([d[k]['comp'] for k in keys])
    return dict(mean=cens.mean(), p99=np.percentile(cens, 99), mean_c=comp.mean(), p99_c=np.percentile(comp, 99))
def compare(C, a, b, draws=2000):
    keys = sorted(C[a]); rng = np.random.default_rng(42)
    sa, sb = stats(C[a], keys), stats(C[b], keys)
    dm, dp = [], []
    for _ in range(draws):
        ks = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        x, y = stats(C[a], ks), stats(C[b], ks)
        dm.append((x['mean'] / y['mean'] - 1) * 100); dp.append((x['p99'] / y['p99'] - 1) * 100)
    return dict(mean_pct=(sa['mean'] / sb['mean'] - 1) * 100, mean_ci=np.percentile(dm, [2.5, 97.5]).round(2).tolist(),
                p99_pct=(sa['p99'] / sb['p99'] - 1) * 100, p99_ci=np.percentile(dp, [2.5, 97.5]).round(2).tolist())
S = {}
for name, rows in (('no_penalty', base), ('penalty', pen)):
    C = cells(rows); S[name] = {}
    for v in C:
        keys = sorted(C[v]); st = stats(C[v], keys)
        o = dict(fct_mean_cens=st['mean'], fct_p99_cens=st['p99'], fct_mean=st['mean_c'], fct_p99=st['p99_c'],
                 retx_pct=sum(C[v][k]['retx'] for k in keys) / sum(C[v][k]['off'] for k in keys) * 100,
                 spur_pct=sum(C[v][k]['spur'] for k in keys) / sum(C[v][k]['off'] for k in keys) * 100,
                 reroutes=float(np.mean([C[v][k]['rr'] for k in keys])))
        if v != 'ecmp':
            o['vs_ecmp'] = compare(C, v, 'ecmp')
        if v != 'guarded_score':
            o['vs_guarded_score'] = compare(C, v, 'guarded_score')
        S[name][v] = {k: (float(x) if isinstance(x, (np.floating, float)) else x) for k, x in o.items()}
json.dump(S, open(os.path.join(R, 'rev4_c4_summary.json'), 'w'), indent=1)
for name in S:
    print('==', name)
    for v, o in S[name].items():
        print(f"  {v:20s} mean {o['fct_mean_cens']:.1f} p99 {o['fct_p99_cens']:.0f} (completed {o['fct_mean']:.1f}/{o['fct_p99']:.0f}) retx {o['retx_pct']:.1f}% spur {o['spur_pct']:.1f}% rr {o['reroutes']:.0f}",
              ('vsECMP mean %+.2f%% %s p99 %+.2f%% %s' % (o['vs_ecmp']['mean_pct'], o['vs_ecmp']['mean_ci'], o['vs_ecmp']['p99_pct'], o['vs_ecmp']['p99_ci'])) if 'vs_ecmp' in o else '',
              ('| vsGS mean %+.2f%% %s p99 %+.2f%% %s' % (o['vs_guarded_score']['mean_pct'], o['vs_guarded_score']['mean_ci'], o['vs_guarded_score']['p99_pct'], o['vs_guarded_score']['p99_ci'])) if 'vs_guarded_score' in o else '')
