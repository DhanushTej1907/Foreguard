"""Summaries for the second review round -> results/rev2_summary.json"""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os
from collections import defaultdict
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
ROOT = os.environ.get('NF_ROOT', NF_ROOT)
load = lambda n: [json.loads(l) for l in open(os.path.join(R, n + '.jsonl'))] if os.path.exists(os.path.join(R, n + '.jsonl')) else []
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']


def boot(d, draws=2000, seed=42):
    v = np.array([d[k] for k in sorted(d)])
    rng = np.random.default_rng(seed)
    return v[rng.integers(0, len(v), size=(draws, len(v)))].mean(1)


def cells(rows, **cond):
    """(family, seed) -> row-average over model seeds"""
    acc = defaultdict(list)
    for r in rows:
        if all(r.get(k) == v for k, v in cond.items()):
            acc[(r['family'], r['seed'])].append(r)
    return {k: {m: float(np.mean([x[m] for x in v])) for m in ('total_drops', 'reroutes', 'delivered_p99_latency_us')}
            for k, v in acc.items()}


S = {}
rep = load('replicate')
ecmp_u = cells(rep, variant='ecmp')
E = np.mean([v['total_drops'] for v in ecmp_u.values()])
S['ecmp_unseen'] = dict(total_drops=float(E), p99_ms=float(np.mean([v['delivered_p99_latency_us'] for v in ecmp_u.values()]) / 1000))
abl = load('ablation')
full_score = cells(abl, variant='full', signal='score')


def summ(c, ref=None):
    out = dict(n=len(c), total_pct=float((np.mean([v['total_drops'] for v in c.values()]) / E - 1) * 100),
               reroutes=float(np.mean([v['reroutes'] for v in c.values()])),
               p99_ms=float(np.mean([v['delivered_p99_latency_us'] for v in c.values()]) / 1000))
    if ref is not None:
        d = {k: (c[k]['total_drops'] - ref[k]['total_drops']) / E * 100 for k in c}
        b = boot(d)
        out['vs_ref_pp'] = float(np.mean(list(d.values())))
        out['vs_ref_ci95'] = np.percentile(b, [2.5, 97.5]).round(3).tolist()
    return out


# add-one-in
ad = load('rev2_addone')
none = cells(ad, variant='none')
S['addone'] = {v: summ(cells(ad, variant=v), none) for v in ('none', '+capacity', '+budget', '+stretch', '+min_improvement', '+cooldown')}
S['addone']['full'] = summ(full_score, none)
# controls (default rules) vs guarded score
ct = load('rev2_controls')
S['controls'] = {v: summ(cells(ct, variant=v), full_score) for v in ('random', 'static', 'oracle_score', 'oracle_label')}
S['controls']['guarded_score'] = summ(full_score)
pers = cells(abl, variant='full', signal='persistence')
S['controls']['guarded_persistence'] = summ(pers)
S['controls']['oracle_label_vs_persistence'] = summ(cells(ct, variant='oracle_label'), pers)

# Table V row + reordering for score without rules (benchmark seeds)
ng = [r for r in load('rev2_noguard') if r['variant'] == 'score_no_rules']
ins = load('instrument')
ecmp_b = [r for r in ins if r['variant'] == 'ecmp']
m = lambda rows, k: float(np.mean([r[k] for r in rows]))
if ng:
    S['table5_score_no_rules'] = dict(
        runs=len(ng), total_vs_ecmp=(m(ng, 'total_drops') / m(ecmp_b, 'total_drops') - 1) * 100,
        fabric_vs_ecmp=(m(ng, 'fabric_link_drops') / m(ecmp_b, 'fabric_link_drops') - 1) * 100,
        host_per_run=m(ng, 'host_link_drops'), delivered_per_tick=m(ng, 'total_delivered') / 150,
        mean_delay_ms=m(ng, 'delivered_mean_latency_us') / 1000, p99_ms=m(ng, 'delivered_p99_latency_us') / 1000,
        reroutes=m(ng, 'reroutes'),
        reordered_pct=float(np.sum([r['reordered_packets'] for r in ng]) / np.sum([r['total_delivered'] for r in ng]) * 100),
        flows_with_reordering_pct=float(np.mean([r['flows_with_reordering'] / r['n_flows'] for r in ng]) * 100))
    # paired CI vs guarded score on benchmark seeds (percent of ECMP total drops)
    gs = {(r['family'], r['seed']): r['total_drops'] for r in ins if r['variant'] == 'guarded_score'}
    Eb = m(ecmp_b, 'total_drops')
    d = {(r['family'], r['seed']): (r['total_drops'] - gs[(r['family'], r['seed'])]) / Eb * 100 for r in ng}
    S['table5_score_no_rules']['vs_guarded_score_pp'] = float(np.mean(list(d.values())))
    S['table5_score_no_rules']['vs_guarded_score_ci95'] = np.percentile(boot(d), [2.5, 97.5]).round(3).tolist()

# Table IX: closed loop, completed flows only (as published) and with unfinished flows censored at the run end
import sys
sys.path.insert(0, ROOT)
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
g = build_fat_tree()
starts = {}
cl = [r for r in load('closedloop') if r['variant'] != 'guarded_lstm'] + \
     [dict(r, variant='score_no_rules') for r in load('rev2_noguard') if r['variant'] == 'score_no_rules_cl']
for r in cl:
    key = (r['family'], r['seed'])
    if key not in starts:
        fl = TrafficGenerator(g, seed=r['seed']).generate_scenario(total_ticks=150, pattern=r['family'])
        starts[key] = {f.flow_id: f.start_tick for f in fl}
agg = defaultdict(lambda: defaultdict(list))
for r in cl:
    a = agg[r['variant']]
    for fid, ftype, size, ideal, fct in r['per_flow']:
        if size <= 0:
            continue
        cens = fct if fct is not None else r['makespan'] - starts[(r['family'], r['seed'])][fid]
        a['cens'].append(cens); a['cens_slow'].append(cens / ideal if ideal else np.nan)
        if fct is not None:
            a['fct'].append(fct); a['slow'].append(fct / ideal if ideal else np.nan)
    a['incomplete'].append(r['incomplete']); a['retx'].append(r['retx_packets']); a['offered'].append(r['offered_unique'])
    a['reroutes'].append(r['reroutes'])
S['table9'] = {}
for v, a in agg.items():
    S['table9'][v] = dict(runs=len(a['reroutes']), incomplete=int(np.sum(a['incomplete'])),
                          fct_mean=float(np.mean(a['fct'])), fct_p99=float(np.percentile(a['fct'], 99)),
                          slow_mean=float(np.nanmean(a['slow'])), slow_p99=float(np.nanpercentile(a['slow'], 99)),
                          cens_fct_mean=float(np.mean(a['cens'])), cens_fct_p99=float(np.percentile(a['cens'], 99)),
                          cens_slow_mean=float(np.nanmean(a['cens_slow'])), cens_slow_p99=float(np.nanpercentile(a['cens_slow'], 99)),
                          retx_pct=float(np.sum(a['retx']) / np.sum(a['offered']) * 100), reroutes=float(np.mean(a['reroutes'])))
json.dump(S, open(os.path.join(R, 'rev2_summary.json'), 'w'), indent=1)
for k, v in S.items():
    print(k)
    if isinstance(v, dict) and all(isinstance(x, dict) for x in v.values()):
        for kk, vv in v.items():
            print('   ', kk, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in vv.items()})
    else:
        print('   ', {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
