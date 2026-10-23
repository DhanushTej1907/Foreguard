"""Summaries of every new experiment group -> results/summary.json (works on partial data)."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os, sys
from collections import defaultdict
import numpy as np

R = RESULTS
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
UNSEEN = list(range(4000, 4010)); BENCH = list(range(3000, 3010))


def load(group):
    p = os.path.join(R, f'{group}.jsonl')
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        r = json.loads(line)
        if 'error' not in r:
            out.append(r)
    return out


def ecmp_unseen():
    """ECMP on seeds 4000-4009 from the original guard sweep (deterministic) or the replicate group."""
    e = {}
    gs = json.load(open(NF_ROOT + '/research/guard_sweep_v4.json'))
    for r in gs['runs']:
        if r['strategy'] == 'ecmp':
            e[(r['pattern'], r['seed'])] = r
    for r in load('replicate'):
        if r['variant'] == 'ecmp':
            e[(r['family'], r['seed'])] = r
    return e


def avg_over_mseeds(rows, keys=('total_drops', 'fabric_link_drops', 'host_link_drops', 'avg_throughput', 'reroutes',
                                'delivered_mean_latency_us', 'delivered_p99_latency_us')):
    """(variant, signal, family, seed) -> metrics averaged over model seeds."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r['variant'], r['signal'], r['family'], r['seed'])].append(r)
    out = {}
    for k, v in acc.items():
        out[k] = {key: float(np.mean([x[key] for x in v if x.get(key) is not None])) if any(x.get(key) is not None for x in v) else None
                  for key in keys}
        out[k]['n_mseeds'] = len(v)
    return out


def boot(d, draws=2000, seed=42, cluster=False):
    """d: {(family, seed): value}. Returns bootstrap distribution of the mean."""
    rng = np.random.default_rng(seed)
    if not cluster:
        v = np.array(list(d.values()))
        return v[rng.integers(0, len(v), size=(draws, len(v)))].mean(1)
    fams = sorted({f for f, _ in d})
    byf = {f: np.array([d[(f, s)] for (ff, s) in d if ff == f]) for f in fams}
    out = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, len(fams), size=len(fams))
        out[i] = np.mean([byf[fams[j]][rng.integers(0, len(byf[fams[j]]), size=len(byf[fams[j]]))].mean() for j in pick])
    return out


def pct_change(vals, base):
    return (np.mean(list(vals.values())) / np.mean(list(base.values())) - 1) * 100


def compare(A, B, base_mean, key='total_drops'):
    """paired difference A-B as % of ECMP mean, with pair and cluster 95% / 90% CIs."""
    common = sorted(set(A) & set(B))
    d = {k: A[k][key] - B[k][key] for k in common}
    m = np.mean(list(d.values()))
    pb, cb = boot(d), boot(d, cluster=True)
    f = lambda x: (np.array(x) / base_mean * 100).round(3).tolist()
    return dict(n=len(common), mean=float(m), mean_pct=float(m / base_mean * 100),
                ci95_pct=f(np.percentile(pb, [2.5, 97.5])), ci90_pct=f(np.percentile(pb, [5, 95])),
                ci95_cluster_pct=f(np.percentile(cb, [2.5, 97.5])), ci90_cluster_pct=f(np.percentile(cb, [5, 95])),
                wins=int(sum(v < 0 for v in d.values())), ties=int(sum(v == 0 for v in d.values())),
                losses=int(sum(v > 0 for v in d.values())))


S = {}
E = ecmp_unseen()
pairsU = [(f, s) for f in FAM for s in UNSEEN]
ecmpU = {k: {kk: E[k][kk] for kk in ('total_drops', 'fabric_link_drops', 'host_link_drops', 'avg_throughput',
                                     'delivered_mean_latency_us', 'delivered_p99_latency_us')} for k in pairsU if k in E}
ecmp_total_mean_U = float(np.mean([v['total_drops'] for v in ecmpU.values()]))
S['ecmp_unseen'] = dict(n=len(ecmpU), mean_total=ecmp_total_mean_U,
                        mean_fabric=float(np.mean([v['fabric_link_drops'] for v in ecmpU.values()])),
                        mean_host=float(np.mean([v['host_link_drops'] for v in ecmpU.values()])),
                        mean_delivered=float(np.mean([v['avg_throughput'] for v in ecmpU.values()])))


def vs_ecmp(cells):
    """cells: {(fam, seed): metrics}. % change of means vs ECMP on the same pairs."""
    ks = [k for k in cells if k in ecmpU]
    if not ks:
        return None
    out = {}
    for key in ('total_drops', 'fabric_link_drops', 'host_link_drops', 'avg_throughput'):
        a = np.mean([cells[k][key] for k in ks]); b = np.mean([ecmpU[k][key] for k in ks])
        out[key] = float((a / b - 1) * 100)
    out['reroutes'] = float(np.mean([cells[k]['reroutes'] for k in ks]))
    for key, nm in (('delivered_mean_latency_us', 'mean_delay_ms'), ('delivered_p99_latency_us', 'p99_delay_ms')):
        if all(cells[k].get(key) is not None for k in ks):
            out[nm] = float(np.mean([cells[k][key] for k in ks]) / 1000)
            out['ecmp_' + nm] = float(np.mean([ecmpU[k][key] for k in ks]) / 1000)
    out['n'] = len(ks)
    d = {k: cells[k]['total_drops'] - ecmpU[k]['total_drops'] for k in ks}
    pb, cb = boot(d), boot(d, cluster=True)
    base = np.mean([ecmpU[k]['total_drops'] for k in ks])
    out['total_ci95_pct'] = (np.percentile(pb, [2.5, 97.5]) / base * 100).round(3).tolist()
    out['total_ci95_cluster_pct'] = (np.percentile(cb, [2.5, 97.5]) / base * 100).round(3).tolist()
    return out


def cells_of(avg, variant, signal):
    return {(f, s): v for (vv, sg, f, s), v in avg.items() if vv == variant and sg == signal}


# ── replicate (unseen) ──
rep = load('replicate')
if rep:
    A = avg_over_mseeds(rep, keys=('total_drops', 'fabric_link_drops', 'host_link_drops', 'avg_throughput', 'reroutes',
                                   'delivered_mean_latency_us', 'delivered_p99_latency_us'))
    S['replicate'] = {}
    for v, sig in (('ecmp', None), ('reactive', None), ('unguarded_gnn', 'gnn'), ('guarded_gnn', 'gnn'), ('guarded_lstm', 'lstm_only'),
                   ('guarded_persistence', 'persistence'), ('guarded_score', 'score')):
        c = cells_of(A, v, sig)
        if c:
            S['replicate'][v] = vs_ecmp(c)
            S['replicate'][v]['mean_delay_ms'] = float(np.mean([x['delivered_mean_latency_us'] for x in c.values()]) / 1000)
            S['replicate'][v]['p99_delay_ms'] = float(np.mean([x['delivered_p99_latency_us'] for x in c.values()]) / 1000)
    g = cells_of(A, 'guarded_gnn', 'gnn')
    S['replicate_tost'] = {}
    for v, sig in (('guarded_score', 'score'), ('guarded_lstm', 'lstm_only'), ('guarded_persistence', 'persistence')):
        o = cells_of(A, v, sig)
        if g and o and len(set(g) & set(o)) >= 10:
            S['replicate_tost'][v] = compare(g, o, ecmp_total_mean_U)

# ── ablation ──
ab = load('ablation')
if ab:
    A = avg_over_mseeds(ab)
    S['ablation'] = {}
    for sig in ('score', 'persistence', 'gnn'):
        full = cells_of(A, 'full', sig)
        S['ablation'][sig] = {}
        for v in ('full', 'no_capacity', 'no_reservation', 'no_budget', 'no_stretch', 'no_min_improvement', 'no_cooldown',
                  'fixed_threshold_0.7', 'all_rules_off'):
            c = cells_of(A, v, sig)
            if not c:
                continue
            e = vs_ecmp(c)
            if v != 'full' and full:
                e['vs_full'] = compare(c, full, ecmp_total_mean_U)
            e['n_mseeds'] = int(np.mean([x['n_mseeds'] for x in c.values()]))
            S['ablation'][sig][v] = e
    # GNN vs score under each ablation
    S['ablation_gnn_minus_score'] = {}
    for v in ('full', 'no_capacity', 'no_reservation', 'no_budget', 'no_stretch', 'no_min_improvement', 'no_cooldown',
              'fixed_threshold_0.7', 'all_rules_off'):
        a, b = cells_of(A, v, 'gnn'), cells_of(A, v, 'score')
        if a and b and len(set(a) & set(b)) >= 10:
            S['ablation_gnn_minus_score'][v] = compare(a, b, ecmp_total_mean_U)

# ── sweep ──
sw = load('sweep')
if sw:
    A = avg_over_mseeds(sw)
    S['sweep'] = defaultdict(dict)
    for (v, sig, f, s) in A:
        pass
    for sig in ('score', 'gnn'):
        for v in sorted({k[0] for k in A if k[1] == sig}):
            S['sweep'][sig][v] = vs_ecmp(cells_of(A, v, sig))
    S['sweep'] = dict(S['sweep'])
# M sweep from original guard_sweep_v4.json (score, unseen)
gs = json.load(open(NF_ROOT + '/research/guard_sweep_v4.json'))
S['sweep_M_score'] = {}
for p in gs['summary']['max_changes']:
    mc = p['max_changes']
    rs = [r for r in gs['runs'] if r.get('max_changes') == mc]
    S['sweep_M_score'][mc] = dict(total=p['total_change_vs_ecmp_percent'], reroutes=p['mean_reroutes_per_run'],
                                  p99_delay_ms=float(np.mean([r['delivered_p99_latency_us'] for r in rs]) / 1000),
                                  mean_delay_ms=float(np.mean([r['delivered_mean_latency_us'] for r in rs]) / 1000),
                                  delivered=(np.mean([r['avg_throughput'] for r in rs]) / S['ecmp_unseen']['mean_delivered'] - 1) * 100,
                                  n=len(rs))


# ── delay ──
dl = load('delay')
if dl:
    A = avg_over_mseeds(dl)
    S['delay'] = defaultdict(dict)
    for v in sorted({k[0] for k in A}):
        for sig in ('score', 'persistence', 'gnn'):
            c = cells_of(A, v, sig)
            if c:
                S['delay'][v][sig] = vs_ecmp(c)
        a, b = cells_of(A, v, 'gnn'), cells_of(A, v, 'score')
        if a and b and len(set(a) & set(b)) >= 10:
            S['delay'][v]['gnn_minus_score'] = compare(a, b, ecmp_total_mean_U)
    S['delay'] = dict(S['delay'])

# ── interval ──
iv = load('interval')
if iv:
    A = avg_over_mseeds(iv)
    S['interval'] = {v: vs_ecmp(cells_of(A, v, 'gnn')) for v in sorted({k[0] for k in A})}
    base_rows = [r for r in load('replicate') if r['variant'] == 'guarded_gnn' and r['mseed'] == 0]
    Ab = avg_over_mseeds(base_rows)
    S['interval_base'] = vs_ecmp(cells_of(Ab, 'guarded_gnn', 'gnn'))
    for I in ('I=2', 'I=5'):
        a = cells_of(A, I, 'gnn'); b = cells_of(Ab, 'guarded_gnn', 'gnn')
        if a and b:
            S['interval'][I]['vs_every_tick'] = compare(a, b, ecmp_total_mean_U)

# ── platt ──
pl = load('platt')
if pl:
    A = avg_over_mseeds(pl)
    S['platt'] = vs_ecmp(cells_of(A, 'guarded_gnn_platt', 'gnn_platt'))
    if rep:
        Ar = avg_over_mseeds(rep)
        S['platt_minus_temperature'] = compare(cells_of(A, 'guarded_gnn_platt', 'gnn_platt'), cells_of(Ar, 'guarded_gnn', 'gnn'), ecmp_total_mean_U)

# ── instrument (benchmark seeds) ──
ins = load('instrument')
if ins:
    agg = defaultdict(lambda: defaultdict(list))
    for r in ins:
        agg[r['variant']]['reordered'].append(r['reordered_packets'])
        agg[r['variant']]['delivered'].append(r['total_delivered'])
        agg[r['variant']]['flows_reord'].append(r['flows_with_reordering'] / r['n_flows'])
        agg[r['variant']]['loss_p50'].append(r['flow_loss_p50']); agg[r['variant']]['loss_p90'].append(r['flow_loss_p90'])
        agg[r['variant']]['loss_mean'].append(r['flow_loss_mean'])
        agg[r['variant']]['reroutes'].append(r['reroutes'])
    S['instrument'] = {v: dict(reordered_pct=float(np.sum(a['reordered']) / np.sum(a['delivered']) * 100),
                               flows_with_reordering_pct=float(np.mean(a['flows_reord']) * 100),
                               flow_loss_p50_pct=float(np.mean(a['loss_p50']) * 100), flow_loss_p90_pct=float(np.mean(a['loss_p90']) * 100),
                               flow_loss_mean_pct=float(np.mean(a['loss_mean']) * 100),
                               reroutes=float(np.mean(a['reroutes'])), runs=len(a['reroutes']))
                       for v, a in agg.items()}

# ── closed loop (benchmark seeds) ──
cl = load('closedloop')
if cl:
    agg = defaultdict(lambda: defaultdict(list))
    fam_agg = defaultdict(lambda: defaultdict(list))
    for r in cl:
        pf = [x for x in r['per_flow'] if x[2] > 0]
        fct = [x[4] for x in pf if x[4] is not None]
        slow = [x[4] / x[3] for x in pf if x[4] is not None and x[3]]
        a = agg[r['variant']]
        a['fct'].extend(fct); a['slow'].extend(slow); a['incomplete'].append(r['incomplete'])
        a['run_mean_fct'].append(np.mean(fct) if fct else np.nan)
        a['makespan'].append(r['makespan']); a['drops'].append(r['total_drops']); a['retx'].append(r['retx_packets'])
        a['offered'].append(r['offered_unique']); a['reroutes'].append(r['reroutes'])
        a['key'].append((r['family'], r['seed']))
        fam_agg[(r['variant'], r['family'])]['fct'].extend(fct)
        fam_agg[(r['variant'], r['family'])]['slow'].extend(slow)
    S['closedloop'] = {}
    for v, a in agg.items():
        S['closedloop'][v] = dict(flows=len(a['fct']), incomplete=int(np.sum(a['incomplete'])),
                                  fct_mean=float(np.mean(a['fct'])), fct_p99=float(np.percentile(a['fct'], 99)),
                                  slow_mean=float(np.mean(a['slow'])), slow_p50=float(np.median(a['slow'])),
                                  slow_p99=float(np.percentile(a['slow'], 99)),
                                  makespan=float(np.mean(a['makespan'])), drops=float(np.mean(a['drops'])),
                                  retx_pct=float(np.sum(a['retx']) / np.sum(a['offered']) * 100),
                                  reroutes=float(np.mean(a['reroutes'])), runs=len(a['makespan']))
    S['closedloop_family'] = {f'{v}|{f}': dict(fct_mean=float(np.mean(a['fct'])), slow_mean=float(np.mean(a['slow'])),
                                                slow_p99=float(np.percentile(a['slow'], 99)))
                              for (v, f), a in fam_agg.items() if a['fct']}
    # paired per-run mean-slowdown difference vs ECMP
    per_run = defaultdict(dict)
    for r in cl:
        pf = [x for x in r['per_flow'] if x[2] > 0 and x[4] is not None and x[3]]
        if pf:
            per_run[r['variant']].setdefault((r['family'], r['seed']), []).append(np.mean([x[4] / x[3] for x in pf]))
    base = {k: np.mean(v) for k, v in per_run.get('ecmp', {}).items()}
    S['closedloop_vs_ecmp'] = {}
    for v, d in per_run.items():
        if v == 'ecmp':
            continue
        dd = {k: np.mean(x) - base[k] for k, x in d.items() if k in base}
        if len(dd) >= 10:
            pb = boot(dd)
            S['closedloop_vs_ecmp'][v] = dict(mean_slowdown_diff=float(np.mean(list(dd.values()))),
                                               ci95=np.percentile(pb, [2.5, 97.5]).tolist(), n=len(dd))

# ── timing ──
tm = load('timing')
if tm:
    S['timing'] = {}
    for v in ('guarded_gnn', 'guarded_score', 'guarded_lstm'):
        rs = [r for r in tm if r['variant'] == v]
        if rs:
            c = np.concatenate([r['controller_ms'] for r in rs])
            inf = np.concatenate([r['inference_ms'] for r in rs]) if rs[0].get('inference_ms') else np.array([])
            S['timing'][v] = dict(ctrl_p50=float(np.median(c)), ctrl_p99=float(np.percentile(c, 99)), ctrl_mean=float(c.mean()),
                                  inf_p50=float(np.median(inf)) if inf.size else None,
                                  inf_p99=float(np.percentile(inf, 99)) if inf.size else None, ticks=int(c.size))

json.dump(S, open(os.path.join(R, 'summary.json'), 'w'), indent=1, default=float)
print(json.dumps({k: (list(v.keys()) if isinstance(v, dict) else v) for k, v in S.items()}, indent=0)[:3000])
