"""Concern 3 analysis -> revision/results/c3/summary.json and tables.md.

Checks: S1 (model-free reruns = rev3_load.jsonl), I2 (M0 GNN reruns vs rev3_load.jsonl, informational),
S2 (instrumented = uninstrumented on decisions and drops), S3 (no errors, packet conservation, drop split),
S4 (oracle reruns at load 1.0 = rev4_c1.jsonl). Routing: per-load means, % vs ECMP, route changes, reordering;
paired differences in % of the mean ECMP drops with pair and family-clustered bootstrap CIs (rev5_stats.compare:
2,000 draws, seed 42); model variants averaged over the three training seeds per (family, seed) cell. At a load
whose ECMP drops are 0 the differences are reported in packets. Findings (a) forecaster and (b) guard effectiveness
are kept in separate sections."""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c3_common as C
import numpy as np
import rev5_stats

R3 = os.path.join(C.ROOT, 'revision', 'results')
KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'total_offered', 'total_delivered',
        'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes']
C1_KEEP = KEEP + ['reordered_packets', 'reordered_fraction', 'flows_with_reordering', 'n_flows']
MODEL_FREE = ['ecmp', 'guarded_score', 'score_no_rules', 'random']
LABEL = {'ecmp': 'ECMP', 'guarded_score': 'score/0.851', 'score_pmix': 'score/Pmix', 'guarded_gnn': 'M0/P1.0',
         'm0p_p10': "M0'/P1.0", 'm1_pmix': 'M1/Pmix', 'score_no_rules': 'score, no rules', 'random': 'random/0.851',
         'm1_p10': 'M1/P1.0', 'm0_pmix': 'M0/Pmix', 'm0p_pmix': "M0'/Pmix", 'oracle_label': 'oracle label/0.5',
         'oracle_score_pmix': 'oracle score/Pmix', 'oracle_score_0851': 'oracle score/0.851'}
COMPARISONS = [  # (name, a, b, section)
    ('PRIMARY M1/Pmix - score/Pmix', 'm1_pmix', 'score_pmix', 'a'),
    ("M1/Pmix - M0'/P1.0 (mixed-training effect, same environment)", 'm1_pmix', 'm0p_p10', 'a'),
    ("M0'/P1.0 - score/0.851 (same-environment replicate)", 'm0p_p10', 'guarded_score', 'a'),
    ('M0/P1.0 - score/0.851 (original checkpoints)', 'guarded_gnn', 'guarded_score', 'a'),
    ('M1/Pmix - M0/P1.0', 'm1_pmix', 'guarded_gnn', 'a'),
    ('score/Pmix - score/0.851 (threshold effect, score)', 'score_pmix', 'guarded_score', 'a'),
    ("M1/P1.0 - M0'/P1.0 (model effect at P1.0)", 'm1_p10', 'm0p_p10', 'a2'),
    ("M1/Pmix - M0'/Pmix (model effect at Pmix)", 'm1_pmix', 'm0p_pmix', 'a2'),
    ('M1/Pmix - M1/P1.0 (threshold effect, M1)', 'm1_pmix', 'm1_p10', 'a2'),
    ("M0'/Pmix - M0'/P1.0 (threshold effect, M0')", 'm0p_pmix', 'm0p_p10', 'a2'),
    ('M0/Pmix - M0/P1.0 (threshold effect, M0)', 'm0_pmix', 'guarded_gnn', 'a2'),
    ('score/0.851 - ECMP', 'guarded_score', 'ecmp', 'b'),
    ('score/Pmix - ECMP', 'score_pmix', 'ecmp', 'b'),
    ('score, no rules - ECMP', 'score_no_rules', 'ecmp', 'b'),
    ('random/0.851 - ECMP', 'random', 'ecmp', 'b'),
]
ORACLE_COMPARISONS = [
    ('oracle label - score/Pmix', 'oracle_label', 'score_pmix'), ('oracle score/Pmix - score/Pmix', 'oracle_score_pmix', 'score_pmix'),
    ('oracle score/0.851 - score/0.851', 'oracle_score_0851', 'guarded_score'),
    ('oracle label - ECMP', 'oracle_label', 'ecmp'), ('oracle score/Pmix - ECMP', 'oracle_score_pmix', 'ecmp'),
    ('oracle score/0.851 - ECMP', 'oracle_score_0851', 'ecmp'),
    ('oracle label - M1/Pmix', 'oracle_label', 'm1_pmix'), ('oracle score/Pmix - M1/Pmix', 'oracle_score_pmix', 'm1_pmix'),
]


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def jload(name):
    p = os.path.join(C.OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def cells(rows):
    """variant -> load -> (family, seed) -> row (numeric fields averaged over training seeds)."""
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        g[r['variant']][r['load']][(r['family'], r['seed'])].append(r)
    out = {}
    for v, d in g.items():
        out[v] = {}
        for lf, cs in d.items():
            out[v][lf] = {}
            for c, rs in cs.items():
                avg = {k: float(np.mean([x[k] for x in rs])) for k in KEEP + ['reordered_packets', 'flows_with_reordering']
                       if all(x.get(k) is not None for x in rs)}
                avg['n_seeds'] = len(rs)
                out[v][lf][c] = avg
    return out


def diff(A, B, E):
    """paired A - B over common (family, seed) cells."""
    common = sorted(set(A) & set(B))
    if not common:
        return None
    if E > 0:
        r = rev5_stats.compare({k: A[k] for k in common}, {k: B[k] for k in common}, E)
        r['unit'] = '% of mean ECMP drops'
        return r
    d = {k: A[k]['total_drops'] - B[k]['total_drops'] for k in common}
    pb, cb = rev5_stats.boot(d), rev5_stats.boot(d, cluster=True)
    return dict(n=len(common), mean=float(np.mean(list(d.values()))), unit='packets (ECMP drops are 0)',
                ci95_pkts=np.percentile(pb, [2.5, 97.5]).round(3).tolist(),
                ci95_cluster_pkts=np.percentile(cb, [2.5, 97.5]).round(3).tolist(),
                wins=int(sum(v < 0 for v in d.values())), ties=int(sum(v == 0 for v in d.values())),
                losses=int(sum(v > 0 for v in d.values())))


def checks(load_rows, oracle_rows):
    rev3 = {r['id']: r for r in jl(os.path.join(R3, 'rev3_load.jsonl'))}
    out = {}
    # S3
    allrows = load_rows + oracle_rows
    errs = [r['id'] for r in allrows if 'error' in r]
    bad = [r['id'] for r in allrows if 'error' not in r and not (r['packet_conservation'] and r['drop_split_ok'])]
    out['S3'] = dict(ok=not errs and not bad, rows=len(allrows), errors=errs[:20], invariant_failures=bad[:20])
    ok_rows = [r for r in load_rows if 'error' not in r]
    # S1
    mf = [r for r in ok_rows if r['variant'] in MODEL_FREE]
    mism = [r['id'] for r in mf if r['id'] not in rev3 or any(r[k] != rev3[r['id']][k] for k in KEEP)]
    out['S1'] = dict(ok=bool(mf) and not mism, compared=len(mf), mismatches=mism[:20], n_mismatch=len(mism))
    # I2
    gg = [r for r in ok_rows if r['variant'] == 'guarded_gnn' and r['id'] in rev3]
    dd = [r['total_drops'] - rev3[r['id']]['total_drops'] for r in gg]
    neq = [r for r in gg if any(r[k] != rev3[r['id']][k] for k in KEEP)]
    out['I2'] = dict(compared=len(gg), identical=len(gg) - len(neq), differing=len(neq),
                     max_abs_total_drops_diff=float(max(map(abs, dd))) if dd else None,
                     mean_total_drops_diff=float(np.mean(dd)) if dd else None,
                     reroutes_diff_rows=int(sum(r['reroutes'] != rev3[r['id']]['reroutes'] for r in gg)))
    # S2
    plain = {r['id']: r for r in jl(os.path.join(C.OUT, 'instrument_check_plain.jsonl')) if 'error' not in r}
    inst = {r['id']: r for r in ok_rows}
    fields = KEEP + ['decision_digest', 'signal_digest']
    s2 = [i for i in plain if i in inst and any(plain[i].get(k) != inst[i].get(k) for k in fields)]
    out['S2'] = dict(ok=bool(plain) and all(i in inst for i in plain) and not s2, compared=len(plain), mismatches=s2)
    # S4
    c1 = {r['id']: r for r in jl(os.path.join(R3, 'rev4_c1.jsonl'))}
    rep = [r for r in jl(os.path.join(C.OUT, 'oracle_repro.jsonl')) if 'error' not in r]
    name = {'oracle_label': 'oracle_label', 'oracle_score_0851': 'oracle_score'}
    s4 = []
    for r in rep:
        ref = c1.get(f"exact|{name[r['variant']]}|None|{r['family']}|{r['seed']}")
        if ref is None or any(r[k] != ref[k] for k in C1_KEEP):
            s4.append(r['id'])
    out['S4'] = dict(ok=bool(rep) and not s4, compared=len(rep), mismatches=s4)
    return out


def fmt_ci(x):
    if x is None:
        return 'not run'
    if 'mean_pct' in x:
        return f"{x['mean_pct']:+.2f} [{x['ci95_pct'][0]:+.2f}, {x['ci95_pct'][1]:+.2f}] / [{x['ci95_cluster_pct'][0]:+.2f}, {x['ci95_cluster_pct'][1]:+.2f}]"
    return f"{x['mean']:+.1f} pkts [{x['ci95_pkts'][0]:+.1f}, {x['ci95_pkts'][1]:+.1f}] / [{x['ci95_cluster_pkts'][0]:+.1f}, {x['ci95_cluster_pkts'][1]:+.1f}]"


def main():
    load_rows = jl(os.path.join(C.OUT, 'load.jsonl'))
    oracle_rows = jl(os.path.join(C.OUT, 'oracle.jsonl'))
    S = dict(meta=C.meta(step='analyze'), checks=checks(load_rows, oracle_rows))
    good = [r for r in load_rows if 'error' not in r]
    cl = cells(good)
    ocl = cells([r for r in oracle_rows if 'error' not in r])
    S['variants_run'] = {v: {str(lf): len(cs) for lf, cs in d.items()} for v, d in cl.items()}
    routing = {}
    for lf in C.LOADS:
        e = cl.get('ecmp', {}).get(lf)
        if not e:
            continue
        E = float(np.mean([x['total_drops'] for x in e.values()]))
        blk = dict(ecmp_mean_drops=E, ecmp_loss_pct=float(sum(x['total_drops'] for x in e.values()) /
                                                          sum(x['total_offered'] for x in e.values()) * 100), variants={}, diffs={})
        for v, d in cl.items():
            c = d.get(lf)
            if not c:
                continue
            D = float(np.mean([x['total_drops'] for x in c.values()]))
            blk['variants'][v] = dict(
                cells=len(c), mean_drops=D, total_pct_vs_ecmp=(D / E - 1) * 100 if E > 0 else None,
                reroutes=float(np.mean([x['reroutes'] for x in c.values()])),
                reordered_pct=float(sum(x.get('reordered_packets', 0) for x in c.values()) / sum(x['total_delivered'] for x in c.values()) * 100),
                p99_ms=float(np.mean([x['delivered_p99_latency_us'] for x in c.values()]) / 1000))
        for name, a, b, sec in COMPARISONS:
            A, B = cl.get(a, {}).get(lf), cl.get(b, {}).get(lf)
            blk['diffs'][name] = dict(section=sec, result=diff(A, B, E) if A and B else None)
        routing[str(lf)] = blk
    # aggregate over all loads: per base scenario, drops summed over the loads
    agg = {}
    def summed(v):
        d = cl.get(v)
        if not d or set(d) != set(C.LOADS):
            return None
        keys = set.intersection(*(set(d[lf]) for lf in C.LOADS))
        return {k: dict(total_drops=sum(d[lf][k]['total_drops'] for lf in C.LOADS)) for k in keys}
    E_all = summed('ecmp')
    if E_all:
        Em = float(np.mean([x['total_drops'] for x in E_all.values()]))
        for name, a, b, sec in COMPARISONS:
            A, B = summed(a), summed(b)
            agg[name] = dict(section=sec, result=diff(A, B, Em) if A and B else None)
    S['routing'], S['routing_all_loads'] = routing, agg
    # oracle (loads 0.2, 0.25)
    orc = {}
    for lf in (0.2, 0.25):
        e = cl.get('ecmp', {}).get(lf)
        if not e:
            continue
        E = float(np.mean([x['total_drops'] for x in e.values()]))
        both = {**{v: d for v, d in cl.items()}, **{v: d for v, d in ocl.items()}}
        blk = dict(variants={}, diffs={})
        for v, d in ocl.items():
            c = d.get(lf)
            if c:
                D = float(np.mean([x['total_drops'] for x in c.values()]))
                Ec = float(np.mean([e[k]['total_drops'] for k in c]))      # ECMP over the same cells
                blk['variants'][v] = dict(cells=len(c), families=sorted({f for f, _ in c}), mean_drops=D, ecmp_mean_drops_same_cells=Ec,
                                          total_pct_vs_ecmp=(D / Ec - 1) * 100 if Ec > 0 else None,
                                          reroutes=float(np.mean([x['reroutes'] for x in c.values()])),
                                          reordered_pct=float(sum(x.get('reordered_packets', 0) for x in c.values()) /
                                                              sum(x['total_delivered'] for x in c.values()) * 100))
        for name, a, b in ORACLE_COMPARISONS:
            A, B = both.get(a, {}).get(lf), both.get(b, {}).get(lf)
            if A and B:
                common = set(A) & set(B)
                Ec = float(np.mean([e[k]['total_drops'] for k in common]))   # normalise by ECMP over the compared cells
                blk['diffs'][name] = diff({k: A[k] for k in common}, {k: B[k] for k in common}, Ec)
            else:
                blk['diffs'][name] = None
        orc[str(lf)] = blk
    S['oracle'] = orc
    C.write_json(os.path.join(C.OUT, 'summary.json'), S)
    tables(S)
    print(json.dumps(S['checks'], indent=1))


def tables(S):
    L = ['# Concern 3 tables (generated by c3_analyze.py)', '']
    # forecast
    F = jload('forecast.json')
    if F:
        for tset in ('unseen', 'benchmark'):
            if tset not in F['sets']:
                continue
            L += [f'## Forecast — {tset} test set (AP: mean ± sd over 3 training seeds; CIs: scenario bootstrap)', '',
                  '| load | positives | scen. w/ pos | AP M0 | AP M0′ | AP M1 | AP persistence | onset AP M0′ / M1 / pers. | Brier M0′ / M1 | flags | M1 − M0′ AP [95% CI] | M1 − persistence AP [95% CI] | onset M1 − M0′ [95% CI] |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
            for lf in [str(x) for x in C.LOADS] + ['pooled']:
                b = F['sets'][tset].get(lf)
                if b is None:
                    continue
                m = b['models']
                ap = lambda n, key='ap': 'undef.' if m[n][key] is None else f"{m[n][key]['mean']:.3f} ± {m[n][key]['sd']:.3f}"
                pa = 'undef.' if m['persistence']['ap'] is None else f"{m['persistence']['ap']:.3f}"
                oa = lambda n: 'undef.' if m[n]['onset_ap'] is None else f"{m[n]['onset_ap']['mean']:.3f}"
                po = 'undef.' if m['persistence']['onset_ap'] is None else f"{m['persistence']['onset_ap']:.3f}"
                br = lambda n: '—' if m[n]['brier'] is None else f"{m[n]['brier']['mean']:.4f}"
                flags = 'AP undefined' if b['ap_undefined'] else ('unstable' if b['unstable'] else 'ok')
                if not b['onset_ap_undefined'] and b['onset_unstable']:
                    flags += '; onset unstable'
                bs = b['bootstrap']
                ci = lambda part, c: '—' if not bs or bs[part].get(c) is None or bs[part][c]['point'] is None else \
                    f"{bs[part][c]['point']:+.4f} [{bs[part][c]['ci95'][0]:+.4f}, {bs[part][c]['ci95'][1]:+.4f}]"
                L.append(f"| {lf} | {b['positives']} | {b['scenarios_with_positives']}/{b['scenarios']} | {ap('M0')} | {ap('M0p')} | {ap('M1')} | {pa} | "
                         f"{oa('M0p')} / {oa('M1')} / {po} | {br('M0p')} / {br('M1')} | {flags} | {ci('overall', 'M1-M0p')} | {ci('overall', 'M1-persistence')} | {ci('onset', 'M1-M0p')} |")
            L.append('')
    # routing
    R = S['routing']
    if R:
        cols = ['ecmp', 'guarded_score', 'score_pmix', 'guarded_gnn', 'm0p_p10', 'm1_pmix', 'm1_p10', 'm0_pmix', 'm0p_pmix', 'score_no_rules', 'random']
        L += ['## Routing — total drops vs ECMP (%), route changes per run, reordered packets (% of delivered)', '',
              '| load | ECMP loss % | ' + ' | '.join(LABEL[c] for c in cols) + ' |', '|---|---|' + '---|' * len(cols)]
        for lf, b in R.items():
            cell = lambda c: 'not run' if c not in b['variants'] else (
                ('n/a' if b['variants'][c]['total_pct_vs_ecmp'] is None else f"{b['variants'][c]['total_pct_vs_ecmp']:+.2f}%")
                + f"; {b['variants'][c]['reroutes']:.1f} rc; {b['variants'][c]['reordered_pct']:.3f}% ro")
            L.append(f"| {lf} | {b['ecmp_loss_pct']:.2f} | " + ' | '.join(cell(c) for c in cols) + ' |')
        L += ['', '## Routing — paired differences (mean [pair 95% CI] / [family-clustered 95% CI]; negative = fewer drops than the reference)', '']
        names = [n for n, *_ in COMPARISONS]
        L += ['| comparison | section | ' + ' | '.join(R) + ' | all loads |', '|---|---|' + '---|' * (len(R) + 1)]
        for n, a, b2, sec in COMPARISONS:
            row = [fmt_ci(R[lf]['diffs'][n]['result']) for lf in R]
            ag = S['routing_all_loads'].get(n)
            L.append(f'| {n} | {sec} | ' + ' | '.join(row) + f" | {fmt_ci(ag['result']) if ag else 'not run'} |")
        L.append('')
    O = S['oracle']
    if O:
        L += ['## Passive oracle (loads 0.2, 0.25)', '', '| load | variant | cells | families | vs ECMP | route changes | reordered % |', '|---|---|---|---|---|---|---|']
        for lf, b in O.items():
            for v, x in b['variants'].items():
                L.append(f"| {lf} | {LABEL[v]} | {x['cells']} | {len(x['families'])} | {x['total_pct_vs_ecmp']:+.2f}% | {x['reroutes']:.1f} | {x['reordered_pct']:.3f} |")
        L += ['', '| comparison | ' + ' | '.join(O) + ' |', '|---|' + '---|' * len(O)]
        for n, *_ in ORACLE_COMPARISONS:
            L.append(f'| {n} | ' + ' | '.join(fmt_ci(O[lf]['diffs'].get(n)) for lf in O) + ' |')
        L.append('')
    path = os.path.join(C.OUT, 'tables.md')
    C.guard_output(path)
    open(path, 'w', encoding='utf8').write('\n'.join(L))


if __name__ == '__main__':
    main()
