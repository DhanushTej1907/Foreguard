"""Concern 4 follow-up: family-clustered 95% CIs for the closed-loop FCT contrasts (Tables A6 and A10). Analysis only.

  python revision/scripts/c4_clustered.py      -> results/c4/clustered.json, tables_clustered.md

Reads results/c4/runs.jsonl (9 settings x 8 strategies x 30 cells). No new simulation runs.

Two bootstraps, both 2,000 draws with seed 42, the same sampled cells for every strategy in a draw, statistics on the
concatenated flows of the sampled cells (pooled mean and pooled p99 FCT, unfinished flows censored as in Table A6):
  cell       30 (family, traffic seed) cells resampled with replacement (the existing analysis);
  clustered  6 families resampled with replacement, then the 5 seeds of each drawn family resampled with replacement;
             a family drawn twice gets independent within-family draws (design of summarize_all.boot(cluster=True),
             with the same order of random-number calls).
Gate (blocker): the cell path must reproduce results/c4/summary.json (points, tick and % CIs, verdicts),
rev4_c4_summary.json (Table A6) and rev4_fb_summary.json closed (FlowBender rows) exactly. Nothing is written otherwise.

Provenance (copied, not imported, so that no module that writes files at import time is loaded):
  start, cells, stats      rev4_c4_analyze.py lines 9-31 (FCT per flow, censoring, pooled mean / p99), unchanged;
  boot_draws, ci, verdict  c4_analyze.py (summary section), unchanged;
  contrast                 c4_analyze.contrast, generalised to take the draw distributions of either bootstrap;
  cluster_draws            summarize_all.boot(cluster=True): same rng call order, cell keys instead of values.
"""
from paths import NF_ROOT, RESULTS  # folder locations (see paths.py)
import json, math, os, sys, time
from collections import defaultdict
import numpy as np

C4 = os.path.join(RESULTS, 'c4')
RUNS = os.path.join(C4, 'runs.jsonl')
OUT_JSON = os.path.join(C4, 'clustered.json')
OUT_MD = os.path.join(C4, 'tables_clustered.md')

VARIANTS = ['ecmp', 'reactive', 'unguarded_gnn', 'score_no_rules', 'fb_default', 'guarded_gnn', 'guarded_persistence',
            'guarded_score']
TABLE_A6 = [v for v in VARIANTS if v != 'fb_default']
PRIMARY = ['score_no_rules', 'fb_default', 'ecmp']
SETTINGS = ['off', 'W0', 'W1', 'W2', 'W4', 'W8', 'W16', 'rtt4', 'dup3']
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
CELLS = [(f, s) for f in FAM for s in range(3000, 3005)]
DRAWS, SEED = 2000, 42
LABEL = dict(ecmp='ECMP', reactive='reactive', unguarded_gnn='unguarded GNN', score_no_rules='score without rules',
             fb_default='FlowBender-style', guarded_gnn='guarded GNN', guarded_persistence='guarded persistence',
             guarded_score='guarded score')

# ─────────────── copied from rev4_c4_analyze.py (lines 9-31), unchanged ───────────────
_g, _starts = None, {}


def start(f, s, fid):
    global _g
    if (f, s) not in _starts:
        sys.path.insert(0, os.environ.get('NF_ROOT', NF_ROOT))
        from simulation.topology import build_fat_tree
        from simulation.traffic_generator import TrafficGenerator
        if _g is None:
            _g = build_fat_tree()
        _starts[(f, s)] = {x.flow_id: x.start_tick for x in TrafficGenerator(_g, seed=s).generate_scenario(total_ticks=150, pattern=f)}
    return _starts[(f, s)][fid]


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


# ─────────────── copied from c4_analyze.py, unchanged ───────────────
def boot_draws(n):
    rng = np.random.default_rng(SEED)            # identical draw sequence to rev4_c4_analyze.compare
    return [rng.integers(0, n, n) for _ in range(DRAWS)]


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def verdict(c):
    return '+' if c[0] > 0 else ('-' if c[1] < 0 else '0')


# ─────────────── new ───────────────
def cell_draws(keys):
    """Single-stage: lists of cell keys, the same index sequence as c4_analyze.boot_draws."""
    return [[keys[j] for j in ix] for ix in boot_draws(len(keys))]


def cluster_draws(keys, draws=DRAWS, seed=SEED):
    """Two-stage: families with replacement, then each drawn family's seeds with replacement (independently each time
    the family is drawn). rng call order as in summarize_all.boot(cluster=True)."""
    rng = np.random.default_rng(seed)
    fams = sorted({f for f, _ in keys})
    byf = {f: [k for k in keys if k[0] == f] for f in fams}
    out = []
    for _ in range(draws):
        pick = rng.integers(0, len(fams), size=len(fams))
        ks = []
        for j in pick:
            m = byf[fams[j]]
            ks += [m[i] for i in rng.integers(0, len(m), size=len(m))]
        out.append(ks)
    return out


def distribution(Cv, draws):
    dm, dp = np.empty(len(draws)), np.empty(len(draws))
    for i, ks in enumerate(draws):
        x = stats(Cv, ks)
        dm[i], dp[i] = x['mean'], x['p99']
    return dict(mean=dm, p99=dp)


def interval(Da, Db, m):
    d, p = Da[m] - Db[m], (Da[m] / Db[m] - 1) * 100
    cd, cp = ci(d), ci(p)
    return dict(diff_ci=cd, pct_ci=cp, pct_ci_r2=np.round(np.percentile(p, [2.5, 97.5]), 2).tolist(),
                verdict=verdict(cd), verdict_pct=verdict(cp))


def contrast(P, Dc, Dk, a, b):
    """FCT(a) - FCT(b): point (ticks, %), cell and clustered intervals. Positive = a slower than b."""
    out = {}
    for m in ('mean', 'p99'):
        out[m] = dict(diff=P[a][m] - P[b][m], pct=(P[a][m] / P[b][m] - 1) * 100,
                      cell=interval(Dc[a], Dc[b], m), clustered=interval(Dk[a], Dk[b], m))
        out[m]['verdict_change'] = out[m]['cell']['verdict'] != out[m]['clustered']['verdict']
        out[m]['verdict_change_pct'] = out[m]['cell']['verdict_pct'] != out[m]['clustered']['verdict_pct']
    return out


def load_rows():
    R = {}
    for l in open(RUNS):
        r = json.loads(l)
        if 'error' not in r:
            R[r['id']] = r
    return R


def per_family(C, keys, a, b='guarded_score'):
    """Per-family pooled mean-FCT contrast, its exact share of the pooled contrast, and the pooled contrast without it."""
    N = sum(len(C[a][k]['cens']) for k in keys)
    pa, pb = stats(C[a], keys)['mean'], stats(C[b], keys)['mean']
    out = {}
    for f in FAM:
        fk = [k for k in keys if k[0] == f]
        ok = [k for k in keys if k[0] != f]
        n = sum(len(C[a][k]['cens']) for k in fk)
        ma, mb = stats(C[a], fk)['mean'], stats(C[b], fk)['mean']
        xa, xb = stats(C[a], ok)['mean'], stats(C[b], ok)['mean']
        seed_d = [float(C[a][k]['cens'].mean() - C[b][k]['cens'].mean()) for k in fk]
        out[f] = dict(n_flows=n, flow_share=n / N, mean_a=float(ma), mean_b=float(mb), diff=float(ma - mb),
                      pct=float((ma / mb - 1) * 100), contribution=float(n * (ma - mb) / N),
                      contribution_share=float(n * (ma - mb) / N / (pa - pb)) if pa != pb else None,
                      seed_diffs=seed_d, seeds_positive=sum(x > 0 for x in seed_d), seeds_zero=sum(x == 0 for x in seed_d),
                      p99_diff=float(np.percentile(np.concatenate([C[a][k]['cens'] for k in fk]), 99)
                                     - np.percentile(np.concatenate([C[b][k]['cens'] for k in fk]), 99)),
                      loo_diff=float(xa - xb), loo_pct=float((xa / xb - 1) * 100))
    return out


# ─────────────── gate ───────────────
def gate(res, R):
    """Cell path vs the saved analyses. Exact equality after a JSON round trip (as the saved files were written)."""
    rt = lambda x: json.loads(json.dumps(x, default=float))
    checks, n = [], 0
    S = json.load(open(os.path.join(C4, 'summary.json')))['settings']
    for s in SETTINGS:
        for v in VARIANTS:
            mine, ref = res[s]['variants'][v], S[s]['variants'][v]
            for k in ('fct_mean', 'fct_p99'):
                n += 1
                if rt(mine[k]) != ref[k]:
                    checks.append(f'summary.json {s} {v} {k}')
            for ref_name in ('vs_ecmp', 'vs_guarded_score'):
                if ref_name not in ref:
                    continue
                for m in ('mean', 'p99'):
                    a, b = rt(mine[ref_name][m]), ref[ref_name][m]
                    for k in ('diff', 'pct'):
                        n += 1
                        if a[k] != b[k]:
                            checks.append(f'summary.json {s} {v} {ref_name} {m} {k}')
                    for k in ('diff_ci', 'pct_ci', 'pct_ci_r2', 'verdict', 'verdict_pct'):
                        n += 1
                        if a['cell'][k] != b[k]:
                            checks.append(f'summary.json {s} {v} {ref_name} {m} {k}')
    S4 = json.load(open(os.path.join(RESULTS, 'rev4_c4_summary.json')))
    SF = json.load(open(os.path.join(RESULTS, 'rev4_fb_summary.json')))['closed']
    for name, s in (('no_penalty', 'off'), ('penalty', 'W0')):
        for src, ref_blk, vs in (('rev4_c4_summary', S4[name], TABLE_A6), ('rev4_fb_summary.closed', {'fb_default': SF[name]}, ['fb_default'])):
            for v in vs:
                mine, ref = res[s]['variants'][v], ref_blk[v]
                for ref_name in ('vs_ecmp', 'vs_guarded_score'):
                    if ref_name not in ref:
                        continue
                    for m in ('mean', 'p99'):
                        n += 2
                        if rt(mine[ref_name][m]['pct']) != ref[ref_name][f'{m}_pct']:
                            checks.append(f'{src} {name} {v} {ref_name} {m}_pct')
                        if mine[ref_name][m]['cell']['pct_ci_r2'] != ref[ref_name][f'{m}_ci']:
                            checks.append(f'{src} {name} {v} {ref_name} {m}_ci')
    return dict(ok=not checks, values_compared=n, mismatches=checks)


# ─────────────── main ───────────────
def analyse():
    t0 = time.time()
    R = load_rows()
    res, fam = {}, {}
    D_cell = D_clu = None
    for s in SETTINGS:
        rows = [R[f'{s}|{v}|{f}|{sd}'] for v in VARIANTS for f, sd in CELLS]
        C = cells(rows)
        keys = sorted(C['ecmp'])
        assert all(sorted(C[v]) == keys for v in VARIANTS) and len(keys) == 30
        assert all(len(C[v][k]['cens']) == len(C['ecmp'][k]['cens']) for v in VARIANTS for k in keys)
        if D_cell is None:
            D_cell, D_clu = cell_draws(keys), cluster_draws(keys)
        P, Dc, Dk = {}, {}, {}
        for v in VARIANTS:
            st = stats(C[v], keys)
            P[v] = dict(mean=float(st['mean']), p99=float(st['p99']))
            Dc[v], Dk[v] = distribution(C[v], D_cell), distribution(C[v], D_clu)
        V = {}
        for v in VARIANTS:
            o = dict(fct_mean=P[v]['mean'], fct_p99=P[v]['p99'])
            if v != 'ecmp':
                o['vs_ecmp'] = contrast(P, Dc, Dk, v, 'ecmp')
            if v != 'guarded_score':
                o['vs_guarded_score'] = contrast(P, Dc, Dk, v, 'guarded_score')
            V[v] = o
        res[s] = dict(variants=V)
        fam[s] = {v: per_family(C, keys, v) for v in VARIANTS if v != 'guarded_score'}
        print('done', s, f'{time.time() - t0:.0f}s', flush=True)
    G = gate(res, R)
    print('gate', G['ok'], G['values_compared'], G['mismatches'][:10])
    if not G['ok']:
        sys.exit('GATE FAILED: nothing written')
    # draw-design diagnostics
    fams = sorted(FAM)
    dup = [len(ks) // 5 - len({k[0] for k in ks}) for ks in D_clu]
    changes = [dict(setting=s, strategy=v, reference=ref, metric=m, cell=c['cell']['verdict'], clustered=c['clustered']['verdict'],
                    cell_pct=c['cell']['verdict_pct'], clustered_pct=c['clustered']['verdict_pct'])
               for s in SETTINGS for v, o in res[s]['variants'].items() for ref in ('vs_guarded_score', 'vs_ecmp') if ref in o
               for m, c in o[ref].items() if c['verdict_change'] or c['verdict_change_pct']]
    flows = {}
    for f in FAM:
        sc = [R[f'off|ecmp|{f}|{sd}']['per_flow'] for sd in range(3000, 3005)]
        flows[f] = dict(n_flows_per_seed=[len([x for x in p if x[2] > 0]) for p in sc],
                        same_flow_ids_and_sizes_across_seeds=all([(x[0], x[2]) for x in p] == [(x[0], x[2]) for x in sc[0]] for p in sc),
                        ecmp_fabric_drops_per_seed=[R[f'off|ecmp|{f}|{sd}']['fabric_link_drops'] for sd in range(3000, 3005)])
    out = dict(meta=dict(draws=DRAWS, seed=SEED, cells=len(CELLS), families=fams,
                         cell='30 (family, traffic seed) cells with replacement (c4_analyze.boot_draws)',
                         clustered='6 families with replacement, then 5 seeds within each drawn family with replacement; '
                                   'independent within-family draws for repeated families (summarize_all.boot(cluster=True) rng order)',
                         statistic='pooled mean / p99 FCT over the concatenated flows of the sampled cells; censored at the horizon',
                         contrast='FCT(strategy) - FCT(reference), ticks and %; positive = strategy slower',
                         verdict="tick CI: '+' entire CI > 0, '-' entire CI < 0, '0' includes 0; verdict_pct on the % CI",
                         clustered_draws_with_a_repeated_family=sum(d > 0 for d in dup) / DRAWS,
                         mean_distinct_families_per_draw=float(np.mean([len({k[0] for k in ks}) for ks in D_clu])),
                         runtime_s=round(time.time() - t0, 1)),
               gate=G, settings=res, verdict_changes=changes, per_family=fam, family_flows=flows)
    json.dump(out, open(OUT_JSON, 'w'), indent=1, default=float)
    open(OUT_MD, 'w', encoding='utf-8').write(tables(out))
    print('verdict changes', len(changes), 'runtime', out['meta']['runtime_s'])
    return out


def tables(out):
    pc = lambda c: f"[{c[0]:+.1f}, {c[1]:+.1f}]"
    L = ['# Concern 4 — family-clustered CIs for closed-loop FCT contrasts (generated by c4_clustered.py)', '',
         'Contrast = FCT(strategy) − FCT(reference), shown in % of the reference (positive = strategy slower). '
         'Cell CI: 30 (family, seed) cells resampled. Clustered CI: 6 families resampled, then 5 seeds within each drawn '
         'family. Both 95%, 2,000 draws, seed 42, same cells for both strategies in each draw. Verdict on the tick CI: '
         '+ entire CI > 0, − entire CI < 0, 0 includes 0 (vs guarded score, + = guard advantage detected). '
         '`*` = the % CI gives a different verdict than the tick CI.', '']
    sym = lambda x: {'+': '+', '-': '−', '0': '0'}[x]
    def cellfmt(c):
        a = sym(c['cell']['verdict']) + ('*' if c['cell']['verdict_pct'] != c['cell']['verdict'] else '')
        b = sym(c['clustered']['verdict']) + ('*' if c['clustered']['verdict_pct'] != c['clustered']['verdict'] else '')
        flag = ' **≠**' if c['verdict_change'] else ''
        return f"{c['pct']:+.1f}% | {pc(c['cell']['pct_ci'])} {a} | {pc(c['clustered']['pct_ci'])} {b}{flag}"
    for ref, title in (('vs_guarded_score', 'vs guarded score'), ('vs_ecmp', 'vs ECMP')):
        L += [f'## Contrasts {title}', '']
        for s in SETTINGS:
            L += [f'### Setting {s}', '', '| strategy | mean Δ% | mean cell CI v | mean clustered CI v | p99 Δ% | p99 cell CI v | p99 clustered CI v |',
                  '|---|---|---|---|---|---|---|']
            for v, o in out['settings'][s]['variants'].items():
                if ref in o:
                    L.append(f"| {LABEL[v]} | {cellfmt(o[ref]['mean'])} | {cellfmt(o[ref]['p99'])} |")
            L.append('')
    L += ['## Verdict changes (cell → clustered)', '', '| setting | strategy | reference | metric | cell (tick / %) | clustered (tick / %) |',
          '|---|---|---|---|---|---|']
    for c in out['verdict_changes']:
        L.append(f"| {c['setting']} | {LABEL[c['strategy']]} | {c['reference'][3:]} | {c['metric']} | {sym(c['cell'])} / {sym(c['cell_pct'])} | "
                 f"{sym(c['clustered'])} / {sym(c['clustered_pct'])} |")
    if not out['verdict_changes']:
        L.append('| — | none | | | | |')
    L += ['', '## Per-family mean-FCT contrast vs guarded score (ticks; contribution = flow-weighted share of the pooled Δ)', '']
    for s in SETTINGS:
        L += [f'### Setting {s}', '', '| comparator | family | flows | Δ ticks | Δ % | seeds Δ>0 | contribution (ticks) | share of pooled Δ | pooled Δ% without family |',
              '|---|---|---|---|---|---|---|---|---|']
        for v in PRIMARY:
            for f, x in out['per_family'][s][v].items():
                sh = f"{x['contribution_share'] * 100:.0f}%" if x['contribution_share'] is not None else '—'
                L.append(f"| {LABEL[v]} | {f} | {x['n_flows']} | {x['diff']:+.2f} | {x['pct']:+.1f}% | {x['seeds_positive']}/5 | "
                         f"{x['contribution']:+.2f} | {sh} | {x['loo_pct']:+.1f}% |")
        L.append('')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    analyse()
