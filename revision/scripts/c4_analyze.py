"""Concern 4 (4th review) analysis of results/c4/runs.jsonl (protocol: results/c4/PROTOCOL.md).

  python revision/scripts/c4_analyze.py gate      G1-G4 -> results/c4/gate.json
  python revision/scripts/c4_analyze.py summary   per-setting tables, contrasts, crossover -> results/c4/summary.json, tables.md

The FCT cells, pooled statistics and the paired cell bootstrap are the functions of rev4_c4_analyze.py (loaded, not
re-implemented; that script itself is not run, so its summary file is never rewritten)."""
from paths import NF_ROOT, RESULTS  # folder locations (see paths.py)
import json, math, os, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C4 = os.path.join(RESULTS, 'c4')
RUNS = os.path.join(C4, 'runs.jsonl')
_ns = {'__file__': os.path.join(HERE, 'rev4_c4_analyze.py')}
exec(open(os.path.join(HERE, 'rev4_c4_analyze.py')).read().split('S = {}')[0], _ns)
cells, stats, compare = _ns['cells'], _ns['stats'], _ns['compare']

VARIANTS = ['ecmp', 'reactive', 'unguarded_gnn', 'score_no_rules', 'fb_default', 'guarded_gnn', 'guarded_persistence',
            'guarded_score']
TABLE_A6 = [v for v in VARIANTS if v != 'fb_default']
AXIS = [('W0', 0), ('W1', 1), ('W2', 2), ('W4', 4), ('W8', 8), ('W16', 16), ('off', math.inf)]
EXTRA = ['rtt4', 'dup3']
SETTINGS = [s for s, _ in AXIS] + EXTRA
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
CELLS = [(f, s) for f in FAM for s in range(3000, 3005)]
DRAWS, SEED = 2000, 42
META = {'id', 'group', 'variant', 'strategy', 'signal', 'mseed', 'family', 'seed', 'guard', 'threshold_used', 'wall_s',
        'sec', 'closed_loop', 'mode', 'penalty',
        'inference_ms_p50', 'inference_ms_p99'}   # wall-clock timing of the forecaster: machine-dependent, not an outcome
load = lambda path: [json.loads(l) for l in open(path)]
saved = lambda name: load(os.path.join(RESULTS, name))
rt = lambda x: json.loads(json.dumps(x, default=float))


def runs():
    R = {}
    for r in load(RUNS):
        if 'error' not in r:
            R[r['id']] = r
    return R


# ─────────────────────────────── gates ─────────────────────────────────────
def row_gate(R, pairs):
    """pairs: (saved row, new id). Every saved metric field must equal the new row's value."""
    bad, n_fields = [], 0
    for s, nid in pairs:
        r = R.get(nid)
        if r is None:
            bad.append(dict(id=nid, missing=True))
            continue
        diff = [k for k in s if k not in META and (k not in r or r[k] != s[k])]
        n_fields += sum(1 for k in s if k not in META)
        if diff:
            bad.append(dict(id=nid, fields=diff))
    return dict(ok=not bad, rows=len(pairs), fields_compared=n_fields, mismatches=bad[:20], n_mismatches=len(bad))


def a6_block(rows):
    """rev4_c4_analyze.py lines 41-53, verbatim logic, on the given rows."""
    C = cells(rows)
    out = {}
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
        out[v] = {k: (float(x) if isinstance(x, (np.floating, float)) else x) for k, x in o.items()}
    return out


def fb_block(rows):
    """rev4_fb_analyze.py lines 30-35, verbatim logic."""
    C = cells(rows); keys = sorted(C['fb_default']); st = stats(C['fb_default'], keys)
    return dict(fct_mean=float(st['mean']), fct_p99=float(st['p99']),
                vs_ecmp=compare(C, 'fb_default', 'ecmp'), vs_guarded_score=compare(C, 'fb_default', 'guarded_score'),
                spur_pct=float(sum(C['fb_default'][k]['spur'] for k in keys) / sum(C['fb_default'][k]['off'] for k in keys) * 100),
                reroutes=float(np.mean([C['fb_default'][k]['rr'] for k in keys])))


def setting_rows(R, name, variants=VARIANTS):
    return [R[f'{name}|{v}|{f}|{s}'] for v in variants for f, s in CELLS if f'{name}|{v}|{f}|{s}' in R]


def gate():
    R = runs()
    G = {}
    cl = [r for r in saved('closedloop.jsonl') if r['variant'] != 'guarded_lstm']
    ng = [r for r in saved('rev2_noguard.jsonl') if r['variant'] == 'score_no_rules_cl']
    fb = [r for r in saved('rev4_fb.jsonl') if r['mode'] == 'closed']
    c4 = saved('rev4_c4.jsonl')
    nid = lambda st, v, r: f"{st}|{v}|{r['family']}|{r['seed']}"
    G['G1_no_penalty_rows'] = row_gate(R, [(r, nid('off', r['variant'], r)) for r in cl] +
                                       [(r, nid('off', 'score_no_rules', r)) for r in ng] +
                                       [(r, nid('off', 'fb_default', r)) for r in fb if r['penalty'] is False])
    G['G2_full_penalty_rows'] = row_gate(R, [(r, nid('W0', r['variant'], r)) for r in c4] +
                                         [(r, nid('W0', 'fb_default', r)) for r in fb if r['penalty'] is True])
    # G3: Table A6 statistics from the new rows
    S4 = json.load(open(os.path.join(RESULTS, 'rev4_c4_summary.json')))
    SF = json.load(open(os.path.join(RESULTS, 'rev4_fb_summary.json')))['closed']
    g3 = {}
    for name, setting in (('no_penalty', 'off'), ('penalty', 'W0')):
        got = rt(a6_block(setting_rows(R, setting, TABLE_A6)))
        g3[f'rev4_c4_summary.{name}'] = dict(ok=got == S4[name], n_variants=len(got),
                                              differing=[v for v in S4[name] if got.get(v) != S4[name][v]])
        got_fb = rt(fb_block(setting_rows(R, setting, ['ecmp', 'guarded_score', 'fb_default'])))
        g3[f'rev4_fb_summary.closed.{name}'] = dict(ok=got_fb == SF[name],
                                                     differing=[k for k in SF[name] if got_fb.get(k) != SF[name][k]])
    G['G3_statistics'] = dict(ok=all(x['ok'] for x in g3.values()), checks=g3)
    # G4: completeness, conservation, no errors (for every setting present)
    present = sorted({r['setting'] for r in R.values()}, key=SETTINGS.index)
    errors = [r['id'] for r in load(RUNS) if 'error' in r and r['id'] not in R]
    g4 = {}
    for s in present:
        rows = setting_rows(R, s)
        g4[s] = dict(runs=len(rows), expected=len(VARIANTS) * len(CELLS),
                     conservation=sum(bool(r['packet_conservation']) for r in rows),
                     drop_split=sum(bool(r['drop_split_ok']) for r in rows))
        g4[s]['ok'] = g4[s]['runs'] == g4[s]['expected'] == g4[s]['conservation'] == g4[s]['drop_split']
    G['G4_conservation'] = dict(ok=all(x['ok'] for x in g4.values()) and not errors, settings=g4, unresolved_errors=errors)
    G['ok'] = all(G[k]['ok'] for k in G)
    json.dump(G, open(os.path.join(C4, 'gate.json'), 'w'), indent=1)
    for k, v in G.items():
        print(k, v if k == 'ok' else v['ok'])
    return G


# ─────────────────────────────── summary ───────────────────────────────────
def boot_draws(n):
    rng = np.random.default_rng(SEED)            # identical draw sequence to rev4_c4_analyze.compare
    return [rng.integers(0, n, n) for _ in range(DRAWS)]


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def verdict(c):
    return '+' if c[0] > 0 else ('-' if c[1] < 0 else '0')


def contrast(P, D, a, b):
    """FCT(a) - FCT(b) (ticks) and (FCT(a)/FCT(b) - 1) * 100, point + 95% CI, same paired draws, mean and p99."""
    out = {}
    for m in ('mean', 'p99'):
        d, p = D[a][m] - D[b][m], (D[a][m] / D[b][m] - 1) * 100
        cd, cp = ci(d), ci(p)
        out[m] = dict(diff=P[a][m] - P[b][m], diff_ci=cd, pct=(P[a][m] / P[b][m] - 1) * 100, pct_ci=cp,
                      pct_ci_r2=np.round(np.percentile(p, [2.5, 97.5]), 2).tolist(),
                      verdict=verdict(cd), verdict_pct=verdict(cp))
        out[m]['basis_disagree'] = out[m]['verdict'] != out[m]['verdict_pct']
    return out


def hist_sum(rows, key):
    h = Counter()
    for r in rows:
        for k, v in r['reorder_diag'][key].items():
            h[k] += v
    return h


def distributions(rows):
    size = {int(k): v for k, v in hist_sum(rows, 'size_hist').items()}
    disp = {int(k): v for k, v in hist_sum(rows, 'disp_hist').items()}
    dup, dupk = hist_sum(rows, 'dupack_hist'), hist_sum(rows, 'dupack_pk')
    rtt = {int(k): v for k, v in hist_sum(rows, 'rtt_hist').items()}
    nb = sum(size.values()); npk = sum(k * v for k, v in size.items())
    if not nb:
        return dict(reordered_batches=0)
    q = np.repeat(np.array(sorted(size)), [size[k] for k in sorted(size)])
    return dict(
        reordered_batches=nb, reordered_packets=npk,
        size_share_batches={'1': size.get(1, 0) / nb, '2': size.get(2, 0) / nb, '>=3': sum(v for k, v in size.items() if k >= 3) / nb},
        size_share_packets={'1': size.get(1, 0) / npk, '2': 2 * size.get(2, 0) / npk,
                            '>=3': sum(k * v for k, v in size.items() if k >= 3) / npk},
        size_quantiles={p: float(np.percentile(q, p)) for p in (10, 25, 50, 75, 90, 99)},
        size_mean=float(q.mean()),
        dupack_below3_share_batches=sum(dup.get(k, 0) for k in ('0', '1', '2')) / nb,
        dupack_below3_share_packets=sum(dupk.get(k, 0) for k in ('0', '1', '2')) / npk,
        dupack_hist=dict(dup),
        disp_le_W_share_batches={W: sum(v for d, v in disp.items() if d <= W) / nb for W in (1, 2, 4, 8, 16)},
        disp_quantiles={p: float(np.percentile(np.repeat(np.array(sorted(disp)), [disp[k] for k in sorted(disp)]), p))
                        for p in (50, 90, 99)},
        rtt_hist=rtt, rtt4_window_hist={f'{k / 4:g}': v for k, v in sorted(rtt.items())},
        rtt_all_multiple_of_4=all(k % 4 == 0 for k in rtt))


def summary():
    R = runs()
    out = dict(settings={}, crossover={}, meta=dict(draws=DRAWS, seed=SEED, unit='(family, traffic seed) cell, 30 cells',
                                                   axis=[[s, w if w != math.inf else 'inf'] for s, w in AXIS]))
    D_idx = boot_draws(len(CELLS))
    selfcheck = []
    for s in SETTINGS:
        rows = setting_rows(R, s)
        if len(rows) != len(VARIANTS) * len(CELLS):
            print('skip', s, len(rows)); continue
        C = cells(rows)
        keys = sorted(C['ecmp'])
        assert all(sorted(C[v]) == keys for v in VARIANTS)
        P, D = {}, {}
        for v in VARIANTS:
            st = stats(C[v], keys)
            P[v] = dict(mean=float(st['mean']), p99=float(st['p99']))
            dm, dp = np.empty(DRAWS), np.empty(DRAWS)
            for i, ix in enumerate(D_idx):
                x = stats(C[v], [keys[j] for j in ix])
                dm[i], dp[i] = x['mean'], x['p99']
            D[v] = dict(mean=dm, p99=dp)
        S = {}
        for v in VARIANTS:
            vr = [r for r in rows if r['variant'] == v]
            rd = lambda k: sum(r['reorder_diag'][k] for r in vr)
            off = sum(C[v][k]['off'] for k in keys)
            o = dict(fct_mean=P[v]['mean'], fct_p99=P[v]['p99'],
                     spur_pct=sum(C[v][k]['spur'] for k in keys) / off * 100,
                     retx_pct=sum(C[v][k]['retx'] for k in keys) / off * 100,
                     reroutes=float(np.mean([C[v][k]['rr'] for k in keys])),
                     incomplete=sum(r['incomplete'] for r in vr), completed=sum(r['completed'] for r in vr),
                     reordered_batches=rd('reordered_batches'), reordered_batches_open=rd('reordered_batches_open'),
                     reordered_packets=sum(r['cl_reordered_packets'] for r in vr),
                     reordered_packets_open=rd('reordered_packets_open'),
                     penalised_batches=rd('penalised_batches'), penalised_packets=rd('penalised_packets'),
                     wall_s=sum(r['wall_s'] for r in vr))
            o['penalised_share'] = dict(
                batches=o['penalised_batches'] / o['reordered_batches'] if o['reordered_batches'] else None,
                batches_open=o['penalised_batches'] / o['reordered_batches_open'] if o['reordered_batches_open'] else None,
                packets=o['penalised_packets'] / o['reordered_packets'] if o['reordered_packets'] else None,
                packets_open=o['penalised_packets'] / o['reordered_packets_open'] if o['reordered_packets_open'] else None)
            if v != 'ecmp':
                o['vs_ecmp'] = contrast(P, D, v, 'ecmp')
            if v != 'guarded_score':
                o['vs_guarded_score'] = contrast(P, D, v, 'guarded_score')
            o['distribution'] = distributions(vr)
            S[v] = o
        # self-check: the shared draws give exactly rev4_c4_analyze.compare's percentages and rounded CIs
        for a, b in (('score_no_rules', 'guarded_score'), ('fb_default', 'ecmp')):
            ref = compare(C, a, b)
            mine = S[a]['vs_guarded_score' if b == 'guarded_score' else 'vs_ecmp']
            selfcheck.append(dict(setting=s, pair=f'{a} vs {b}',
                                  ok=ref['mean_pct'] == mine['mean']['pct'] and ref['mean_ci'] == mine['mean']['pct_ci_r2']
                                  and ref['p99_pct'] == mine['p99']['pct'] and ref['p99_ci'] == mine['p99']['pct_ci_r2']))
        tot = dict(distribution=distributions(rows))
        all_pen = sum(S[v]['penalised_batches'] for v in VARIANTS)
        tot['penalised_share'] = dict(
            batches=all_pen / max(1, sum(S[v]['reordered_batches'] for v in VARIANTS)),
            batches_open=all_pen / max(1, sum(S[v]['reordered_batches_open'] for v in VARIANTS)),
            packets=sum(S[v]['penalised_packets'] for v in VARIANTS) / max(1, sum(S[v]['reordered_packets'] for v in VARIANTS)),
            packets_open=sum(S[v]['penalised_packets'] for v in VARIANTS) / max(1, sum(S[v]['reordered_packets_open'] for v in VARIANTS)))
        out['settings'][s] = dict(variants=S, pooled=tot)
        print('done', s, flush=True)
    out['selfcheck_compare'] = dict(ok=all(x['ok'] for x in selfcheck), checks=selfcheck)
    # crossover along W = 0, 1, 2, 4, 8, 16, inf (off)
    have = [(s, w) for s, w in AXIS if s in out['settings']]
    for v in VARIANTS:
        if v == 'guarded_score':
            continue
        out['crossover'][v] = {}
        for m in ('mean', 'p99'):
            ver = [(s, w, out['settings'][s]['variants'][v]['vs_guarded_score'][m]['verdict']) for s, w in have]
            det = [w for s, w, x in ver if x == '+']
            nd = [w for s, w, x in ver if x != '+']
            a = max(det) if det else None
            b = min(nd) if nd else None
            seq = [x == '+' for _, _, x in ver]
            monotone = all(seq[i] or not seq[i + 1] for i in range(len(seq) - 1))   # detected prefix, then never again
            ws = [w for _, w in have]
            adjacent = a is not None and b is not None and ws.index(b) == ws.index(a) + 1
            fmt = lambda w: None if w is None else ('inf' if w == math.inf else w)
            out['crossover'][v][m] = dict(
                verdicts={s: x for s, _, x in ver},
                extra={s: out['settings'][s]['variants'][v]['vs_guarded_score'][m]['verdict'] for s in EXTRA if s in out['settings']},
                largest_W_detected=fmt(a), smallest_W_not_detected=fmt(b), monotone=monotone,
                bracket=[fmt(a), fmt(b)] if monotone and adjacent else None,
                comparator_better_at=[s for s, _, x in ver if x == '-'] +
                                     [s for s in EXTRA if s in out['settings'] and
                                      out['settings'][s]['variants'][v]['vs_guarded_score'][m]['verdict'] == '-'],
                basis_disagree=[s for s in [x for x, _ in have] + [x for x in EXTRA if x in out['settings']]
                                if out['settings'][s]['variants'][v]['vs_guarded_score'][m]['basis_disagree']])
    json.dump(out, open(os.path.join(C4, 'summary.json'), 'w'), indent=1, default=float)
    open(os.path.join(C4, 'tables.md'), 'w', encoding='utf-8').write(tables(out))
    print('selfcheck', out['selfcheck_compare']['ok'])
    return out


def tables(out):
    f1 = lambda x: f'{x:.1f}'
    cis = lambda c, d=1: f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'
    L = ['# Concern 4 — closed-loop FCT under reordering models (generated by c4_analyze.py)', '',
         'FCT in ticks; unfinished flows censored at the 600-tick horizon. Contrasts: Δ = FCT(row) − FCT(reference) in ticks '
         'and in % of the reference, 95% paired cell-bootstrap CI (2,000 draws, seed 42). Verdict on the tick CI: '
         '+ = entire CI > 0 (vs guarded score: guard advantage detected), − = entire CI < 0, 0 = CI includes 0.', '']
    for s, blk in out['settings'].items():
        L += [f'## Setting {s}', '',
              '| strategy | mean FCT | p99 FCT | spurious % | vs ECMP mean Δ [CI] (%) | vs ECMP p99 Δ [CI] (%) | '
              'vs guarded score mean Δ [CI] (%) v | vs guarded score p99 Δ [CI] (%) v |', '|---|---|---|---|---|---|---|---|']
        for v, o in blk['variants'].items():
            e = o.get('vs_ecmp'); g = o.get('vs_guarded_score')
            ce = lambda m: f"{e[m]['diff']:+.1f} {cis(e[m]['diff_ci'])} ({e[m]['pct']:+.1f}%)" if e else '—'
            cg = lambda m: f"{g[m]['diff']:+.1f} {cis(g[m]['diff_ci'])} ({g[m]['pct']:+.1f}%) {g[m]['verdict']}" if g else '—'
            L.append(f"| {v} | {f1(o['fct_mean'])} | {f1(o['fct_p99'])} | {o['spur_pct']:.2f} | {ce('mean')} | {ce('p99')} | "
                     f"{cg('mean')} | {cg('p99')} |")
        L.append('')
    L += ['## Penalised share of reordered batches / packets (all strategies pooled; open = flow not yet complete)', '',
          '| setting | reordered batches | penalised / all | penalised / open | reordered packets | penalised / all | penalised / open |',
          '|---|---|---|---|---|---|---|']
    for s, blk in out['settings'].items():
        p, d = blk['pooled']['penalised_share'], blk['pooled']['distribution']
        L.append(f"| {s} | {d.get('reordered_batches', 0):,} | {p['batches']:.3f} | {p['batches_open']:.3f} | "
                 f"{d.get('reordered_packets', 0):,} | {p['packets']:.3f} | {p['packets_open']:.3f} |")
    L += ['', '## Reordered-batch distributions (all strategies pooled)', '',
          '| setting | size p10/p50/p90/p99 (pkts) | share size 1 / 2 / ≥3 (batches) | dupACK < 3 share (batches / pkts) | '
          'd ≤ 1/2/4/8/16 share | RTT/4 windows (ticks: batches) |', '|---|---|---|---|---|---|']
    for s, blk in out['settings'].items():
        d = blk['pooled']['distribution']
        if not d.get('reordered_batches'):
            continue
        q, sh, dl = d['size_quantiles'], d['size_share_batches'], d['disp_le_W_share_batches']
        L.append(f"| {s} | {q[10]:.0f}/{q[50]:.0f}/{q[90]:.0f}/{q[99]:.0f} | {sh['1']:.3f} / {sh['2']:.3f} / {sh['>=3']:.3f} | "
                 f"{d['dupack_below3_share_batches']:.3f} / {d['dupack_below3_share_packets']:.3f} | "
                 + ' / '.join(f'{dl[W]:.2f}' for W in (1, 2, 4, 8, 16)) + ' | '
                 + ', '.join(f'{k}: {v}' for k, v in d['rtt4_window_hist'].items()) + ' |')
    L += ['', '## Crossover (contrast FCT(comparator) − FCT(guarded score); W axis 0, 1, 2, 4, 8, 16, ∞ = off)', '',
          '| comparator | metric | verdicts W0 W1 W2 W4 W8 W16 off | rtt4 | dup3 | largest W detected | smallest W not detected | monotone | bracket | comparator better at |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for v, mm in out['crossover'].items():
        for m, c in mm.items():
            L.append(f"| {v} | {m} | {' '.join(c['verdicts'].values())} | {c['extra'].get('rtt4', '')} | {c['extra'].get('dup3', '')} | "
                     f"{c['largest_W_detected']} | {c['smallest_W_not_detected']} | {c['monotone']} | {c['bracket']} | "
                     f"{', '.join(c['comparator_better_at']) or '—'} |")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    {'gate': gate, 'summary': summary}[sys.argv[1]]()
