"""Concern 7.3: raw results for the load-sweep extensions (unseen seeds 4000-4009, six families, 60 cells per load).

  knee    Table A7   ECMP and guarded score at loads 0.2 / 0.25 with the 16 host-link buffers at 2,000 / 4,000 /
                     8,000 packets (fabric buffers 2,000). The 2,000-packet runs must equal rev3_load.jsonl (gate).
  fbload  Table A13  FlowBender-style (10% marking, two consecutive ticks) open loop, instrumented, at every load.
  clload  Table A11  closed loop without reordering penalty: ECMP, guarded score, FlowBender-style at every load.

Flows are generated as in rev3_load.py (default graph, every rate multiplied by the load factor); FlowBender uses
rng seed 1000 * family index + traffic seed as in rev4_fb.py and c4_sweep.py.

  python revision/scripts/rev7_loadext.py run --groups knee,fbload,clload --workers 6
  python revision/scripts/rev7_loadext.py gate
  python revision/scripts/rev7_loadext.py analyze

Results: results/rev7_loadext.jsonl (resumable), results/rev7_loadext_summary.json, results/rev7_loadext_tables.md.
"""
from paths import NF_ROOT, RESULTS  # folder locations (see paths.py)
import argparse, copy, json, os, sys, time, traceback
import multiprocessing as mp
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(RESULTS, 'rev7_loadext.jsonl')
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
SEEDS = list(range(4000, 4010))
LOADS = [0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
KNEE, BUFFERS = [0.2, 0.25], [2000, 4000, 8000]
FB = dict(k_frac=0.1, n_consec=2)
OPEN_KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'unroutable_drops', 'total_offered',
             'total_delivered', 'avg_throughput', 'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes',
             'reordered_packets', 'reordered_fraction', 'flows_with_reordering', 'n_flows']
CL_KEEP = ['total_drops', 'reroutes', 'completed', 'incomplete', 'fct_mean', 'fct_p99', 'retx_packets',
           'offered_unique', 'spurious_packets', 'makespan', 'per_flow']
DRAWS, SEED = 2000, 42


def tasks(groups):
    T = []
    for fi, fam in enumerate(FAM):
        for s in SEEDS:
            rs = 1000 * fi + s
            if 'knee' in groups:
                for lf in KNEE:
                    for B in BUFFERS:
                        T.append(dict(id=f'knee|{lf}|{B}|ecmp|{fam}|{s}', group='knee', load=lf, buf=B, variant='ecmp',
                                      strategy='ecmp', family=fam, seed=s))
                        T.append(dict(id=f'knee|{lf}|{B}|guarded_score|{fam}|{s}', group='knee', load=lf, buf=B,
                                      variant='guarded_score', strategy='guarded', signal='score', threshold=0.851,
                                      family=fam, seed=s))
            for lf in LOADS:
                if 'fbload' in groups:
                    T.append(dict(id=f'fbload|{lf}|fb_default|{fam}|{s}', group='fbload', load=lf, variant='fb_default',
                                  strategy='flowbender', guard=FB, family=fam, seed=s, rs=rs))
                if 'clload' in groups:
                    T.append(dict(id=f'clload|{lf}|ecmp|{fam}|{s}', group='clload', load=lf, variant='ecmp',
                                  strategy='ecmp', family=fam, seed=s))
                    T.append(dict(id=f'clload|{lf}|guarded_score|{fam}|{s}', group='clload', load=lf,
                                  variant='guarded_score', strategy='guarded', signal='score', threshold=0.851,
                                  family=fam, seed=s))
                    T.append(dict(id=f'clload|{lf}|fb_default|{fam}|{s}', group='clload', load=lf, variant='fb_default',
                                  strategy='flowbender', guard=FB, family=fam, seed=s, rs=rs))
    return T


_W = {}


def graph(B=2000):
    from simulation.topology import build_fat_tree
    g = build_fat_tree()
    if B != 2000:
        n = 0
        for u, v, d in g.edges(data=True):
            if d.get('is_host_link'):
                d['buffer_size'] = B
                n += 1
        assert n == 16, n
    return g


def _init():
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    sys.path.insert(0, HERE)
    import nf_exp
    _W['X'], _W['g'], _W['flows'] = nf_exp, {}, {}


def _run(t):
    try:
        X = _W['X']
        from simulation.traffic_generator import TrafficGenerator
        B = t.get('buf', 2000)
        if B not in _W['g']:
            _W['g'][B] = graph(B)
        key = (t['family'], t['seed'])
        if key not in _W['flows']:   # traffic is generated on the default graph, as in rev3_load.py and rev6_hostbuf.py
            _W['flows'] = {key: TrafficGenerator(graph(2000), seed=t['seed']).generate_scenario(
                total_ticks=150, pattern=t['family'])}
        flows = copy.deepcopy(_W['flows'][key])
        for f in flows:
            f.rate *= t['load']
        guard = dict(t.get('guard', {}))
        if t['strategy'] == 'guarded':
            guard['signal'] = t['signal']
        kw = dict(threshold=t.get('threshold'), guard=guard, rng_seed=t.get('rs', 0))
        s0 = time.perf_counter()
        if t['group'] == 'clload':
            r = X.run_case_x(_W['g'][B], flows, 150, t['strategy'], closed_loop=True, reorder_penalty=False, **kw)
            keep = CL_KEEP
        else:
            r = X.run_case_x(_W['g'][B], flows, 150, t['strategy'], instrument=True, **kw)
            keep = OPEN_KEEP
        assert r['packet_conservation'] and r['drop_split_ok'], t['id']
        out = {k: v for k, v in t.items() if k != 'guard'}
        out['wall_s'] = time.perf_counter() - s0
        out.update({k: r[k] for k in keep if k in r})
        return out
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())


def load_rows():
    R = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if 'error' not in r:
                R[r['id']] = r
    return R


def cmd_run(a):
    done = set(load_rows())
    T = [t for t in tasks(a.groups.split(',')) if t['id'] not in done]
    if a.only:
        f, s = a.only.split(':')
        T = [t for t in T if t['family'] == f and t['seed'] == int(s)]
    T.sort(key=lambda t: (t['family'], t['seed']))
    print(f'{len(T)} runs to do ({len(done)} done)', flush=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers, initializer=_init) as pool, open(OUT, 'a') as fh:
        for i, r in enumerate(pool.imap(_run, T, chunksize=2)):
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            if 'error' in r:
                print('ERROR', r['id'], r['error'][-800:], flush=True)
            if (i + 1) % 50 == 0 or i + 1 == len(T):
                el = time.time() - t0
                print(f'  {i + 1}/{len(T)} {el / 60:.1f} min, eta {el / (i + 1) * (len(T) - i - 1) / 60:.1f} min', flush=True)


def rev3():
    R = {}
    for line in open(os.path.join(RESULTS, 'rev3_load.jsonl')):
        r = json.loads(line)
        if 'error' not in r and r['variant'] in ('ecmp', 'guarded_score'):
            R[(r['load'], r['variant'], r['family'], r['seed'])] = r
    return R


def cmd_gate(a):
    """The 2,000-packet knee runs must equal the published load sweep (rev3_load.jsonl) field by field."""
    R, ref = load_rows(), rev3()
    fields = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'total_delivered', 'reroutes']
    n = bad = 0
    for r in R.values():
        if r['group'] != 'knee' or r['buf'] != 2000:
            continue
        p = ref[(r['load'], r['variant'], r['family'], r['seed'])]
        n += 1
        diff = {f: (r[f], p[f]) for f in fields if r[f] != p[f]}
        if diff:
            bad += 1
            if bad <= 10:
                print('MISMATCH', r['id'], diff)
    print(f'gate: {n} runs compared with rev3_load.jsonl, {bad} mismatches')
    print('GATE PASSED' if n and not bad else 'GATE FAILED')


# ── statistics: open loop as rev6_hostbuf.boot / pts; closed loop as c4_clustered (cells, stats, draws) ──
def boot(d, cluster=False):
    rng = np.random.default_rng(SEED)
    if not cluster:
        v = np.array(list(d.values()))
        return v[rng.integers(0, len(v), size=(DRAWS, len(v)))].mean(1)
    fams = sorted({f for f, _ in d})
    byf = {f: np.array([d[(f, s)] for (ff, s) in d if ff == f]) for f in fams}
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.integers(0, len(fams), size=len(fams))
        out[i] = np.mean([byf[fams[j]][rng.integers(0, len(byf[fams[j]]), size=len(byf[fams[j]]))].mean() for j in pick])
    return out


def pts(A, B, E, cells, cluster=False):
    """Paired difference A - B per cell, in % of the mean of E (points of mean ECMP total drops)."""
    base = float(np.mean(E))
    if base == 0:
        return float('nan'), [float('nan'), float('nan')]
    d = {c: a - b for c, a, b in zip(cells, A, B)}
    lo, hi = np.percentile(boot(d, cluster), [2.5, 97.5]) / base * 100
    return float(np.mean(list(d.values())) / base * 100), [float(lo), float(hi)]


_starts = {}


def start(f, s, fid):
    if (f, s) not in _starts:
        sys.path.insert(0, os.environ.get('NF_ROOT', NF_ROOT))
        from simulation.traffic_generator import TrafficGenerator
        _starts[(f, s)] = {x.flow_id: x.start_tick for x in
                           TrafficGenerator(graph(2000), seed=s).generate_scenario(total_ticks=150, pattern=f)}
    return _starts[(f, s)][fid]


def fct_cells(rows):
    C = defaultdict(dict)
    for r in rows:
        fl = [(x[4] if x[4] is not None else r['makespan'] - start(r['family'], r['seed'], x[0])) for x in r['per_flow'] if x[2] > 0]
        C[r['variant']][(r['family'], r['seed'])] = np.array(fl)
    return C


def fct_stats(d, keys):
    x = np.concatenate([d[k] for k in keys])
    return dict(mean=float(x.mean()), p99=float(np.percentile(x, 99)))


def fct_draws(keys, cluster):
    rng = np.random.default_rng(SEED)
    if not cluster:
        return [[keys[j] for j in rng.integers(0, len(keys), len(keys))] for _ in range(DRAWS)]
    fams = sorted({f for f, _ in keys})
    byf = {f: [k for k in keys if k[0] == f] for f in fams}
    out = []
    for _ in range(DRAWS):
        ks = []
        for j in rng.integers(0, len(fams), size=len(fams)):
            m = byf[fams[j]]
            ks += [m[i] for i in rng.integers(0, len(m), size=len(m))]
        out.append(ks)
    return out


def fct_rel(C, a, b, keys, m, cluster=False):
    """(FCT_a / FCT_b - 1) * 100 for the pooled mean or p99, with a 95% bootstrap CI."""
    p = (fct_stats(C[a], keys)[m] / fct_stats(C[b], keys)[m] - 1) * 100
    d = [(fct_stats(C[a], ks)[m] / fct_stats(C[b], ks)[m] - 1) * 100 for ks in fct_draws(keys, cluster)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(p), [float(lo), float(hi)]


def cmd_analyze(a):
    R, ref = load_rows(), rev3()
    cells = sorted((f, s) for f in FAM for s in SEEDS)   # pair order as rev5_stats.compare / rev4_c4_analyze.compare
    S, L = dict(A7={}, A13={}, A11={}), []
    pct = lambda x, e: 100 * (np.mean(x) - np.mean(e)) / np.mean(e) if np.mean(e) else float('nan')
    f2 = lambda c: f'[{c[0]:+.2f}, {c[1]:+.2f}]'

    # Table A7: host-link buffers at the knee
    L += ['## Table A7: host-link buffers at the knee (ECMP vs guarded score, seeds 4000-4009, 60 runs per row)', '',
          '| Load | Host buffer | ECMP host drops/run | Fabric / host (%) | Total Δ (%) | 95% CI paired | 95% CI clustered | Route chg. |',
          '|---|---|---|---|---|---|---|---|']
    for lf in KNEE:
        for B in BUFFERS:
            g = lambda v, k: np.array([R[f'knee|{lf}|{B}|{v}|{f}|{s}'][k] for f, s in cells], dtype=float)
            if any(f'knee|{lf}|{B}|ecmp|{f}|{s}' not in R or f'knee|{lf}|{B}|guarded_score|{f}|{s}' not in R for f, s in cells):
                continue
            E, G = g('ecmp', 'total_drops'), g('guarded_score', 'total_drops')
            d, ci = pts(G, E, E, cells)
            _, cic = pts(G, E, E, cells, True)
            o = dict(ecmp_host=float(g('ecmp', 'host_link_drops').mean()),
                     fabric_pct=pct(g('guarded_score', 'fabric_link_drops'), g('ecmp', 'fabric_link_drops')),
                     host_pct=pct(g('guarded_score', 'host_link_drops'), g('ecmp', 'host_link_drops')),
                     total_pct=d, ci=ci, ci_clustered=cic, reroutes=float(g('guarded_score', 'reroutes').mean()))
            S['A7'][f'{lf}|{B}'] = o
            L.append(f"| {lf} | {B:,} | {o['ecmp_host']:,.0f} | {o['fabric_pct']:+.1f} / {o['host_pct']:+.1f} | "
                     f"{d:+.2f} | {f2(ci)} | {f2(cic)} | {o['reroutes']:.1f} |")

    # Table A13: FlowBender-style across loads (open loop); ECMP and guarded score from rev3_load.jsonl
    L += ['', '## Table A13: FlowBender-style across loads (open loop, seeds 4000-4009, 60 runs per row)', '',
          '| Load | Total Δ (%) [95% CI] | Clustered CI | Fabric / host (%) | FlowB. − guard [95% CI] | Clustered CI | Route chg. / reord. (%) |',
          '|---|---|---|---|---|---|---|']
    for lf in LOADS:
        if any(f'fbload|{lf}|fb_default|{f}|{s}' not in R for f, s in cells):
            continue
        F = lambda k: np.array([R[f'fbload|{lf}|fb_default|{f}|{s}'][k] for f, s in cells], dtype=float)
        P = lambda v, k: np.array([ref[(lf, v, f, s)][k] for f, s in cells], dtype=float)
        E, G, Fb = P('ecmp', 'total_drops'), P('guarded_score', 'total_drops'), F('total_drops')
        d, ci = pts(Fb, E, E, cells)
        _, cic = pts(Fb, E, E, cells, True)
        dg, cig = pts(Fb, G, E, cells)
        _, cigc = pts(Fb, G, E, cells, True)
        o = dict(total_pct=d, ci=ci, ci_clustered=cic,
                 fabric_pct=pct(F('fabric_link_drops'), P('ecmp', 'fabric_link_drops')),
                 host_pct=pct(F('host_link_drops'), P('ecmp', 'host_link_drops')),
                 minus_guard=dg, minus_guard_ci=cig, minus_guard_ci_clustered=cigc,
                 reroutes=float(F('reroutes').mean()),   # pooled share of delivered packets, as c3_analyze.py
                 reordered_pct=float(100 * F('reordered_packets').sum() / F('total_delivered').sum()))
        S['A13'][str(lf)] = o
        L.append(f"| {lf} | {d:+.2f} {f2(ci)} | {f2(cic)} | {o['fabric_pct']:+.1f} / {o['host_pct']:+.1f} | "
                 f"{dg:+.2f} {f2(cig)} | {f2(cigc)} | {o['reroutes']:,.0f} / {o['reordered_pct']:.2f} |")

    # Table A11: closed-loop FCT across loads (no reordering penalty, unfinished flows censored at 600 ticks)
    L += ['', '## Table A11: closed-loop FCT across loads (no reordering penalty, seeds 4000-4009, 60 cells per load)', '',
          '| Load | ECMP mean / p99 | Guard vs ECMP, mean | Clustered CI | Guard vs ECMP, p99 | FlowB. vs guard, mean | Clustered CI |',
          '|---|---|---|---|---|---|---|']
    for lf in LOADS:
        rows = [R.get(f'clload|{lf}|{v}|{f}|{s}') for v in ('ecmp', 'guarded_score', 'fb_default') for f, s in cells]
        if any(r is None for r in rows):
            continue
        C = fct_cells(rows)
        e = fct_stats(C['ecmp'], cells)
        gm, gmc = fct_rel(C, 'guarded_score', 'ecmp', cells, 'mean'), fct_rel(C, 'guarded_score', 'ecmp', cells, 'mean', True)
        gp = fct_rel(C, 'guarded_score', 'ecmp', cells, 'p99')
        fm, fmc = fct_rel(C, 'fb_default', 'guarded_score', cells, 'mean'), fct_rel(C, 'fb_default', 'guarded_score', cells, 'mean', True)
        S['A11'][str(lf)] = dict(ecmp_mean=e['mean'], ecmp_p99=e['p99'], guard_mean=gm, guard_mean_clustered=gmc,
                                 guard_p99=gp, fb_mean=fm, fb_mean_clustered=fmc)
        L.append(f"| {lf} | {e['mean']:.1f} / {e['p99']:.0f} | {gm[0]:+.2f} {f2(gm[1])} | {f2(gmc[1])} | "
                 f"{gp[0]:+.2f} {f2(gp[1])} | {fm[0]:+.2f} {f2(fm[1])} | {f2(fmc[1])} |")

    json.dump(S, open(os.path.join(RESULTS, 'rev7_loadext_summary.json'), 'w'), indent=1)
    open(os.path.join(RESULTS, 'rev7_loadext_tables.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'gate', 'analyze'])
    ap.add_argument('--groups', default='knee,fbload,clload')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--only', default=None, help='family:seed, for a quick test (e.g. uniform:4000)')
    a = ap.parse_args()
    {'run': cmd_run, 'gate': cmd_gate, 'analyze': cmd_analyze}[a.cmd](a)
