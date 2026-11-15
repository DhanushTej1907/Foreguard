"""Concern 7.1: rerun the Table V benchmark with deeper host-link buffers.

Every Table V strategy on the benchmark seeds 3000-3009 (six families, 60 cells), with the host-link
buffers (both directions of the 16 host-edge links) set to 2,000, 4,000 and 8,000 packets. Fabric
buffers stay at 2,000. Model-based strategies use the three training seeds, as in Table V.

  python3 revision/scripts/rev6_hostbuf.py run --workers 2          # about 30-60 min on 2 cores
  python3 revision/scripts/rev6_hostbuf.py gate                     # 2,000-packet runs must equal Table V
  python3 revision/scripts/rev6_hostbuf.py analyze                  # tables + paired / clustered CIs

Results: results/rev6_hostbuf.jsonl (resumable), results/rev6_hostbuf_summary.json,
results/rev6_hostbuf_tables.md.
"""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import argparse, json, os, sys, time, traceback
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(RESULTS, 'rev6_hostbuf.jsonl')
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
BENCH = list(range(3000, 3010))
TH_SCORE, TH_PERS = 0.851, 0.5
NONE = dict(headroom=None, max_changes=None, stretch=None, min_improvement=0.0, cooldown=0)
BUFFERS = [2000, 4000, 8000]
KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'unroutable_drops', 'total_offered',
        'total_delivered', 'avg_throughput', 'delivered_mean_latency_us', 'delivered_p99_latency_us',
        'reroutes', 'ticks_run', 'reordered_packets', 'reordered_fraction', 'flows_with_reordering', 'n_flows']
# (variant, strategy, signal, model seeds, guard, threshold)
STRATS = [('ecmp', 'ecmp', None, [None], {}, None),
          ('reactive', 'reactive', None, [None], {}, None),
          ('unguarded_gnn', 'unguarded', 'gnn', [0, 1, 2], {}, None),
          ('score_no_rules', 'guarded', 'score', [None], NONE, TH_SCORE),
          ('flowbender', 'flowbender', None, [None], dict(k_frac=0.1, n_consec=2), None),
          ('flowbender_5pct_1tick', 'flowbender', None, [None], dict(k_frac=0.05, n_consec=1), None),
          ('guarded_gnn', 'guarded', 'gnn', [0, 1, 2], {}, None),
          ('guarded_lstm', 'guarded', 'lstm_only', [0, 1, 2], {}, None),
          ('guarded_persistence', 'guarded', 'persistence', [None], {}, TH_PERS),
          ('guarded_score', 'guarded', 'score', [None], {}, TH_SCORE)]


def tasks(buffers):
    T = []
    for B in buffers:
        for fi, fam in enumerate(FAM):
            for s in BENCH:
                for v, st, sig, ms, g, th in STRATS:
                    for m in ms:
                        T.append(dict(id=f'hb|{B}|{v}|{m}|{fam}|{s}', buf=B, variant=v, strategy=st, signal=sig,
                                      mseed=m, family=fam, seed=s, guard=g, threshold=th,
                                      rs=(1000 * fi + s) if st == 'flowbender' else 0))
    return T


_W = {}


def host_buffer_graph(B):
    from simulation.topology import build_fat_tree
    g = build_fat_tree()
    n = 0
    for u, v, d in g.edges(data=True):
        if d.get('is_host_link'):
            d['buffer_size'] = B
            n += 1
    assert n == 16, n
    return g


def _init(threads):
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    sys.path.insert(0, HERE)
    import nf_exp
    _W['X'] = nf_exp
    _W['g'] = {}
    _W['flows'] = {}
    _W['threads'] = threads


def _run(t):
    try:
        X = _W['X']
        from simulation.traffic_generator import TrafficGenerator
        if t['buf'] not in _W['g']:
            _W['g'][t['buf']] = host_buffer_graph(t['buf'])
        g = _W['g'][t['buf']]
        key = (t['family'], t['seed'])
        if key not in _W['flows']:   # traffic is generated on the default graph, identical for every buffer
            _W['flows'] = {key: TrafficGenerator(host_buffer_graph(2000), seed=t['seed']).generate_scenario(
                total_ticks=150, pattern=t['family'])}
        flows = _W['flows'][key]
        model, threshold, sig = None, t['threshold'], t['signal']
        if sig in ('gnn', 'lstm_only'):
            if not X._MODELS:
                X.load_models(threads=_W['threads'])
            mm = X._MODELS[(sig, t['mseed'])]
            model, threshold = mm['model'], mm['threshold']
        guard = dict(t['guard'])
        if t['strategy'] == 'guarded':
            guard['signal'] = 'model' if sig in ('gnn', 'lstm_only') else sig
        t0 = time.perf_counter()
        r = X.run_case_x(g, flows, 150, t['strategy'], threshold=threshold, model=model, guard=guard,
                         instrument=True, rng_seed=t['rs'])
        assert r['packet_conservation'] and r['drop_split_ok'], t['id']
        out = {k: v for k, v in t.items()}
        out['threshold_used'] = threshold
        out['wall_s'] = time.perf_counter() - t0
        out.update({k: r[k] for k in KEEP if k in r})
        return out
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())


def load():
    R = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if 'error' not in r:
                R[r['id']] = r
    return R


def cmd_run(a):
    done = set(load())
    buffers = [int(x) for x in a.buffers.split(',')]
    T = [t for t in tasks(buffers) if t['id'] not in done]
    if a.only:
        T = [t for t in T if t['family'] == a.only.split(':')[0] and t['seed'] == int(a.only.split(':')[1])]
    T.sort(key=lambda t: (t['family'], t['seed'], t['buf']))
    print(f'{len(T)} runs to do ({len(done)} done)', flush=True)
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.threads,)) as pool, open(OUT, 'a') as fh:
        for i, r in enumerate(pool.imap(_run, T, chunksize=4)):
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            if 'error' in r:
                print('ERROR', r['id'], r['error'][-800:], flush=True)
            if (i + 1) % 50 == 0 or i + 1 == len(T):
                el = time.time() - t0
                print(f'  {i + 1}/{len(T)} {el / 60:.1f} min, eta {el / (i + 1) * (len(T) - i - 1) / 60:.1f} min', flush=True)


def cmd_gate(a):
    """The 2,000-packet runs must equal the published runs field by field."""
    R = load()
    W = os.path.dirname(RESULTS)
    ref = {}
    name = {'ecmp': 'ecmp', 'reactive': 'reactive', 'gnn': 'unguarded_gnn', 'guarded_gnn': 'guarded_gnn',
            'guarded_lstm': 'guarded_lstm', 'guarded_persistence': 'guarded_persistence',
            'guarded_score': 'guarded_score'}
    for r in json.load(open(os.path.join(NF_ROOT, 'research', 'benchmark_results_v4.json')))['runs']:
        m = None if r['train_seed'] is None else r['train_seed'] - 42
        ref[f"hb|2000|{name[r['strategy']]}|{m}|{r['pattern']}|{r['seed']}"] = r
    for line in open(os.path.join(RESULTS, 'rev2_noguard.jsonl')):
        r = json.loads(line)
        if r.get('variant') == 'score_no_rules':
            ref[f"hb|2000|score_no_rules|None|{r['family']}|{r['seed']}"] = r
    for line in open(os.path.join(RESULTS, 'rev4_fb.jsonl')):
        r = json.loads(line)
        if r.get('mode') == 'open':
            v = {'fb_default': 'flowbender', 'fb_aggressive': 'flowbender_5pct_1tick'}[r['variant']]
            ref[f"hb|2000|{v}|None|{r['family']}|{r['seed']}"] = r
    fields = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'total_delivered', 'reroutes']
    n = bad = missing = 0
    for k, r in R.items():
        if r['buf'] != 2000:
            continue
        if k not in ref:
            missing += 1
            continue
        n += 1
        diff = {f: (r.get(f), ref[k].get(f)) for f in fields if f in ref[k] and r.get(f) != ref[k].get(f)}
        if diff:
            bad += 1
            if bad <= 10:
                print('MISMATCH', k, diff)
    print(f'gate: {n} runs compared, {bad} mismatches, {missing} without a published reference')
    print('GATE PASSED' if n and not bad else 'GATE FAILED')


def boot(d, draws=2000, seed=42, cluster=False):
    """Copied from rev5_stats.boot: d = {(family, seed): value}; bootstrap distribution of the mean
    (cluster=True: families with replacement, then seeds within each drawn family)."""
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


def pts(A, E, cells, cluster=False):
    """Paired difference A - E per cell, as % of the full-sample mean of E (the paper's 'points' scaling)."""
    d = {c: a - e for c, a, e in zip(cells, A, E)}
    base = float(np.mean(E))
    m = float(np.mean(list(d.values())) / base * 100)
    lo, hi = np.percentile(boot(d, cluster=cluster), [2.5, 97.5]) / base * 100
    return m, [float(lo), float(hi)]


def cmd_analyze(a):
    R = load()
    buffers = sorted({r['buf'] for r in R.values()})
    cells = [(f, s) for f in FAM for s in BENCH]
    variants = [v for v, *_ in STRATS]
    S, lines = {}, []

    def val(B, v, f, s, field):
        xs = [r[field] for r in R.values() if r['buf'] == B and r['variant'] == v and r['family'] == f
              and r['seed'] == s and field in r]
        return float(np.mean(xs)) if xs else np.nan       # mean over training seeds

    for B in buffers:
        get = lambda v, field: np.array([val(B, v, f, s, field) for f, s in cells])
        E = {k: get('ecmp', k) for k in ('total_drops', 'fabric_link_drops', 'host_link_drops', 'avg_throughput')}
        S[B] = {}
        lines += [f'\n### Host-link buffers = {B:,} packets (fabric 2,000; seeds 3000-3009, 60 cells)\n',
                  '| Strategy | Total drops vs ECMP [95% CI paired; clustered] | Fabric vs ECMP | Host vs ECMP | '
                  'Host drops/run | Delivered pkt/tick (Δ vs ECMP [95% CI]) | Mean / p99 delay (ms) | Reroutes/run |',
                  '|---|---|---|---|---|---|---|---|']
        for v in variants:
            X = {k: get(v, k) for k in ('total_drops', 'fabric_link_drops', 'host_link_drops', 'avg_throughput',
                                        'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes',
                                        'reordered_fraction')}
            if np.isnan(X['total_drops']).any():
                continue
            pct = lambda x, e: 100 * (x.mean() - e.mean()) / e.mean()
            d, ci = pts(X['total_drops'], E['total_drops'], cells)
            _, cic = pts(X['total_drops'], E['total_drops'], cells, cluster=True)
            dv, civ = pts(X['avg_throughput'], E['avg_throughput'], cells)
            fab = pct(X['fabric_link_drops'], E['fabric_link_drops']) if E['fabric_link_drops'].mean() else float('nan')
            host = pct(X['host_link_drops'], E['host_link_drops'])
            S[B][v] = dict(total_pct=d, total_ci=ci, total_ci_clustered=cic, fabric_pct=fab, host_pct=host,
                           host_per_run=float(X['host_link_drops'].mean()), ecmp_total_per_run=float(E['total_drops'].mean()),
                           delivered=float(X['avg_throughput'].mean()), delivered_pct=dv, delivered_ci=civ,
                           mean_delay_ms=float(X['delivered_mean_latency_us'].mean() / 1000),
                           p99_delay_ms=float(X['delivered_p99_latency_us'].mean() / 1000),
                           reroutes=float(X['reroutes'].mean()), reordered_pct=float(100 * np.nanmean(X['reordered_fraction'])))
            q = S[B][v]
            lines.append(f"| {v} | {q['total_pct']:+.2f}% [{ci[0]:+.2f}, {ci[1]:+.2f}; {cic[0]:+.2f}, {cic[1]:+.2f}] | "
                         f"{q['fabric_pct']:+.1f}% | {q['host_pct']:+.1f}% | {q['host_per_run']:,.0f} | "
                         f"{q['delivered']:,.1f} ({dv:+.2f}% [{civ[0]:+.2f}, {civ[1]:+.2f}]) | "
                         f"{q['mean_delay_ms']:.2f} / {q['p99_delay_ms']:.2f} | {q['reroutes']:,.1f} |")
        # guarded vs FlowBender-style, points of ECMP total drops
        lines.append('\nGuarded minus FlowBender-style (points of mean ECMP total drops, paired 95% CI):')
        F = get('flowbender', 'total_drops')
        for v in ('guarded_gnn', 'guarded_lstm', 'guarded_persistence', 'guarded_score'):
            G = get(v, 'total_drops')
            dd = {c: g - f for c, g, f in zip(cells, G, F)}
            base = float(E['total_drops'].mean())
            m = float(np.mean(list(dd.values())) / base * 100)
            ci = (np.percentile(boot(dd), [2.5, 97.5]) / base * 100).tolist()
            cic = (np.percentile(boot(dd, cluster=True), [2.5, 97.5]) / base * 100).tolist()
            S[B][v]['minus_flowbender_pts'] = [m, ci, cic]
            lines.append(f'- {v}: {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]; clustered [{cic[0]:+.2f}, {cic[1]:+.2f}]')
    json.dump(S, open(os.path.join(RESULTS, 'rev6_hostbuf_summary.json'), 'w'), indent=1)
    open(os.path.join(RESULTS, 'rev6_hostbuf_tables.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'gate', 'analyze'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--buffers', default=','.join(map(str, BUFFERS)))
    ap.add_argument('--only', default=None, help='family:seed, for a quick test (e.g. uniform:3000)')
    a = ap.parse_args()
    {'run': cmd_run, 'gate': cmd_gate, 'analyze': cmd_analyze}[a.cmd](a)
