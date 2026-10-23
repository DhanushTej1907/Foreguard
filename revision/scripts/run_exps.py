"""Run experiment groups for the revision with a small process pool; results are appended to
results/<group>.jsonl (resumable: finished task ids are skipped).

  python scripts/run_exps.py <group> [--workers 2] [--threads 1]

Groups (unseen traffic seeds 4000-4009, benchmark seeds 3000-3009, six workload families each)
  verify_model   guarded/unguarded GNN + guarded LSTM on benchmark seeds 3000-3001 (exactness check
                 against research/benchmark_results_v4.json)
  replicate      every strategy with default parameters on unseen seeds 4000-4009 (fresh benchmark)
  ablation       leave-one-rule-out on unseen seeds; signals score, persistence, GNN (3 training seeds)
  sweep          alpha / delta / kappa sweeps for guarded score on unseen seeds
  delay          telemetry delay d_T and actuation delay d_A (2 and 5 ticks, and both 5) on unseen
                 seeds; guarded score and guarded GNN (3 training seeds)
  interval       forecaster invoked every I = 2 or 5 ticks, guarded GNN (training seed 0), unseen seeds
  instrument     every strategy on benchmark seeds 3000-3009, per-flow loss and packet reordering
                 (model-based strategies use training seed 0)
  closedloop     closed-loop transport (retransmission + AIMD, flow completion time) on benchmark
                 seeds 3000-3004; guarded LSTM omitted, model-based strategies use training seed 0
  timing         per-tick controller and inference time (run with --workers 1 --threads 2)
  platt          guarded GNN with Platt-recalibrated probabilities (needs results/platt_params.json)
"""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import argparse, json, os, sys, time, traceback
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = RESULTS
PATTERNS = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
TH_SCORE, TH_PERS = 0.851, 0.5          # validation thresholds from benchmark_results_v4.json
BENCH = list(range(3000, 3010))
UNSEEN = list(range(4000, 4010))
KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'unroutable_drops', 'total_offered',
        'total_delivered', 'avg_throughput', 'delivered_mean_latency_us', 'delivered_p99_latency_us',
        'reroutes', 'ticks_run', 'inference_ms_p50', 'inference_ms_p99', 'rejections',
        'reordered_packets', 'reordered_fraction', 'flows_with_reordering', 'n_flows',
        'flow_loss_p50', 'flow_loss_p90', 'flow_loss_mean', 'completed', 'incomplete', 'fct_mean',
        'fct_p50', 'fct_p99', 'slowdown_mean', 'slowdown_p50', 'slowdown_p99', 'makespan',
        'retx_packets', 'offered_unique', 'per_flow', 'controller_ms', 'inference_ms']


def task(group, variant, strategy, signal=None, mseed=None, family=None, seed=None, guard=None, **kw):
    tid = '|'.join(str(x) for x in (group, variant, strategy, signal, mseed, family, seed))
    return dict(id=tid, group=group, variant=variant, strategy=strategy, signal=signal, mseed=mseed,
                family=family, seed=seed, guard=guard or {}, **kw)


def strategies_default():
    """(variant label, strategy, signal, model seeds)"""
    return [('ecmp', 'ecmp', None, [None]), ('reactive', 'reactive', None, [None]),
            ('unguarded_gnn', 'unguarded', 'gnn', [0, 1, 2]),
            ('guarded_gnn', 'guarded', 'gnn', [0, 1, 2]), ('guarded_lstm', 'guarded', 'lstm_only', [0, 1, 2]),
            ('guarded_persistence', 'guarded', 'persistence', [None]), ('guarded_score', 'guarded', 'score', [None])]


ABLATIONS = {
    'full': {},
    'no_capacity': dict(headroom=None),
    'no_reservation': dict(reserve=False),
    'no_budget': dict(max_changes=None),
    'no_stretch': dict(stretch=None),
    'no_min_improvement': dict(min_improvement=0.0),
    'no_cooldown': dict(cooldown=0),
    'fixed_threshold_0.7': '__th07__',
    'all_rules_off': dict(headroom=None, max_changes=None, stretch=None, min_improvement=0.0, cooldown=0),
}
SWEEP_SCORE = {'alpha': [0.6, 0.75, 1.0, 1.2], 'delta': [0.05, 0.3, 0.5], 'kappa': [2, 10, 20]}
SWEEP_GNN = {'alpha': [0.75, 1.0], 'delta': [0.05, 0.3], 'kappa': [2, 10]}
PARAM = {'alpha': 'headroom', 'delta': 'min_improvement', 'kappa': 'cooldown'}
DELAYS = [('dT', d, 0) for d in (2, 5)] + [('dA', 0, d) for d in (2, 5)] + [('dTdA', 5, 5)]


def build(group):
    T = []
    if group == 'verify_model':
        for fam in PATTERNS:
            for s in (3000, 3001):
                for v, st, sig, ms in strategies_default():
                    if sig in ('gnn', 'lstm_only'):
                        for m in ms:
                            T.append(task(group, v, st, sig, m, fam, s))
    elif group == 'replicate':
        for fam in PATTERNS:
            for s in UNSEEN:
                for v, st, sig, ms in strategies_default():
                    for m in ms:
                        T.append(task(group, v, st, sig, m, fam, s))
    elif group == 'ablation':
        for fam in PATTERNS:
            for s in UNSEEN:
                for name, g in ABLATIONS.items():
                    for sig, ms in (('score', [None]), ('persistence', [None]), ('gnn', [0, 1, 2])):
                        if name == 'fixed_threshold_0.7' and sig == 'persistence':
                            continue
                        for m in ms:
                            if g == '__th07__':
                                T.append(task(group, name, 'guarded', sig, m, fam, s, threshold=0.7))
                            else:
                                T.append(task(group, name, 'guarded', sig, m, fam, s, guard=g))
    elif group == 'sweep':
        for fam in PATTERNS:
            for s in UNSEEN:
                for sig, grid, ms in (('score', SWEEP_SCORE, [None]),):
                    for p, values in grid.items():
                        for val in values:
                            for m in ms:
                                T.append(task(group, f'{p}={val}', 'guarded', sig, m, fam, s, guard={PARAM[p]: val}))
    elif group == 'delay':
        for fam in PATTERNS:
            for s in UNSEEN:
                for name, dT, dA in DELAYS:
                    for sig, ms in (('score', [None]), ('gnn', [0, 1, 2])):
                        for m in ms:
                            T.append(task(group, f'{name}:{dT},{dA}', 'guarded', sig, m, fam, s, d_T=dT, d_A=dA))
    elif group == 'interval':
        for fam in PATTERNS:
            for s in UNSEEN:
                for I in (2, 5):
                    for m in (0,):
                        T.append(task(group, f'I={I}', 'guarded', 'gnn', m, fam, s, interval=I))
    elif group == 'instrument':
        for fam in PATTERNS:
            for s in BENCH:
                for v, st, sig, ms in strategies_default():
                    for m in (ms if sig not in ('gnn', 'lstm_only') else [0]):
                        T.append(task(group, v, st, sig, m, fam, s, instrument=True))
    elif group == 'closedloop':
        for fam in PATTERNS:
            for s in BENCH[:5]:
                for v, st, sig, ms in strategies_default():
                    if v == 'guarded_lstm':
                        continue
                    for m in (ms if sig not in ('gnn', 'lstm_only') else [0]):
                        T.append(task(group, v, st, sig, m, fam, s, closed_loop=True))
    elif group == 'platt':
        for fam in PATTERNS:
            for s in UNSEEN:
                for m in (0, 1, 2):
                    T.append(task(group, 'guarded_gnn_platt', 'guarded', 'gnn_platt', m, fam, s))
    elif group == 'timing':
        for fam in PATTERNS:
            for s in (3000, 3001):
                T.append(task(group, 'guarded_gnn', 'guarded', 'gnn', 0, fam, s, time_controller=True))
                T.append(task(group, 'guarded_score', 'guarded', 'score', None, fam, s, time_controller=True))
                T.append(task(group, 'guarded_lstm', 'guarded', 'lstm_only', 0, fam, s, time_controller=True))
    else:
        raise SystemExit(f'unknown group {group}')
    return T


_W = {}


def _init(threads):
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    sys.path.insert(0, HERE)
    import nf_exp
    from simulation.topology import build_fat_tree
    _W['X'] = nf_exp
    _W['g'] = build_fat_tree()
    _W['flows'] = {}
    _W['threads'] = threads


def _run(t):
    try:
        X, g = _W['X'], _W['g']
        from simulation.traffic_generator import TrafficGenerator
        key = (t['family'], t['seed'])
        if key not in _W['flows']:
            _W['flows'] = {key: TrafficGenerator(g, seed=t['seed']).generate_scenario(total_ticks=150, pattern=t['family'])}
        flows = _W['flows'][key]
        model, threshold = None, None
        sig = t['signal']
        if sig in ('gnn', 'lstm_only'):
            if not X._MODELS:
                X.load_models(threads=_W['threads'])
            mm = X._MODELS[(sig, t['mseed'])]
            model, threshold = mm['model'], mm['threshold']
        elif sig == 'gnn_platt':
            if not X._MODELS:
                X.load_models(threads=_W['threads'])
            pp = json.load(open(os.path.join(OUTDIR, 'platt_params.json')))[str(t['mseed'])]
            model = X.PlattScaled(X._MODELS[('gnn', t['mseed'])]['model'].model, pp['a'], pp['b'])
            threshold = pp['threshold']
        elif sig == 'score':
            threshold = TH_SCORE
        elif sig == 'persistence':
            threshold = TH_PERS
        if t.get('threshold') is not None:
            threshold = t['threshold']
        guard = dict(t['guard'])
        if t['strategy'] == 'guarded':
            guard['signal'] = 'model' if sig in ('gnn', 'lstm_only', 'gnn_platt') else sig
        t0 = time.perf_counter()
        r = X.run_case_x(g, flows, 150, t['strategy'], threshold=threshold, model=model, guard=guard,
                         d_T=t.get('d_T', 0), d_A=t.get('d_A', 0), interval=t.get('interval', 1),
                         instrument=t.get('instrument', False), closed_loop=t.get('closed_loop', False),
                         time_controller=t.get('time_controller', False))
        assert r['packet_conservation'] and r['drop_split_ok'], t['id']
        out = {k: v for k, v in t.items() if k != 'guard'}
        out['guard'] = t['guard']
        out['threshold_used'] = threshold
        out['wall_s'] = time.perf_counter() - t0
        out.update({k: r[k] for k in KEEP if k in r})
        return out
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('group')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--signals', default=None, help='comma list: only tasks whose signal is in it (none = model-free)')
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f'{args.group}.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                if 'error' not in r:
                    done.add(r['id'])
            except Exception:
                pass
    tasks = [t for t in build(args.group) if t['id'] not in done]
    if args.signals:
        keep = set(args.signals.split(','))
        tasks = [t for t in tasks if (t['signal'] or 'none') in keep]
    # keep (family, seed) blocks together for the flow cache
    tasks.sort(key=lambda t: (t['family'], t['seed']))
    print(f'{args.group}: {len(tasks)} tasks to run ({len(done)} done)', flush=True)
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.threads,)) as pool, open(path, 'a') as fh:
        for i, r in enumerate(pool.imap(_run, tasks, chunksize=4)):
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            if 'error' in r:
                print('ERROR', r['id'], r['error'][-500:], flush=True)
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                el = time.time() - t0
                print(f'  {i + 1}/{len(tasks)} {el / 60:.1f} min, eta {el / (i + 1) * (len(tasks) - i - 1) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
