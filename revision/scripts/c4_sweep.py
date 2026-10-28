"""Concern 4 (4th review): closed-loop FCT under reordering models between no penalty and the full penalty.

Settings (see results/c4/PROTOCOL.md): off, W0, W1, W2, W4, W8, W16 (penalise a reordered batch only if its displacement
exceeds W ticks; W0 = the full penalty of rev4_c4.py), rtt4 (W = RTT/4 of the batch's path), dup3 (dupACK threshold 3).
Strategies = Table A6 + FlowBender-style, with the model / seed conventions of rev4_c4.py and rev4_fb.py.
Cells: benchmark seeds 3000-3004 x 6 families.

  python revision/scripts/c4_sweep.py --settings off,W0 [--workers 3] [--out PATH] [--pilot]

Appends to results/c4/runs.jsonl (resumable: ids already present without error are skipped).
--pilot: only seed 3000 of each family (timing pilot; use --out to a scratch path)."""
from paths import NF_ROOT, RESULTS  # folder locations (see paths.py)
import argparse, json, os, sys, time, traceback
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(RESULTS, 'c4')
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
SEEDS = list(range(3000, 3005))
NONE = dict(headroom=None, max_changes=None, stretch=None, min_improvement=0.0, cooldown=0)
FB = dict(k_frac=0.1, n_consec=2)
# (variant, strategy, signal, threshold, guard) -- as rev4_c4.VARS, plus FlowBender-style as rev4_fb.py (fb_default)
VARS = [('ecmp', 'ecmp', None, None, {}), ('reactive', 'reactive', None, None, {}),
        ('unguarded_gnn', 'unguarded', 'gnn', None, {}), ('guarded_gnn', 'guarded', 'gnn', None, {}),
        ('guarded_persistence', 'guarded', 'persistence', 0.5, {}), ('guarded_score', 'guarded', 'score', 0.851, {}),
        ('score_no_rules', 'guarded', 'score', 0.851, NONE), ('fb_default', 'flowbender', None, None, FB)]
SETTINGS = {'off': dict(reorder_penalty=False),
            **{f'W{w}': dict(reorder_penalty=True, reorder_window=w) for w in (0, 1, 2, 4, 8, 16)},
            'rtt4': dict(reorder_penalty=True, reorder_window='rtt/4'),
            'dup3': dict(reorder_penalty=True, dupack_threshold=3)}
DROP = ('inference_ms',)   # raw per-call timing lists are not kept


def tasks(settings, pilot=False):
    T = []
    for name in settings:
        for fi, f in enumerate(FAM):
            for s in (SEEDS[:1] if pilot else SEEDS):
                for v, st, sig, th, g in VARS:
                    T.append(dict(id=f'{name}|{v}|{f}|{s}', setting=name, variant=v, strategy=st, signal=sig,
                                  threshold=th, guard=g, family=f, seed=s,
                                  rng_seed=1000 * fi + s if st == 'flowbender' else 0))
    return T


_W = {}


def _init():
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    sys.path.insert(0, HERE)
    import nf_exp
    from simulation.topology import build_fat_tree
    _W['X'], _W['g'], _W['flows'] = nf_exp, build_fat_tree(), {}


def run_task(t, X, g, flows):
    model, th, guard = None, t['threshold'], dict(t['guard'])
    if t['signal'] == 'gnn':
        if ('gnn', 0) not in X._MODELS:
            X.load_models(kinds=('gnn',), seeds=(0,), threads=1)
        model = X._MODELS[('gnn', 0)]['model']
        th = X._MODELS[('gnn', 0)]['threshold'] if t['strategy'] == 'guarded' else None
    if t['strategy'] == 'guarded':
        guard['signal'] = 'model' if t['signal'] == 'gnn' else t['signal']
    s = time.perf_counter()
    r = X.run_case_x(g, flows, 150, t['strategy'], threshold=th, model=model, guard=guard, closed_loop=True,
                     rng_seed=t['rng_seed'], **SETTINGS[t['setting']])
    wall = time.perf_counter() - s
    assert r['packet_conservation'] and r['drop_split_ok'], t['id']
    out = {k: t[k] for k in ('id', 'setting', 'variant', 'family', 'seed', 'rng_seed')}
    out['threshold_used'] = th
    out['wall_s'] = wall
    out.update({k: v for k, v in r.items() if k not in DROP})
    return out


def _run(t):
    try:
        X, g = _W['X'], _W['g']
        from simulation.traffic_generator import TrafficGenerator
        key = (t['family'], t['seed'])
        if key not in _W['flows']:
            _W['flows'] = {key: TrafficGenerator(g, seed=t['seed']).generate_scenario(total_ticks=150, pattern=t['family'])}
        return run_task(t, X, g, _W['flows'][key])
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())


def _check_out(path):
    """Never write into an existing result file outside results/c4 (protected)."""
    ap = os.path.abspath(path)
    res = os.path.abspath(RESULTS)
    if ap.startswith(res + os.sep) and not ap.startswith(os.path.abspath(OUTDIR) + os.sep):
        raise SystemExit(f'refusing to write {ap}: outside results/c4')
    for prot in ('checkpoints', 'data', 'research'):
        if ap.startswith(os.path.join(os.path.abspath(NF_ROOT), prot) + os.sep):
            raise SystemExit(f'refusing to write {ap}: protected folder')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', required=True, help='comma list of ' + ','.join(SETTINGS))
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(OUTDIR, 'runs.jsonl'))
    ap.add_argument('--pilot', action='store_true')
    a = ap.parse_args()
    names = a.settings.split(',')
    bad = [n for n in names if n not in SETTINGS]
    if bad:
        raise SystemExit(f'unknown settings {bad}')
    _check_out(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if 'error' not in r:
                done.add(r['id'])
    T = [t for t in tasks(names, a.pilot) if t['id'] not in done]
    T.sort(key=lambda t: (t['family'], t['seed'], t['setting']))   # keep (family, seed) blocks together
    print(f'{len(T)} tasks ({len(done)} done) -> {a.out}', flush=True)
    t0, errors = time.time(), 0
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, initializer=_init) as pool, open(a.out, 'a') as fh:
        for i, r in enumerate(pool.imap(_run, T, chunksize=2), 1):
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            if 'error' in r:
                errors += 1
                print('ERROR', r['id'], r['error'][-800:], flush=True)
            if i % 60 == 0 or i == len(T):
                el = time.time() - t0
                print(f'  {i}/{len(T)} {el / 60:.1f} min, eta {el / i * (len(T) - i) / 60:.1f} min', flush=True)
    print(f'done: {len(T)} tasks, {errors} errors, {(time.time() - t0) / 60:.2f} min wall', flush=True)


if __name__ == '__main__':
    main()
