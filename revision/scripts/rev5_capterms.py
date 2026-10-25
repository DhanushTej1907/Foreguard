"""Capacity-check term ablation (paper Table A7), unseen traffic seeds 4000-4009, six families.

  python revision/scripts/rev5_capterms.py campaign [--gate-only] [--workers 2]
  python revision/scripts/rev5_capterms.py audit       [--workers 2]
  python revision/scripts/rev5_capterms.py validation  [--workers 2]
  python revision/scripts/rev5_capterms.py determinism --out <file outside results/>

Groups
  campaign     6 configs x 2 signals x 6 families x 10 seeds = 720 runs, audit OFF -> rev5_capterms.jsonl
               (every Table A7 outcome). --gate-only runs only the 180 runs that repeat saved results
               (A|score, A|random, nocheck|score).
  audit        config A with the opt-in shadow audit, both signals: 120 runs -> rev5_capterms_audit.jsonl
               (counterfactual term attribution only; outcomes must equal the audit-off A runs)
  validation   ECMP, 60 runs, plain engine as in replicate.jsonl -> rev5_capterms_validation.jsonl (gate G0)
  determinism  12 campaign tasks (one per config x signal) rerun into --out for a repeatability check

Configs (Table A7): A full check; B no queue term; Cp no observed arrivals (summed rates, queue kept);
D summed rates only; E as D with every flow at the mean active sending rate; nocheck capacity check removed.
Every other setting is the default guard. Files are appended to and resumable (finished ids are skipped);
the script refuses to append when the code differs from the code recorded in rev5_capterms_meta.json.
"""
from paths import NF_ROOT, RESULTS  # folder locations (see paths.py)
import argparse, datetime, glob, hashlib, json, os, platform, subprocess, sys, time, traceback
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
SEEDS = list(range(4000, 4010))
SIGNALS = ['score', 'random']
THRESHOLD = 0.851                       # validation threshold of the score; also used for the random signal
TICKS = 150
# the default guard, written out in full so that the recorded parameters are the ones passed
BASE_GUARD = dict(max_changes=4, headroom=0.90, reserve=True, min_improvement=0.15, stretch=2, cooldown=5,
                  k_paths=3, max_age=1, rate_noise=0.0, rate_mode='exact', cap_arrivals=True, cap_queue=True,
                  audit=False)
CONFIGS = {'A': {}, 'B': dict(cap_queue=False), 'Cp': dict(cap_arrivals=False),
           'D': dict(cap_arrivals=False, cap_queue=False),
           'E': dict(cap_arrivals=False, cap_queue=False, rate_mode='blind'),
           'nocheck': dict(headroom=None)}
GATE = {('A', 'score'), ('A', 'random'), ('nocheck', 'score')}
OUT = {'campaign': 'rev5_capterms.jsonl', 'audit': 'rev5_capterms_audit.jsonl',
       'validation': 'rev5_capterms_validation.jsonl'}
META = os.path.join(RESULTS, 'rev5_capterms_meta.json')
LOG = os.path.join(RESULTS, 'log_rev5_capterms.txt')
KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'unroutable_drops', 'total_offered', 'total_delivered',
        'avg_throughput', 'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes', 'rejections',
        'reordered_packets', 'reordered_fraction', 'flows_with_reordering', 'n_flows', 'packet_conservation',
        'drop_split_ok', 'ticks_run', 'decision_digest', 'signal_digest', 'cap_audit', 'audit_definition_version']


def resolved_guard(config, audit=False):
    g = dict(BASE_GUARD, **CONFIGS[config])
    g['audit'] = audit
    return g


def _base(family, seed, signal):
    fi = FAM.index(family)
    return dict(family=family, family_index=fi, seed=seed, signal=signal, rng_seed=1000 * fi + seed,
                noise_seed=1000 * fi + seed, ticks=TICKS)


def tasks(group, gate_only=False):
    T = []
    for fam in FAM:
        for s in SEEDS:
            if group == 'validation':
                T.append(dict(_base(fam, s, None), id=f'capvalid|ecmp|-|{fam}|{s}', group=group, config='ecmp',
                              strategy='ecmp', threshold=None, guard_params=None, engine='plain'))
                continue
            for sig in SIGNALS:
                if group == 'audit':
                    T.append(dict(_base(fam, s, sig), id=f'capaudit|A|{sig}|{fam}|{s}', group=group, config='A',
                                  strategy='guarded', threshold=THRESHOLD, guard_params=resolved_guard('A', audit=True),
                                  engine='instrumented'))
                    continue
                for cfg in CONFIGS:
                    if gate_only and (cfg, sig) not in GATE:
                        continue
                    T.append(dict(_base(fam, s, sig), id=f'capterms|{cfg}|{sig}|{fam}|{s}', group='campaign',
                                  config=cfg, strategy='guarded', threshold=THRESHOLD,
                                  guard_params=resolved_guard(cfg), engine='instrumented'))
    return T


def determinism_tasks():
    T = tasks('campaign')
    pick, out = {}, []
    for i, (cfg, sig) in enumerate((c, s) for c in CONFIGS for s in SIGNALS):
        fam, seed = FAM[i % len(FAM)], SEEDS[(3 * i) % len(SEEDS)]
        out.append(next(t for t in T if (t['config'], t['signal'], t['family'], t['seed']) == (cfg, sig, fam, seed)))
    return out


# ───────────────────────────── provenance ──────────────────────────────
def code_hashes():
    files = [os.path.join(HERE, 'nf_exp.py'), os.path.join(HERE, 'rev5_capterms.py'), os.path.join(HERE, 'paths.py'),
             os.path.join(NF_ROOT, 'config.py')] + sorted(glob.glob(os.path.join(NF_ROOT, 'simulation', '*.py'))) + \
            [os.path.join(NF_ROOT, 'routing', 'rerouter.py')]
    out = {}
    for p in files:
        with open(p, 'rb') as fh:
            out[os.path.relpath(p, NF_ROOT).replace('\\', '/')] = hashlib.sha256(fh.read()).hexdigest()
    return out


def environment():
    import numpy, networkx
    try:
        import scipy
        sv = scipy.__version__
    except Exception:
        sv = None
    try:
        head = subprocess.run(['git', '-C', NF_ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', NF_ROOT, 'status', '--porcelain'], capture_output=True,
                                    text=True).stdout.strip())
    except Exception:
        head, dirty = None, None
    return dict(python=sys.version, numpy=numpy.__version__, networkx=networkx.__version__, scipy=sv,
                platform=platform.platform(), machine=platform.machine(), git_head=head, git_dirty=dirty)


def check_meta(group):
    hashes = code_hashes()
    meta = json.load(open(META)) if os.path.exists(META) else None
    if meta is not None and meta['code_sha256'] != hashes:
        changed = sorted(k for k in set(hashes) | set(meta['code_sha256'])
                         if hashes.get(k) != meta['code_sha256'].get(k))
        raise SystemExit(f'refusing to append: code differs from {META}: {changed}')
    if meta is None:
        meta = dict(code_sha256=hashes, environment=environment(), base_guard=BASE_GUARD, configs=CONFIGS,
                    threshold=THRESHOLD, families=FAM, seeds=SEEDS, ticks=TICKS, invocations=[])
    else:
        now, then = environment(), meta['environment']
        pinned = ('python', 'numpy', 'networkx', 'scipy', 'platform', 'machine')
        if any(now[k] != then[k] for k in pinned):
            raise SystemExit(f'refusing to append: environment differs from {META}')
    return meta


def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf8') as fh:
        fh.write(msg + '\n')


# ───────────────────────────── workers ─────────────────────────────────
_W = {}


def _init():
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    sys.path.insert(0, HERE)
    import nf_exp
    from simulation.topology import build_fat_tree
    _W['X'], _W['g'], _W['flows'] = nf_exp, build_fat_tree(), {}


def _run(t):
    try:
        X, g = _W['X'], _W['g']
        from simulation.traffic_generator import TrafficGenerator
        key = (t['family'], t['seed'])
        if key not in _W['flows']:
            _W['flows'] = {key: TrafficGenerator(g, seed=t['seed']).generate_scenario(total_ticks=TICKS,
                                                                                      pattern=t['family'])}
        flows = _W['flows'][key]
        guard = None
        if t['strategy'] == 'guarded':
            guard = dict(t['guard_params'], signal=t['signal'], noise_seed=t['noise_seed'])
        t0 = time.perf_counter()
        r = X.run_case_x(g, flows, TICKS, t['strategy'], threshold=t['threshold'], guard=guard,
                         rng_seed=t['rng_seed'], instrument=t['engine'] == 'instrumented')
        out = dict(t)
        out.update({k: r[k] for k in KEEP if k in r})
        out['wall_s'] = time.perf_counter() - t0
        return out
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())


def run_pool(T, path, workers):
    T = sorted(T, key=lambda t: (t['family'], t['seed']))      # keep (family, seed) blocks for the flow cache
    t0 = time.time()
    ctx = mp.get_context('spawn')
    n_err = 0
    with ctx.Pool(workers, initializer=_init) as pool, open(path, 'a') as fh:
        for i, r in enumerate(pool.imap(_run, T, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            if 'error' in r:
                n_err += 1
                log(f'ERROR {r["id"]} {r["error"][-400:]}')
            if i % 50 == 0 or i == len(T):
                el = time.time() - t0
                log(f'  {i}/{len(T)} {el / 60:.1f} min, eta {el / i * (len(T) - i) / 60:.1f} min')
    return n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('group', choices=['campaign', 'audit', 'validation', 'determinism'])
    ap.add_argument('--gate-only', action='store_true')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default=None, help='determinism only: output file (must be outside results/)')
    args = ap.parse_args()
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    if args.group == 'determinism':
        if not args.out or os.path.abspath(args.out).startswith(os.path.abspath(RESULTS)):
            raise SystemExit('--out must be given and lie outside results/')
        if os.path.exists(args.out):
            raise SystemExit(f'{args.out} exists')
        n = run_pool(determinism_tasks(), args.out, args.workers)
        print(f'determinism: {len(determinism_tasks())} runs, {n} errors -> {args.out}')
        return
    meta = check_meta(args.group)
    path = os.path.join(RESULTS, OUT[args.group])
    assert os.path.basename(path).startswith('rev5_capterms')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if 'error' not in r:
                done.add(r['id'])
    T = [t for t in tasks(args.group, args.gate_only) if t['id'] not in done]
    start = datetime.datetime.now().isoformat(timespec='seconds')
    log(f'[{start}] {args.group}{" (gate only)" if args.gate_only else ""}: {len(T)} tasks to run ({len(done)} done)')
    n_err = run_pool(T, path, args.workers) if T else 0
    end = datetime.datetime.now().isoformat(timespec='seconds')
    meta['invocations'].append(dict(group=args.group, gate_only=args.gate_only, start=start, end=end, n_tasks=len(T),
                                    n_errors=n_err, workers=args.workers))
    with open(META, 'w') as fh:
        json.dump(meta, fh, indent=1)
    log(f'[{end}] {args.group}: done, {n_err} errors')


if __name__ == '__main__':
    main()
