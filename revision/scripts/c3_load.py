"""Concern 3 routing sweep (instrumented), passive oracle and checks. Results are appended (resumable) to
revision/results/c3/<out>.jsonl. Every run reproduces rev3_load._run exactly (same flows, scaling, ids and
run_case_x arguments) plus instrument=True; ids of the seven rev3_load variants are identical, so S1 / I2 match by id.

  python revision/scripts/c3_load.py run    --variants v1,v2 [--loads ..] [--families ..] [--seeds ..] [--workers 3] [--out load]
  python revision/scripts/c3_load.py pilot  [--workers 3]       # loads 0.25, 1.0 x 6 families x seed 4000 x all routing variants
  python revision/scripts/c3_load.py oracle-pilot | oracle      # loads 0.2, 0.25 (pilot: family 'mixed') -> oracle.jsonl
  python revision/scripts/c3_load.py oracle-repro                # S4: load 1.0, mixed, seeds 4000-4001 vs rev4_c1.jsonl
  python revision/scripts/c3_load.py instrument-check            # S2: uninstrumented reruns of pilot cells

Variants (PROTOCOL.md): tier 1 = ecmp, guarded_score (score, tau 0.851), score_pmix, guarded_gnn (M0/P1.0, x3),
m0p_p10 (M0'/P1.0, x3), m1_pmix (M1/Pmix, x3), score_no_rules, random; tier 2 = m1_p10, m0_pmix, m0p_pmix (x3 each);
oracle = oracle_label (tau 0.5), oracle_score_pmix (score Pmix tau), oracle_score_0851 (only if Pmix tau != 0.851)."""
import argparse, json, os, sys, time, traceback
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c3_common as C

NONE = dict(headroom=None, max_changes=None, stretch=None, min_improvement=0.0, cooldown=0)   # = rev3_load.NONE
KEEP = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'total_offered', 'total_delivered',
        'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes']                    # = rev3_load.KEEP
EXTRA = ['unroutable_drops', 'avg_throughput', 'outstanding_packets', 'reordered_packets', 'reordered_fraction',
         'flows_with_reordering', 'n_flows', 'flow_loss_p50', 'flow_loss_p90', 'flow_loss_mean', 'decision_digest',
         'signal_digest', 'packet_conservation', 'drop_split_ok', 'rejections', 'inference_ms_p50', 'inference_ms_p99',
         'ticks_run']
# name -> (tier, strategy, signal, model spec (tag, stem, protocol) or None, threshold spec, guard)
MODEL_SPECS = {'guarded_gnn': (None, None, None), 'm0p_p10': ('m0p', 'M0p', None), 'm1_pmix': ('m1', 'M1', None),
               'm1_p10': ('m1_p10', 'M1', 'P1.0'), 'm0_pmix': ('m0_pmix', 'M0', 'Pmix'), 'm0p_pmix': ('m0p_pmix', 'M0p', 'Pmix')}
TIER1 = ['ecmp', 'guarded_score', 'score_pmix', 'guarded_gnn', 'm0p_p10', 'm1_pmix', 'score_no_rules', 'random']
TIER2 = ['m1_p10', 'm0_pmix', 'm0p_pmix']
ORACLE = ['oracle_label', 'oracle_score_pmix', 'oracle_score_0851']
REV3_VARIANTS = ['ecmp', 'guarded_score', 'guarded_gnn', 'score_no_rules', 'random']


def calibration():
    return json.load(open(os.path.join(C.OUT, 'calibration.json')))


def make_tasks(variants, loads, families, seeds, force_0851=False):
    need = {'score_pmix', 'oracle_score_pmix', 'oracle_score_0851'} & set(variants)
    score_pmix = calibration()['score']['Pmix_used'] if need else None
    T = []
    for lf in loads:
        for fam in families:
            fi = C.FAM.index(fam)
            for s in seeds:
                base = dict(load=lf, family=fam, seed=s)
                for v in variants:
                    if v == 'ecmp':
                        T.append(dict(base, variant=v, strategy='ecmp'))
                    elif v == 'guarded_score':
                        T.append(dict(base, variant=v, strategy='guarded', signal='score', threshold=C.SCORE_TH_P10, guard={}))
                    elif v == 'score_pmix':
                        T.append(dict(base, variant=v, strategy='guarded', signal='score', threshold=score_pmix, guard={}))
                    elif v == 'score_no_rules':
                        T.append(dict(base, variant=v, strategy='guarded', signal='score', threshold=C.SCORE_TH_P10, guard=dict(NONE)))
                    elif v == 'random':
                        T.append(dict(base, variant=v, strategy='guarded', signal='random', threshold=C.SCORE_TH_P10, guard={},
                                      rng_seed=1000 * fi + s))
                    elif v in MODEL_SPECS:
                        for m in C.TRAIN_SEEDS:
                            T.append(dict(base, variant=v, strategy='guarded', signal='gnn', mseed=m, guard={}))
                    elif v == 'oracle_label':
                        T.append(dict(base, variant=v, strategy='guarded', signal='oracle_label', threshold=0.5, guard={}))
                    elif v == 'oracle_score_pmix':
                        T.append(dict(base, variant=v, strategy='guarded', signal='oracle_score', threshold=score_pmix, guard={}))
                    elif v == 'oracle_score_0851':
                        if force_0851 or score_pmix != C.SCORE_TH_P10:
                            T.append(dict(base, variant=v, strategy='guarded', signal='oracle_score', threshold=C.SCORE_TH_P10, guard={}))
                    else:
                        raise SystemExit(f'unknown variant {v}')
    for t in T:
        t['id'] = '|'.join(str(t.get(k)) for k in ('load', 'variant', 'mseed', 'family', 'seed'))
    return T


_W = {}


def _init():
    import nf_exp
    from simulation.topology import build_fat_tree
    _W['X'], _W['g'], _W['flows'], _W['cal'] = nf_exp, build_fat_tree(), {}, calibration()


def _model(variant, m):
    X = _W['X']
    tag, stem_name, proto = MODEL_SPECS[variant]
    key = ('gnn', m) if tag is None else (tag, 'gnn', m)
    if key not in X._MODELS:
        stems = {'M0': None, 'M0p': C.STEM_M0P, 'M1': C.STEM_M1}
        cal = None
        if proto is not None:
            e = _W['cal']['models'][stem_name]
            cal = {('gnn', k): (e[str(k)][proto]['temperature'], e[str(k)][proto]['threshold']) for k in C.TRAIN_SEEDS}
        X.load_models(kinds=('gnn',), threads=1, stem=stems[stem_name] if stem_name else None, tag=tag, calibration=cal)
    return X._MODELS[key]


def _run(t, instrument=True):
    try:
        X, g = _W['X'], _W['g']
        key = (t['family'], t['seed'], t['load'])
        if key not in _W['flows']:
            _W['flows'] = {key: C.scaled_flows(g, t['seed'], t['family'], t['load'])}
        flows = _W['flows'][key]
        model, th, guard = None, t.get('threshold'), dict(t.get('guard', {}))
        if t.get('signal') == 'gnn':
            mm = _model(t['variant'], t['mseed'])
            model, th = mm['model'], mm['threshold']
            guard['signal'] = 'model'
        elif t['strategy'] == 'guarded':
            guard['signal'] = t['signal']
        t0 = time.perf_counter()
        r = X.run_case_x(g, flows, C.TICKS, t['strategy'], threshold=th, model=model, guard=guard,
                         rng_seed=t.get('rng_seed', 0), instrument=instrument)
        out = {k: t[k] for k in ('id', 'load', 'variant', 'family', 'seed')}
        out['mseed'] = t.get('mseed')
        out['threshold_used'] = th
        out['instrument'] = instrument
        out['wall_s'] = time.perf_counter() - t0
        out.update({k: r[k] for k in KEEP})
        out.update({k: r[k] for k in EXTRA if k in r})
        return out
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())


def _run_plain(t):
    return _run(t, instrument=False)


def execute(tasks, out_name, workers, fn=_run):
    path = os.path.join(C.OUT, out_name + '.jsonl')
    C.guard_output(path)
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if 'error' not in r:
                done.add(r['id'])
    tasks = [t for t in tasks if t['id'] not in done]
    tasks.sort(key=lambda t: (t['load'], t['family'], t['seed']))
    print(f'{out_name}: {len(tasks)} tasks ({len(done)} done)', flush=True)
    t0, errors = time.time(), 0
    if not tasks:
        return 0
    with mp.get_context('spawn').Pool(workers, initializer=_init) as pool, open(path, 'a') as fh:
        for i, r in enumerate(pool.imap(fn, tasks, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if 'error' in r:
                errors += 1
                print('ERROR', r['id'], r['error'][-400:], flush=True)
            if i % 200 == 0 or i == len(tasks):
                el = (time.time() - t0) / 60
                print(f'  {i}/{len(tasks)} {el:.1f} min, eta {el / i * (len(tasks) - i):.1f} min', flush=True)
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['run', 'pilot', 'oracle-pilot', 'oracle', 'oracle-repro', 'instrument-check'])
    ap.add_argument('--variants')
    ap.add_argument('--loads')
    ap.add_argument('--families')
    ap.add_argument('--seeds')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default='load')
    a = ap.parse_args()
    lst = lambda s, f, d: [f(x) for x in s.split(',')] if s else d
    loads, fams, seeds = lst(a.loads, float, C.LOADS), lst(a.families, str, C.FAM), lst(a.seeds, int, C.UNSEEN)
    if a.what == 'run':
        errs = execute(make_tasks(lst(a.variants, str, TIER1), loads, fams, seeds), a.out, a.workers)
    elif a.what == 'pilot':
        errs = execute(make_tasks(TIER1 + TIER2, [0.25, 1.0], C.FAM, [4000]), 'load', a.workers)
    elif a.what == 'oracle-pilot':
        errs = execute(make_tasks(ORACLE, [0.2, 0.25], ['mixed'], C.UNSEEN), 'oracle', a.workers)
    elif a.what == 'oracle':
        errs = execute(make_tasks(ORACLE, [0.2, 0.25], fams, seeds), 'oracle', a.workers)
    elif a.what == 'oracle-repro':          # S4: load 1.0 rows of rev4_c1.jsonl (tau 0.851 / 0.5)
        errs = execute(make_tasks(['oracle_label', 'oracle_score_0851'], [1.0], ['mixed'], [4000, 4001], force_0851=True),
                       'oracle_repro', a.workers)
    else:
        tasks = [t for t in make_tasks(['ecmp', 'guarded_score', 'm1_pmix', 'guarded_gnn', 'random'], [0.25, 1.0], C.FAM, [4000])
                 if t.get('mseed') in (None, 0)]
        errs = execute(tasks, 'instrument_check_plain', a.workers, fn=_run_plain)
    sys.exit(1 if errs else 0)


if __name__ == '__main__':
    main()
