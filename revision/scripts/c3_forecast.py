"""Concern 3 open-loop forecast evaluation (threshold-free core test of finding (a)) -> revision/results/c3/forecast.json

Test sets (ECMP telemetry, no controller):
  unseen     traffic seeds 4000-4009 x 6 families (60 base scenarios, the routing sweep's traffic) -- primary
  benchmark  the 6 benchmark test base scenarios (test partition of the mixed dataset) -- secondary
at each of the 8 loads, plus all loads pooled (base scenarios resampled with all their loads).

Per load: positives, base scenarios with positives, AP / onset AP / Brier per training seed (M0, M0', M1) and for
continuous persistence; Brier and F1 under both protocols. Scenario bootstrap of AP differences (2,000 draws, seed 42)
with exactly the draws and statistic of model.metrics.bootstrap_ap_difference (AP averaged over training seeds in
every draw; draws without positives skipped), computed with a sort-once weighted AP (tests/test_c3.py checks the
equality). Flags: AP undefined with 0 positives; unstable with < 100 positives or < 5 base scenarios with positives."""
import json, os, sys, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c3_common as C
import numpy as np

DRAWS, BOOT_SEED = 2000, 42
COMPARE = {'M1-M0p': ('M1', 'M0p'), 'M1-persistence': ('M1', 'persistence'), 'M0p-persistence': ('M0p', 'persistence'),
           'M0p-M0': ('M0p', 'M0'), 'M1-M0': ('M1', 'M0')}
POOLED_COMPARE = ['M1-M0p', 'M1-persistence', 'M0p-persistence']


# ───────────────────────── sort-once weighted AP (scenario bootstrap) ─────────────────────────
class SortedSeries:
    """One score array, pre-sorted as sklearn's _binary_clf_curve does (stable sort, descending)."""

    def __init__(self, s, y, g):
        order = np.argsort(s, kind='mergesort')[::-1]
        ss = s[order]
        self.y = y[order].astype(np.uint8)
        self.g = g[order].astype(np.int32)
        self.ends = np.r_[np.where(np.diff(ss))[0], ss.size - 1]

    def ap(self, w):
        """AP with per-scenario weights w (counts of each scenario in the draw); None if no positive."""
        we = w[self.g]
        wy = we * self.y
        tp = np.cumsum(wy)[self.ends]
        fp = np.cumsum(we - wy)[self.ends]
        if tp[-1] <= 0:
            return None
        ps = tp + fp
        prec = np.divide(tp, ps, out=np.zeros_like(tp), where=ps > 0)
        rec = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def fast_bootstrap(series, y, last, groups, comparisons, draws=DRAWS, seed=BOOT_SEED):
    """series: {name: [score arrays]} (lists averaged in every draw); y, last: [n_windows, n_links];
    groups: [n_windows] base-scenario ids. Same draws as model.metrics.bootstrap_ap_difference."""
    groups = np.asarray(groups)
    ids = np.unique(groups)
    gi = np.searchsorted(ids, groups)                      # scenario index per window
    rng = np.random.default_rng(seed)
    picks = [rng.choice(ids, size=len(ids), replace=True) for _ in range(draws)]
    weights = [np.bincount(np.searchsorted(ids, p), minlength=len(ids)).astype(np.float64) for p in picks]
    out = {}
    for mname, mask in (('overall', np.ones_like(y, dtype=bool)), ('onset', np.asarray(last) == 0)):
        gl = np.broadcast_to(gi[:, None], y.shape)[mask]
        yl = np.asarray(y)[mask]
        need = {n for c in comparisons for n in comparisons[c]}
        srt = {n: [SortedSeries(np.asarray(s, dtype=np.float64)[mask], yl, gl) for s in series[n]] for n in need}

        def stat(w):
            v = {}
            for n, lst in srt.items():
                aps = [x.ap(w) for x in lst]
                if any(a is None for a in aps):
                    return None
                v[n] = float(np.mean(aps))
            return v
        point = stat(np.ones(len(ids)))
        boots = [stat(w) for w in weights]
        boots = [b for b in boots if b is not None]
        out[mname] = {}
        for c, (a, b) in comparisons.items():
            d = [x[a] - x[b] for x in boots]
            out[mname][c] = dict(point=None if point is None else point[a] - point[b],
                                 ci95=[float(v) for v in np.percentile(d, [2.5, 97.5])] if d else None,
                                 valid_draws=len(d), draws=draws, scenarios=int(len(ids)))
    return out


# ───────────────────────────────────────── data ─────────────────────────────────────────
def unseen_windows(load):
    """Open-loop ECMP windows of the 60 unseen base scenarios at one load; groups = base-scenario index."""
    from simulation.telemetry import run_simulation_and_collect
    from simulation.topology import build_fat_tree
    g = build_fat_tree()
    samples, groups = [], []
    for fi, fam in enumerate(C.FAM):
        for s in C.UNSEEN:
            for w in run_simulation_and_collect(g, C.scaled_flows(g, s, fam, load), C.TICKS).get_windowed_samples():
                samples.append(w); groups.append(fi * 100 + s)
    return samples, np.array(groups)


_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from model.gnn_model import CongestionPredictor, load_compatible_state_dict
    from simulation.network_engine import NetworkEngine
    from simulation.telemetry import TelemetryCollector
    from simulation.topology import build_fat_tree
    g = build_fat_tree()
    col = TelemetryCollector(g, NetworkEngine(g))
    edge_pairs = [(col.switch_to_idx[u], col.switch_to_idx[v]) for u, v in col.switch_edges]
    cal = json.load(open(os.path.join(C.OUT, 'calibration.json')))
    stems = {'M0': os.path.join(C.ROOT, 'checkpoints', 'best_model_v3'), 'M0p': C.STEM_M0P, 'M1': C.STEM_M1}
    _W['models'] = {}
    for name, stem in stems.items():
        for k in C.TRAIN_SEEDS:
            ck = torch.load(stem + ('' if k == 0 else f'_seed{k}') + '.pt', map_location='cpu', weights_only=False)
            m = CongestionPredictor(num_edges=len(col.switch_edges), edge_pairs=edge_pairs)
            load_compatible_state_dict(m, ck['model_state_dict'])
            m.eval()
            _W['models'][(name, k)] = m
    _W['cal'] = cal
    _W['edge_index'] = col.get_pyg_edge_index()


def evaluate(job):
    """job = (set name, load or 'pooled'). Returns the per-load block."""
    from model.dataset import CongestionDataset, load_datasets
    from model.train import predict_probabilities
    from model.calibration import apply_temperature
    from model.metrics import forecast_metrics
    t0 = time.time()
    tset, load = job
    loads = C.LOADS if load == 'pooled' else [load]
    samples, groups = [], []
    for lf in loads:
        if tset == 'unseen':
            s, gr = unseen_windows(lf)
        else:
            if 'test' not in _W:
                _W['test'] = load_datasets(C.DATA)[2]
            s = [x for x in _W['test'].samples if x['load'] == lf]
            gr = np.array([x['scenario_id'] for x in s])
        samples += s; groups.append(gr)
    groups = np.concatenate(groups)
    ds = CongestionDataset(samples, _W['edge_index'])
    y, last, last_s = ds.stacked('labels'), ds.stacked('last_labels'), ds.stacked('last_scores')
    pos = int(y.sum())
    scen_pos = int(len({gid for gid, row in zip(groups, y) if row.sum() > 0}))
    onset = last == 0
    onset_pos = int(y[onset].sum())
    onset_scen = int(len({gid for gid, row, o in zip(groups, y, onset) if (row * o).sum() > 0}))
    blk = dict(windows=len(ds), link_windows=int(y.size), positives=pos, positive_rate=float(y.mean()),
               scenarios=int(len(np.unique(groups))), scenarios_with_positives=scen_pos,
               onset_link_windows=int(onset.sum()), onset_positives=onset_pos, onset_scenarios_with_positives=onset_scen,
               ap_undefined=pos == 0, unstable=pos < C.UNSTABLE_MIN_POS or scen_pos < C.UNSTABLE_MIN_SCEN,
               onset_ap_undefined=onset_pos == 0,
               onset_unstable=onset_pos < C.UNSTABLE_MIN_POS or onset_scen < C.UNSTABLE_MIN_SCEN, models={})
    cal = _W['cal']
    series = {}
    for name in ('M0', 'M0p', 'M1'):
        per_seed, cals = [], []
        for k in C.TRAIN_SEEDS:
            raw = predict_probabilities(_W['models'][(name, k)], ds)
            e = cal['models'][name][str(k)]
            r = dict(seed=42 + k, natural=e['natural'])
            for p in ('P1.0', 'Pmix'):
                pc = apply_temperature(raw, e[p]['temperature'])
                r[p] = forecast_metrics(y, pc, e[p]['threshold'], last)
                if p == e['natural']:
                    cals.append(pc)
            per_seed.append(r)
        series[name] = cals
        nat = [r[r['natural']] for r in per_seed]
        summ = lambda f: (lambda v: None if any(x is None for x in v) else dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))))([f(m) for m in nat])
        blk['models'][name] = dict(per_seed=per_seed, ap=summ(lambda m: m['overall']['average_precision']),
                                   onset_ap=summ(lambda m: m['onset']['average_precision']),
                                   brier=summ(lambda m: m['overall']['brier']))
    series['persistence'] = [last_s]
    pm = {p: forecast_metrics(y, last_s, cal['score'][p + '_used'], last, is_probability=False) for p in ('P1.0', 'Pmix')}
    blk['models']['persistence'] = dict(per_protocol=pm, ap=pm['P1.0']['overall']['average_precision'],
                                        onset_ap=pm['P1.0']['onset']['average_precision'], brier=pm['P1.0']['overall']['brier'])
    comps = COMPARE if load != 'pooled' else {c: COMPARE[c] for c in POOLED_COMPARE}
    blk['bootstrap'] = fast_bootstrap(series, y, last, groups, comps) if pos else None
    blk['seconds'] = time.time() - t0
    return tset, str(load), blk


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.time()
    jobs = [('unseen', 'pooled')] + [('unseen', lf) for lf in reversed(C.LOADS)] + \
           [('benchmark', 'pooled')] + [('benchmark', lf) for lf in reversed(C.LOADS)]
    out_path = os.path.join(C.OUT, 'forecast.json')
    C.guard_output(out_path)
    rep = dict(meta=C.meta(step='forecast', workers=workers), draws=DRAWS, bootstrap_seed=BOOT_SEED,
               unstable_rule=dict(min_positives=C.UNSTABLE_MIN_POS, min_scenarios=C.UNSTABLE_MIN_SCEN), sets={})
    with mp.get_context('spawn').Pool(workers, initializer=_init) as pool:
        for tset, load, blk in pool.imap_unordered(evaluate, jobs):
            rep['sets'].setdefault(tset, {})[load] = blk
            m = blk['models']
            ap = lambda n: None if m[n]['ap'] is None else round(m[n]['ap']['mean'], 4)
            print(f'{tset:9s} {load:7s} pos {blk["positives"]:7d} scen {blk["scenarios_with_positives"]:2d} '
                  f'AP M0 {ap("M0")} M0p {ap("M0p")} M1 {ap("M1")} pers {m["persistence"]["ap"]} '
                  f'{blk["seconds"]:.0f}s', flush=True)
    rep['seconds'] = time.time() - t0
    C.write_json(out_path, rep)


if __name__ == '__main__':
    main()
