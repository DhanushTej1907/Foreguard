"""Concern 3 mixed-load dataset + integrity checks B5 -> data/datasets_c3_mixedload.pkl, revision/results/c3/data.json.

The 30 benchmark scenarios at the 8 sweep loads (rates scaled exactly as rev3_load.py). Every window keeps its base
scenario_id, so the stratified split (which depends only on {scenario_id: pattern} and Random(42)) puts every load
copy of scenario i in the same partition. Exit 1 if any check fails (the pickle is then not written)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c3_common as C
import numpy as np

from model.dataset import build_datasets, save_datasets
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from simulation.topology import build_fat_tree
import c3_gate


def main():
    t0 = time.time()
    g = build_fat_tree()
    col = TelemetryCollector(g, NetworkEngine(g))
    samples = C.generate_mixed_samples(g)
    train, val, test = build_datasets(samples, col.get_pyg_edge_index(), split='stratified')
    bench = json.load(open(os.path.join(C.ROOT, 'research', 'benchmark_results_v4.json')))['forecast']['splits']
    rep = dict(meta=C.meta(step='data'), checks={}, counts={})
    ck = rep['checks']
    parts = dict(train=train, val=val, test=test)
    # one partition per base scenario, equal to the benchmark lists at every load
    where = {}
    for n, ds in parts.items():
        for s in ds.samples:
            where.setdefault(s['scenario_id'], set()).add(n)
    ck['one_partition_per_scenario'] = all(len(v) == 1 for v in where.values()) and len(where) == 30
    ck['partitions_equal_benchmark_at_every_load'] = all(
        sorted({s['scenario_id'] for s in ds.samples if s['load'] == lf}) == bench[n] for n, ds in parts.items() for lf in C.LOADS)
    # window counts: 136 windows per scenario run
    ck['window_counts'] = (len(samples) == 136 * 30 * len(C.LOADS) and len(train) == 136 * 18 * 8
                           and len(val) == 136 * 6 * 8 and len(test) == 136 * 6 * 8)
    # load-1.0 slice bitwise equal to the original rebuild (also checks order within each partition)
    _, _, ref = c3_gate.rebuild()
    eq = {}
    for (n, ds), r in zip(parts.items(), ref):
        sl = [{k: v for k, v in s.items() if k != 'load'} for s in ds.samples if s['load'] == 1.0]
        eq[n] = c3_gate.same_samples(sl, r.samples)
    ck['load1_slice_bitwise_equal'] = all(v[0] for v in eq.values())
    ck['load1_slice_detail'] = {n: v[1] for n, v in eq.items()}
    # positive counts per load x partition
    for n, ds in parts.items():
        rep['counts'][n] = {}
        for lf in C.LOADS:
            sub = [s for s in ds.samples if s['load'] == lf]
            y = np.stack([s['labels'] for s in sub])
            pos_scen = sorted({s['scenario_id'] for s in sub if s['labels'].sum() > 0})
            rep['counts'][n][str(lf)] = dict(windows=len(sub), link_windows=int(y.size), positives=int(y.sum()),
                                             positive_rate=float(y.mean()), scenarios_with_positives=len(pos_scen))
    rep['ok'] = all(v for k, v in ck.items() if k != 'load1_slice_detail')
    print(json.dumps(ck, indent=1))
    if rep['ok']:
        C.guard_output(C.DATA)
        save_datasets(train, val, test, C.DATA)
        rep['pickle'] = dict(path=C.DATA, sha256=C.sha256(C.DATA), bytes=os.path.getsize(C.DATA))
    rep['seconds'] = time.time() - t0
    C.write_json(os.path.join(C.OUT, 'data.json'), rep)
    for n in parts:
        print(n, {lf: (v['positives'], v['scenarios_with_positives']) for lf, v in rep['counts'][n].items()})
    sys.exit(0 if rep['ok'] else 1)


if __name__ == '__main__':
    main()
