"""Concern 3 integrity gates G1 (B3) and G2 (B4) -> revision/results/c3/gate.json. Exit 1 on any failure.

G1  rebuild the load-1.0 dataset in memory exactly as tools/make_dataset.py does (that script is not run: it writes
    data/datasets_v3.pkl) and compare it bitwise with the local data/datasets_v3.pkl (read only).
G2  the rebuilt data is the data the stored checkpoints were trained and validated on: split lists, stored val AP
    (batch 32, as in the training loop), val Brier / temperature / alert threshold (batch 64, as in the post-training
    calibration), test AP against the benchmark JSON, and the score threshold 0.8510000109672546."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c3_common as C
import numpy as np
import torch

from sklearn.metrics import average_precision_score
import research_benchmark as RB
from model.dataset import build_datasets, load_datasets
from model.gnn_model import CongestionPredictor, load_compatible_state_dict, check_checkpoint_schema
from model.baselines import LSTMOnly
from model.train import predict_probabilities
from model.calibration import fit_temperature, apply_temperature
from model.metrics import tune_threshold
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from simulation.topology import build_fat_tree

ARRAYS = ['node_features', 'edge_features', 'labels', 'last_labels', 'last_scores']
SCALARS = ['scenario_id', 'pattern', 'tick', 'feature_start', 'feature_end', 'target_tick']


def rebuild():
    RB.seed_everything(42)
    g = build_fat_tree()
    col = TelemetryCollector(g, NetworkEngine(g))
    samples, _ = RB.generate_samples(g, 150)
    return g, col, build_datasets(samples, col.get_pyg_edge_index(), split='stratified')


def same_samples(a, b):
    """Bitwise equality of two sample lists (order, keys, dtypes, values)."""
    if len(a) != len(b):
        return False, f'length {len(a)} != {len(b)}'
    for i, (x, y) in enumerate(zip(a, b)):
        if set(x) != set(y):
            return False, f'sample {i}: keys {sorted(set(x) ^ set(y))}'
        for k in ARRAYS:
            if x[k].dtype != y[k].dtype or x[k].shape != y[k].shape or not np.array_equal(x[k], y[k]):
                return False, f'sample {i}: {k} differs'
        for k in SCALARS:
            if x[k] != y[k]:
                return False, f'sample {i}: {k} {x[k]!r} != {y[k]!r}'
    return True, 'identical'


def main():
    torch.set_num_threads(2)
    g, col, (train, val, test) = rebuild()
    rep = dict(meta=C.meta(gate='G1+G2'), G1={}, G2={})
    # G1
    old = load_datasets(os.path.join(C.ROOT, 'data', 'datasets_v3.pkl'))
    for name, new, ref in zip(('train', 'val', 'test'), (train, val, test), old):
        ok, why = same_samples(new.samples, ref.samples)
        rep['G1'][name] = dict(ok=ok, detail=why, n=len(new))
    rep['G1']['ok'] = all(rep['G1'][n]['ok'] for n in ('train', 'val', 'test'))
    # G2
    bench = json.load(open(os.path.join(C.ROOT, 'research', 'benchmark_results_v4.json')))
    f = bench['forecast']
    splits = {n: sorted({s['scenario_id'] for s in ds.samples}) for n, ds in (('train', train), ('val', val), ('test', test))}
    rep['G2']['splits'] = dict(ok=splits == f['splits'], rebuilt=splits)
    y_val, y_test = val.stacked('labels'), test.stacked('labels')
    edge_pairs = [(col.switch_to_idx[u], col.switch_to_idx[v]) for u, v in col.switch_edges]
    checks = []
    for kind in ('gnn', 'lstm_only'):
        for k in C.TRAIN_SEEDS:
            path = os.path.join(C.ROOT, 'checkpoints', 'best_model_v3' + ('' if k == 0 else f'_seed{k}') +
                                ('_lstm_only' if kind == 'lstm_only' else '') + '.pt')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            check_checkpoint_schema(ck)
            m = CongestionPredictor(num_edges=len(col.switch_edges), edge_pairs=edge_pairs) if kind == 'gnn' else LSTMOnly()
            load_compatible_state_dict(m, ck['model_state_dict'])
            m.eval()
            raw32 = predict_probabilities(m, val, batch_size=32)
            raw64 = predict_probabilities(m, val)
            val_ap = float(average_precision_score(y_val.ravel(), raw32.ravel()))
            brier = float(np.mean((raw64 - y_val) ** 2))
            T = fit_temperature(raw64, y_val)
            th = tune_threshold(y_val, apply_temperature(raw64, T))
            test_ap = float(average_precision_score(y_test.ravel(), apply_temperature(predict_probabilities(m, test), ck['temperature']).ravel()))
            ref_test = f['metrics'][kind]['per_seed'][k]['overall']['average_precision']
            c = dict(kind=kind, seed=42 + k, val_ap=[val_ap, ck['val_ap']], val_brier_raw=[brier, ck['val_brier_raw']],
                     temperature=[T, ck['temperature']], alert_threshold=[th, ck['alert_threshold']],
                     test_ap=[test_ap, ref_test])
            c['ok'] = (abs(val_ap - ck['val_ap']) <= 1e-6 and abs(brier - ck['val_brier_raw']) <= 1e-6 and
                       abs(T - ck['temperature']) <= 1e-6 * max(1, abs(ck['temperature'])) and
                       abs(th - ck['alert_threshold']) <= 1e-6 and abs(test_ap - ref_test) <= 1e-4)
            checks.append(c)
            print(kind, 42 + k, 'ok' if c['ok'] else 'FAIL', {x: c[x] for x in ('val_ap', 'temperature', 'alert_threshold', 'test_ap')}, flush=True)
    rep['G2']['checkpoints'] = checks
    sc = tune_threshold(y_val, val.stacked('last_scores'))
    ref_sc = f['metrics']['persistence_continuous']['per_seed'][0]['threshold']
    rep['G2']['score_threshold'] = dict(ok=sc == ref_sc, rebuilt=sc, benchmark=ref_sc)
    rep['G2']['ok'] = rep['G2']['splits']['ok'] and all(c['ok'] for c in checks) and rep['G2']['score_threshold']['ok']
    rep['ok'] = rep['G1']['ok'] and rep['G2']['ok']
    C.write_json(os.path.join(C.OUT, 'gate.json'), rep)
    print('G1', rep['G1']['ok'], '| G2', rep['G2']['ok'], '| score th', sc)
    sys.exit(0 if rep['ok'] else 1)


if __name__ == '__main__':
    main()
