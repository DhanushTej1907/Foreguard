"""Concern 3 training: M0' (load 1.0 only) and M1 (all 8 loads) with the benchmark procedure, unchanged.

  python revision/scripts/c3_train.py load1      # M0' -> checkpoints/c3/best_model_c3_load1[_seedK].pt
  python revision/scripts/c3_train.py mixed      # M1  -> checkpoints/c3/best_model_c3_mixed[_seedK].pt

Same as research_benchmark.run_forecast for the GNN: torch threads 2, batch 32, seed_everything(42 + k) right before
the model is built, model.train.train_model (max 50 epochs, early stop 10 on pooled val AP, temperature + alert
threshold fitted on the group's validation set). M0' uses the load-1.0 slice of the mixed dataset, which data.json
shows is bitwise equal to the original training data. For M0' the run doubles as reproduction check I1 (G3):
per-epoch val AP, best epoch and weights against the stored best_model_v3 checkpoints (informational only).
-> revision/results/c3/train_<group>.json"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c3_common as C
import numpy as np
import torch

import research_benchmark as RB
from config import TRAINING
from model.dataset import load_datasets
from model.gnn_model import CongestionPredictor, check_checkpoint_schema
from model.train import train_model, predict_probabilities
from model.calibration import apply_temperature
from sklearn.metrics import average_precision_score
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from simulation.topology import build_fat_tree


def stem_for(group):
    return {'load1': C.STEM_M0P, 'mixed': C.STEM_M1}[group]


def repro_check(k, h, path, test, bench):
    """I1: compare an M0' run with the stored best_model_v3 checkpoint of the same seed (informational)."""
    ref = os.path.join(C.ROOT, 'checkpoints', 'best_model_v3' + ('' if k == 0 else f'_seed{k}'))
    hist = json.load(open(ref + '.history.json'))
    new_ck = torch.load(path, map_location='cpu', weights_only=False)
    old_ck = torch.load(ref + '.pt', map_location='cpu', weights_only=False)
    diffs = [float((new_ck['model_state_dict'][n].float() - old_ck['model_state_dict'][n].float()).abs().max())
             for n in old_ck['model_state_dict']]
    a, b = h['val_ap'], hist['val_ap']
    n = min(len(a), len(b))
    first_div = next((i + 1 for i in range(n) if a[i] != b[i]), None)
    return dict(seed=42 + k, best_epoch=[h['best_epoch'], hist['best_epoch']], epochs_run=[h['epochs_run'], hist['epochs_run']],
                best_val_ap=[h['best_val_ap'], hist['best_val_ap']], temperature=[h['temperature'], hist['temperature']],
                alert_threshold=[h['alert_threshold'], hist['alert_threshold']],
                val_ap_curve_identical=a == b, first_differing_epoch=first_div,
                max_abs_val_ap_diff=float(max(abs(x - y) for x, y in zip(a[:n], b[:n]))) if n else None,
                weights_identical=max(diffs) == 0.0, weights_max_abs_diff=max(diffs),
                test_ap=[h.get('test_ap'), bench['metrics']['gnn']['per_seed'][k]['overall']['average_precision']])


def main():
    group = sys.argv[1]
    assert group in ('load1', 'mixed')
    torch.set_num_threads(2)
    TRAINING['batch_size'] = 32
    os.makedirs(C.CKDIR, exist_ok=True)
    data_sha = C.sha256(C.DATA)
    train, val, test = load_datasets(C.DATA)
    if group == 'load1':
        train, val, test = (C.load_subset(d, 1.0) for d in (train, val, test))
    loads = sorted({s['load'] for s in train.samples})
    g = build_fat_tree()
    col = TelemetryCollector(g, NetworkEngine(g))
    edge_pairs = [(col.switch_to_idx[u], col.switch_to_idx[v]) for u, v in col.switch_edges]
    bench = json.load(open(os.path.join(C.ROOT, 'research', 'benchmark_results_v4.json')))['forecast']
    rep = dict(meta=C.meta(step='train', group=group), group=group, train_loads=loads, data_sha256=data_sha,
               n_train=len(train), n_val=len(val), batch_size=TRAINING['batch_size'], runs=[], repro=[])
    out_json = os.path.join(C.OUT, f'train_{group}.json')
    C.guard_output(out_json)
    for k in C.TRAIN_SEEDS:
        path = stem_for(group) + ('' if k == 0 else f'_seed{k}') + '.pt'
        C.guard_output(path)
        if os.path.exists(path):
            raise SystemExit(f'{path} exists: no reruns over existing results')
        RB.seed_everything(RB.SEED_BASE + k)
        model = CongestionPredictor(num_edges=len(col.switch_edges), edge_pairs=edge_pairs)
        t0 = time.time()
        h = train_model(model, train, val, epochs=TRAINING['epochs'], checkpoint_path=path, verbose=False)
        wall = time.time() - t0
        model.eval()
        y_test = test.stacked('labels')
        h['test_ap'] = float(average_precision_score(y_test.ravel(), apply_temperature(predict_probabilities(model, test), h['temperature']).ravel()))
        steps = h['epochs_run'] * math.ceil(len(train) / TRAINING['batch_size'])
        ck = torch.load(path, map_location='cpu', weights_only=False)
        check_checkpoint_schema(ck)          # B7: schema 3 and label_mode
        ck['c3'] = dict(group=group, train_loads=[float(x) for x in loads], data_sha256=data_sha, n_train=len(train),
                        n_val=len(val), gradient_steps=int(steps), seed=RB.SEED_BASE + k)
        torch.save(ck, path)
        run = dict(seed=RB.SEED_BASE + k, checkpoint=os.path.relpath(path, C.ROOT), best_epoch=h['best_epoch'],
                   epochs_run=h['epochs_run'], best_val_ap=h['best_val_ap'], temperature=h['temperature'],
                   alert_threshold=h['alert_threshold'], val_brier_raw=h['val_brier_raw'],
                   val_brier_calibrated=h['val_brier_calibrated'], gradient_steps=int(steps),
                   batches_per_epoch=math.ceil(len(train) / TRAINING['batch_size']), train_seconds=h['train_seconds'],
                   wall_seconds=wall, sha256=C.sha256(path), test_ap_info_only=h['test_ap'])
        rep['runs'].append(run)
        print(f'{group} seed {42 + k}: best epoch {h["best_epoch"]}/{h["epochs_run"]}, val AP {h["best_val_ap"]:.4f}, '
              f'T {h["temperature"]:.3f}, tau {h["alert_threshold"]:.4f}, {wall:.0f}s, {steps} steps', flush=True)
        if group == 'load1':
            r = repro_check(k, h, path, test, bench)
            rep['repro'].append(r)
            print('  I1:', {x: r[x] for x in ('best_epoch', 'val_ap_curve_identical', 'first_differing_epoch', 'weights_identical')}, flush=True)
        C.write_json(out_json, rep)
    # B7 sanity (not performance)
    rep['B7_ok'] = all(np.isfinite(r['temperature']) and np.isfinite(r['alert_threshold']) and r['best_val_ap'] > 0
                       for r in rep['runs'])
    C.write_json(out_json, rep)
    sys.exit(0 if rep['B7_ok'] else 1)


if __name__ == '__main__':
    main()
