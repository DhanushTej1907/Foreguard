"""Concern 3 calibration protocols -> revision/results/c3/calibration.json.

P1.0 = temperature + alert threshold fitted on the 6 validation base scenarios at load 1.0 (the original val set);
Pmix = the same 6 scenarios pooled over the 8 loads. Same functions as model/train.py (predict_probabilities with
batch 64, fit_temperature, tune_threshold on the calibrated probabilities). Score: tune_threshold on val last_scores.
Each signal gets exactly one fit per protocol. No test data is read.
Natural protocols (M0/P1.0, M0'/P1.0, M1/Pmix) are recomputed and must equal the values stored by training."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c3_common as C
import torch

from model.dataset import load_datasets
from model.gnn_model import CongestionPredictor, load_compatible_state_dict, check_checkpoint_schema
from model.train import predict_probabilities
from model.calibration import fit_temperature, apply_temperature
from model.metrics import tune_threshold
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from simulation.topology import build_fat_tree

MODELS = {'M0': os.path.join(C.ROOT, 'checkpoints', 'best_model_v3'), 'M0p': C.STEM_M0P, 'M1': C.STEM_M1}
NATURAL = {'M0': 'P1.0', 'M0p': 'P1.0', 'M1': 'Pmix'}


def fit(raw, y):
    T = fit_temperature(raw, y)
    return float(T), float(tune_threshold(y, apply_temperature(raw, T)))


def main():
    torch.set_num_threads(2)
    _, val, _ = load_datasets(C.DATA)
    vsets = {'P1.0': C.load_subset(val, 1.0), 'Pmix': val}
    g = build_fat_tree()
    col = TelemetryCollector(g, NetworkEngine(g))
    edge_pairs = [(col.switch_to_idx[u], col.switch_to_idx[v]) for u, v in col.switch_edges]
    rep = dict(meta=C.meta(step='calibrate'), val_windows={p: len(v) for p, v in vsets.items()}, models={}, score={})
    ok = True
    for name, stem in MODELS.items():
        rep['models'][name] = {}
        for k in C.TRAIN_SEEDS:
            path = stem + ('' if k == 0 else f'_seed{k}') + '.pt'
            ck = torch.load(path, map_location='cpu', weights_only=False)
            check_checkpoint_schema(ck)
            m = CongestionPredictor(num_edges=len(col.switch_edges), edge_pairs=edge_pairs)
            load_compatible_state_dict(m, ck['model_state_dict'])
            m.eval()
            e = {}
            for p, vs in vsets.items():
                T, th = fit(predict_probabilities(m, vs), vs.stacked('labels'))
                e[p] = dict(temperature=T, threshold=th)
            nat = e[NATURAL[name]]
            e['natural'] = NATURAL[name]
            e['natural_matches_checkpoint'] = (abs(nat['temperature'] - ck['temperature']) <= 1e-6 * max(1, ck['temperature'])
                                               and abs(nat['threshold'] - ck['alert_threshold']) <= 1e-6)
            e['checkpoint'] = dict(temperature=float(ck['temperature']), threshold=float(ck['alert_threshold']))
            # the natural protocol always uses the values stored by training
            e[NATURAL[name]] = dict(e['checkpoint'])
            ok &= e['natural_matches_checkpoint']
            rep['models'][name][str(k)] = e
            print(name, 42 + k, {p: (round(e[p]['temperature'], 4), round(e[p]['threshold'], 4)) for p in vsets},
                  'natural ok' if e['natural_matches_checkpoint'] else 'NATURAL MISMATCH', flush=True)
    for p, vs in vsets.items():
        rep['score'][p] = float(tune_threshold(vs.stacked('labels'), vs.stacked('last_scores')))
    rep['score']['P1.0_used'] = C.SCORE_TH_P10          # the rounded value every revision script uses
    rep['score']['Pmix_used'] = rep['score']['Pmix']
    rep['score_P10_matches_benchmark'] = rep['score']['P1.0'] == 0.8510000109672546
    ok &= rep['score_P10_matches_benchmark']
    rep['ok'] = bool(ok)
    print('score', rep['score'])
    C.write_json(os.path.join(C.OUT, 'calibration.json'), rep)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
