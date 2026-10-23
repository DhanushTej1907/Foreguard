from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os, sys
import numpy as np
ROOT = NF_ROOT; sys.path.insert(0, ROOT); sys.path.insert(0, SCRIPTS)
import torch; torch.set_num_threads(1)
from scipy.special import expit, logit
from simulation.topology import build_fat_tree
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from model.dataset import build_datasets
from model.train import predict_probabilities
from model.metrics import tune_threshold
import research_benchmark as RB, nf_exp as X
fa = json.load(open(RESULTS + '/forecast_analysis.json'))
RB.seed_everything(42); g = build_fat_tree(); col = TelemetryCollector(g, NetworkEngine(g))
samples, meta = RB.generate_samples(g, 150)
train, val, test = build_datasets(samples, col.get_pyg_edge_index(), split='stratified')
X.load_models(kinds=('gnn',), threads=1)
out = {}
for k in (0, 1, 2):
    pr = fa['models']['gnn'][k]['platt']; a, b = pr['a'], pr['b']
    raw_v = predict_probabilities(X._MODELS[('gnn', k)]['model'].model, val)
    pv = expit(a * logit(np.clip(raw_v, 1e-6, 1 - 1e-6)) + b)
    out[str(k)] = dict(a=a, b=b, threshold=tune_threshold(val.stacked('labels'), pv), temp_threshold=X._MODELS[('gnn', k)]['threshold'])
    print(k, out[str(k)])
json.dump(out, open(RESULTS + '/platt_params.json', 'w'), indent=1)
