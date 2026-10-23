"""Forecast-side analyses for the revision: calibration (ECE, reliability, Platt), per-scenario AP,
leakage-robust subsets (without the structurally repeated stride / LLM-training test scenarios).
Regenerates the stratified dataset exactly as research_benchmark.py does and scores the stored
checkpoints; asserts the reproduced test AP matches benchmark_results_v4.json."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os, sys
import numpy as np
ROOT = os.environ.get('NF_ROOT', NF_ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(1)
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.optimize import minimize
from scipy.special import expit, logit
from simulation.topology import build_fat_tree
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from model.dataset import build_datasets
from model.train import predict_probabilities
from model.baselines import PerLinkLogReg
from model.metrics import tune_threshold
import research_benchmark as RB
import nf_exp as X

OUT = sys.argv[1]
bench = json.load(open(os.path.join(ROOT, 'research', 'benchmark_results_v4.json')))
RB.seed_everything(42)
g = build_fat_tree()
col = TelemetryCollector(g, NetworkEngine(g))
samples, meta = RB.generate_samples(g, 150)
train, val, test = build_datasets(samples, col.get_pyg_edge_index(), split='stratified')
assert sorted({s['scenario_id'] for s in test.samples}) == bench['forecast']['splits']['test']
y_val, y_test = val.stacked('labels'), test.stacked('labels')
last_test, last_val = test.stacked('last_labels'), val.stacked('last_labels')
groups = np.array([s['scenario_id'] for s in test.samples])
pattern_of = {m['scenario_id']: m['pattern'] for m in meta}
X.load_models(threads=1)

EPS = 1e-6
def ece(y, p, bins=10, mode='width'):
    y, p = np.ravel(y), np.ravel(p)
    if mode == 'width':
        edges = np.linspace(0, 1, bins + 1)
    else:
        edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = 0, 1
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, bins - 1)
    tot, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            rows.append(None); continue
        conf, acc = p[m].mean(), y[m].mean()
        tot += m.mean() * abs(conf - acc)
        rows.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=int(m.sum()), mean_p=float(conf), frac_pos=float(acc)))
    return float(tot), rows

def platt_fit(p, y):
    z = logit(np.clip(np.ravel(p), EPS, 1 - EPS)); y = np.ravel(y)
    def nll(ab):
        q = np.clip(expit(ab[0] * z + ab[1]), EPS, 1 - EPS)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return minimize(nll, x0=[1.0, 0.0], method='Nelder-Mead').x

def platt_apply(p, ab):
    return expit(ab[0] * logit(np.clip(p, EPS, 1 - EPS)) + ab[1])

res = dict(models={}, persistence={}, per_scenario={}, subsets={})
preds = {}
for kind in ('gnn', 'lstm_only'):
    res['models'][kind] = []
    for k in (0, 1, 2):
        mm = X._MODELS[(kind, k)]
        base = mm['model'].model            # raw (uncalibrated) network
        raw_t = predict_probabilities(base, test); raw_v = predict_probabilities(base, val)
        T = mm['temperature']
        cal_t = expit(logit(np.clip(raw_t, EPS, 1 - EPS)) / T); cal_v = expit(logit(np.clip(raw_v, EPS, 1 - EPS)) / T)
        ab = platt_fit(raw_v, y_val); pl_t = platt_apply(raw_t, ab)
        ap = average_precision_score(y_test.ravel(), cal_t.ravel())
        ref = bench['forecast']['metrics'][kind]['per_seed'][k]['overall']['average_precision']
        assert abs(ap - ref) < 1e-4, (kind, k, ap, ref)
        e_raw, rel_raw = ece(y_test, raw_t); e_cal, rel_cal = ece(y_test, cal_t); e_pl, rel_pl = ece(y_test, pl_t)
        e_cal_m, _ = ece(y_test, cal_t, mode='mass')
        e_raw_m, _ = ece(y_test, raw_t, mode='mass')
        e_pl_m, _ = ece(y_test, pl_t, mode='mass')
        res['models'][kind].append(dict(
            seed=42 + k, temperature=T, ap=ap, platt=dict(a=float(ab[0]), b=float(ab[1])),
            mean_p_raw=float(raw_t.mean()), mean_p_cal=float(cal_t.mean()), mean_p_platt=float(pl_t.mean()),
            pos_rate=float(y_test.mean()),
            ece_raw=e_raw, ece_cal=e_cal, ece_platt=e_pl, ece_mass_raw=e_raw_m, ece_mass_cal=e_cal_m, ece_mass_platt=e_pl_m,
            brier_raw=float(np.mean((raw_t - y_test) ** 2)), brier_cal=float(np.mean((cal_t - y_test) ** 2)),
            brier_platt=float(np.mean((pl_t - y_test) ** 2)),
            rel_raw=rel_raw, rel_cal=rel_cal, rel_platt=rel_pl))
        preds[(kind, k)] = cal_t
        print(f'{kind} seed {42+k}: AP {ap:.4f} T {T:.3f} ECE raw {e_raw:.4f} cal {e_cal:.4f} platt {e_pl:.4f} '
              f'(mass {e_raw_m:.4f}/{e_cal_m:.4f}/{e_pl_m:.4f}) meanp {raw_t.mean():.3f}/{cal_t.mean():.3f} pos {y_test.mean():.3f}', flush=True)
# logistic regression + persistence
logreg = PerLinkLogReg().fit(train.stacked('edge_features'), train.stacked('labels'))
lr_t = logreg.predict_proba(test.stacked('edge_features'))
e_lr, rel_lr = ece(y_test, lr_t)
res['logreg'] = dict(ece=e_lr, rel=rel_lr, brier=float(np.mean((lr_t - y_test) ** 2)))
pc_t = test.stacked('last_scores'); pb_t = last_test
res['persistence']['binary_ece'] = ece(y_test, pb_t)[0]
print('logreg ECE', round(e_lr, 4), 'binary persistence ECE', round(res['persistence']['binary_ece'], 4))

# per-scenario AP (overall and onset)
def ap_on(mask, s):
    yy = y_test[mask].ravel()
    return float(average_precision_score(yy, s[mask].ravel())) if yy.sum() else None
gnn_avg = lambda mask, onset=False: np.mean([ap_on(mask if not onset else mask, preds[('gnn', k)]) for k in range(3)])
for sid in sorted(set(groups)):
    rows = groups == sid
    m = np.zeros_like(y_test, dtype=bool); m[rows] = True
    on = m & (last_test == 0)
    entry = dict(pattern=pattern_of[sid], positives=int(y_test[rows].sum()))
    for name, mask in (('overall', m), ('onset', on)):
        g_ap = [ap_on(mask, preds[('gnn', k)]) for k in range(3)]
        l_ap = [ap_on(mask, preds[('lstm_only', k)]) for k in range(3)]
        entry[name] = dict(gnn=None if None in g_ap else float(np.mean(g_ap)), lstm=None if None in l_ap else float(np.mean(l_ap)),
                           pers_cont=ap_on(mask, pc_t), pers_bin=ap_on(mask, pb_t), logreg=ap_on(mask, lr_t))
    res['per_scenario'][int(sid)] = entry
    o = entry['overall']
    print(f"scenario {sid:2d} {entry['pattern']:13s} pos {entry['positives']:5d} AP gnn {o['gnn']:.3f} lstm {o['lstm']:.3f} pc {o['pers_cont']:.3f} diff {o['gnn']-o['pers_cont']:+.3f} | onset gnn {entry['onset']['gnn'] if entry['onset']['gnn'] is None else round(entry['onset']['gnn'],3)} pc {entry['onset']['pers_cont'] if entry['onset']['pers_cont'] is None else round(entry['onset']['pers_cont'],3)}")

# subsets excluding structurally repeated families
from model.metrics import bootstrap_ap_difference
for label, excl in (('all', set()), ('no_stride', {'stride'}), ('no_stride_no_llm', {'stride', 'llm_training'})):
    keep = np.array([pattern_of[s] not in excl for s in groups])
    yk, lk, gk = y_test[keep], last_test[keep], groups[keep]
    gnn_scores = [preds[('gnn', k)][keep] for k in range(3)]
    lstm_scores = [preds[('lstm_only', k)][keep] for k in range(3)]
    b1 = bootstrap_ap_difference(yk, gnn_scores, pc_t[keep], lk, gk, 2000)
    b2 = bootstrap_ap_difference(yk, gnn_scores, lstm_scores, lk, gk, 2000)
    apg = np.mean([average_precision_score(yk.ravel(), s.ravel()) for s in gnn_scores])
    apl = np.mean([average_precision_score(yk.ravel(), s.ravel()) for s in lstm_scores])
    app = average_precision_score(yk.ravel(), pc_t[keep].ravel())
    on = (lk == 0)
    apg_on = np.mean([average_precision_score(yk[on], s[on]) for s in gnn_scores])
    app_on = average_precision_score(yk[on], pc_t[keep][on])
    res['subsets'][label] = dict(n_scenarios=int(len(set(gk))), ap_gnn=float(apg), ap_lstm=float(apl), ap_pers_cont=float(app),
                                 onset_ap_gnn=float(apg_on), onset_ap_pers_cont=float(app_on),
                                 gnn_minus_pc=b1, gnn_minus_lstm=b2)
    print(label, 'scen', len(set(gk)), f'AP gnn {apg:.3f} lstm {apl:.3f} pc {app:.3f}; diff {b1["overall"]["point"]:+.3f} {np.round(b1["overall"]["ci95"],3)} onset {b1["onset"]["point"]:+.3f} {np.round(b1["onset"]["ci95"],3)}; gnn-lstm {b2["overall"]["point"]:+.4f} {np.round(b2["overall"]["ci95"],3)}')
json.dump(res, open(OUT, 'w'), indent=1, default=float)
np.savez_compressed(OUT.replace('.json', '_preds.npz'), y=y_test, last=last_test, groups=groups,
                    **{f'{k}_{s}': preds[(k, s)] for k, s in preds}, pc=pc_t, lr=lr_t)
print('wrote', OUT)
