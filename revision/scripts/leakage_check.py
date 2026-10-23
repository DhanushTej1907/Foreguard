"""Structural similarity of forecasting scenarios within each traffic family (stride leakage check)."""
import json, os, sys
import numpy as np
ROOT = sys.argv[1]; sys.path.insert(0, ROOT)
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.telemetry import run_simulation_and_collect
from simulation.network_engine import NetworkEngine

PATTERNS = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
SPLIT = {'train': [0, 1, 2, 3, 4, 5, 7, 9, 14, 15, 16, 18, 23, 24, 25, 26, 28, 29], 'val': [6, 8, 10, 13, 17, 21], 'test': [11, 12, 19, 20, 22, 27]}
role = {i: k for k, v in SPLIT.items() for i in v}
g = build_fat_tree(); eng = NetworkEngine(g)
info = {}
for i in range(30):
    pat = PATTERNS[i % 6]
    flows = TrafficGenerator(g, seed=42 + i).generate_scenario(total_ticks=150, pattern=pat)
    keys = {(f.src, f.dst) for f in flows}
    paths = {(f.src, f.dst, tuple(eng.get_ecmp_path(f))) for f in flows}
    col = run_simulation_and_collect(g, flows, 150)
    lab = np.array(col.edge_labels_history)  # [T, 32]
    info[i] = dict(pattern=pat, role=role[i], pairs=keys, paths=paths, lab=lab, nflows=len(flows))
jac = lambda a, b: len(a & b) / len(a | b) if a | b else 1.0
out = {}
for pat in PATTERNS:
    ids = [i for i in info if info[i]['pattern'] == pat]
    test = [i for i in ids if info[i]['role'] == 'test'][0]
    train = [i for i in ids if info[i]['role'] == 'train']
    pj = [jac(info[test]['pairs'], info[j]['pairs']) for j in train]
    pathj = [jac(info[test]['paths'], info[j]['paths']) for j in train]
    rate_t = info[test]['lab'].mean(0)
    corr = [float(np.corrcoef(rate_t, info[j]['lab'].mean(0))[0, 1]) if rate_t.std() and info[j]['lab'].mean(0).std() else float('nan') for j in train]
    # exact label-sequence agreement on ticks where both exist
    agree = [float((info[test]['lab'] == info[j]['lab']).mean()) for j in train]
    out[pat] = dict(test=test, train=train, nflows=[info[i]['nflows'] for i in ids],
                    pair_jaccard=pj, ecmp_path_jaccard=pathj, per_link_congestion_rate_corr=corr,
                    label_agreement=agree, test_pos_rate=float(info[test]['lab'].mean()))
    print(f"{pat:14s} test={test} flows={out[pat]['nflows']} pairJ={np.round(pj,2)} pathJ={np.round(pathj,2)} corr={np.round(corr,2)} agree={np.round(agree,3)} pos={out[pat]['test_pos_rate']:.3f}")
# routing seeds: stride flows across 3000-3009
st = [TrafficGenerator(g, seed=s).generate_scenario(total_ticks=150, pattern='stride') for s in range(3000, 3010)]
same = all({(f.flow_id, f.src, f.dst) for f in st[0]} == {(f.flow_id, f.src, f.dst) for f in x} for x in st)
print('stride routing seeds share identical (id,src,dst):', same)
# do stride forecasting scenarios share ids with routing seeds?
fs = TrafficGenerator(g, seed=42 + 3).generate_scenario(total_ticks=150, pattern='stride')
print('stride forecast scenario shares (id,src,dst) with routing seeds:', {(f.flow_id, f.src, f.dst) for f in fs} == {(f.flow_id, f.src, f.dst) for f in st[0]})
json.dump(out, open(sys.argv[2], 'w'), indent=1)
