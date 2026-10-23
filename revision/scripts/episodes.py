from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import sys, json
import numpy as np
sys.path.insert(0, NF_ROOT)
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.telemetry import run_simulation_and_collect
P = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
g = build_fat_tree(); lens = []; byfam = {}
for i in range(30):
    pat = P[i % 6]
    col = run_simulation_and_collect(g, TrafficGenerator(g, seed=42 + i).generate_scenario(150, pat), 150)
    lab = np.array(col.edge_labels_history)
    for e in range(lab.shape[1]):
        run = 0
        for t in range(lab.shape[0]):
            if lab[t, e]:
                run += 1
            elif run:
                lens.append(run); byfam.setdefault(pat, []).append(run); run = 0
        if run:
            lens.append(run); byfam.setdefault(pat, []).append(run)
lens = np.array(lens)
w = lens / lens.sum()
out = dict(n=int(lens.size), median=float(np.median(lens)), mean=float(lens.mean()),
           share_ticks_in_episodes_ge_10=float(lens[lens >= 10].sum() / lens.sum()),
           share_ticks_in_episodes_gt_5=float(lens[lens > 5].sum() / lens.sum()),
           per_family={f: dict(median=float(np.median(v)), mean=float(np.mean(v)), n=len(v)) for f, v in byfam.items()})
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS + '/episodes.json', 'w'), indent=1)
