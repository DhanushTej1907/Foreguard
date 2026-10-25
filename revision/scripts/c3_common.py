"""Concern 3 (mixed-load retraining): shared constants, load scaling, output-path guard and run metadata.

Protocol: revision/results/c3/PROTOCOL.md. Outputs go only to revision/results/c3/, checkpoints/c3/ and
data/datasets_c3_mixedload.pkl. The paper's artifacts are protected by guard_output()."""
from paths import NF_ROOT  # noqa: F401  (also puts the folder layout in place)
import copy, hashlib, json, os, platform, subprocess, sys, time

ROOT = os.environ.get('NF_ROOT', NF_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LOADS = [0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]          # = rev3_load.LOADS
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
UNSEEN = list(range(4000, 4010))                                 # = rev3_load.SEEDS
TICKS = 150
TRAIN_SEEDS = (0, 1, 2)                                          # training seed 42 + k
OUT = os.path.join(ROOT, 'revision', 'results', 'c3')
CKDIR = os.path.join(ROOT, 'checkpoints', 'c3')
DATA = os.path.join(ROOT, 'data', 'datasets_c3_mixedload.pkl')
STEM_M0P = os.path.join(CKDIR, 'best_model_c3_load1')           # M0': load-1.0 controls, current environment
STEM_M1 = os.path.join(CKDIR, 'best_model_c3_mixed')            # M1: mixed-load models
SCORE_TH_P10 = 0.851                                             # score threshold used by every revision script
UNSTABLE_MIN_POS, UNSTABLE_MIN_SCEN = 100, 5                     # unstable-AP rule (PROTOCOL.md)


def scaled_flows(graph, seed, family, load, ticks=TICKS):
    """Exactly the rev3_load.py procedure: generate, deep-copy, multiply every flow rate by `load`."""
    from simulation.traffic_generator import TrafficGenerator
    flows = copy.deepcopy(TrafficGenerator(graph, seed=seed).generate_scenario(total_ticks=ticks, pattern=family))
    for f in flows:
        f.rate *= load
    return flows


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def guard_output(path):
    """Raise if `path` is a protected artifact; return it otherwise."""
    from config import check_output_path
    check_output_path(path)
    p, root = _norm(path), _norm(ROOT)
    rel = os.path.relpath(p, root)
    parts = rel.split(os.sep)
    bad = ((parts[0] == 'checkpoints' and len(parts) == 2 and parts[1].startswith('best_model_v3'))
           or rel == os.path.normcase(os.path.join('data', 'datasets_v3.pkl'))
           or parts[0] == 'research'
           or (parts[:2] == ['revision', 'results'] and len(parts) == 3))
    if bad:
        raise ValueError(f'refusing to write protected artifact {path}')
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def meta(**extra):
    import numpy as np
    import torch
    try:
        head = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    return dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), git_head=head, python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                torch_threads=torch.get_num_threads(), argv=sys.argv, **extra)


def write_json(path, obj):
    guard_output(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf8') as fh:
        json.dump(obj, fh, indent=1, allow_nan=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    return path


def generate_mixed_samples(graph, loads=LOADS, ticks=TICKS):
    """The 30 benchmark scenarios (seed 42 + i, family PATTERNS[i % 6]) at every load. Every window keeps its base
    scenario_id and pattern and gets a `load` tag; order: scenario, load, tick."""
    from research_benchmark import NUM_SCENARIOS, PATTERNS
    from simulation.telemetry import run_simulation_and_collect
    samples = []
    for i in range(NUM_SCENARIOS):
        pattern = PATTERNS[i % len(PATTERNS)]
        for load in loads:
            flows = scaled_flows(graph, 42 + i, pattern, load, ticks)
            for s in run_simulation_and_collect(graph, flows, ticks).get_windowed_samples():
                s['scenario_id'] = i; s['pattern'] = pattern; s['load'] = load
                samples.append(s)
    return samples


def load_subset(ds, load):
    """CongestionDataset restricted to one load tag (order preserved)."""
    from model.dataset import CongestionDataset
    out = CongestionDataset([s for s in ds.samples if s['load'] == load], ds.edge_index.numpy())
    return out
