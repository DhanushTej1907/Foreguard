"""Rebuild data/datasets_v3.pkl (gitignored) without retraining.

Regenerates the stratified dataset exactly as research_benchmark.py does (global seed 42, 150 ticks) and
checks that the test scenarios match research/benchmark_results_v4.json, so the stored checkpoint
checkpoints/best_model_v3.pt evaluates on the same split it was trained for.

Usage (from the repo root):  python tools/make_dataset.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import research_benchmark as RB
from config import DATA_OUT
from model.dataset import build_datasets, save_datasets
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from simulation.topology import build_fat_tree

RB.seed_everything(42)
graph = build_fat_tree()
collector = TelemetryCollector(graph, NetworkEngine(graph))
print('Generating 30 scenarios (about a minute)...', flush=True)
samples, _ = RB.generate_samples(graph, 150)
train, val, test = build_datasets(samples, collector.get_pyg_edge_index(), split='stratified')

bench = json.load(open(os.path.join(ROOT, 'research', 'benchmark_results_v4.json')))
expected = bench['forecast']['splits']['test']
found = sorted({s['scenario_id'] for s in test.samples})
assert found == expected, f'test split {found} differs from the benchmark {expected}'

save_datasets(train, val, test, DATA_OUT)
print(f'Saved {DATA_OUT}: {len(train)} train / {len(val)} val / {len(test)} test samples (split matches the benchmark)')
