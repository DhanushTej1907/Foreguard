"""Reproducible research run: grouped training, forecast baselines, paired held-out traffic runs.

Forecast predictors (same split, same metrics, thresholds tuned on validation):
  gnn, lstm_only, logreg, persistence_binary (last label, fixed 0.5), persistence_continuous
  (last congestion score). Metrics are reported overall and onset-only.

  python research_benchmark.py                              # forecast + routing, default _v3/_v4 outputs
  python research_benchmark.py --forecast-only --results-out research/pilot_forecast_v3.json
  python research_benchmark.py --forecast-only --holdout-family all  # train on 5 families, test on the 6th
"""
import argparse, copy, json, os, random, time, platform
from pathlib import Path
import numpy as np
import torch
from config import (MODEL, TRAINING, CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_OUT, RESULTS_OUT, DATA_OUT,
                    check_output_path)
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector, run_simulation_and_collect
from model.dataset import build_datasets, save_datasets
from model.gnn_model import CongestionPredictor
from model.baselines import LSTMOnly, PerLinkLogReg
from model.metrics import forecast_metrics, tune_threshold, bootstrap_ap_difference
from model.train import train_model, predict_probabilities
from routing.rerouter import ProactiveRerouter, ReactiveRerouter
from routing.guarded import GuardedRerouter

PATTERNS = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
PREDICTORS = ['gnn', 'lstm_only', 'logreg', 'persistence_binary', 'persistence_continuous']
STRATEGIES = ['ecmp', 'reactive', 'persistence', 'gnn', 'guarded_gnn', 'guarded_persistence', 'guarded_score']
NUM_SCENARIOS = 30


def seed_everything(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def generate_samples(graph, ticks):
    """30 independent scenarios, each traffic family represented 5 times."""
    samples, metadata = [], []
    for i in range(NUM_SCENARIOS):
        pattern = PATTERNS[i % len(PATTERNS)]
        flows = TrafficGenerator(graph, seed=42 + i).generate_scenario(total_ticks=ticks, pattern=pattern)
        for s in run_simulation_and_collect(graph, flows, ticks).get_windowed_samples():
            s['scenario_id'] = i; s['pattern'] = pattern
            samples.append(s)
        metadata.append(dict(scenario_id=i, seed=42 + i, pattern=pattern))
    return samples, metadata


def lstm_checkpoint(path):
    return os.path.splitext(path)[0] + '_lstm_only.pt'


def run_forecast(train, val, test, collector, epochs, checkpoint_path, draws):
    """Train/fit every predictor on one split and score it on the test set."""
    edge_pairs = [(collector.switch_to_idx[u], collector.switch_to_idx[v]) for u, v in collector.switch_edges]
    result = dict(splits={name: sorted({s['scenario_id'] for s in ds.samples})
                          for name, ds in [('train', train), ('val', val), ('test', test)]},
                  training_seconds={}, parameters={}, training={}, metrics={})
    val_scores, test_scores, models, thresholds = {}, {}, {}, {}
    for name, factory, path in [
            ('gnn', lambda: CongestionPredictor(num_edges=len(collector.switch_edges), edge_pairs=edge_pairs),
             checkpoint_path),
            ('lstm_only', LSTMOnly, lstm_checkpoint(checkpoint_path))]:
        seed_everything(42)
        model = factory()
        history = train_model(model, train, val, epochs=epochs, checkpoint_path=path, verbose=False)
        result['training_seconds'][name] = history['train_seconds']
        result['parameters'][name] = model.count_parameters()
        result['training'][name] = {k: history[k] for k in
                                    ('checkpoint_path', 'best_epoch', 'best_val_ap', 'val_ap', 'val_f1', 'train_loss', 'val_loss')}
        thresholds[name] = history['alert_threshold']
        val_scores[name] = predict_probabilities(model, val)
        test_scores[name] = predict_probabilities(model, test)
        models[name] = model
        print(f'  trained {name}: best epoch {history["best_epoch"]}, val AP {history["best_val_ap"]:.4f}, '
              f'{history["train_seconds"]:.1f}s', flush=True)
    start = time.perf_counter()
    logreg = PerLinkLogReg().fit(train.stacked('edge_features'), train.stacked('labels'))
    result['training_seconds']['logreg'] = time.perf_counter() - start
    val_scores['logreg'] = logreg.predict_proba(val.stacked('edge_features'))
    test_scores['logreg'] = logreg.predict_proba(test.stacked('edge_features'))
    for name, key in [('persistence_binary', 'last_labels'), ('persistence_continuous', 'last_scores')]:
        val_scores[name], test_scores[name] = val.stacked(key), test.stacked(key)
    thresholds['persistence_binary'] = 0.5
    y_val, y_test, last_test = val.stacked('labels'), test.stacked('labels'), test.stacked('last_labels')
    for name in PREDICTORS:
        if name not in thresholds:
            thresholds[name] = tune_threshold(y_val, val_scores[name])
        result['metrics'][name] = forecast_metrics(y_test, test_scores[name], thresholds[name], last_test,
                                                   is_probability=name != 'persistence_continuous')
    groups = np.array([s['scenario_id'] for s in test.samples])
    result['bootstrap_gnn_minus_persistence_continuous'] = bootstrap_ap_difference(
        y_test, test_scores['gnn'], test_scores['persistence_continuous'], last_test, groups, draws)
    return result, models['gnn'], thresholds


def predict(model, collector):
    w = MODEL['window_size']
    with torch.no_grad():
        return model(torch.tensor(np.stack(collector.node_features_history[-w:])).unsqueeze(0),
                     torch.tensor(np.stack(collector.edge_features_history[-w:])).unsqueeze(0),
                     torch.tensor(collector.get_pyg_edge_index())).squeeze(0).numpy()


def run_case(graph, flows, ticks, strategy, model, thresholds):
    engine = NetworkEngine(graph); flows = copy.deepcopy(flows)
    for f in flows: f.current_path = []; f.active = False
    engine.set_flows(flows); c = TelemetryCollector(graph, engine)
    guard_threshold = dict(guarded_gnn=thresholds['gnn'], guarded_persistence=thresholds['persistence_binary'],
                           guarded_score=thresholds['persistence_continuous'])
    controller = (ReactiveRerouter(graph) if strategy == 'reactive' else
                  GuardedRerouter(graph, c.edge_to_idx, threshold=guard_threshold[strategy])
                  if strategy in guard_threshold else ProactiveRerouter(graph, c.edge_to_idx))
    inference = []
    for _ in range(ticks):
        c.collect_tick(engine.step())
        if strategy == 'reactive': controller.evaluate_and_reroute(flows, engine, engine.tick)
        elif strategy != 'ecmp' and engine.tick >= MODEL['window_size']:
            if strategy in ('persistence', 'guarded_persistence'):
                p = c.edge_labels_history[-1]
            elif strategy == 'guarded_score':
                p = c.edge_scores_history[-1]
            else:
                start = time.perf_counter(); p = predict(model, c); inference.append((time.perf_counter() - start) * 1000)
            controller.evaluate_and_reroute(flows, p, engine, engine.tick)
    result = engine.metrics()
    result['reroutes'] = len(controller.reroute_log)
    result['inference_ms_p50'] = float(np.median(inference)) if inference else None
    result['inference_ms_p99'] = float(np.percentile(inference, 99)) if inference else None
    if isinstance(controller, GuardedRerouter): result['rejections'] = dict(controller.rejections)
    return result


def summarize_routing(runs, strategies=STRATEGIES, seed=42):
    """Per-strategy means (incl. host/fabric/unroutable drops), reductions vs ECMP, paired bootstrap."""
    grouped = {}
    for r in runs:
        grouped.setdefault((r['pattern'], r['seed']), {})[r['strategy']] = r
    matched = [v for v in grouped.values() if all(s in v for s in strategies)]

    def mean(s, key):
        values = [v[s][key] for v in matched if v[s][key] is not None]
        return float(np.mean(values)) if values else None

    keys = dict(mean_total_drops='total_drops', mean_host_link_drops='host_link_drops',
                mean_fabric_link_drops='fabric_link_drops', mean_unroutable_drops='unroutable_drops',
                mean_delivered_packets_per_tick='avg_throughput', mean_delivered_latency_us='delivered_mean_latency_us',
                mean_run_p99_latency_us='delivered_p99_latency_us', mean_reroutes_per_run='reroutes')
    overall = {s: {name: mean(s, key) for name, key in keys.items()} for s in strategies}
    reduction = lambda s, k: (1 - overall[s][k] / overall['ecmp'][k]) * 100 if overall['ecmp'][k] else None
    vs_ecmp = {}
    for s in strategies:
        if s == 'ecmp':
            continue
        d = [v[s]['total_drops'] - v['ecmp']['total_drops'] for v in matched]
        vs_ecmp[s] = dict(matched_trials=len(matched), wins=int(sum(x < 0 for x in d)),
                          ties=int(sum(x == 0 for x in d)), losses=int(sum(x > 0 for x in d)),
                          total_drop_reduction_percent=reduction(s, 'mean_total_drops'),
                          fabric_drop_reduction_percent=reduction(s, 'mean_fabric_link_drops'),
                          host_drop_reduction_percent=reduction(s, 'mean_host_link_drops'))
    rng = np.random.default_rng(seed); paired = []
    for pattern in dict.fromkeys(r['pattern'] for r in runs):
        rows = [v for (p, _), v in sorted(grouped.items()) if p == pattern and v in matched]
        for s in strategies[1:]:
            diffs = np.array([v[s]['total_drops'] - v['ecmp']['total_drops'] for v in rows])
            if not len(diffs):
                continue
            means = np.mean(rng.choice(diffs, size=(2000, len(diffs)), replace=True), axis=1)
            paired.append(dict(pattern=pattern, strategy=s, mean_drop_difference=float(diffs.mean()),
                               paired_bootstrap_95_percent_interval=[float(v) for v in np.percentile(means, [2.5, 97.5])]))
    return dict(overall_summary=overall, vs_ecmp=vs_ecmp, paired_summary=paired)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--ticks', type=int, default=150); parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--forecast-only', action='store_true', help='skip routing runs')
    parser.add_argument('--holdout-family', choices=PATTERNS + ['all'],
                        help='train on the other 5 traffic families, test on this one (requires --forecast-only)')
    parser.add_argument('--checkpoint-out', default=CHECKPOINT_OUT)
    parser.add_argument('--results-out', default=RESULTS_OUT)
    parser.add_argument('--data-out', default=DATA_OUT, help='saved only for the standard split')
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    args = parser.parse_args()
    if args.ticks < 50 or args.seeds < 1 or args.epochs < 1: parser.error('ticks >= 50, seeds >= 1, epochs >= 1')
    if args.holdout_family and not args.forecast_only: parser.error('--holdout-family requires --forecast-only')
    for path in (args.checkpoint_out, lstm_checkpoint(args.checkpoint_out), args.results_out, args.data_out):
        check_output_path(path)
    torch.set_num_threads(2); seed_everything(42)
    TRAINING['batch_size'] = 32
    graph = build_fat_tree()
    collector = TelemetryCollector(graph, NetworkEngine(graph))
    edge_index = collector.get_pyg_edge_index()
    samples, metadata = generate_samples(graph, args.ticks)
    # schema_version here is the results-JSON schema, not config.CHECKPOINT_SCHEMA_VERSION.
    output = dict(schema_version=4, checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
                  label_mode=MODEL['label_mode'], settings=vars(args),
                  python=platform.python_version(), torch=torch.__version__,
                  scope='synthetic full-duplex batch simulator; no PFC/RDMA; no real-time deployment claim',
                  reproducibility={'global_seed': 42, 'model_training_seed': 42, 'routing_seed_start': 3000,
                                   'bootstrap_seed': 42},
                  scenarios=metadata)
    if args.holdout_family:
        output['holdout'] = {}
        stem, ext = os.path.splitext(args.checkpoint_out)
        for family in (PATTERNS if args.holdout_family == 'all' else [args.holdout_family]):
            print(f'Holdout family: {family}', flush=True)
            train, val, test = build_datasets(samples, edge_index, holdout_family=family)
            output['holdout'][family], _, _ = run_forecast(train, val, test, collector, args.epochs,
                                                           f'{stem}_holdout_{family}{ext}', args.bootstrap_draws)
    else:
        train, val, test = build_datasets(samples, edge_index)
        save_datasets(train, val, test, args.data_out)
        output['forecast'], model, thresholds = run_forecast(train, val, test, collector, args.epochs,
                                                             args.checkpoint_out, args.bootstrap_draws)
        if not args.forecast_only:
            output['runs'] = []
            for pattern in PATTERNS:
                for seed in range(3000, 3000 + args.seeds):
                    flows = TrafficGenerator(graph, seed=seed).generate_scenario(total_ticks=args.ticks, pattern=pattern)
                    for strategy in STRATEGIES:
                        result = run_case(graph, flows, args.ticks, strategy, model, thresholds)
                        assert result['packet_conservation'] and result['drop_split_ok']
                        output['runs'].append(dict(pattern=pattern, seed=seed, strategy=strategy, **result))
                    print(f'Completed {pattern}, seed {seed}', flush=True)
            output.update(summarize_routing(output['runs']))
    path = Path(args.results_out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False,
                               default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf-8')
    print(f'Wrote {path}')


if __name__ == '__main__': main()
