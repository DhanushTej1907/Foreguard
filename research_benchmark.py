"""Reproducible research run: grouped training, forecast baselines, paired held-out traffic runs.

Forecast predictors (same split, same metrics, thresholds tuned on validation):
  gnn, lstm_only (each trained with --train-seeds seeds, temperature-scaled on validation),
  logreg, persistence_binary (last label, fixed 0.5), persistence_continuous (last congestion
  score). Metrics are reported overall and onset-only, as mean +- std over training seeds.

Routing: model strategies (gnn, guarded_gnn, guarded_lstm) run once per training seed;
non-model strategies run once. Summaries and paired bootstraps use seed-averaged drops.

  python research_benchmark.py                              # forecast + routing, default _v3/_v4 outputs
  python research_benchmark.py --forecast-only --results-out research/pilot_forecast_v3.json
  python research_benchmark.py --forecast-only --holdout-family all  # train on 5 families, test on the 6th
"""
import argparse, copy, json, os, random, time, platform
from pathlib import Path
import numpy as np
import torch
from config import (MODEL, TRAINING, SIMULATION, CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_OUT, RESULTS_OUT, DATA_OUT,
                    check_output_path)
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector, run_simulation_and_collect
from model.dataset import build_datasets, save_datasets
from model.gnn_model import CongestionPredictor
from model.baselines import LSTMOnly, PerLinkLogReg
from model.calibration import TemperatureScaled, apply_temperature
from model.metrics import forecast_metrics, tune_threshold, bootstrap_ap_difference
from model.train import train_model, predict_probabilities
from routing.rerouter import ProactiveRerouter, ReactiveRerouter
from routing.guarded import GuardedRerouter

PATTERNS = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
PREDICTORS = ['gnn', 'lstm_only', 'logreg', 'persistence_binary', 'persistence_continuous']
LEARNED = ['gnn', 'lstm_only']
STRATEGIES = ['ecmp', 'reactive', 'gnn', 'guarded_gnn', 'guarded_lstm', 'guarded_persistence', 'guarded_score']
MODEL_STRATEGIES = {'gnn': 'gnn', 'guarded_gnn': 'gnn', 'guarded_lstm': 'lstm_only'}
GUARD_COMPARISONS = ['guarded_persistence', 'guarded_score', 'guarded_lstm', 'ecmp']
ROUTING_KEYS = ['total_drops', 'host_link_drops', 'fabric_link_drops', 'unroutable_drops', 'avg_throughput',
                'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes']
NUM_SCENARIOS = 30
SEED_BASE = 42


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


def seed_checkpoint(path, k):
    """Training seed k > 0 gets its own checkpoint; seed 0 uses `path` itself."""
    stem, ext = os.path.splitext(path)
    return path if k == 0 else f'{stem}_seed{k}{ext}'


def aggregate(values):
    """Field-wise mean/std (sample std, ddof=1) over a list of (nested) metric dicts.
    Leaves that are None or non-numeric in any entry stay as-is if identical, else None."""
    first = values[0]
    if isinstance(first, dict):
        return {k: aggregate([v[k] for v in values]) for k in first}
    if isinstance(first, bool) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return first if all(v == first for v in values) else None
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def run_forecast(train, val, test, collector, epochs, checkpoint_path, draws, train_seeds=1):
    """Train/fit every predictor on one split and score it on the test set.

    Returns (result, models, thresholds): models[name][k] is the temperature-scaled model of
    training seed k and thresholds[name][k] its calibrated validation threshold.
    """
    edge_pairs = [(collector.switch_to_idx[u], collector.switch_to_idx[v]) for u, v in collector.switch_edges]
    factories = dict(gnn=lambda: CongestionPredictor(num_edges=len(collector.switch_edges), edge_pairs=edge_pairs),
                     lstm_only=LSTMOnly)
    y_val, y_test, last_test = val.stacked('labels'), test.stacked('labels'), test.stacked('last_labels')
    result = dict(splits={name: sorted({s['scenario_id'] for s in ds.samples})
                          for name, ds in [('train', train), ('val', val), ('test', test)]},
                  training_seeds=[SEED_BASE + k for k in range(train_seeds)],
                  training_seconds={}, parameters={}, training={n: [] for n in LEARNED}, metrics={})
    per_seed = {n: [] for n in PREDICTORS}
    scores = {n: [] for n in LEARNED}
    models = {n: [] for n in LEARNED}
    thresholds = {n: [] for n in LEARNED}
    for k in range(train_seeds):
        for name in LEARNED:
            path = seed_checkpoint(checkpoint_path, k)
            path = lstm_checkpoint(path) if name == 'lstm_only' else path
            seed_everything(SEED_BASE + k)
            model = factories[name]()
            h = train_model(model, train, val, epochs=epochs, checkpoint_path=path, verbose=False)
            model.eval()
            raw = predict_probabilities(model, test)
            cal = apply_temperature(raw, h['temperature'])
            m = forecast_metrics(y_test, cal, h['alert_threshold'], last_test)
            raw_m = forecast_metrics(y_test, raw, h['alert_threshold'], last_test)
            m['brier_uncalibrated'] = dict(overall=raw_m['overall']['brier'], onset=raw_m['onset']['brier'])
            per_seed[name].append(m); scores[name].append(cal)
            models[name].append(TemperatureScaled(model, h['temperature']).eval())
            thresholds[name].append(h['alert_threshold'])
            result['parameters'][name] = model.count_parameters()
            result['training_seconds'].setdefault(name, []).append(h['train_seconds'])
            result['training'][name].append(dict(seed=SEED_BASE + k, **{key: h[key] for key in (
                'checkpoint_path', 'best_epoch', 'epochs_run', 'best_val_ap', 'temperature', 'alert_threshold',
                'val_brier_raw', 'val_brier_calibrated', 'train_seconds', 'val_ap')}))
            print(f'  seed {SEED_BASE + k} {name}: best epoch {h["best_epoch"]}/{h["epochs_run"]}, '
                  f'val AP {h["best_val_ap"]:.4f}, T {h["temperature"]:.3f}, {h["train_seconds"]:.1f}s', flush=True)
    start = time.perf_counter()
    logreg = PerLinkLogReg().fit(train.stacked('edge_features'), train.stacked('labels'))
    result['training_seconds']['logreg'] = [time.perf_counter() - start]
    fixed_val = dict(logreg=logreg.predict_proba(val.stacked('edge_features')),
                     persistence_binary=val.stacked('last_labels'), persistence_continuous=val.stacked('last_scores'))
    fixed_test = dict(logreg=logreg.predict_proba(test.stacked('edge_features')),
                      persistence_binary=test.stacked('last_labels'), persistence_continuous=test.stacked('last_scores'))
    fixed_thresholds = {}
    for name in fixed_test:
        fixed_thresholds[name] = 0.5 if name == 'persistence_binary' else tune_threshold(y_val, fixed_val[name])
        per_seed[name].append(forecast_metrics(y_test, fixed_test[name], fixed_thresholds[name], last_test,
                                               is_probability=name != 'persistence_continuous'))
    for name in PREDICTORS:
        result['metrics'][name] = dict(per_seed=per_seed[name], summary=aggregate(per_seed[name]))
    groups = np.array([s['scenario_id'] for s in test.samples])
    result['bootstrap'] = {
        'gnn_minus_persistence_continuous': bootstrap_ap_difference(
            y_test, scores['gnn'], fixed_test['persistence_continuous'], last_test, groups, draws),
        'gnn_minus_lstm_only': bootstrap_ap_difference(y_test, scores['gnn'], scores['lstm_only'], last_test, groups, draws),
    }
    thresholds.update(fixed_thresholds)
    return result, models, thresholds


def predict(model, collector):
    w = MODEL['window_size']
    with torch.no_grad():
        return model(torch.tensor(np.stack(collector.node_features_history[-w:])).unsqueeze(0),
                     torch.tensor(np.stack(collector.edge_features_history[-w:])).unsqueeze(0),
                     torch.tensor(collector.get_pyg_edge_index())).squeeze(0).numpy()


def run_case(graph, flows, ticks, strategy, model, thresholds):
    """One routing run. thresholds holds the alert thresholds for this run's models
    ('gnn', 'lstm_only') and the guard-only baselines. Returns (metrics, per-tick inference ms)."""
    engine = NetworkEngine(graph); flows = copy.deepcopy(flows)
    for f in flows: f.current_path = []; f.active = False
    engine.set_flows(flows); c = TelemetryCollector(graph, engine)
    guard_threshold = dict(guarded_gnn=thresholds.get('gnn'), guarded_lstm=thresholds.get('lstm_only'),
                           guarded_persistence=thresholds['persistence_binary'],
                           guarded_score=thresholds['persistence_continuous'])
    controller = (ReactiveRerouter(graph) if strategy == 'reactive' else
                  GuardedRerouter(graph, c.edge_to_idx, threshold=guard_threshold[strategy])
                  if strategy in guard_threshold else ProactiveRerouter(graph, c.edge_to_idx))
    inference = []
    for _ in range(ticks):
        c.collect_tick(engine.step())
        if strategy == 'reactive': controller.evaluate_and_reroute(flows, engine, engine.tick)
        elif strategy != 'ecmp' and engine.tick >= MODEL['window_size']:
            if strategy == 'guarded_persistence':
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
    return result, inference


def seed_average_runs(runs):
    """One run per (pattern, traffic seed, strategy): numeric fields averaged over training seeds."""
    groups = {}
    for r in runs:
        groups.setdefault((r['pattern'], r['seed'], r['strategy']), []).append(r)
    out = []
    for (pattern, seed, strategy), rs in groups.items():
        avg = dict(pattern=pattern, seed=seed, strategy=strategy, train_seeds=len(rs))
        for key in ROUTING_KEYS:
            values = [r[key] for r in rs]
            avg[key] = None if any(v is None for v in values) else float(np.mean(values))
        out.append(avg)
    return out


def routing_seed_spread(runs):
    """For model strategies: per training seed, the mean drops over all traffic runs; then mean/std."""
    out = {}
    for strategy in MODEL_STRATEGIES:
        rs = [r for r in runs if r['strategy'] == strategy]
        seeds = sorted({r['train_seed'] for r in rs})
        out[strategy] = {}
        for key in ['total_drops', 'fabric_link_drops', 'host_link_drops', 'reroutes']:
            per_seed = [float(np.mean([r[key] for r in rs if r['train_seed'] == s])) for s in seeds]
            out[strategy][key] = dict(per_seed=per_seed, **aggregate(per_seed))
    return out


def paired_bootstrap(avg_runs, a, b, key, draws=2000, seed=42):
    """Bootstrap over (pattern, traffic seed) pairs of mean(a - b) for `key` (seed-averaged runs)."""
    by = {(r['pattern'], r['seed'], r['strategy']): r for r in avg_runs}
    pairs = sorted({(p, s) for p, s, st in by if st == a and (p, s, b) in by})
    diffs = np.array([by[(p, s, a)][key] - by[(p, s, b)][key] for p, s in pairs])
    mean_b = float(np.mean([by[(p, s, b)][key] for p, s in pairs]))
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(draws, len(diffs)))].mean(axis=1)
    return dict(a=a, b=b, key=key, pairs=len(pairs), mean_difference=float(diffs.mean()),
                ci95=[float(v) for v in np.percentile(means, [2.5, 97.5])],
                mean_a=float(np.mean([by[(p, s, a)][key] for p, s in pairs])), mean_b=mean_b,
                relative_difference_percent=float(diffs.mean() / mean_b * 100) if mean_b else None,
                wins=int((diffs < 0).sum()), ties=int((diffs == 0).sum()), losses=int((diffs > 0).sum()))


def timing_summary(values):
    horizon_ms = MODEL['prediction_horizon'] * SIMULATION['tick_duration_ms']
    v = np.asarray(values, dtype=float)
    return dict(n=int(v.size), p50_ms=float(np.median(v)), p99_ms=float(np.percentile(v, 99)),
                mean_ms=float(v.mean()), max_ms=float(v.max()), horizon_ms=float(horizon_ms),
                share_over_horizon=float((v > horizon_ms).mean()))


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


def run_routing(graph, ticks, traffic_seeds, models, thresholds, train_seeds):
    runs, timings = [], {n: [] for n in LEARNED}
    for pattern in PATTERNS:
        for seed in range(3000, 3000 + traffic_seeds):
            flows = TrafficGenerator(graph, seed=seed).generate_scenario(total_ticks=ticks, pattern=pattern)
            for strategy in STRATEGIES:
                model_name = MODEL_STRATEGIES.get(strategy)
                for k in (range(train_seeds) if model_name else [None]):
                    th = dict(persistence_binary=thresholds['persistence_binary'],
                              persistence_continuous=thresholds['persistence_continuous'])
                    if k is not None:
                        th.update({n: thresholds[n][k] for n in LEARNED})
                    result, inference = run_case(graph, flows, ticks, strategy,
                                                 models[model_name][k] if model_name else None, th)
                    assert result['packet_conservation'] and result['drop_split_ok']
                    if model_name: timings[model_name].extend(inference)
                    runs.append(dict(pattern=pattern, seed=seed, strategy=strategy,
                                     train_seed=None if k is None else SEED_BASE + k, **result))
            print(f'Completed {pattern}, seed {seed}', flush=True)
    avg = seed_average_runs(runs)
    summary = summarize_routing(avg)
    summary['routing_seed_spread'] = routing_seed_spread(runs)
    summary['guarded_gnn_paired'] = [paired_bootstrap(avg, 'guarded_gnn', b, key)
                                     for b in GUARD_COMPARISONS for key in ('total_drops', 'fabric_link_drops')]
    summary['inference_timing'] = {n: timing_summary(v) for n, v in timings.items() if v}
    return dict(runs=runs, **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAINING['epochs'])
    parser.add_argument('--ticks', type=int, default=150)
    parser.add_argument('--seeds', type=int, default=10, help='traffic seeds per family (3000, 3001, ...)')
    parser.add_argument('--train-seeds', type=int, default=3, help='training seeds for gnn and lstm_only')
    parser.add_argument('--split', choices=['stratified', 'random'], default='stratified')
    parser.add_argument('--forecast-only', action='store_true', help='skip routing runs')
    parser.add_argument('--holdout-family', choices=PATTERNS + ['all'],
                        help='train on the other 5 traffic families, test on this one (requires --forecast-only)')
    parser.add_argument('--checkpoint-out', default=CHECKPOINT_OUT)
    parser.add_argument('--results-out', default=RESULTS_OUT)
    parser.add_argument('--data-out', default=DATA_OUT, help='saved only for the standard split')
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    args = parser.parse_args()
    if args.ticks < 50 or args.seeds < 1 or args.epochs < 1 or args.train_seeds < 1:
        parser.error('ticks >= 50, seeds >= 1, epochs >= 1, train-seeds >= 1')
    if args.holdout_family and not args.forecast_only: parser.error('--holdout-family requires --forecast-only')
    for path in (args.checkpoint_out, lstm_checkpoint(args.checkpoint_out), args.results_out, args.data_out):
        check_output_path(path)
    started = time.perf_counter()
    torch.set_num_threads(2); seed_everything(42)
    TRAINING['batch_size'] = 32
    graph = build_fat_tree()
    collector = TelemetryCollector(graph, NetworkEngine(graph))
    edge_index = collector.get_pyg_edge_index()
    samples, metadata = generate_samples(graph, args.ticks)
    # schema_version here is the results-JSON schema, not config.CHECKPOINT_SCHEMA_VERSION.
    output = dict(schema_version=4, checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
                  label_mode=MODEL['label_mode'], settings=vars(args),
                  training_config={k: TRAINING[k] for k in ('epochs', 'lr_patience', 'early_stop_patience',
                                                            'learning_rate', 'pos_weight')},
                  python=platform.python_version(), torch=torch.__version__,
                  scope='synthetic full-duplex batch simulator; no PFC/RDMA; no real-time deployment claim',
                  reproducibility={'global_seed': 42, 'training_seeds': [SEED_BASE + k for k in range(args.train_seeds)],
                                   'routing_seed_start': 3000, 'bootstrap_seed': 42,
                                   'routing_models': 'model strategies run once per training seed; '
                                                     'non-model strategies run once'},
                  scenarios=metadata)
    if args.holdout_family:
        output['holdout'] = {}
        stem, ext = os.path.splitext(args.checkpoint_out)
        for family in (PATTERNS if args.holdout_family == 'all' else [args.holdout_family]):
            print(f'Holdout family: {family}', flush=True)
            train, val, test = build_datasets(samples, edge_index, holdout_family=family)
            output['holdout'][family], _, _ = run_forecast(train, val, test, collector, args.epochs,
                                                           f'{stem}_holdout_{family}{ext}', args.bootstrap_draws,
                                                           args.train_seeds)
    else:
        train, val, test = build_datasets(samples, edge_index, split=args.split)
        save_datasets(train, val, test, args.data_out)
        output['forecast'], models, thresholds = run_forecast(train, val, test, collector, args.epochs,
                                                              args.checkpoint_out, args.bootstrap_draws,
                                                              args.train_seeds)
        output['forecast_seconds'] = time.perf_counter() - started
        if not args.forecast_only:
            output.update(run_routing(graph, args.ticks, args.seeds, models, thresholds, args.train_seeds))
    output['wall_clock_seconds'] = time.perf_counter() - started
    path = Path(args.results_out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False,
                               default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf-8')
    print(f'Wrote {path} after {output["wall_clock_seconds"] / 60:.1f} min')


if __name__ == '__main__': main()
