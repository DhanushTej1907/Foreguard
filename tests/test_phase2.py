import os
import json
import pickle
import tempfile
import unittest
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from config import (MODEL, CHECKPOINT_SCHEMA_VERSION, LEGACY_PATHS, PLOTS_DIR, CHECKPOINT_OUT, RESULTS_OUT,
                    DATA_OUT, check_output_path)
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector, congestion_score, congestion_label, run_simulation_and_collect
from model.dataset import CongestionDataset, save_datasets, load_datasets, check_dataset_schema
from model.gnn_model import check_checkpoint_schema
from model.metrics import tune_threshold, forecast_metrics, bootstrap_ap_difference
from model.baselines import LSTMOnly, PerLinkLogReg
from model.splits import holdout_split

PATTERNS = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']


def scenario_samples(n=4, ticks=60):
    g = build_fat_tree(); samples = []
    for i in range(n):
        flows = TrafficGenerator(g, seed=42 + i).generate_scenario(total_ticks=ticks, pattern=PATTERNS[i % 6])
        c = run_simulation_and_collect(g, flows, ticks)
        for s in c.get_windowed_samples():
            s['scenario_id'] = i; s['pattern'] = PATTERNS[i % 6]; samples.append(s)
    return g, c, samples


class OutputPathTests(unittest.TestCase):
    def test_legacy_paths_refused(self):
        for path in LEGACY_PATHS + [os.path.join(PLOTS_DIR, 'roc_curve.png')]:
            with self.assertRaises(ValueError): check_output_path(path)
        for path in [CHECKPOINT_OUT, RESULTS_OUT, DATA_OUT, os.path.join(PLOTS_DIR, 'v3', 'roc_curve.png')]:
            self.assertEqual(check_output_path(path), path)
        self.assertTrue(CHECKPOINT_OUT.endswith('best_model_v3.pt'))
        self.assertTrue(RESULTS_OUT.endswith(os.path.join('research', 'benchmark_results_v4.json')))


class SchemaTests(unittest.TestCase):
    def test_dataset_roundtrip_and_rejection(self):
        _, c, samples = scenario_samples(1, 30)
        ds = CongestionDataset(samples, c.get_pyg_edge_index())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'd.pkl')
            save_datasets(ds, ds, ds, path)
            self.assertEqual(len(load_datasets(path)[0]), len(ds))
            for bad in [{}, {'schema_version': 2, 'label_mode': MODEL['label_mode']},
                        {'schema_version': CHECKPOINT_SCHEMA_VERSION, 'label_mode': 'offered'}]:
                with open(path, 'wb') as f: pickle.dump(dict(bad, train=ds, val=ds, test=ds), f)
                with self.assertRaises(ValueError): load_datasets(path)

    def test_checkpoint_label_mode(self):
        check_checkpoint_schema({'schema_version': CHECKPOINT_SCHEMA_VERSION, 'label_mode': MODEL['label_mode']})
        with self.assertRaises(ValueError):
            check_checkpoint_schema({'schema_version': CHECKPOINT_SCHEMA_VERSION, 'label_mode': 'offered'})
        self.assertEqual(MODEL['label_mode'], 'served_or_queued')


class ScoreAndFeatureTests(unittest.TestCase):
    def test_score_reproduces_label_exactly(self):
        cap, buf, T, f = 500, 2000, MODEL['congestion_threshold'], MODEL['queue_label_fraction']
        for out in range(0, cap + 1, 5):
            for queue in range(0, buf + 1, 10):
                data = dict(utilization=0., packets_out=out, queue_depth=queue)
                s = congestion_score(data, cap, buf, 'served_or_queued')
                self.assertTrue(0 <= s <= 1)
                self.assertEqual(s > T, out / cap > T or queue > f * buf, (out, queue))
                self.assertEqual(congestion_label(data, cap, buf, 'served_or_queued'), float(s > T))
        for util in np.linspace(0, 2, 81):
            self.assertEqual(congestion_score(dict(utilization=util), cap, buf, 'offered') > T, util > T)

    def test_all_features_in_unit_interval(self):
        g = build_fat_tree()
        for pattern in ['hotspot', 'mixed']:
            flows = TrafficGenerator(g, seed=3000).generate_scenario(total_ticks=80, pattern=pattern)
            c = run_simulation_and_collect(g, flows, 80)
            for name in ['node_features_history', 'edge_features_history', 'edge_scores_history']:
                values = np.stack(getattr(c, name))
                self.assertGreaterEqual(values.min(), 0, (pattern, name))
                self.assertLessEqual(values.max(), 1, (pattern, name))
            # Feature 3 is served utilization; labels agree with scores.
            edge = np.stack(c.edge_features_history)
            self.assertGreater(edge[..., 3].max(), 0)
            np.testing.assert_array_equal(np.stack(c.edge_labels_history),
                                          (np.stack(c.edge_scores_history) > MODEL['congestion_threshold']).astype(np.float32))


class MetricTests(unittest.TestCase):
    def test_tune_threshold_max_f1(self):
        y = np.array([0, 0, 1, 1, 0, 1]); s = np.array([.1, .2, .6, .8, .7, .9])
        thr = tune_threshold(y, s)
        best = max(forecast_metrics(y, s, t, np.zeros(6))['overall']['f1'] for t in np.linspace(0, 1, 101))
        self.assertAlmostEqual(forecast_metrics(y, s, thr, np.zeros(6))['overall']['f1'], best)
        self.assertTrue(.2 < thr < .6)        # all of {.6,.7,.8,.9} predicted positive, ties never split
        self.assertEqual(tune_threshold(np.zeros(3), np.ones(3)), .5)

    def test_forecast_metrics_and_onset(self):
        y = np.array([[1, 0], [1, 1], [0, 0]]); s = np.array([[.9, .1], [.8, .2], [.3, .6]])
        last = np.array([[1, 0], [0, 0], [0, 1]])
        m = forecast_metrics(y, s, .5, last)
        o = m['overall']
        self.assertEqual((o['n'], o['positives']), (6, 3))
        self.assertAlmostEqual(o['precision'], 2 / 3); self.assertAlmostEqual(o['recall'], 2 / 3)
        self.assertAlmostEqual(o['average_precision'], average_precision_score(y.ravel(), s.ravel()))
        on = m['onset']     # last==0: (0,1),(1,0),(1,1),(2,0) -> y 0,1,1,0 ; s .1,.8,.2,.3
        self.assertEqual((on['n'], on['positives']), (4, 2))
        self.assertAlmostEqual(on['precision'], 1.); self.assertAlmostEqual(on['recall'], .5)
        # Undefined values are None, not 0.
        none = forecast_metrics(np.zeros(4), np.zeros(4), .5, np.zeros(4))['overall']
        self.assertIsNone(none['precision']); self.assertIsNone(none['recall'])
        self.assertIsNone(none['average_precision']); self.assertIsNone(none['roc_auc'])

    def test_bootstrap(self):
        rng = np.random.default_rng(0)
        y = (rng.random((40, 8)) < .3).astype(float); groups = np.repeat(np.arange(8), 5)
        last = (rng.random((40, 8)) < .5).astype(float)
        same = bootstrap_ap_difference(y, y * .5 + .2, y * .5 + .2, last, groups, draws=50)
        self.assertEqual(same['overall']['point'], 0); self.assertEqual(same['overall']['ci95'], [0, 0])
        better = bootstrap_ap_difference(y, y, rng.random(y.shape), last, groups, draws=50)
        for part in ['overall', 'onset']:
            self.assertGreater(better[part]['ci95'][0], 0)
            self.assertEqual(better[part]['scenarios'], 8); self.assertLessEqual(better[part]['valid_draws'], 50)


class BaselineTests(unittest.TestCase):
    def test_lstm_only_shape(self):
        m = LSTMOnly()
        out = m(torch.zeros(3, 10, 20, 4), torch.rand(3, 10, 32, 4), None)
        self.assertEqual(tuple(out.shape), (3, 32)); self.assertTrue(((out > 0) & (out < 1)).all())

    def test_per_link_logreg(self):
        rng = np.random.default_rng(1)
        x = rng.random((200, 10, 3, 4)).astype(np.float32)
        y = np.zeros((200, 3)); y[:, 0] = x[:, -1, 0, 0] > .5; y[:, 1] = x[:, -1, 1, 3] > .5   # link 2 constant
        p = PerLinkLogReg().fit(x, y).predict_proba(x)
        self.assertEqual(p.shape, (200, 3))
        self.assertGreater(average_precision_score(y[:, 0], p[:, 0]), .9)
        np.testing.assert_array_equal(p[:, 2], 0.)


class SplitTests(unittest.TestCase):
    def test_holdout_split(self):
        samples = [dict(scenario_id=i, pattern=PATTERNS[i % 6], tick=t) for i in range(30) for t in range(3)]
        for family in PATTERNS:
            train, val, test = holdout_split(samples, family)
            self.assertEqual({s['pattern'] for s in test}, {family})
            self.assertEqual(len({s['scenario_id'] for s in test}), 5)
            self.assertEqual(sorted(s['pattern'] for s in val if s['tick'] == 0), sorted(set(PATTERNS) - {family}))
            self.assertNotIn(family, {s['pattern'] for s in train + val})
            ids = [{s['scenario_id'] for s in part} for part in (train, val, test)]
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        with self.assertRaises(ValueError): holdout_split(samples, 'nope')


class TrainingAndBenchmarkTests(unittest.TestCase):
    def test_train_selects_by_val_ap_and_stores_threshold(self):
        from model.train import train_model
        _, c, samples = scenario_samples(3, 50)
        ds = CongestionDataset(samples, c.get_pyg_edge_index())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'm.pt')
            torch.manual_seed(0)
            h = train_model(LSTMOnly(), ds, ds, epochs=3, verbose=False, checkpoint_path=path)
            ckpt = torch.load(path, weights_only=True)
            self.assertEqual(ckpt['epoch'], int(np.argmax(h['val_ap'])) + 1)
            self.assertAlmostEqual(ckpt['val_ap'], max(h['val_ap']))
            self.assertEqual(ckpt['alert_threshold'], h['alert_threshold'])
            check_checkpoint_schema(ckpt)
            self.assertGreater(h['train_seconds'], 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'm.history.json')))
        with self.assertRaises(ValueError):
            train_model(LSTMOnly(), ds, ds, epochs=1, verbose=False, checkpoint_path=LEGACY_PATHS[0])

    def test_run_forecast_smoke(self):
        import research_benchmark as rb
        from model.dataset import build_datasets
        g, c, samples = scenario_samples(4, 50)
        train, val, test = build_datasets(samples, c.get_pyg_edge_index())
        with tempfile.TemporaryDirectory() as tmp:
            result, models, thresholds = rb.run_forecast(train, val, test, c, 1, os.path.join(tmp, 'g.pt'), 20,
                                                         train_seeds=2)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'g_seed1_lstm_only.pt')))
        self.assertEqual(set(result['metrics']), set(rb.PREDICTORS))
        self.assertEqual(set(result['training_seconds']), {'gnn', 'lstm_only', 'logreg'})
        self.assertEqual(len(result['training_seconds']['gnn']), 2)
        self.assertEqual(thresholds['persistence_binary'], .5)
        self.assertEqual(len(models['lstm_only']), 2); self.assertEqual(len(thresholds['gnn']), 2)
        for name, m in result['metrics'].items():
            self.assertEqual(len(m['per_seed']), 2 if name in rb.LEARNED else 1)
            self.assertEqual(set(m['per_seed'][0]) - {'brier_uncalibrated'},
                             {'threshold', 'brier_is_probability', 'overall', 'onset'})
            self.assertIn('mean', m['summary']['overall']['positive_rate'])
        self.assertIn('brier_uncalibrated', result['metrics']['gnn']['per_seed'][0])
        self.assertFalse(result['metrics']['persistence_continuous']['summary']['brier_is_probability'])
        for key in ['gnn_minus_persistence_continuous', 'gnn_minus_lstm_only']:
            self.assertIn('ci95', result['bootstrap'][key]['onset'])
        json.dumps(result, allow_nan=False, default=lambda o: o.item())

    def test_summarize_routing(self):
        import research_benchmark as rb
        runs = []
        for seed in (1, 2):
            for s in rb.STRATEGIES:
                guarded = s.startswith('guarded')
                runs.append(dict(pattern='p', seed=seed, strategy=s, total_drops=100 if not guarded else 70,
                                 host_link_drops=60, fabric_link_drops=40 if not guarded else 10, unroutable_drops=0,
                                 avg_throughput=1., delivered_mean_latency_us=1., delivered_p99_latency_us=None,
                                 reroutes=seed))
        out = rb.summarize_routing(runs)
        g = out['vs_ecmp']['guarded_gnn']
        self.assertAlmostEqual(g['fabric_drop_reduction_percent'], 75.)
        self.assertAlmostEqual(g['total_drop_reduction_percent'], 30.)
        self.assertAlmostEqual(g['host_drop_reduction_percent'], 0.)
        self.assertEqual((g['wins'], g['ties'], g['losses']), (2, 0, 0))
        self.assertEqual(out['vs_ecmp']['reactive']['ties'], 2)
        self.assertEqual(out['overall_summary']['ecmp']['mean_fabric_link_drops'], 40.)
        self.assertIsNone(out['overall_summary']['ecmp']['mean_run_p99_latency_us'])


if __name__ == '__main__': unittest.main()
