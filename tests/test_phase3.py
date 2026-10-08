import os
import tempfile
import unittest
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from config import TRAINING
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from model.baselines import LSTMOnly
from model.calibration import fit_temperature, apply_temperature, TemperatureScaled
from model.metrics import bootstrap_ap_difference
from model.splits import stratified_split
import research_benchmark as rb

PATTERNS = rb.PATTERNS


class TrainingConfigTests(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(TRAINING['epochs'], 50)
        self.assertEqual(TRAINING['early_stop_patience'], 10)
        self.assertLess(TRAINING['lr_patience'], TRAINING['early_stop_patience'])


class StratifiedSplitTests(unittest.TestCase):
    def test_every_family_in_every_split(self):
        samples = [dict(scenario_id=i, pattern=PATTERNS[i % 6], tick=t) for i in range(30) for t in range(2)]
        parts = stratified_split(samples)
        ids = [{s['scenario_id'] for s in p} for p in parts]
        self.assertEqual([len(x) for x in ids], [18, 6, 6])
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for p, n in zip(parts, (3, 1, 1)):
            for fam in PATTERNS:
                self.assertEqual(len({s['scenario_id'] for s in p if s['pattern'] == fam}), n)
        self.assertEqual(ids, [{s['scenario_id'] for s in p} for p in stratified_split(samples)])  # deterministic


class CalibrationTests(unittest.TestCase):
    def test_temperature_fixes_overconfidence(self):
        rng = np.random.default_rng(0)
        true_p = rng.random(5000); y = (rng.random(5000) < true_p).astype(float)
        logit = np.log(true_p / (1 - true_p)) * 3          # overconfident by a factor of 3
        raw = 1 / (1 + np.exp(-logit))
        t = fit_temperature(raw, y)
        self.assertAlmostEqual(t, 3, delta=.4)
        cal = apply_temperature(raw, t)
        self.assertLess(np.mean((cal - y) ** 2), np.mean((raw - y) ** 2))
        # Monotonic: ranking metrics unchanged.
        self.assertAlmostEqual(average_precision_score(y, cal), average_precision_score(y, raw), places=6)
        self.assertAlmostEqual(roc_auc_score(y, cal), roc_auc_score(y, raw), places=6)

    def test_wrapper_matches_numpy(self):
        m = LSTMOnly().eval(); x = torch.rand(2, 10, 32, 4)
        with torch.no_grad():
            raw = m(None, x).numpy(); wrapped = TemperatureScaled(m, 2.).eval()(None, x).numpy()
        np.testing.assert_allclose(wrapped, apply_temperature(raw, 2.), rtol=1e-4, atol=1e-6)
        self.assertEqual(TemperatureScaled(m, 2.).count_parameters(), m.count_parameters())

    def test_train_model_stores_calibration(self):
        from model.train import train_model
        from model.dataset import CongestionDataset
        from test_phase2 import scenario_samples
        _, c, samples = scenario_samples(2, 50)
        ds = CongestionDataset(samples, c.get_pyg_edge_index())
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(0)
            h = train_model(LSTMOnly(), ds, ds, epochs=2, verbose=False, checkpoint_path=os.path.join(tmp, 'm.pt'))
            ckpt = torch.load(os.path.join(tmp, 'm.pt'), weights_only=True)
        for key in ('temperature', 'alert_threshold', 'val_brier_raw', 'val_brier_calibrated'):
            self.assertEqual(ckpt[key], h[key])
        self.assertLessEqual(h['val_brier_calibrated'], h['val_brier_raw'] + 1e-9)
        self.assertEqual(h['epochs_run'], 2)


class AggregationTests(unittest.TestCase):
    def test_aggregate(self):
        out = rb.aggregate([dict(a=1., b=None, c=dict(d=2.), e=True), dict(a=3., b=None, c=dict(d=4.), e=True)])
        self.assertEqual(out['a'], dict(mean=2., std=float(np.std([1, 3], ddof=1))))
        self.assertIsNone(out['b']); self.assertEqual(out['c']['d']['mean'], 3.); self.assertIs(out['e'], True)
        self.assertEqual(rb.aggregate([5.])['std'], 0.)

    def test_bootstrap_seed_lists(self):
        rng = np.random.default_rng(2)
        y = (rng.random((30, 4)) < .3).astype(float); last = np.zeros_like(y); groups = np.repeat(np.arange(6), 5)
        s1, s2, b = rng.random(y.shape), rng.random(y.shape), rng.random(y.shape)
        single = bootstrap_ap_difference(y, s1, b, last, groups, draws=30)
        listed = bootstrap_ap_difference(y, [s1], [b], last, groups, draws=30)
        self.assertEqual(single, listed)
        avg = bootstrap_ap_difference(y, [s1, s2], b, last, groups, draws=30)
        expected = np.mean([average_precision_score(y.ravel(), s.ravel()) for s in (s1, s2)]) \
            - average_precision_score(y.ravel(), b.ravel())
        self.assertAlmostEqual(avg['overall']['point'], expected)


class RoutingTests(unittest.TestCase):
    def make_runs(self):
        runs = []
        for seed in range(4):
            for st in rb.STRATEGIES:
                for k in ([42, 43] if st in rb.MODEL_STRATEGIES else [None]):
                    drops = {'ecmp': 100, 'guarded_gnn': 60 + (k == 43) * 20}.get(st, 90)
                    runs.append(dict(pattern='p', seed=seed, strategy=st, train_seed=k, total_drops=drops,
                                     host_link_drops=50, fabric_link_drops=drops - 50, unroutable_drops=0,
                                     avg_throughput=1., delivered_mean_latency_us=1., delivered_p99_latency_us=2.,
                                     reroutes=1))
        return runs

    def test_seed_average_and_spread(self):
        runs = self.make_runs()
        avg = rb.seed_average_runs(runs)
        self.assertEqual(len(avg), 4 * len(rb.STRATEGIES))
        gg = [r for r in avg if r['strategy'] == 'guarded_gnn']
        self.assertTrue(all(r['total_drops'] == 70 and r['train_seeds'] == 2 for r in gg))
        spread = rb.routing_seed_spread(runs)['guarded_gnn']['total_drops']
        self.assertEqual(spread['per_seed'], [60., 80.]); self.assertEqual(spread['mean'], 70.)

    def test_paired_bootstrap(self):
        avg = rb.seed_average_runs(self.make_runs())
        p = rb.paired_bootstrap(avg, 'guarded_gnn', 'guarded_persistence', 'total_drops', draws=100)
        self.assertEqual(p['pairs'], 4); self.assertEqual(p['mean_difference'], -20.)
        self.assertEqual(p['ci95'], [-20., -20.]); self.assertAlmostEqual(p['relative_difference_percent'], -20 / 90 * 100)
        self.assertEqual((p['wins'], p['ties'], p['losses']), (4, 0, 0))
        f = rb.paired_bootstrap(avg, 'guarded_gnn', 'ecmp', 'fabric_link_drops', draws=100)
        self.assertEqual(f['mean_difference'], -30.)

    def test_guarded_lstm_run_case(self):
        g = build_fat_tree(); c = TelemetryCollector(g, NetworkEngine(g))
        flows = TrafficGenerator(g, seed=3000).generate_scenario(total_ticks=20, pattern='hotspot')
        model = TemperatureScaled(LSTMOnly().eval(), 1.5).eval()
        th = dict(gnn=.5, lstm_only=.01, persistence_binary=.5, persistence_continuous=.85)
        result, inference = rb.run_case(g, flows, 20, 'guarded_lstm', model, th)
        self.assertTrue(result['packet_conservation'] and result['drop_split_ok'])
        self.assertEqual(len(inference), 20 - 10 + 1)
        self.assertIsNotNone(result['inference_ms_p99'])
        timing = rb.timing_summary(inference)
        self.assertEqual(timing['horizon_ms'], 5.); self.assertEqual(timing['n'], len(inference))


if __name__ == '__main__': unittest.main()
