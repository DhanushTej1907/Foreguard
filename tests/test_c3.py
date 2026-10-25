"""Concern 3 (mixed-load retraining) helpers: load scaling = rev3_load, one partition per base scenario, protected
outputs, the sort-once bootstrap = model.metrics.bootstrap_ap_difference, instrumentation changes nothing, the passive
oracle's look-ahead alignment, and c3_load task ids / settings = rev3_load for the shared variants."""
import copy, os, random, sys, unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'revision', 'scripts')
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
import c3_common as C  # noqa: E402
import c3_forecast  # noqa: E402
import c3_load  # noqa: E402
import nf_exp  # noqa: E402
from model.metrics import bootstrap_ap_difference  # noqa: E402
from model.splits import stratified_split  # noqa: E402
from simulation.telemetry import run_simulation_and_collect  # noqa: E402
from simulation.topology import build_fat_tree  # noqa: E402
from simulation.traffic_generator import TrafficGenerator  # noqa: E402

G = build_fat_tree()
ATTRS = ['flow_id', 'src', 'dst', 'rate', 'start_tick', 'duration', 'flow_type']


class TestScaling(unittest.TestCase):
    def test_scaled_flows_equal_rev3_load_procedure(self):
        for seed, fam, load in [(4000, 'hotspot', 0.2), (4003, 'mixed', 0.125), (42, 'stride', 0.75), (4009, 'llm_training', 1.0)]:
            ref = copy.deepcopy(TrafficGenerator(G, seed=seed).generate_scenario(total_ticks=150, pattern=fam))
            for f in ref:
                f.rate *= load
            got = C.scaled_flows(G, seed, fam, load)
            self.assertEqual([[getattr(f, a) for a in ATTRS] for f in got], [[getattr(f, a) for a in ATTRS] for f in ref])

    def test_load_one_is_exact(self):
        raw = TrafficGenerator(G, seed=43).generate_scenario(total_ticks=150, pattern='hotspot')
        got = C.scaled_flows(G, 43, 'hotspot', 1.0)
        self.assertEqual([f.rate for f in raw], [f.rate for f in got])

    def test_loads_match_rev3_load(self):
        import rev3_load
        self.assertEqual(C.LOADS, rev3_load.LOADS)
        self.assertEqual(C.UNSEEN, rev3_load.SEEDS)
        self.assertEqual(C.FAM, rev3_load.FAM)


class TestSplit(unittest.TestCase):
    def test_one_partition_per_base_scenario(self):
        fams = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
        base = [dict(scenario_id=i, pattern=fams[i % 6]) for i in range(30)]
        ref = [{s['scenario_id'] for s in p} for p in stratified_split(base)]
        mixed = [dict(scenario_id=i, pattern=fams[i % 6], load=lf) for i in range(30) for lf in C.LOADS]
        parts = stratified_split(mixed)
        for p, r in zip(parts, ref):
            self.assertEqual({s['scenario_id'] for s in p}, r)
            for lf in C.LOADS:
                self.assertEqual({s['scenario_id'] for s in p if s['load'] == lf}, r)
        self.assertEqual(ref[2], {11, 12, 19, 20, 22, 27})       # benchmark test split


class TestGuard(unittest.TestCase):
    def test_protected_paths_refused(self):
        for p in ['checkpoints/best_model_v3.pt', 'checkpoints/best_model_v3_seed2_lstm_only.pt', 'data/datasets_v3.pkl',
                  'research/benchmark_results_v4.json', 'research/new.json', 'revision/results/rev3_load.jsonl',
                  'revision/results/anything_new.json', 'checkpoints/best_model.pt']:
            with self.assertRaises(ValueError, msg=p):
                C.guard_output(os.path.join(ROOT, p))

    def test_new_paths_allowed(self):
        for p in ['revision/results/c3/load.jsonl', 'checkpoints/c3/best_model_c3_mixed_seed1.pt', 'data/datasets_c3_mixedload.pkl']:
            self.assertEqual(C.guard_output(os.path.join(ROOT, p)), os.path.join(ROOT, p))


class TestFastBootstrap(unittest.TestCase):
    def test_equals_bootstrap_ap_difference(self):
        rng = np.random.default_rng(0)
        n_scen, per, links = 7, 12, 5
        groups = np.repeat(np.arange(n_scen) * 3 + 100, per)
        y = (rng.random((n_scen * per, links)) < 0.15).astype(float)
        y[groups == 103] = 0                                          # a scenario without positives
        last = (rng.random(y.shape) < 0.3).astype(float)
        a = [np.round(rng.random(y.shape), 2) for _ in range(2)]      # rounded: ties
        b = np.round(0.5 * y + 0.5 * rng.random(y.shape), 1)
        ref = bootstrap_ap_difference(y, a, b, last, groups, draws=300, seed=42)
        got = c3_forecast.fast_bootstrap(dict(A=a, B=[b]), y, last, groups, {'d': ('A', 'B')}, draws=300, seed=42)
        for part in ('overall', 'onset'):
            r, g = ref[part], got[part]['d']
            self.assertAlmostEqual(r['point'], g['point'], places=12)
            self.assertEqual(r['valid_draws'], g['valid_draws'])
            for x, z in zip(r['ci95'], g['ci95']):
                self.assertAlmostEqual(x, z, places=12)


class TestRuns(unittest.TestCase):
    def test_instrument_changes_nothing(self):
        flows = C.scaled_flows(G, 4000, 'hotspot', 1.0)
        kw = dict(threshold=0.851, guard=dict(signal='score'))
        plain = nf_exp.run_case_x(G, flows, 150, 'guarded', **kw)
        inst = nf_exp.run_case_x(G, flows, 150, 'guarded', instrument=True, **kw)
        self.assertGreater(plain['reroutes'], 0)
        for k in c3_load.KEEP + ['decision_digest', 'packet_conservation', 'drop_split_ok']:
            self.assertEqual(plain[k], inst[k], k)
        self.assertIn('reordered_packets', inst)

    def test_oracle_lookahead_alignment(self):
        """With routing frozen (the guard never moves a flow) the oracle label at tick T equals the ECMP label T+4."""
        flows = C.scaled_flows(G, 4001, 'mixed', 0.25)
        ecmp = run_simulation_and_collect(G, copy.deepcopy(flows), 150).edge_labels_history
        seen = []
        orig = nf_exp.GuardX.evaluate_and_reroute

        def spy(self, flows, predictions, engine, tick, **kw):
            seen.append((tick, np.array(predictions)))
            return []
        nf_exp.GuardX.evaluate_and_reroute = spy
        try:
            nf_exp.run_case_x(G, flows, 150, 'guarded', threshold=0.5, guard=dict(signal='oracle_label'))
        finally:
            nf_exp.GuardX.evaluate_and_reroute = orig
        self.assertEqual(len(seen), 150 - 10 + 1)
        for tick, p in seen:
            if tick + 4 < len(ecmp):
                np.testing.assert_array_equal(p, ecmp[tick + 4])
        self.assertTrue(any(p.sum() > 0 for _, p in seen))


class TestTasks(unittest.TestCase):
    def test_shared_variants_equal_rev3_load_tasks(self):
        import rev3_load
        keys = ('load', 'variant', 'strategy', 'signal', 'threshold', 'guard', 'mseed', 'rng_seed', 'family', 'seed')
        ref = {t['id']: {k: t.get(k) for k in keys} for t in rev3_load.tasks()}
        got = {t['id']: {k: t.get(k) for k in keys}
               for t in c3_load.make_tasks(c3_load.REV3_VARIANTS, C.LOADS, C.FAM, C.UNSEEN)}
        self.assertEqual(got, ref)

    def test_model_variants_one_task_per_training_seed(self):
        T = c3_load.make_tasks(['m1_pmix', 'm0p_p10'], [0.2], ['uniform'], [4000])
        self.assertEqual(sorted((t['variant'], t['mseed']) for t in T),
                         sorted((v, m) for v in ('m1_pmix', 'm0p_p10') for m in (0, 1, 2)))


if __name__ == '__main__':
    unittest.main()
