import importlib.util, os, unittest
from collections import Counter
from unittest import mock
import research_benchmark as rb
from routing.guarded import GuardedRerouter
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('guard_sweep', os.path.join(ROOT, 'tools', 'guard_sweep.py'))
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

THRESHOLDS = dict(persistence_binary=.5, persistence_continuous=.851)


class GuardKwargsTests(unittest.TestCase):
    def run_guarded(self, guard_kwargs):
        """run_case for guarded_score on a short hotspot scenario; returns (metrics, the guard)."""
        created = []

        class Spy(GuardedRerouter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

        graph = build_fat_tree()
        flows = TrafficGenerator(graph, seed=7).generate_scenario(total_ticks=60, pattern='hotspot')
        with mock.patch.object(rb, 'GuardedRerouter', Spy):
            result, _ = rb.run_case(graph, flows, 60, 'guarded_score', None, THRESHOLDS, guard_kwargs=guard_kwargs)
        self.assertEqual(len(created), 1)
        return result, created[0]

    def test_max_changes_reaches_the_guard_and_caps_reroutes_per_tick(self):
        default, guard = self.run_guarded(None)
        self.assertEqual(guard.max_changes, 4)
        per_tick = Counter(e['tick'] for e in guard.reroute_log)
        self.assertGreater(max(per_tick.values()), 1)  # the budget binds in this scenario
        capped, guard = self.run_guarded(dict(max_changes=1))
        self.assertEqual(guard.max_changes, 1)
        self.assertEqual(max(Counter(e['tick'] for e in guard.reroute_log).values()), 1)
        self.assertLess(capped['reroutes'], default['reroutes'])
        # guard_kwargs=None keeps the benchmark's guard exactly (max_changes=4 is the default).
        explicit, _ = self.run_guarded(dict(max_changes=4))
        self.assertEqual(explicit, default)


class SweepSummaryTests(unittest.TestCase):
    def test_summary_is_paired_against_ecmp_on_the_same_traffic(self):
        runs = []
        for pattern, ecmp, mc2, mc4 in (('uniform', 100, 90, 100), ('hotspot', 200, 150, 210)):
            for strategy, total in (('ecmp', ecmp), ('guarded_score_mc2', mc2), ('guarded_score_mc4', mc4)):
                runs.append(dict(pattern=pattern, seed=4000, strategy=strategy, total_drops=total,
                                 fabric_link_drops=total // 2, host_link_drops=total - total // 2,
                                 reroutes=0 if strategy == 'ecmp' else 10))
        out = sweep.summarize(runs, max_changes=[2, 4], draws=200)
        mc2, mc4 = out['max_changes']
        self.assertEqual(out['pairs'], 2)
        self.assertEqual(out['ecmp']['mean_total_drops'], 150)
        self.assertAlmostEqual(mc2['total_change_vs_ecmp_percent'], (120 / 150 - 1) * 100)
        self.assertEqual((mc2['wins'], mc2['ties'], mc2['losses']), (2, 0, 0))
        self.assertEqual((mc4['wins'], mc4['ties'], mc4['losses']), (0, 1, 1))
        self.assertAlmostEqual(mc2['per_family']['hotspot']['total_change_vs_ecmp_percent'], -25.0)
        self.assertEqual(mc2['paired_vs_ecmp']['total_drops']['mean_difference'], -30.0)
        self.assertEqual(mc2['mean_reroutes_per_run'], 10)

    def test_sweep_seeds_are_new_traffic(self):
        self.assertFalse(set(sweep.SEEDS) & set(range(3000, 3010)))
        self.assertEqual(sweep.MAX_CHANGES, [2, 4, 8, 16])


if __name__ == '__main__':
    unittest.main()
