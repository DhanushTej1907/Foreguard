"""Capacity-check term ablation (paper Table A7): GuardX flags cap_arrivals / cap_queue / rate_mode='blind',
the opt-in shadow audit, and the copied bootstrap (revision/scripts/rev5_stats.py)."""
import json, os, sys, unittest
from types import SimpleNamespace

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'revision', 'scripts')
RESULTS = os.path.join(ROOT, 'revision', 'results')
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
import nf_exp  # noqa: E402
import rev5_stats  # noqa: E402

CONFIGS = {'A': dict(), 'B': dict(cap_queue=False), 'Cp': dict(cap_arrivals=False),
           'D': dict(cap_arrivals=False, cap_queue=False),
           'E': dict(cap_arrivals=False, cap_queue=False, rate_mode='blind')}
OLD = ['h1', 's1', 'a', 's2', 'h2']      # risky branch (risk 0.95)
NEW = ['h1', 's1', 'b', 's2', 'h2']      # safe branch (risk 0); the only alternative
B_LINKS = [('s1', 'b'), ('b', 's2')]     # the edges a move from OLD to NEW adds
A_LINKS = [('s1', 'a'), ('a', 's2')]


class Diamond:
    """h1 - s1 - {a | b} - s2 - h2 plus an isolated x1 - x2 link for flows that only shift the mean rate.
    Capacity 500 and alpha 0.9 give alpha * C = 450 on every link."""

    def __init__(self, b_in=0, b_queue=0, a_in=0):
        self.G = nx.Graph([('h1', 's1'), ('s1', 'a'), ('a', 's2'), ('s1', 'b'), ('b', 's2'), ('s2', 'h2'),
                           ('x1', 'x2')])
        self.edge_to_idx = {}
        for i, (u, v) in enumerate([('s1', 'a'), ('a', 's2'), ('s1', 'b'), ('b', 's2')]):
            self.edge_to_idx[(u, v)] = self.edge_to_idx[(v, u)] = i
        self.pred = np.array([.95, .95, 0., 0.])
        self.link_states = {}
        for u, v in self.G.edges:
            for e in ((u, v), (v, u)):
                self.link_states[e] = SimpleNamespace(capacity=500, queue=0, packets_in=0)
        for e in B_LINKS:
            self.link_states[e].packets_in, self.link_states[e].queue = b_in, b_queue
        for e in A_LINKS:
            self.link_states[e].packets_in = a_in

    @staticmethod
    def _edge_key(u, v):
        return (u, v)


def flow(fid, rate, path):
    return SimpleNamespace(flow_id=fid, rate=float(rate), active=True, current_path=list(path),
                           src=path[0], dst=path[-1])


def run(net, flows, audit=False, **kw):
    g = nf_exp.GuardX(net.G, net.edge_to_idx, threshold=.851, audit=audit, **kw)
    events = g.evaluate_and_reroute(flows, net.pred, net, tick=10)
    return g, events


class TermIsolationTests(unittest.TestCase):
    # (R on the b links, arrivals O on the b links, queue Q on the b links) with a mover at r_f = 50
    CASES = {
        'queue_binds':    ((300, 0, 120),   dict(A=False, B=True, Cp=False, D=True)),
        'arrivals_bind':  ((300, 420, 0),   dict(A=False, B=False, Cp=True, D=True)),
        'both_needed':    ((300, 420, 120), dict(A=False, B=False, Cp=False, D=True)),
        'none_binds':     ((300, 0, 0),     dict(A=True, B=True, Cp=True, D=True)),
    }

    def scenario(self, R, O, Q):
        net = Diamond(b_in=O, b_queue=Q)
        flows = [flow('mover', 50, OLD), flow('bg', R, NEW)]
        return net, flows

    def test_each_variant_removes_only_its_term(self):
        for name, ((R, O, Q), expected) in self.CASES.items():
            for cfg, accept in expected.items():
                with self.subTest(case=name, config=cfg):
                    net, flows = self.scenario(R, O, Q)
                    g, events = run(net, flows, **CONFIGS[cfg])
                    moved = flows[0].current_path == NEW
                    self.assertEqual(moved, accept)
                    self.assertEqual(len(events), int(accept))
                    self.assertEqual(g.rejections['no_admissible_path'], int(not accept))

    def test_audit_counts_on_A_trajectory(self):
        expect = {
            'queue_binds':   dict(tests_queued=1, tests_arrivals_raise=0, pass_A=0, pass_B=1, pass_Cp=0, pass_D=1,
                                  A_ne_B=1, A_ne_Cp=0, B_ne_D=0, Cp_ne_D=1, A_ne_D=1),
            'arrivals_bind': dict(tests_queued=0, tests_arrivals_raise=1, pass_A=0, pass_B=0, pass_Cp=1, pass_D=1,
                                  A_ne_B=0, A_ne_Cp=1, B_ne_D=1, Cp_ne_D=0, A_ne_D=1),
            'both_needed':   dict(tests_queued=1, tests_arrivals_raise=1, pass_A=0, pass_B=0, pass_Cp=0, pass_D=1,
                                  A_ne_B=0, A_ne_Cp=0, B_ne_D=1, Cp_ne_D=1, A_ne_D=1),
            'none_binds':    dict(tests_queued=0, tests_arrivals_raise=0, pass_A=1, pass_B=1, pass_Cp=1, pass_D=1,
                                  A_ne_B=0, A_ne_Cp=0, B_ne_D=0, Cp_ne_D=0, A_ne_D=0),
        }
        for name, ((R, O, Q), _) in self.CASES.items():
            with self.subTest(case=name):
                net, flows = self.scenario(R, O, Q)
                g, _ = run(net, flows, audit=True)
                a = g.cap_audit
                self.assertEqual(a['tests'], 1)
                self.assertEqual(a['self_mismatch'], 0)
                for k, v in expect[name].items():
                    self.assertEqual(a[k], v, k)

    def test_rate_blind_changes_only_the_estimate(self):
        # exact demand passes, blind demand fails: far flow (isolated link) raises the mean rate to 450
        net = Diamond()
        flows = [flow('mover', 50, OLD), flow('bg', 300, NEW), flow('far', 1000, ['x1', 'x2'])]
        _, ev_d = run(net, flows, **CONFIGS['D'])
        self.assertEqual(len(ev_d), 1)
        net = Diamond()
        flows = [flow('mover', 50, OLD), flow('bg', 300, NEW), flow('far', 1000, ['x1', 'x2'])]
        g, ev_e = run(net, flows, audit=True, **CONFIGS['E'])
        self.assertEqual(len(ev_e), 0)
        self.assertEqual(g.cap_audit['pass_E'], 0)
        self.assertEqual(g.cap_audit['self_mismatch'], 0)
        # on an A trajectory (exact rates) the shadow D passes and the shadow E fails at the same test
        net = Diamond()
        flows = [flow('mover', 50, OLD), flow('bg', 300, NEW), flow('far', 1000, ['x1', 'x2'])]
        g, _ = run(net, flows, audit=True)
        self.assertEqual((g.cap_audit['pass_D'], g.cap_audit['pass_E'], g.cap_audit['A_ne_E']), (1, 0, 1))
        # reverse: exact demand fails (420 + 50 > 450), blind passes (3 x 117.75 = 353.25)
        mk = lambda: [flow('mover', 50, OLD), flow('bg1', 210, NEW), flow('bg2', 210, NEW), flow('far', 1, ['x1', 'x2'])]
        flows = mk()
        _, ev_d = run(Diamond(), flows, **CONFIGS['D'])
        self.assertEqual(len(ev_d), 0)
        flows = mk()
        _, ev_e = run(Diamond(), flows, **CONFIGS['E'])
        self.assertEqual(len(ev_e), 1)

    def test_reservation_updates_full_and_rate_only_demand(self):
        for cfg in ('A', 'D'):
            with self.subTest(config=cfg):
                net = Diamond()
                flows = [flow('m1', 50, OLD), flow('m2', 50, OLD), flow('bg', 360, NEW)]
                g, events = run(net, flows, audit=True, **CONFIGS[cfg])
                # 360 + 50 = 410 fits; the reserved 410 + 50 = 460 does not
                self.assertEqual(len(events), 1)
                st = g.last_state
                for e in B_LINKS:
                    self.assertEqual(st['demand'][e], 410.0)
                    self.assertEqual(st['rates'][e], 410.0)
                for e in A_LINKS:
                    self.assertEqual(st['demand'][e], 50.0)
                    self.assertEqual(st['rates'][e], 50.0)
                self.assertEqual(g.cap_audit['tests'], 2)
                self.assertEqual(g.cap_audit['self_mismatch'], 0)

    def test_arrival_bound_old_link_after_move(self):
        net = Diamond(a_in=500)
        flows = [flow('mover', 50, OLD)]
        g, events = run(net, flows, audit=True)
        self.assertEqual(len(events), 1)
        for e in A_LINKS:
            self.assertEqual(g.last_state['demand'][e], 450.0)   # max(50, 500) - 50
            self.assertEqual(g.last_state['rates'][e], 0.0)      # 50 - 50

    def test_audit_off_by_default_and_pure(self):
        net, flows = self.scenario(300, 420, 120)
        g, ev_off = run(net, flows)
        self.assertFalse(g.audit)
        self.assertEqual(dict(g.cap_audit), {})
        self.assertNotIn('rates', g.last_state)
        for cfg in CONFIGS:
            for case in self.CASES.values():
                (R, O, Q), _ = case
                with self.subTest(config=cfg, case=(R, O, Q)):
                    n1, f1 = self.scenario(R, O, Q)
                    n2, f2 = self.scenario(R, O, Q)
                    g1, e1 = run(n1, f1, **CONFIGS[cfg])
                    g2, e2 = run(n2, f2, audit=True, **CONFIGS[cfg])
                    self.assertEqual(e1, e2)
                    self.assertEqual(dict(g1.last_state['demand']), dict(g2.last_state['demand']))
                    self.assertEqual([f.current_path for f in f1], [f.current_path for f in f2])
                    self.assertEqual(g2.cap_audit['self_mismatch'], 0)


def _saved(path, pred):
    with open(os.path.join(RESULTS, path)) as fh:
        return {(r['family'], r['seed']): r for r in map(json.loads, fh) if pred(r)}


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from simulation.topology import build_fat_tree
        cls.g = build_fat_tree()

    def run_case(self, family, fi, seed, signal, **guard):
        from simulation.traffic_generator import TrafficGenerator
        flows = TrafficGenerator(self.g, seed=seed).generate_scenario(total_ticks=150, pattern=family)
        return nf_exp.run_case_x(self.g, flows, 150, 'guarded', threshold=.851,
                                 guard=dict(guard, signal=signal, noise_seed=1000 * fi + seed),
                                 rng_seed=1000 * fi + seed, instrument=True)

    def test_default_reproduces_saved_run(self):
        saved = _saved('rev4_c1.jsonl', lambda r: r['mode'] == 'exact' and r['variant'] == 'score')[('uniform', 4000)]
        r = self.run_case('uniform', 0, 4000, 'score')
        for k in ('total_drops', 'fabric_link_drops', 'host_link_drops', 'total_offered', 'total_delivered',
                  'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes', 'reordered_packets',
                  'reordered_fraction', 'flows_with_reordering', 'n_flows'):
            self.assertEqual(r[k], saved[k], k)
        self.assertNotIn('cap_audit', r)

    def test_audit_does_not_change_outcomes(self):
        for signal in ('score', 'random'):
            with self.subTest(signal=signal):
                off = self.run_case('hotspot', 1, 4000, signal)
                on = self.run_case('hotspot', 1, 4000, signal, audit=True)
                self.assertEqual(on.pop('audit_definition_version'), 1)
                audit = on.pop('cap_audit')
                self.assertEqual(audit['self_mismatch'], 0)
                self.assertGreater(audit['tests'], 0)
                self.assertEqual(off, on)


class BootstrapCalibrationTests(unittest.TestCase):
    def test_reproduces_saved_intervals(self):
        E = np.mean([r['total_drops'] for r in _saved('replicate.jsonl', lambda r: r['variant'] == 'ecmp').values()])
        nc = _saved('ablation.jsonl', lambda r: r.get('variant') == 'no_capacity' and r.get('signal') == 'score')
        fu = _saved('ablation.jsonl', lambda r: r.get('variant') == 'full' and r.get('signal') == 'score')
        c = rev5_stats.compare(nc, fu, E)
        self.assertEqual((c['n'], c['ci95_pct'], c['ci95_cluster_pct']), (60, [-0.009, 2.446], [-0.509, 3.224]))
        self.assertAlmostEqual(c['mean_pct'], 1.220107667097269, places=12)
        rd = _saved('rev4_c1.jsonl', lambda r: r['mode'] == 'exact' and r['variant'] == 'random')
        sc = _saved('rev4_c1.jsonl', lambda r: r['mode'] == 'exact' and r['variant'] == 'score')
        self.assertEqual(rev5_stats.compare(rd, sc, E)['ci95_pct'], [-1.182, 0.526])

    def test_cluster_resamples_families_not_seeds(self):
        # one family with value 1 at every seed, five with 0: a draw can only be k/6 for whole families k
        d = {(f, s): float(f == 'f0') for f in ('f0', 'f1', 'f2', 'f3', 'f4', 'f5') for s in range(10)}
        vals = np.round(rev5_stats.boot(d, cluster=True) * 6, 9)
        self.assertTrue(set(vals) <= set(range(7)))


if __name__ == '__main__':
    unittest.main()
