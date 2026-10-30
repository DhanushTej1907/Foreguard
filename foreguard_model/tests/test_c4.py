"""Concern 4 (4th review): reordering tolerance rules of the closed-loop engine (revision/scripts/nf_exp.py).

W = 0 equals the legacy full penalty; a very large window never penalises; the dupACK count matches hand-computed
values (same flow, delivered before, strictly later birth; spurious copies, other flows and equal births excluded);
same-tick deliveries count in FIFO order; the saved no-penalty and full-penalty rows are reproduced through the legacy
flags and the new parameters; invalid parameter combinations are rejected."""
import json, os, sys, unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'revision', 'scripts')
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import networkx as nx  # noqa: E402
import nf_exp  # noqa: E402
from simulation.topology import build_fat_tree  # noqa: E402
from simulation.traffic_generator import Flow, TrafficGenerator  # noqa: E402

G = build_fat_tree()
RES = os.path.join(ROOT, 'revision', 'results')
A_PATHS = sorted(nx.all_shortest_paths(G, 'host_0_0_0', 'host_1_0_0'))   # 6 hops, RTT 12


def synthetic(**kw):
    """Flow 0 shares a queued path with flow 1, then moves at tick 15 to an idle equal-cost path: its first packets on
    the new path overtake the queued ones (2 reordered batches of 300 packets, displacement 2 ticks)."""
    e = nf_exp.ClosedLoopEngine(G, **kw)
    a = Flow(0, 'host_0_0_0', 'host_1_0_0', 300.0, 0, 40)
    b = Flow(1, 'host_0_0_1', 'host_1_1_0', 400.0, 0, 40)
    a.current_path = list(A_PATHS[0])
    b.current_path = ['host_0_0_1'] + list(A_PATHS[0][1:-2]) + ['edge_1_1', 'host_1_1_0']
    e.set_flows([a, b])
    for t in range(600):
        if t == 15:
            a.current_path = list(A_PATHS[-1])
        e.step()
        if e.all_done():
            break
    return e


def outcome(e):
    m = e.metrics()
    return dict(m, tick=e.tick, spurious=e.spurious_packets, retx=e.retx_packets,
                flows={k: (s.fct, s.delivered, s.spur_sent, s.retx_sent, s.reordered) for k, s in e.cl.items()})


class TestSyntheticFlow(unittest.TestCase):
    def test_w0_equals_legacy_full_penalty(self):
        legacy, w0 = synthetic(reorder_penalty=True), synthetic(reorder_penalty=True, reorder_window=0)
        self.assertEqual(outcome(legacy), outcome(w0))
        self.assertGreater(legacy.reordered_packets, 0)
        self.assertGreater(legacy.spurious_packets, 0)
        self.assertEqual(w0.penalised_batches, w0.reordered_batches_open)
        self.assertTrue(legacy.metrics()['packet_conservation'])

    def test_large_window_never_penalises(self):
        off = synthetic()
        for kw in (dict(reorder_window=10 ** 9), dict(dupack_threshold=10 ** 9)):
            e = synthetic(reorder_penalty=True, **kw)
            self.assertGreater(e.reordered_batches, 0)
            self.assertEqual(e.penalised_batches, 0)
            self.assertEqual(e.spurious_packets, 0)
            self.assertEqual(outcome(e), outcome(off))

    def test_window_boundary(self):
        self.assertEqual(dict(synthetic().reorder_disp_hist), {2: 2})
        self.assertEqual(synthetic(reorder_penalty=True, reorder_window=1).penalised_batches, 2)    # d = 2 > 1
        self.assertEqual(synthetic(reorder_penalty=True, reorder_window=2).penalised_batches, 0)    # d = 2 not > 2

    def test_rtt_quarter(self):
        e = synthetic(reorder_penalty=True, reorder_window='rtt/4')
        self.assertEqual(dict(e.reorder_rtt_hist), {12: 2})        # 6 hops -> RTT 12 -> window 3 ticks
        self.assertEqual(e.penalised_batches, 0)                   # 4 * 2 = 8 not > 12


class TestDupackCount(unittest.TestCase):
    def setUp(self):
        self.e = nf_exp.ClosedLoopEngine(G, reorder_penalty=True, dupack_threshold=3)
        flows = [Flow(i, 'host_0_0_0', 'host_1_0_0', 100.0, 0, 100) for i in range(3)]   # size 10,000: never done
        for f in flows:
            f.current_path = list(A_PATHS[0])
        self.e.set_flows(flows)
        self.path = tuple(A_PATHS[0])

    def deliver(self, fid, born, count, spurious=False):
        """Returns (dupACK count seen by this batch, penalised?)."""
        st = self.e.cl[fid]
        dup = self.e.dupack_count(st, born)
        before = len(st.spur_events)
        self.e._deliver(50, count, self.path, born, fid, spurious)
        return dup, len(st.spur_events) > before

    def test_hand_computed_sequence(self):
        d = self.deliver
        self.assertEqual(d(0, 0, 2), (0, False))            # in order
        self.assertEqual(d(0, 3, 1), (0, False))            # in order
        self.assertEqual(d(1, 9, 50), (0, False))           # other flow: never counted for flow 0
        self.assertEqual(d(0, 5, 1), (0, False))            # in order, max_born = 5
        self.assertEqual(d(0, 7, 10, spurious=True)[1], False)   # spurious copy: not counted, max_born stays 5
        self.assertEqual(self.e.cl[0].max_born, 5)
        self.assertEqual(d(0, 2, 4), (2, False))            # reordered: births 3 (1) + 5 (1) = 2 < 3
        self.assertEqual(d(0, 5, 2), (0, False))            # equal to max_born: not reordered (no decision)
        self.assertEqual(d(0, 1, 1), (8, True))             # births 2 (4) + 3 (1) + 5 (3) = 8
        self.assertEqual(d(0, 4, 1), (3, True))             # births 5 (1 + 2) = 3 >= 3 (boundary)
        self.assertEqual(d(0, 4, 2), (3, True))             # the earlier birth-4 packet is not counted (equal birth)
        self.assertEqual(self.e.reordered_batches, 4)
        self.assertEqual(self.e.penalised_batches, 3)
        self.assertEqual(self.e.penalised_packets, 4)
        self.assertEqual(self.e.cl[1].reordered, 0)

    def test_equal_births_excluded(self):
        d = self.deliver
        self.assertEqual(d(2, 2, 2), (0, False))
        self.assertEqual(d(2, 1, 2), (2, False))
        self.assertEqual(d(2, 1, 5), (2, False))            # 4 if equal births counted -> would be penalised

    def test_same_tick_fifo_order(self):
        """Two batches of one flow leave the last hop in the same tick: the later-born one first."""
        for order, reordered in (((5, 4), True), ((4, 5), False)):
            e = nf_exp.ClosedLoopEngine(G, reorder_penalty=True, dupack_threshold=3)
            f = Flow(0, 'host_0_0_0', 'host_1_0_0', 100.0, 10 ** 6, 100)     # never starts: no injections
            f.current_path = list(A_PATHS[0])
            e.set_flows([f])
            path = tuple(A_PATHS[0])
            link = e.link_states[(path[-2], path[-1])]
            for born, cnt in zip(order, (3, 1) if order[0] == 5 else (1, 3)):
                link.enqueue(cnt, (path, len(path) - 2, born, 0))
            e.step()
            st = e.cl[0]
            self.assertEqual(st.delivered, 4)
            self.assertEqual(e.reordered_batches, int(reordered))
            if reordered:
                self.assertEqual(dict(e.reorder_disp_hist), {1: 1})
                self.assertEqual(dict(e.reorder_dupack_hist), {'3': 1})   # the 3 born-5 packets delivered just before
                self.assertEqual(e.penalised_batches, 1)


def _saved(name, key):
    for line in open(os.path.join(RES, name)):
        r = json.loads(line)
        if r['id'] == key:
            return r
    raise KeyError(key)


META = {'id', 'group', 'variant', 'strategy', 'signal', 'mseed', 'family', 'seed', 'guard', 'threshold_used', 'wall_s',
        'sec', 'closed_loop', 'mode', 'penalty'}


class TestSavedBehaviour(unittest.TestCase):
    """reactive | uniform | 3000: saved rows with reordering (rev4_c4: 30,447 reordered, 30,299 spurious packets)."""
    @classmethod
    def setUpClass(cls):
        cls.flows = TrafficGenerator(G, seed=3000).generate_scenario(total_ticks=150, pattern='uniform')

    def run_cell(self, **kw):
        r = nf_exp.run_case_x(G, self.flows, 150, 'reactive', closed_loop=True, **kw)
        return json.loads(json.dumps(r))

    def assert_matches(self, saved, got):
        keys = [k for k in saved if k not in META]
        self.assertGreater(len(keys), 8)
        for k in keys:
            self.assertEqual(saved[k], got[k], k)

    def test_no_penalty(self):
        saved = _saved('closedloop.jsonl', 'closedloop|reactive|reactive|None|None|uniform|3000')
        legacy, explicit = self.run_cell(), self.run_cell(reorder_penalty=False)
        self.assertEqual(legacy, explicit)
        self.assert_matches(saved, legacy)
        self.assertEqual(legacy['spurious_packets'], 0)

    def test_full_penalty(self):
        saved = _saved('rev4_c4.jsonl', 'reactive|uniform|3000')
        legacy, w0 = self.run_cell(reorder_penalty=True), self.run_cell(reorder_penalty=True, reorder_window=0)
        self.assertEqual(legacy, w0)
        self.assert_matches(saved, legacy)
        self.assertGreater(legacy['spurious_packets'], 0)
        self.assertEqual(legacy['reorder_diag']['penalised_batches'], legacy['reorder_diag']['reordered_batches_open'])


class TestValidation(unittest.TestCase):
    def test_rejected(self):
        bad = [dict(reorder_window=2), dict(dupack_threshold=3), dict(reorder_penalty=False, reorder_window=0),
               dict(reorder_penalty=True, reorder_window=2, dupack_threshold=3),
               dict(reorder_penalty=True, reorder_window=-1), dict(reorder_penalty=True, reorder_window=1.5),
               dict(reorder_penalty=True, reorder_window='rtt/2'), dict(reorder_penalty=True, reorder_window=True),
               dict(reorder_penalty=True, dupack_threshold=0)]
        for kw in bad:
            with self.assertRaises(ValueError, msg=kw):
                nf_exp.ClosedLoopEngine(G, **kw)

    def test_accepted(self):
        for kw in (dict(), dict(reorder_penalty=False), dict(reorder_penalty=True), dict(reorder_penalty=True, reorder_window=0),
                   dict(reorder_penalty=True, reorder_window='rtt/4'), dict(reorder_penalty=True, dupack_threshold=3)):
            nf_exp.ClosedLoopEngine(G, **kw)

    def test_open_loop_rejects_penalty(self):
        flows = TrafficGenerator(G, seed=3000).generate_scenario(total_ticks=150, pattern='uniform')
        with self.assertRaises(ValueError):
            nf_exp.run_case_x(G, flows, 150, 'ecmp', reorder_window=0)


if __name__ == '__main__':
    unittest.main()
