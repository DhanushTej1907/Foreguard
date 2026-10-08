import unittest
import copy
import numpy as np
import networkx as nx
from simulation.network_engine import NetworkEngine, LinkState
from simulation.traffic_generator import Flow, TrafficGenerator
from simulation.topology import build_fat_tree
from simulation.telemetry import TelemetryCollector
from model.splits import split_scenarios
from routing.guarded import GuardedRerouter

def line():
    g=nx.Graph()
    for u,v in [('a','b'),('b','c')]:
        g.add_edge(u,v,capacity=2,buffer_size=10,propagation_delay_us=.5)
    g.graph.update(all_switches=['b'],hosts=['a','c'])
    return g

class SimulatorTests(unittest.TestCase):
    def test_causality_and_conservation(self):
        e=NetworkEngine(line()); e.set_flows([Flow(1,'a','c',3,0,1)])
        e.step(); self.assertEqual(e.total_delivered,0)
        self.assertEqual(e.get_link_state('b','c').packets_in,0)
        for _ in range(5):
            e.step(); self.assertTrue(e.metrics()['packet_conservation'])
        self.assertEqual(e.total_delivered,3)
        self.assertEqual(e.outstanding_packets,0)

    def test_drops_not_duplicated(self):
        g=line()
        for _,_,d in g.edges(data=True): d['buffer_size']=2
        e=NetworkEngine(g); e.set_flows([Flow(1,'a','c',10,0,1)])
        for _ in range(4): e.step()
        self.assertEqual(e.total_drops,8); self.assertEqual(e.total_delivered,2)
        self.assertTrue(e.metrics()['packet_conservation'])

    def test_fractional_rates_and_tick_zero(self):
        e=NetworkEngine(line()); e.set_flows([Flow(1,'a','c',.5,0,4)])
        for _ in range(6): e.step()
        self.assertEqual(e.total_packets,2); self.assertEqual(e.total_delivered,2)

    def test_units(self):
        from config import SIMULATION
        old=SIMULATION['tick_duration_ms']
        try:
            SIMULATION['tick_duration_ms']=2
            s=LinkState(2,10,.5); s.enqueue(4); s.process_tick()
            self.assertEqual(s.latency_us,2000.5)
        finally: SIMULATION['tick_duration_ms']=old

    def test_ecmp_stable_and_diverse(self):
        g=build_fat_tree(); e=NetworkEngine(g)
        flows=[Flow(i,'host_0_0_0','host_1_0_0',1,0,1) for i in range(40)]
        # Use discovered names, independent of the builder's naming convention.
        for f in flows: f.src=g.graph['hosts'][0]; f.dst=g.graph['hosts'][4]
        paths=[tuple(e.get_ecmp_path(f)) for f in flows]
        self.assertGreater(len(set(paths)),1)
        self.assertEqual(paths,[tuple(NetworkEngine(g).get_ecmp_path(f)) for f in flows])

    def test_reset(self):
        e=NetworkEngine(line()); e.set_flows([Flow(1,'a','c',3,0,1)])
        e.step(); e.reset()
        self.assertEqual(e.outstanding_packets,0); self.assertEqual(e.total_packets,0)

class DatasetTests(unittest.TestCase):
    def test_group_disjoint(self):
        samples=[dict(scenario_id=i,tick=t) for i in range(10) for t in range(20)]
        sets=[{s['scenario_id'] for s in p} for p in split_scenarios(samples)]
        self.assertTrue(all(sets)); self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    def test_reject_legacy(self):
        with self.assertRaises(ValueError): split_scenarios([{'tick':1}])
    def test_horizon(self):
        g=build_fat_tree(); c=TelemetryCollector(g,NetworkEngine(g))
        c.node_features_history=[np.array([[i]]) for i in range(10)]
        c.edge_features_history=[np.array([[i]]) for i in range(10)]
        c.edge_labels_history=[np.array([i]) for i in range(10)]
        samples=c.get_windowed_samples(3,2)
        self.assertEqual(len(samples),6)
        self.assertEqual(samples[0]['feature_end'],2)
        self.assertEqual(samples[0]['labels'][0],4)
        self.assertEqual(samples[-1]['labels'][0],9)

class ControllerTests(unittest.TestCase):
    def setup_case(self):
        g=nx.Graph()
        for u,v in [('s','a'),('a','t'),('s','b'),('b','t')]:
            g.add_edge(u,v,capacity=100,buffer_size=200,propagation_delay_us=.5)
        g.graph.update(all_switches=['a','b'],hosts=['s','t'])
        e=NetworkEngine(g); mapping={e._edge_key(u,v):i for i,(u,v) in enumerate(g.edges())}
        p=np.zeros(4); p[mapping[e._edge_key('s','a')]]=.99
        flows=[Flow(i,'s','t',40,0,20) for i in range(4)]
        for f in flows: f.current_path=['s','a','t']; f.active=True
        return g,e,mapping,p,flows
    def test_capacity_reservation(self):
        g,e,m,p,flows=self.setup_case(); r=GuardedRerouter(g,m)
        events=r.evaluate_and_reroute(flows,p,e,1)
        self.assertEqual(len(events),2) # third move would reserve 120 > 90
        self.assertEqual(len(r.evaluate_and_reroute(flows,p,e,2)),0)
    def test_stale_invalid_and_budget(self):
        g,e,m,p,flows=self.setup_case(); r=GuardedRerouter(g,m,max_changes=1)
        self.assertEqual(r.evaluate_and_reroute(flows,p,e,5,prediction_tick=1),[])
        self.assertEqual(r.evaluate_and_reroute(flows,p*np.nan,e,5),[])
        self.assertEqual(len(r.evaluate_and_reroute(flows,p,e,5)),1)
    def test_overloaded_alternative(self):
        g,e,m,p,flows=self.setup_case(); e.get_link_state('s','b').queue=100
        r=GuardedRerouter(g,m)
        self.assertEqual(r.evaluate_and_reroute(flows,p,e,1),[])

if __name__=='__main__': unittest.main()
