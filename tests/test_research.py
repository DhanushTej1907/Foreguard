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

    def test_directional_queues_independent(self):
        # Opposing full-rate flows: a shared FIFO would see 4 pkts/tick on a 2 pkt/tick link.
        e=NetworkEngine(line()); e.set_flows([Flow(1,'a','c',2,0,5),Flow(2,'c','a',2,0,5)])
        self.assertIsNot(e.get_link_state('a','b'),e.get_link_state('b','a'))
        e.step()  # tick 0 fills first hops only
        for _ in range(3):
            e.step()
            for key in [('a','b'),('b','a'),('b','c'),('c','b')]:
                s=e.get_link_state(*key)
                self.assertEqual(s.packets_in,2); self.assertEqual(s.queue_depth,0)
        for _ in range(4): e.step()
        self.assertEqual(e.total_drops,0); self.assertEqual(e.total_delivered,20)
        self.assertTrue(e.metrics()['packet_conservation'])

    def test_conservation_full_duplex_fat_tree(self):
        g=build_fat_tree(); e=NetworkEngine(g)
        e.set_flows(TrafficGenerator(g,seed=3000).generate_scenario(total_ticks=60,pattern='hotspot'))
        for _ in range(60):
            e.step(); self.assertTrue(e.metrics()['packet_conservation'])
        self.assertEqual(len(e.link_states),2*g.number_of_edges())

    def test_drop_split_sums_to_total(self):
        g=nx.Graph()
        for u,v in [('a1','b'),('a2','b'),('b','c'),('c','d')]:
            g.add_edge(u,v,capacity=2,buffer_size=2,propagation_delay_us=.5)
        g.graph.update(all_switches=['b','c'],hosts=['a1','a2','d'])
        # a2 overloads its host link; a1+a2 overload fabric link b->c.
        e=NetworkEngine(g); e.set_flows([Flow(1,'a1','d',2,0,6),Flow(2,'a2','d',5,0,6)])
        for _ in range(10): e.step()
        m=e.metrics()
        self.assertGreater(m['host_link_drops'],0); self.assertGreater(m['fabric_link_drops'],0)
        self.assertEqual(m['unroutable_drops'],0); self.assertTrue(m['drop_split_ok'])
        self.assertEqual(m['host_link_drops']+m['fabric_link_drops'],m['total_drops'])
        self.assertTrue(m['packet_conservation'])
        # A flow with no route (src == dst) is counted separately and the identity still holds.
        e.reset(); e.active_flows.append(Flow(3,'d','d',3,0,2)); e.set_flows(e.active_flows)
        for _ in range(10): e.step()
        m=e.metrics()
        self.assertEqual(m['unroutable_drops'],6); self.assertTrue(m['drop_split_ok'])
        self.assertEqual(m['host_link_drops']+m['fabric_link_drops']+m['unroutable_drops'],m['total_drops'])

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

class TelemetryTests(unittest.TestCase):
    def snapshot(self):
        g=build_fat_tree(); e=NetworkEngine(g); c=TelemetryCollector(g,e)
        return c, e.step()
    def test_max_over_directions(self):
        c,snap=self.snapshot(); u,v=c.switch_edges[0]
        snap['links'][(v,u)].update(utilization=.95,queue_depth=500)  # one direction only
        c.collect_tick(snap); idx=c.edge_to_idx[(u,v)]
        feats,labels=c.edge_features_history[-1],c.edge_labels_history[-1]
        self.assertEqual(feats.shape,(32,4)); self.assertEqual(labels.shape,(32,))
        self.assertAlmostEqual(feats[idx,0],.95,places=6); self.assertAlmostEqual(feats[idx,1],.25)
        self.assertEqual(labels[idx],1.); self.assertEqual(labels.sum(),1.)
    def test_label_modes(self):
        from config import MODEL
        from simulation.telemetry import congestion_label
        queued=dict(utilization=.5,packets_out=250,queue_depth=300)  # 300 > 0.1*2000
        self.assertEqual(congestion_label(queued,500,2000,'offered'),0.)
        self.assertEqual(congestion_label(queued,500,2000,'served_or_queued'),1.)
        self.assertEqual(congestion_label(dict(utilization=.5,packets_out=450,queue_depth=0),500,2000,'served_or_queued'),1.)
        with self.assertRaises(ValueError): congestion_label(queued,500,2000,'bogus')
        old=MODEL['label_mode']
        try:
            for mode,expected in [('offered',0.),('served_or_queued',1.)]:
                MODEL['label_mode']=mode
                c,snap=self.snapshot(); u,v=c.switch_edges[0]
                snap['links'][(u,v)].update(queue_depth=300)
                c.collect_tick(snap)
                self.assertEqual(c.edge_labels_history[-1][c.edge_to_idx[(u,v)]],expected)
        finally: MODEL['label_mode']=old

class ReactiveTests(unittest.TestCase):
    def test_switch_links_only(self):
        from routing.rerouter import ReactiveRerouter
        g=build_fat_tree(); e=NetworkEngine(g)
        f=Flow(1,g.graph['hosts'][0],g.graph['hosts'][4],1,0,10)
        e.set_flows([f]); f.active=True; path=list(f.current_path)
        r=ReactiveRerouter(g)
        e.get_link_state(path[-2],path[-1]).utilization=1.  # edge -> host link
        self.assertEqual(r.evaluate_and_reroute([f],e,1),[])
        e.get_link_state(path[1],path[2]).utilization=1.    # edge -> agg fabric link
        events=r.evaluate_and_reroute([f],e,2)
        self.assertEqual(len(events),1)
        new=events[0]['new_path']; self.assertNotIn((path[1],path[2]),list(zip(new,new[1:])))

class CheckpointTests(unittest.TestCase):
    def test_schema_version(self):
        from config import CHECKPOINT_SCHEMA_VERSION
        from model.gnn_model import check_checkpoint_schema
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION,3)
        for legacy in [{'schema_version':2},{}]:
            with self.assertRaises(ValueError): check_checkpoint_schema(legacy)
        check_checkpoint_schema({'schema_version':3})

class ControllerTests(unittest.TestCase):
    def setup_case(self):
        g=nx.Graph()
        for u,v in [('s','a'),('a','t'),('s','b'),('b','t')]:
            g.add_edge(u,v,capacity=100,buffer_size=200,propagation_delay_us=.5)
        g.graph.update(all_switches=['a','b'],hosts=['s','t'])
        e=NetworkEngine(g)
        # Per-physical-link prediction index, mapped for both directions (as TelemetryCollector does).
        mapping={k:i for i,(u,v) in enumerate(g.edges()) for k in ((u,v),(v,u))}
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
