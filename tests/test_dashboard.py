import unittest
import time
import numpy as np
import networkx as nx
import torch
from pathlib import Path
from config import CHECKPOINT_OUT
from model.gnn_model import check_checkpoint_schema
from fastapi.testclient import TestClient
from dashboard.server import app, sim_state, _build_update_message, SCORE_THRESHOLD
from routing.guarded import GuardedRerouter
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector


def current_checkpoint():
    """The checkpoint the dashboard may load (current schema and label mode), else None (ECMP fallback)."""
    path = Path(CHECKPOINT_OUT)
    if not path.exists(): return None
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    try: check_checkpoint_schema(ckpt)
    except ValueError: return None
    return ckpt


def link_id(u, v): return '--'.join(sorted((u, v)))


def receive(ws, matches, limit=300):
    for _ in range(limit):
        msg = ws.receive_json()
        if matches(msg): return msg
    raise AssertionError('no matching WebSocket message')


class DashboardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        sim_state['running'] = False

    tearDown = setUp

    def test_config_start_stop_spike_and_shared_clock(self):
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(client.post('/api/config',json={'speed':0}).status_code,422)
            self.assertEqual(client.post('/api/config',json={'threshold':2}).status_code,422)
            self.assertEqual(client.post('/api/config',json={'speed':1,'threshold':.8}).status_code,200)
            self.assertEqual(client.post('/api/start').status_code,200)
            # Only a current-schema, current-label-mode checkpoint at CHECKPOINT_OUT may load;
            # otherwise the dashboard falls back to ECMP.
            ckpt=current_checkpoint()
            status=client.get('/api/status').json()
            self.assertEqual(status['model_loaded'],ckpt is not None)
            if ckpt is not None:
                # The live threshold starts at the checkpoint's calibrated threshold.
                self.assertEqual(sim_state['threshold'],ckpt['alert_threshold'])
                self.assertEqual(status['alert_threshold'],ckpt['alert_threshold'])
                self.assertEqual(status['controller'],'guarded_gnn')
            else:
                self.assertEqual(status['controller'],'ecmp')
                self.assertFalse(status['controllers']['guarded_gnn']['available'])
            time.sleep(.3)
            before=client.get('/api/status').json()['tick']
            self.assertGreater(before,0) # runs without any viewers
            with client.websocket_connect('/ws') as a:
                with client.websocket_connect('/ws') as b:
                    self.assertEqual(a.receive_json()['type'],'topology')
                    self.assertEqual(b.receive_json()['type'],'topology')
                    # a may be registered a tick before b; align on the same tick, then both
                    # viewers must get the identical metrics from the single producer.
                    ma,mb=a.receive_json()['metrics'],b.receive_json()['metrics']
                    while ma['tick']<mb['tick']: ma=a.receive_json()['metrics']
                    while mb['tick']<ma['tick']: mb=b.receive_json()['metrics']
                    self.assertEqual(ma,mb)
            self.assertEqual(client.post('/api/spike').json()['status'],'spike_injected')
            ids=[f.flow_id for f in sim_state['flows']]
            self.assertEqual(len(ids),len(set(ids)))
            client.post('/api/stop')
            tick=client.get('/api/status').json()['tick']; time.sleep(.15)
            self.assertEqual(client.get('/api/status').json()['tick'],tick)

    def test_update_message_uses_topology_link_ids_and_calibrated_threshold(self):
        ckpt = current_checkpoint()
        with TestClient(app) as client:
            client.post('/api/config', json={'speed': 20})
            client.post('/api/start')
            collector = sim_state['collector']
            with client.websocket_connect('/ws') as ws:
                topo = receive(ws, lambda m: m['type'] == 'topology')
                msg = receive(ws, lambda m: m['type'] == 'update' and (ckpt is None or m['predictions']))
            client.post('/api/stop')
        topology_ids = {link_id(l['source'], l['target']) for l in topo['data']['links']}
        # One entry per physical link, under the same IDs the frontend builds from the topology.
        self.assertEqual(set(msg['links']), topology_ids)
        for link in msg['links'].values():
            self.assertIsInstance(link['congested'], bool)
            self.assertTrue(0 <= link['score'] <= 1)
        self.assertEqual(set(msg['since_switch']), {'ticks', 'total_drops', 'fabric_drops', 'host_drops', 'reroutes'})
        if ckpt is None:
            self.assertEqual(msg['predictions'], {})
            self.assertIsNone(msg['alert_threshold'])
            return
        # Predictions: the 32 fabric links, in collector.switch_edges order, as calibrated probabilities.
        self.assertEqual(list(msg['predictions']), [link_id(u, v) for u, v in collector.switch_edges])
        self.assertEqual(len(msg['predictions']), 32)
        self.assertTrue(set(msg['predictions']) <= topology_ids)
        self.assertTrue(all(0 <= p <= 1 for p in msg['predictions'].values()))
        self.assertEqual(msg['alert_threshold'], ckpt['alert_threshold'])
        self.assertEqual(msg['threshold'], ckpt['alert_threshold'])

    def test_build_update_message_maps_switch_edges_and_current_labels(self):
        G = build_fat_tree(); engine = NetworkEngine(G)
        flows = TrafficGenerator(G, seed=7).generate_scenario(total_ticks=60, pattern='hotspot')
        engine.set_flows(flows); collector = TelemetryCollector(G, engine)
        for _ in range(30):
            snapshot = engine.step(); collector.collect_tick(snapshot)
        predictions = np.linspace(0, 1, len(collector.switch_edges), dtype=np.float32)
        path = engine.get_ecmp_path(flows[0])
        event = dict(flow_id=flows[0].flow_id, src=flows[0].src, dst=flows[0].dst, reason='guarded', new_path=tuple(path))
        msg = _build_update_message(snapshot, collector, predictions, [event])
        self.assertEqual(len(msg['links']), G.number_of_edges())
        for i, (u, v) in enumerate(collector.switch_edges):
            self.assertEqual(msg['predictions'][f'{u}--{v}'], round(float(predictions[i]), 4))
            # Current congestion is the training label (served_or_queued), not utilization.
            self.assertEqual(msg['links'][f'{u}--{v}']['congested'], bool(collector.edge_labels_history[-1][i]))
        self.assertTrue(any(link['congested'] for link in msg['links'].values()))
        self.assertEqual(msg['reroute_events'][0]['new_path'], path)
        self.assertEqual(_build_update_message(snapshot, collector)['predictions'], {})
        with self.assertRaises(ValueError):  # a prediction vector of the wrong length is not silently truncated
            _build_update_message(snapshot, collector, predictions[:-1])

    def test_controller_switch_restores_ecmp_and_resets_counters(self):
        ckpt = current_checkpoint()
        zero = dict(ticks=0, total_drops=0, fabric_drops=0, host_drops=0, reroutes=0)
        with TestClient(app) as client:
            client.post('/api/config', json={'speed': 20})
            client.post('/api/start'); time.sleep(.3)
            client.post('/api/stop')
            engine, flows = sim_state['engine'], sim_state['flows']
            on_ecmp = lambda: all(f.current_path == engine.get_ecmp_path(f) for f in flows)
            # Move an inter-pod flow off its ECMP path, as a reroute would.
            f = next(f for f in flows if len(f.current_path) == 7)
            f.current_path = next(p for p in nx.all_shortest_paths(sim_state['graph'], f.src, f.dst)
                                  if p != engine.get_ecmp_path(f))
            self.assertFalse(on_ecmp())

            r = client.post('/api/controller', json={'controller': 'ecmp'})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()['controller'], 'ecmp')
            self.assertEqual(r.json()['since_switch'], zero)
            self.assertIsNone(sim_state['rerouter'])
            self.assertTrue(on_ecmp())

            with client.websocket_connect('/ws') as ws:
                receive(ws, lambda m: m['type'] == 'topology')
                ws.send_json({'action': 'toggle'})  # resume
                msg = receive(ws, lambda m: m['type'] == 'update' and m['since_switch']['ticks'] >= 20)
                self.assertEqual(msg['controller'], 'ecmp')
                self.assertEqual(msg['since_switch']['reroutes'], 0)  # ECMP never reroutes
                s = msg['since_switch']
                self.assertGreaterEqual(s['total_drops'], s['fabric_drops'] + s['host_drops'])
                # Switch over the WebSocket; counters restart at the switch.
                ws.send_json({'action': 'controller', 'value': 'guarded_score'})
                msg = receive(ws, lambda m: m['type'] == 'update' and m['controller'] == 'guarded_score')
                self.assertEqual(msg['controller_label'], 'Guarded score (no model)')
                self.assertEqual(msg['since_switch']['ticks'], 1)
                ws.send_json({'action': 'controller', 'value': 'bogus'})
                self.assertIn('unknown controller', receive(ws, lambda m: m['type'] == 'error')['message'])
            self.assertIsInstance(sim_state['rerouter'], GuardedRerouter)

            # The slider threshold drives guarded_gnn only; guarded_score keeps the label threshold.
            client.post('/api/config', json={'threshold': .5}); time.sleep(.1)
            self.assertEqual(sim_state['rerouter'].threshold, SCORE_THRESHOLD)
            if ckpt is not None:
                r = client.post('/api/controller', json={'controller': 'guarded_gnn'})
                self.assertEqual(r.status_code, 200)
                client.post('/api/config', json={'threshold': .9}); time.sleep(.1)
                self.assertEqual(sim_state['rerouter'].threshold, .9)

            self.assertEqual(client.post('/api/controller', json={'controller': 'reactive'}).status_code, 422)
            client.post('/api/stop')
            model, sim_state['model'] = sim_state['model'], None
            try:
                self.assertEqual(client.post('/api/controller', json={'controller': 'guarded_gnn'}).status_code, 409)
            finally:
                sim_state['model'] = model
            self.assertEqual(client.post('/api/controller', json={'controller': 'ecmp'}).json()['since_switch'], zero)
            self.assertTrue(on_ecmp())

if __name__=='__main__': unittest.main()
