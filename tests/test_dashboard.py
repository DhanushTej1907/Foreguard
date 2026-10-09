import unittest
import time
import torch
from pathlib import Path
from config import CHECKPOINT_OUT
from model.gnn_model import check_checkpoint_schema
from fastapi.testclient import TestClient
from dashboard.server import app, sim_state

class DashboardTests(unittest.TestCase):
    def test_config_start_stop_spike_and_shared_clock(self):
        torch.set_num_threads(2)
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(client.post('/api/config',json={'speed':0}).status_code,422)
            self.assertEqual(client.post('/api/config',json={'threshold':2}).status_code,422)
            self.assertEqual(client.post('/api/config',json={'speed':1,'threshold':.8}).status_code,200)
            self.assertEqual(client.post('/api/start').status_code,200)
            # Only a current-schema, current-label-mode checkpoint at CHECKPOINT_OUT may load;
            # otherwise the dashboard falls back to ECMP.
            path=Path(CHECKPOINT_OUT); current=False
            if path.exists():
                ckpt=torch.load(path,map_location='cpu',weights_only=True)
                try: check_checkpoint_schema(ckpt); current=True
                except ValueError: pass
            status=client.get('/api/status').json()
            self.assertEqual(status['model_loaded'],current)
            if current: self.assertEqual(sim_state['threshold'],ckpt['alert_threshold'])
            else: self.assertEqual(status['controller'],'static ECMP')
            time.sleep(.3)
            before=client.get('/api/status').json()['tick']
            self.assertGreater(before,0) # runs without any viewers
            with client.websocket_connect('/ws') as a:
                with client.websocket_connect('/ws') as b:
                    self.assertEqual(a.receive_json()['type'],'topology')
                    self.assertEqual(b.receive_json()['type'],'topology')
                    self.assertEqual(a.receive_json()['metrics']['tick'],b.receive_json()['metrics']['tick'])
            self.assertEqual(client.post('/api/spike').json()['status'],'spike_injected')
            ids=[f.flow_id for f in sim_state['flows']]
            self.assertEqual(len(ids),len(set(ids)))
            client.post('/api/stop')
            tick=client.get('/api/status').json()['tick']; time.sleep(.15)
            self.assertEqual(client.get('/api/status').json()['tick'],tick)

if __name__=='__main__': unittest.main()
