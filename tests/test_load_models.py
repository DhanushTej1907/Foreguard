"""revision/scripts/nf_exp.load_models: defaults unchanged; explicit stem / tag / calibration never touch the default entries."""
import os, sys, unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'revision', 'scripts')
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
import nf_exp  # noqa: E402

CK = os.path.join(ROOT, 'checkpoints')


def stored(kind, k):
    name = 'best_model_v3' + ('' if k == 0 else f'_seed{k}') + ('_lstm_only' if kind == 'lstm_only' else '') + '.pt'
    ck = torch.load(os.path.join(CK, name), map_location='cpu', weights_only=False)
    return os.path.join(CK, name), float(ck['temperature']), float(ck['alert_threshold'])


class TestLoadModels(unittest.TestCase):
    def setUp(self):
        self._saved = dict(nf_exp._MODELS)
        nf_exp._MODELS.clear()

    def tearDown(self):
        nf_exp._MODELS.clear()
        nf_exp._MODELS.update(self._saved)

    def test_default_paths_keys_and_calibration(self):
        nf_exp.load_models(threads=1)
        self.assertEqual(sorted(nf_exp._MODELS), sorted((kind, k) for kind in ('gnn', 'lstm_only') for k in (0, 1, 2)))
        for (kind, k), mm in nf_exp._MODELS.items():
            path, T, th = stored(kind, k)
            self.assertEqual(os.path.normcase(mm['path']), os.path.normcase(path))
            self.assertEqual(mm['temperature'], T)
            self.assertEqual(mm['threshold'], th)
            self.assertEqual(float(mm['model'].temperature), T)

    def test_tagged_load_leaves_default_entries(self):
        nf_exp.load_models(kinds=('gnn',), threads=1)
        default = nf_exp._MODELS[('gnn', 0)]
        # same files under an explicit stem + tag, with a calibration override for seed 1 only
        nf_exp.load_models(kinds=('gnn',), threads=1, stem=os.path.join(CK, 'best_model_v3'), tag='x',
                           calibration={('gnn', 1): (1.5, 0.42)})
        self.assertIs(nf_exp._MODELS[('gnn', 0)], default)
        self.assertEqual(nf_exp._MODELS[('x', 'gnn', 1)]['threshold'], 0.42)
        self.assertEqual(nf_exp._MODELS[('x', 'gnn', 1)]['temperature'], 1.5)
        _, T0, th0 = stored('gnn', 0)
        self.assertEqual(nf_exp._MODELS[('x', 'gnn', 0)]['threshold'], th0)
        self.assertEqual(nf_exp._MODELS[('x', 'gnn', 0)]['temperature'], T0)
        self.assertEqual(nf_exp._MODELS[('gnn', 1)]['threshold'], stored('gnn', 1)[2])

    def test_missing_stem_raises(self):
        with self.assertRaises(FileNotFoundError):
            nf_exp.load_models(kinds=('gnn',), seeds=(0,), threads=1, stem=os.path.join(CK, 'no_such_model'), tag='y')
        self.assertNotIn(('y', 'gnn', 0), nf_exp._MODELS)


if __name__ == '__main__':
    unittest.main()
