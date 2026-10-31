"""Concern 4 follow-up: family-clustered bootstrap for closed-loop FCT contrasts (revision/scripts/c4_clustered.py).

The cell draws are the c4_analyze index sequence; the clustered draws follow summarize_all.boot(cluster=True) (same rng
call order: families first, then seeds within each drawn family, independently for a repeated family); a constant
per-flow shift gives a degenerate interval at the shift; per-family contributions add up to the pooled contrast;
importing the script writes nothing; the saved output passed its gate."""
import ast, json, os, sys, unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'revision', 'scripts')
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
C4 = os.path.join(ROOT, 'revision', 'results', 'c4')
_before = {f: os.path.getmtime(os.path.join(C4, f)) for f in os.listdir(C4)}
import c4_clustered as cc  # noqa: E402

FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
KEYS = sorted((f, s) for f in FAM for s in range(3000, 3005))


def summarize_all_boot():
    """summarize_all.boot, taken from the source (that module writes summary.json at import, so it is not imported)."""
    with open(os.path.join(SCRIPTS, 'summarize_all.py')) as fh:
        src = fh.read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'boot')
    ns = {'np': np}
    exec(ast.get_source_segment(src, fn), ns)
    return ns['boot']


def synthetic_cells(shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    d = {}
    for k in KEYS:
        x = rng.integers(5, 300, size=int(rng.integers(10, 40))).astype(float)
        d[k] = dict(cens=x + shift, comp=x + shift)
    return d


class TestDraws(unittest.TestCase):
    def test_cell_draws_match_c4_analyze_sequence(self):
        rng = np.random.default_rng(42)
        ref = [[KEYS[j] for j in rng.integers(0, 30, 30)] for _ in range(cc.DRAWS)]
        self.assertEqual(cc.cell_draws(KEYS), ref)

    def test_cluster_draws_match_summarize_all_rng(self):
        vals = {k: float(i * 7 % 13) + i / 10 for i, k in enumerate(KEYS)}
        ref = summarize_all_boot()(vals, draws=500, seed=42, cluster=True)
        mine = [np.mean([vals[k] for k in ks]) for ks in cc.cluster_draws(KEYS, draws=500, seed=42)]
        np.testing.assert_allclose(mine, ref, rtol=0, atol=1e-12)   # equal block sizes: mean of means = pooled mean

    def test_cluster_structure(self):
        draws = cc.cluster_draws(KEYS, draws=300)
        repeated_differ = 0
        for ks in draws:
            self.assertEqual(len(ks), 30)
            blocks = [ks[i:i + 5] for i in range(0, 30, 5)]
            for b in blocks:
                self.assertEqual(len({f for f, _ in b}), 1)            # one family per block
            fams = [b[0][0] for b in blocks]
            for f in set(fams):
                bs = [tuple(b) for b in blocks if b[0][0] == f]
                if len(bs) > 1 and len(set(bs)) > 1:
                    repeated_differ += 1
        self.assertGreater(repeated_differ, 100)                       # repeated families get independent draws
        self.assertLess(np.mean([len({k[0] for k in ks}) for ks in draws]), 5)


class TestContrast(unittest.TestCase):
    def test_constant_shift(self):
        a, b = synthetic_cells(shift=3.0), synthetic_cells()
        C = {'a': a, 'b': b}
        P = {v: dict(mean=float(cc.stats(C[v], KEYS)['mean']), p99=float(cc.stats(C[v], KEYS)['p99'])) for v in C}
        dc, dk = cc.cell_draws(KEYS)[:200], cc.cluster_draws(KEYS, draws=200)
        Dc = {v: cc.distribution(C[v], dc) for v in C}
        Dk = {v: cc.distribution(C[v], dk) for v in C}
        out = cc.contrast(P, Dc, Dk, 'a', 'b')
        for m in ('mean', 'p99'):
            self.assertAlmostEqual(out[m]['diff'], 3.0, places=9)
            for k in ('cell', 'clustered'):
                np.testing.assert_allclose(out[m][k]['diff_ci'], [3.0, 3.0], atol=1e-9)
                self.assertEqual(out[m][k]['verdict'], '+')
            self.assertFalse(out[m]['verdict_change'])

    def test_family_contributions_sum_to_pooled(self):
        a, b = synthetic_cells(seed=1), synthetic_cells(seed=1)
        for k in KEYS:
            b[k] = dict(cens=b[k]['cens'] * (1.1 if k[0] == 'mixed' else 1.0), comp=b[k]['comp'])
        C = {'a': a, 'guarded_score': b}
        fam = cc.per_family(C, KEYS, 'a')
        pooled = cc.stats(a, KEYS)['mean'] - cc.stats(b, KEYS)['mean']
        self.assertAlmostEqual(sum(x['contribution'] for x in fam.values()), pooled, places=9)
        self.assertAlmostEqual(sum(x['contribution_share'] for x in fam.values()), 1.0, places=9)
        self.assertAlmostEqual(fam['mixed']['contribution_share'], 1.0, places=9)
        self.assertEqual(fam['uniform']['diff'], 0.0)


class TestOutputs(unittest.TestCase):
    def test_import_writes_nothing(self):
        self.assertEqual({f: os.path.getmtime(os.path.join(C4, f)) for f in os.listdir(C4)}, _before)

    def test_saved_gate_passed(self):
        path = os.path.join(C4, 'clustered.json')
        if not os.path.exists(path):
            self.skipTest('clustered.json not generated')
        with open(path) as fh:
            o = json.load(fh)
        self.assertTrue(o['gate']['ok'])
        self.assertGreater(o['gate']['values_compared'], 2000)
        self.assertEqual(sorted(o['settings']), sorted(cc.SETTINGS))


if __name__ == '__main__':
    unittest.main()
