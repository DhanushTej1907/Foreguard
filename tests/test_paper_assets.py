import importlib.util, json, os, re, tempfile, unittest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('make_paper_assets', os.path.join(ROOT, 'tools', 'make_paper_assets.py'))
assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assets)


def tabular_rows(tex):
    """(columns in the tabular spec, columns a row spans, row) for every row of every tabular."""
    for spec, body in re.findall(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}', tex, re.S):
        ncols = len(re.findall(r'[lcr]', spec))
        for row in body.split('\\\\'):
            row = re.sub(r'\\(toprule|midrule|bottomrule|addlinespace)|\\cmidrule\([a-z]*\)\{[^}]*\}', '', row).strip()
            if not row:
                continue
            cells = re.split(r'(?<!\\)&', row)
            span = sum(int(m.group(1)) if (m := re.search(r'\\multicolumn\{(\d+)\}', c)) else 1 for c in cells)
            yield ncols, span, row


class PaperAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = cls.tmp.name
        cls.written = assets.build_all(assets.RESULTS, cls.out, sweep_path=None)
        with open(assets.RESULTS) as fh:
            cls.j = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, name):
        with open(os.path.join(self.out, name), encoding='utf-8') as fh:
            return fh.read()

    def test_every_asset_written(self):
        expected = {f'{n}.{e}' for n in assets.FIGURES for e in ('pdf', 'png')} | {'tables.tex', 'KEY_NUMBERS.md'}
        self.assertEqual({os.path.basename(p) for p in self.written}, expected)
        self.assertTrue(all(os.path.getsize(p) > 0 for p in self.written))

    def test_figures_are_single_column_300_dpi_with_embedded_truetype(self):
        for name in assets.FIGURES:
            with Image.open(os.path.join(self.out, name + '.png')) as im:
                self.assertEqual(im.size[0], 1050, name)  # exactly 3.5 in at 300 dpi
                self.assertEqual([round(d) for d in im.info['dpi']], [300, 300], name)
            with open(os.path.join(self.out, name + '.pdf'), 'rb') as fh:
                pdf = fh.read()
            self.assertNotIn(b'/Type3', pdf, name)  # IEEE PDF eXpress rejects Type 3 fonts
            self.assertIn(b'/FontFile2', pdf, name)

    def test_tables_escape_percent_and_rows_fill_their_columns(self):
        tex = self.read('tables.tex')
        body = '\n'.join(line for line in tex.splitlines() if not line.startswith('%'))
        self.assertIsNone(re.search(r'(?<!\\)%', body))  # every % sign escaped
        self.assertEqual(body.count('{'), body.count('}'))
        rows = list(tabular_rows(tex))
        self.assertEqual(len(rows), (2 + 5) + (1 + 7) + (1 + 6))  # header + data rows of the 3 tables
        for ncols, span, row in rows:
            self.assertEqual(span, ncols, row)
        for label in ('tab:forecast', 'tab:routing', 'tab:paired'):
            self.assertIn(f'\\label{{{label}}}', tex)
        # Paired table: guarded GNN vs each guard-only baseline and guarded LSTM, total and fabric.
        self.assertEqual(len(assets.paired_rows(self.j)), 6)
        self.assertIn('Every interval includes 0.', tex)

    def test_key_numbers_match_the_json(self):
        md = self.read('KEY_NUMBERS.md')
        j = self.j
        self.assertIn(f"total -{j['vs_ecmp']['guarded_score']['total_drop_reduction_percent']:.2f}%", md)
        self.assertIn(f"GNN {j['forecast']['metrics']['gnn']['summary']['overall']['average_precision']['mean']:.3f}", md)
        self.assertIn(f"p99 {j['inference_timing']['gnn']['p99_ms']:.2f} ms", md)
        # Every plain JSON path cited in backticks exists.
        cited = [p for p in re.findall(r'`([a-z_0-9.]+)`', md) if '.' in p or p in j]
        self.assertGreater(len(cited), 15)
        for path in cited:
            assets.at(j, path)
        self.assertIn('the GNN does not beat the guard-only baselines', md)


if __name__ == '__main__':
    unittest.main()
