"""FlowBender-style baseline summary -> results/rev4_fb_summary.json"""
import json, os, sys
import numpy as np
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
sys.argv = [sys.argv[0]]
ns = {'__file__': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rev4_c4_analyze.py')}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rev4_c4_analyze.py')).read().split('S = {}')[0], ns)
load = lambda n: [json.loads(l) for l in open(os.path.join(R, n))]
fb = load('rev4_fb.jsonl'); ins = load('instrument.jsonl')
ecmp = {(r['family'], r['seed']): r for r in ins if r['variant'] == 'ecmp'}
gs = {(r['family'], r['seed']): r for r in ins if r['variant'] == 'guarded_score'}
E = np.mean([r['total_drops'] for r in ecmp.values()]); EF = np.mean([r['fabric_link_drops'] for r in ecmp.values()])
def boot(d):
    v = np.array([d[k] for k in sorted(d)]); rng = np.random.default_rng(42)
    return np.percentile(v[rng.integers(0, len(v), (2000, len(v)))].mean(1), [2.5, 97.5]).round(2).tolist()
S = {'open': {}, 'closed': {}}
for v in ('fb_default', 'fb_aggressive'):
    rows = {(r['family'], r['seed']): r for r in fb if r['mode'] == 'open' and r['variant'] == v}
    m = lambda k: float(np.mean([r[k] for r in rows.values()]))
    d = {c: (rows[c]['total_drops'] - gs[c]['total_drops']) / E * 100 for c in rows}
    S['open'][v] = dict(total_vs_ecmp=(m('total_drops') / E - 1) * 100, fabric_vs_ecmp=(m('fabric_link_drops') / EF - 1) * 100,
                        host_per_run=m('host_link_drops'), delivered_per_tick=m('total_delivered') / 150,
                        mean_ms=m('delivered_mean_latency_us') / 1000, p99_ms=m('delivered_p99_latency_us') / 1000, reroutes=m('reroutes'),
                        reordered_pct=float(sum(r['reordered_packets'] for r in rows.values()) / sum(r['total_delivered'] for r in rows.values()) * 100),
                        flows_reordered_pct=float(np.mean([r['flows_with_reordering'] / r['n_flows'] for r in rows.values()]) * 100),
                        vs_guarded_score_pp=float(np.mean(list(d.values()))), vs_guarded_score_ci=boot(d))
base = [r for r in load('closedloop.jsonl') if r['variant'] in ('ecmp', 'guarded_score')]
pen = [r for r in load('rev4_c4.jsonl') if r['variant'] in ('ecmp', 'guarded_score')]
for pflag, ref in ((False, base), (True, pen)):
    rows = ref + [dict(r, variant='fb_default') for r in fb if r['mode'] == 'closed' and r['penalty'] == pflag]
    C = ns['cells'](rows); keys = sorted(C['fb_default']); st = ns['stats'](C['fb_default'], keys)
    S['closed']['penalty' if pflag else 'no_penalty'] = dict(fct_mean=float(st['mean']), fct_p99=float(st['p99']),
        vs_ecmp=ns['compare'](C, 'fb_default', 'ecmp'), vs_guarded_score=ns['compare'](C, 'fb_default', 'guarded_score'),
        spur_pct=float(sum(C['fb_default'][k]['spur'] for k in keys) / sum(C['fb_default'][k]['off'] for k in keys) * 100),
        reroutes=float(np.mean([C['fb_default'][k]['rr'] for k in keys])))
json.dump(S, open(os.path.join(R, 'rev4_fb_summary.json'), 'w'), indent=1, default=float)
print(json.dumps(S, indent=1, default=float))
