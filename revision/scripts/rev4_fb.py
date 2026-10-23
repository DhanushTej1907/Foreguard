"""FlowBender-style baseline: open loop (instrumented) on benchmark seeds 3000-3009 for two settings, and
closed loop with and without the reordering penalty on seeds 3000-3004 (default setting)."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os, sys, traceback
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results', 'rev4_fb.jsonl')
FAM = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
SET = {'fb_default': dict(k_frac=0.1, n_consec=2), 'fb_aggressive': dict(k_frac=0.05, n_consec=1)}
def tasks():
    T = []
    for fi, f in enumerate(FAM):
        for s in range(3000, 3010):
            for v, kw in SET.items():
                T.append(dict(id=f'open|{v}|{f}|{s}', mode='open', variant=v, guard=kw, family=f, seed=s, rs=1000 * fi + s))
        for s in range(3000, 3005):
            for pen in (False, True):
                T.append(dict(id=f'closed{int(pen)}|fb_default|{f}|{s}', mode='closed', penalty=pen, variant='fb_default',
                              guard=SET['fb_default'], family=f, seed=s, rs=1000 * fi + s))
    return T
_W = {}
def _init():
    sys.path.insert(0, HERE)
    import nf_exp
    from simulation.topology import build_fat_tree
    _W['X'], _W['g'] = nf_exp, build_fat_tree()
def _run(t):
    try:
        X, g = _W['X'], _W['g']
        from simulation.traffic_generator import TrafficGenerator
        flows = TrafficGenerator(g, seed=t['seed']).generate_scenario(total_ticks=150, pattern=t['family'])
        if t['mode'] == 'open':
            r = X.run_case_x(g, flows, 150, 'flowbender', guard=dict(t['guard']), instrument=True, rng_seed=t['rs'])
        else:
            r = X.run_case_x(g, flows, 150, 'flowbender', guard=dict(t['guard']), closed_loop=True,
                             reorder_penalty=t['penalty'], rng_seed=t['rs'])
        out = {k: t[k] for k in ('id', 'mode', 'variant', 'family', 'seed')}
        out['penalty'] = t.get('penalty')
        out.update({k: v for k, v in r.items() if k in ('total_drops', 'fabric_link_drops', 'host_link_drops', 'total_offered',
                    'total_delivered', 'delivered_mean_latency_us', 'delivered_p99_latency_us', 'reroutes', 'reordered_packets',
                    'flows_with_reordering', 'n_flows', 'completed', 'incomplete', 'fct_mean', 'fct_p99', 'retx_packets',
                    'offered_unique', 'spurious_packets', 'makespan', 'per_flow')})
        return out
    except Exception:
        return dict(id=t['id'], error=traceback.format_exc())
if __name__ == '__main__':
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    done = {json.loads(l)['id'] for l in open(OUT)} if os.path.exists(OUT) else set()
    T = [t for t in tasks() if t['id'] not in done]
    print(len(T), 'tasks', flush=True)
    with mp.Pool(2, initializer=_init) as pool, open(OUT, 'a') as fh:
        for r in pool.imap_unordered(_run, T):
            fh.write(json.dumps(r) + '\n'); fh.flush()
    print('done')
