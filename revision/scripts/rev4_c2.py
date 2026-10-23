"""Concern 2: route changes, reordering and delay of the control signals (instrumented runs, unseen seeds).
Adds oracle (future score / future label) and static-random runs to results/rev4_c1.jsonl (mode 'exact')."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, os, sys, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev4_c1 as C

def tasks():
    T = []
    for fi, fam in enumerate(C.FAM):
        for s in C.SEEDS:
            base = dict(family=fam, seed=s, noise_seed=1000 * fi + s, mode='exact', guard={})
            T.append(dict(base, variant='oracle_score', signal='oracle_score', threshold=0.851))
            T.append(dict(base, variant='oracle_label', signal='oracle_label', threshold=0.5))
            T.append(dict(base, variant='static', signal='static', threshold=0.851))
    for t in T:
        t['id'] = '|'.join(str(t.get(k)) for k in ('mode', 'variant', 'mseed', 'family', 'seed'))
    return T

if __name__ == '__main__':
    os.environ.setdefault('NF_ROOT', NF_ROOT)
    done = {json.loads(l)['id'] for l in open(C.OUT)}
    T = [t for t in tasks() if t['id'] not in done]
    print(len(T), 'tasks', flush=True)
    with mp.Pool(2, initializer=C._init) as pool, open(C.OUT, 'a') as fh:
        for i, r in enumerate(pool.imap_unordered(C._run, T), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
    print('done')
