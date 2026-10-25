"""Paired and family-clustered bootstrap used for the capacity-check term ablation (rev5).

boot() and compare() are copied verbatim from summarize_all.py (which cannot be imported: it rewrites
results/summary.json at import time). tests/test_capacity_terms.py checks that they reproduce the
saved summary.json and rev4_c1_summary.json intervals.

  pairing unit      (family, traffic seed); compare() orders pairs by sorted(key), i.e. family name
                    alphabetically, then seed
  difference        d = A - B in packets (A = variant, B = reference): positive = more drops than reference
  pair bootstrap    2,000 resamples of the n differences with replacement, statistic = mean,
                    np.random.default_rng(42) reset on every call, percentile 95% (and 90%) interval
  cluster bootstrap two-stage: resample the families (clusters) with replacement, then the seeds within each
                    drawn family; statistic = mean of the family means (equals the overall mean only when every
                    family has the same number of seeds, which callers must assert)
  scaling           every value is divided by the mean ECMP total drops and multiplied by 100 (points)
"""
import numpy as np


def boot(d, draws=2000, seed=42, cluster=False):
    """d: {(family, seed): value}. Returns bootstrap distribution of the mean."""
    rng = np.random.default_rng(seed)
    if not cluster:
        v = np.array(list(d.values()))
        return v[rng.integers(0, len(v), size=(draws, len(v)))].mean(1)
    fams = sorted({f for f, _ in d})
    byf = {f: np.array([d[(f, s)] for (ff, s) in d if ff == f]) for f in fams}
    out = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, len(fams), size=len(fams))
        out[i] = np.mean([byf[fams[j]][rng.integers(0, len(byf[fams[j]]), size=len(byf[fams[j]]))].mean() for j in pick])
    return out


def compare(A, B, base_mean, key='total_drops'):
    """paired difference A-B as % of ECMP mean, with pair and cluster 95% / 90% CIs."""
    common = sorted(set(A) & set(B))
    d = {k: A[k][key] - B[k][key] for k in common}
    m = np.mean(list(d.values()))
    pb, cb = boot(d), boot(d, cluster=True)
    f = lambda x: (np.array(x) / base_mean * 100).round(3).tolist()
    return dict(n=len(common), mean=float(m), mean_pct=float(m / base_mean * 100),
                ci95_pct=f(np.percentile(pb, [2.5, 97.5])), ci90_pct=f(np.percentile(pb, [5, 95])),
                ci95_cluster_pct=f(np.percentile(cb, [2.5, 97.5])), ci90_cluster_pct=f(np.percentile(cb, [5, 95])),
                wins=int(sum(v < 0 for v in d.values())), ties=int(sum(v == 0 for v in d.values())),
                losses=int(sum(v > 0 for v in d.values())))
