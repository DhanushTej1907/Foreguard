"""Forecast metrics shared by training, evaluation and the research benchmark.

Decision rule everywhere: predict congested iff score > threshold (the same strict
comparison the guarded controller uses). Onset windows are link-windows that are not
congested at the last observed tick (last_labels == 0).
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def tune_threshold(y, scores):
    """Threshold maximizing F1 on (y, scores) under the rule `score > threshold`.

    Returns the midpoint between the chosen cut value and the next lower distinct score,
    so ties are never split. Returns 0.5 if y has no positives.
    """
    y = np.asarray(y, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    positives = y.sum()
    if positives == 0:
        return 0.5
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # Last index of each block of equal scores: predicting positive for s >= block value.
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], s.size - 1]
    tp = np.cumsum(y_sorted)[last]
    fp = np.cumsum(1 - y_sorted)[last]
    f1 = 2 * tp / (tp + fp + positives)
    k = int(np.argmax(f1))
    cut = s_sorted[last[k]]
    lower = s_sorted[last[k] + 1] if last[k] + 1 < s.size else cut - 1.0
    return float((cut + lower) / 2)


def _block(y, s, threshold):
    n, pos = int(y.size), int(y.sum())
    pred = s > threshold
    tp, npred = int((pred & (y == 1)).sum()), int(pred.sum())
    precision = tp / npred if npred else None
    recall = tp / pos if pos else None
    f1 = 2 * tp / (npred + pos) if (npred + pos) else None
    return dict(n=n, positives=pos, positive_rate=pos / n if n else None,
                precision=precision, recall=recall, f1=f1,
                average_precision=float(average_precision_score(y, s)) if pos else None,
                roc_auc=float(roc_auc_score(y, s)) if 0 < pos < n else None,
                brier=float(brier_score_loss(y, np.clip(s, 0, 1))) if n else None)


def forecast_metrics(y, scores, threshold, last_labels, is_probability=True):
    """Overall and onset-only F1/precision/recall/AP/AUC/Brier. Undefined values are None.

    is_probability=False marks scores (e.g. continuous persistence) whose Brier score is
    computed on the clipped score rather than a calibrated probability.
    """
    y = np.asarray(y, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    onset = np.asarray(last_labels).ravel() == 0
    return dict(threshold=float(threshold), brier_is_probability=bool(is_probability),
                overall=_block(y, s, threshold), onset=_block(y[onset], s[onset], threshold))


def bootstrap_ap_difference(y, a, b, last_labels, groups, draws=2000, seed=42):
    """Scenario-level bootstrap of AP(a) - AP(b), overall and onset-only.

    y, a, b, last_labels: [n_windows, n_links]; groups: [n_windows] scenario ids.
    Each draw resamples scenarios with replacement; draws whose subset has no positive
    are skipped and counted out of valid_draws.
    """
    y, a, b, last = (np.asarray(x, dtype=float) for x in (y, a, b, last_labels))
    groups = np.asarray(groups)
    ids = np.unique(groups)
    rows = {i: np.nonzero(groups == i)[0] for i in ids}
    rng = np.random.default_rng(seed)
    picks = [rng.choice(ids, size=len(ids), replace=True) for _ in range(draws)]
    out = {}
    for name, mask in [("overall", np.ones_like(y, dtype=bool)), ("onset", last == 0)]:
        def diff(sel_rows):
            m = mask[sel_rows]
            ys = y[sel_rows][m]
            if ys.sum() == 0:
                return None
            return (average_precision_score(ys, a[sel_rows][m])
                    - average_precision_score(ys, b[sel_rows][m]))
        point = diff(np.arange(len(groups)))
        diffs = [d for d in (diff(np.concatenate([rows[i] for i in p])) for p in picks) if d is not None]
        out[name] = dict(point=None if point is None else float(point),
                         ci95=[float(v) for v in np.percentile(diffs, [2.5, 97.5])] if diffs else None,
                         valid_draws=len(diffs), draws=draws, scenarios=int(len(ids)))
    return out
