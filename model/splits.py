"""Group split independent of PyTorch, with no overlapping scenario windows."""
import random

def split_scenarios(samples, train_frac=.7, val_frac=.15, seed=42):
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError('fractions must leave nonempty train, validation and test sets')
    if not samples or any('scenario_id' not in s for s in samples):
        raise ValueError('regenerate data with scenario_id; legacy windows are unsupported')
    groups = sorted({s['scenario_id'] for s in samples})
    if len(groups) < 3:
        raise ValueError('at least three independent scenarios required')
    random.Random(seed).shuffle(groups)
    ntrain = min(len(groups)-2, max(1, int(len(groups)*train_frac)))
    nval = min(len(groups)-ntrain-1, max(1, int(len(groups)*val_frac)))
    partitions = [set(groups[:ntrain]), set(groups[ntrain:ntrain+nval]), set(groups[ntrain+nval:])]
    return tuple([s for s in samples if s['scenario_id'] in ids] for ids in partitions)
