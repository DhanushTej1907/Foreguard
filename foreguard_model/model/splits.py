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

def stratified_split(samples, n_val=1, n_test=1, seed=42):
    """Per traffic family: n_test test, n_val validation, the rest training scenarios,
    so every family appears in every partition (30 scenarios -> 18/6/6)."""
    if not samples or any('scenario_id' not in s or 'pattern' not in s for s in samples):
        raise ValueError('stratified split needs scenario_id and pattern on every sample')
    pattern = {s['scenario_id']: s['pattern'] for s in samples}
    rng = random.Random(seed); partitions = [set(), set(), set()]
    for fam in sorted(set(pattern.values())):
        ids = sorted(i for i, p in pattern.items() if p == fam)
        if len(ids) < n_val + n_test + 1:
            raise ValueError(f'family {fam!r} needs at least {n_val + n_test + 1} scenarios')
        rng.shuffle(ids)
        partitions[2].update(ids[:n_test]); partitions[1].update(ids[n_test:n_test + n_val])
        partitions[0].update(ids[n_test + n_val:])
    return tuple([s for s in samples if s['scenario_id'] in ids] for ids in partitions)

def holdout_split(samples, family, seed=42):
    """Test on every scenario of traffic `family`; validate on one scenario per
    remaining family; train on the rest."""
    if not samples or any('scenario_id' not in s or 'pattern' not in s for s in samples):
        raise ValueError('holdout split needs scenario_id and pattern on every sample')
    pattern = {s['scenario_id']: s['pattern'] for s in samples}
    test_ids = {i for i, p in pattern.items() if p == family}
    if not test_ids:
        raise ValueError(f'no scenarios of family {family!r}')
    rng = random.Random(seed); val_ids = set()
    for fam in sorted(set(pattern.values()) - {family}):
        ids = sorted(i for i, p in pattern.items() if p == fam)
        if len(ids) < 2:
            raise ValueError(f'family {fam!r} needs at least two scenarios')
        val_ids.add(rng.choice(ids))
    partitions = [set(pattern) - test_ids - val_ids, val_ids, test_ids]
    return tuple([s for s in samples if s['scenario_id'] in ids] for ids in partitions)
