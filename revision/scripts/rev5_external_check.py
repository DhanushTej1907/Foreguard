"""Final validation: per-run comparison of rev5_capterms.jsonl with an independently produced rev5_cap.jsonl.

The external file arrives in a ZIP and is treated as untrusted data: it is extracted into a new, empty
directory, nothing in it is run or imported, and only the JSONL (plus a metadata JSON next to it, if any) is read.

  python -I revision/scripts/rev5_external_check.py extract --zip <file.zip> --dest <new empty dir>
  python -I revision/scripts/rev5_external_check.py inspect --file <.../revision/results/rev5_cap.jsonl>
  python -I revision/scripts/rev5_external_check.py compare --file <...> --mapping <mapping.json>

inspect lists field names, variant / signal labels and per-field types, and writes nothing. compare needs an
explicit mapping file, written after inspection and confirmed by the user:
  {"variant": {"<their label>": "A|B|Cp|D|E|nocheck", ...}, "signal": {"<their label>": "score|random"},
   "fields": {"<their field>": "<our field>", ...},            # optional renames; identity otherwise
   "variant_field": "variant", "signal_field": "signal"}     # where their labels live
Labels without a mapping are reported, never guessed. compare writes results/rev5_capterms_external_check.json.

Rules: key = (family, seed, signal, variant). Inputs (threshold, rng_seed, resolved guard parameters,
total_offered, n_flows) are compared first; a run whose recorded inputs differ is reported as non-equivalent and is
counted neither as a reproduction failure nor as a success. For equivalent runs every deterministic outcome field
must be exactly equal; every inequality is listed with both values (float gaps within 1e-12 relative are
labelled representation-only but still listed). Differences in attribution / audit fields count as expected
method differences only when the matching A run agrees on every outcome field.
"""
import argparse, json, os, sys, zipfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), 'results')
OURS = os.path.join(RESULTS, 'rev5_capterms.jsonl')
OURS_AUDIT = os.path.join(RESULTS, 'rev5_capterms_audit.jsonl')
OURS_VALID = os.path.join(RESULTS, 'rev5_capterms_validation.jsonl')
OUT = os.path.join(RESULTS, 'rev5_capterms_external_check.json')
INPUT_FIELDS = ['threshold', 'rng_seed', 'total_offered', 'n_flows']
OUTCOME_INT = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'unroutable_drops', 'total_delivered', 'reroutes',
               'reordered_packets', 'flows_with_reordering', 'rejections']
OUTCOME_FLOAT = ['delivered_mean_latency_us', 'delivered_p99_latency_us', 'reordered_fraction', 'avg_throughput']
OUTCOME_OTHER = ['decision_digest', 'signal_digest', 'packet_conservation', 'drop_split_ok']
ATTRIBUTION = ['cap_audit']


def extract(zpath, dest):
    if os.path.exists(dest) and os.listdir(dest):
        raise SystemExit(f'{dest} is not empty')
    os.makedirs(dest, exist_ok=True)
    root = os.path.realpath(dest)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = os.path.realpath(os.path.join(dest, m.filename))
            if not target.startswith(root + os.sep) and target != root:
                raise SystemExit(f'unsafe member path {m.filename!r}')
        z.extractall(dest)
        print('\n'.join(m.filename for m in z.infolist()))


def read_jsonl(path):
    rows, bad = [], 0
    with open(path, encoding='utf8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def inspect(path):
    rows, bad = read_jsonl(path)
    types = defaultdict(Counter)
    for r in rows:
        for k, v in r.items():
            types[k][type(v).__name__] += 1
    labels = {k: Counter(str(r.get(k)) for r in rows) for k in ('variant', 'config', 'signal', 'family', 'mode', 'group')
              if any(k in r for r in rows)}
    print(json.dumps(dict(rows=len(rows), unparsable_lines=bad, fields={k: dict(v) for k, v in types.items()},
                          labels={k: dict(v) for k, v in labels.items()}), indent=1))
    meta = [f for f in os.listdir(os.path.dirname(os.path.abspath(path))) if f.endswith('.json')]
    print('metadata files next to it:', meta)


def _ne(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if a == b:
            return None
        return 'representation-only' if abs(a - b) <= 1e-12 * max(1.0, abs(b)) else 'substantive'
    return None if a == b else 'exact'


def compare(path, mapping_path):
    M = json.load(open(mapping_path))
    vf, sf = M.get('variant_field', 'variant'), M.get('signal_field', 'signal')
    ren = M.get('fields', {})
    theirs_raw, bad_lines = read_jsonl(path)
    unmapped = Counter()
    theirs, dup = {}, []
    for r in theirs_raw:
        r = {ren.get(k, k): v for k, v in r.items()}
        v, s = M['variant'].get(str(r.get(vf))), M['signal'].get(str(r.get(sf)))
        if v is None or s is None:
            unmapped[(str(r.get(vf)), str(r.get(sf)))] += 1
            continue
        k = (r['family'], int(r['seed']), s, v)
        if k in theirs:
            dup.append(k)
        theirs[k] = r
    # campaign rows, plus the 60 ECMP validation runs (signal None -> 'none') for external ECMP rows
    ours = {(r['family'], r['seed'], r['signal'] or 'none', r['config']): r
            for p in (OURS, OURS_VALID) for r in read_jsonl(p)[0] if 'error' not in r}
    audit = {(r['family'], r['seed'], r['signal']): r for r in read_jsonl(OURS_AUDIT)[0] if 'error' not in r}
    common = sorted(set(ours) & set(theirs))
    rep = dict(status='done', external_file=os.path.abspath(path), mapping=M, unparsable_lines=bad_lines,
               unmapped_labels={f'{a}|{b}': n for (a, b), n in unmapped.items()}, duplicates=[list(k) for k in dup],
               missing_in_external=[list(k) for k in sorted(set(ours) - set(theirs))],
               extra_in_external=[list(k) for k in sorted(set(theirs) - set(ours))], matched=len(common),
               non_equivalent_inputs=[], outcome_mismatches=[], attribution_differences=[], fields_not_in_external=[])
    absent, absent_ours = Counter(), Counter()
    for k in common:
        o, t = ours[k], theirs[k]
        inp = []
        for f in INPUT_FIELDS:
            if f in t and f not in o:
                absent_ours[f] += 1
            elif f in t and _ne(t[f], o[f]):
                inp.append(dict(field=f, ours=o[f], external=t[f]))
        if 'guard_params' in t and t['guard_params'] != o['guard_params']:
            inp.append(dict(field='guard_params', ours=o['guard_params'], external=t['guard_params']))
        if inp:
            rep['non_equivalent_inputs'].append(dict(key=list(k), differences=inp))
            continue
        diffs = []
        for f in OUTCOME_INT + OUTCOME_FLOAT + OUTCOME_OTHER:
            if f not in t:
                absent[f] += 1
                continue
            if f not in o:
                absent_ours[f] += 1
                continue
            kind = _ne(o.get(f), t[f])
            if kind:
                diffs.append(dict(field=f, ours=o.get(f), external=t[f], kind=kind))
        if diffs:
            rep['outcome_mismatches'].append(dict(key=list(k), differences=diffs, classification='unexpected'))
        if k[3] == 'A' and any(f in t for f in ATTRIBUTION):
            a = audit.get(k[:3])
            for f in ATTRIBUTION:
                if f in t and a is not None and t[f] != a.get(f):
                    rep['attribution_differences'].append(dict(
                        key=list(k), field=f, ours=a.get(f), external=t[f],
                        classification='expected method difference' if not diffs else
                        'unexplained (A outcomes differ)'))
    rep['fields_not_in_external'] = dict(absent)
    rep['fields_not_in_ours'] = dict(absent_ours)       # e.g. reordering is not recorded by the plain ECMP engine
    rep['input_fields_checkable'] = sorted({f for k in common for f in INPUT_FIELDS + ['guard_params'] if f in theirs[k]})
    rep['summary'] = dict(matched=len(common), equivalent=len(common) - len(rep['non_equivalent_inputs']),
                          runs_with_outcome_mismatch=len(rep['outcome_mismatches']),
                          all_equivalent_runs_identical=not rep['outcome_mismatches'])
    with open(OUT, 'w') as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep['summary'], indent=1))
    print(f'unmapped {dict(unmapped)}; missing {len(rep["missing_in_external"])}; extra {len(rep["extra_in_external"])}; '
          f'non-equivalent {len(rep["non_equivalent_inputs"])}; absent fields {dict(absent)} -> {OUT}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['extract', 'inspect', 'compare'])
    ap.add_argument('--zip')
    ap.add_argument('--dest')
    ap.add_argument('--file')
    ap.add_argument('--mapping')
    a = ap.parse_args()
    if a.what == 'extract':
        extract(a.zip, a.dest)
    elif a.what == 'inspect':
        inspect(a.file)
    else:
        compare(a.file, a.mapping)


if __name__ == '__main__':
    main()
