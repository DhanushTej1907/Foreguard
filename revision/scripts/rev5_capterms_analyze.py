"""Capacity-check term ablation (Table A7): gates, validation and summary.

  python revision/scripts/rev5_capterms_analyze.py gate                  # G0 (ECMP) + G1 (180 baseline runs)
  python revision/scripts/rev5_capterms_analyze.py summary [--determinism FILE]
                                                                         # -> results/rev5_capterms_summary.json

Comparison policy for runs that repeat saved results (gates): identifiers, parameters and every integer
count must be equal; the integer-derived floats (mean / p99 delay, reordered fraction, delivered per tick)
must agree to 1e-12 relative (representation level). Any other difference is a mismatch; nothing is relaxed.
The summary is written only when all 720 campaign and 120 audit runs exist without errors (60 pairs per cell,
10 per family). Statistics: rev5_stats.compare (copied from summarize_all.py; see its docstring).
"""
from paths import RESULTS  # folder locations (see paths.py)
import argparse, json, os, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev5_capterms as C  # noqa: E402
import rev5_stats  # noqa: E402

INT_FIELDS = ['total_drops', 'fabric_link_drops', 'host_link_drops', 'unroutable_drops', 'total_offered',
              'total_delivered', 'reroutes', 'reordered_packets', 'flows_with_reordering', 'n_flows', 'rejections']
FLOAT_FIELDS = ['delivered_mean_latency_us', 'delivered_p99_latency_us', 'reordered_fraction', 'avg_throughput']
REL_TOL = 1e-12
SUMMARY = os.path.join(RESULTS, 'rev5_capterms_summary.json')
EXTERNAL = os.path.join(RESULTS, 'rev5_capterms_external_check.json')


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return [], []
    rows = [json.loads(l) for l in open(p)]
    return [r for r in rows if 'error' not in r], [r for r in rows if 'error' in r]


def keyed(rows, pred=lambda r: True):
    out = {}
    for r in rows:
        if pred(r):
            k = (r['family'], r['seed'])
            assert k not in out, f'duplicate {k}'
            out[k] = r
    return out


def diff_run(new, old, fields_int=INT_FIELDS, fields_float=FLOAT_FIELDS):
    """Field-by-field comparison on the fields both rows carry. Returns a list of mismatches."""
    out = []
    for k in fields_int:
        if k in new and k in old and new[k] != old[k]:
            out.append(dict(field=k, new=new[k], saved=old[k], kind='exact'))
    for k in fields_float:
        if k in new and k in old:
            a, b = new[k], old[k]
            if a is None or b is None:
                if a is not b:
                    out.append(dict(field=k, new=a, saved=b, kind='none'))
            elif a != b:
                rep = abs(a - b) <= REL_TOL * max(1.0, abs(b))
                out.append(dict(field=k, new=a, saved=b, kind='representation-only' if rep else 'substantive'))
    return out


def saved_refs():
    rep, _ = load('replicate.jsonl')
    c1, _ = load('rev4_c1.jsonl')
    ab, _ = load('ablation.jsonl')
    return dict(
        ecmp=keyed(rep, lambda r: r['variant'] == 'ecmp'),
        A_score=keyed(c1, lambda r: r['mode'] == 'exact' and r['variant'] == 'score'),
        A_random=keyed(c1, lambda r: r['mode'] == 'exact' and r['variant'] == 'random'),
        nocheck_score=keyed(ab, lambda r: r.get('variant') == 'no_capacity' and r.get('signal') == 'score'),
        blind_score=keyed(c1, lambda r: r['mode'] == 'blind' and r['variant'] == 'score'),
        blind_random=keyed(c1, lambda r: r['mode'] == 'blind' and r['variant'] == 'random'))


def gate():
    refs = saved_refs()
    val, val_err = load(C.OUT['validation'])
    camp, camp_err = load(C.OUT['campaign'])
    rep = dict(G0=dict(source='replicate.jsonl variant ecmp'), G1={})
    # G0
    new = keyed(val)
    mism = {f'{k[0]}|{k[1]}': d for k in refs['ecmp'] if k in new and (d := diff_run(new[k], refs['ecmp'][k]))}
    rep['G0'].update(n_expected=60, n_found=len(new), errors=len(val_err), missing=sorted(f'{f}|{s}' for f, s in
                     refs['ecmp'] if (f, s) not in new), mismatches=mism)
    rep['G0']['pass'] = len(new) == 60 and not val_err and not mism
    # G1
    for (cfg, sig), ref, src in ((('A', 'score'), refs['A_score'], 'rev4_c1.jsonl exact|score'),
                                 (('A', 'random'), refs['A_random'], 'rev4_c1.jsonl exact|random'),
                                 (('nocheck', 'score'), refs['nocheck_score'], 'ablation.jsonl no_capacity|score')):
        new = keyed(camp, lambda r: r['config'] == cfg and r['signal'] == sig)
        mism = {f'{k[0]}|{k[1]}': d for k in ref if k in new and (d := diff_run(new[k], ref[k]))}
        compared = sorted({f for k in ref if k in new for f in INT_FIELDS + FLOAT_FIELDS if f in new[k] and f in ref[k]})
        rep['G1'][f'{cfg}|{sig}'] = dict(source=src, n_found=len(new), fields_compared=compared, mismatches=mism,
                                         missing=sorted(f'{f}|{s}' for f, s in ref if (f, s) not in new),
                                         **{'pass': len(new) == 60 and not mism})
    rep['G1_pass'] = all(v['pass'] for v in rep['G1'].values()) and not camp_err
    rep['campaign_errors'] = len(camp_err)
    return rep


# ───────────────────────────── summary ─────────────────────────────────
def pct(a, b):
    return float((a / b - 1) * 100)


def cell_stats(cell, E):
    k = sorted(cell)
    m = lambda f: float(np.mean([cell[x][f] for x in k]))
    return dict(n=len(k), total_pct=pct(m('total_drops'), E['total']), fabric_pct=pct(m('fabric_link_drops'), E['fabric']),
                host_pct=pct(m('host_link_drops'), E['host']), deliv_per_tick=m('avg_throughput'),
                deliv_pct=pct(m('avg_throughput'), E['deliv']), mean_ms=m('delivered_mean_latency_us') / 1000,
                p99_ms=m('delivered_p99_latency_us') / 1000, reroutes=m('reroutes'),
                reordered_pct=float(sum(cell[x]['reordered_packets'] for x in k) /
                                    sum(cell[x]['total_delivered'] for x in k) * 100),
                flows_reordered_pct=float(np.mean([cell[x]['flows_with_reordering'] / cell[x]['n_flows'] for x in k]) * 100))


EXPECTED_DIFF = {'A': set(), 'B': {'cap_queue'}, 'Cp': {'cap_arrivals'}, 'D': {'cap_arrivals', 'cap_queue'},
                 'E': {'cap_arrivals', 'cap_queue', 'rate_mode'}, 'nocheck': {'headroom'}}
OUTCOME = INT_FIELDS + FLOAT_FIELDS + ['decision_digest', 'signal_digest', 'packet_conservation', 'drop_split_ok']


def summary(determinism=None):
    camp, camp_err = load(C.OUT['campaign'])
    aud, aud_err = load(C.OUT['audit'])
    want_c = {t['id'] for t in C.tasks('campaign')}
    want_a = {t['id'] for t in C.tasks('audit')}
    have_c, have_a = {r['id'] for r in camp}, {r['id'] for r in aud}
    missing = sorted(want_c - have_c) + sorted(want_a - have_a)
    extra = sorted(have_c - want_c) + sorted(have_a - want_a)
    dup = len(camp) - len(have_c) + len(aud) - len(have_a)
    if missing or extra or dup or camp_err or aud_err:
        print(f'INCOMPLETE: {len(missing)} missing, {len(extra)} unexpected, {dup} duplicate, '
              f'{len(camp_err)} + {len(aud_err)} error rows; summary not written')
        for i in missing[:20]:
            print('  missing', i)
        raise SystemExit(1)
    cells = defaultdict(dict)
    for r in camp:
        cells[(r['config'], r['signal'])][(r['family'], r['seed'])] = r
    for k, v in cells.items():
        assert len(v) == 60 and all(sum(1 for f, _ in v if f == fam) == 10 for fam in C.FAM), k
    audit = {(r['signal'], r['family'], r['seed']): r for r in aud}
    V = {}
    g = gate()
    V['G0_ecmp'] = g['G0']['pass']
    V['G1_baselines'] = g['G1_pass']
    # per-run integrity
    V['conservation_and_drop_split'] = all(r['packet_conservation'] and r['drop_split_ok'] for r in camp + aud)
    offered = defaultdict(set)
    for r in camp:
        offered[(r['family'], r['seed'])].add(r['total_offered'])
    V['offered_identical_per_scenario'] = all(len(v) == 1 for v in offered.values())
    gp_ok = True
    for r in camp:
        a = cells[('A', r['signal'])][(r['family'], r['seed'])]['guard_params']
        if {k for k in a if a[k] != r['guard_params'][k]} != EXPECTED_DIFF[r['config']]:
            gp_ok = False
    V['guard_params_differ_only_in_intended_keys'] = gp_ok
    V['campaign_runs_audit_off'] = all(not r['guard_params']['audit'] and 'cap_audit' not in r for r in camp)
    sigd = defaultdict(set)
    for r in camp:
        if r['signal'] == 'random':
            sigd[(r['family'], r['seed'])].add(r['signal_digest'])
    V['random_signal_identical_across_configs'] = all(len(v) == 1 and None not in v for v in sigd.values())
    # audit purity and self-consistency
    pur = []
    for (sig, fam, s), r in audit.items():
        a = cells[('A', sig)][(fam, s)]
        bad = [k for k in OUTCOME if r.get(k) != a.get(k)]
        if bad:
            pur.append(dict(run=f'{sig}|{fam}|{s}', fields=bad))
    V['audit_purity'] = not pur
    V['audit_self_mismatch_zero'] = all(r['cap_audit'].get('self_mismatch', 0) == 0 for r in aud)
    # executed variant == A  <=>  zero counterfactual disagreements on A's trajectory (B, Cp, D loosen A)
    equiv = {}
    for X in ('B', 'Cp', 'D'):
        bad = []
        for (sig, fam, s), r in audit.items():
            same = cells[(X, sig)][(fam, s)]['decision_digest'] == cells[('A', sig)][(fam, s)]['decision_digest']
            zero = r['cap_audit'].get(f'A_ne_{X}', 0) == 0
            if same != zero:
                bad.append(f'{sig}|{fam}|{s}')
        equiv[X] = bad
    V['digest_equivalence'] = {X: not b for X, b in equiv.items()}
    if determinism:
        rows = [json.loads(l) for l in open(determinism)]
        det = []
        for r in rows:
            c = next(x for x in camp if x['id'] == r['id'])
            bad = [k for k in r if k != 'wall_s' and r.get(k) != c.get(k)]
            if 'error' in r or bad:
                det.append(dict(id=r['id'], fields=bad or ['error']))
        V['determinism'] = dict(n=len(rows), identical=not det, mismatches=det)
    else:
        V['determinism'] = 'not run'
    if os.path.exists(EXTERNAL):
        x = json.load(open(EXTERNAL))
        V['external_comparison'] = dict(status=x['status'], file=os.path.basename(x['external_file']), **x['summary'],
                                        missing=len(x['missing_in_external']), extra=len(x['extra_in_external']),
                                        unmapped=x['unmapped_labels'], fields_not_in_external=x['fields_not_in_external'])
    else:
        V['external_comparison'] = 'pending'

    # ECMP reference: the saved 60 deterministic runs (G0 shows the current code reproduces them)
    refs = saved_refs()
    ec = refs['ecmp']
    E = dict(total=float(np.mean([r['total_drops'] for r in ec.values()])),
             fabric=float(np.mean([r['fabric_link_drops'] for r in ec.values()])),
             host=float(np.mean([r['host_link_drops'] for r in ec.values()])),
             deliv=float(np.mean([r['avg_throughput'] for r in ec.values()])),
             mean_ms=float(np.mean([r['delivered_mean_latency_us'] for r in ec.values()]) / 1000),
             p99_ms=float(np.mean([r['delivered_p99_latency_us'] for r in ec.values()]) / 1000), n=len(ec))
    S = dict(description='Capacity-check term ablation (paper Table A7); unseen seeds 4000-4009, six families; '
                         'outcomes from audit-off runs, attribution from audit-on A runs (counterfactual, per test, '
                         'along A\'s own trajectory).',
             ecmp_reference=dict(source='replicate.jsonl (60 ECMP runs)', **E),
             statistics=dict(pairing_unit='(family, traffic seed)', n_pairs=60, difference='variant - reference, '
                             'packets, in points of mean ECMP total drops; positive = more drops', resamples=2000,
                             rng='numpy default_rng(42), reset per comparison', interval='percentile 95% (90% also)',
                             cluster='two-stage: 6 families with replacement, then 10 seeds within each drawn family; '
                                     'mean of family means', order='sorted (family name, seed)'),
             cells={}, comparisons={}, attribution={}, validation=V, gates=g)
    for (cfg, sig), cell in sorted(cells.items()):
        S['cells'][f'{cfg}|{sig}'] = cell_stats(cell, E)
    cmp = S['comparisons']
    for sig in C.SIGNALS:
        A = cells[('A', sig)]
        for X in ('B', 'Cp', 'D', 'E', 'nocheck'):
            cmp[f'{X}_vs_A|{sig}'] = rev5_stats.compare(cells[(X, sig)], A, E['total'])
        cmp[f'D_vs_nocheck|{sig}'] = rev5_stats.compare(cells[('D', sig)], cells[('nocheck', sig)], E['total'])
        cmp[f'E_vs_D|{sig}'] = rev5_stats.compare(cells[('E', sig)], cells[('D', sig)], E['total'])
        if g['G1_pass']:
            saved_blind = refs[f'blind_{sig}']
            cmp[f'E_vs_saved_blind_with_terms|{sig}'] = dict(
                rev5_stats.compare(cells[('E', sig)], saved_blind, E['total']),
                note='diagnostic: saved rev4_c1 mode blind keeps the queue and arrival terms (Table A4)')
    for sig in C.SIGNALS:
        rows = [r for (s, _, _), r in audit.items() if s == sig]
        tot = defaultdict(int)
        for r in rows:
            for k, v in r['cap_audit'].items():
                tot[k] += v
        n = tot['tests']
        share = lambda k: (tot[k] / n * 100) if n else None
        per_run = lambda k: [r['cap_audit'].get(k, 0) / r['cap_audit']['tests'] * 100 for r in rows if r['cap_audit'].get('tests')]
        S['attribution'][sig] = dict(
            kind='counterfactual: reduced checks evaluated at every capacity test of the full check (A), '
                 'on A\'s state; pooled over 60 runs', audit_definition_version=rows[0]['audit_definition_version'],
            totals=dict(tot), runs_with_tests=sum(1 for r in rows if r['cap_audit'].get('tests')),
            pct_tests_with_queued_link=share('tests_queued'), pct_tests_arrivals_raise_demand=share('tests_arrivals_raise'),
            pct_queue_decides_arrivals_kept=share('A_ne_B'), pct_arrivals_decide_queue_kept=share('A_ne_Cp'),
            pct_arrivals_decide_queue_removed=share('B_ne_D'), pct_queue_decides_arrivals_removed=share('Cp_ne_D'),
            pct_A_vs_D=share('A_ne_D'), pct_A_vs_E=share('A_ne_E'),
            per_run_pct_arrivals_decide_queue_kept=dict(zip(('min', 'median', 'max'),
                                                            np.percentile(per_run('A_ne_Cp'), [0, 50, 100]).tolist())),
            per_run_pct_queue_decides_arrivals_kept=dict(zip(('min', 'median', 'max'),
                                                             np.percentile(per_run('A_ne_B'), [0, 50, 100]).tolist())))
    with open(SUMMARY, 'w') as fh:
        json.dump(S, fh, indent=1)
    return S


def print_table(S):
    print(f"ECMP mean total drops {S['ecmp_reference']['total']:.1f} (n={S['ecmp_reference']['n']})")
    print(f"{'config':8s} sig  total%   Δ vs A [95% CI]            fabric/host %    deliv/t  mean/p99 ms   reroutes")
    for cfg in C.CONFIGS:
        for sig in C.SIGNALS:
            c = S['cells'][f'{cfg}|{sig}']
            d = S['comparisons'].get(f'{cfg}_vs_A|{sig}')
            ds = f"{d['mean_pct']:+.2f} [{d['ci95_pct'][0]:+.2f}, {d['ci95_pct'][1]:+.2f}]" if d else '—'
            print(f"{cfg:8s} {sig[0].upper()}   {c['total_pct']:6.2f}   {ds:26s} {c['fabric_pct']:6.1f} / {c['host_pct']:+.1f}"
                  f"   {c['deliv_per_tick']:7.1f}  {c['mean_ms']:.2f} / {c['p99_ms']:.2f}  {c['reroutes']:7.1f}")
    for k, v in S['comparisons'].items():
        print(f"  {k:36s} {v['mean_pct']:+.3f}  pair {v['ci95_pct']}  cluster {v['ci95_cluster_pct']}  W/T/L {v['wins']}/{v['ties']}/{v['losses']}")
    for sig, a in S['attribution'].items():
        print(f"  attribution {sig}: tests {a['totals']['tests']}, queued {a['pct_tests_with_queued_link']:.2f}%, "
              f"queue decides {a['pct_queue_decides_arrivals_kept']:.3f}%, arrivals decide {a['pct_arrivals_decide_queue_kept']:.3f}%"
              f" (queue removed: {a['pct_arrivals_decide_queue_removed']:.3f}%), A vs E {a['pct_A_vs_E']:.3f}%")
    print('validation:', json.dumps(S['validation'], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['gate', 'summary'])
    ap.add_argument('--determinism', default=None)
    args = ap.parse_args()
    if args.what == 'gate':
        g = gate()
        print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'fields_compared'} if isinstance(v, dict) else v)
                          for k, v in g.items()}, indent=1)[:6000])
        print('GATES PASS' if g['G0']['pass'] and g['G1_pass'] else 'GATES FAIL')
        raise SystemExit(0 if g['G0']['pass'] and g['G1_pass'] else 2)
    print_table(summary(args.determinism))


if __name__ == '__main__':
    main()
