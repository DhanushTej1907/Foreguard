"""New figures for the revision (IEEE single column, the paper's style)."""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, sys
sys.path.insert(0, NF_ROOT + '/tools'); sys.path.insert(0, NF_ROOT)
import numpy as np
import make_paper_assets as M
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
M.apply_style()
S = json.load(open(RESULTS + '/summary.json'))
SIG = [('score', 'Guarded score', M.VERMILLION, 'D'), ('persistence', 'Guarded persistence', M.GREEN, 'v'),
       ('gnn', 'Guarded GNN', M.BLUE, 'o')]


def fig_ablation():
    rules = [('full', 'None (full guard)'), ('no_capacity', 'Capacity check'), ('no_reservation', 'Reservation'),
             ('no_budget', 'Change budget'), ('no_stretch', 'Stretch bound'),
             ('no_min_improvement', 'Min. improvement'), ('no_cooldown', 'Cooldown'),
             ('fixed_threshold_0.7', 'Tuned threshold'), ('all_rules_off', 'All rules')]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(M.WIDTH_IN, 2.75), sharey=True,
                                     gridspec_kw=dict(width_ratios=[1.25, 0.9, 0.9]))
    y = np.arange(len(rules))[::-1]
    off = {'score': -0.22, 'persistence': 0.0, 'gnn': 0.22}
    ecmp_p99 = None
    for sig, lab, col, mk in SIG:
        A = S['ablation'].get(sig, {})
        for (key, _), yy in zip(rules, y):
            e = A.get(key)
            if not e:
                continue
            ecmp_p99 = e.get('ecmp_p99_delay_ms', ecmp_p99)
            if key == 'full':
                a1.plot(0, yy + off[sig], mk, color=col, ms=3.2)
            else:
                v = e['vs_full']; lo, hi = v['ci95_pct']
                a1.errorbar(v['mean_pct'], yy + off[sig], xerr=[[v['mean_pct'] - lo], [hi - v['mean_pct']]], fmt=mk,
                            color=col, ms=3.2, elinewidth=.8, capsize=0)
            a2.plot(e['reroutes'], yy + off[sig], mk, color=col, ms=3.2)
            a3.plot(e['p99_delay_ms'], yy + off[sig], mk, color=col, ms=3.2)
    a1.axvline(0, color='#7F7F7F', lw=.6)
    if ecmp_p99:
        a3.axvline(ecmp_p99, color=M.BLACK, lw=.6, ls='--')
        a3.text(ecmp_p99 + 0.12, y[0] + 0.45, 'ECMP', fontsize=6.5, ha='left', va='center', color=M.BLACK)
    a1.set_yticks(y, [r[1] for r in rules], fontsize=7)
    a1.set_ylabel('Rule removed')
    a2.set_xscale('log')
    for ax, t in zip((a1, a2, a3), ('(a) $\\Delta$ total drops\n(pp of ECMP)', '(b) Reroutes\nper run', '(c) p99 delay\n(ms)')):
        ax.set_title(t, fontsize=7); ax.tick_params(axis='x', labelsize=7)
    handles = [Line2D([], [], color=c, marker=m, ls='', ms=3.5, label=l) for _, l, c, m in SIG]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0), handletextpad=0.2, columnspacing=0.9,
               fontsize=7)
    fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.925), wspace=0.02)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGS}/fig_ablation.{ext}', dpi=300)


def fig_delay():
    D = S.get('delay', {}); R = S.get('replicate', {})
    conds = [('none', 'None'), ('dT:2,0', '$d_T$=2'), ('dT:5,0', '$d_T$=5'), ('dA:0,2', '$d_A$=2'), ('dA:0,5', '$d_A$=5'),
             ('dTdA:5,5', 'Both=5')]
    x = np.arange(len(conds))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(M.WIDTH_IN, 2.9), sharex=True, gridspec_kw=dict(height_ratios=[1, 1]))
    for sig, lab, col, mk in [s_ for s_ in SIG if s_[0] != 'persistence']:
        ys = []
        for key, _ in conds:
            e = R.get('guarded_score' if sig == 'score' else 'guarded_gnn') if key == 'none' else D.get(key, {}).get(sig)
            ys.append(e['total_drops'] if e else np.nan)
        a1.plot(x + (-0.08 if sig == 'score' else 0.08), ys, mk, color=col, ms=3.6, label=lab)
    a1.set_ylabel('Total drops\nvs. ECMP (%)', fontsize=7)
    a1.set_ylim(-5.5, -3)
    a1.legend(loc='lower left', ncol=2, fontsize=7, handletextpad=0.2, borderaxespad=0.1)
    a1.set_title('(a) Drop reduction of each strategy', fontsize=7)
    # paired difference
    for i, (key, _) in enumerate(conds):
        if key == 'none':
            g = S['replicate_tost']['guarded_score']
            m, lo, hi = g['mean_pct'], None, None
            # 95% pair CI from replicate comparison
            m = g['mean_pct']; lo, hi = g['ci95_pct'] if 'ci95_pct' in g else (np.nan, np.nan)
        else:
            g = D[key]['gnn_minus_score']; m = g['mean_pct']; lo, hi = g['ci95_pct']
        a2.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt='o', color=M.BLUE, ms=3.6, elinewidth=.9, capsize=0)
    a2.axhline(0, color='#7F7F7F', lw=.6)
    a2.set_ylabel('GNN $-$ score\n(pp of ECMP)', fontsize=7)
    a2.set_title('(b) Guarded GNN minus guarded score (95% CI)', fontsize=7)
    a2.set_ylim(-0.95, 1.7)
    a2.text(len(conds) - 0.55, -0.78, 'forecast better', fontsize=6.5, ha='right', va='center', color='#4D4D4D')
    a2.text(len(conds) - 0.55, 1.5, 'score better', fontsize=6.5, ha='right', va='center', color='#4D4D4D')
    a2.set_xticks(x, [c[1] for c in conds], fontsize=7)
    a2.set_xlim(-0.5, len(conds) - 0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGS}/fig_delay.{ext}', dpi=300)


if __name__ == '__main__':
    for name in sys.argv[1:] or ['ablation', 'delay']:
        globals()['fig_' + name]()
    print('ok')
