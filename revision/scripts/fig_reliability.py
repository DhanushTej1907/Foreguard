from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import json, sys, os
sys.path.insert(0, NF_ROOT + '/tools'); sys.path.insert(0, NF_ROOT)
import numpy as np
import make_paper_assets as M
import matplotlib.pyplot as plt
M.apply_style()
fa = json.load(open(RESULTS + '/forecast_analysis.json'))
fig, axes = plt.subplots(1, 2, figsize=(M.WIDTH_IN, 2.45), sharey=True)
GREY = '#7F7F7F'
for ax, kind, title in zip(axes, ('gnn', 'lstm_only'), ('(a) GCN-LSTM', '(b) LSTM-only')):
    runs = fa['models'][kind]
    eces = globals().setdefault('eces', {})
    ax.plot([0, 1], [0, 1], color='#BBBBBB', lw=.6, ls=':', zorder=0)
    for key, lab, col, mk, ls in (('rel_raw', 'Uncalibrated', GREY, 'x', '--'),
                                  ('rel_cal', 'Temperature', M.BLUE, 'o', '-'),
                                  ('rel_platt', 'Platt', M.VERMILLION, 'D', '-')):
        nb = len(runs[0][key]); xs, ys = [], []
        for b in range(nb):
            cells = [r[key][b] for r in runs if r[key][b] is not None]
            if not cells: continue
            n = np.array([c['n'] for c in cells], float)
            xs.append(np.average([c['mean_p'] for c in cells], weights=n))
            ys.append(np.average([c['frac_pos'] for c in cells], weights=n))
        ece_key = {'rel_raw': 'ece_raw', 'rel_cal': 'ece_cal', 'rel_platt': 'ece_platt'}[key]
        ece = np.mean([r[ece_key] for r in runs])
        ax.plot(xs, ys, color=col, marker=mk, ms=3, ls=ls, lw=.9, label=lab)
        eces.setdefault(lab, []).append(ece)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([0, .5, 1], ['0', '0.5', '1']); ax.set_yticks([0, .5, 1], ['0', '0.5', '1'])
    ax.set_title(title); ax.set_xlabel('Mean predicted probability', fontsize=7)
axes[0].set_ylabel('Fraction congested')
h, l = axes[0].get_legend_handles_labels()
l = [f"{x} (ECE {eces[x][0]:.3f} / {eces[x][1]:.3f})" for x in l]
fig.legend(h, l, loc='lower center', ncol=1, bbox_to_anchor=(0.5, 0.0), fontsize=7, handlelength=1.8)
fig.get_layout_engine().set(rect=(0, 0.2, 1, 0.8))
for ext in ('pdf', 'png'):
    fig.savefig(f'{FIGS}/fig_reliability.{ext}', dpi=300)
print('ok')
