import os
"""Regenerate fig_mc_lc.pdf from mc_results.json (legend outside,
below both panels)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'serif',
                     'font.serif': ['cmr10', 'CMU Serif', 'DejaVu Serif'],
                     'axes.unicode_minus': False})
from matplotlib.lines import Line2D

res = json.load(open('mc_results.json'))
FORK = os.path.dirname(os.path.abspath(__file__))
STYLE = {'CR-LC':  ('#d62728', 'o', 'corner, Lao-Criger schedule'),
         'CR-opt': ('#ff7f0e', 's', 'corner, optimised (serialised)'),
         'MR-LC':  ('#1f77b4', '^', 'central, Lao-Criger schedule'),
         'MR-opt': ('#2ca02c', 'D', 'central, optimised (serialised)')}
FILL = {3: 'full', 5: 'left', 7: 'bottom', 9: 'top', 11: 'none'}
XOFF = {3: 1.0, 5: 1.07, 7: 1.15, 9: 1.23, 11: 1.32}
LINES_A = {'CR-LC': 8/15, 'CR-opt': 6/15, 'MR-LC': 7/15, 'MR-opt': 3/15}
LINES_B = {'CR-LC': 48/15, 'CR-opt': 46/15, 'MR-LC': 32/15, 'MR-opt': 28/15}


def parse(key):
    parts = key.split('|')
    if len(parts) == 3:
        m, n, p = parts; d = 3
    else:
        m, n, ds, p = parts; d = int(ds[1:])
    return m, n, d, float(p)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.3))
for ax, model, LINES, claims in [
        (ax1, 'p2', LINES_A, [(9/15, "Lao-Criger's $9/15$ (both placements)")]),
        (ax2, 'all', LINES_B, [(49/15, None), (34/15, None)])]:
    for name, (col, mk, lab) in STYLE.items():
        for d in (3, 5, 7, 9, 11):
            ps, ys, es = [], [], []
            for key, (acc, rej, err) in res.items():
                m, n, dd, p = parse(key)
                if m == model and n == name and dd == d:
                    ps.append(p); ys.append(err/acc/p)
                    es.append(np.sqrt(err)/acc/p)
            if not ps: continue
            srt = np.argsort(ps)
            ps = np.array(ps)[srt] * XOFF[d]
            ys, es = np.array(ys)[srt], np.array(es)[srt]
            ax.errorbar(ps, ys, yerr=es, color=col, marker=mk, ms=4.5,
                        ls='none', capsize=2, fillstyle=FILL[d],
                        label=(lab if (ax is ax1 and d == 3) else None))
        ax.axhline(LINES[name], color=col, lw=1.0, alpha=0.75)
    for y, lab in claims:
        ax.axhline(y, color='k', ls='--', lw=1.0, label=lab)
    ax.set_xscale('log')
ax1.set_xlabel(r'$p_2$'); ax2.set_xlabel(r'$p$  ($p_2{=}p_I{=}p_1{=}p_M{=}p$)')
ax1.set_ylabel(r'$p_L/p_2$'); ax2.set_ylabel(r'$p_L/p$')
ax1.set_title('(a) two-qubit gate noise only', fontsize=13, fontweight='bold')
ax2.set_title('(b) all rates equal', fontsize=13, fontweight='bold')
ax2.annotate("Lao-Criger's $49/15$", xy=(1.15e-4, 49/15 + 0.015), fontsize=8)
ax2.annotate("Lao-Criger's $34/15$", xy=(1.15e-4, 34/15 + 0.015), fontsize=8)
for y, t in [(48/15, '$48/15$'), (46/15, '$46/15$'), (32/15, '$32/15$'),
             (28/15, '$28/15$')]:
    ax2.annotate(t, xy=(2.9e-3, y), fontsize=8, va='center')
ax2.set_xlim(right=4.2e-3)
extra = [Line2D([], [], color='gray', marker='o', ls='none',
                fillstyle=FILL[d], label=f'$d={d}$') for d in (3, 5, 7, 9, 11)]
h, l = ax1.get_legend_handles_labels()
fig.legend(h + extra, l + [f'$d={d}$' for d in (3, 5, 7, 9, 11)],
           fontsize=7.5, ncol=5, loc='lower center',
           bbox_to_anchor=(0.5, 0.0), framealpha=0.95)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(FORK + 'fig_mc_lc.pdf')
print('saved')
