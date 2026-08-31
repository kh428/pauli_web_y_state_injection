import os
"""Regenerate fig_t_inj.pdf from t_results_final.json."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

res = json.load(open('t_results_final.json'))
FORK = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots(figsize=(5.6, 3.6))
STYLE = {'Y': ('#2ca02c', 'D', 3/15, r'$|Y\rangle$ injection'),
         'T': ('#9467bd', 'o', 2.5/15, r'$|T\rangle$ injection')}
FILL = {3: 'full', 5: 'none', 7: 'bottom'}
XOFF = {3: 1.0, 5: 1.09, 7: 1.19}
for state, (col, mk, line, lab) in STYLE.items():
    for d in (3, 5, 7):
        ps, ys, es = [], [], []
        for key, (acc, bad) in res.items():
            st, ds, p = key.split('|')
            if st == state and int(ds[1:]) == d:
                p = float(p)
                ps.append(p); ys.append(bad/acc/p)
                es.append(np.sqrt(bad)/acc/p)
        srt = np.argsort(ps)
        ps = np.array(ps)[srt] * XOFF[d]
        ys, es = np.array(ys)[srt], np.array(es)[srt]
        ax.errorbar(ps, ys, yerr=es, color=col, marker=mk, ms=6, ls='none',
                    capsize=2.5, fillstyle=FILL[d],
                    label=(lab if d == 3 else None))
    ax.axhline(line, color=col, lw=1.0, alpha=0.8)
pp = np.geomspace(4e-4, 2.6e-3, 50)
ax.plot(pp, 3/15 + 649/225 * pp, color='#2ca02c', ls='--', lw=1.1,
        label=r'$p_2/5 + \frac{649}{225}p_2^2$ (exact pair enumeration)')
ax.annotate('$3/15$', xy=(3.05e-3, 3/15 - 0.004), fontsize=9)
ax.annotate('$2.5/15 = 1/6$', xy=(2.35e-3, 2.5/15 - 0.0055), fontsize=9)
extra = [Line2D([], [], color='gray', marker='o', ls='none',
                fillstyle=FILL[d], label=f'$d={d}$') for d in (3, 5, 7)]
h, l = ax.get_legend_handles_labels()
ax.legend(h + extra, l + ['$d=3$', '$d=5$', '$d=7$'], fontsize=7.5, ncol=2,
          loc='upper center')
ax.set_xscale('log')
ax.set_xlabel(r'$p_2$'); ax.set_ylabel(r'$p_L/p_2$')
ax.set_xlim(right=4.4e-3); ax.set_ylim(0.15, 0.245)
fig.tight_layout()
fig.savefig(FORK + 'fig_t_inj.pdf')
print('saved')
