import sys, os, json
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fractions import Fraction
import runpy
g = runpy.run_path(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'exact_T_cosets.py'))
probs = g['probs']
tr = json.load(open(SRC + '/../t_results_final.json'))
fig, ax = plt.subplots(figsize=(5.4, 3.3))
pp = np.logspace(np.log10(3e-4), np.log10(6e-2), 240)
exY = [float(probs(Fraction(p).limit_denominator(10**9))[0]) / p
       for p in pp]
exT = [float(probs(Fraction(p).limit_denominator(10**9))[1]) / p
       for p in pp]
ax.plot(pp, exY, color='#2ca02c', lw=1.3,
        label=r'$|Y\rangle$ exact (all orders)')
ax.plot(pp, exT, color='#9467bd', lw=1.3,
        label=r'$|T\rangle$ exact (all orders)')
for st, col, mk in [('Y', '#2ca02c', 'D'), ('T', '#9467bd', 'o')]:
    xs, ys, es = [], [], []
    for key, (accn, badn) in tr.items():
        s_, dd, p2s_ = key.split('|')
        if s_ != st or dd != 'd3': continue
        p2v = float(p2s_); samp = badn / accn
        xs.append(p2v); ys.append(samp / p2v)
        es.append((samp * (1 - samp) / accn) ** 0.5 / p2v)
    ax.errorbar(xs, ys, yerr=es, fmt=mk, color=col, ms=4, mfc='white',
                capsize=2, lw=1, ls='none')
ax.axhline(1/5, color='#bbbbbb', ls=':', lw=0.9)
ax.axhline(1/6, color='#bbbbbb', ls=':', lw=0.9)
ax.set_xscale('log')
ax.set_xlabel(r'$p_2$'); ax.set_ylabel(r'$p_L/p_2$')
ax.legend(fontsize=8, loc='upper left')
fig.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'figs')
fig.savefig(os.path.join(OUT, 'fig_exact_T.pdf'))
print('saved')
