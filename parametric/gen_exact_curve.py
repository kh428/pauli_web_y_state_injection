"""Figure for v3 section 9: the exact all-orders campaign p_L(p2)
(from the tsim-derived linear forms) against direct stim sampling,
with the first- and second-order truncations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import stim as _stim
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central

# exact function from the logged srepr
lines = open('../RESULTS_tsim_campaign.txt').read().split('\n')
expr = sp.sympify(lines[1], evaluate=True)
p = [s for s in expr.free_symbols][0]
f = sp.lambdify(p, expr, 'numpy')

d = 3
kinds = spiral_kinds(d)
site0 = next(c for c, k in kinds.items() if k == 'Y')
_, info = build(d, kinds, sched=opt_central, p2=0.0, tail=False)
tail, lw, col_off, row_off = readout_tail(d, kinds, info)
CONSTS = {}
for basis, kk, post in [('Z', '+', f'H {lw}'), ('X', '0', '')]:
    kcal = dict(kinds); kcal[site0] = kk
    ct, _ = build(d, kcal, sched=opt_central, tail=False,
                  site_override=site0)
    cfull = ct + '\n' + tail + ('\n' + post if post else '') + f'\nM {lw}'
    m = _stim.Circuit(cfull).compile_sampler().sample(64).astype('uint8')
    offs = col_off if basis == 'Z' else row_off
    nc = m.shape[1]
    par = (m[:, [nc - 1 + o for o in offs]].sum(axis=1) + m[:, -1]) % 2
    CONSTS[basis] = int(par[0])
fix = []
for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
if CONSTS['Z']: fix.append(f'Z {lw}')
for o in row_off: fix.append(f'CX rec[{o}] {lw}')
if CONSTS['X']: fix.append(f'X {lw}')
TAIL = '\n' + tail + '\n' + '\n'.join(fix) + f'\nS_DAG {lw}\nH {lw}\nM {lw}'
dets = info['dets']

PVALS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3.5e-2, 5e-2]
pts = []
for pv in PVALS:
    txt, _ = build(d, kinds, sched=opt_central, p2=pv,
                   noisy_rounds=2, extra_rounds=2, tail=False)
    c = _stim.Circuit(txt + TAIL)
    s_ = c.compile_sampler()
    shots = 60_000_000 if pv < 5e-3 else 150_000_000
    acc = bad = 0
    left = shots
    while left > 0:
        n = min(4_000_000, left); left -= n
        m = s_.sample(n)
        det = np.zeros((n, len(dets)), bool)
        for i, (lab, recs) in enumerate(dets):
            for r in recs:
                det[:, i] ^= m[:, r].astype(bool)
        keep = ~det.any(axis=1)
        acc += int(keep.sum()); bad += int(m[keep, -1].sum())
    pl = bad / acc
    er = np.sqrt(pl * (1 - pl) / acc)
    pts.append((pv, pl, er))
    print(f'p={pv}: pL={pl:.6f}({er:.6f}), exact={float(f(pv)):.6f}',
          flush=True)

fig, ax = plt.subplots(figsize=(5.6, 3.4))
pp = np.logspace(np.log10(6e-4), np.log10(6.5e-2), 300)
ax.plot(pp, f(pp) / pp, color='black', lw=1.3,
        label='exact (all orders)')
ax.plot(pp, np.full_like(pp, 1/5), color='#888888', ls=':', lw=1.0,
        label=r'$p_2/5$')
ax.plot(pp, 1/5 + (649/225) * pp, color='#2ca02c', ls='--', lw=1.1,
        label=r'$p_2/5 + \frac{649}{225}p_2^2$')
xs = [q[0] for q in pts]; ys = [q[1]/q[0] for q in pts]
es = [q[2]/q[0] for q in pts]
ax.errorbar(xs, ys, yerr=es, fmt='D', color='#2ca02c', ms=4,
            mfc='white', capsize=2, lw=1,
            label=r'\texttt{stim} sampling' if False else 'sampled')
ax.set_xscale('log')
ax.set_xlabel(r'$p_2$'); ax.set_ylabel(r'$p_L/p_2$')
ax.legend(fontsize=8, loc='upper left')
fig.tight_layout()
OUT = os.path.dirname(os.path.abspath(__file__))
fig.savefig(OUT + 'fig_exact_curve.pdf')
print('saved')
