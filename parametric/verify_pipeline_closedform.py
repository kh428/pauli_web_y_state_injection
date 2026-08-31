"""Validate the series pipeline of verify_25over9 against the EXACT
closed form: the one-noisy-round deflation protocol with channels on
the site's four check CNOTs only (the q=0 multichannel formula),
p_L = 8p(10125 - 20700p + 15360p^2 - 4096p^3) / (225 (15-8p)^2).
If the pipeline's O(p^2) series matches this formula's series, the
pair machinery is validated end-to-end."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
import sympy as sp
from collections import defaultdict
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central

d = 3
P2 = [(a, b) for a in 'IXYZ' for b in 'IXYZ' if (a, b) != ('I', 'I')]
kinds = spiral_kinds(d)
site = next(c for c, k in kinds.items() if k == 'Y')

def assemble(fault=None):
    txt, info = build(d, kinds, sched=opt_central, p2=0.0,
                      noisy_rounds=1, extra_rounds=0, tail=False,
                      fault=fault)
    tail, lw, col_off, row_off = readout_tail(d, kinds, info)
    global CONSTS
    if 'CONSTS' not in globals():
        CONSTS = {}
        for basis, kk, post in [('Z', '+', f'H {lw}'), ('X', '0', '')]:
            kcal = dict(kinds); kcal[site] = kk
            ct, _ = build(d, kcal, sched=opt_central, noisy_rounds=1,
                          extra_rounds=0, tail=False, site_override=site)
            cfull = ct + '\n' + tail + ('\n' + post if post else '') \
                + f'\nM {lw}'
            m = stim.Circuit(cfull).compile_sampler().sample(64)
            m = m.astype('uint8')
            offs = col_off if basis == 'Z' else row_off
            nc = m.shape[1]
            par = (m[:, [nc - 1 + o for o in offs]].sum(axis=1)
                   + m[:, -1]) % 2
            assert (par == par[0]).all()
            CONSTS[basis] = int(par[0])
    fix = []
    for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
    if CONSTS['Z']: fix.append(f'Z {lw}')
    for o in row_off: fix.append(f'CX rec[{o}] {lw}')
    if CONSTS['X']: fix.append(f'X {lw}')
    full = (txt + '\n' + tail + '\n' + '\n'.join(fix) +
            f'\nS_DAG {lw}\nH {lw}\nM {lw}')
    total = info['nmeas'] + d * d - 1 + 1
    out = [full]
    for lab, recs in info['dets']:
        out.append('DETECTOR ' + ' '.join(f'rec[{r-total}]' for r in recs))
    out.append('OBSERVABLE_INCLUDE(0) rec[-1]')
    return '\n'.join(out), info

base, info = assemble()
dv, ov = stim.Circuit(base).compile_detector_sampler() \
    .sample(16, separate_observables=True)
assert not dv.any() and not ov.any()

# channels = the site's four check CNOTs only (round 0)
site_ch = [(rd, ci, slot) for (rd, ci, slot, typ, pos, cell)
           in info['catalogue'] if cell == site]
assert len(site_ch) == 4, site_ch
effects = []
for idx, (rd, ci, slot) in enumerate(site_ch):
    for (Pa, Pt) in P2:
        txt, _ = assemble(fault=('cnot', rd, ci, slot, Pa, Pt))
        dvv, ovv = stim.Circuit(txt).compile_detector_sampler() \
            .sample(4, separate_observables=True)
        assert (dvv == dvv[0]).all() and (ovv == ovv[0]).all()
        effects.append((idx, dvv[0].tobytes(), int(ovv[0][0])))

A1 = sum(1 for i, s, o in effects if not any(np.frombuffer(s, bool)))
B1 = sum(1 for i, s, o in effects
         if not any(np.frombuffer(s, bool)) and o)
groups = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for i, s, o in effects:
    groups[s][i][o] += 1
A2 = B2 = 0
for s, per in groups.items():
    tot0 = sum(v[0] for v in per.values())
    tot1 = sum(v[1] for v in per.values())
    n = tot0 + tot1
    A2 += n * (n - 1) // 2 - sum((v[0]+v[1])*(v[0]+v[1]-1)//2
                                 for v in per.values())
    B2 += tot0 * tot1 - sum(v[0]*v[1] for v in per.values())
print(f'singles A1={A1} B1={B1}; pairs A2={A2} B2={B2}')

p = sp.symbols('p'); u = p / 15; K = 4
N = u*(1-p)**(K-1)*B1 + u**2*(1-p)**(K-2)*B2
D = (1-p)**K + u*(1-p)**(K-1)*A1 + u**2*(1-p)**(K-2)*A2
mine = sp.series(sp.together(N/D), p, 0, 3).removeO().expand()
exact = 8*p*(10125 - 20700*p + 15360*p**2 - 4096*p**3) \
    / (225*(15-8*p)**2)
ex = sp.series(sp.expand(exact), p, 0, 3).removeO().expand()
print('pipeline series:', mine)
print('closed-form    :', ex)
print('linear match   :', sp.simplify(mine.coeff(p,1)-ex.coeff(p,1))==0)
print('quadratic match:', sp.simplify(mine.coeff(p,2)-ex.coeff(p,2))==0)
