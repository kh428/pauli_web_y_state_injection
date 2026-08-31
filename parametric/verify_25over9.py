"""Independent check of the quadratic coefficient in
p_L^{|Y>} = (1/5) p2 + (25/9) p2^2 + O(p2^3)  (optimised central, d=3,
CNOT depolarising only, deflation readout, accept iff all detectors 0).

Method (independent of the original DEM pair enumeration): for every
one of the 48 noisy CNOTs and each of its 15 Pauli classes, insert the
class deterministically (E(1)) into the NOISELESS campaign circuit and
read off its exact detector-flip vector and observable flip with
stim's detector sampler (Pauli-frame linearity makes pair effects the
XOR of singles, exactly). Then the exact series of P(flip&acc)/P(acc)
to O(p2^2) in rational arithmetic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
import sympy as sp
from itertools import combinations
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central

d = 3
P2 = [(a, b) for a in 'IXYZ' for b in 'IXYZ' if (a, b) != ('I', 'I')]

def assemble(fault=None):
    kinds = spiral_kinds(d)
    site0 = next(c for c, k in kinds.items() if k == 'Y')
    txt, info = build(d, kinds, sched=opt_central, p2=0.0, tail=False,
                      fault=fault)
    tail, lw, col_off, row_off = readout_tail(d, kinds, info)
    # frame constants, calibrated once (noiseless Clifford runs)
    global CONSTS
    if 'CONSTS' not in globals():
        CONSTS = {}
        nmeas = info['nmeas']
        for basis, kk, post in [('Z', '+', f'H {lw}'), ('X', '0', '')]:
            kcal = dict(kinds); kcal[site0] = kk
            ct, _ = build(d, kcal, sched=opt_central, tail=False,
                          site_override=site0)
            cfull = ct + '\n' + tail + ('\n' + post if post else '') \
                + f'\nM {lw}'
            m = stim.Circuit(cfull).compile_sampler().sample(64)
            m = m.astype('uint8')
            offs = col_off if basis == 'Z' else row_off
            ncols = m.shape[1]
            par = (m[:, [ncols - 1 + o for o in offs]].sum(axis=1)
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
    nmeas = info['nmeas']
    n_tail = d * d - 1
    total = nmeas + n_tail + 1
    lines = [full]
    for lab, recs in info['dets']:
        lines.append('DETECTOR ' +
                     ' '.join(f'rec[{r - total}]' for r in recs))
    lines.append('OBSERVABLE_INCLUDE(0) rec[-1]')
    return '\n'.join(lines), info

# noiseless baseline: everything deterministic and zero
base_txt, info = assemble()
det, obs = stim.Circuit(base_txt).compile_detector_sampler() \
    .sample(16, separate_observables=True)
assert not det.any() and not obs.any(), 'baseline not clean'
ndet = det.shape[1]
print(f'baseline clean: {ndet} detectors, obs deterministic 0')

# effect vector of every (channel, class)
channels = sorted({(rd, ci, slot) for (rd, ci, slot, *_ )
                   in info['catalogue']})
K = len(channels)
print(f'{K} noisy CNOT channels x 15 classes = {K*15} faulted circuits')
effects = []          # (channel_index, syndrome_bytes, obs_flip)
for idx, (rd, ci, slot) in enumerate(channels):
    for (Pa, Pt) in P2:
        txt, _ = assemble(fault=('cnot', rd, ci, slot, Pa, Pt))
        dv, ov = stim.Circuit(txt).compile_detector_sampler() \
            .sample(4, separate_observables=True)
        assert (dv == dv[0]).all() and (ov == ov[0]).all(), \
            (rd, ci, slot, Pa, Pt)
        effects.append((idx, dv[0].tobytes(), int(ov[0][0])))

A1 = sum(1 for i, s, o in effects if not any(s))
B1 = sum(1 for i, s, o in effects if not any(s) and o)
print(f'singles: A1 (accepted) = {A1}, B1 (accepted & flip) = {B1}')

# pairs on DIFFERENT channels with matching syndromes
from collections import defaultdict
groups = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for i, s, o in effects:
    groups[s][i][o] += 1
A2 = B2 = 0
for s, per in groups.items():
    tot0 = sum(v[0] for v in per.values())
    tot1 = sum(v[1] for v in per.values())
    n = tot0 + tot1
    pairs_all = n * (n - 1) // 2
    odd_all = tot0 * tot1
    same_pairs = sum((v[0] + v[1]) * (v[0] + v[1] - 1) // 2
                     for v in per.values())
    same_odd = sum(v[0] * v[1] for v in per.values())
    A2 += pairs_all - same_pairs
    B2 += odd_all - same_odd
print(f'pairs: A2 = {A2}, B2 = {B2}')

p = sp.symbols('p')
u = p / 15
N = u * (1 - p) ** (K - 1) * B1 + u ** 2 * (1 - p) ** (K - 2) * B2
D = (1 - p) ** K + u * (1 - p) ** (K - 1) * A1 \
    + u ** 2 * (1 - p) ** (K - 2) * A2
series = sp.series(sp.together(N / D), p, 0, 3).removeO().expand()
c1 = series.coeff(p, 1)
c2 = series.coeff(p, 2)
print(f'p_L = {c1} p2 + {c2} p2^2 + O(p2^3)')
print(f'linear:    {c1} = {sp.nsimplify(c1)}   (paper: 1/5)')
print(f'quadratic: {c2} = {float(c2):.6f}      (paper: 25/9 = {25/9:.6f})')
