"""Hardening for verify_25over9: (1) spot-check that a two-fault
insertion's effect is the XOR of the single-fault effects (Pauli-frame
linearity, on the real circuit); (2) reproduce the DEM-based pair
enumeration to locate the discrepancy."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
import sympy as sp
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central
import verify_25over9 as V   # reuses assemble/effects (reruns them)

d = 3
P2 = V.P2
channels = V.channels
eff = {}
for k, (i, s, o) in enumerate(V.effects):
    ch = channels[i]
    cls = P2[k % 15]
    eff[(ch, cls)] = (np.frombuffer(s, dtype=bool).copy(), o)

# (1) XOR spot-check with double insertions
def assemble2(f1, f2):
    kinds = spiral_kinds(d)
    txt, info = build(d, kinds, sched=opt_central, p2=0.0, tail=False,
                      fault=f1)
    # insert the second fault manually: find its CNOT and append E(1)
    (_, rd2, ci2, slot2, Pa2, Pt2) = f2
    checks = info['checks']
    q = lambda c: d * c[0] + c[1]
    anc = {ci: d * d + ci for ci in range(len(checks))}
    typ, pos, sup = checks[ci2]
    try: name = opt_central(typ, pos)[slot2]
    except TypeError: name = opt_central(typ)[slot2]
    cell = sup[name]; a, dq = anc[ci2], q(cell)
    tg = ([f'{Pa2}{a}'] if Pa2 != 'I' else []) + \
         ([f'{Pt2}{dq}'] if Pt2 != 'I' else [])
    ins = 'E(1) ' + ' '.join(tg)
    # locate the CNOT occurrence: count CNOTs of that (rd, ci, slot)
    lines = txt.split('\n')
    count = -1; target_line = None
    pair = f'CX {a} {dq}' if typ == 'X' else f'CX {dq} {a}'
    occ = [i for i, l in enumerate(lines) if l == pair]
    # occurrences of this CNOT line appear once per round, in order
    target_line = occ[rd2]
    lines.insert(target_line + 1, ins)
    txt2 = '\n'.join(lines)
    tail, lw, col_off, row_off = readout_tail(d, kinds, info)
    fix = []
    for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
    if V.CONSTS['Z']: fix.append(f'Z {lw}')
    for o in row_off: fix.append(f'CX rec[{o}] {lw}')
    if V.CONSTS['X']: fix.append(f'X {lw}')
    full = (txt2 + '\n' + tail + '\n' + '\n'.join(fix) +
            f'\nS_DAG {lw}\nH {lw}\nM {lw}')
    nmeas = info['nmeas']; total = nmeas + d * d - 1 + 1
    out = [full]
    for lab, recs in info['dets']:
        out.append('DETECTOR ' + ' '.join(f'rec[{r-total}]' for r in recs))
    out.append('OBSERVABLE_INCLUDE(0) rec[-1]')
    return '\n'.join(out)

rng = random.Random(0)
ok = 0
for _ in range(12):
    ch1, ch2 = rng.sample(channels, 2)
    c1, c2 = rng.choice(P2), rng.choice(P2)
    f1 = ('cnot',) + ch1 + c1
    f2 = ('cnot',) + ch2 + c2
    txt = assemble2(f1, f2)
    dv, ov = stim.Circuit(txt).compile_detector_sampler() \
        .sample(4, separate_observables=True)
    assert (dv == dv[0]).all() and (ov == ov[0]).all()
    s1, o1 = eff[(ch1, c1)]; s2, o2 = eff[(ch2, c2)]
    assert (dv[0] == (s1 ^ s2)).all() and int(ov[0][0]) == (o1 ^ o2), \
        (ch1, c1, ch2, c2)
    ok += 1
print(f'(1) XOR linearity spot-check: {ok}/12 random double-faults match')

# (2) reproduce the DEM pair enumeration (the original method)
EPS = 1e-5
kinds = spiral_kinds(d)
txt, info = build(d, kinds, sched=opt_central, p2=EPS, tail=False)
tail, lw, col_off, row_off = readout_tail(d, kinds, info)
fix = []
for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
if V.CONSTS['Z']: fix.append(f'Z {lw}')
for o in row_off: fix.append(f'CX rec[{o}] {lw}')
if V.CONSTS['X']: fix.append(f'X {lw}')
full = (txt + '\n' + tail + '\n' + '\n'.join(fix) +
        f'\nS_DAG {lw}\nH {lw}\nM {lw}')
nmeas = info['nmeas']; total = nmeas + d * d - 1 + 1
out = [full]
for lab, recs in info['dets']:
    out.append('DETECTOR ' + ' '.join(f'rec[{r-total}]' for r in recs))
out.append('OBSERVABLE_INCLUDE(0) rec[-1]')
c = stim.Circuit('\n'.join(out))
dem = c.detector_error_model(flatten_loops=True)
errs = []           # (multiplicity n, frozenset dets, obs)
for inst in dem:
    if inst.type != 'error': continue
    pr = inst.args_copy()[0]
    n = pr / (EPS / 15)
    dets = frozenset(t.val for t in inst.targets_copy()
                     if t.is_relative_detector_id())
    obs = sum(1 for t in inst.targets_copy()
              if t.is_logical_observable_id()) % 2
    errs.append((n, dets, obs))
print(f'(2) DEM has {len(errs)} merged errors; '
      f'total multiplicity {sum(round(n) for n, _, _ in errs)}')
A1d = sum(round(n) for n, dts, o in errs if not dts)
B1d = sum(round(n) for n, dts, o in errs if not dts and o)
A2d = B2d = 0
E = len(errs)
for i in range(E):
    ni, di, oi = errs[i]
    for j in range(i + 1, E):
        nj, dj, oj = errs[j]
        if di == dj:
            A2d += round(ni) * round(nj)
            if oi ^ oj: B2d += round(ni) * round(nj)
print(f'    DEM-based: A1={A1d}, B1={B1d}, A2={A2d}, B2={B2d}')
p = sp.symbols('p'); u = p / 15; K = 48
N = u*(1-p)**(K-1)*B1d + u**2*(1-p)**(K-2)*B2d
D = (1-p)**K + u*(1-p)**(K-1)*A1d + u**2*(1-p)**(K-2)*A2d
s2 = sp.series(sp.together(N/D), p, 0, 3).removeO().expand()
print(f'    DEM-based series: {s2.coeff(p,1)} p2 + {s2.coeff(p,2)} p2^2')
