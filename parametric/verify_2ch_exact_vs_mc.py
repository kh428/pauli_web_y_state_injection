"""Decisive pipeline validation on the CAMPAIGN protocol: restrict
depolarising noise to two chosen CNOT channels. Exact P(flip|acc)
from the verify_25over9 effect vectors (256 configs, XOR-composed)
vs brute-force stim MC of the identical circuit at p2=0.1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from itertools import product
import verify_25over9 as V   # reruns the 720 effect extraction

d = 3
P2 = V.P2
channels = V.channels
# choose the site's round-1 X-check and Z-check CNOT channels (both
# carry malignant classes) - find them via the catalogue
_, info = V.assemble()
site = info['site'] if 'site' in info else None
from circuits import spiral_kinds
kinds = spiral_kinds(3)
site = next(c for c, k in kinds.items() if k == 'Y')
site_ch = [(rd, ci, slot) for (rd, ci, slot, typ, pos, cell)
           in info['catalogue'] if cell == site and rd == 0]
oth_ch = [(rd, ci, slot) for (rd, ci, slot, typ, pos, cell)
          in info['catalogue'] if rd == 1 and cell != site]
chA, chB = site_ch[1], oth_ch[7]
iA, iB = channels.index(chA), channels.index(chB)
print('channels:', chA, chB)

eff = {}
for k, (i, s, o) in enumerate(V.effects):
    eff[(i, P2[k % 15])] = (np.frombuffer(s, dtype=bool).copy(), o)
Z = np.zeros(V.ndet, bool)

def exact(p):
    q = p / 15
    accP = flipP = 0.0
    for ca in [None] + P2:
        pa = (1 - p) if ca is None else q
        sa, oa = (Z, 0) if ca is None else eff[(iA, ca)]
        for cb in [None] + P2:
            pb = (1 - p) if cb is None else q
            sb, ob = (Z, 0) if cb is None else eff[(iB, cb)]
            if (sa ^ sb).any():
                continue
            w = pa * pb
            accP += w
            if oa ^ ob:
                flipP += w
    return flipP / accP, accP

p = 0.1
ex, exacc = exact(p)
print(f'exact from effect vectors: P(flip|acc) = {ex:.6f}, '
      f'P(acc) = {exacc:.6f}')

# brute-force MC: same circuit, DEPOLARIZE2(p) on just those two CNOTs
from circuits import build
from t_experiment import readout_tail, opt_central
def mc_circuit():
    txt, info = build(d, kinds, sched=opt_central, p2=0.0, tail=False)
    lines = txt.split('\n')
    # insert DEPOLARIZE2 after the two chosen CNOTs
    checks = info['checks']
    q_ = lambda c: 3 * c[0] + c[1]
    anc = {ci: 9 + ci for ci in range(len(checks))}
    for (rd, ci, slot) in (chA, chB):
        typ, pos, sup = checks[ci]
        try: name = opt_central(typ, pos)[slot]
        except TypeError: name = opt_central(typ)[slot]
        cell = sup[name]; a, dq = anc[ci], q_(cell)
        pair = f'CX {a} {dq}' if typ == 'X' else f'CX {dq} {a}'
        occ = [i for i, l in enumerate(lines) if l == pair]
        lines.insert(occ[rd] + 1, f'DEPOLARIZE2({p}) {a} {dq}')
    txt = '\n'.join(lines)
    tail, lw, col_off, row_off = readout_tail(d, kinds, info)
    fix = []
    for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
    if V.CONSTS['Z']: fix.append(f'Z {lw}')
    for o in row_off: fix.append(f'CX rec[{o}] {lw}')
    if V.CONSTS['X']: fix.append(f'X {lw}')
    full = (txt + '\n' + tail + '\n' + '\n'.join(fix) +
            f'\nS_DAG {lw}\nH {lw}\nM {lw}')
    return full, info

full, info = mc_circuit()
c = stim.Circuit(full)
s = c.compile_sampler()
dets = info['dets']
acc = bad = 0
for _ in range(50):
    m = s.sample(2_000_000)
    det = np.zeros((len(m), len(dets)), bool)
    for i, (lab, recs) in enumerate(dets):
        for r in recs:
            det[:, i] ^= m[:, r].astype(bool)
    keep = ~det.any(axis=1)
    acc += int(keep.sum())
    bad += int(m[keep, -1].sum())
mc = bad / acc
err = np.sqrt(mc * (1 - mc) / acc)
print(f'MC ({acc} accepted): P(flip|acc) = {mc:.6f} +- {err:.6f}')
print(f'agreement: {abs(mc - ex) / err:.2f} sigma')
