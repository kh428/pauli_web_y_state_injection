"""tsim MC of the record-clamped conditional LER at p=0.05, against the
exact contraction formulas. state via argv: T (default) or Y."""
import sys as _s
STATE = _s.argv[1] if len(_s.argv) > 1 else "T"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tsim
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central

d = 3
p = 0.05
kinds = dict(spiral_kinds(d)); site = next(c for c, k in kinds.items()
                                           if k == 'Y')
kinds[site] = STATE
txt, info = build(d, kinds, sched=opt_central, noisy_rounds=1,
                  extra_rounds=0, tail=False)
tail, lw, _, _ = readout_tail(d, kinds, info)
lines = txt.split('\n')
out = []
for l in lines:
    out.append(l)
    ps = l.split()
    if ps and ps[0] == ('S' if STATE == 'Y' else 'T') and int(ps[1]) == lw:
        out.append(f'X_ERROR({p}) {lw}')
    if ps and ps[0] == 'CX' and lw in (int(ps[1]), int(ps[2])):
        out.append(f'DEPOLARIZE2({p}) {ps[1]} {ps[2]}')
undo = 'S_DAG' if STATE == 'Y' else 'T_DAG'
full = ('\n'.join(out) + '\n' + tail + f'\n{undo} {lw}\nH {lw}\nM {lw}')
c = tsim.Circuit(full)
s = c.compile_sampler()
nm = c.num_measurements
acc = flip = 0
for _ in range(120):
    m = s.sample(2_000_000)
    keep = ~m[:, :nm-1].any(axis=1)     # all-zero record
    acc += int(keep.sum())
    flip += int(m[keep, -1].sum())
print(f'accepted {acc}, P(flip | record=0) = {flip/acc:.5f} '
      f'+- {np.sqrt(flip)/acc:.5f}')
print('T exact: 0.09732 (leading 0.10500) | Y exact: 0.11851 (leading 0.13000)')
