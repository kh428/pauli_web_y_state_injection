"""FIXED |Y> distance-independence run: Y-site builder with the
channel bookkeeping in exact_ler's format (variable NAMES 'eN').
Asserts the d=3 formula equals the paper's equation (2), then runs
every odd distance to 29."""
import sys, os, time
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import sympy as sp
import exact_ler
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central

def build_noisy_Y(d=3):
    kinds = dict(spiral_kinds(d))
    site = next(c for c, k in kinds.items() if k == 'Y')
    txt, info = build(d, kinds, sched=opt_central, noisy_rounds=1,
                      extra_rounds=0, tail=False)
    tail, lw, _, _ = readout_tail(d, kinds, info)
    full = txt + '\n' + tail + f'\nS_DAG {lw}\nH {lw}\nM {lw}'
    noisy, channels, ne = [], [], 0
    for l in full.split('\n'):
        noisy.append(l)
        ps = l.split()
        if ps and ps[0] == 'S' and len(ps) == 2 and int(ps[1]) == lw:
            noisy.append(f'X_ERROR(0.001) {lw}')
            channels.append(('X_ERROR', [f'e{ne}'])); ne += 1
        if ps and ps[0] == 'CX' and 'rec' not in l and \
                lw in (int(ps[1]), int(ps[2])):
            noisy.append(f'DEPOLARIZE2(0.001) {ps[1]} {ps[2]}')
            channels.append(('DEP2', [f'e{ne+k}' for k in range(4)]))
            ne += 4
    assert len(channels) == 5, channels
    return '\n'.join(noisy), channels, lw

exact_ler.build_noisy = build_noisy_Y
from verify_d_independence import exact_ler_stable

expr, p2, q = exact_ler_stable(3)
paper = (65536*p2**4*q - 32768*p2**4 - 245760*p2**3*q
         + 122880*p2**3 + 345600*p2**2*q - 165600*p2**2
         - 216000*p2*q + 81000*p2 + 50625*q) / (225*(8*p2-15)**2)
diff = sp.simplify(expr - paper)
print('d=3 equals paper equation (2):', diff == 0, flush=True)
assert diff == 0, sp.factor(expr)
for d in (5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29):
    t0 = time.time()
    e2, _, _ = exact_ler_stable(d)
    ok = sp.simplify(e2 - paper) == 0
    print(f'd={d} ({time.time()-t0:.0f}s): '
          f'{"IDENTICAL" if ok else "DIFFERS"}', flush=True)
    assert ok
print('|Y> closed form identical at every odd distance 3-29')
