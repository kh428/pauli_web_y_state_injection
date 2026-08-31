"""Independent confirmation of the multi-channel exact LER via the
hand-parser branch sum: enumerate every Pauli configuration of the five
channels (16^4 x 2 branches), contract each amplitude with the
pyzx-based parser (a fully independent code path from the pyzx_param
reduction + DP), and assemble the same conditional LER symbolically."""
import sys, os, itertools
import sys as _s
STATE = _s.argv[1] if len(_s.argv) > 1 else "T"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from zx_from_stim import parse, amplitude
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central

d = 3
kinds = dict(spiral_kinds(d)); site = next(c for c, k in kinds.items()
                                           if k == 'Y')
kinds[site] = STATE
txt, info = build(d, kinds, sched=opt_central, noisy_rounds=1,
                  extra_rounds=0, tail=False)
tail, lw, _, _ = readout_tail(d, kinds, info)
lines = txt.split('\n')
# channel locations: after site init (X_ERROR) and after the four
# site-touching CNOTs (DEP2), matching exact_ler.py's build_noisy
locs = []
for i, l in enumerate(lines):
    ps = l.split()
    if ps and ps[0] == ('S' if STATE == 'Y' else 'T') and int(ps[1]) == lw:
        locs.append(('X_ERROR', i, (lw,)))
    if ps and ps[0] == 'CX' and lw in (int(ps[1]), int(ps[2])):
        locs.append(('DEP2', i, (int(ps[1]), int(ps[2]))))
print('channels:', [(k, q) for k, _, q in locs], flush=True)

p2, q = sp.symbols('p_2 q', positive=True)
PAULI = ['I', 'X', 'Y', 'Z']
opts = []
for kind, i, qs in locs:
    if kind == 'X_ERROR':
        opts.append([((), 1 - q), ((('X', qs[0]),), q)])
    else:
        o = [((), 1 - p2)]
        for Pa in PAULI:
            for Pt in PAULI:
                if Pa == 'I' and Pt == 'I': continue
                ins = tuple((P, qq) for P, qq in
                            [(Pa, qs[0]), (Pt, qs[1])] if P != 'I')
                o.append((ins, p2 / 15))
        opts.append(o)

num = 0; den = 0
count = 0
for combo in itertools.product(*opts):
    w = 1
    inserts = {}
    for (ins, wt) in combo:
        w *= wt
        for P, qq in ins:
            inserts.setdefault(qq, []).append(P)
    # skip weight-0 shortcut none; build circuit text with inserts after
    # each channel location (order within a wire irrelevant up to sign)
    ll = list(lines)
    # insert after each location, from bottom to top to keep indices
    for (kind, i, qs), (ins, wt) in sorted(zip(locs, combo),
                                           key=lambda t: -t[0][1]):
        for P, qq in ins:
            ll.insert(i + 1, f'{P} {qq}')
    undo = 'S_DAG' if STATE == 'Y' else 'T_DAG'
    full = ('\n'.join(ll) + '\n' + tail +
            f'\n{undo} {lw}\nH {lw}\nM {lw}')
    a = []
    for b in (0, 1):
        g, _ = parse(full, {info['nmeas'] + 8: b})
        a.append(abs(amplitude(g))**2)
    r0 = sp.nsimplify(a[0], tolerance=1e-8, rational=True)
    r1 = sp.nsimplify(a[1], tolerance=1e-8, rational=True)
    num += w * r1
    den += w * (r0 + r1)
    count += 1
    if count % 5000 == 0:
        print(f'{count} configurations done', flush=True)
LER = sp.simplify(num / den)
print('branch-sum EXACT LER =', sp.factor(LER))
print('series (q=p2):', sp.series(sp.simplify(LER.subs(q, p2)), p2, 0, 2))
