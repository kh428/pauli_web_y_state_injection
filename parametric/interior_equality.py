"""The finite geometric check behind the distance-independence proof:
at every odd d, the instruction sequence of the noisy protocol
restricted to Chebyshev radius R of the injected site is IDENTICAL
after recentring. With cone confinement (radius 3/2, cone_check)
this proves the reduced object is the same at every odd d >= 7."""
import sys, os
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from circuits import build, spiral_kinds, rect_checks
from exact_ler import build_noisy

R = 1.5   # the fault-cone radius of cone_check

def interior_signature(d):
    noisy_txt, channels, lw = build_noisy(d)
    kinds = spiral_kinds(d)
    site = next(c for c, k in kinds.items() if k == 'Y')
    checks = rect_checks(d)
    qpos = {d * i + j: (i, j) for i in range(d) for j in range(d)}
    for ci, (typ, pos, sup) in enumerate(checks):
        qpos[d * d + ci] = pos
    def rel(qq):
        p = qpos[qq]
        return (p[0] - site[0], p[1] - site[1])
    def near(qq):
        r = rel(qq)
        return max(abs(r[0]), abs(r[1])) <= R
    sig = []
    for line in noisy_txt.split('\n'):
        ps = line.split()
        if not ps: continue
        op = ps[0]
        if op in ('TICK',):
            sig.append(('TICK',))
            continue
        if op.startswith('DETECTOR') or op.startswith('OBSERVABLE') \
                or 'rec' in line:
            continue
        args = []
        keep = False
        for tok in ps[1:]:
            if tok.lstrip('-').replace('.', '').isdigit() and \
                    '.' not in tok:
                qq = int(tok)
                if qq in qpos:
                    args.append(rel(qq))
                    if near(qq): keep = True
                else:
                    args.append(('EXT',))
            else:
                args.append(tok)
        if keep:
            sig.append((op, tuple(args)))
    # drop trailing/leading empty ticks and collapse runs of TICKs
    out = []
    for item in sig:
        if item == ('TICK',) and out and out[-1] == ('TICK',):
            continue
        out.append(item)
    return tuple(out)

ref = None
refd = None
for d in (5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29):
    s = interior_signature(d)
    if ref is None:
        ref = s; refd = d
        print(f'd={d}: {len(s)} interior instructions (radius {R})')
    else:
        same = (s == ref)
        print(f'd={d}: {"IDENTICAL to d=%d" % refd if same else "DIFFERS"}'
              f' ({len(s)} instructions)')
        pass

