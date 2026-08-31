import os
"""Step 1: decoded-convention malignant counts (append 2 future
rounds; count preparation faults with no detector ever firing that
flip the logical), for corner+central x {N/Z, TS} at d=3,5.
Anchors to reproduce: TS corner 8, TS central 7 (stim-verified)."""
import sys, os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'figures_3d', 'src')
sys.path.insert(0, SRC)
from iter_lc_reconstruct import build, count, lc_sched
from physical import rect, injection_kind
from schemes import spiral_kinds

def nz_sched(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NW, SW, NE, SE] if typ == 'X' else [NW, NE, SW, SE])
    ss = set(sup)
    return [c if c in ss else None for c in order]

for d in (3, 5):
    for name, kinds in [('corner', {c: injection_kind(c, 0, 0, d)
                                    for c in rect(0, 0, d, d)}),
                        ('central', dict(spiral_kinds(0, 0, d)))]:
        for sname, fn in [('TS', lc_sched), ('N/Z', nz_sched)]:
            res = count(kinds, d, fn, f'{name}/{sname}/d{d}')
