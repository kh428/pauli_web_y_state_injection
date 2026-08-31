import os
"""Step 2: optimised serialised schedules under the decoded
convention. Anchors: corner 6/15 (= Li's 2p2/5), central 3/15."""
import sys, os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'figures_3d', 'src')
sys.path.insert(0, SRC)
import iter_lc_reconstruct as ilr
from gates import GateBuilder
from physical import rect as _rect, region_checks

def build_sorted(kinds, d, sched_fn, prep_rounds=2, extra_rounds=2):
    b = GateBuilder()
    b.init_cells(kinds)
    P = _rect(0, 0, d, d)
    xs, zs = region_checks(P)
    schedules = ([('X', pos, sched_fn(pos, sup, 'X')) for pos, sup in xs] +
                 [('Z', pos, sched_fn(pos, sup, 'Z')) for pos, sup in zs])
    schedules.sort(key=lambda t: (t[1][0], t[1][1]))   # face-scan order
    tags = []
    for r in range(prep_rounds + extra_rounds):
        tag = f'p{r}' if r < prep_rounds else f'x{r}'
        tags.append(tag)
        b.gate_round(P, tag,
                     schedules=[(t, p, list(s)) for t, p, s in schedules])
    b.open_outputs(P)
    prep_tags = set(tags[:prep_rounds])
    return b.finish() + (prep_tags,)

ilr.build = build_sorted
count = ilr.count
from physical import rect, injection_kind
from schemes import spiral_kinds

def opt_corner(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NE, NW, SE, SW] if typ == 'X' else [SE, NE, NW, SW])
    ss = set(sup)
    return [c if c in ss else None for c in order]

def opt_central(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NW, NE, SE, SW] if typ == 'X' else [NE, SE, NW, SW])
    ss = set(sup)
    return [c if c in ss else None for c in order]

for d in (3, 5):
    ck = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
    sk = dict(spiral_kinds(0, 0, d))
    count(ck, d, opt_corner, f'corner/opt/d{d}')
    count(sk, d, opt_central, f'central/opt/d{d}')
