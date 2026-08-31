import os
"""Step 4: the full uniform-pair schedule search (24 x 24 = 576) at
d=3 under the DECODED convention (2 prep rounds + 2 appended future
rounds, face-scan-sorted build so colliding pairs serialise in face
order). For each valid candidate: malignant CNOT-class count n2 for
the corner and the central placement. Verifies the draft claims:
(a) no collision-free pair beats TS's 8/15 (corner) / 7/15 (central);
(b) the minimum over all 576 is 6 (corner) / 3 (central)."""
import sys, os, itertools, traceback
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'figures_3d', 'src')
sys.path.insert(0, SRC)
import iter_lc_reconstruct as ilr
from gates import GateBuilder
from physical import rect as _rect, region_checks, rect, injection_kind
from schemes import spiral_kinds

def build_sorted(kinds, d, sched_fn, prep_rounds=2, extra_rounds=2):
    b = GateBuilder()
    b.init_cells(kinds)
    P = _rect(0, 0, d, d)
    xs, zs = region_checks(P)
    schedules = ([('X', pos, sched_fn(pos, sup, 'X')) for pos, sup in xs] +
                 [('Z', pos, sched_fn(pos, sup, 'Z')) for pos, sup in zs])
    schedules.sort(key=lambda t: (t[1][0], t[1][1]))
    tags = []
    for r in range(prep_rounds + extra_rounds):
        tag = f'p{r}' if r < prep_rounds else f'x{r}'
        tags.append(tag)
        b.gate_round(P, tag,
                     schedules=[(t, p, list(s)) for t, p, s in schedules])
    b.open_outputs(P)
    return b.finish() + (set(tags[:prep_rounds]),)

ilr.build = build_sorted

CORNERS = ['NW', 'NE', 'SW', 'SE']
def sched_from(xperm, zperm):
    def fn(pos, sup, typ):
        fi, fj = pos[0] - 0.5, pos[1] - 0.5
        C = {'NW': (fi, fj + 1), 'NE': (fi + 1, fj + 1),
             'SW': (fi, fj),     'SE': (fi + 1, fj)}
        order = [C[c] for c in (xperm if typ == 'X' else zperm)]
        ss = set(sup)
        return [c if c in ss else None for c in order]
    return fn

def collision_free(xperm, zperm, d=3):
    P = _rect(0, 0, d, d)
    xs, zs = region_checks(P)
    fn = sched_from(xperm, zperm)
    for k in range(4):
        seen = {}
        for typ, lst in (('X', xs), ('Z', zs)):
            for pos, sup in lst:
                c = fn(pos, sup, typ)[k]
                if c is None: continue
                if c in seen and seen[c] != typ:
                    return False
                seen[c] = typ
    return True

d = 3
ck = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
sk = dict(spiral_kinds(0, 0, d))
results = []
perms = list(itertools.permutations(CORNERS))
for i, xp in enumerate(perms):
    for j, zp in enumerate(perms):
        fn = sched_from(xp, zp)
        cf = collision_free(xp, zp, d)
        row = {'x': ''.join(c[-1] if len(c)==2 else c for c in xp),
               'z': ''.join(c[-1] if len(c)==2 else c for c in zp),
               'xp': xp, 'zp': zp, 'cf': cf}
        for tag, kinds in (('corner', ck), ('central', sk)):
            try:
                n2 = ilr.count(kinds, d, fn, f'{tag}/{i},{j}')[0]
            except Exception:
                n2 = None
            row[tag] = n2
        results.append(row)
        print(f"{','.join(xp)} | {','.join(zp)} cf={cf} "
              f"corner={row['corner']} central={row['central']}", flush=True)

valid = [r for r in results if r['corner'] is not None
         and r['central'] is not None]
print(f'\n=== {len(valid)} valid of {len(results)} pairs ===')
for tag in ('corner', 'central'):
    vs = [r[tag] for r in valid]
    cfv = [r[tag] for r in valid if r['cf']]
    print(f'{tag}: min over all = {min(vs)}, '
          f'min collision-free = {min(cfv)} '
          f'({len(cfv)} collision-free valid)')
    best = min(vs)
    for r in valid:
        if r[tag] == best:
            print(f'  best {tag}: X={r["xp"]} Z={r["zp"]} cf={r["cf"]}')
