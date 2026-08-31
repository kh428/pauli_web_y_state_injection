import os
"""Step 5b: heterogeneous-ordering search under the DECODED
convention, d=3, central (then corner) placement. Every check carries
its own visiting order (a permutation of its plaquette corners);
random restarts + single-check hill climbing. Re-establishes the
draft claim 'nothing below 3/15 (central) / 6/15 (corner)'."""
import sys, os, itertools, random
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
PERMS = list(itertools.permutations(range(4)))
d = 3
P = _rect(0, 0, d, d)
xs, zs = region_checks(P)
CHECKS = [('X', pos, sup) for pos, sup in xs] + \
         [('Z', pos, sup) for pos, sup in zs]

def corners_of(pos):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    return {'NW': (fi, fj + 1), 'NE': (fi + 1, fj + 1),
            'SW': (fi, fj),     'SE': (fi + 1, fj)}

def sched_from_assign(assign):
    key = {(t, p): perm for (t, p), perm in assign.items()}
    def fn(pos, sup, typ):
        C = corners_of(pos)
        perm = key[(typ, pos)]
        order = [C[CORNERS[k]] for k in perm]
        ss = set(sup)
        return [c if c in ss else None for c in order]
    return fn

evals = {}
def score(kinds, assign, tag):
    k = tuple(sorted(assign.items()))
    if k in evals: return evals[k]
    try:
        n2 = ilr.count(kinds, d, sched_from_assign(assign), tag)[0]
    except Exception:
        n2 = 10**9
    evals[k] = n2
    return n2

def search(kinds, name, restarts=40, budget=2500, seed=1):
    rng = random.Random(seed)
    global evals; evals = {}
    best_overall = 10**9; nev = 0
    for r in range(restarts):
        assign = {(t, p): rng.choice(PERMS) for t, p, s in CHECKS}
        cur = score(kinds, assign, f'{name}/r{r}/init'); nev += 1
        improved = True
        while improved and nev < budget:
            improved = False
            order = list(assign.keys()); rng.shuffle(order)
            for ck in order:
                base = assign[ck]
                for perm in PERMS:
                    if perm == base: continue
                    assign[ck] = perm
                    v = score(kinds, assign, f'{name}/r{r}/hc'); nev += 1
                    if v < cur:
                        cur = v; base = perm; improved = True
                    else:
                        assign[ck] = base
                    if nev >= budget: break
                if nev >= budget: break
        if cur < best_overall:
            best_overall = cur
            print(f'[{name}] restart {r}: new best n2={cur} '
                  f'(evals so far {nev})', flush=True)
        if nev >= budget:
            print(f'[{name}] budget reached at restart {r}', flush=True)
            break
    print(f'[{name}] FINAL best n2={best_overall}, evaluations={nev}, '
          f'unique candidates={len(evals)}', flush=True)
    return best_overall

sk = dict(spiral_kinds(0, 0, d))
ck = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
b1 = search(sk, 'central', restarts=40, budget=2500, seed=1)
b2 = search(ck, 'corner', restarts=40, budget=2500, seed=2)
print(f'\n=== decoded-convention heterogeneous floors (d=3): '
      f'central {b1}, corner {b2} ===')
