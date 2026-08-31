"""P4a: exhaustive search over uniform interleaved schedules. Every
X-check follows one permutation of (NE,NW,SE,SW), every Z-check another:
24 x 24 = 576 combos. Validity = deterministic baseline (a bad
interleave breaks the measured stabiliser group). Objective = malignant
CNOT classes (downstream-decoded criterion), tie-broken by reject slope."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circuits import corner_kinds, spiral_kinds
from dem_census import species_census, is_valid

OFFS = ['NE', 'NW', 'SE', 'SW']
d = 3

results = {}
for name, kinds in [('corner', corner_kinds(d)), ('central', spiral_kinds(d))]:
    best = []
    valid = 0
    hist = {}
    for px in itertools.permutations(OFFS):
        for pz in itertools.permutations(OFFS):
            sched = lambda typ, px=px, pz=pz: list(px if typ == 'X' else pz)
            if not is_valid(d, kinds, sched):
                continue
            valid += 1
            mal, rej, down = species_census(d, kinds, sched, 'p2')
            hist[mal] = hist.get(mal, 0) + 1
            best.append((mal, rej, px, pz))
    best.sort()
    results[name] = (valid, hist, best)
    print(f'{name}: {valid} valid schedules of 576; malignant-count '
          f'histogram {dict(sorted(hist.items()))}')
    for mal, rej, px, pz in best[:4]:
        print(f'   mal {mal}/15  rej {rej}/15   X:{"".join(o[0]+o[1] for o in px)} '
              f'Z:{"".join(o[0]+o[1] for o in pz)}')

# verify the best central schedule at d=5
name = 'central'
mal, rej, px, pz = results[name][2][0]
sched = lambda typ, px=px, pz=pz: list(px if typ == 'X' else pz)
m5, r5, _ = species_census(5, spiral_kinds(5), sched, 'p2')
print(f'best central schedule at d=5: mal {m5}/15, rej {r5}/15')
m5c, r5c, _ = species_census(5, corner_kinds(5), sched, 'p2')
print(f'same schedule, corner at d=5: mal {m5c}/15')
