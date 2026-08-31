"""P4a corrected: schedule search with the MWPM-aware objective. The
leading-order logical error rate under matching is
  (malignant + mis-decoded) classes:
malignant = zero syndrome & logical flip; mis-decoded = detected but the
MWPM correction restores the wrong logical. Both read off the DEM: every
DEM error mechanism is decoded and compared against its own observable."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim, pymatching
from circuits import build, corner_kinds, spiral_kinds, sched_lc
from dem_census import is_valid

P = 1e-5


def mwpm_census(d, kinds, sched):
    txt, info = build(d, kinds, sched=sched, p2=P)
    c = stim.Circuit(txt)
    dem = c.detector_error_model(decompose_errors=True)
    match = pymatching.Matching.from_detector_error_model(dem)
    ndet = c.num_detectors
    n_post = info['n_post']
    raw = c.detector_error_model(flatten_loops=True)
    mal = mis = 0
    rows, obss, cnts = [], [], []
    for inst in raw:
        if inst.type != 'error': continue
        n = round(inst.args_copy()[0] / (P / 15))
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        obs = any(t.is_logical_observable_id() for t in inst.targets_copy())
        if not dets:
            if obs: mal += n
            continue
        if any(dv < n_post for dv in dets):
            continue                    # fires a post-selected detector: rejected
        row = np.zeros(ndet, dtype=bool); row[dets] = True
        rows.append(row); obss.append(obs); cnts.append(n)
    if rows:
        pred = match.decode_batch(np.array(rows))[:, 0].astype(bool)
        for pr, ob, n in zip(pred, obss, cnts):
            if pr != ob: mis += n
    return mal, mis


if __name__ == '__main__':
    OFFS = ['NE', 'NW', 'SE', 'SW']
    d = 3
    for name, kinds in [('corner', corner_kinds(d)),
                        ('central', spiral_kinds(d))]:
        best = []
        for px in itertools.permutations(OFFS):
            for pz in itertools.permutations(OFFS):
                sched = lambda typ, px=px, pz=pz: list(px if typ == 'X' else pz)
                if not is_valid(d, kinds, sched): continue
                mal, mis = mwpm_census(d, kinds, sched)
                best.append((mal + mis, mal, mis, px, pz))
        best.sort()
        tot0 = best[0][0]
        print(f'{name}: best MWPM-aware total {tot0}/15 '
              f'({sum(1 for b in best if b[0]==tot0)} schedules)')
        for tot, mal, mis, px, pz in best[:4]:
            print(f'   tot {tot} = mal {mal} + mis {mis}  '
                  f'X:{"".join(o[0]+o[1] for o in px)} '
                  f'Z:{"".join(o[0]+o[1] for o in pz)}')
        # reference: LC's schedule
        mal, mis = mwpm_census(d, kinds, sched_lc)
        print(f'   [LC-TS schedule: mal {mal} + mis {mis} = {mal+mis}]')
