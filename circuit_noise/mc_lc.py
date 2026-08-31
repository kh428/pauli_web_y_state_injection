"""P2-lite: Monte-Carlo validation. Samples the four scheme/schedule
combinations, post-selects, decodes accepted shots with MWPM (only
shots with non-trivial downstream syndrome need the decoder), and
reports p_L and reject rate. Two error models:
  (a) p2-only depolarizing CNOTs  -> p_L/p2 vs the counted slopes
  (b) all-equal p2=pI=p1=pM=p     -> p_L/p  vs the corrected totals"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim, pymatching
from circuits import build, corner_kinds, spiral_kinds, sched_lc

best_central = lambda typ: (['NW','NE','SE','SW'] if typ == 'X'
                            else ['NE','SE','NW','SW'])
best_corner = lambda typ: (['NE','SE','NW','SW'] if typ == 'X'
                           else ['SE','NE','NW','SW'])
SCHEMES = [('CR-LC',  corner_kinds(3), sched_lc),
           ('CR-opt', corner_kinds(3), best_corner),
           ('MR-LC',  spiral_kinds(3), sched_lc),
           ('MR-opt', spiral_kinds(3), best_central)]


def run_point(kinds, sched, model, p, shots):
    kw = dict(p2=p) if model == 'p2' else dict(p2=p, pI=p, p1=p, pM=p)
    txt, info = build(3, kinds, sched=sched, **kw)
    c = stim.Circuit(txt)
    dem = c.detector_error_model(decompose_errors=True)
    match = pymatching.Matching.from_detector_error_model(dem)
    n_post = info['n_post']
    sampler = c.compile_detector_sampler()
    acc = rej = err = 0
    CHUNK = 5_000_000
    left = shots
    while left > 0:
        n = min(CHUNK, left)
        left -= n
        det, obs = sampler.sample(n, separate_observables=True)
        keep = ~det[:, :n_post].any(axis=1)
        rej += int(n - keep.sum())
        det, obs = det[keep], obs[keep, 0]
        acc += det.shape[0]
        down = det[:, n_post:]
        nz = down.any(axis=1)
        pred = np.zeros(det.shape[0], dtype=bool)
        if nz.any():
            pred[nz] = match.decode_batch(det[nz])[:, 0].astype(bool)
        err += int((pred != obs).sum())
    return acc, rej, err


results = {}
for model, ps in [('p2', [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]),
                  ('all', [1e-4, 3e-4, 1e-3])]:
    for name, kinds, sched in SCHEMES:
        for p in ps:
            shots = min(int(4000 / (0.4 * p)), 60_000_000)
            acc, rej, err = run_point(kinds, sched, model, p, shots)
            pl = err / acc
            results[(model, name, p)] = (acc, rej, err)
            print(f'{model:3s} {name:6s} p={p:.0e}: shots={acc+rej}, '
                  f'reject={rej/(acc+rej):.4f}, p_L={pl:.3e}, '
                  f'p_L/p={pl/p:.3f} +- {np.sqrt(err)/acc/p:.3f}')

with open('mc_results.json', 'w') as fh:
    json.dump({f'{m}|{n}|{p}': v for (m, n, p), v in results.items()}, fh)
print('saved mc_results.json')
