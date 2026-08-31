"""Parallel Monte-Carlo at d=9 and d=11 (10-worker pool)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def one(task):
    d, name, model, p, shots = task
    import stim, pymatching
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from circuits import build, corner_kinds, spiral_kinds, sched_lc
    opt_corner = lambda typ: (['NE','NW','SE','SW'] if typ == 'X'
                              else ['SE','NE','NW','SW'])
    opt_central = lambda typ: (['NW','NE','SE','SW'] if typ == 'X'
                               else ['NE','SE','NW','SW'])
    KINDS = {'CR-LC': (corner_kinds(d), sched_lc),
             'CR-opt': (corner_kinds(d), opt_corner),
             'MR-LC': (spiral_kinds(d), sched_lc),
             'MR-opt': (spiral_kinds(d), opt_central)}
    kinds, sched = KINDS[name]
    kw = dict(p2=p) if model == 'p2' else dict(p2=p, pI=p, p1=p, pM=p)
    txt, info = build(d, kinds, sched=sched, **kw)
    c = stim.Circuit(txt)
    match = pymatching.Matching.from_detector_error_model(
        c.detector_error_model(decompose_errors=True))
    n_post = info['n_post']
    sampler = c.compile_detector_sampler()
    acc = rej = err = 0
    left = shots
    while left > 0:
        n = min(2_000_000, left); left -= n
        det, obs = sampler.sample(n, separate_observables=True)
        keep = ~det[:, :n_post].any(axis=1)
        rej += int(n - keep.sum())
        det, obs = det[keep], obs[keep, 0]
        acc += det.shape[0]
        nz = det[:, n_post:].any(axis=1)
        pred = np.zeros(det.shape[0], dtype=bool)
        if nz.any():
            pred[nz] = match.decode_batch(det[nz])[:, 0].astype(bool)
        err += int((pred != obs).sum())
    return (f'{model}|{name}|d{d}|{p}', acc, rej, err)


if __name__ == '__main__':
    tasks = []
    for d in (9, 11):
        cap = 20_000_000 if d == 9 else 12_000_000
        for name in ('CR-LC', 'CR-opt', 'MR-LC', 'MR-opt'):
            for model, ps in [('p2', [2e-4, 5e-4, 2e-3]),
                              ('all', [3e-4, 1e-3])]:
                for p in ps:
                    shots = min(int(2000 / (0.4 * p)), cap)
                    tasks.append((d, name, model, p, shots))
    res = json.load(open('mc_results.json'))
    with ProcessPoolExecutor(max_workers=10) as ex:
        for key, acc, rej, err in ex.map(one, tasks):
            res[key] = (acc, rej, err)
            m, n, ds, p = key.split('|')
            p = float(p)
            print(f'{m} {n} {ds}: p={p:.0e} p_L/p={err/acc/p:.3f} '
                  f'+- {err**0.5/acc/p:.3f}', flush=True)
    json.dump(res, open('mc_results.json', 'w'))
    print('saved')
