"""Fast leading-order census via stim's detector error model: give every
channel of one species the same tiny probability EPS; each DEM error's
probability is then (count x EPS), and its detector/observable signature
classifies the count as malignant (no detectors, observable flips),
rejected (fires a post-selected detector) or downstream-detected (fires
only later detectors, removed by the decoder). One stim call per census.
Note the post-selected dets are the FIRST n_post detectors by
construction in circuits.build."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stim
from circuits import build, corner_kinds, spiral_kinds, sched_lc, sched_nz

EPS = 1e-9


UNIT = {'p2': EPS / 15, 'p1': EPS / 3, 'pI': EPS, 'pM': EPS}


def species_census(d, kinds, sched, species, noisy_rounds=2, extra_rounds=2):
    kw = {species: EPS}
    unit = UNIT[species]
    txt, info = build(d, kinds, sched=sched, noisy_rounds=noisy_rounds,
                      extra_rounds=extra_rounds, **kw)
    c = stim.Circuit(txt)
    dem = c.detector_error_model(flatten_loops=True)
    n_post = info['n_post']
    mal = rej = down = 0
    for inst in dem:
        if inst.type != 'error':
            continue
        n = round(inst.args_copy()[0] / unit)
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        obs = any(t.is_logical_observable_id() for t in inst.targets_copy())
        if not dets:
            if obs: mal += n
        elif any(dv < n_post for dv in dets):
            rej += n
        else:
            down += n
    return mal, rej, down


def is_valid(d, kinds, sched):
    txt, _ = build(d, kinds, sched=sched)
    c = stim.Circuit(txt)
    det, obs = c.compile_detector_sampler().sample(4, separate_observables=True)
    return not det.any() and not obs.any()


if __name__ == '__main__':
    for d in (3, 5):
        for name, kinds in [('corner ', corner_kinds(d)),
                            ('central', spiral_kinds(d))]:
            for sname, sched in [('LC-TS', sched_lc), ('N/Z', sched_nz)]:
                m2, r2, d2_ = species_census(d, kinds, sched, 'p2')
                mI, rI, dI = species_census(d, kinds, sched, 'pI')
                mM, rM, dM = species_census(d, kinds, sched, 'pM')
                m1, r1_, d1_ = species_census(d, kinds, sched, 'p1')
                print(f'{name} d={d} {sname:5s}: CNOT mal {m2}/15, rej {r2}/15,'
                      f' down {d2_}/15 | init mal {mI}, rej {rI} |'
                      f' meas mal {mM}, rej {rM} | rot mal {m1}/3 of 3')
