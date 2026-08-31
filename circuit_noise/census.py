"""Deterministic single-fault census via stim: for every CNOT class and
init flip in the noisy (post-selected) rounds, insert the fault as a
probability-1 Pauli and read the detector/observable response of one
shot. Malignant = no detector fires AND the logical Y observable flips.
This is Li's / Lao-Criger's leading-order counting, computed by stim,
fully independent of the GF(2) web solver."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stim
from circuits import build, corner_kinds, spiral_kinds, sched_lc, sched_nz

P2 = ['I', 'X', 'Y', 'Z']


def response(txt):
    c = stim.Circuit(txt)
    det, obs = c.compile_detector_sampler().sample(
        1, separate_observables=True)
    return bool(det[0].any()), bool(obs[0][0])


def census(d, kinds, sched, verbose=False):
    base, info = build(d, kinds, sched=sched)
    fired, flip = response(base)
    assert not fired and not flip, 'baseline not clean'
    n2 = 0
    gates = {}
    for (rd, ci, slot, typ, pos, cell) in info['catalogue']:
        mal = []
        for Pa in P2:
            for Pt in P2:
                if Pa == 'I' and Pt == 'I': continue
                txt, _ = build(d, kinds, sched=sched,
                               fault=('cnot', rd, ci, slot, Pa, Pt))
                fired, flip = response(txt)
                if not fired and flip:
                    mal.append((Pa, Pt))
        if mal:
            n2 += len(mal)
            gates[(rd, typ, pos, cell)] = mal
            if verbose:
                if typ == 'Z':          # data is control: (C,T)=(data,anc)
                    conv = [(Pt, Pa) for Pa, Pt in mal]
                else:
                    conv = list(mal)
                print(f'  round {rd} {typ}{pos} data {cell} slot: '
                      f'{len(mal)} classes (C,T)={conv}')
    nI = 0
    for cell in sorted(kinds):
        txt, _ = build(d, kinds, sched=sched, fault=('init', cell))
        fired, flip = response(txt)
        if not fired and flip:
            nI += 1
            if verbose: print(f'  init flip {cell}')
    return n2, nI, gates


if __name__ == '__main__':
    for d in (3, 5):
        for name, kinds in [('CR/corner', corner_kinds(d)),
                            ('MR/central', spiral_kinds(d))]:
            for sname, sched in [('LC-TS', sched_lc), ('N/Z', sched_nz)]:
                v = (d == 3 and sname == 'LC-TS')
                n2, nI, _ = census(d, kinds, sched, verbose=v)
                print(f'{name:11s} d={d} sched={sname:5s}: '
                      f'({n2}/15) p2 + {nI} pI')
