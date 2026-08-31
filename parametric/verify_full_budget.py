"""Recreate the FULL central-scheme error budget from deterministic
fault insertion with the tsim engine: every CNOT class (p2), every
data/site initialisation flip (pI), the three rotation classes (p1)
and every noisy-round readout flip (pM), classified by their
detector/observable flips on the campaign circuit (2 noisy + 2 clean
rounds + deflation readout, accept iff all detectors zero).
Run for both the Tomita-Svore ordering and the optimised serialised
schedule."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tsim
import stim as _stim
from circuits import build, spiral_kinds, sched_lc
from t_experiment import readout_tail, opt_central

d = 3
P2 = [(a, b) for a in 'IXYZ' for b in 'IXYZ' if (a, b) != ('I', 'I')]

def run_schedule(sched, name):
    kinds = spiral_kinds(d)
    site0 = next(c for c, k in kinds.items() if k == 'Y')
    txt, info = build(d, kinds, sched=sched, p2=0.0,
                      noisy_rounds=2, extra_rounds=2, tail=False)
    tail, lw, col_off, row_off = readout_tail(d, kinds, info)
    CONSTS = {}
    for basis, kk, post in [('Z', '+', f'H {lw}'), ('X', '0', '')]:
        kcal = dict(kinds); kcal[site0] = kk
        ct, _ = build(d, kcal, sched=sched, tail=False,
                      site_override=site0)
        cfull = ct + '\n' + tail + ('\n' + post if post else '') \
            + f'\nM {lw}'
        m = _stim.Circuit(cfull).compile_sampler().sample(64)
        m = m.astype('uint8')
        offs = col_off if basis == 'Z' else row_off
        nc = m.shape[1]
        par = (m[:, [nc - 1 + o for o in offs]].sum(axis=1)
               + m[:, -1]) % 2
        assert (par == par[0]).all()
        CONSTS[basis] = int(par[0])
    fix = []
    for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
    if CONSTS['Z']: fix.append(f'Z {lw}')
    for o in row_off: fix.append(f'CX rec[{o}] {lw}')
    if CONSTS['X']: fix.append(f'X {lw}')
    TAIL = '\n' + tail + '\n' + '\n'.join(fix) + \
        f'\nS_DAG {lw}\nH {lw}\nM {lw}'
    dets = info['dets']
    lines = txt.split('\n')
    q_ = lambda c: d * c[0] + c[1]

    def flips(insert_at, gates):
        L = list(lines)
        L.insert(insert_at + 1, gates)
        m = tsim.Circuit('\n'.join(L) + TAIL).compile_sampler() \
            .sample(4).astype('uint8')
        det = np.zeros((4, len(dets)), np.uint8)
        for i, (lab, recs) in enumerate(dets):
            for r in recs:
                det[:, i] ^= m[:, r]
        ob = m[:, -1]
        assert (det == det[0]).all() and (ob == ob[0]).all()
        return det[0], int(ob[0])

    m0 = tsim.Circuit('\n'.join(lines) + TAIL).compile_sampler() \
        .sample(4).astype('uint8')
    base_det = np.zeros((4, len(dets)), np.uint8)
    for i, (lab, recs) in enumerate(dets):
        for r in recs:
            base_det[:, i] ^= m0[:, r]
    assert not base_det.any() and not m0[:, -1].any()

    # p2: CNOT channels of rounds 0,1
    seen = {}
    n2 = 0
    for li, l in enumerate(lines):
        ps = l.split()
        if ps and ps[0] == 'CX':
            pair = (int(ps[1]), int(ps[2]))
            k = seen.get(pair, 0); seen[pair] = k + 1
            if k >= 2: continue
            for (Pa, Pt) in P2:
                ops = []
                if Pa != 'I': ops.append(f'{Pa} {pair[0]}')
                if Pt != 'I': ops.append(f'{Pt} {pair[1]}')
                dv, ov = flips(li, '\n'.join(ops))
                if not dv.any() and ov: n2 += 1

    # pI: one flip per initialisation (Z on |+> and |Y>, X on |0>)
    nI = 0
    site_q = q_(site0)
    for li, l in enumerate(lines):
        ps = l.split()
        if ps and ps[0] in ('R', 'RX') and len(ps) == 2 \
                and int(ps[1]) < d * d:
            qq = int(ps[1])
            if qq == site_q: continue          # handled after its S
            dv, ov = flips(li, ('X' if ps[0] == 'R' else 'Z') + f' {qq}')
            if not dv.any() and ov: nI += 1
    i_s = next(i for i, l in enumerate(lines)
               if l == f'S {site_q}')
    dv, ov = flips(i_s, f'Z {site_q}')
    if not dv.any() and ov: nI += 1

    # p1: rotation channel, X/Y/Z after the S preparing the site
    n1 = 0
    for P in 'XYZ':
        dv, ov = flips(i_s, f'{P} {site_q}')
        if not dv.any() and ov: n1 += 1

    # pM: readout flips of the two noisy rounds
    nM = 0
    seenm = {}
    for li, l in enumerate(lines):
        ps = l.split()
        if ps and ps[0] in ('M', 'MX') and len(ps) == 2 \
                and int(ps[1]) >= d * d:
            qq = int(ps[1])
            k = seenm.get(qq, 0); seenm[qq] = k + 1
            if k >= 2: continue
            L = list(lines)
            L.insert(li, ('X' if ps[0] == 'M' else 'Z') + f' {qq}')
            m = tsim.Circuit('\n'.join(L) + TAIL).compile_sampler() \
                .sample(4).astype('uint8')
            det = np.zeros((4, len(dets)), np.uint8)
            for i, (lab, recs) in enumerate(dets):
                for r in recs:
                    det[:, i] ^= m[:, r]
            assert (det == det[0]).all() and (m[:, -1] == m[0, -1]).all()
            if not det[0].any() and m[0, -1]: nM += 1

    print(f'{name}:  p_L = ({n2}/15) p2 + {nI} pI + ({n1}/3) p1 '
          f'+ {nM} pM')
    return n2, nI, n1, nM

run_schedule(sched_lc, 'central, Tomita-Svore ')
run_schedule(opt_central, 'central, optimised    ')
