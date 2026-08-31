"""|T> injection experiment (user's route A readout): noisy protocol,
error-free tail = frame fix (destabilisers conditioned on the final
perfect round) + tableau unencoding to the site wire + T_DAG, H, M.
LER = P(1 | all detectors zero). Identical pipeline runs |Y> (S_DAG
readout) as a sanity anchor. Predictions from iter_t_types: |T> zero-
syndrome infidelity = [n_Z + (n_X+n_Y)/2] p2/15; |Y>: (n_X+n_Z) p2/15."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from circuits import build, spiral_kinds, rect_checks

opt_central = lambda typ: (['NW','NE','SE','SW'] if typ == 'X'
                           else ['NE','SE','NW','SW'])
d = 3


def gf2_fit(A, x):
    """Solve A a = x (mod 2) with an affine constant; A shape (shots, n)."""
    import numpy as np
    A = np.hstack([A, np.ones((A.shape[0], 1), np.uint8)])
    AA = np.hstack([A, x[:, None]]).astype(np.uint8) % 2
    rows, cols = AA.shape
    piv, r = [], 0
    for cidx in range(cols - 1):
        rr = next((i for i in range(r, rows) if AA[i, cidx]), None)
        if rr is None: continue
        AA[[r, rr]] = AA[[rr, r]]
        for i in range(rows):
            if i != r and AA[i, cidx]: AA[i] ^= AA[r]
        piv.append(cidx); r += 1
    a = np.zeros(cols - 1, np.uint8)
    for i, cidx in enumerate(piv): a[cidx] = AA[i, -1]
    assert not ((A @ a + x) % 2).any(), 'gf2 fit failed'
    return [i for i in range(cols - 2) if a[i]], int(a[-1])


def calibrate_eigenmap(d, kinds, info, base_txt):
    """For each check, the GF(2) map from raw measurement record to the
    ACTUAL current eigenvalue after the last round (serialised schedules
    dress the measured operators, so raw final-round bits are not it).
    Calibrated on the Clifford (|Y>) circuit; propagation is
    state-independent."""
    import numpy as np, stim
    q = lambda c: d * c[0] + c[1]
    L = [base_txt]
    for typ, pos, sup in info['checks']:
        terms = [('X' if typ == 'X' else 'Z') + str(q(cell))
                 for cell in sup.values()]
        L.append('MPP ' + '*'.join(terms))
    m = stim.Circuit('\n'.join(L)).compile_sampler().sample(300)
    m = m.astype(np.uint8)
    nmeas = info['nmeas']
    fits = []
    for k in range(len(info['checks'])):
        fits.append(gf2_fit(m[:, :nmeas], m[:, nmeas + k]))
    return fits


def readout_tail(d, kinds, info):
    """Noiseless projection back to the single site qubit (Litinski-
    style patch deflation): measure every other data qubit in X along
    the logical column, Z along the logical row, Z elsewhere. Then
    Xbar = X_site * (product of column X outcomes) and
    Zbar = Z_site * (product of row Z outcomes) hold as operator
    identities, so the site is the logical qubit up to Z^a X^b with
    a, b the recorded parities. Returns (tail_text, site_wire,
    col_offsets, row_offsets): offsets are rec positions counted from
    the end of the tail block."""
    q = lambda c: d * c[0] + c[1]
    site = info['site']
    col = [(site[0], r) for r in range(d) if r != site[1]]
    row = [(c, site[1]) for c in range(d) if c != site[0]]
    rest = [(c, r) for c in range(d) for r in range(d)
            if (c, r) != site and (c, r) not in col and (c, r) not in row]
    L = []
    order = []
    for cell in col: L.append(f'MX {q(cell)}'); order.append(('col', cell))
    for cell in row: L.append(f'M {q(cell)}');  order.append(('row', cell))
    for cell in rest: L.append(f'M {q(cell)}'); order.append(('rest', cell))
    n = len(order)
    col_off = [i - n for i, (kind, _) in enumerate(order) if kind == 'col']
    row_off = [i - n for i, (kind, _) in enumerate(order) if kind == 'row']
    return '\n'.join(L), q(site), col_off, row_off


def run(state, p2, shots, sampler_mod, dd=None):
    global d
    if dd is not None: d = dd
    import stim as _stim
    kinds = spiral_kinds(d)
    site0 = next(c for c, k in kinds.items() if k == 'Y')
    kinds[site0] = state                    # 'Y' or 'T'
    txt, info = build(d, kinds, sched=opt_central, p2=p2, tail=False)
    tail, lw, col_off, row_off = readout_tail(d, kinds, info)
    nmeas = info['nmeas']
    # frame: Z^a on site with a = parity of column X outcomes, X^b with
    # b = parity of row Z outcomes; plus a possible constant per basis,
    # calibrated on one noiseless Clifford run each (exact frames, so
    # the constant is support-independent)
    consts = {}
    for basis, kk, post, gate in [('Z', '+', f'H {lw}', 'Z'),
                                  ('X', '0', '', 'X')]:
        kcal = dict(kinds); kcal[site0] = kk
        ct, _ = build(d, kcal, sched=opt_central, tail=False,
                      site_override=site0)
        cfull = ct + '\n' + tail + ('\n' + post if post else '') + f'\nM {lw}'
        m = _stim.Circuit(cfull).compile_sampler().sample(64).astype('uint8')
        offs = col_off if basis == 'Z' else row_off
        par = (m[:, [nmeas + len(m[0]) - nmeas - 1 + o for o in offs]]
               .sum(axis=1) + m[:, -1]) % 2
        assert (par == par[0]).all(), f'{basis} const not constant'
        consts[basis] = int(par[0])
    fix = []
    for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
    if consts['Z']: fix.append(f'Z {lw}')
    for o in row_off: fix.append(f'CX rec[{o}] {lw}')
    if consts['X']: fix.append(f'X {lw}')
    und = 'S_DAG' if state == 'Y' else 'T_DAG'
    full = (txt + '\n' + tail + '\n' + '\n'.join(fix) +
            f'\n{und} {lw}\nH {lw}\nM {lw}')
    c = sampler_mod.Circuit(full)
    s = c.compile_sampler()
    nmeas = info['nmeas']
    dets = info['dets']
    acc = bad = 0
    CH = 2_000_000
    left = shots
    while left > 0:
        n = min(CH, left); left -= n
        m = s.sample(n)
        det = np.zeros((n, len(dets)), bool)
        for i, (lab, recs) in enumerate(dets):
            for r in recs:
                det[:, i] ^= m[:, r].astype(bool)
        keep = ~det.any(axis=1)
        acc += int(keep.sum())
        bad += int(m[keep, -1].sum())
    return acc, bad


if __name__ == '__main__':
    import tsim
    # noiseless validation: P(1) must be 0 for both states
    for state in ('Y', 'T'):
        acc, bad = run(state, 0.0, 20000, tsim)
        print(f'noiseless {state}: accepted {acc}, P(1) = {bad/acc:.5f}')
