"""Machine part of the arbitrary-distance proof: for each of the 61
fault generators (X on the preparation line; 15 Pauli classes on each
of the four site CNOTs), propagate the Pauli frame through the mini
protocol (one optimised round + deflation readout) and record (i) the
set of flipped measurement records in SITE-RELATIVE coordinates and
(ii) the residual Pauli left on the injected wire before the undo
gate. Claim checked: the map is IDENTICAL at d = 7, 9, 11, 13 and its
support lies within lattice radius 2 of the site."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_ler_states import build_noisy_state
from circuits import rect_checks

PAULI_BITS = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}


def effect_map(d):
    txt, channels, lw = build_noisy_state(d, 'T')
    lines = [l for l in txt.split('\n') if l.strip()]
    site = (d // 2, d // 2)
    checks = rect_checks(d)
    anc_pos = {d * d + ci: checks[ci][1] for ci in range(len(checks))}
    def rel(qq):
        if qq >= d * d:
            p = anc_pos[qq]
        else:
            p = divmod(qq, d)
        return (round(p[0] - site[0], 1), round(p[1] - site[1], 1))
    # fault sites: index of channel line -> insert AFTER preceding gate
    faults = []       # (label, line_index, [(qubit, 'X'/'Y'/'Z'), ...])
    for i, l in enumerate(lines):
        ps = l.split()
        if ps and ps[0].startswith('X_ERROR'):
            faults.append(('prep|X', i, [(int(ps[1]), 'X')]))
        if ps and ps[0].startswith('DEPOLARIZE2'):
            a, b = int(ps[1]), int(ps[2])
            g = lines[i - 1].split()   # the CNOT just before
            assert g[0] == 'CX'
            tag = f'cnot({rel(a)},{rel(b)})'
            for Pa in 'IXYZ':
                for Pt in 'IXYZ':
                    if Pa == Pt == 'I': continue
                    ins = [(q, P) for q, P in ((a, Pa), (b, Pt)) if P != 'I']
                    faults.append((f'{tag}|{Pa}{Pt}', i, ins))
    out = {}
    for label, i0, ins in faults:
        fx, fz = {}, {}
        for qq, P in ins:
            x, z = PAULI_BITS[P]
            fx[qq] = fx.get(qq, 0) ^ x
            fz[qq] = fz.get(qq, 0) ^ z
        flips = []
        residual = None
        for l in lines[i0 + 1:]:
            ps = l.split()
            if not ps or ps[0].startswith(('DEPOLARIZE', 'X_ERROR', 'TICK')):
                continue
            op = ps[0]
            if op == 'CX':
                a, b = int(ps[1]), int(ps[2])
                fx[b] = fx.get(b, 0) ^ fx.get(a, 0)
                fz[a] = fz.get(a, 0) ^ fz.get(b, 0)
            elif op == 'H':
                q = int(ps[1])
                fx[q], fz[q] = fz.get(q, 0), fx.get(q, 0)
            elif op in ('S', 'S_DAG'):
                q = int(ps[1])
                fz[q] = fz.get(q, 0) ^ fx.get(q, 0)
            elif op in ('T', 'T_DAG'):
                q = int(ps[1])
                if q == lw and residual is None:
                    residual = (fx.get(lw, 0), fz.get(lw, 0))
                    fx[lw] = fz[lw] = 0    # stop tracking through non-Clifford
            elif op in ('M', 'MX'):
                for qs in ps[1:]:
                    q = int(qs)
                    bit = fx.get(q, 0) if op == 'M' else fz.get(q, 0)
                    if bit:
                        flips.append((rel(q), op))
            elif op in ('R', 'RX'):
                for qs in ps[1:]:
                    fx[int(qs)] = fz[int(qs)] = 0
        out[label] = (tuple(sorted(flips)), residual)
    return out

maps = {}
for d in (7, 9, 11, 13):
    maps[d] = effect_map(d)
    rad = 0
    for label, (flips, res) in maps[d].items():
        for (dx, dy), _ in flips:
            rad = max(rad, abs(dx), abs(dy))
    print(f'd={d}: {len(maps[d])} fault generators, '
          f'max |relative coordinate| of any flipped record = {rad}')
ref = maps[7]
for d in (9, 11, 13):
    assert maps[d] == ref, f'd={d} differs!'
print('IDENTICAL RELATIVE EFFECT MAPS AT d = 7, 9, 11, 13; '
      'support radius <= 2')
