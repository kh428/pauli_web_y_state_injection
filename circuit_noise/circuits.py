"""Stim circuit builder for rotated-surface-code |Y> injection schemes.
Self-contained re-implementation of the layout/kind conventions of
../figures_3d (column-major cells (c,r),
r up; X-face iff (fi+fj) even; boundary pairs: X on N/S, Z on E/W).

Protocol (Li / Lao-Criger): init data, `noisy_rounds` rounds of scheduled
CNOT extraction (post-selected), `extra_rounds` perfect rounds
(downstream QEC), final perfect MPP of the logical Y representative.
Faults are inserted as deterministic Pauli gates; with no probabilistic
noise every detector value IS the fault's syndrome and the observable
bit IS the logical flip."""


def rect_checks(d):
    """[(typ, pos, {offset_name: cell}), ...] for a d x d patch."""
    checks = []
    for fi in range(-1, d):
        for fj in range(-1, d):
            quad = {'SW': (fi, fj), 'SE': (fi + 1, fj),
                    'NW': (fi, fj + 1), 'NE': (fi + 1, fj + 1)}
            present = {k: c for k, c in quad.items()
                       if 0 <= c[0] < d and 0 <= c[1] < d}
            typ = 'X' if (fi + fj) % 2 == 0 else 'Z'
            if len(present) == 4:
                checks.append((typ, (fi + .5, fj + .5), present))
            elif len(present) == 2:
                ns = (fj == -1 or fj == d - 1)      # horizontal pair exposed N/S
                ew = (fi == -1 or fi == d - 1)      # vertical pair exposed E/W
                if (typ == 'X' and ns and not ew) or (typ == 'Z' and ew and not ns):
                    checks.append((typ, (fi + .5, fj + .5), present))
    return checks


def corner_kinds(d):
    out = {}
    for c in range(d):
        for r in range(d):
            if (c, r) == (0, d - 1): out[(c, r)] = 'Y'
            else: out[(c, r)] = '+' if c + r <= d - 1 else '0'
    return out


def spiral_kinds(d):
    site = (d // 2, d // 2)
    plus = set()
    for k in range(1, (d - 1) // 2 + 1):
        for c in range(site[0] - k, site[0] + k):
            plus.add((c, site[1] + k))
    full = plus | {(2 * site[0] - c, 2 * site[1] - r) for c, r in plus}
    return {(c, r): ('Y' if (c, r) == site else
                     ('+' if (c, r) in full else '0'))
            for c in range(d) for r in range(d)}


def sched_lc(typ):
    """Lao-Criger / Tomita-Svore (their fig 4): absolute slots."""
    return ['NE', 'NW', 'SE', 'SW'] if typ == 'X' else ['NE', 'SE', 'NW', 'SW']


def sched_nz(typ):
    """Our generic N/Z interleave."""
    return ['NW', 'SW', 'NE', 'SE'] if typ == 'X' else ['NW', 'NE', 'SW', 'SE']


def build(d, kinds, sched=sched_lc, noisy_rounds=2, extra_rounds=2,
          p2=0.0, pI=0.0, p1=0.0, pM=0.0, fault=None, obs_basis='Y',
          frame=None, site_override=None, tail=True):
    """Returns (stim_circuit_text, info). fault: ('cnot', round, ci, slot,
    Pa, Pt) with Pa on the ancilla, Pt on the data, applied after that
    CNOT; or ('init', cell). info: dets list, per-CNOT catalogue."""
    checks = rect_checks(d)
    q = lambda c: d * c[0] + c[1]
    anc = {ci: d * d + ci for ci in range(len(checks))}
    L = []
    # data init
    for cell in sorted(kinds):
        k = kinds[cell]
        if k == '0': L.append(f'R {q(cell)}')
        elif k == '+': L.append(f'RX {q(cell)}')
        elif k == 'Y':
            L.append(f'RX {q(cell)}'); L.append(f'S {q(cell)}')
            if p1 > 0: L.append(f'DEPOLARIZE1({p1}) {q(cell)}')
        else:                                    # 'T' magic site
            L.append(f'RX {q(cell)}'); L.append(f'T {q(cell)}')
            if p1 > 0: L.append(f'DEPOLARIZE1({p1}) {q(cell)}')
        if fault == ('init', cell):
            L.append(('X_ERROR(1)' if k == '0' else 'Z_ERROR(1)') +
                     f' {q(cell)}')
        elif pI > 0:
            L.append(('X_ERROR' if k == '0' else 'Z_ERROR') +
                     f'({pI}) {q(cell)}')
    nmeas = 0
    mrec = {}                       # (round, ci) -> rec index
    catalogue = []                  # (round, ci, slot, typ, pos, cell)
    total_rounds = noisy_rounds + extra_rounds
    for rd in range(total_rounds):
        noisy = rd < noisy_rounds
        for ci, (typ, pos, sup) in enumerate(checks):
            L.append(('RX ' if typ == 'X' else 'R ') + str(anc[ci]))
        for slot, off in enumerate(['s0', 's1', 's2', 's3']):
            L.append('TICK')
            for ci, (typ, pos, sup) in enumerate(checks):
                try:
                    name = sched(typ, pos)[slot]
                except TypeError:
                    name = sched(typ)[slot]
                if name not in sup: continue
                cell = sup[name]
                a, dq = anc[ci], q(cell)
                if typ == 'X': L.append(f'CX {a} {dq}')
                else: L.append(f'CX {dq} {a}')
                if noisy and p2 > 0:
                    L.append(f'DEPOLARIZE2({p2}) {a} {dq}')
                if fault is not None and fault[0] == 'cnot' \
                        and fault[1:4] == (rd, ci, slot):
                    Pa, Pt = fault[4], fault[5]
                    tg = ([f'{Pa}{a}'] if Pa != 'I' else []) + \
                         ([f'{Pt}{dq}'] if Pt != 'I' else [])
                    L.append('E(1) ' + ' '.join(tg))
                if noisy:
                    catalogue.append((rd, ci, slot, typ, pos, cell))
        for ci, (typ, pos, sup) in enumerate(checks):
            noise = f'({pM}) ' if (noisy and pM > 0) else ' '
            L.append(('MX' if typ == 'X' else 'M') + noise + str(anc[ci]))
            mrec[(rd, ci)] = nmeas
            nmeas += 1
    # detectors: round-1 hatched; consecutive-round comparisons for all
    dets = []                       # (label, [rec indices])
    n_post = 0                      # leading dets that are post-selected
    for ci, (typ, pos, sup) in enumerate(checks):
        hatched = all(kinds[c] == ('+' if typ == 'X' else '0')
                      for c in sup.values())
        if hatched:
            dets.append((f'r0 {typ}{pos}', [mrec[(0, ci)]]))
            n_post += 1
    for rd in range(1, total_rounds):
        for ci, (typ, pos, sup) in enumerate(checks):
            dets.append((f'r{rd}-r{rd-1} {typ}{pos}',
                         [mrec[(rd, ci)], mrec[(rd - 1, ci)]]))
            if rd < noisy_rounds:
                n_post += 1
    # final perfect logical measurement, then detectors (rec offsets
    # are relative to the end, after the MPP = measurement nmeas).
    # obs_basis: 'Y' (default) = Y at site, X up the column, Z along the
    # row; 'X'/'Z' = the bare column/row logical. frame: extra absolute
    # rec indices XORed into the observable (learned frame function).
    if site_override is not None:
        site = site_override
    else:
        site = next(c for c, k in kinds.items() if k in ('Y', 'T'))
    if not tail:
        return '\n'.join(L), {'checks': checks, 'dets': dets,
                              'catalogue': catalogue, 'mrec': mrec,
                              'n_post': n_post, 'nmeas': nmeas,
                              'site': site, 'q': q(site)}
    if obs_basis == 'Y':
        terms = [f'Y{q(site)}']
        terms += [f'X{q((site[0], r))}' for r in range(d) if r != site[1]]
        terms += [f'Z{q((c, site[1]))}' for c in range(d) if c != site[0]]
    elif obs_basis == 'X':
        terms = [f'X{q((site[0], r))}' for r in range(d)]
    else:
        terms = [f'Z{q((c, site[1]))}' for c in range(d)]
    L.append('MPP ' + '*'.join(terms))
    total = nmeas + 1
    for label, recs in dets:
        L.append('DETECTOR ' + ' '.join(f'rec[{r - total}]' for r in recs))
    obs = ['rec[-1]'] + [f'rec[{r - total}]' for r in (frame or [])]
    L.append('OBSERVABLE_INCLUDE(0) ' + ' '.join(obs))
    return '\n'.join(L), {'checks': checks, 'dets': dets,
                           'catalogue': catalogue, 'mrec': mrec,
                           'n_post': n_post}
