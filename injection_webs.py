"""Pauli webs of the |Y> state injection protocol on the rotated surface code.

Companion code to "Pauli web of the |Y> state surface code injection"
(arXiv:2501.15566). Everything here is derived, not asserted: the spider rule is
checked against the spider tensors, and closed Pauli webs are the null space of a
linear system over GF(2).

Conventions (matching the paper):
  * green web = Z-type decoration, red web = X-type decoration
  * green (Z) ancilla = X-type check, measured in X
  * red   (X) ancilla = Z-type check, measured in Z
  * measurement outcomes are classical: the measurement effect is a one-legged
    spider of the ancilla's own colour, so it fuses away and is NOT drawn.
    A check enters a web exactly when the web covers its ancilla on every leg.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pyzx as zx
from pyzx import VertexType

__all__ = ['XCHECK', 'ZCHECK', 'init_kind', 'build', 'web_system', 'nullspace',
           'closed_webs', 'solve_with', 'vec_to_web', 'audit', 'detectors',
           'spider_tensor', 'rule_allowed', 'brute_force_allowed', 'versions']

D = 5

# plaquette supports and ancilla positions, read out of the paper's TikZ
XCHECK = {0: [0, 1, 5, 6], 1: [2, 3, 7, 8], 2: [4, 9], 3: [5, 10], 4: [6, 7, 11, 12],
          5: [8, 9, 13, 14], 6: [10, 11, 15, 16], 7: [12, 13, 17, 18], 8: [14, 19],
          9: [15, 20], 10: [16, 17, 21, 22], 11: [18, 19, 23, 24]}
ZCHECK = {0: [0, 1], 1: [2, 3], 2: [1, 2, 6, 7], 3: [3, 4, 8, 9], 4: [5, 6, 10, 11],
          5: [7, 8, 12, 13], 6: [11, 12, 16, 17], 7: [13, 14, 18, 19], 8: [15, 16, 20, 21],
          9: [17, 18, 22, 23], 10: [21, 22], 11: [23, 24]}
XPOS = {0: (.5, .5), 1: (.5, 2.5), 2: (.5, 4.5), 3: (1.5, -.5), 4: (1.5, 1.5), 5: (1.5, 3.5),
        6: (2.5, .5), 7: (2.5, 2.5), 8: (2.5, 4.5), 9: (3.5, -.5), 10: (3.5, 1.5),
        11: (3.5, 3.5)}
ZPOS = {0: (-.5, .5), 1: (-.5, 2.5), 2: (.5, 1.5), 3: (.5, 3.5), 4: (1.5, .5), 5: (1.5, 2.5),
        6: (2.5, 1.5), 7: (2.5, 3.5), 8: (3.5, .5), 9: (3.5, 2.5), 10: (4.5, 1.5),
        11: (4.5, 3.5)}


def cr(i):
    """data qubit index -> (column, row)"""
    return i // D, i % D


def init_kind(i, corner='Y'):
    """Lao-Criger initialisation: |+> on and below the anti-diagonal, |0> above,
    the injected state on the top-left corner."""
    c, r = cr(i)
    if (c, r) == (0, D - 1):
        return corner
    return 'plus' if c + r <= D - 1 else 'zero'


# --------------------------------------------------------------------- diagram
def build(rounds=1, corner='Y'):
    """The injection spacetime diagram with `rounds` full rounds of syndrome
    extraction (each round = an X-check layer then a Z-check layer).

    Returns (graph, V, meta). V maps a key to a vertex; keys are
      ('init', i)          initial state of data qubit i
      ('d', r, 'x'|'z', i) data qubit i in round r, X-layer or Z-layer
      ('bx', r, j)         X-check ancilla j in round r   (green, measured in X)
      ('az', r, k)         Z-check ancilla k in round r   (red,   measured in Z)
      ('out', i)           open output boundary of data qubit i
    """
    g, V, meta = zx.Graph(), {}, {}

    def add(key, ty, x, y, z, phase=0, role=''):
        v = g.add_vertex(ty, qubit=y, row=x, phase=phase)
        g.set_vdata(v, 'z', z)
        V[key] = v
        meta[v] = dict(key=key, role=role)
        return v

    dz = 2.0                                   # vertical spacing between layers
    for i in range(D * D):
        c, r = cr(i)
        k = init_kind(i, corner)
        ty = VertexType.X if k == 'zero' else VertexType.Z
        ph = Fraction(1, 2) if k == 'Y' else (Fraction(1, 4) if k == 'T' else 0)
        add(('init', i), ty, c, r, 0.0, ph, 'init_' + k)
        prev = ('init', i)
        for rd in range(1, rounds + 1):
            for lay, ty_ in (('x', VertexType.X), ('z', VertexType.Z)):
                z = dz * (2 * (rd - 1) + (1 if lay == 'x' else 2))
                add(('d', rd, lay, i), ty_, c, r, z, 0, f'data_r{rd}{lay}')
                g.add_edge((V[prev], V[('d', rd, lay, i)]))
                prev = ('d', rd, lay, i)
        add(('out', i), VertexType.BOUNDARY, c, r, dz * (2 * rounds + 1), 0, 'output')
        g.add_edge((V[prev], V[('out', i)]))

    for rd in range(1, rounds + 1):
        for j, qs in XCHECK.items():           # green ancilla, X-type check
            x, y = XPOS[j]
            add(('bx', rd, j), VertexType.Z, x, y, dz * (2 * (rd - 1) + 1), 0, f'ancX_r{rd}')
            for i in qs:
                g.add_edge((V[('bx', rd, j)], V[('d', rd, 'x', i)]))
        for k, qs in ZCHECK.items():           # red ancilla, Z-type check
            x, y = ZPOS[k]
            add(('az', rd, k), VertexType.X, x, y, dz * (2 * (rd - 1) + 2), 0, f'ancZ_r{rd}')
            for i in qs:
                g.add_edge((V[('az', rd, k)], V[('d', rd, 'z', i)]))
    return g, V, meta


# ------------------------------------------------------------------ GF(2) core
def rref(M):
    M = M.copy() % 2
    rows, cols = M.shape
    piv, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i, c]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return M[:r], piv


def nullspace(M):
    if M.shape[0] == 0:
        return np.eye(M.shape[1], dtype=np.uint8)
    R, piv = rref(M)
    n = M.shape[1]
    basis = []
    for f in [c for c in range(n) if c not in piv]:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for r, c in enumerate(piv):
            if R[r, f]:
                v[c] = 1
        basis.append(v)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, n), np.uint8)


def web_system(g):
    """Closed Pauli webs = null space of this system.

    Unknowns: x_e, z_e per edge (is it decorated red / green), plus one flag
    lambda_v per spider for the all-or-none opposite-colour condition.
    """
    edges = [tuple(sorted(g.edge_st(e))) for e in g.edges()]
    ei = {e: k for k, e in enumerate(edges)}
    spiders = [v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY]
    nE = len(edges)
    xv = {e: 2 * ei[e] for e in edges}
    zv = {e: 2 * ei[e] + 1 for e in edges}
    lam = {v: 2 * nE + k for k, v in enumerate(spiders)}
    ncols = 2 * nE + len(spiders)
    rows = []
    for v in spiders:
        legs = [tuple(sorted((v, w))) for w in g.neighbors(v)]
        own_is_z = (g.type(v) == VertexType.Z)
        half = (Fraction(g.phase(v)).denominator == 2)
        for e in legs:                         # opposite colour = lambda_v on every leg
            r = np.zeros(ncols, np.uint8)
            r[(xv if own_is_z else zv)[e]] = 1
            r[lam[v]] = 1
            rows.append(r)
        r = np.zeros(ncols, np.uint8)          # own-colour parity
        for e in legs:
            r[(zv if own_is_z else xv)[e]] ^= 1
        if half:
            r[lam[v]] ^= 1
        rows.append(r)
    return np.array(rows, np.uint8), edges, xv, zv, ncols


def solve_with(g, fixed=None):
    """Closed webs subject to {column: bit}. Returns (particular, basis, maps) or None
    if the constraints are inconsistent (i.e. no such web exists)."""
    A, edges, xv, zv, ncols = web_system(g)
    rows, rhs = [A], [np.zeros(A.shape[0], np.uint8)]
    for col, bit in (fixed or {}).items():
        r = np.zeros(ncols, np.uint8)
        r[col] = 1
        rows.append(r[None, :])
        rhs.append(np.array([bit], np.uint8))
    M = np.vstack(rows)
    b = np.concatenate(rhs)
    R, piv = rref(np.hstack([M, b[:, None]]))
    if ncols in piv:
        return None
    part = np.zeros(ncols, np.uint8)
    for r, c in enumerate(piv):
        part[c] = R[r, ncols]
    return part, nullspace(M), (edges, xv, zv)


def closed_webs(g):
    A, edges, xv, zv, _ = web_system(g)
    return nullspace(A), edges, xv, zv


def vec_to_web(vec, edges, xv, zv):
    m = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}
    out = {}
    for e in edges:
        p = m[(int(vec[xv[e]]), int(vec[zv[e]]))]
        if p != 'I':
            out[e] = p
    return out


def audit(g, web, meta):
    """Which spiders does this edge decoration violate? Empty list = closed web."""
    bad = []
    for v in g.vertices():
        if g.type(v) == VertexType.BOUNDARY:
            continue
        legs = [tuple(sorted((v, w))) for w in g.neighbors(v)]
        ps = [web.get(e, 'I') for e in legs]
        z = [1 if p in 'ZY' else 0 for p in ps]
        x = [1 if p in 'XY' else 0 for p in ps]
        own, opp = (z, x) if g.type(v) == VertexType.Z else (x, z)
        half = (Fraction(g.phase(v)).denominator == 2)
        why = []
        if len(set(opp)) != 1:
            why.append(f'opposite colour on {sum(opp)}/{len(opp)} legs (needs all or none)')
        else:
            need = opp[0] if half else 0
            if sum(own) % 2 != need:
                why.append(f'own colour on {sum(own)}/{len(own)} legs (needs parity {need})')
        if why:
            bad.append((meta[v]['key'], meta[v]['role'], '; '.join(why)))
    return bad


def covers(g, V, web, kind, rd, idx):
    """Does `web` cover this check's ancilla on EVERY leg? That is what puts the
    check's outcome into the web (there is no measurement leg to decorate)."""
    key = ('bx', rd, idx) if kind == 'X' else ('az', rd, idx)
    want = ('X', 'Y') if kind == 'X' else ('Z', 'Y')
    v = V[key]
    legs = [tuple(sorted((v, w))) for w in g.neighbors(v)]
    return bool(legs) and all(web.get(e, 'I') in want for e in legs)


def detectors(g, V, rounds):
    """Closed webs with no support on the open output boundary. Returns
    (dimension, {(kind, round, index)} covered across a basis)."""
    A, edges, xv, zv, ncols = web_system(g)
    fixed = {}
    for i in range(D * D):
        e = tuple(sorted((V[('out', i)],
                          next(iter(g.neighbors(V[('out', i)]))))))
        fixed[xv[e]] = 0
        fixed[zv[e]] = 0
    res = solve_with(g, fixed)
    if res is None:
        return 0, set()
    _, basis, (edges, xv, zv) = res
    found = set()
    for bv in basis:
        w = vec_to_web(bv, edges, xv, zv)
        for rd in range(1, rounds + 1):
            for j in XCHECK:
                if covers(g, V, w, 'X', rd, j):
                    found.add(('X', rd, j))
            for k in ZCHECK:
                if covers(g, V, w, 'Z', rd, k):
                    found.add(('Z', rd, k))
    return len(basis), found


# --------------------------------------------------- the rule, from the tensors
I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)
Ym = 1j * Xm @ Zm
PAULI = {'I': I2, 'X': Xm, 'Y': Ym, 'Z': Zm}


def spider_tensor(colour, n, phase_frac):
    ph = np.exp(1j * np.pi * float(phase_frac))
    if colour == 'Z':
        v = np.zeros(2 ** n, dtype=complex)
        v[0] = 1.0
        v[-1] = ph
        return v
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    a, b = plus, minus
    for _ in range(n - 1):
        a, b = np.kron(a, plus), np.kron(b, minus)
    return a + ph * b


def brute_force_allowed(colour, n, phase_frac):
    """Pauli strings P with P|spider> = +-|spider> (stabiliser or anti-stabiliser)."""
    v = spider_tensor(colour, n, phase_frac)
    v = v / np.linalg.norm(v)
    ok = []
    for s in itertools.product('IXYZ', repeat=n):
        M = np.array([[1.0 + 0j]])
        for p in s:
            M = np.kron(M, PAULI[p])
        w = M @ v
        if np.allclose(w, v, atol=1e-9) or np.allclose(w, -v, atol=1e-9):
            ok.append(''.join(s))
    return ok


def rule_allowed(colour, n, phase_frac):
    """The same set, predicted by the linear rule the solver encodes."""
    own_is_z = (colour == 'Z')
    half = (Fraction(phase_frac).denominator == 2)
    out = []
    for s in itertools.product('IXYZ', repeat=n):
        z = [1 if p in 'ZY' else 0 for p in s]
        x = [1 if p in 'XY' else 0 for p in s]
        own, opp = (z, x) if own_is_z else (x, z)
        if len(set(opp)) != 1:
            continue
        if sum(own) % 2 != (opp[0] if half else 0):
            continue
        out.append(''.join(s))
    return out


def versions():
    import sys
    import platform
    return {'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'pyzx': getattr(zx, '__version__', 'unknown')}
