"""Web-driven search over |Y> injection initialisation patterns.

A pattern = (assign, site): assign maps each cell of a d x d rotated patch to
'+' or '0', and the injected qubit replaces the cell at `site`. For each
pattern the closed-web (GF(2)) machinery answers, exactly and basis-free:

  injects_Y / injects_any  transport feasibility: site input class -> logical
                           class at the open outputs (canonical reps pinned;
                           legitimate by the interface-compression lemma)
  det_dim                  dimension of the round-1 detector space (webs
                           terminating only on the initial states)
  n_indiv                  how many INDIVIDUAL check hubs are deterministic
                           (covered-hub set exactly {j})
  site_covered             does any detector terminate on the injected leg
                           (the blind-spot test)
  undet                    initialisation-leg faults (edge, P) that commute
                           with every detector -- undetected at this level

Closure at the degree-1 init spiders already enforces the termination classes
(|+> leg carries I/X, |0> I/Z, |Y> I/Y), so no pinning is needed there.
"""
import sys, os
from fractions import Fraction
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'vendor'))

import numpy as np
import pyzx as zx
from pyzx import VertexType
from physical import SpacetimeBuilder, rect
from injection_webs import web_system
import gf2

B = VertexType.BOUNDARY


class Study:
    """Packed closed-web solver + lambda (hub-coverage) columns."""
    def __init__(self, g):
        A, self.edges, _, _, self.ncols = web_system(g)
        self.Ap = gf2.pack(A, self.ncols)
        self.ei = {e: k for k, e in enumerate(self.edges)}
        spiders = [v for v in g.vertices() if g.type(v) != B]
        nE = len(self.edges)
        self.lam = {v: 2 * nE + k for k, v in enumerate(spiders)}

    def cols(self, e):
        return 2 * self.ei[e], 2 * self.ei[e] + 1

    def solve(self, fixed):
        return gf2.solve_affine_packed(self.Ap, self.ncols, fixed)

    def pin_edge(self, f, e, P):
        xc, zc = self.cols(e)
        f[xc] = 1 if P in ('X', 'Y') else 0
        f[zc] = 1 if P in ('Z', 'Y') else 0


def rep_pattern(P, d):
    """Canonical logical representative on the open outputs of a d x d patch."""
    out = {}
    if P in ('Z', 'Y'):
        for i in range(d):
            out[(i, 0)] = 'Z'
    if P in ('X', 'Y'):
        for j in range(d):
            c = (0, j)
            out[c] = 'Y' if out.get(c) == 'Z' else 'X'
    return out


def build(assign, site, d, site_mode):
    """One round of checks. site_mode: 'open' (transport tests) or 'Y'."""
    b = SpacetimeBuilder()
    P = rect(0, 0, d, d)
    if site_mode == 'open':
        b.open_inputs([site])
        b.init_cells({c: assign[c] for c in P if c != site})
    else:
        kinds = dict(assign)
        kinds[site] = 'Y'
        b.init_cells(kinds)
    b.round(P, 'r')
    b.open_outputs(P)
    return b.finish()


def edge_of(g, meta, role, cell=None):
    for v in g.vertices():
        if g.type(v) == B and meta[v]['role'] == role and \
           (cell is None or meta[v]['cell'] == cell):
            yield meta[v]['cell'], tuple(sorted((v, next(iter(g.neighbors(v))))))


def transports(assign, site, d):
    g, meta = build(assign, site, d, 'open')
    W = Study(g)
    outs = dict(edge_of(g, meta, 'out'))
    site_e = next(e for c, e in edge_of(g, meta, 'in') if c == site)
    def ok(P):
        f = {}
        W.pin_edge(f, site_e, P)
        pat = rep_pattern(P, d)
        for c, e in outs.items():
            W.pin_edge(f, e, pat.get(c, 'I'))
        return W.solve(f) is not None
    y = ok('Y')
    return y, (y and ok('X') and ok('Z'))


def detector_metrics(assign, site, d):
    g, meta = build(assign, site, d, 'Y')
    W = Study(g)
    outs = dict(edge_of(g, meta, 'out'))
    f0 = {}
    for e in outs.values():
        W.pin_edge(f0, e, 'I')
    sol = W.solve(f0)
    if sol is None:
        return dict(det_dim=0, n_indiv=0, site_covered=False, undet=None)
    part, basis = sol
    det_dim = len(basis)

    hubs = [v for v in g.vertices()
            if meta[v]['role'].startswith('chk_')]
    n_indiv = 0
    indiv = []
    for j in hubs:
        f = dict(f0)
        for k in hubs:
            f[W.lam[k]] = 1 if k == j else 0
        if W.solve(f) is not None:
            n_indiv += 1
            indiv.append(meta[j]['role'])

    # blind-spot test: a detector terminating on the injected leg
    site_v = next(v for v in g.vertices()
                  if meta[v]['role'] == 'init_Y' and meta[v]['cell'] == site)
    site_e = tuple(sorted((site_v, next(iter(g.neighbors(site_v))))))
    f = dict(f0)
    xc, zc = W.cols(site_e)
    f[xc] = 1                                   # green pi/2 forces z = x: Y
    site_covered = W.solve(f) is not None

    # undetected init faults: decorations realised by the detector space on
    # each init leg; P is detected iff it anticommutes with one of them
    init_edges = {}
    for v in g.vertices():
        if meta[v]['role'].startswith('init_'):
            e = tuple(sorted((v, next(iter(g.neighbors(v))))))
            init_edges[meta[v]['cell']] = e
    vecs = [part] if np.any(part) else []
    vecs += list(basis)
    seen = {}
    for vec in vecs:
        for c, e in init_edges.items():
            xc, zc = W.cols(e)
            xb = int(vec[xc // 64] >> np.uint64(xc % 64)) & 1
            zb = int(vec[zc // 64] >> np.uint64(zc % 64)) & 1
            seen.setdefault(c, set()).add((xb, zb))
    SYMP = {'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
    undet = []
    for c in init_edges:
        got = seen.get(c, {(0, 0)})
        for P, (px, pz) in SYMP.items():
            if not any((px * wz + pz * wx) % 2 for wx, wz in got):
                undet.append((c, P))
    return dict(det_dim=det_dim, n_indiv=n_indiv, site_covered=site_covered,
                undet=undet, indiv=indiv)


def lao_criger(d):
    from physical import injection_kind
    site = (0, d - 1)
    assign = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
    assign[site] = '+'          # placeholder; site replaces it
    return assign, site


def staircase_family(d):
    """All monotone non-increasing boundaries f: col -> 0..d,
    assign(c,r) = '+' if r < f(c) else '0'; sites = cells with an
    opposite-kind orthogonal neighbour."""
    def gen(prev, col, f):
        if col == d:
            yield tuple(f)
            return
        for v in range(prev, -1, -1):
            yield from gen(v, col + 1, f + [v])
    for f in gen(d, 0, []):
        assign = {(c, r): ('+' if r < f[c] else '0')
                  for c in range(d) for r in range(d)}
        sites = []
        for (c, r), k in assign.items():
            for (dc, dr) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (c + dc, r + dr)
                if q in assign and assign[q] != k:
                    sites.append((c, r))
                    break
        yield f, assign, sites


TRIVIAL = {'init_+': 'X', 'init_0': 'Z', 'init_Y': 'Y'}


def fault_profile(assign, site, d):
    """Undetected NONTRIVIAL init faults, split benign/malicious. Malicious =
    anticommutes with the output-Y correlator (well-defined: the fault commutes
    with every detector, so the pairing is representative-independent)."""
    g, meta = build(assign, site, d, 'Y')
    W = Study(g)
    outs = dict(edge_of(g, meta, 'out'))
    f0 = {}
    for e in outs.values():
        W.pin_edge(f0, e, 'I')
    part, basis = W.solve(f0)
    # correlator: same graph, outputs pinned to rep(Y)
    fc = {}
    pat = rep_pattern('Y', d)
    for c, e in outs.items():
        W.pin_edge(fc, e, pat.get(c, 'I'))
    corr = W.solve(fc)
    assert corr is not None
    corr = corr[0]

    def bits(vec, e):
        xc, zc = W.cols(e)
        return (int(vec[xc // 64] >> np.uint64(xc % 64)) & 1,
                int(vec[zc // 64] >> np.uint64(zc % 64)) & 1)

    init_edges, kinds = {}, {}
    for v in g.vertices():
        r = meta[v]['role']
        if r.startswith('init_'):
            init_edges[meta[v]['cell']] = tuple(sorted((v, next(iter(g.neighbors(v))))))
            kinds[meta[v]['cell']] = r
    vecs = ([part] if np.any(part) else []) + list(basis)
    SYMP = {'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
    benign, malicious = [], []
    for c, e in init_edges.items():
        got = {bits(v, e) for v in vecs} | {(0, 0)}
        cw = bits(corr, e)
        for P, (px, pz) in SYMP.items():
            if P == TRIVIAL[kinds[c]]:
                continue
            if any((px * wz + pz * wx) % 2 for wx, wz in got):
                continue                          # detected
            if (px * cw[1] + pz * cw[0]) % 2:
                malicious.append((c, P))
            else:
                benign.append((c, P))
    return benign, malicious
