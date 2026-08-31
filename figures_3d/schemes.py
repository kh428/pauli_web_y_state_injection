"""New |Y>-injection schemes designed directly as spacetime ZX-diagrams, web-surveyed.

Schemes (d=3, one post-selected check round, open outputs):
  lc          Lao-Criger corner product init (baseline)
  precone_*   LC pattern + CNOT layer(s) entangling the site before round 1
  unitary     unitary spreading: X-repetition down the west column and
              Z-repetition along the bottom row from a corner input, bulk in
              |0>/|+>, then one verification round (unitary injection in ZX)
  hub_*       hook-style: all data |0>, one X-check HUB carries pi/2 -- the
              state enters through the measurement ancilla (ZX analogue of
              Gidney's hook injection; XY-plane states only, as there)

Metrics per scheme: det_dim (= post-selected bits), n_indiv, produces |Ybar>
(correlator web feasible), undetected-malicious prep faults (edge-level census
over init legs + cone edges; init legs filtered by their trivial stabiliser).
"""
import sys, os
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'vendor'))

import numpy as np
import pyzx as zx
from pyzx import VertexType
from physical import SpacetimeBuilder, rect, injection_kind, GREEN, RED
from patterns import Study, rep_pattern, edge_of

B = VertexType.BOUNDARY
SYMP = {'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
TRIVIAL = {'init_+': 'X', 'init_0': 'Z', 'init_Y': 'Y'}


class Builder(SpacetimeBuilder):
    def cnot_layer(self, pairs, tag=''):
        """pairs: [(ctrl_cell, tgt_cell)]; green on ctrl wire, red on tgt."""
        done = set()
        for c, t in pairs:
            gv = self.V(VertexType.Z, *c, 0, f'cnot_c{tag}', c)
            rv = self.V(VertexType.X, *t, 0, f'cnot_t{tag}', t)
            self.g.add_edge((self.frontier[c], gv))
            self.g.add_edge((self.frontier[t], rv))
            self.g.add_edge((gv, rv))
            self.frontier[c], self.frontier[t] = gv, rv
            done.update((c, t))
        self.t += 1


def bits(vec, W, e):
    xc, zc = W.cols(e)
    return (int(vec[xc // 64] >> np.uint64(xc % 64)) & 1,
            int(vec[zc // 64] >> np.uint64(zc % 64)) & 1)


def analyze(g, meta, d):
    W = Study(g)
    outs = dict(edge_of(g, meta, 'out'))
    f0 = {}
    for e in outs.values():
        W.pin_edge(f0, e, 'I')
    sol = W.solve(f0)
    det_dim, vecs = 0, []
    if sol is not None:
        part, basis = sol
        det_dim = len(basis)
        vecs = ([part] if np.any(part) else []) + list(basis)

    hubs = [v for v in g.vertices() if meta[v]['role'].startswith('chk_')]
    n_indiv = 0
    for j in hubs:
        f = dict(f0)
        for k in hubs:
            f[W.lam[k]] = 1 if k == j else 0
        if W.solve(f) is not None:
            n_indiv += 1

    fc = {}
    pat = rep_pattern('Y', d)
    for c, e in outs.items():
        W.pin_edge(fc, e, pat.get(c, 'I'))
    corr = W.solve(fc)
    produces = corr is not None

    ben = mal = 0
    mal_list = []
    if produces:
        cv = corr[0]
        prep = []
        for v in g.vertices():
            role = meta[v]['role']
            if role.startswith('init_') or role.startswith('cnot') \
               or role in ('in', 'hook'):
                for w in g.neighbors(v):
                    e = tuple(sorted((v, w)))
                    prep.append((e, role if role.startswith('init_') else None))
        seen_e = set()
        for e, initrole in prep:
            if e in seen_e:
                continue
            seen_e.add(e)
            got = {bits(v, W, e) for v in vecs} | {(0, 0)}
            cw = bits(cv, W, e)
            for P, (px, pz) in SYMP.items():
                if initrole and P == TRIVIAL.get(initrole, ''):
                    continue
                if any((px * wz + pz * wx) % 2 for wx, wz in got):
                    continue
                if (px * cw[1] + pz * cw[0]) % 2:
                    mal += 1
                    mal_list.append((meta[e[0]]['cell'], P))
                else:
                    ben += 1
    corr_web = decode(corr[0], W) if produces else {}
    det_webs = [decode(v, W) for v in vecs]
    return dict(det_dim=det_dim, n_indiv=n_indiv, produces=produces,
                mal=mal, ben=ben, mal_list=mal_list,
                det_webs=det_webs, corr_web=corr_web)


def decode(vec, W):
    out = {}
    for e in W.edges:
        xb, zb = bits(vec, W, e)
        p = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}[(xb, zb)]
        if p != 'I':
            out[e] = p
    return out


# ------------------------------------------------------------------ schemes
def sc_lc(d=3):
    b = Builder()
    P = rect(0, 0, d, d)
    b.init_cells({c: injection_kind(c, 0, 0, d) for c in P})
    b.round(P, 'r')
    b.open_outputs(P)
    return b.finish()


def sc_precone(layers, d=3):
    b = Builder()
    P = rect(0, 0, d, d)
    b.init_cells({c: injection_kind(c, 0, 0, d) for c in P})
    for i, lay in enumerate(layers):
        b.cnot_layer(lay, str(i))
    b.round(P, 'r')
    b.open_outputs(P)
    return b.finish()


def sc_unitary(d=3):
    """Corner input at (0,0); X-repetition down west column (CNOT chain,
    input as control), Z-repetition along bottom row (input as target of
    |+> controls), bulk |0>; one verification round."""
    b = Builder()
    P = rect(0, 0, d, d)
    kinds = {}
    for c in P:
        if c == (0, 0):
            kinds[c] = 'Y'
        elif c[0] == 0:
            kinds[c] = '0'          # west column: X-repetition targets
        elif c[1] == 0:
            kinds[c] = '+'          # bottom row: Z-repetition controls
        else:
            kinds[c] = '0'
    b.init_cells(kinds)
    for i in range(d - 1):
        b.cnot_layer([((0, i), (0, i + 1)), ((i + 1, 0), (i, 0))], str(i))
    b.round(P, 'r')
    b.open_outputs(P)
    return b.finish()


def sc_hub(hub_pos, d=3):
    b = Builder()
    P = rect(0, 0, d, d)
    b.init_cells({c: '0' for c in P})
    b.round(P, 'r')
    b.open_outputs(P)
    g, meta = b.finish()
    tgt = None
    for v in g.vertices():
        if meta[v]['role'] == 'chk_xr' and \
           (g.row(v), g.qubit(v)) == hub_pos:
            tgt = v
    assert tgt is not None, f'no X hub at {hub_pos}'
    g.set_phase(tgt, Fraction(1, 2))
    return g, meta


def x_hub_positions(d=3):
    from physical import region_checks
    xs, _ = region_checks(rect(0, 0, d, d))
    return [tuple(map(float, pos)) for pos, _ in xs]


def sc_hook(pair, d=3, pre=0, post=0):
    """All data |0>; insert e^{i pi/4 X_c1 X_c2} at the round-1 X layer via a
    phase gadget (red taps -> green collector -> red pi/2 leaf): the ZX-level
    hook -- a rotation about HALF an X-check's support, i.e. the partial
    parity present on the ancilla mid-ladder."""
    b = Builder()
    P = rect(0, 0, d, d)
    b.init_cells({c: '0' for c in P})
    for r in range(pre):
        b.round(P, f'p{r}')
    b.round(P, 'r')
    for r in range(post):
        b.round(P, f'q{r}')
    b.open_outputs(P)
    g, meta = b.finish()
    taps = {}
    for v in g.vertices():
        if meta[v]['role'] == 'data_xr':
            taps[meta[v]['cell']] = v
    t1, t2 = taps[pair[0]], taps[pair[1]]
    zc = g.vdata(t1, 'z', 0.0)
    col = g.add_vertex(VertexType.Z, qubit=(g.qubit(t1)+g.qubit(t2))/2+0.35,
                       row=(g.row(t1)+g.row(t2))/2+0.35)
    leaf = g.add_vertex(VertexType.X, qubit=g.qubit(col)+0.3,
                        row=g.row(col)+0.3, phase=Fraction(1, 2))
    for v in (col, leaf):
        g.set_vdata(v, 'z', zc)
        meta[v] = dict(cell=None, role='hook', t=zc)
    g.add_edge((t1, col)); g.add_edge((t2, col)); g.add_edge((col, leaf))
    return g, meta


def hook_pairs(d=3):
    from physical import region_checks
    xs, _ = region_checks(rect(0, 0, d, d))
    from itertools import combinations
    for pos, sup in xs:
        for pr in combinations(sup, 2):
            yield pos, pr


def sc_2copy():
    """2-copy Y-checked injection, logical level: |Y> -- S -- tap -- Sdag -- out
    per copy, green hub joining the red taps = post-selected YbarYbar parity
    (S-sandwich turns XX into YY). Detects EVERY single injection-leg fault."""
    g = zx.Graph()
    meta = {}
    def V(ty, q, z, ph=0, role=''):
        v = g.add_vertex(ty, qubit=q, row=z, phase=ph)
        g.set_vdata(v, 'z', float(z))
        meta[v] = dict(cell=(q,), role=role, t=z)
        return v
    taps = []
    for q in (0, 1):
        i = V(VertexType.Z, q, 0, Fraction(1, 2), 'init_Y')
        s1 = V(VertexType.Z, q, 1, Fraction(1, 2), 'S')
        t = V(VertexType.X, q, 2, 0, 'tap')
        s2 = V(VertexType.Z, q, 3, Fraction(3, 2), 'Sdag')
        o = V(VertexType.BOUNDARY, q, 4, 0, 'out')
        for a, b in ((i, s1), (s1, t), (t, s2), (s2, o)):
            g.add_edge((a, b))
        taps.append(t)
    hub = V(VertexType.Z, 0.5, 2.3, 0, 'hub')
    g.add_edge((taps[0], hub)); g.add_edge((taps[1], hub))
    return g, meta


def analyze_2copy(g, meta):
    W = Study(g)
    outs = [tuple(sorted((v, next(iter(g.neighbors(v)))))) for v in g.vertices()
            if meta[v]['role'] == 'out']
    f0 = {}
    for e in outs:
        W.pin_edge(f0, e, 'I')
    part, basis = W.solve(f0)
    vecs = ([part] if np.any(part) else []) + list(basis)
    fc = {}
    W.pin_edge(fc, outs[0], 'Y'); W.pin_edge(fc, outs[1], 'I')
    corr = W.solve(fc)
    return dict(det_dim=len(basis), n_indiv=1, produces=corr is not None,
                mal=0, ben=0, mal_list=[],
                det_webs=[decode(v, W) for v in vecs],
                corr_web=decode(corr[0], W) if corr is not None else {})


class Builder2(Builder):
    def s_layer(self, phases, tag=''):
        """phases: {cell: Fraction} -- single-qubit Z-rotations on wires."""
        for c, ph in phases.items():
            v = self.V(VertexType.Z, *c, ph, f's{tag}', c)
            self.g.add_edge((self.frontier[c], v))
            self.frontier[c] = v
        self.t += 1

    def cz_layer(self, pairs, tag=''):
        """CZ between cell pairs: green taps joined by H = Z(pi/2)X(pi/2)Z(pi/2)
        chain (keeps the diagram in plain edges for the web solver)."""
        H = Fraction(1, 2)
        for a, b in pairs:
            ta = self.V(VertexType.Z, *a, 0, f'cz{tag}', a)
            tb = self.V(VertexType.Z, *b, 0, f'cz{tag}', b)
            self.g.add_edge((self.frontier[a], ta))
            self.g.add_edge((self.frontier[b], tb))
            self.frontier[a], self.frontier[b] = ta, tb
            mx = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            h1 = self.V(VertexType.Z, mx[0] + 0.2, mx[1], H, f'h{tag}', None)
            h2 = self.V(VertexType.X, mx[0] + 0.35, mx[1], H, f'h{tag}', None)
            h3 = self.V(VertexType.Z, mx[0] + 0.5, mx[1], H, f'h{tag}', None)
            self.g.add_edge((ta, h1)); self.g.add_edge((h1, h2))
            self.g.add_edge((h2, h3)); self.g.add_edge((h3, tb))
        self.t += 1

    def fold_s(self, x0, y0, d, dagger=False, tag='f'):
        """Fold-transversal S on a d x d patch at (x0,y0): S on the fold
        diagonal, CZ across mirror pairs."""
        ph = Fraction(3, 2) if dagger else Fraction(1, 2)
        diag = {}
        pairs = []
        for i in range(d):
            for j in range(i, d):
                a, b = (x0 + i, y0 + j), (x0 + j, y0 + i)
                if a == b:
                    diag[a] = ph
                else:
                    pairs.append((a, b))
        self.s_layer(diag, tag)
        self.cz_layer(pairs, tag)


def spiral_kinds(x0, y0, d):
    site = (x0 + d // 2, y0 + d // 2)
    plus = set()
    for k in range(1, (d - 1) // 2 + 1):
        for c in range(site[0] - k, site[0] + k):
            plus.add((c, site[1] + k))
    full = set(plus) | {(2*site[0]-c, 2*site[1]-r) for c, r in plus}
    out = {}
    for i in range(d):
        for j in range(d):
            c = (x0 + i, y0 + j)
            out[c] = 'Y' if c == site else ('+' if c in full else '0')
    return out


def sc_2copy_physical(d=3, r_merge=1, seed='lc'):
    """Physical 2-copy Y-checked injection: two injected patches (seed =
    'lc' corner pattern or 'spiral' center pattern), fold-S on both, rough
    merge (XbarXbar) across the seam, fold-S-dagger, final round.
    Sandwich identity: (S x S) M_XX (S x S)^dag = M_YY."""
    b = Builder2()
    P1 = rect(0, 0, d, d)
    P2 = rect(d + 1, 0, d, d)
    seam = [(d, j) for j in range(d)]
    if seed == 'spiral':
        kinds = spiral_kinds(0, 0, d)
        kinds.update(spiral_kinds(d + 1, 0, d))
    else:
        kinds = {c: injection_kind(c, 0, 0, d) for c in P1}
        kinds.update({c: injection_kind(c, d + 1, 0, d) for c in P2})
    b.init_cells(kinds)
    b.round(P1 + P2, 'a')
    b.fold_s(0, 0, d, tag='f1')
    b.fold_s(d + 1, 0, d, tag='f2')
    b.init_cells({c: '0' for c in seam})
    for r in range(r_merge):
        b.round(P1 + seam + P2, f'm{r}')
    b.measure_cells(seam, 'Z')
    b.fold_s(0, 0, d, dagger=True, tag='g1')
    b.fold_s(d + 1, 0, d, dagger=True, tag='g2')
    b.round(P1 + P2, 'b')
    b.open_outputs(P1 + P2)
    return b.finish()


def sc_spiral(d):
    """The spiral pinwheel, unified rule (loop iterations 2-4, 13): site at
    center; row center+k carries |+> on exactly the 2k cells
    [center-k, center+k-1] for k = 1..(d-1)/2; C2-rotated below; all other
    cells |0>. Verified OPTIMAL at d=3,5,7,9: malicious = {site X, site Z}
    and det_dim = (d^2-1)/2 (half of all checks deterministic)."""
    site = (d // 2, d // 2)
    plus = set()
    for k in range(1, (d - 1) // 2 + 1):
        for c in range(site[0] - k, site[0] + k):
            plus.add((c, site[1] + k))
    full = set(plus) | {(2 * site[0] - c, 2 * site[1] - r) for c, r in plus}
    P = rect(0, 0, d, d)
    b = Builder()
    kinds = {c: ('+' if c in full else '0') for c in P if c != site}
    kinds[site] = 'Y'
    b.init_cells(kinds)
    b.round(P, 'r')
    b.open_outputs(P)
    return b.finish()
