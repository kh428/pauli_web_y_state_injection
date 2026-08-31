"""d=5 web studies for appendices B-F: ZZ injection grown 2->5, hook grown
2->5, transversal (open inputs), unitary cone, deformation re-injection."""
import sys, os
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vendor'))
import numpy as np
from pyzx import VertexType
from schemes import Builder, rect, decode
from patterns import Study, rep_pattern, edge_of

GREEN, RED, B = VertexType.Z, VertexType.X, VertexType.BOUNDARY
SYMP = {'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
TRIV = {'init_+': 'X', 'init_0': 'Z', 'init_Y': 'Y'}
D = 5


def growth_kinds():
    """new cells for 2->5 growth of a SW-corner seed: columns 0-1 extended
    north in |+>, columns 2-4 in |0>."""
    k = {}
    for c in range(D):
        for r in range(D):
            if c < 2 and r < 2:
                continue                      # seed cells
            k[(c, r)] = '+' if c < 2 else '0'
    return k


def gadget(b, cells, tap_ty, col_ty, tag, pos=None, leaf_pos=None):
    """phase gadget e^{+-i pi/4 P P} on two frontier wires: taps of tap_ty,
    collector of col_ty, leaf of tap_ty with phase pi/2. pos places the
    collector (the leaf sits beyond it), so the gadget can stand clear of
    the patch for legible figures."""
    taps = []
    for cell in cells:
        t = b.V(tap_ty, *cell, 0, f'gad_t{tag}', cell)
        b.g.add_edge((b.frontier[cell], t))
        b.frontier[cell] = t
        taps.append(t)
    if pos is None:
        pos = (sum(c[0] for c in cells) / 2 + 0.4,
               sum(c[1] for c in cells) / 2 + 0.4)
    if leaf_pos is None:
        leaf_pos = (pos[0] - 0.7, pos[1] + 0.5)
    col = b.V(col_ty, pos[0], pos[1], 0, f'gad_c{tag}', None)
    leaf = b.V(tap_ty, leaf_pos[0], leaf_pos[1], Fraction(1, 2), f'gad_l{tag}', None)
    for t in taps:
        b.g.add_edge((t, col))
    b.g.add_edge((col, leaf))
    b.t += 1


def build_zz():
    b = Builder()
    seed = rect(0, 0, 2, 2)
    b.init_cells({c: '+' for c in seed})
    b.round(seed, 's')
    gadget(b, [(0, 0), (1, 0)], GREEN, RED, 'zz',
           pos=(0.5, -1.8), leaf_pos=(0.5, -2.9))  # Zbar rep = ROW pair here
    b.init_cells(growth_kinds())
    P = rect(0, 0, D, D)
    b.round(P, 'g'); b.round(P, 'h')
    b.open_outputs(P)
    return b.finish()


def build_hook():
    b = Builder()
    seed = rect(0, 0, 2, 2)
    b.init_cells({c: '0' for c in seed})
    b.round(seed, 's')
    g, meta = b.g, b.meta
    # attach XX gadget to the X-layer taps of the seed round on the col pair
    taps = {meta[v]['cell']: v for v in g.vertices()
            if meta[v]['role'] == 'data_xs'}
    t1, t2 = taps[(0, 0)], taps[(0, 1)]
    zc = b.t
    col = b.V(GREEN, -1.1, 0.5, 0, 'gad_chk', None)
    leaf = b.V(RED, -1.8, 1.0, Fraction(1, 2), 'gad_lhk', None)
    g.add_edge((t1, col)); g.add_edge((t2, col)); g.add_edge((col, leaf))
    b.init_cells(growth_kinds())
    P = rect(0, 0, D, D)
    b.round(P, 'g'); b.round(P, 'h')
    b.open_outputs(P)
    return b.finish()


def study(g, meta, name, census_roles=('init_', 'gad_')):
    W = Study(g)
    outs = dict(edge_of(g, meta, 'out'))
    f0 = {}
    for e in outs.values():
        W.pin_edge(f0, e, 'I')
    sol = W.solve(f0)
    part, basis = sol
    vecs = ([part] if np.any(part) else []) + list(basis)
    fc = {}
    pat = rep_pattern('Y', D)
    for c, e in outs.items():
        W.pin_edge(fc, e, pat.get(c, 'I'))
    corr = W.solve(fc)
    produces = corr is not None
    mal = ben = 0
    mal_list = []
    if produces:
        cv = corr[0]
        def bits(vec, e):
            xc, zc = W.cols(e)
            return (int(vec[xc//64] >> np.uint64(xc%64)) & 1,
                    int(vec[zc//64] >> np.uint64(zc%64)) & 1)
        seen = set()
        for v in g.vertices():
            role = meta[v]['role']
            if not any(role.startswith(p) for p in census_roles):
                continue
            for w_ in g.neighbors(v):
                e = tuple(sorted((v, w_)))
                if e in seen: continue
                seen.add(e)
                got = {bits(x, e) for x in vecs} | {(0, 0)}
                cw = bits(cv, e)
                for P_, (px, pz) in SYMP.items():
                    if role.startswith('init_') and P_ == TRIV.get(role, ''):
                        continue
                    if any((px*wz + pz*wx) % 2 for wx, wz in got): continue
                    if (px*cw[1] + pz*cw[0]) % 2:
                        mal += 1
                        mal_list.append((meta[v]['cell'], role, P_))
                    else:
                        ben += 1
    print(f'{name:12s} verts={g.num_vertices():4d} det={len(basis):3d} '
          f'produces_Y={produces} mal={mal} ben={ben}')
    if mal_list:
        print('   ', mal_list[:8])
    return W, vecs, (corr[0] if produces else None)


def build_transversal(rounds=2):
    b = Builder()
    P = rect(0, 0, D, D)
    b.open_inputs(P)
    for r in range(rounds):
        b.round(P, f'r{r}')
    b.open_outputs(P)
    return b.finish()


def build_deformation():
    b = Builder()
    P = rect(0, 0, D, D)
    q = (2, 2)
    b.open_inputs(P)
    b.round(P, 'a')
    b.measure_cells([q], 'X')
    b.init_cells({q: 'Y'})
    b.round(P, 'b'); b.round(P, 'c')
    b.open_outputs(P)
    return b.finish()


if __name__ == '__main__':
    g, meta = build_zz()
    study(g, meta, 'ZZ 2->5')
    g, meta = build_hook()
    study(g, meta, 'hook 2->5')

    # transversal: dets with 1 and 2 rounds (open inputs = non-stabiliser)
    for rd in (1, 2):
        g, meta = build_transversal(rd)
        W = Study(g)
        outs = dict(edge_of(g, meta, 'out'))
        ins = dict(edge_of(g, meta, 'in'))
        f0 = {}
        for e in outs.values(): W.pin_edge(f0, e, 'I')
        for e in ins.values(): W.pin_edge(f0, e, 'I')
        part, basis = W.solve(f0)
        print(f'transversal {rd} round(s): det_dim = {len(basis)}')

    # deformation: dets + blind spot + the two adjacent Z-plaquettes
    g, meta = build_deformation()
    W = Study(g)
    outs = dict(edge_of(g, meta, 'out'))
    ins = dict(edge_of(g, meta, 'in'))
    f0 = {}
    for e in outs.values(): W.pin_edge(f0, e, 'I')
    for e in ins.values(): W.pin_edge(f0, e, 'I')
    part, basis = W.solve(f0)
    yv = next(v for v in g.vertices() if meta[v]['role'] == 'init_Y')
    ye = tuple(sorted((yv, next(iter(g.neighbors(yv))))))
    f = dict(f0); f[W.cols(ye)[0]] = 1
    cov = W.solve(f) is not None
    print(f'deformation: verts={g.num_vertices()} det_dim={len(basis)} '
          f're-injected qubit covered by a detector: {cov}')
