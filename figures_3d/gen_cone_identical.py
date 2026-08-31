"""Pictorial distance-independence: the mini protocol at d=5 and d=9
with every edge inside the fault cone (Chebyshev radius 1.5 of the
injected qubit) marked violet. The violet sub-circuit is identical
in the two panels; that is the whole argument."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyzx import VertexType
from gates import GateBuilder
from physical import rect, region_checks
from schemes import spiral_kinds
from paper3d import emit

GREEN, RED = VertexType.Z, VertexType.X
def opt_sched(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NW, NE, SE, SW] if typ == 'X' else [NE, SE, NW, SW])
    ss = set(sup)
    return [c if c in ss else None for c in order]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'latex_pauli_web_y_state_injection', 'version_5_v3')
NODE = {5: '0.3cm', 9: '0.17cm'}
for d in (5, 9):
    kinds = spiral_kinds(0, 0, d)
    site = next(c for c, k in kinds.items() if k == 'Y')
    b = GateBuilder()
    b.init_cells(kinds)
    P = rect(0, 0, d, d)
    xs, zs = region_checks(P)
    schedules = ([('X', pos, opt_sched(pos, sup, 'X'))
                  for pos, sup in xs] +
                 [('Z', pos, opt_sched(pos, sup, 'Z'))
                  for pos, sup in zs])
    schedules.sort(key=lambda t: (t[1][0], t[1][1]))
    b.gate_round(P, 'r', schedules=[(t, p, list(s))
                                    for t, p, s in schedules])
    col = {(site[0], r) for r in range(d)}
    for c in P:
        if tuple(c) == site: continue
        ty = GREEN if tuple(c) in col else RED
        cap = b.V(ty, c[0], c[1], 0, 'defl', c)
        b.g.add_edge((b.frontier[c], cap))
        del b.frontier[c]
    b.open_outputs([site])
    g, meta = b.finish()
    def coords(cell):
        if len(cell) == 3 and cell[1] in ('X', 'Z'):
            return cell[2]
        try:
            return (float(cell[0]), float(cell[1]))
        except (TypeError, ValueError):
            return None
    def cheb(cell):
        c = coords(cell)
        if c is None: return 99
        return max(abs(c[0] - site[0]), abs(c[1] - site[1]))
    marks = {}
    for e in g.edges():
        u, v = g.edge_st(e)
        cu = meta[u].get('cell'); cv = meta[v].get('cell')
        if cu is None or cv is None: continue
        if cheb(cu) <= 1.5 and cheb(cv) <= 1.5:
            marks[tuple(sorted((u, v)))] = 'M'
    emit(g, [marks], os.path.join(OUT, f'fig_cone_d{d}.tex'),
         zscale=2.4, node_size=NODE[d], planes=[])
    print(f'd={d}: {g.num_vertices()} spiders, {len(marks)} cone edges')
