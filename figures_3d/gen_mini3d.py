"""3D spacetime ZX diagrams of the mini injection protocol (init, one
optimised round, deflation readout) at d = 3, 5, 7, 9 -- the BEFORE
pictures of the parameterised reduction. The five noise-channel
locations (site init edge, outgoing edges of the four site-touching
CNOTs) are marked in violet. The AFTER picture is the three-spider
remnant, identical at every d."""
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
                   'arXiv-2501.15566v5_draft_post_LC_read')
NODE = {3: '0.4cm', 5: '0.3cm', 7: '0.22cm', 9: '0.17cm'}

for d in (3, 5, 7, 9):
    kinds = spiral_kinds(0, 0, d)
    site = next(c for c, k in kinds.items() if k == 'Y')
    b = GateBuilder()
    b.init_cells(kinds)
    P = rect(0, 0, d, d)
    xs, zs = region_checks(P)
    schedules = ([('X', pos, opt_sched(pos, sup, 'X')) for pos, sup in xs] +
                 [('Z', pos, opt_sched(pos, sup, 'Z')) for pos, sup in zs])
    schedules.sort(key=lambda t: (t[1][0], t[1][1]))
    b.gate_round(P, 'r', schedules=[(t, p, list(s)) for t, p, s in schedules])
    # deflation caps: X-basis (green) along the site column, Z (red) else
    col = {(site[0], r) for r in range(d)}
    for c in P:
        if tuple(c) == site:
            continue
        ty = GREEN if tuple(c) in col else RED
        cap = b.V(ty, c[0], c[1], 0, 'defl', c)
        b.g.add_edge((b.frontier[c], cap))
        del b.frontier[c]
    b.open_outputs([site])
    g, meta = b.finish()
    # violet noise locations: site init edge + out-edges of site CNOTs
    def out_edge(v):
        key = (g.vdata(v, 'z', 0.0), v)
        cands = [w for w in g.neighbors(v)
                 if (g.vdata(w, 'z', 0.0), w) > key
                 and meta[w].get('cell') == meta[v].get('cell')]
        if not cands: return None
        return tuple(sorted((v, min(cands,
                    key=lambda w: (g.vdata(w, 'z', 0.0), w)))))
    marks = {}
    for v in g.vertices():
        m = meta[v]
        if m['role'].startswith('init') and tuple(m['cell']) == site:
            e = out_edge(v)
            if e: marks[e] = 'M'
        if m['role'].startswith('dtap') and tuple(m['cell']) == site:
            e = out_edge(v)
            if e: marks[e] = 'M'
            for w in g.neighbors(v):
                if meta[w]['role'].startswith(('actrl', 'atgt')):
                    e2 = out_edge(w)
                    if e2: marks[e2] = 'M'
    reps = {}
    for v in g.vertices():
        r = meta[v]['role']
        if r.startswith('ancinit_'):
            reps.setdefault('r', v)
    emit(g, [marks], os.path.join(OUT, f'fig_mini3d_d{d}.tex'),
         zscale=2.4, node_size=NODE[d], planes=list(reps.values()))
    print(f'd={d}: {g.num_vertices()} spiders, {len(marks)} violet marks')
