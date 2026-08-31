"""Generic paper-style 3D tikz emitter for spacetime builder graphs:
tdplot_main_coords family conventions (zx_green/zx_red 1cm nodes, 0.05cm
black wires, 0.25cm webs at opacity 0.8, +-0.2 offsets when X and Z share
an edge). z levels compacted via viewer.compact_z, spacing 4."""
import sys, os
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyzx import VertexType

GREEN = r'\node[fill=zx_green,shape=circle,draw=black]'
BLUE = r'\node[fill=zx_blue,shape=circle,draw=black]'
RED = r'\node[fill=zx_red,shape=circle,draw=black]'
NONE = r'\node[fill=none,shape=circle,draw=none]'
WIRE = r'\draw[black,fill=black,opacity=1,line width = 0.05cm]'
HEDGE = (r'\draw[-, dashed, dash pattern=on 2pt off 1.2pt, '
         r'line width=0.06cm, draw=zx_hadamard]')
WEB = {'Z': r'\draw[green,fill=green,opacity=0.8,line width = 0.25cm]',
       'X': r'\draw[red,fill=red,opacity=0.8,line width = 0.25cm]',
       'M': r'\draw[violet,fill=violet,opacity=0.8,line width = 0.25cm]'}


def emit(g, webs, path, zscale=4.0, node_size='0.7cm', lift=None,
         swap_y=None, planes=None):
    """webs: list of {edge: 'X'|'Y'|'Z'} dicts, overlaid. planes: list of
    vertex ids; a translucent constant-time slice is drawn at the
    (compacted) height of each."""
    from viewer import compact_z
    compact_z(g)
    name = {v: f'v{v}' for v in g.vertices()}
    verts = sorted(g.vertices(), key=lambda v: (g.vdata(v, 'z', 0.0),
                                                g.row(v), g.qubit(v)))
    L = ['\\begin{tikzpicture}[tdplot_main_coords,every node/.style='
         f'{{minimum size={node_size}}}]']
    lastz = None
    lift = lift or {}
    swap_y = swap_y or set()
    for v in verts:
        z = g.vdata(v, 'z', 0.0) + lift.get(v, 0.0)
        if z != lastz:
            if lastz is not None:
                L.append(r'\end{scope}')
            L.append(r'\begin{scope}')
            lastz = z
        x = 2 + 4 * g.row(v)
        y = 2 + 4 * g.qubit(v)
        h = zscale * z
        ty = g.type(v)
        half = g.phase(v) == Fraction(1, 2)
        if ty == VertexType.Z:
            style = BLUE if half else GREEN
        elif ty == VertexType.X:
            style = BLUE if half else RED
        else:
            style = NONE
        L.append(f'{style} ({name[v]}) at ({x:g},{y:g},{h:g}) {{}};')
    L.append(r'\end{scope}')
    hnodes = {v for v in g.vertices()
              if g.type(v) == VertexType.X and g.phase(v) == Fraction(1, 2)}
    # layering (bottom to top): plain wires < webs < Hadamard edges < nodes
    L.append(r'\begin{pgfonlayer}{background}')
    if planes:
        xs_ = [2 + 4 * g.row(v) for v in g.vertices()]
        ys_ = [2 + 4 * g.qubit(v) for v in g.vertices()]
        x0, x1 = min(xs_) - 2, max(xs_) + 2
        y0, y1 = min(ys_) - 2, max(ys_) + 2
        for pv in planes:
            h = zscale * (g.vdata(pv, 'z', 0.0) + (lift or {}).get(pv, 0.0))
            L.append(r'\filldraw[fill=gray, opacity=0.18, draw=gray] '
                     f'({x0:g},{y0:g},{h:g}) -- ({x1:g},{y0:g},{h:g}) -- '
                     f'({x1:g},{y1:g},{h:g}) -- ({x0:g},{y1:g},{h:g}) '
                     '-- cycle;')
    hedges = []
    for e in g.edges():
        a, b = g.edge_s(e), g.edge_t(e)
        if a in hnodes or b in hnodes:
            hedges.append((a, b))
        else:
            L.append(f'{WIRE} ({name[a]}) to ({name[b]});')
    SWAP = {'X': 'Z', 'Z': 'X'}
    per_edge = {}
    for w in webs:
        for e, p in w.items():
            s = per_edge.setdefault(e, set())
            s.update({'X', 'Z'} if p == 'Y' else {p})
    for e, cols in sorted(per_edge.items()):
        va, vb = e
        a, b = name[va], name[vb]
        hedge = va in hnodes or vb in hnodes
        both = len(cols) == 2
        if hedge:
            # decoration changes colour across the Hadamard edge; the half
            # nearest each node is the decoration valid at THAT node: the
            # solved decoration at the plain end, its H-conjugate at the
            # blue (converted X pi/2) end.
            if vb in hnodes:
                o, bn = a, b
            else:
                o, bn = b, a
            if both:
                # Y = both strands, each swapping colour at the midpoint
                for p in sorted(cols):
                    off = '-0.2,-0.2,0' if p == 'Z' else '0.2,0.2,0'
                    L.append(f'{WEB[p]} ($({o})+({off})$) to '
                             f'($($({o})!0.5!({bn})$)+({off})$);')
                    L.append(f'{WEB[SWAP[p]]} ($($({o})!0.5!({bn})$)+({off})$)'
                             f' to ($({bn})+({off})$);')
            else:
                p = next(iter(cols))
                L.append(f'{WEB[p]} ({o}) to ($({o})!0.5!({bn})$);')
                L.append(f'{WEB[SWAP[p]]} ($({o})!0.5!({bn})$) to ({bn});')
            continue
        swapped = swap_y and (va in swap_y or vb in swap_y)
        for p in sorted(cols):
            if both:
                if swapped:
                    off = '0.2,0.2,0' if p == 'Z' else '-0.2,-0.2,0'
                else:
                    off = '-0.2,-0.2,0' if p == 'Z' else '0.2,0.2,0'
                L.append(f'{WEB[p]} ($({a})+({off})$) to ($({b})+({off})$);')
            else:
                L.append(f'{WEB[p]} ({a}) to ({b});')
    for a, b in hedges:
        L.append(f'{HEDGE} ({name[a]}) to ({name[b]});')
    L.append(r'\end{pgfonlayer}')
    L.append(r'\end{tikzpicture}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('wrote', path, f'({g.num_vertices()} nodes, {sum(len(w) for w in webs)} web edges)')
