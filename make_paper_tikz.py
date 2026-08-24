"""Emit the two overlaid-web figures as TikZ in the paper's own style.

Produces one tikzpicture per panel, using the same conventions as the existing
figures in main.tex: tdplot_main_coords, node names n<i>t<layer> / b<j>t1 / a<k>t2,
data qubit i at (2+4*(i//5), 2+4*(i%5)), layers at heights 0 / 10 / 20 / 22.5,
0.05cm wires and 0.25cm webs at opacity 0.8.
"""
import itertools, random
from injection_webs import build, web_system, nullspace, vec_to_web, audit, XCHECK, ZCHECK, init_kind
from make_viewers import System, ALL

HEI = {'t0': 0, 't1': 10, 't2': 20, 't3': 22.5}


def dpos(i):
    return 2 + 4 * (i // 5), 2 + 4 * (i % 5)


def apos(xy):
    return 2 + 4 * xy[0], 2 + 4 * xy[1]


XPOS = {0: (.5, .5), 1: (.5, 2.5), 2: (.5, 4.5), 3: (1.5, -.5), 4: (1.5, 1.5), 5: (1.5, 3.5),
        6: (2.5, .5), 7: (2.5, 2.5), 8: (2.5, 4.5), 9: (3.5, -.5), 10: (3.5, 1.5), 11: (3.5, 3.5)}
ZPOS = {0: (-.5, .5), 1: (-.5, 2.5), 2: (.5, 1.5), 3: (.5, 3.5), 4: (1.5, .5), 5: (1.5, 2.5),
        6: (2.5, 1.5), 7: (2.5, 3.5), 8: (3.5, .5), 9: (3.5, 2.5), 10: (4.5, 1.5), 11: (4.5, 3.5)}

WIRE = r'\draw[black,fill=black,opacity=1,line width = 0.05cm]'
WEB = {'Z': r'\draw[green,fill=green,opacity=0.8,line width = 0.25cm]',
       'X': r'\draw[red,fill=red,opacity=0.8,line width = 0.25cm]'}
GREEN = r'\node[fill=zx_green,shape=circle,draw=black]'
RED = r'\node[fill=zx_red,shape=circle,draw=black]'
NONE = r'\node[fill=none,shape=circle,draw=none]'


def name_map(V):
    """my vertex keys -> the paper's TikZ node names"""
    m = {}
    for i in range(25):
        m[V[('init', i)]] = f'n{i}t0'
        m[V[('d', 1, 'x', i)]] = f'n{i}t1'
        m[V[('d', 1, 'z', i)]] = f'n{i}t2'
        m[V[('out', i)]] = f'n{i}t3'
    for j in XCHECK:
        m[V[('bx', 1, j)]] = f'b{j}t1'
    for k in ZCHECK:
        m[V[('az', 1, k)]] = f'a{k}t2'
    return m


def tikz_panel(webs):
    """webs: list of edge dicts {(u,v): 'X'|'Y'|'Z'} on the 1-round graph."""
    g, V, meta = build(rounds=1)
    NM = name_map(V)
    L = [r'\begin{tikzpicture}[tdplot_main_coords,every node/.style={minimum size=1cm}]']

    # --- nodes ------------------------------------------------------------
    L.append(r'\begin{scope}')                       # t0, the initial states
    for i in range(25):
        x, y = dpos(i)
        k = init_kind(i)
        style = RED if k == 'zero' else GREEN
        lab = r'{$\pi/2$}' if k == 'Y' else '{}'
        L.append(f'{style} (n{i}t0) at ({x},{y},{HEI["t0"]}) {lab};')
    L.append(r'\end{scope}')

    L.append(r'\begin{scope}')                       # t1, X-check round
    for i in range(25):
        x, y = dpos(i)
        L.append(f'{RED} (n{i}t1) at ({x},{y},{HEI["t1"]}) {{}};')
    for j in sorted(XCHECK):
        x, y = apos(XPOS[j])
        L.append(f'{GREEN} (b{j}t1) at ({x},{y},{HEI["t1"]}) {{}};')
    L.append(r'\end{scope}')

    L.append(r'\begin{scope}')                       # t2, Z-check round
    for i in range(25):
        x, y = dpos(i)
        L.append(f'{GREEN} (n{i}t2) at ({x},{y},{HEI["t2"]}) {{}};')
    for k in sorted(ZCHECK):
        x, y = apos(ZPOS[k])
        L.append(f'{RED} (a{k}t2) at ({x},{y},{HEI["t2"]}) {{}};')
    L.append(r'\end{scope}')

    L.append(r'\begin{scope}')                       # t3, invisible output stubs
    for i in range(25):
        x, y = dpos(i)
        L.append(f'{NONE} (n{i}t3) at ({x},{y},{HEI["t3"]}) {{}};')
    L.append(r'\end{scope}')

    # --- ancilla wires ----------------------------------------------------
    L.append(r'\begin{scope}')
    for j in sorted(XCHECK):
        for i in XCHECK[j]:
            L.append(f'{WIRE} (b{j}t1) to (n{i}t1);')
    for k in sorted(ZCHECK):
        for i in ZCHECK[k]:
            L.append(f'{WIRE} (a{k}t2) to (n{i}t2);')
    L.append(r'\end{scope}')

    # --- worldlines -------------------------------------------------------
    L.append(r'\begin{pgfonlayer}{background}')
    for lo, hi in (('t0', 't1'), ('t1', 't2'), ('t2', 't3')):
        for i in range(25):
            L.append(f'{WIRE} (n{i}{lo}) to (n{i}{hi});')
    L.append(r'\end{pgfonlayer}')

    # --- the webs, squished together -------------------------------------
    # union over the selected webs; an edge carrying both colours gets the two
    # drawn at opposite small offsets, exactly as the existing figures do.
    per_edge = {}
    for w in webs:
        for e, p in w.items():
            s = per_edge.setdefault(e, set())
            s.update({'X', 'Z'} if p == 'Y' else {p})
    L.append(r'\begin{pgfonlayer}{background}')
    n_edges = 0
    for e, cols in sorted(per_edge.items(), key=lambda kv: (NM[kv[0][0]], NM[kv[0][1]])):
        u, v = e
        a, b = NM[u], NM[v]
        both = len(cols) == 2
        for p in sorted(cols):
            off = ''
            if both:
                d = '-0.2,-0.2,0' if p == 'Z' else '0.2,0.2,0'
                off = d
            if off:
                L.append(f'{WEB[p]} ($({a})+({off})$) to ($({b})+({off})$);')
            else:
                L.append(f'{WEB[p]} ({a}) to ({b});')
            n_edges += 1
    L.append(r'\end{pgfonlayer}')
    L.append(r'\end{tikzpicture}')
    return '\n'.join(L), len(per_edge), n_edges


# ---------------------------------------------------------------- the two web sets
g1, V1, m1 = build(rounds=1)
S1 = System(g1, V1)
det_webs, nodet_webs = [], []
for kind, j in ALL:
    f = dict(S1.no_outputs())
    f.update(S1.cover(kind, 1, j))
    res = S1.solve(f)
    det = res is not None
    if not det:
        res = S1.solve(S1.cover(kind, 1, j))
    v, bw = S1.small(*res, seed=j)
    w = S1.web(v)
    assert not audit(g1, w, m1), (kind, j)
    (det_webs if det else nodet_webs).append(w)

print(f'{len(det_webs)} webs with a round-1 detector, {len(nodet_webs)} without')

pa, ea, da = tikz_panel(det_webs)
pb, eb, db = tikz_panel(nodet_webs)
print(f'  post-selection panel: {ea} decorated edges ({da} draw commands)')
print(f'  no-detector panel   : {eb} decorated edges ({db} draw commands)')

FIG = r"""
\begin{figure}[!h]
    \centering
    \tdplotsetmaincoords{70}{23}
    \subfloat[\label{fig:webs_postselect}\add{The $%d$ plaquettes that do admit a first round
    Pauli web, all drawn together. Every one of these webs terminates on the initial states at
    the bottom. They are the same plaquettes labelled `$+1$' in figure
    \ref{fig:surface_code_Y_inj_post_selection}.}]{
    \resizebox{0.46\linewidth}{!}{
%s
    }
    }
    \quad
    \subfloat[\label{fig:webs_nodetector}\add{The $%d$ plaquettes that do not, again all
    drawn together. These webs run only forward in time to the last time slice and reach none
    of the initial states.}]{
    \resizebox{0.46\linewidth}{!}{
%s
    }
    }
    \caption{\label{fig:web_mesh}\add{The first round Pauli webs of the $\ket{Y}$ injection
    protocol. Neither panel is a single Pauli web. Each one shows all of the individual webs
    superposed, so an edge that several webs share is drawn several times over. Green edges
    carry a $Z$-type decoration and red edges an $X$-type one.}}
\end{figure}
""" % (len(det_webs), pa, len(nodet_webs), pb)

open('paper_web_mesh.tex', 'w').write(FIG)
print(f'wrote paper_web_mesh.tex ({len(FIG)} chars)')
