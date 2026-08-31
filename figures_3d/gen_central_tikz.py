"""Generate the 3D layered ZX/web TikZ panels for the Lao-Criger CENTRAL
qubit injection at d=5, in the exact conventions of the paper's figure 12
(make_paper_tikz.py): tdplot layers t0/t1/t2/t3, node names n{i}t{l},
b{j}t1, a{k}t2, 0.05cm wires, 0.25cm webs at opacity 0.8.

Outputs (to arXiv-2501.15566v5_draft/): fig_central_web_single.tex,
fig_central_webs_det.tex, fig_central_webs_nodet.tex."""
import sys, os

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', 'pauli_web_y_state_injection')
sys.path.insert(0, REPO)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'arXiv-2501.15566v5_draft')

import injection_webs as iw
from injection_webs import XCHECK, ZCHECK, audit, cr

# --- central (spiral-filled) initialisation, paper qubit numbering i=5c+r ---
PLUS = {(0, 4), (1, 4), (2, 4), (3, 4), (1, 3), (2, 3),
        (2, 1), (3, 1), (1, 0), (2, 0), (3, 0), (4, 0)}
SITE = (2, 2)                                   # data qubit 12

def central_kind(i, corner='Y'):
    c, r = cr(i)
    if (c, r) == SITE:
        return corner
    return 'plus' if (c, r) in PLUS else 'zero'

iw.init_kind = central_kind                     # patch BEFORE build
from make_viewers import System, ALL

HEI = {'t0': 0, 't1': 10, 't2': 20, 't3': 22.5}
XPOS = {0: (.5, .5), 1: (.5, 2.5), 2: (.5, 4.5), 3: (1.5, -.5), 4: (1.5, 1.5),
        5: (1.5, 3.5), 6: (2.5, .5), 7: (2.5, 2.5), 8: (2.5, 4.5),
        9: (3.5, -.5), 10: (3.5, 1.5), 11: (3.5, 3.5)}
ZPOS = {0: (-.5, .5), 1: (-.5, 2.5), 2: (.5, 1.5), 3: (.5, 3.5), 4: (1.5, .5),
        5: (1.5, 2.5), 6: (2.5, 1.5), 7: (2.5, 3.5), 8: (3.5, .5),
        9: (3.5, 2.5), 10: (4.5, 1.5), 11: (4.5, 3.5)}

def dpos(i):
    return 2 + 4 * (i // 5), 2 + 4 * (i % 5)

def apos(xy):
    return 2 + 4 * xy[0], 2 + 4 * xy[1]

WIRE = r'\draw[black,fill=black,opacity=1,line width = 0.05cm]'
WEB = {'Z': r'\draw[green,fill=green,opacity=0.8,line width = 0.25cm]',
       'X': r'\draw[red,fill=red,opacity=0.8,line width = 0.25cm]'}
GREEN = r'\node[fill=zx_green,shape=circle,draw=black]'
BLUE = r'\node[fill=zx_blue,shape=circle,draw=black]'
RED = r'\node[fill=zx_red,shape=circle,draw=black]'
NONE = r'\node[fill=none,shape=circle,draw=none]'

def name_map(V):
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

def tikz_panel(webs, V, labels=()):
    NM = name_map(V)
    L = [r'\begin{tikzpicture}[tdplot_main_coords,every node/.style={minimum size=1cm}]']
    L.append(r'\begin{scope}')
    for i in range(25):
        x, y = dpos(i)
        k = central_kind(i)
        style = RED if k == 'zero' else (BLUE if k == 'Y' else GREEN)
        L.append(f'{style} (n{i}t0) at ({x},{y},{HEI["t0"]}) {{}};')
    for i in labels:
        x, y = dpos(i)
        L.append(r'\node[draw=none,fill=none,text=cyan,scale=1.6] at '
                 f'({x + 0.8},{y + 0.7},{HEI["t0"]}) {{${i}$}};')
    L.append(r'\end{scope}')
    L.append(r'\begin{scope}')
    for i in range(25):
        x, y = dpos(i)
        L.append(f'{RED} (n{i}t1) at ({x},{y},{HEI["t1"]}) {{}};')
    for j in sorted(XCHECK):
        x, y = apos(XPOS[j])
        L.append(f'{GREEN} (b{j}t1) at ({x},{y},{HEI["t1"]}) {{}};')
    L.append(r'\end{scope}')
    L.append(r'\begin{scope}')
    for i in range(25):
        x, y = dpos(i)
        L.append(f'{GREEN} (n{i}t2) at ({x},{y},{HEI["t2"]}) {{}};')
    for k in sorted(ZCHECK):
        x, y = apos(ZPOS[k])
        L.append(f'{RED} (a{k}t2) at ({x},{y},{HEI["t2"]}) {{}};')
    L.append(r'\end{scope}')
    L.append(r'\begin{scope}')
    for i in range(25):
        x, y = dpos(i)
        L.append(f'{NONE} (n{i}t3) at ({x},{y},{HEI["t3"]}) {{}};')
    L.append(r'\end{scope}')
    L.append(r'\begin{scope}')
    for j in sorted(XCHECK):
        for i in XCHECK[j]:
            L.append(f'{WIRE} (b{j}t1) to (n{i}t1);')
    for k in sorted(ZCHECK):
        for i in ZCHECK[k]:
            L.append(f'{WIRE} (a{k}t2) to (n{i}t2);')
    L.append(r'\end{scope}')
    L.append(r'\begin{pgfonlayer}{background}')
    for lo, hi in (('t0', 't1'), ('t1', 't2'), ('t2', 't3')):
        for i in range(25):
            L.append(f'{WIRE} (n{i}{lo}) to (n{i}{hi});')
    L.append(r'\end{pgfonlayer}')
    per_edge = {}
    for w in webs:
        for e, p in w.items():
            s = per_edge.setdefault(e, set())
            s.update({'X', 'Z'} if p == 'Y' else {p})
    L.append(r'\begin{pgfonlayer}{background}')
    for e, cols in sorted(per_edge.items(), key=lambda kv: (NM[kv[0][0]], NM[kv[0][1]])):
        a, b = NM[e[0]], NM[e[1]]
        both = len(cols) == 2
        for p in sorted(cols):
            if both:
                off = '-0.2,-0.2,0' if p == 'Z' else '0.2,0.2,0'
                L.append(f'{WEB[p]} ($({a})+({off})$) to ($({b})+({off})$);')
            else:
                L.append(f'{WEB[p]} ({a}) to ({b});')
    L.append(r'\end{pgfonlayer}')
    L.append(r'\end{tikzpicture}')
    return '\n'.join(L)

g1, V1, m1 = iw.build(rounds=1)
S1 = System(g1, V1)
det_webs, nodet_webs = [], []
det_ids, nodet_ids = [], []
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
    (det_ids if det else nodet_ids).append((kind, j))
print(f'{len(det_webs)} webs with a round-1 detector: {det_ids}')
print(f'{len(nodet_webs)} without: {nodet_ids}')

# single-web illustration: the X-plaquette watching the qubit above the site
single = det_webs[det_ids.index(('X', 5))]
for name, content in (('fig_central_web_single.tex',
                       tikz_panel([single], V1, labels=(8, 9, 13, 14))),
                      ('fig_central_webs_det.tex', tikz_panel(det_webs, V1)),
                      ('fig_central_webs_nodet.tex', tikz_panel(nodet_webs, V1))):
    with open(os.path.join(OUT, name), 'w') as fh:
        fh.write(content + '\n')
    print('wrote', name)
