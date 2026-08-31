"""Full circuit-level diagram of the corner scheme at d=3, exactly the
circuit the census runs on: initialisations, two rounds of N/Z-scheduled
CNOT extraction (round 1 + post-selected verify round), ancilla readouts,
open data outputs. Violet annotations: the three noise species, and at
every malignant location the number of malignant classes it carries
(from iter_li_pergate). Written as 2D tikz in the paper style."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physical import rect, region_checks, injection_kind
from iter_li_pergate import pergate

d = 3
P = rect(0, 0, d, d)
xs, zs = region_checks(P)
kinds = {c: injection_kind(c, 0, 0, d) for c in P}
mal_cnot, mal_init = pergate(kinds, d)

def sched(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NW, SW, NE, SE] if typ == 'X' else [NW, NE, SW, SE])
    return [c if c in set(sup) else None for c in order]

checks = ([('X', pos, sched(pos, sup, 'X')) for pos, sup in xs] +
          [('Z', pos, sched(pos, sup, 'Z')) for pos, sup in zs])

def qid(c):
    return 3 * int(c[0]) + int(c[1])

SP = 1.5
DY = {i: 21 - SP * i for i in range(9)}
AY = {j: 7 - SP * (j + 1) for j in range(len(checks))}
XI = 0.0
STEP = 1.9
def xs_round(x0):
    return [x0 + STEP * (k + 1) for k in range(4)]
XS1 = xs_round(XI)                # 1.9 .. 7.6
XC1 = XS1[-1] + STEP              # caps round 1
XI2 = XC1 + STEP                  # ancilla re-init
XS2 = xs_round(XI2)
XC2 = XS2[-1] + STEP
XE = XC2 + 1.6                    # data wires run off (open outputs)

GREEN = r'\node[fill=zx_green,shape=circle,draw=black,minimum size=0.4cm]'
RED = r'\node[fill=zx_red,shape=circle,draw=black,minimum size=0.4cm]'
BLUE = r'\node[fill=zx_blue,shape=circle,draw=black,minimum size=0.4cm]'
W = r'\draw[black,line width=0.045cm]'

L = [r'\begin{tikzpicture}[scale=.55,every node/.style={minimum size=0.4cm},on grid]']
# data wires
for c in P:
    i = qid(c)
    y = DY[i]
    k = kinds[c]
    style = {'0': RED, '+': GREEN, 'Y': BLUE}[k]
    lab = {'0': r'\ket{0}', '+': r'\ket{+}', 'Y': r'\ket{Y}'}[k]
    L.append(f'{style} (di{i}) at ({XI},{y}) {{}};')
    L.append(r'\node[draw=none,fill=none,scale=0.85] at '
             f'({XI-1.2},{y}) {{${lab}$}};')
    L.append(r'\node[draw=none,fill=none,text=cyan,scale=0.8] at '
             f'({XI-2.3},{y}) {{${i}$}};')
    L.append(f'{W} (di{i}) -- ({XE},{y});')
    L.append(r'\node[draw=none,fill=none,scale=1.1] at '
             f'({XE+0.55},{y}) {{$\\cdots$}};')
    if tuple(c) in [tuple(x) for x in mal_init]:
        L.append(r'\draw[->,violet,line width=0.03cm] '
                 f'({XI-0.62},22.55) to[bend right=18] ({XI-0.3},{y+0.35});')
# ancilla wires, both rounds
for j, (typ, pos, sup) in enumerate(checks):
    y = AY[j]
    node = GREEN if typ == 'X' else RED
    lab = r'\ket{+}' if typ == 'X' else r'\ket{0}'
    L.append(r'\node[draw=none,fill=none,scale=0.75] at '
             f'({XI-2.3},{y}) {{${typ}$}};')
    for (xi, xc, tag) in [(XI, XC1, 'r'), (XI2, XC2, 'v0')]:
        L.append(f'{node} (ai{j}{tag}) at ({xi},{y}) {{}};')
        if tag == 'r':
            L.append(r'\node[draw=none,fill=none,scale=0.8] at '
                     f'({xi-1.2},{y}) {{${lab}$}};')
        L.append(f'{W} (ai{j}{tag}) -- ({xc},{y});')
        L.append(f'{node} (ac{j}{tag}) at ({xc},{y}) {{}};')
# CNOTs at their scheduled steps, both rounds, with malignant counts
for (XSr, tag) in [(XS1, 'r'), (XS2, 'v0')]:
    for j, (typ, pos, sup) in enumerate(checks):
        for k, c in enumerate(sup):
            if c is None:
                continue
            x = XSr[k]
            i = qid(c)
            da, aa = (RED, GREEN) if typ == 'X' else (GREEN, RED)
            L.append(f'{da} (g{tag}{j}k{k}d) at ({x},{DY[i]}) {{}};')
            L.append(f'{aa} (g{tag}{j}k{k}a) at ({x},{AY[j]}) {{}};')
            ang = 4 + 2 * (j % 4)
            side = 'left' if typ == 'X' else 'right'
            L.append(f'{W} (g{tag}{j}k{k}d) to[bend {side}={ang}] '
                     f'(g{tag}{j}k{k}a);')
            key = (tag, typ, tuple(pos), (float(c[0]), float(c[1])))
            if key in mal_cnot:
                n = len(mal_cnot[key])
                L.append(r'\node[draw=none,fill=none,text=violet,scale=0.85] at '
                         f'({x+0.55},{DY[i]-0.55}) {{${n}$}};')
# noise-species labels
ytop = DY[0] + 1.9
L.append(r'\node[draw=none,fill=none,text=violet,scale=0.9] at '
         f'({XI-0.62},{ytop+0.9}) {{$p_I$}};')
L.append(r'\node[draw=none,fill=none,text=violet,scale=0.9] at '
         f'({(XS1[0]+XS1[-1])/2},{ytop}) {{$P_c \\otimes P_t$, each $p_2/15$}};')
L.append(f'\\draw[->,violet,line width=0.03cm] '
         f'({XS1[1]},{ytop-0.45}) -- ({XS1[1]},{DY[0]+0.5});')
L.append(r'\node[draw=none,fill=none,text=violet,scale=0.9] at '
         f'({XC1},{ytop}) {{$p_M$}};')
L.append(f'\\draw[->,violet,line width=0.03cm] '
         f'({XC1},{ytop-0.45}) -- ({XC1},{AY[0]+0.5});')
# round braces
ybot = AY[len(checks)-1] - 1.1
L.append(r'\draw[decorate,decoration={brace,mirror,amplitude=5pt}] '
         f'({XI+0.6},{ybot}) -- ({XC1+0.4},{ybot}) '
         r'node[draw=none,fill=none,midway,below=6pt,scale=0.85] {round $1$};')
L.append(r'\draw[decorate,decoration={brace,mirror,amplitude=5pt}] '
         f'({XI2-0.4},{ybot}) -- ({XC2+0.4},{ybot}) '
         r'node[draw=none,fill=none,midway,below=6pt,scale=0.85] '
         r'{round $2$ (post-selected)};')
L.append(r'\end{tikzpicture}')
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'arXiv-2501.15566v5_draft_post_LC_read', 'fig_circuit_full.tex')
open(out, 'w').write('\n'.join(L) + '\n')
print('wrote fig_circuit_full.tex:', len(checks), 'ancillas,',
      sum(len(m) for m in mal_cnot.values()), 'CNOT classes marked,',
      len(mal_init), 'init flips marked')
