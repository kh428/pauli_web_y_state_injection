"""Slot-by-slot top views of the two syndrome-extraction orderings of
the circuit-level section -- the generic N/Z interleave and the
Tomita-Svore ordering of Lao-Criger -- at d = 3, in the style of the
optimised-schedule slice figure but smaller. VERIFIED BY CONSTRUCTION:
the gates in each panel are parsed from the very stim circuits
(circuits.build with sched_nz / sched_lc) that the census table and
the Monte-Carlo run on; a collision-free assertion runs per tick."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'circuit_noise'))
import circuits
from circuits import build, spiral_kinds, rect_checks, sched_lc, sched_nz

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'latex_pauli_web_y_state_injection', 'version_5_v1')
d = 3
checks = rect_checks(d)
q = lambda c: d * c[0] + c[1]
anc_inv = {d * d + ci: checks[ci] for ci in range(len(checks))}

def ticks_of(sched):
    txt, info = build(d, spiral_kinds(d), sched=sched, noisy_rounds=1,
                     extra_rounds=0, tail=False)
    ticks, cur, started = [], [], False
    for l in txt.split('\n'):
        if l == 'TICK':
            if started: ticks.append(cur)
            started, cur = True, []
        elif started and l.startswith('CX '):
            cur.append(tuple(int(x) for x in l.split()[1:]))
    if cur: ticks.append(cur)
    ticks = ticks[:4]
    for k, gates in enumerate(ticks):
        touched = [g[0] for g in gates] + [g[1] for g in gates]
        assert len(touched) == len(set(touched)), \
            f'collision in tick {k+1}!'
    return ticks

def panel(gates):
    P = [r'\begin{tikzpicture}[scale=0.55]']
    for typ, pos, sup in checks:
        col = 'red!7' if typ == 'X' else 'green!9'
        cs = list(sup.values())
        if len(cs) == 4:
            fi, fj = pos[0] - 0.5, pos[1] - 0.5
            P.append(f'\\fill[{col}] ({fi},{fj}) rectangle ({fi+1},{fj+1});')
        else:
            (x0, y0), (x1, y1) = sorted(cs)
            cmx, cmy = (x0+x1)/2, (y0+y1)/2
            r = math.hypot(x1-x0, y1-y0)/2
            th0 = math.degrees(math.atan2(y0-cmy, x0-cmx))
            probe = (math.cos(math.radians(th0+90)),
                     math.sin(math.radians(th0+90)))
            out = (pos[0]-cmx, pos[1]-cmy)
            th1 = th0 + (180 if probe[0]*out[0]+probe[1]*out[1] > 0 else -180)
            P.append(f'\\fill[{col}] ({x0},{y0}) arc[start angle={th0:.1f}, '
                     f'end angle={th1:.1f}, radius={r:.3f}] -- cycle;')
    active = set()
    for a, b in gates:
        for qq in (a, b):
            if qq < d * d:
                active.add(divmod(qq, d))
    for i in range(d):
        for j in range(d):
            if (i, j) not in active:
                P.append(f'\\draw[gray!60, fill=gray!25] ({i},{j}) '
                         'circle (0.09);')
    for a, b in gates:
        cpos = anc_inv[a][1] if a >= d*d else divmod(a, d)
        tpos = anc_inv[b][1] if b >= d*d else divmod(b, d)
        P.append(f'\\draw[very thick] {cpos} -- {tpos};')
        P.append(f'\\draw[fill={{rgb,255: red,216; green,248; blue,216}}] '
                 f'{cpos} circle (0.15);')
        P.append(f'\\draw[fill={{rgb,255: red,232; green,165; blue,165}}] '
                 f'{tpos} circle (0.15);')
    P.append(r'\end{tikzpicture}')
    return '\n'.join(P)

rows = []
for name, sched in [('N/Z', sched_nz), ('TS', sched_lc)]:
    ticks = ticks_of(sched)
    cells = ' &\n'.join(panel(g) for g in ticks)
    rows.append((name, cells))
L = [r'\begin{tabular}{ccccc}']
L.append(' & {\\footnotesize layer 1} & {\\footnotesize layer 2} & '
         '{\\footnotesize layer 3} & {\\footnotesize layer 4}\\\\')
for name, cells in rows:
    L.append(f'\\rotatebox{{90}}{{\\footnotesize {name}}} &\n'
             + cells + r'\\[1.5mm]')
L.append(r'\end{tabular}')
open(os.path.join(OUT, 'fig_sched_slices_d3.tex'), 'w').write(
    '\n'.join(L) + '\n')
print('fig_sched_slices_d3.tex written; both schedules collision-free')
