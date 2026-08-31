"""Top-view time slices of one syndrome-extraction cycle of the
optimised (serialised) central schedule at d = 5, drawn as the 2D
ZX diagram seen from above: green dot on the control end, red dot on
the target end of every CNOT in the slice. Colliding slots are split
into their two serialisation passes (stim/circuit face-scan order)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'circuit_noise'))
import circuits
from t_experiment import opt_central

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'latex_pauli_web_y_state_injection', 'version_5_v1')
d = 5
site = (d // 2, d // 2)
checks = circuits.rect_checks(d)

panels = []          # (label, [(typ, anc_pos, cell)])
for slot in range(4):
    passes = [[], []]
    seen = {}
    for typ, pos, sup in checks:
        name = opt_central(typ)[slot]
        if name not in sup: continue
        cell = sup[name]
        lvl = seen.get(cell, 0)
        seen[cell] = lvl + 1
        passes[lvl].append((typ, pos, cell))
    if passes[1]:
        panels.append((f'layer {slot+1}a', passes[0]))
        panels.append((f'layer {slot+1}b', passes[1]))
    else:
        panels.append((f'layer {slot+1}', passes[0]))

L = [r'\begin{tabular}{ccc}']
cellrows = []
for pi, (lab, gates) in enumerate(panels):
    P = [r'\begin{tikzpicture}[scale=0.62]']
    # faces for orientation
    for typ, pos, sup in checks:
        col = 'red!7' if typ == 'X' else 'green!9'
        fi, fj = pos[0] - 0.5, pos[1] - 0.5
        cs = [c for c in sup.values()]
        if len(cs) == 4:
            P.append(f'\\fill[{col}] ({fi},{fj}) rectangle '
                     f'({fi+1},{fj+1});')
        else:
            import math
            (x0, y0), (x1, y1) = sorted(cs)
            cmx, cmy = (x0 + x1) / 2, (y0 + y1) / 2
            r = math.hypot(x1 - x0, y1 - y0) / 2
            th0 = math.degrees(math.atan2(y0 - cmy, x0 - cmx))
            probe = (math.cos(math.radians(th0 + 90)),
                     math.sin(math.radians(th0 + 90)))
            out = (pos[0] - cmx, pos[1] - cmy)
            th1 = th0 + (180 if probe[0]*out[0] + probe[1]*out[1] > 0
                         else -180)
            P.append(f'\\fill[{col}] ({x0},{y0}) '
                     f'arc[start angle={th0:.1f}, end angle={th1:.1f}, '
                     f'radius={r:.3f}] -- cycle;')
    # idle data qubits
    active = {g[2] for g in gates}
    for i in range(d):
        for j in range(d):
            if (i, j) not in active:
                P.append(f'\\draw[gray!60, fill=gray!25] ({i},{j}) '
                         'circle (0.09);')
    # site ring
    P.append(f'\\draw[blue, very thick] {site} circle (0.30);')
    # gates: edge + green control dot + red target dot
    for typ, pos, cell in gates:
        if typ == 'X':
            c, t = pos, cell
        else:
            c, t = cell, pos
        P.append(f'\\draw[very thick] {c} -- {t};')
        P.append(f'\\draw[fill={{rgb,255: red,216; green,248; blue,216}}] '
                 f'{c} circle (0.16);')
        P.append(f'\\draw[fill={{rgb,255: red,232; green,165; blue,165}}] '
                 f'{t} circle (0.16);')
    P.append(r'\end{tikzpicture}')
    cellrows.append(('\n'.join(P), lab))
rows = []
for r in range(0, len(cellrows), 3):
    chunk = cellrows[r:r+3]
    rows.append(' &\n'.join(c[0] for c in chunk) + r'\\')
    rows.append(' & '.join(f'{{\\footnotesize ({chr(97+r+i)}) {c[1]}}}'
                           for i, c in enumerate(chunk)) +
                (r'\\[2mm]' if r + 3 < len(cellrows) else ''))
L += rows
L.append(r'\end{tabular}')
open(os.path.join(OUT, 'fig_slices_opt_d5.tex'), 'w').write('\n'.join(L) + '\n')
print(f'{len(panels)} panels:', [p[0] for p in panels])
print('site collides in:', [lab for lab, gs in panels
                            if any(c == site for _, _, c in gs)])
