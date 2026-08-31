"""Two-panel 2D figure: why the corner scheme has four malicious
initialisation errors and the central scheme only two. Post-selectable
plaquettes are read from circuits.build's detector list (the same
machinery as the census), so the '+1' sets are verified, not drawn by
hand. House lattice style (light faces, semicircular boundary lobes,
init-coloured data dots)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'circuit_noise'))
from circuits import build, rect_checks, corner_kinds, spiral_kinds, sched_nz

OUT = os.path.dirname(os.path.abspath(__file__))
d = 5
checks = rect_checks(d)

def postselectable(kinds):
    txt, info = build(d, kinds, sched=sched_nz, noisy_rounds=1,
                      extra_rounds=0, tail=False)
    import re
    out = set()
    for lab, recs in info['dets']:
        m = re.match(r'r0 ([XZ])\(([-\d.]+),\s*([-\d.]+)\)', lab)
        out.add((m.group(1), (float(m.group(2)), float(m.group(3)))))
    return out

def faces_of(cell):
    return [(typ, pos, sup) for typ, pos, sup in checks
            if cell in sup.values()]

def panel(kinds, site, neighbour, site_err, nb_err, nb_num):
    ps = postselectable(kinds)
    P = [r'\begin{tikzpicture}[scale=0.68]']
    for typ, pos, sup in checks:
        col = 'red!12' if typ == 'X' else 'green!14'
        cs = list(sup.values())
        if len(cs) == 4:
            fi, fj = pos[0]-0.5, pos[1]-0.5
            P.append(f'\\fill[{col}] ({fi},{fj}) rectangle ({fi+1},{fj+1});')
        else:
            (x0,y0),(x1,y1) = sorted(cs)
            cmx, cmy = (x0+x1)/2, (y0+y1)/2
            r = math.hypot(x1-x0, y1-y0)/2
            th0 = math.degrees(math.atan2(y0-cmy, x0-cmx))
            pr = (math.cos(math.radians(th0+90)),
                  math.sin(math.radians(th0+90)))
            outv = (pos[0]-cmx, pos[1]-cmy)
            th1 = th0 + (180 if pr[0]*outv[0]+pr[1]*outv[1] > 0 else -180)
            P.append(f'\\fill[{col}] ({x0},{y0}) arc[start angle={th0:.1f},'
                     f' end angle={th1:.1f}, radius={r:.3f}] -- cycle;')
    # '+1' marks on verified post-selectable faces
    for typ, pos, sup in checks:
        if (typ, pos) in ps:
            P.append(f'\\node[draw=none,fill=none,scale=0.55] at '
                     f'({pos[0]},{pos[1]}) {{$+1$}};')
    # thick dashed outline on the neighbour's faces
    for typ, pos, sup in faces_of(neighbour):
        cs = list(sup.values())
        if len(cs) == 4:
            fi, fj = pos[0]-0.5, pos[1]-0.5
            P.append(f'\\draw[black, dashed, very thick] ({fi},{fj}) '
                     f'rectangle ({fi+1},{fj+1});')
        else:
            (x0,y0),(x1,y1) = sorted(cs)
            cmx, cmy = (x0+x1)/2, (y0+y1)/2
            r = math.hypot(x1-x0, y1-y0)/2
            th0 = math.degrees(math.atan2(y0-cmy, x0-cmx))
            pr = (math.cos(math.radians(th0+90)),
                  math.sin(math.radians(th0+90)))
            outv = (pos[0]-cmx, pos[1]-cmy)
            th1 = th0 + (180 if pr[0]*outv[0]+pr[1]*outv[1] > 0 else -180)
            P.append(f'\\draw[black, dashed, very thick] ({x0},{y0}) '
                     f'arc[start angle={th0:.1f}, end angle={th1:.1f}, '
                     f'radius={r:.3f}] -- cycle;')
    # data qubits coloured by initial state
    for i in range(d):
        for j in range(d):
            k = kinds[(i, j)]
            fill = {'+': 'zx_green', '0': 'zx_red', 'Y': 'blue!40'}[k]
            P.append(f'\\draw[black, fill={fill}] ({i},{j}) circle (0.14);')
    # site: blue ring + malicious error label
    P.append(f'\\draw[blue, very thick] {site} circle (0.34);')
    P.append(f'\\node[draw=none,fill=none,text=blue,scale=0.75] at '
             f'({site[0]-0.05},{site[1]-0.62}) {{${site_err}$}};')
    # neighbour: ring + label
    P.append(f'\\draw[violet, very thick] {neighbour} circle (0.34);')
    if nb_err:
        P.append(f'\\node[draw=none,fill=none,text=violet,scale=0.75] at '
                 f'({neighbour[0]+0.05},{neighbour[1]+0.62}) {{${nb_err}$}};')
    P.append(f'\\node[draw=none,fill=none,text=cyan,scale=0.7] at '
             f'({neighbour[0]+0.42},{neighbour[1]-0.3}) {{{nb_num}}};')
    P.append(r'\end{tikzpicture}')
    return '\n'.join(P)

ck = {c: k for c, k in corner_kinds(d).items()}
sk = dict(spiral_kinds(d))
pa = panel(ck, (0, 4), (1, 4), 'X, Z', 'X, Y', 9)
pb = panel(sk, (2, 2), (2, 3), 'X, Z', '', 13)
L = [r'\begin{tabular}{cc}', pa + ' &', pb + r'\\',
     r'{\footnotesize (a) corner scheme} & {\footnotesize (b) central scheme}',
     r'\end{tabular}']
open(os.path.join(OUT, 'fig_malicious_2panel.tex'), 'w').write(
    '\n'.join(L) + '\n')
print('corner postselectable:', len(postselectable(ck)),
      '| central:', len(postselectable(sk)))
print('written fig_malicious_2panel.tex')
