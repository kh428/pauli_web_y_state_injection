"""Cartoon of the central-qubit initialisation pattern at d = 9: data
qubits coloured by initial state (verified spiral_kinds(9)), the two
|+> staircase wedges outlined, pinwheel quadrants around the injected
centre. House 2D lattice style."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'circuit_noise'))
from circuits import rect_checks, spiral_kinds

OUT = os.path.dirname(os.path.abspath(__file__))
d = 9
c = d // 2
checks = rect_checks(d)
kinds = dict(spiral_kinds(d))

P = [r'\begin{tikzpicture}[scale=0.6]']
for typ, pos, sup in checks:
    col = 'red!10' if typ == 'X' else 'green!12'
    cs = list(sup.values())
    if len(cs) == 4:
        fi, fj = pos[0]-0.5, pos[1]-0.5
        P.append(f'\\fill[{col}] ({fi},{fj}) rectangle ({fi+1},{fj+1});')
    else:
        (x0,y0),(x1,y1) = sorted(cs)
        cmx, cmy = (x0+x1)/2, (y0+y1)/2
        r = math.hypot(x1-x0, y1-y0)/2
        th0 = math.degrees(math.atan2(y0-cmy, x0-cmx))
        pr = (math.cos(math.radians(th0+90)), math.sin(math.radians(th0+90)))
        outv = (pos[0]-cmx, pos[1]-cmy)
        th1 = th0 + (180 if pr[0]*outv[0]+pr[1]*outv[1] > 0 else -180)
        P.append(f'\\fill[{col}] ({x0},{y0}) arc[start angle={th0:.1f}, '
                 f'end angle={th1:.1f}, radius={r:.3f}] -- cycle;')

# staircase outline of the upper |+> wedge, and its 180-degree rotation
delta = 0.42
pts = []
# up the right staircase
for k in range(1, c + 1):
    pts.append((c + k - 1 + delta, c + k - delta))
    pts.append((c + k - 1 + delta, c + k + delta))
# across the top (right to left)
pts.append((c - c - delta, 2 * c + delta))
# down the left staircase
for k in range(c, 0, -1):
    pts.append((c - k - delta, c + k + delta))
    pts.append((c - k - delta, c + k - delta))
# close along the bottom of row k=1
pts.append((c + delta, c + 1 - delta))
def path(ps):
    return ' -- '.join(f'({x:.2f},{y:.2f})' for x, y in ps) + ' -- cycle'
P.append(f'\\draw[green!45!black, very thick, dashed] {path(pts)};')
rot = [(2*c - x, 2*c - y) for x, y in pts]
P.append(f'\\draw[green!45!black, very thick, dashed] {path(rot)};')

for i in range(d):
    for j in range(d):
        k = kinds[(i, j)]
        fill = {'+': 'zx_green', '0': 'zx_red', 'Y': 'blue!40'}[k]
        P.append(f'\\draw[black, fill={fill}] ({i},{j}) circle (0.16);')
P.append(f'\\draw[blue, very thick] ({c},{c}) circle (0.40);')
# cartoon continuation: extend the dashed wedge staircases outward
def stair(x0, y0, sx, sy, steps=2):
    """staircase continuation from (x0,y0), horizontal first."""
    pts, x, y = [(x0, y0)], x0, y0
    for _ in range(steps):
        x += sx; pts.append((x, y))
        y += sy; pts.append((x, y))
    return pts
def draw_open(ps):
    seg = ' -- '.join(f'({x:.2f},{y:.2f})' for x, y in ps)
    P.append(f'\\draw[green!45!black, very thick, dashed] {seg};')
dlt = 0.42
# upper wedge: top-right and top-left corners
tr = (2*c - 1 + dlt, 2*c + dlt)
tl = (-dlt, 2*c + dlt)
draw_open(stair(*tr, 1, 1))
draw_open(stair(*tl, -1, 1))
# lower wedge = 180-degree rotation
br = (2*c - tl[0], 2*c - tl[1])
bl = (2*c - tr[0], 2*c - tr[1])
draw_open(stair(*br, 1, -1))
draw_open(stair(*bl, -1, -1))
# small continuation dots at the open ends
for (x0, y0), (sx, sy) in [(tr, (1, 1)), (tl, (-1, 1)),
                           (br, (1, -1)), (bl, (-1, -1))]:
    ex, ey = x0 + 2*sx + 0.7*sx, y0 + 2*sy + 0.7*sy
    rot = 45 if sx*sy > 0 else -45
    P.append(f'\\node[draw=none,fill=none,scale=0.8,'
             f'text=green!45!black, rotate={rot}] at '
             f'({ex:.2f},{ey:.2f}) {{$\\cdots$}};')

# quadrant labels, outside the lattice
P.append(f'\\node[draw=none,fill=none,scale=1.0,text=green!45!black] at '
         f'({c},{2*c+1.0}) {{$\\ket{{+}}$}};')
P.append(f'\\node[draw=none,fill=none,scale=1.0,text=green!45!black] at '
         f'({c},{-1.0}) {{$\\ket{{+}}$}};')
P.append(f'\\node[draw=none,fill=none,scale=1.0] at '
         f'({2*c+1.2},{c}) {{$\\ket{{0}}$}};')
P.append(f'\\node[draw=none,fill=none,scale=1.0] at '
         f'({-1.2},{c}) {{$\\ket{{0}}$}};')
P.append(r'\end{tikzpicture}')
open(os.path.join(OUT, 'fig_pinwheel_d9.tex'), 'w').write(
    '\n'.join(P) + '\n')
nplus = sum(1 for v in kinds.values() if v == '+')
print(f'd=9 pattern: {nplus} |+> qubits, {d*d-1-nplus} |0>, 1 |Y>; '
      'fig_pinwheel_d9.tex written')
