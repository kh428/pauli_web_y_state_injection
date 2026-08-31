"""Exact frame functionals for the |T> experiment, from the webs: with
the injected site an OPEN INPUT, the Xbar (Zbar) correlator web relates
the site input X (Z) to the final logical representative and a set of
measurement outcomes -- its decorated measurement caps. Those rec sets
are the frame functions alpha, beta: exact, state- and support-
independent. Emitted as JSON keyed to the stim builder's (round, check)
record layout (checks in face-scan order, tags r, v0, x1, x2)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gates import GateBuilder
from physical import rect, region_checks
from schemes import spiral_kinds
from patterns import Study, edge_of

d = 5 if len(sys.argv) > 1 and sys.argv[1] == '5' else 3

def opt_sched(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NW, NE, SE, SW] if typ == 'X' else [NE, SE, NW, SW])
    ss = set(sup)
    return [c if c in ss else None for c in order]

kinds = spiral_kinds(0, 0, d)
site = next(c for c, k in kinds.items() if k == 'Y')
TAGS = ['r', 'v0', 'x1', 'x2']

b = GateBuilder()
b.open_inputs([site])
b.init_cells({c: k for c, k in kinds.items() if c != site})
P = rect(0, 0, d, d)
xs, zs = region_checks(P)
schedules = ([('X', pos, opt_sched(pos, sup, 'X')) for pos, sup in xs] +
             [('Z', pos, opt_sched(pos, sup, 'Z')) for pos, sup in zs])
schedules.sort(key=lambda t: (t[1][0], t[1][1]))
check_pos = [(t, p) for t, p, s in schedules]
for tag in TAGS:
    b.gate_round(P, tag, schedules=[(t, p, list(s)) for t, p, s in schedules])
b.open_outputs(P)
g, meta = b.finish()

W = Study(g)
outs = dict(edge_of(g, meta, 'out'))
site_in = next(e for c, e in edge_of(g, meta, 'in') if tuple(c) == site)

def solve_frame(pauli, rep_cells):
    f = {}
    for c, e in outs.items():
        W.pin_edge(f, e, pauli if tuple(c) in rep_cells else 'I')
    W.pin_edge(f, site_in, pauli)
    sol = W.solve(f)
    assert sol is not None and sol[0] is not None, f'no {pauli} correlator'
    vec = sol[0]
    recs = []
    for v in g.vertices():
        rv = meta[v]['role']
        if not rv.startswith('ancmeas'): continue
        typ = 'X' if rv.startswith('ancmeas_X') else 'Z'
        tag = rv[len('ancmeas_X'):]
        e = tuple(sorted((v, next(iter(g.neighbors(v))))))
        xc, zc = W.cols(e)
        bx = int(vec[xc // 64] >> np.uint64(xc % 64)) & 1
        bz = int(vec[zc // 64] >> np.uint64(zc % 64)) & 1
        flip = bz if typ == 'X' else bx      # MX flipped by Z, M by X
        if flip:
            _, ctyp, cpos = meta[v]['cell']
            ci = check_pos.index((ctyp, tuple(cpos)))
            rd = TAGS.index(tag)
            recs.append(rd * len(check_pos) + ci)
    return sorted(recs)

col = {(site[0], r) for r in range(d)}
row = {(c, site[1]) for c in range(d)}
alpha = solve_frame('X', col)
beta = solve_frame('Z', row)

def solve_stab(k):
    typ, pos = check_pos[k]
    sup = next(s for t, p, s in schedules if (t, p) == (typ, pos))
    cells = {tuple(c) for c in sup if c is not None}
    f = {}
    for c, e in outs.items():
        W.pin_edge(f, e, typ if tuple(c) in cells else 'I')
    W.pin_edge(f, site_in, 'I')     # pure record relation, no site leg
    sol = W.solve(f)
    assert sol is not None and sol[0] is not None, f'no stab web {k}'
    vec = sol[0]
    recs = []
    for v in g.vertices():
        rv = meta[v]['role']
        if not rv.startswith('ancmeas'): continue
        ctyp = 'X' if rv.startswith('ancmeas_X') else 'Z'
        tag = rv[len('ancmeas_X'):]
        e = tuple(sorted((v, next(iter(g.neighbors(v))))))
        xc, zc = W.cols(e)
        bx = int(vec[xc // 64] >> np.uint64(xc % 64)) & 1
        bz = int(vec[zc // 64] >> np.uint64(zc % 64)) & 1
        if (bz if ctyp == 'X' else bx):
            _, cct, ccp = meta[v]['cell']
            ci = check_pos.index((cct, tuple(ccp)))
            rd = TAGS.index(tag)
            recs.append(rd * len(check_pos) + ci)
    return sorted(recs)

stabsets = [solve_stab(k) for k in range(len(check_pos))]
out = {'d': d, 'alpha': alpha, 'beta': beta, 'stabsets': stabsets}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                    'circuit_noise', f't_frames_d{d}.json')
json.dump(out, open(path, 'w'))
print(f'd={d}: alpha recs {alpha}')
print(f'd={d}: beta  recs {beta}')
