"""Generate the d=5 web figures for appendices B-F."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from schemes import Builder, rect, decode, sc_unitary
from patterns import Study, rep_pattern, edge_of
from iter_d5_schemes import (build_zz, build_hook, build_transversal,
                             build_deformation, D)
from paper3d import emit

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'arXiv-2501.15566v5_draft_post_LC_read')

def correlator(g, meta):
    W = Study(g)
    outs = dict(edge_of(g, meta, 'out'))
    ins = dict(edge_of(g, meta, 'in'))
    f = {}
    pat = rep_pattern('Y', D)
    for c, e in outs.items():
        W.pin_edge(f, e, pat.get(c, 'I'))
    for e in ins.values():
        W.pin_edge(f, e, 'I')
    sol = W.solve(f)
    return decode(sol[0], W)

g, meta = build_zz()
lift = {}
for v in g.vertices():
    if meta[v]['role'] == 'gad_czz':
        lift[v] = 0.35
    elif meta[v]['role'] == 'gad_lzz':
        lift[v] = 0.75
swapv = {v for v in g.vertices() if meta[v]['role'] == 'gad_lzz'}
emit(g, [correlator(g, meta)], os.path.join(OUT, 'fig_zz_d5_web.tex'),
     lift=lift, swap_y=swapv)
g, meta = build_hook()
hlift = {}
for v in g.vertices():
    if meta[v]['role'] == 'gad_chk':
        hlift[v] = -0.5
    elif meta[v]['role'] == 'gad_lhk':
        hlift[v] = -0.5
emit(g, [correlator(g, meta)], os.path.join(OUT, 'fig_hook_d5_web.tex'),
     lift=hlift)

# transversal: ONE X-type and ONE Z-type round-parity cube, well separated
g, meta = build_transversal(2)
W = Study(g)
outs = dict(edge_of(g, meta, 'out'))
ins = dict(edge_of(g, meta, 'in'))
f0 = {}
for e in outs.values(): W.pin_edge(f0, e, 'I')
for e in ins.values(): W.pin_edge(f0, e, 'I')
hubs = {}
for v in g.vertices():
    r = meta[v]['role']
    if r.startswith('chk_'):
        hubs.setdefault((r[4], round(g.row(v), 1), round(g.qubit(v), 1)),
                        {})[r[-2:]] = v
webs = []
for typ, px, py in (('x', 1.5, 1.5), ('z', 2.5, 3.5)):
    pair = hubs[(typ, px + 0.0, py + 0.0)] if (typ, px, py) in hubs else None
    key = min((k for k in hubs if k[0] == typ),
              key=lambda k: (k[1]-px)**2 + (k[2]-py)**2)
    pair = hubs[key]
    f = dict(f0)
    for k2, d2 in hubs.items():
        for rd, hv in d2.items():
            if hv == pair['r0']:
                f[W.lam[hv]] = 1
            elif hv != pair.get('r1'):
                f[W.lam[hv]] = 0
    sol = W.solve(f)
    webs.append(decode(sol[0], W))
emit(g, webs, os.path.join(OUT, 'fig_transversal_d5_webs.tex'))

# unitary cone: correlator through the cone (input pinned Y, outputs rep(Y))
def unitary_open(d):
    from schemes import Builder
    from physical import rect as R
    b = Builder()
    P = R(0, 0, d, d)
    kinds = {}
    site = (0, 0)
    for c in P:
        if c == site: continue
        kinds[c] = '0' if c[0] == 0 else ('+' if c[1] == 0 else '0')
    b.open_inputs([site])
    b.init_cells(kinds)
    for i in range(d - 1):
        if i == 0:
            # both layer-0 gates touch the site; serialise them so the
            # drawing keeps one spider per point
            b.cnot_layer([((0, 0), (0, 1))], '0a')
            b.cnot_layer([((1, 0), (0, 0))], '0b')
        else:
            b.cnot_layer([((0, i), (0, i + 1)), ((i + 1, 0), (i, 0))],
                         str(i))
    b.round(P, 'r')
    b.open_outputs(P)
    return b.finish()

g, meta = unitary_open(5)
W = Study(g)
outs = dict(edge_of(g, meta, 'out'))
site_e = next(e for c, e in edge_of(g, meta, 'in'))
f = {}
W.pin_edge(f, site_e, 'Y')
pat = rep_pattern('Y', 5)
for c, e in outs.items():
    W.pin_edge(f, e, pat.get(c, 'I'))
sol = W.solve(f)
emit(g, [decode(sol[0], W)], os.path.join(OUT, 'fig_unitary_d5_web.tex'))

# deformation: forward-only webs of the two Z-plaquettes adjacent to q in
# the re-insertion round
g, meta = build_deformation()
W = Study(g)
ins = dict(edge_of(g, meta, 'in'))
q = (2, 2)
hubs = []
for v in g.vertices():
    if meta[v]['role'] == 'chk_zb':
        if any(meta[w]['cell'] == q for w in g.neighbors(v)):
            hubs.append(v)
webs = []
f0 = {}
for e in ins.values(): W.pin_edge(f0, e, 'I')
for hb in hubs[:2]:
    f = dict(f0)
    allh = [v for v in g.vertices() if meta[v]['role'].startswith('chk_')]
    for k in allh:
        f[W.lam[k]] = 1 if k == hb else 0
    sol = W.solve(f)
    if sol is not None:
        webs.append(decode(sol[0], W))
# the running code's logical Zbar correlator, transported input -> output
fz = {}
patz = rep_pattern('Z', 5)
for c, e in ins.items():
    W.pin_edge(fz, e, patz.get(c, 'I'))
outs_d = dict(edge_of(g, meta, 'out'))
for c, e in outs_d.items():
    W.pin_edge(fz, e, patz.get(c, 'I'))
solz = W.solve(fz)
if solz is not None:
    webs.append(decode(solz[0], W))
print(f'deformation: {len(hubs)} adjacent Z-hubs in round b, {len(webs)} webs drawn '
      f'(incl. Zbar correlator: {solz is not None})')
emit(g, webs, os.path.join(OUT, 'fig_deformation_d5_webs.tex'))
