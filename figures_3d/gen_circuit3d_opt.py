"""3D spacetime ZX diagram of the OPTIMISED central schedule (X-checks
NW,NE,SE,SW; Z-checks NE,SE,NW,SW; absolute slots), with the complete
unprotected set in violet: the malignant CNOT classes' outgoing
segments and the malignant init segment. Census computed on the
extended (downstream-decoded) spacetime; drawing shows the two
post-selected rounds."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gates import GateBuilder
from physical import rect, region_checks
from schemes import spiral_kinds
from patterns import Study, rep_pattern, edge_of
from iter_li_counting import SYMP, FLIP, P2
from paper3d import emit

d = 3

def opt_sched(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NW, NE, SE, SW] if typ == 'X' else [NE, SE, NW, SW])
    ss = set(sup)
    return [c if c in ss else None for c in order]

def build(kinds, rounds, tags):
    b = GateBuilder()
    b.init_cells(kinds)
    P = rect(0, 0, d, d)
    xs, zs = region_checks(P)
    schedules = ([('X', pos, opt_sched(pos, sup, 'X')) for pos, sup in xs] +
                 [('Z', pos, opt_sched(pos, sup, 'Z')) for pos, sup in zs])
    # colliding same-slot taps are serialised in face-scan order, exactly
    # as in the stim builder (circuits.py rect_checks ordering)
    schedules.sort(key=lambda t: (t[1][0], t[1][1]))
    for tag in tags[:rounds]:
        b.gate_round(P, tag, schedules=[(t, p, list(s)) for t, p, s in schedules])
    b.open_outputs(P)
    return b.finish()

kinds = spiral_kinds(0, 0, d)
# census on the extended spacetime
g, meta = build(kinds, 4, ['r', 'v0', 'x1', 'x2'])
W = Study(g)
outs = dict(edge_of(g, meta, 'out'))
f0 = {}
for e in outs.values(): W.pin_edge(f0, e, 'I')
part, basis = W.solve(f0)
vecs = ([part] if np.any(part) else []) + list(basis)
fc = {}
pat = rep_pattern('Y', d)
for c, e in outs.items(): W.pin_edge(fc, e, pat.get(c, 'I'))
cv = W.solve(fc)[0]
def bits(vec, e):
    xc, zc = W.cols(e)
    return (int(vec[xc//64] >> np.uint64(xc%64)) & 1,
            int(vec[zc//64] >> np.uint64(zc%64)) & 1)
def syn_log(e, P_):
    px, pz = SYMP[P_]
    syn = tuple((px*wz + pz*wx) % 2 for wx, wz in (bits(x, e) for x in vecs))
    cw = bits(cv, e)
    return syn, (px*cw[1] + pz*cw[0]) % 2
zero = tuple(0 for _ in vecs)
def out_edge(v):
    # next tap along the wire: order by (time, insertion id) so that
    # serialised same-tick taps chain correctly
    key = (g.vdata(v, 'z', 0.0), v)
    cands = [w for w in g.neighbors(v)
             if (g.vdata(w, 'z', 0.0), w) > key
             and meta[w].get('cell') == meta[v].get('cell')]
    if not cands: return None
    w = min(cands, key=lambda w: (g.vdata(w, 'z', 0.0), w))
    return tuple(sorted((v, w)))

mal_marks = []       # (tag, check pos+typ, data cell, need_ea, need_et)
for v in sorted(g.vertices(), key=lambda v: g.vdata(v, 'z', 0.0)):
    rv = meta[v]['role']
    if not (rv.startswith('actrl') or rv.startswith('atgt')): continue
    if not (rv.endswith('r') or rv.endswith('v0')): continue
    dp = [w for w in g.neighbors(v) if meta[w]['role'].startswith('dtap')]
    if not dp: continue
    t = dp[0]
    ea, et = out_edge(v), out_edge(t)
    if ea is None or et is None: continue
    mal = []
    for Pa in P2:
        for Pt in P2:
            if Pa == 'I' and Pt == 'I': continue
            syn = zero; log = 0
            if Pa != 'I':
                s_, l = syn_log(ea, Pa)
                syn = tuple(a^b for a,b in zip(syn,s_)); log ^= l
            if Pt != 'I':
                s_, l = syn_log(et, Pt)
                syn = tuple(a^b for a,b in zip(syn,s_)); log ^= l
            if syn == zero and log: mal.append((Pa, Pt))
    if mal:
        _, typ, pos = meta[v]['cell']
        mal_marks.append((rv, typ, tuple(pos), tuple(meta[t]['cell']),
                          any(Pa != 'I' for Pa, _ in mal),
                          any(Pt != 'I' for _, Pt in mal), len(mal)))
mal_init = []
for v in g.vertices():
    rv = meta[v]['role']
    if rv in FLIP:
        e = tuple(sorted((v, next(iter(g.neighbors(v))))))
        syn, log = syn_log(e, FLIP[rv])
        if syn == zero and log:
            mal_init.append(tuple(meta[v]['cell']))
print('malignant gates:', [(m[1], m[2], m[3], m[6]) for m in mal_marks])
print('malignant inits:', mal_init)

# drawing graph: the two post-selected rounds only
g2, meta2 = build(kinds, 2, ['r', 'v0'])
marks = {}
def out_edge2(v):
    key = (g2.vdata(v, 'z', 0.0), v)
    cands = [w for w in g2.neighbors(v)
             if (g2.vdata(w, 'z', 0.0), w) > key
             and meta2[w].get('cell') == meta2[v].get('cell')]
    if not cands: return None
    return tuple(sorted((v, min(cands,
                                key=lambda w: (g2.vdata(w, 'z', 0.0), w)))))
for v in g2.vertices():
    rv = meta2[v]['role']
    if rv.startswith('actrl') or rv.startswith('atgt'):
        dp = [w for w in g2.neighbors(v) if meta2[w]['role'].startswith('dtap')]
        if not dp: continue
        t = dp[0]
        for (mrv, typ, pos, cell, nea, net, n) in mal_marks:
            if (rv == mrv and meta2[v]['cell'][1] == typ and
                    tuple(meta2[v]['cell'][2]) == pos and
                    tuple(meta2[t]['cell']) == cell):
                if nea: marks[out_edge2(v)] = 'M'
                if net: marks[out_edge2(t)] = 'M'
    if rv in FLIP and tuple(meta2[v]['cell']) in mal_init:
        e = tuple(sorted((v, next(iter(g2.neighbors(v))))))
        marks[e] = 'M'
print('violet segments in drawing:', len(marks))
reps = {}
for v in g2.vertices():
    r = meta2[v]['role']
    if r.startswith('ancinit_'):
        reps.setdefault(r[-2:], v)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'arXiv-2501.15566v5_draft_post_LC_read')
emit(g2, [marks], os.path.join(OUT, 'fig_circuit3d_opt.tex'),
     zscale=2.2, node_size='0.4cm', planes=list(reps.values()))
