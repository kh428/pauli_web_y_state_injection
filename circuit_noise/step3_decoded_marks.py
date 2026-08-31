import os
"""Step 3: regenerate the mal-spacetime figures (fig 22 style) and the
flat-circuit violet marks under the DECODED convention. Classification
runs on the 4-round graph; drawing on the 2-round graph (identical
prep-portion vertex ids by construction). Output tex goes to THIS
folder for review, not to version_5_v1."""
import sys, os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'figures_3d', 'src')
sys.path.insert(0, SRC)
import numpy as np
from gates import GateBuilder
from physical import rect, region_checks, injection_kind
from schemes import spiral_kinds
from patterns import Study, rep_pattern, edge_of
from iter_li_counting import SYMP, FLIP, P2
from iter_lc_reconstruct import lc_sched
from paper3d import emit

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_n(kinds, d, sched_fn, nrounds):
    b = GateBuilder()
    b.init_cells(kinds)
    P = rect(0, 0, d, d)
    xs, zs = region_checks(P)
    schedules = ([('X', pos, sched_fn(pos, sup, 'X')) for pos, sup in xs] +
                 [('Z', pos, sched_fn(pos, sup, 'Z')) for pos, sup in zs])
    schedules.sort(key=lambda t: (t[1][0], t[1][1]))
    tags = []
    for r in range(nrounds):
        tag = f'p{r}' if r < 2 else f'x{r}'
        tags.append(tag)
        b.gate_round(P, tag,
                     schedules=[(t, p, list(s)) for t, p, s in schedules])
    b.open_outputs(P)
    return b.finish()

def decoded_mal(kinds, d, sched_fn):
    g, meta = build_n(kinds, d, sched_fn, 4)
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
    def syn_log(e, P):
        px, pz = SYMP[P]
        syn = tuple((px*wz + pz*wx) % 2
                    for wx, wz in (bits(x, e) for x in vecs))
        cw = bits(cv, e)
        return syn, (px*cw[1] + pz*cw[0]) % 2
    zero = tuple(0 for _ in vecs)
    def out_edge(v):
        z = g.vdata(v, 'z', 0.0)
        cands = [w for w in g.neighbors(v)
                 if g.vdata(w, 'z', 0.0) > z
                 and meta[w].get('cell') == meta[v].get('cell')]
        if not cands: return None
        w = min(cands, key=lambda w: g.vdata(w, 'z', 0.0))
        return tuple(sorted((v, w)))
    mal_taps = {}          # tap vertex -> which sides malignant (a/t)
    n2 = 0
    for v in sorted(g.vertices(), key=lambda v: g.vdata(v, 'z', 0.0)):
        rv = meta[v]['role']
        pfx = 'actrl' if rv.startswith('actrl') else (
              'atgt' if rv.startswith('atgt') else None)
        if pfx is None or rv[len(pfx):] not in ('p0', 'p1'):
            continue
        dp = [w for w in g.neighbors(v)
              if meta[w]['role'].startswith('dtap')]
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
                    syn = tuple(a ^ b for a, b in zip(syn, s_)); log ^= l
                if Pt != 'I':
                    s_, l = syn_log(et, Pt)
                    syn = tuple(a ^ b for a, b in zip(syn, s_)); log ^= l
                if syn == zero and log:
                    mal.append((Pa, Pt))
        if mal:
            n2 += len(mal)
            mal_taps[v] = ({p for p, _ in mal} - {'I'},
                           {p for _, p in mal} - {'I'}, t)
    # malignant init flips
    mal_init = []
    for v in g.vertices():
        rv = meta[v]['role']
        if rv in FLIP:
            e = tuple(sorted((v, next(iter(g.neighbors(v))))))
            syn, log = syn_log(e, FLIP[rv])
            if not any(syn) and log:
                mal_init.append(v)
    return g, meta, mal_taps, mal_init, n2

def draw(kinds, d, sched_fn, label, fname):
    g4, meta4, mal_taps, mal_init4, n2 = decoded_mal(kinds, d, sched_fn)
    g2, meta2 = build_n(kinds, d, sched_fn, 2)
    def out_edge2(v):
        z = g2.vdata(v, 'z', 0.0)
        cands = [w for w in g2.neighbors(v)
                 if g2.vdata(w, 'z', 0.0) > z
                 and meta2[w].get('cell') == meta2[v].get('cell')]
        if not cands: return None
        return tuple(sorted((v, min(cands,
                    key=lambda w: g2.vdata(w, 'z', 0.0)))))
    marks = {}
    for v, (pa, pt, t) in mal_taps.items():
        assert meta2[v]['role'] == meta4[v]['role'], 'vertex map broke'
        if pa:
            e = out_edge2(v)
            if e: marks[e] = 'M'
        if pt:
            e = out_edge2(t)
            if e: marks[e] = 'M'
    for v in mal_init4:
        e = tuple(sorted((v, next(iter(g2.neighbors(v))))))
        marks[e] = 'M'
    reps = {}
    for v in g2.vertices():
        r = meta2[v]['role']
        if r.startswith('ancinit_'):
            reps.setdefault(r[-2:], v)
    emit(g2, [marks], os.path.join(OUT, fname), zscale=2.2,
         node_size='0.4cm', planes=list(reps.values()))
    print(f'{label}: n2={n2}, {len(marks)} violet segments, '
          f'{len(mal_init4)} init flips -> {fname}')

d = 3
ck = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
sk = dict(spiral_kinds(0, 0, d))
draw(ck, d, lc_sched, 'corner/TS', 'fig_mal_decoded_corner.tex')
draw(sk, d, lc_sched, 'central/TS', 'fig_mal_decoded_central.tex')
