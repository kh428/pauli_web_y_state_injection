"""Full reconstruction of Lao-Criger's circuit-level argument with the
web DEM. Their schedule (read off their fig 4b/c, = uniform Tomita-Svore
pattern with absolute slots): X-checks visit NE,NW,SE,SW; Z-checks visit
NE,SE,NW,SW; weight-2 checks keep their absolute time slots. Their
counting criterion: a single fault in the PREPARATION (init + 2 measured
rounds) is malignant iff it triggers no detector at all -- including
comparisons against FUTURE rounds (downstream QEC) -- and flips the
logical correlator. We append 2 extra rounds so late faults meet their
future detectors, and count only preparation-round faults."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gates import GateBuilder
from physical import rect, region_checks, injection_kind
from schemes import spiral_kinds
from patterns import Study, rep_pattern, edge_of
from iter_li_counting import SYMP, FLIP, P2

def lc_sched(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NE, NW, SE, SW] if typ == 'X' else [NE, SE, NW, SW])
    ss = set(sup)
    return [c if c in ss else None for c in order]

def build(kinds, d, sched_fn, prep_rounds=2, extra_rounds=2):
    b = GateBuilder()
    b.init_cells(kinds)
    P = rect(0, 0, d, d)
    xs, zs = region_checks(P)
    schedules = ([('X', pos, sched_fn(pos, sup, 'X')) for pos, sup in xs] +
                 [('Z', pos, sched_fn(pos, sup, 'Z')) for pos, sup in zs])
    tags = []
    for r in range(prep_rounds + extra_rounds):
        tag = f'p{r}' if r < prep_rounds else f'x{r}'
        tags.append(tag)
        b.gate_round(P, tag, schedules=[(t, p, list(s)) for t, p, s in schedules])
    b.open_outputs(P)
    prep_tags = set(tags[:prep_rounds])
    return b.finish() + (prep_tags,)

def count(kinds, d, sched_fn, label, verbose=False):
    g, meta, prep_tags = build(kinds, d, sched_fn)
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
    def tag_of(role, pfx):
        return role[len(pfx):]
    n2 = 0
    for v in sorted(g.vertices(), key=lambda v: g.vdata(v, 'z', 0.0)):
        rv = meta[v]['role']
        pfx = 'actrl' if rv.startswith('actrl') else (
              'atgt' if rv.startswith('atgt') else None)
        if pfx is None or tag_of(rv, pfx) not in prep_tags:
            continue
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
            n2 += len(mal)
            if verbose:
                _, typ, pos = meta[v]['cell']
                # convert to LC's (P_C, P_T) control/target convention
                if typ == 'Z':   # data is control
                    conv = [(Pt, Pa) for Pa, Pt in mal]
                else:            # ancilla is control
                    conv = list(mal)
                z = g.vdata(v, 'z', 0.0)
                print(f'  {rv:9s} {typ}{pos} data {meta[t]["cell"]} '
                      f't={z}: {len(mal)} classes (C,T)={conv}')
    nI = 0
    for v in g.vertices():
        rv = meta[v]['role']
        if rv in FLIP:
            e = tuple(sorted((v, next(iter(g.neighbors(v))))))
            syn, log = syn_log(e, FLIP[rv])
            if syn == zero and log:
                nI += 1
                if verbose: print(f'  init flip {meta[v]["cell"]} ({FLIP[rv]})')
    nM = 0
    for v in g.vertices():
        rv = meta[v]['role']
        if not rv.startswith('ancmeas'): continue
        if rv.split('_')[-1][1:] not in [t[1:] for t in prep_tags]: pass
        tag = rv[len('ancmeas_X'):]
        if tag not in prep_tags: continue
        e = tuple(sorted((v, next(iter(g.neighbors(v))))))
        Pf = 'Z' if '_X' in rv else 'X'
        syn, log = syn_log(e, Pf)
        if syn == zero and log: nM += 1
    print(f'{label}: n2={n2} -> ({n2}/15) p2 + {nI} pI + {nM} pM')
    return n2, nI, nM

d = 3
corner = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
central = spiral_kinds(0, 0, d)
print('=== LC schedule (their fig 4), downstream-decoded criterion ===')
count(corner, d, lc_sched, 'CR  d=3', verbose=True)
count(central, d, lc_sched, 'MR  d=3', verbose=True)

def nz_sched(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NW, SW, NE, SE] if typ == 'X' else [NW, NE, SW, SE])
    ss = set(sup)
    return [c if c in ss else None for c in order]

print('=== our N/Z schedule, same corrected criterion ===')
count(corner, d, nz_sched, 'corner  d=3')
count(central, d, nz_sched, 'central d=3')
print('=== LC schedule at d=5 (their d-independence claim) ===')
d = 5
count({c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}, d, lc_sched, 'CR  d=5')
count(spiral_kinds(0, 0, d), d, lc_sched, 'MR  d=5')
