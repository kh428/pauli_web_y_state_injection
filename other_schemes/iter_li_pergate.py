"""Per-gate malignant class breakdown (N/Z schedule, one verify round):
which CNOTs, inits and readouts carry malignant classes, and how many
each. analyse() keeps the graph and the fault edges so the malignant
locations can be drawn; pergate() is the summary wrapper."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from iter_li_counting import build, SYMP, FLIP, P2
from physical import rect, injection_kind
from patterns import Study, rep_pattern, edge_of


def analyse(kinds, d):
    g, meta = build(kinds, d)
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

    cnot = {}
    for v in sorted(g.vertices(), key=lambda v: g.vdata(v, 'z', 0.0)):
        rv = meta[v]['role']
        if not (rv.startswith('actrl') or rv.startswith('atgt')): continue
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
                    s, l = syn_log(ea, Pa)
                    syn = tuple(a^b for a,b in zip(syn,s)); log ^= l
                if Pt != 'I':
                    s, l = syn_log(et, Pt)
                    syn = tuple(a^b for a,b in zip(syn,s)); log ^= l
                if syn == zero and log: mal.append((Pa, Pt))
        if mal:
            _, typ, pos = meta[v]['cell']
            tag = rv.replace('actrl', '').replace('atgt', '')
            cnot[(tag, typ, pos, meta[t]['cell'])] = {
                'classes': mal, 'ea': ea, 'et': et}
    init = []
    for v in g.vertices():
        rv = meta[v]['role']
        if rv in FLIP:
            e = tuple(sorted((v, next(iter(g.neighbors(v))))))
            syn, log = syn_log(e, FLIP[rv])
            if syn == zero and log:
                init.append((meta[v]['cell'], e))
    return g, meta, cnot, init


def pergate(kinds, d):
    _, _, cnot, init = analyse(kinds, d)
    return ({k: v['classes'] for k, v in cnot.items()},
            [c for c, _ in init])


if __name__ == '__main__':
    d = 3
    kinds = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
    cnot, init = pergate(kinds, d)
    for k, mal in cnot.items():
        print(k, len(mal), mal)
    print('malignant inits:', init)
    print('total n2 =', sum(len(m) for m in cnot.values()))
