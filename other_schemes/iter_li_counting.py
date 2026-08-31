"""Li-style malignant fault-class counting from the web DEM, for the
Lao-Criger corner and central schemes at gate level (N/Z schedule, one
post-selected verification round). Classes:
  CNOT: 15 two-qubit Pauli classes per gate, weight p2/15, applied as the
        correlated pair (Pa on the ancilla-out edge, Pt on the data-out);
  init: one orthogonal-flip class per data qubit, weight pI;
  meas: one outcome-flip class per ancilla cap, weight pM.
A class is malignant if its combined syndrome is zero and it flips the
logical Y correlator. Output: pL ~ (n2/15) p2 + nI pI + nM pM [+ 2p1/3]."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gates import GateBuilder
from physical import rect, injection_kind
from schemes import spiral_kinds
from patterns import Study, rep_pattern, edge_of

SYMP = {'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
FLIP = {'init_0': 'X', 'init_+': 'Z', 'init_Y': 'X'}
P2 = ['I', 'X', 'Y', 'Z']

def build(kinds, d, verify=1):
    b = GateBuilder()
    b.init_cells(kinds)
    P = rect(0, 0, d, d)
    b.gate_round(P, 'r')
    for r in range(verify):
        b.gate_round(P, f'v{r}')
    b.open_outputs(P)
    return b.finish()

def count(kinds, d):
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
        if not cands:
            return None
        w = min(cands, key=lambda w: g.vdata(w, 'z', 0.0))
        return tuple(sorted((v, w)))

    n2 = 0
    for v in g.vertices():
        rv = meta[v]['role']
        if not (rv.startswith('actrl') or rv.startswith('atgt')):
            continue
        dpartner = [w for w in g.neighbors(v)
                    if meta[w]['role'].startswith('dtap')]
        if not dpartner:
            continue
        t = dpartner[0]
        ea, et = out_edge(v), out_edge(t)
        if ea is None or et is None:
            continue
        for Pa in P2:
            for Pt in P2:
                if Pa == 'I' and Pt == 'I':
                    continue
                syn = zero
                log = 0
                if Pa != 'I':
                    s, l = syn_log(ea, Pa)
                    syn = tuple(a ^ b for a, b in zip(syn, s)); log ^= l
                if Pt != 'I':
                    s, l = syn_log(et, Pt)
                    syn = tuple(a ^ b for a, b in zip(syn, s)); log ^= l
                if syn == zero and log:
                    n2 += 1
    nI = 0
    for v in g.vertices():
        rv = meta[v]['role']
        if rv in FLIP:
            e = tuple(sorted((v, next(iter(g.neighbors(v))))))
            syn, log = syn_log(e, FLIP[rv])
            if syn == zero and log:
                nI += 1
    nM = 0
    for v in g.vertices():
        rv = meta[v]['role']
        if rv.startswith('ancmeas'):
            e = tuple(sorted((v, next(iter(g.neighbors(v))))))
            Pf = 'Z' if '_X' in rv else 'X'
            syn, log = syn_log(e, Pf)
            if syn == zero and log:
                nM += 1
    return n2, nI, nM

for d in (3, 5):
    lc = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
    n2, nI, nM = count(lc, d)
    print(f'corner  d={d}: pL ~ ({n2}/15) p2 + {nI} pI + {nM} pM + (2/3) p1'
          f'   [n2/15 = {n2/15:.2f}]')
    n2, nI, nM = count(spiral_kinds(0, 0, d), d)
    print(f'central d={d}: pL ~ ({n2}/15) p2 + {nI} pI + {nM} pM + (2/3) p1'
          f'   [n2/15 = {n2/15:.2f}]')
