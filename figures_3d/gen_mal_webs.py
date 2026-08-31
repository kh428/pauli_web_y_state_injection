"""Why a segment is violet, at web level: two examples on the corner
d=3 spacetime. (a) a detected round-2 fault, drawn with the detector
web that catches it; (b) a malignant round-2 fault, drawn with the
logical Y correlator web it flips (no detector web reaches it)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from iter_li_counting import build, SYMP
from physical import rect, injection_kind
from patterns import Study, rep_pattern, edge_of
from schemes import decode
from paper3d import emit

OUT = os.path.dirname(os.path.abspath(__file__))
d = 3
kinds = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
g, meta = build(kinds, d)
W = Study(g)
outs = dict(edge_of(g, meta, 'out'))
f0 = {}
for e in outs.values():
    W.pin_edge(f0, e, 'I')
part, basis = W.solve(f0)
vecs = ([part] if np.any(part) else []) + list(basis)
fc = {}
pat = rep_pattern('Y', d)
for c, e in outs.items():
    W.pin_edge(fc, e, pat.get(c, 'I'))
cv = W.solve(fc)[0]

def bits(vec, e):
    xc, zc = W.cols(e)
    return (int(vec[xc // 64] >> np.uint64(xc % 64)) & 1,
            int(vec[zc // 64] >> np.uint64(zc % 64)) & 1)

def overlap(vec, e, P):
    px, pz = SYMP[P]
    wx, wz = bits(vec, e)
    return (px * wz + pz * wx) % 2

def out_edge(v):
    z = g.vdata(v, 'z', 0.0)
    cands = [w for w in g.neighbors(v)
             if g.vdata(w, 'z', 0.0) > z
             and meta[w].get('cell') == meta[v].get('cell')]
    if not cands:
        return None
    w = min(cands, key=lambda w: g.vdata(w, 'z', 0.0))
    return tuple(sorted((v, w)))

# collect round-2 CNOT fault locations
r2 = []
for v in sorted(g.vertices(), key=lambda v: g.vdata(v, 'z', 0.0)):
    rv = meta[v]['role']
    if not (rv.startswith('actrl') or rv.startswith('atgt')):
        continue
    if not rv.endswith('v0'):
        continue
    dp = [w for w in g.neighbors(v) if meta[w]['role'].startswith('dtap')]
    if not dp:
        continue
    ea, et = out_edge(v), out_edge(dp[0])
    if ea and et:
        r2.append((v, dp[0], ea, et))

def classify(e, P):
    syn = tuple(overlap(x, e, P) for x in vecs)
    return syn, overlap(cv, e, P)

# (a) detected: a round-2 fault with nonzero syndrome; draw one firing web
det = None
for v, t, ea, et in r2:
    for P in 'XZY':
        syn, log = classify(ea, P)
        if any(syn):
            wi = syn.index(1)
            det = (ea, P, wi)
            break
    if det:
        break
ea, P, wi = det
web_a = decode(vecs[wi], W)

# (b) malignant: zero syndrome, flips the correlator
mal = None
for v, t, ea2, et2 in r2:
    for e in (ea2, et2):
        for Pm in 'XZY':
            syn, log = classify(e, Pm)
            if not any(syn) and log:
                mal = (e, Pm)
                break
        if mal: break
    if mal: break
e_m, P_m = mal

def annotate(path, edge, label):
    a, b = edge
    t = open(path).read()
    t = t.replace('\\end{tikzpicture}',
        f'\\coordinate (fm) at ($(v{a})!0.5!(v{b})$);\n'
        f'\\draw[->,violet,line width=0.06cm] '
        f'($(fm)+(-2.6,1.7)$) -- ($(fm)+(-0.2,0.12)$);\n'
        f'\\node[draw=none,fill=none,text=violet,scale=2.6] at '
        f'($(fm)+(-3.2,2.05)$) {{${label}$}};\n'
        '\\end{tikzpicture}')
    open(path, 'w').write(t)

reps = {}
for v in g.vertices():
    r = meta[v]['role']
    if r.startswith('ancinit_'):
        reps.setdefault('v0' if r.endswith('v0') else 'r', v)
planes = list(reps.values())

pa = os.path.join(OUT, 'fig_malweb_detected.tex')
emit(g, [dict(web_a)], pa, zscale=2.2, node_size='0.4cm', planes=planes)
annotate(pa, ea, P)
print(f'(a) detected example: {P} on {ea}, caught by web {wi}')

# nearest X-decorated detector web to the malignant fault
def mid(e):
    pts = []
    for v in e:
        pts.append((g.row(v), g.qubit(v), g.vdata(v, 'z', 0.0)))
    return tuple(sum(c) / 2 for c in zip(*pts))
mx, my, mz = mid(e_m)
best = None
for k, vec in enumerate(vecs):
    web = decode(vec, W)
    ds = [sum((a - b) ** 2 for a, b in zip(mid(e), (mx, my, mz))) ** 0.5
          for e, p in web.items() if p in ('X', 'Y')]
    if not ds:
        continue
    dmin = min(ds)
    if best is None or dmin < best[0]:
        best = (dmin, k, web)
dmin, kb, web_b = best
assert overlap(vecs[kb], e_m, P_m) == 0
pb = os.path.join(OUT, 'fig_malweb_malignant.tex')
emit(g, [dict(web_b)], pb, zscale=2.2, node_size='0.4cm', planes=planes)
annotate(pb, e_m, P_m)
print(f'(b) malignant: {P_m} on {e_m}; nearest X-decorated web is #{kb} '
      f'at distance {dmin:.2f}, still overlap 0')

# (c) the web that WOULD catch it: allow webs terminating on the open
# outputs (the comparison against a next round, which is not drawn)
Wf = Study(g)
part_f, basis_f = Wf.solve({})
outset = set(dict(edge_of(g, meta, 'out')).values())
cands = []
for vec in list(basis_f):
    if overlap(vec, e_m, P_m) % 2 == 1:
        web = decode(vec, Wf)
        if any(e in outset for e in web):
            cands.append((len(web), web))
assert cands, 'no output-terminating catching web found'
cands.sort(key=lambda t: t[0])
# greedy: XOR with closed detector webs while the weight drops
best_vec = None
for vec in list(basis_f):
    if overlap(vec, e_m, P_m) % 2 == 1:
        w = decode(vec, Wf)
        if any(e in outset for e in w):
            if best_vec is None or len(w) < len(decode(best_vec, Wf)):
                best_vec = vec.copy()
improved = True
while improved:
    improved = False
    for dv in list(vecs) + list(basis_f):
        trial = best_vec ^ dv
        if overlap(trial, e_m, P_m) % 2 != 1:
            continue
        wt = decode(trial, Wf)
        if any(e in outset for e in wt) and \
                len(wt) < len(decode(best_vec, Wf)):
            best_vec = trial
            improved = True
web_c = decode(best_vec, Wf)
assert overlap(best_vec, e_m, P_m) % 2 == 1
pc = os.path.join(OUT, 'fig_malweb_nextround.tex')
emit(g, [dict(web_c)], pc, zscale=2.2, node_size='0.4cm', planes=planes)
annotate(pc, e_m, P_m)
print(f'(c) would-be web: {len(web_c)} decorated edges, reaches the '
      'open outputs')
