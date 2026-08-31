"""D2: exact all-orders |T> campaign curve. Probe residual-class bits
(bZ, bX) per CNOT fault class via the '0'- and '+'-site variants of
the campaign circuit (shared detectors, shared propagation);
consistency bZ^bX == |Y>-campaign L flip. Character sums over four
cosets give P(class | accept) exactly:
p_T = [P(Z) + (P(X)+P(Y))/2] / Pacc, p_Y = [P(X)+P(Z)] / Pacc.
Validation: p_Y matches yesterday's exact function; p_T leading
2.5/15; p_T curve vs the billion-shot t_results_final.json data."""
import sys, os, json
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import numpy as np
from fractions import Fraction
import stim as _stim
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central

d = 3
kinds = spiral_kinds(d)
site0 = next(c for c, k in kinds.items() if k == 'Y')
P2 = [(a, b) for a in 'IXYZ' for b in 'IXYZ' if (a, b) != ('I', 'I')]

def campaign(site_kind, undo):
    kk = dict(kinds); kk[site_kind is not None and site0 or site0] = \
        site_kind if site_kind else kk[site0]
    if site_kind: kk[site0] = site_kind
    txt, info = build(d, kk, sched=opt_central, p2=0.0,
                      noisy_rounds=2, extra_rounds=2, tail=False,
                      site_override=site0)
    tail, lw, col_off, row_off = readout_tail(d, kk, info)
    fix = []
    for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
    for o in row_off: fix.append(f'CX rec[{o}] {lw}')
    TAIL = '\n' + tail + '\n' + '\n'.join(fix) + \
        ('\n' + undo + f' {lw}' if undo else '') + f'\nH {lw}\nM {lw}'
    return txt, info, TAIL, lw

# reference: |Y> campaign
txtY, infoY, TAILY, lw = campaign('Y', 'S_DAG')
# variants: |0> site (Z-bar readout: measure Z -> no H... use M direct)
txt0, info0, TAIL0, _ = campaign('0', None)
TAIL0 = TAIL0.replace(f'\nH {lw}\nM {lw}', f'\nM {lw}')
txtP, infoP, TAILP, _ = campaign('+', None)   # X-bar: H M
dets = infoY['dets']
assert [r for _, r in info0['dets']] == [r for _, r in dets] \
    and [r for _, r in infoP['dets']] == [r for _, r in dets], \
    'detector sets differ between variants'

def probe(txt, TAIL, li, gates):
    L = txt.split('\n')
    L.insert(li + 1, gates)
    m = _stim.Circuit('\n'.join(L) + TAIL).compile_sampler() \
        .sample(8).astype('uint8')
    det = np.zeros((8, len(dets)), np.uint8)
    for i, (lab, recs) in enumerate(dets):
        for r in recs:
            det[:, i] ^= m[:, r]
    ob = m[:, -1]
    assert (det == det[0]).all() and (ob == ob[0]).all(), (li, gates)
    return det[0].copy(), int(ob[0])

def baseline(txt, TAIL):
    m = _stim.Circuit(txt + TAIL).compile_sampler().sample(8) \
        .astype('uint8')
    det = np.zeros((8, len(dets)), np.uint8)
    for i, (lab, recs) in enumerate(dets):
        for r in recs:
            det[:, i] ^= m[:, r]
    assert not det.any() and (m[:, -1] == m[0, -1]).all()
    return int(m[0, -1])
c0 = baseline(txt0, TAIL0)
cP = baseline(txtP, TAILP)
cY = baseline(txtY, TAILY)
assert cY == 0

# channels (rounds 0,1), per-variant line indices
def find_channels(t):
    ch = []; seen = {}
    for li, l in enumerate(t.split('\n')):
        ps = l.split()
        if ps and ps[0] == 'CX':
            pair = (int(ps[1]), int(ps[2]))
            k = seen.get(pair, 0); seen[pair] = k + 1
            if k < 2: ch.append((li, pair))
    return ch
chY = find_channels(txtY)
ch0 = find_channels(txt0)
chP = find_channels(txtP)
assert len(chY) == 48 and [p for _, p in ch0] == [p for _, p in chY] \
    and [p for _, p in chP] == [p for _, p in chY]
channels = list(zip(chY, ch0, chP))

BITS = {'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
DY = []; OY = []; OZb = []; OXb = []
for (liY, pair), (li0, _), (liP, _) in channels:
    for (Pa, Pt) in P2:
        ops = []
        if Pa != 'I': ops.append(f'{Pa} {pair[0]}')
        if Pt != 'I': ops.append(f'{Pt} {pair[1]}')
        g = '\n'.join(ops)
        dv, ov = probe(txtY, TAILY, liY, g)
        dv0, o0 = probe(txt0, TAIL0, li0, g)
        dvP, oP = probe(txtP, TAILP, liP, g)
        assert (dv0 == dv).all() and (dvP == dv).all(), 'det mismatch'
        bZ = o0 ^ c0        # residual anticommutes with Zbar readout
        bX = oP ^ cP
        assert (bZ ^ bX) == ov, 'type bits inconsistent with Y flip'
        DY.append(dv); OZb.append(bZ); OXb.append(bX)
DY = np.array(DY, np.uint8)
OZb = np.array(OZb, np.uint8); OXb = np.array(OXb, np.uint8)
print('720 classes probed; type bits consistent with |Y> flips',
      flush=True)

# span + four-coset weight enumerators (CNOT-only: all 48 channels)
F = DY.T
nzr = [i for i in range(F.shape[0]) if F[i].any()]
basis_rows, piv = [], {}
for r in F[nzr]:
    r = r.copy()
    for c, br in piv.items():
        if r[c]: r ^= br
    w = np.nonzero(r)[0]
    if len(w): piv[w[0]] = r; basis_rows.append(r)
rk = len(basis_rows)
print(f'rank {rk}', flush=True)
Bm = np.array(basis_rows, np.uint8)
N = 1 << rk
idx = np.arange(N, dtype=np.uint32)
PAR = np.zeros(1 << 10, np.uint8)
for i in range(1, 1 << 10):
    PAR[i] = PAR[i >> 1] ^ (i & 1)
def par_mask(mask):
    return PAR[idx & (mask & 0x3FF)] ^ \
        PAR[(idx >> 10) & ((mask >> 10) & 0x3FF)]
nbits = 192
bit_ch = np.repeat(np.arange(48), 4)      # 4 bits per channel? NO:
# bits here are per-class? We probed per CLASS (15/channel), forms are
# per-bit needed. Rebuild per-bit forms from the 4 basis classes.
BASIS4 = [('X','I'), ('Z','I'), ('I','X'), ('I','Z')]
row_of = {cls: i for i, cls in enumerate(P2)}
Dbit = []; Zbit = []; Xbit = []
for ci in range(48):
    for b in BASIS4:
        k = ci * 15 + row_of[b]
        Dbit.append(DY[k]); Zbit.append(OZb[k]); Xbit.append(OXb[k])
Dbit = np.array(Dbit, np.uint8)
Zbit = np.array(Zbit, np.uint8); Xbit = np.array(Xbit, np.uint8)
# verify linearity for all classes
BITS4 = {'I': (0,0), 'X': (1,0), 'Y': (1,1), 'Z': (0,1)}
bad = 0
for ci in range(48):
    for ki, (Pa, Pt) in enumerate(P2):
        bb = BITS4[Pa] + BITS4[Pt]
        pd = np.zeros(len(dets), np.uint8); pz = px = 0
        for j in range(4):
            if bb[j]:
                pd ^= Dbit[4*ci+j]; pz ^= Zbit[4*ci+j]
                px ^= Xbit[4*ci+j]
        k = ci * 15 + ki
        if not (pd == DY[k]).all() or pz != OZb[k] or px != OXb[k]:
            bad += 1
print(f'linearity check: {bad} bad of 720', flush=True)
assert bad == 0
Fb = Dbit.T
# re-derive span on bit-forms (should equal rk)
def rowreduce(rows):
    br, pv = [], {}
    for r in rows:
        r = r.copy()
        for c, b in pv.items():
            if r[c]: r ^= b
        w = np.nonzero(r)[0]
        if len(w): pv[w[0]] = r; br.append(r)
    return br
brows = rowreduce([Fb[i] for i in range(Fb.shape[0]) if Fb[i].any()])
assert len(brows) == rk
Bm = np.array(brows, np.uint8)
fZ = Zbit.copy(); fX = Xbit.copy()

def wenum(offset):
    W = np.zeros(N, np.int16)
    for ci in range(48):
        bs = [4*ci, 4*ci+1, 4*ci+2, 4*ci+3]
        nzf = np.zeros(N, np.bool_)
        for b in bs:
            mask = 0
            for j in range(rk):
                if Bm[j, b]: mask |= (1 << j)
            par = par_mask(mask)
            if offset[b]: par = par ^ 1
            nzf |= par.astype(bool)
        W += nzf
    u, c = np.unique(W, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))
Wnull = wenum(np.zeros(192, np.uint8))
WZ = wenum(fZ)
WX = wenum(fX)
WY = wenum(fZ ^ fX)
print('four coset enumerators done', flush=True)

def evalW(W, x):
    return sum(n * x**w for w, n in W.items())
def probs(p2):
    x = 1 - Fraction(16, 15) * p2
    E0 = Fraction(evalW(Wnull, x), N)
    EZ = Fraction(evalW(WZ, x), N)
    EX = Fraction(evalW(WX, x), N)
    EY = Fraction(evalW(WY, x), N)
    # P(acc, bZ=a, bX=b) = 1/4 (E0 + (-1)^a EZ + (-1)^b EX + (-1)^(a+b) EY)
    P = {}
    for a in (0, 1):
        for b in (0, 1):
            P[(a, b)] = (E0 + (-1)**a * EZ + (-1)**b * EX
                         + (-1)**(a+b) * EY) / 4
    Pacc = sum(P.values())
    pY = (P[(1, 0)] + P[(0, 1)]) / Pacc
    pT = (P[(0, 1)] + Fraction(1, 2) * (P[(1, 0)] + P[(1, 1)])) / Pacc
    return pY, pT, Pacc
# leading terms
eps = Fraction(1, 10**7)
pY1, pT1, _ = probs(eps)
print(f'leading: pY/p2 = {float(pY1/eps):.6f} (target 0.2), '
      f'pT/p2 = {float(pT1/eps):.6f} (target {2.5/15:.6f})')
# compare pY exact with yesterday's function at p=0.05
pY5, pT5, Pacc5 = probs(Fraction(5, 100))
print(f'p2=0.05: pY={float(pY5):.6f} (yesterday 0.019321), '
      f'pT={float(pT5):.6f}')
# validate against t_results_final.json
tr = json.load(open(SRC + '/../t_results_final.json'))
print('t_results d=3 |T> points vs exact:')
for key, rec in sorted(tr.items()):
    if isinstance(rec, dict) and rec.get('state') == 'T' \
            and rec.get('d', 3) == 3:
        p2v = rec['p2']; 
        pTx = probs(Fraction(p2v).limit_denominator(10**9))[1]
        print(f"  p2={p2v}: sampled {rec.get('ratio', rec)}, "
              f"exact ratio {float(pTx)/p2v:.4f}")

# corrected t_results comparison: keys 'T|d3|p2' -> [accepted, bad]
import json as _json
tr = _json.load(open(SRC + '/../t_results_final.json'))
print('d=3 sampled vs exact:')
for key, (accn, badn) in sorted(tr.items()):
    st, dd, p2s_ = key.split('|')
    if dd != 'd3': continue
    p2v = float(p2s_)
    samp = badn / accn
    err = (samp * (1 - samp) / accn) ** 0.5
    pYx, pTx, _ = probs(Fraction(p2v).limit_denominator(10**9))
    ex = float(pTx if st == 'T' else pYx)
    print(f'  {st} p2={p2v}: sampled {samp:.6f}({err:.6f}) '
          f'exact {ex:.6f} -> {abs(samp-ex)/err:.2f} sigma')

# exact series of p_T and p_Y to p^3 (Fraction arithmetic)
from math import comb as _comb
def wser(W, order=3):
    r = Fraction(16, 15)
    out = [Fraction(0)] * (order + 1)
    for w, n in W.items():
        for k in range(min(w, order) + 1):
            out[k] += n * Fraction((-1)**k * _comb(w, k)) * r**k
    return [c / N for c in out]
S0 = wser(Wnull); SZ = wser(WZ); SX = wser(WX); SY = wser(WY)
def coset_prob_series(sa, sb):
    # P(acc, bZ=a, bX=b) series
    return None
def comb4(a, b):
    return [(S0[k] + (-1)**a * SZ[k] + (-1)**b * SX[k]
             + (-1)**(a+b) * SY[k]) / 4 for k in range(4)]
P10 = comb4(1, 0); P01 = comb4(0, 1); P11 = comb4(1, 1)
Pacc_s = [sum(x) for x in zip(comb4(0, 0), P10, P01, P11)]
def sdiv(numer, denom, order=3):
    out = [Fraction(0)] * (order + 1)
    for k in range(order + 1):
        s = numer[k]
        for j in range(k):
            s -= out[j] * denom[k - j]
        out[k] = s / denom[0]
    return out
numY = [a + b for a, b in zip(P10, P01)]
numT = [z + Fraction(1, 2) * (x + y)
        for z, x, y in zip(P01, P10, P11)]
serY = sdiv(numY, Pacc_s)
serT = sdiv(numT, Pacc_s)
print('exact series (CNOT-only):')
print(f'  p_Y = {serY[1]} p + {serY[2]} p^2 + {serY[3]} p^3')
print(f'  p_T = {serT[1]} p + {serT[2]} p^2 + {serT[3]} p^3')
