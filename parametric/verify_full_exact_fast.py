"""Exact multivariate p_L(p2,pI,p1,pM), optimised central campaign,
FAST: stim probes (engine-equivalence already established), numpy
parity enumeration, Fraction series. Outputs leading + second order,
equal-rates series to p^3, exact values, MC validation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fractions import Fraction
from itertools import product
import stim as _stim
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central

d = 3
kinds = spiral_kinds(d)
site0 = next(c for c, k in kinds.items() if k == 'Y')
txt, info = build(d, kinds, sched=opt_central, p2=0.0,
                  noisy_rounds=2, extra_rounds=2, tail=False)
tail, lw, col_off, row_off = readout_tail(d, kinds, info)
CONSTS = {}
for basis, kk, post in [('Z', '+', f'H {lw}'), ('X', '0', '')]:
    kcal = dict(kinds); kcal[site0] = kk
    ct, _ = build(d, kcal, sched=opt_central, tail=False,
                  site_override=site0)
    cfull = ct + '\n' + tail + ('\n' + post if post else '') + f'\nM {lw}'
    m = _stim.Circuit(cfull).compile_sampler().sample(64).astype('uint8')
    offs = col_off if basis == 'Z' else row_off
    nc = m.shape[1]
    par = (m[:, [nc - 1 + o for o in offs]].sum(axis=1) + m[:, -1]) % 2
    CONSTS[basis] = int(par[0])
fix = []
for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
if CONSTS['Z']: fix.append(f'Z {lw}')
for o in row_off: fix.append(f'CX rec[{o}] {lw}')
if CONSTS['X']: fix.append(f'X {lw}')
TAIL = '\n' + tail + '\n' + '\n'.join(fix) + f'\nS_DAG {lw}\nH {lw}\nM {lw}'
dets = info['dets']
lines = txt.split('\n')
q_ = lambda c: d * c[0] + c[1]
site_q = q_(site0)

def flips(L):
    m = _stim.Circuit('\n'.join(L) + TAIL).compile_sampler() \
        .sample(4).astype('uint8')
    det = np.zeros((4, len(dets)), np.uint8)
    for i, (lab, recs) in enumerate(dets):
        for r in recs:
            det[:, i] ^= m[:, r]
    ob = m[:, -1]
    assert (det == det[0]).all() and (ob == ob[0]).all()
    return det[0].copy(), int(ob[0])

locs = []
seen = {}
for li, l in enumerate(lines):
    ps = l.split()
    if ps and ps[0] == 'CX':
        pair = (int(ps[1]), int(ps[2]))
        k = seen.get(pair, 0); seen[pair] = k + 1
        if k < 2:
            locs.append(('p2', li, [f'X {pair[0]}', f'Z {pair[0]}',
                                    f'X {pair[1]}', f'Z {pair[1]}'], 0))
    elif ps and ps[0] in ('R', 'RX') and len(ps) == 2 \
            and int(ps[1]) < d * d and int(ps[1]) != site_q:
        locs.append(('pI', li,
                     [('X' if ps[0] == 'R' else 'Z') + f' {ps[1]}'], 0))
i_s = next(i for i, l in enumerate(lines) if l == f'S {site_q}')
locs.append(('pI', i_s, [f'Z {site_q}'], 0))
locs.append(('p1', i_s, [f'X {site_q}', f'Z {site_q}'], 0))
seenm = {}
for li, l in enumerate(lines):
    ps = l.split()
    if ps and ps[0] in ('M', 'MX') and len(ps) == 2 \
            and int(ps[1]) >= d * d:
        qq = int(ps[1])
        k = seenm.get(qq, 0); seenm[qq] = k + 1
        if k < 2:
            locs.append(('pM', li,
                         [('X' if ps[0] == 'M' else 'Z') + f' {qq}'], 1))
print(f'{len(locs)} locations, species counts:',
      {sp_: sum(1 for x in locs if x[0] == sp_)
       for sp_ in ('p2', 'pI', 'p1', 'pM')}, flush=True)

bits_loc = []; DETv = []; OBSv = []
for gi, (sp_, li, gens, before) in enumerate(locs):
    for gstr in gens:
        L = list(lines)
        L.insert(li + (0 if before else 1), gstr)
        dv, ov = flips(L)
        DETv.append(dv); OBSv.append(ov); bits_loc.append(gi)
DETv = np.array(DETv, np.uint8); OBSv = np.array(OBSv, np.uint8)
nbits = len(bits_loc)
print(f'{nbits} bits probed', flush=True)

F = DETv.T
nz = [i for i in range(F.shape[0]) if F[i].any()]
basis_rows, piv = [], {}
for r in F[nz]:
    r = r.copy()
    for c, br in piv.items():
        if r[c]: r ^= br
    w = np.nonzero(r)[0]
    if len(w): piv[w[0]] = r; basis_rows.append(r)
rk = len(basis_rows)
print(f'nonzero det forms {len(nz)}, rank {rk}', flush=True)
Bm = np.array(basis_rows, np.uint8)          # rk x nbits
N = 1 << rk
idx = np.arange(N, dtype=np.uint32)
PAR = np.zeros(1 << 10, np.uint8)
for i in range(1, 1 << 10):
    PAR[i] = PAR[i >> 1] ^ (i & 1)
def parity_of_mask(mask):
    return PAR[(idx & (mask & 0x3FF))] ^ PAR[((idx >> 10) & 0x3FF) &
                                             ((mask >> 10) & 0x3FF)]
species_idx = {'p2': 0, 'pI': 1, 'p1': 2, 'pM': 3}
def enumerate_weights(offset_bits):
    Wsp = [np.zeros(N, np.int16) for _ in range(4)]
    for gi, (sp_, li, gens, before) in enumerate(locs):
        bs = [b for b in range(nbits) if bits_loc[b] == gi]
        nzflag = np.zeros(N, np.bool_)
        for b in bs:
            mask = 0
            for j in range(rk):
                if Bm[j, b]: mask |= (1 << j)
            par = parity_of_mask(mask)
            if offset_bits[b]: par = par ^ 1
            nzflag |= par.astype(bool)
        Wsp[species_idx[sp_]] += nzflag
    return Wsp
zeros = np.zeros(nbits, np.uint8)
W0sp = enumerate_weights(zeros)
WLsp = enumerate_weights(OBSv)
print('weights enumerated', flush=True)

def counts_of(Wsp):
    code = ((Wsp[0].astype(np.int64) * 16 + Wsp[1]) * 4 + Wsp[2]) \
        * 32 + Wsp[3]
    u, c = np.unique(code, return_counts=True)
    out = {}
    for cd, n in zip(u, c):
        wM = cd % 32; cd //= 32
        w1 = cd % 4; cd //= 4
        wI = cd % 16; cd //= 16
        out[(int(cd), int(wI), int(w1), int(wM))] = int(n)
    return out
C0 = counts_of(W0sp); CL = counts_of(WLsp)
print(f'{len(C0)}/{len(CL)} weight keys', flush=True)

# exact evaluation and Fraction series
def val(C, f2, fI, f1, fM):
    return sum(n * f2**a * fI**b * f1**c * fM**e
               for (a, b, c, e), n in C.items())
def pL_at(p2, pI, p1, pM):
    f2 = 1 - Fraction(16, 15) * p2
    fI = 1 - 2 * pI
    f1 = 1 - Fraction(4, 3) * p1
    fM = 1 - 2 * pM
    P0 = Fraction(val(C0, f2, fI, f1, fM), N)
    PLv = Fraction(val(CL, f2, fI, f1, fM), N)
    return (P0 - PLv) / (2 * P0), P0

# equal-rates series to p^3 by exact finite differences? use symbolic
# via polynomial in one variable: evaluate val at symbolic Fraction
# series: do Taylor by evaluating derivative-free: build series
# coefficients directly: each term (1-16p/15)^a(1-2p)^b(1-4p/3)^c(1-2p)^e
from math import comb
def series_equal(C, order=3):
    coeff = [Fraction(0)] * (order + 1)
    r2, rI, r1, rM = Fraction(16,15), Fraction(2), Fraction(4,3), Fraction(2)
    for (a, b, c, e), n in C.items():
        pref = [(a, r2), (b, rI), (c, r1), (e, rM)]
        term = [Fraction(0)] * (order + 1)
        term[0] = Fraction(1)
        for cnt, rate in pref:
            new = [Fraction(0)] * (order + 1)
            fac = [Fraction((-1)**k * comb(cnt, k)) * rate**k
                   if k <= cnt else Fraction(0) for k in range(order+1)]
            for i in range(order + 1):
                for j in range(order + 1 - i):
                    new[i + j] += term[i] * fac[j]
            term = new
        for k in range(order + 1):
            coeff[k] += n * term[k]
    return [c / N for c in coeff]
S0 = series_equal(C0); SL = series_equal(CL)
# pL = (S0 - SL) / (2 S0) as series
def series_div(numer, denom, order=3):
    out = [Fraction(0)] * (order + 1)
    for k in range(order + 1):
        s = numer[k]
        for j in range(k):
            s -= out[j] * denom[k - j]
        out[k] = s / denom[0]
    return out
num = [(a - b) / 2 for a, b in zip(S0, SL)]
ser = series_div(num, S0)
print('equal-rates series:', ser[0], '+', ser[1], 'p +', ser[2],
      'p^2 +', ser[3], 'p^3', flush=True)

# leading multivariate terms: probe one species at a time
def lead_species(which):
    args = [Fraction(0)] * 4
    args[which] = Fraction(1, 10**6)
    v, _ = pL_at(*args)
    return v * 10**6
print('leading terms (per species, from infinitesimal probes):')
for i, nm in enumerate(('p2', 'pI', 'p1', 'pM')):
    print(f'  {nm}: {float(lead_species(i)):.6f}')
ex5, Pacc5 = pL_at(Fraction(2,100), Fraction(1,100), Fraction(15,1000),
                   Fraction(1,100))
print(f'exact p_L(0.02, 0.01, 0.015, 0.01) = {float(ex5):.6f}, '
      f'P_acc = {float(Pacc5):.6f}', flush=True)
import json
with open('../RESULTS_full_exact.txt', 'w') as f:
    f.write('exact multivariate campaign p_L, optimised central d=3\n')
    f.write('weight enumerators (a=w2,b=wI,c=w1,e=wM: count):\n')
    f.write('C0=' + json.dumps({str(k): v for k, v in C0.items()}) + '\n')
    f.write('CL=' + json.dumps({str(k): v for k, v in CL.items()}) + '\n')
    f.write(f'rank={rk}\n')
    f.write(f'equal-rates series: {ser[1]} p + {ser[2]} p^2 + {ser[3]} p^3\n')

# MC validation, mixed strong rates
ntxt, _ = build(d, kinds, sched=opt_central, p2=0.02, pI=0.01, p1=0.015,
                pM=0.01, noisy_rounds=2, extra_rounds=2, tail=False)
s_ = _stim.Circuit(ntxt + TAIL).compile_sampler()
acc = bad = 0
for _ in range(15):
    m = s_.sample(4_000_000)
    det = np.zeros((len(m), len(dets)), bool)
    for i, (lab, recs) in enumerate(dets):
        for r in recs:
            det[:, i] ^= m[:, r].astype(bool)
    keep = ~det.any(axis=1)
    acc += int(keep.sum()); bad += int(m[keep, -1].sum())
mc = bad / acc
er = np.sqrt(mc * (1 - mc) / acc)
print(f'MC validation: {mc:.6f} +- {er:.6f} vs exact {float(ex5):.6f} '
      f'-> {abs(mc - float(ex5))/er:.2f} sigma')
