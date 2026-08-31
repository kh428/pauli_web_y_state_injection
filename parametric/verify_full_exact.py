"""Exact multivariate p_L(p2, pI, p1, pM) of the optimised-central
campaign protocol, from tsim-derived linear forms over ALL noise
locations: 48 CNOT channels (4 bits each), 10 init flips (1 bit),
the rotation channel (2 bits, X/Y/Z at p1/3), 16 readout flips
(1 bit). Character factors per nonzero restriction:
CNOT 1-16p2/15, init 1-2pI, rotation 1-4p1/3, readout 1-2pM.
Validated against MC at strong mixed rates."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy as sp
import tsim
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
    m = tsim.Circuit('\n'.join(L) + TAIL).compile_sampler() \
        .sample(4).astype('uint8')
    det = np.zeros((4, len(dets)), np.uint8)
    for i, (lab, recs) in enumerate(dets):
        for r in recs:
            det[:, i] ^= m[:, r]
    ob = m[:, -1]
    assert (det == det[0]).all() and (ob == ob[0]).all()
    return det[0].copy(), int(ob[0])

# enumerate locations: (species, [basis-generators as gate strings], line)
locs = []
seen = {}
for li, l in enumerate(lines):
    ps = l.split()
    if ps and ps[0] == 'CX':
        pair = (int(ps[1]), int(ps[2]))
        k = seen.get(pair, 0); seen[pair] = k + 1
        if k < 2:
            gens = [f'X {pair[0]}', f'Z {pair[0]}',
                    f'X {pair[1]}', f'Z {pair[1]}']
            locs.append(('p2', li, gens))
    elif ps and ps[0] in ('R', 'RX') and len(ps) == 2 \
            and int(ps[1]) < d * d and int(ps[1]) != site_q:
        locs.append(('pI', li,
                     [('X' if ps[0] == 'R' else 'Z') + f' {ps[1]}']))
i_s = next(i for i, l in enumerate(lines) if l == f'S {site_q}')
locs.append(('pI', i_s, [f'Z {site_q}']))
locs.append(('p1', i_s, [f'X {site_q}', f'Z {site_q}']))
seenm = {}
for li, l in enumerate(lines):
    ps = l.split()
    if ps and ps[0] in ('M', 'MX') and len(ps) == 2 \
            and int(ps[1]) >= d * d:
        qq = int(ps[1])
        k = seenm.get(qq, 0); seenm[qq] = k + 1
        if k < 2:
            locs.append(('pM', li - 1,
                         [('X' if ps[0] == 'M' else 'Z') + f' {qq}'],
                         'before'))
print(f'{len(locs)} locations '
      f'({sum(1 for x in locs if x[0]=="p2")} p2, '
      f'{sum(1 for x in locs if x[0]=="pI")} pI, '
      f'{sum(1 for x in locs if x[0]=="p1")} p1, '
      f'{sum(1 for x in locs if x[0]=="pM")} pM)')

# basis effect vectors per location bit
bits_meta = []           # (loc_idx, species)
DET = []; OBS = []
for gi, loc in enumerate(locs):
    sp_, li, gens = loc[0], loc[1], loc[2]
    before = len(loc) > 3
    for gstr in gens:
        L = list(lines)
        L.insert(li + (0 if before else 1), gstr)
        dv, ov = flips(L)
        DET.append(dv); OBS.append(ov)
        bits_meta.append((gi, sp_))
DET = np.array(DET, dtype=np.uint8)      # nbits x 28
OBS = np.array(OBS, dtype=np.uint8)
nbits = len(bits_meta)
print(f'{nbits} basis bits probed')

F = DET.T.copy()                         # 28 x nbits
Lf = OBS.copy()
nz = [i for i in range(F.shape[0]) if F[i].any()]
print(f'nonzero detector forms: {len(nz)}')
basis_rows = []
piv = {}
for r in F[nz]:
    r = r.copy()
    for c, br in piv.items():
        if r[c]: r ^= br
    w = np.nonzero(r)[0]
    if len(w): piv[w[0]] = r; basis_rows.append(r)
rk = len(basis_rows)
print(f'rank r = {rk}')
Bm = np.array(basis_rows, dtype=np.uint8)

loc_of_bit = np.array([g for g, s in bits_meta])
species_of_loc = {gi: loc[0] for gi, loc in enumerate(locs)}
nloc = len(locs)

def enum(offset):
    cur = offset.copy()
    def loc_nonzero(vec):
        out = np.zeros(nloc, dtype=bool)
        for b in np.nonzero(vec)[0]:
            out[loc_of_bit[b]] = True
        return out
    act = loc_nonzero(cur)
    counts = {}
    def key():
        k = [0, 0, 0, 0]
        for gi in np.nonzero(act)[0]:
            k['p2 pI p1 pM'.split().index(species_of_loc[gi])] += 1
        return tuple(k)
    # incremental: track per-loc nonzero via nibble xor
    width = {gi: [b for b in range(nbits) if loc_of_bit[b] == gi]
             for gi in range(nloc)}
    counts[key()] = counts.get(key(), 0) + 1
    g = 0
    for k2 in range(1, 2 ** rk):
        g2 = k2 ^ (k2 >> 1); diff = g ^ g2; g = g2
        j = int(diff).bit_length() - 1
        row = Bm[j]
        for gi in set(loc_of_bit[np.nonzero(row)[0]].tolist()):
            bs = width[gi]
            cur[bs] ^= row[bs]
            act[gi] = bool(cur[bs].any())
        kk = key()
        counts[kk] = counts.get(kk, 0) + 1
    return counts
W0 = enum(np.zeros(nbits, dtype=np.uint8))
WL = enum(Lf.copy())
print('enumerators done:', len(W0), len(WL), 'weight keys')

p2s, pIs, p1s, pMs = sp.symbols('p_2 p_I p_1 p_M', positive=True)
f2 = 1 - 16 * p2s / 15
fI = 1 - 2 * pIs
f1 = 1 - 4 * p1s / 3
fM = 1 - 2 * pMs
def poly(W):
    return sum(int(n) * f2**a * fI**b * f1**c * fM**e
               for (a, b, c, e), n in W.items())
Pacc = poly(W0) / 2**rk
PL_ = poly(WL) / 2**rk
pL = sp.simplify((Pacc - PL_) / (2 * Pacc))
# leading terms
lead = sp.expand(pL.subs({p2s: sp.Symbol('t')*p2s, pIs: sp.Symbol('t')*pIs,
                          p1s: sp.Symbol('t')*p1s, pMs: sp.Symbol('t')*pMs}))
t = sp.Symbol('t')
s1 = sp.expand(sp.series(lead, t, 0, 2).removeO()).coeff(t, 1)
s2 = sp.expand(sp.series(lead, t, 0, 3).removeO()).coeff(t, 2)
print('LEADING:', sp.nsimplify(s1))
print('SECOND ORDER:', sp.expand(s2))
peq = sp.Symbol('p', positive=True)
alleq = pL.subs({p2s: peq, pIs: peq, p1s: peq, pMs: peq})
ser = sp.series(sp.together(alleq), peq, 0, 4).removeO().expand()
print('all rates equal series:', sp.nsimplify(ser.coeff(peq, 1)), 'p +',
      sp.nsimplify(ser.coeff(peq, 2)), 'p^2 +',
      sp.nsimplify(ser.coeff(peq, 3)), 'p^3')
with open('../RESULTS_full_exact.txt', 'w') as f:
    f.write('exact multivariate campaign p_L(p2,pI,p1,pM), optimised central d=3\n')
    f.write(sp.srepr(pL) + '\n\npretty:\n' + str(pL) + '\n')
    f.write(f'\nleading: {s1}\nsecond: {sp.expand(s2)}\n')
    f.write(f'equal-rates series: {ser.coeff(peq,1)} p + '
            f'{ser.coeff(peq,2)} p^2 + {ser.coeff(peq,3)} p^3\n')

# MC validation at strong mixed rates
pv2, pvI, pv1, pvM = 0.02, 0.01, 0.015, 0.01
exact_num = float(pL.subs({p2s: pv2, pIs: pvI, p1s: pv1, pMs: pvM}))
ntxt, _ = build(d, kinds, sched=opt_central, p2=pv2, pI=pvI, p1=pv1,
                pM=pvM, noisy_rounds=2, extra_rounds=2, tail=False)
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
print(f'validation p2={pv2} pI={pvI} p1={pv1} pM={pvM}: '
      f'exact {exact_num:.6f} vs MC {mc:.6f}({er:.6f}) '
      f'-> {abs(mc-exact_num)/er:.2f} sigma')
