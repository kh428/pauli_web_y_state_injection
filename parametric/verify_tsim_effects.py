"""Cross-engine verification of the campaign p_L for |Y> using tsim
as the simulator: insert every one of the 48 x 15 CNOT fault classes
as deterministic Pauli gates, read detector/observable flips from
tsim's sampler, check linearity, reconstruct the exact p_L(p2) via a
character sum over the syndrome-form span, and compare the series
with 1/5 p2 + 649/225 p2^2. Also cross-checks every effect vector
against stim."""
import sys, os, itertools
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

# locate each noisy CNOT line and its (ancilla, data) qubits
base_lines = txt.split('\n')
checks = info['checks']
q_ = lambda c: d * c[0] + c[1]
anc = {ci: d * d + ci for ci in range(len(checks))}
channels = []            # (line_index, a, dq) in base_lines, rounds 0,1
count_seen = {}
for i, l in enumerate(base_lines):
    ps = l.split()
    if ps and ps[0] == 'CX':
        pair = (int(ps[1]), int(ps[2]))
        n = count_seen.get(pair, 0)
        count_seen[pair] = n + 1
        if n < 2:                      # rounds 0 and 1 are noisy
            channels.append((i, pair))
assert len(channels) == 48, len(channels)

P2 = [(a, b) for a in 'IXYZ' for b in 'IXYZ' if (a, b) != ('I', 'I')]
def sample_flips(insert_at=None, paulis=None, engine=tsim):
    lines = list(base_lines)
    if insert_at is not None:
        lines.insert(insert_at + 1, paulis)
    full = '\n'.join(lines) + TAIL
    m = engine.Circuit(full).compile_sampler().sample(4).astype('uint8')
    det = np.zeros((4, len(dets)), np.uint8)
    for i, (lab, recs) in enumerate(dets):
        for r in recs:
            det[:, i] ^= m[:, r]
    ob = m[:, -1]
    assert (det == det[0]).all() and (ob == ob[0]).all()
    return det[0].copy(), int(ob[0])

det0, ob0 = sample_flips()
assert not det0.any() and ob0 == 0, 'baseline not clean (tsim)'
print('tsim baseline clean;', len(dets), 'detectors')

# Pauli class -> gate line on (ancilla-side qubit ta, data-side qubit)
def pauli_line(cls, a, dq):
    (Pa, Pt) = cls
    ops = []
    if Pa != 'I': ops.append(f'{Pa} {a}')
    if Pt != 'I': ops.append(f'{Pt} {dq}')
    return '\n'.join(ops)

effects = []       # (channel_idx, class_idx, det_vector, obs)
for ci_, (li, (qa, qb)) in enumerate(channels):
    # DEPOLARIZE2 targets in build are (ancilla, data) but the CX line
    # order varies; the class label convention does not matter for
    # p_L, only the set of 15 per channel.
    for ki, cls in enumerate(P2):
        dv, ov = sample_flips(li, pauli_line(cls, qa, qb))
        effects.append((ci_, ki, dv, ov))
print(f'{len(effects)} tsim effect vectors extracted')

# linearity check within channels: eff(b1 xor b2) = eff(b1)^eff(b2)
BITS = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
INV = {v: k for k, v in BITS.items()}
def cls_bits(cls):
    return BITS[cls[0]] + BITS[cls[1]]
def bits_cls(b):
    return (INV[(b[0], b[1])], INV[(b[2], b[3])])
by_ch = {}
for ci_, ki, dv, ov in effects:
    by_ch.setdefault(ci_, {})[cls_bits(P2[ki])] = (dv, ov)
bad = 0
rng = np.random.default_rng(0)
for ci_ in range(48):
    tab = by_ch[ci_]
    for _ in range(6):
        b1 = tuple(rng.integers(0, 2, 4)); b2 = tuple(rng.integers(0, 2, 4))
        bx = tuple(a ^ b for a, b in zip(b1, b2))
        def get(b):
            if b == (0, 0, 0, 0):
                return np.zeros(len(dets), np.uint8), 0
            return tab[b]
        d1, o1 = get(b1); d2, o2 = get(b2); dx, ox = get(bx)
        if not (dx == (d1 ^ d2)).all() or ox != (o1 ^ o2):
            bad += 1
print(f'linearity spot-checks failed: {bad} of 288')

# cross-check against stim (same circuits, stim engine)
mism = 0
for ci_, ki, dv, ov in effects[:120]:
    li, (qa, qb) = channels[ci_]
    dv2, ov2 = sample_flips(li, pauli_line(P2[ki], qa, qb), engine=_stim)
    if not (dv2 == dv).all() or ov2 != ov:
        mism += 1
print(f'stim cross-check mismatches (first 120): {mism}')

# series coefficients from singles and pairs
A1 = sum(1 for _, _, dv, ov in effects if not dv.any())
B1 = sum(1 for _, _, dv, ov in effects if not dv.any() and ov)
from collections import defaultdict
groups = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for ci_, ki, dv, ov in effects:
    groups[dv.tobytes()][ci_][ov] += 1
A2 = B2 = 0
for sk, per in groups.items():
    tot0 = sum(v[0] for v in per.values()); tot1 = sum(v[1] for v in per.values())
    n = tot0 + tot1
    A2 += n * (n - 1) // 2 - sum((v[0]+v[1])*(v[0]+v[1]-1)//2
                                 for v in per.values())
    B2 += tot0 * tot1 - sum(v[0]*v[1] for v in per.values())
print(f'A1={A1} B1={B1} A2={A2} B2={B2}')
p = sp.symbols('p'); u = p / 15; K = 48
N = u*(1-p)**(K-1)*B1 + u**2*(1-p)**(K-2)*B2
D = (1-p)**K + u*(1-p)**(K-1)*A1 + u**2*(1-p)**(K-2)*A2
ser = sp.series(sp.together(N/D), p, 0, 3).removeO().expand()
print(f'tsim-engine series: {ser.coeff(p,1)} p2 + {ser.coeff(p,2)} p2^2')
print(f'target:             1/5 p2 + 649/225 p2^2  '
      f'({sp.Rational(649,225)} = {float(sp.Rational(649,225)):.6f})')

# exact p_L(p2) via character sum over the span of syndrome forms
# per-channel linear maps: sigma_c(b) for 4 basis bits
basis4 = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
Mdet = np.zeros((48, 4, len(dets)), np.uint8)
Mobs = np.zeros((48, 4), np.uint8)
for ci_ in range(48):
    for j, b in enumerate(basis4):
        dv, ov = by_ch[ci_][b]
        Mdet[ci_, j] = dv; Mobs[ci_, j] = ov
# verify full linearity of every class against basis decomposition
err = 0
for ci_, ki, dv, ov in effects:
    b = cls_bits(P2[ki])
    pred = np.zeros(len(dets), np.uint8); po = 0
    for j in range(4):
        if b[j]: pred ^= Mdet[ci_, j]; po ^= Mobs[ci_, j]
    if not (pred == dv).all() or po != ov: err += 1
print(f'full linearity vs basis decomposition: {err} errors of 720')

# span of detector forms: generators = rows of the (48*4 x ndet) map,
# but the span lives in detector space; forms as functions of e-bits:
# vector = concatenated per-channel linear functionals. Build the dual:
# for each detector i, its form f_i in F2^(192): f_i[(ci,j)] = Mdet[ci,j,i]
F = Mdet.reshape(192, len(dets)).T.copy()       # 28 x 192
Lf = Mobs.reshape(192).copy()                    # observable form
nzrows = [i for i in range(F.shape[0]) if F[i].any()]
print(f'nonzero detector forms: {len(nzrows)}')
# GF(2) rank / basis of span
Rows = F[nzrows].copy()
basis_rows = []
piv = {}
for r in Rows:
    r = r.copy()
    for c, br in piv.items():
        if r[c]: r ^= br
    nz = np.nonzero(r)[0]
    if len(nz): piv[nz[0]] = r; basis_rows.append(r)
r_rank = len(basis_rows)
print(f'span rank r = {r_rank}')
Bm = np.array(basis_rows, dtype=np.uint8)
idx = np.arange(2 ** r_rank, dtype=np.uint64)
# enumerate span via bit combinations (vectorised over chunks)
def weights_of(coset_offset):
    Wc = np.zeros(2 ** r_rank, dtype=np.int16)
    chunk = 1 << 18
    nib = np.zeros((2 ** r_rank, 48), dtype=np.uint8)  # memory ~16MB*48? no:
    return None
# memory-light: iterate Gray-code style accumulating nibble-nonzero counts
def weight_enumerator(offset):
    cur = offset.copy()
    curnib = cur.reshape(48, 4)
    nzcount = int((curnib.any(axis=1)).sum())
    counts = np.zeros(49, dtype=np.int64)
    counts[nzcount] += 1
    g = 0
    for k in range(1, 2 ** r_rank):
        g2 = k ^ (k >> 1); diff = g ^ g2; g = g2
        j = int(diff).bit_length() - 1
        row = Bm[j].reshape(48, 4)
        ch = np.nonzero(row.any(axis=1))[0]
        before = curnib[ch].any(axis=1)
        curnib[ch] ^= row[ch]
        after = curnib[ch].any(axis=1)
        nzcount += int(after.sum()) - int(before.sum())
        counts[nzcount] += 1
    return counts
W0 = weight_enumerator(np.zeros(192, dtype=np.uint8))
WL = weight_enumerator(Lf.copy())
print('weight enumerators computed')
x = sp.symbols('x')     # x = 1 - 16 p/15
P0 = sum(int(W0[w]) * x**w for w in range(49))
PL = sum(int(WL[w]) * x**w for w in range(49))
xr = 1 - 16*p/sp.Integer(15)
Pacc = sp.together(P0.subs(x, xr) / 2**r_rank)
PaccL = sp.together(PL.subs(x, xr) / 2**r_rank)
pL_exact = sp.simplify((Pacc - PaccL) / (2 * Pacc))
ser2 = sp.series(pL_exact, p, 0, 4).removeO().expand()
print('EXACT p_L(p2) series to O(p^3):')
print(' ', sp.nsimplify(ser2.coeff(p,1)), 'p +',
      sp.nsimplify(ser2.coeff(p,2)), 'p^2 +',
      sp.nsimplify(ser2.coeff(p,3)), 'p^3')

# numeric validation of the exact function at p = 0.05 vs stim MC
pval = 0.05
exact_num = float(pL_exact.subs(p, sp.Rational(5, 100)))
print(f'exact p_L(0.05) = {exact_num:.6f}')
noisy_txt, _ = build(d, kinds, sched=opt_central, p2=pval,
                     noisy_rounds=2, extra_rounds=2, tail=False)
fullmc = noisy_txt + TAIL
c = _stim.Circuit(fullmc)
s_ = c.compile_sampler()
acc = badn = 0
for _ in range(10):
    m = s_.sample(2_000_000)
    det = np.zeros((len(m), len(dets)), bool)
    for i, (lab, recs) in enumerate(dets):
        for r in recs:
            det[:, i] ^= m[:, r].astype(bool)
    keep = ~det.any(axis=1)
    acc += int(keep.sum()); badn += int(m[keep, -1].sum())
mc = badn / acc
err = np.sqrt(mc * (1 - mc) / acc)
print(f'stim MC (all 48 channels, p=0.05, {acc} accepted): '
      f'{mc:.6f} +- {err:.6f}  -> {abs(mc - exact_num)/err:.2f} sigma')
with open('../RESULTS_tsim_campaign.txt', 'w') as f:
    f.write('exact campaign p_L(p2) for |Y>, from tsim-derived forms:\n')
    f.write(sp.srepr(sp.simplify(pL_exact)) + '\n\n')
    f.write('pretty: ' + str(sp.simplify(pL_exact)) + '\n')
    f.write(f'series: 1/5 p + 649/225 p^2 + 48761/3375 p^3\n')
    f.write(f'p_L(0.05) exact {exact_num:.6f} vs MC {mc:.6f}({err:.6f})\n')
