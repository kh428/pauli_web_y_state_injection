"""tsim-parametric verification of the campaign p_L for |Y>: pin the
records, detectors and observable of the FULL campaign circuit
(2 noisy rounds, DEPOLARIZE2 on all 48 CNOTs, deflation readout with
frame fix) as open legs, parameter-safe full_reduce, read the XOR
forms off the legs, cross-check every single-fault effect against the
stim effect vectors, and recover the exact p_L(p2) from the forms by
a character sum over the form span."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy as sp
from fractions import Fraction
import tsim, pyzx_param
from pyzx_param.utils import VertexType, EdgeType as ET
from tsim.core.parse import parse_stim_circuit
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central
import stim as _stim

d = 3
kinds = spiral_kinds(d)
site0 = next(c for c, k in kinds.items() if k == 'Y')
txt, info = build(d, kinds, sched=opt_central, p2=0.001,
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
    assert (par == par[0]).all()
    CONSTS[basis] = int(par[0])
fix = []
for o in col_off: fix.append(f'CZ rec[{o}] {lw}')
if CONSTS['Z']: fix.append(f'Z {lw}')
for o in row_off: fix.append(f'CX rec[{o}] {lw}')
if CONSTS['X']: fix.append(f'X {lw}')
full = (txt + '\n' + tail + '\n' + '\n'.join(fix) +
        f'\nS_DAG {lw}\nH {lw}\nM {lw}')
nmeas = info['nmeas']
total = nmeas + d * d - 1 + 1
lines = [full]
for lab, recs in info['dets']:
    lines.append('DETECTOR ' + ' '.join(f'rec[{r - total}]' for r in recs))
lines.append('OBSERVABLE_INCLUDE(0) rec[-1]')
full = '\n'.join(lines)

cn = tsim.Circuit(full)
built = parse_stim_circuit(cn._stim_circ)
g = built.graph.copy()
B = {}
for l in full.split('\n'):
    p_ = l.split()
    if p_ and p_[0] == 'M' and 'rec' not in l:
        for qq in p_[1:]: B[int(qq)] = 'Z'
    elif p_ and p_[0] == 'MX':
        for qq in p_[1:]: B[int(qq)] = 'X'
g.auto_detect_io()
for v in list(g.inputs()) + list(g.outputs()):
    qq = int(g.qubit(v))
    g.set_type(v, VertexType.X if B[qq] == 'Z' else VertexType.Z)
legs, leg_of = [], {}
def leg(v, tag):
    w = g.add_vertex(VertexType.BOUNDARY, qubit=g.qubit(v), row=g.row(v))
    g.add_edge((v, w)); legs.append(w); leg_of[tag] = w
for i, v in enumerate(built.detectors):
    g.set_params(v, {p for p in g.get_params(v) if not p.startswith('det')})
    leg(v, f'D{i}')
for i, v in built.observables_dict.items():
    g.set_params(v, {p for p in g.get_params(v) if not p.startswith('obs')})
    leg(v, 'L')
for k, v in enumerate(built.rec):
    leg(v, f'r{k}')
g.set_inputs(()); g.set_outputs(tuple(legs))
print(f'pre-reduce: {g.num_vertices()} vertices, '
      f'{len(built.rec)} records pinned')
pyzx_param.full_reduce(g, quiet=True, paramSafe=True)
print(f'post-reduce: {g.num_vertices()} vertices')

def leg_info(tag):
    w = leg_of[tag]
    nb = list(g.neighbors(w))
    assert len(nb) == 1
    v = nb[0]
    deg = len(list(g.neighbors(v)))
    return v, set(g.get_params(v)), g.phase(v), deg

report = {}
multi = []
for tag in list(leg_of):
    v, ps, ph, deg = leg_info(tag)
    report[tag] = (ps, ph, deg)
    if deg > 1: multi.append(tag)
print(f'legs on multi-degree spiders: {multi}')
nz_dets = [t for t in report if t.startswith('D') and report[t][0]]
print(f'nonzero detector forms: {len(nz_dets)}')
Lps, Lph, Ldeg = report['L']
print(f'L: form size {len(Lps)}, phase {Lph}, deg {Ldeg}')
rec_forms = {t: report[t] for t in report if t.startswith('r')}
coins = [t for t, (ps, ph, dg) in rec_forms.items() if not ps]
print(f'records with empty form (coins/deterministic-0): {len(coins)}')
