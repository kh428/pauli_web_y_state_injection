"""Export the d=3 noisy protocol -- the tsim CIRCUIT graph (not the
doubled sampling graph) with error parameters, detectors and the
logical observable -- at each stage of the parameter-safe reduction
pipeline, as tsim-tikzit-style tikz. Detector, observable and record
vertices are pinned with open legs (labelled D_j, L, r_k) so the
reduction cannot push their correlations into the scalar; every spider
label shows its phase plus the pi[e XOR] parameter it carries."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
import tsim, pyzx_param
from pyzx_param.utils import VertexType, EdgeType
from tsim.core.parse import parse_stim_circuit
from tsim.core.graph import squash_graph
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'arXiv-2501.15566v5_draft_post_LC_read')

def assemble(d=3):
    kinds = dict(spiral_kinds(d))
    site = next(c for c, k in kinds.items() if k == 'Y')
    kinds[site] = 'T'
    txt, info = build(d, kinds, sched=opt_central, noisy_rounds=1,
                      extra_rounds=0, tail=False)
    tail, lw, col_off, row_off = readout_tail(d, kinds, info)
    fix = [f'CZ rec[{o}] {lw}' for o in col_off] + \
          [f'CX rec[{o}] {lw}' for o in row_off]
    full = (txt + '\n' + tail + '\n' + '\n'.join(fix) +
            f'\nT_DAG {lw}\nH {lw}\nM {lw}')
    noisy = []
    for l in full.split('\n'):
        noisy.append(l)
        ps = l.split()
        if ps and ps[0] == 'T' and ps[1] == str(lw):
            noisy.append(f'X_ERROR(0.001) {lw}')
        if ps and ps[0] == 'CX' and 'rec' not in l and \
                str(lw) in (ps[1], ps[2]):
            noisy.append(f'DEPOLARIZE2(0.001) {ps[1]} {ps[2]}')
    nrecs = sum(len(l.split()) - 1 for l in full.split('\n')
                if l.split() and l.split()[0] in ('M', 'MX')
                and 'rec' not in l)
    det_recs = sorted({r for lab, recs in info['dets'] for r in recs})
    for k in det_recs:
        noisy.append(f'DETECTOR rec[{k - nrecs}]')
    noisy.append('OBSERVABLE_INCLUDE(0) rec[-1]')
    return '\n'.join(noisy), lw

def pname(p):
    if p.startswith('e'): return f'e_{{{p[1:]}}}'
    if p.startswith('rec'): return f'r_{{{p[4:-1]}}}'
    return p

def pkey(p):
    m = re.search(r'\d+', p)
    return (0 if p[0] == 'e' else 1, int(m.group()) if m else 0)

def phase_label(ph, params):
    parts = []
    if ph not in (0, Fraction(0)):
        f = Fraction(ph)
        num = '' if f.numerator == 1 else str(f.numerator)
        parts.append('\\pi' if f == 1 else
                     (f'\\tfrac{{{num}\\pi}}{{{f.denominator}}}'
                      if f.denominator != 1 else f'{num}\\pi'))
    ps = sorted(params, key=pkey)
    if ps:
        xor = '{\\oplus}'.join(pname(p) for p in ps)
        parts.append(f'\\pi[{xor}]' if len(ps) > 1 else f'\\pi {pname(ps[0])}')
    return ('$' + '{+}'.join(parts) + '$') if parts else ''

def export(g, path, xs=0.55, ys=0.5, blabels=None):
    blabels = blabels or {}
    L = ['\\begin{tikzpicture}', '  \\begin{pgfonlayer}{nodelayer}']
    for v in g.vertices():
        t = g.type(v)
        lab = phase_label(g.phase(v), set(g.get_params(v)))
        if t == VertexType.H_BOX:
            st = 'thad'
        elif t == VertexType.BOUNDARY:
            st, lab = 'tnone', blabels.get(v, '')
        else:
            base = 'z' if t == VertexType.Z else 'x'
            st = base + ('pd' if lab else 'd')
        x, y = g.row(v) * xs, -float(g.qubit(v)) * ys
        L.append(f'    \\node [style={st}] ({v}) at ({x:.2f}, {y:.2f}) {{{lab}}};')
    L += ['  \\end{pgfonlayer}', '  \\begin{pgfonlayer}{edgelayer}']
    for e in g.edges():
        s, t2 = g.edge_st(e)
        st = 'thedge' if g.edge_type(e) == EdgeType.HADAMARD else 'tedge'
        L.append(f'    \\draw [style={st}] ({s}) to ({t2});')
    L += ['  \\end{pgfonlayer}', '\\end{tikzpicture}']
    open(path, 'w').write('\n'.join(L) + '\n')

full, lw = assemble(3)
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
legs, blabels = [], {}
def leg(v, lab, below=False):
    dq = 0.55 if below else -1
    w = g.add_vertex(VertexType.BOUNDARY, qubit=g.qubit(v) + dq,
                     row=g.row(v) + (0.45 if below else 0))
    g.add_edge((v, w)); legs.append(w); blabels[w] = lab
for i, v in enumerate(built.detectors):
    g.set_params(v, {p for p in g.get_params(v) if not p.startswith('det')})
    leg(v, f'$D_{{{i}}}$')
for i, v in built.observables_dict.items():
    g.set_params(v, {p for p in g.get_params(v) if not p.startswith('obs')})
    leg(v, '$L$')
for k, v in enumerate(built.rec):
    leg(v, f'$r_{{{k}}}$', below=True)
g.set_inputs(()); g.set_outputs(tuple(legs))

def snap(name, xs=0.55, ys=0.5):
    export(g, os.path.join(OUT, f'fig_tsim_stage_{name}.tex'),
           xs=xs, ys=ys, blabels=blabels)
    print(f'{name}: {g.num_vertices()} vertices, {g.num_edges()} edges')

snap('pre')
pyzx_param.simplify.spider_simp(g, quiet=True); snap('fuse')
pyzx_param.simplify.id_simp(g, quiet=True);    snap('id')
pyzx_param.full_reduce(g, quiet=True, paramSafe=True)

def layout_final(gg):
    def est_w(u):
        lab = phase_label(gg.phase(u), set(gg.get_params(u)))
        return max(0.5, 0.135 * len(lab.replace('{\\oplus}', 'x')
                                     .replace('\\pi', 'p').replace('$', '')
                                     .replace('{', '').replace('}', '')))
    seen, comps = set(), []
    for v in gg.vertices():
        if v in seen: continue
        comp, stack = [], [v]
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u); comp.append(u)
            stack.extend(gg.neighbors(u))
        comps.append(comp)
    comps.sort(key=lambda c: -len(c))
    big = [c for c in comps if len(c) > 2]
    coins = [c for c in comps if len(c) == 2]
    y = 0.0
    for comp in big:
        core = sorted([u for u in comp if gg.type(u) != VertexType.BOUNDARY],
                      key=lambda u: gg.row(u))
        x = 0.0
        for i, u in enumerate(core):
            if i: x += est_w(core[i - 1]) / 2 + est_w(u) / 2 + 0.6
            gg.set_row(u, x); gg.set_qubit(u, -y)
        for u in [u for u in comp if gg.type(u) == VertexType.BOUNDARY]:
            nb = next(iter(gg.neighbors(u)))
            up = blabels.get(u, '').startswith(('$D', '$L'))
            k = sum(1 for w2 in comp if w2 < u and
                    gg.type(w2) == VertexType.BOUNDARY and
                    next(iter(gg.neighbors(w2))) == nb)
            gg.set_row(u, gg.row(nb) + 0.9 * k)
            gg.set_qubit(u, -(y + 1.0) if up else -(y - 1.0))
        y -= 2.6
    x = 0.0
    for comp in coins:
        b = next(u for u in comp if gg.type(u) == VertexType.BOUNDARY)
        c = next(u for u in comp if gg.type(u) != VertexType.BOUNDARY)
        gg.set_row(c, x); gg.set_qubit(c, -y)
        gg.set_row(b, x); gg.set_qubit(b, -(y + 1.0))
        x += 1.5
layout_final(g)
snap('final', xs=1.0, ys=1.0)
