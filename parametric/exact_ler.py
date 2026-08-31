"""EXACT LER as an expectation value over the channel laws, from the
reduced parameterised tsim graph. The reduced object depends on the
error bits only through a few XOR forms; a dynamic program over the
channels gives the exact joint law of those forms, and the LER follows
in closed form in (p2, q)."""
import tsim, pyzx_param, itertools, sys, os
from pyzx_param.utils import VertexType
from fractions import Fraction
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circuits import build, spiral_kinds
from t_experiment import readout_tail, opt_central


class D0(dict):
    def __missing__(self, k): return Fraction(0)


def build_noisy(d=3):
    kinds = dict(spiral_kinds(d))
    site = next(c for c, k in kinds.items() if k == 'Y')
    kinds[site] = 'T'
    txt, info = build(d, kinds, sched=opt_central, noisy_rounds=1,
                      extra_rounds=0, tail=False)
    tail, lw, _, _ = readout_tail(d, kinds, info)
    full = txt + '\n' + tail + f'\nT_DAG {lw}\nH {lw}\nM {lw}'
    noisy, channels, ne = [], [], 0
    for l in full.split('\n'):
        noisy.append(l)
        ps = l.split()
        if ps and ps[0] == 'T' and int(ps[1]) == lw:
            noisy.append(f'X_ERROR(0.001) {lw}')
            channels.append(('X_ERROR', [f'e{ne}'])); ne += 1
        if ps and ps[0] == 'CX' and lw in (int(ps[1]), int(ps[2])):
            noisy.append(f'DEPOLARIZE2(0.001) {ps[1]} {ps[2]}')
            channels.append(('DEP2', [f'e{ne+k}' for k in range(4)])); ne += 4
    return '\n'.join(noisy), channels, lw


def reduced_graph(cn, noisy_txt, lw, bit):
    B = {}
    for l in noisy_txt.split('\n'):
        p_ = l.split()
        if p_ and p_[0] == 'M':
            for qq in p_[1:]: B[int(qq)] = 'Z'
        elif p_ and p_[0] == 'MX':
            for qq in p_[1:]: B[int(qq)] = 'X'
    g = cn.get_graph().copy()
    g.auto_detect_io()
    for v in list(g.inputs()) + list(g.outputs()):
        qq = int(g.qubit(v))
        g.set_type(v, VertexType.X if B[qq] == 'Z' else VertexType.Z)
        if qq == lw: g.set_phase(v, Fraction(bit))
    for v in list(g.vertices()):        # record fixed to all-zero
        g.set_params(v, {x for x in g.get_params(v)
                         if not x.startswith('rec')})
    g.set_inputs(()); g.set_outputs(())
    pyzx_param.full_reduce(g, paramSafe=True)
    return g


def forms_of(g):
    fs = set()
    for v in g.vertices():
        P = frozenset(g.get_params(v))
        if P: fs.add(P)
    s = g.scalar
    for ps in s.phasenodevars:
        if ps: fs.add(frozenset(ps))
    for c, lst in getattr(s, 'phasevars_halfpi', {}).items():
        for ps in lst:
            if ps: fs.add(frozenset(ps))
    for pair in getattr(s, 'phasevars_pi_pair', []):
        for ps in pair:
            if ps: fs.add(frozenset(ps))
    for pp in s.phasepairs:
        if pp.paramsA: fs.add(frozenset(pp.paramsA))
        if pp.paramsB: fs.add(frozenset(pp.paramsB))
    return fs


def amp(g, sub):
    g2 = g.copy()
    for v in list(g2.vertices()):
        P = g2.get_params(v)
        if P:
            par = sum(sub[x] for x in P) % 2
            g2.set_phase(v, g2.phase(v) + Fraction(par))
            g2.set_params(v, set())
    vals = D0({x: Fraction(b) for x, b in sub.items()})
    s_val = g2.scalar.evaluate_scalar(vals)
    if g2.num_vertices():
        t = complex(pyzx_param.tensorfy(g2, preserve_scalar=False))
        return s_val * t
    return s_val


def exact_ler(d=3):
    noisy_txt, channels, lw = build_noisy(d)
    cn = tsim.Circuit(noisy_txt)
    G = {b: reduced_graph(cn, noisy_txt, lw, b) for b in (0, 1)}
    forms = sorted(forms_of(G[0]) | forms_of(G[1]),
                   key=lambda f: sorted(f))
    m = len(forms)
    print(f'remnants {G[0].num_vertices()}/{G[1].num_vertices()} vertices; '
          f'{m} distinct XOR forms over the error bits')
    p2, q = sp.symbols('p_2 q', positive=True)
    # DP over channels: parity-vector -> (symbolic prob, representative e)
    states = {tuple([0] * m): (sp.Integer(1), {})}
    for kind, vs in channels:
        if kind == 'X_ERROR':
            local = [({vs[0]: 0}, 1 - q), ({vs[0]: 1}, q)]
        else:
            local = [({v: 0 for v in vs}, 1 - p2)]
            for bits in itertools.product((0, 1), repeat=4):
                if not any(bits): continue
                local.append((dict(zip(vs, bits)), p2 / 15))
        new = {}
        for pv, (w, rep) in states.items():
            for sub, lw_ in local:
                dpv = tuple((pv[i] + sum(sub.get(x, 0) for x in forms[i]))
                            % 2 for i in range(m))
                if dpv in new:
                    new[dpv] = (new[dpv][0] + w * lw_, new[dpv][1])
                else:
                    new[dpv] = (w * lw_, {**rep, **sub})
        states = new
    print(f'{len(states)} reachable parity patterns')
    num = 0; den = 0
    for pv, (w, rep) in states.items():
        full_rep = D0({x: rep.get(x, 0) for f in forms for x in f})
        a0 = sp.nsimplify(abs(amp(G[0], full_rep))**2,
                          tolerance=1e-8, rational=True)
        a1 = sp.nsimplify(abs(amp(G[1], full_rep))**2,
                          tolerance=1e-8, rational=True)
        num += w * a1
        den += w * (a0 + a1)
    return sp.simplify(num / den), p2, q


if __name__ == '__main__':
    LER, p2, q = exact_ler(3)
    print()
    print('EXACT LER(p2, q) =', LER)
    print()
    print('series (q = p2):',
          sp.series(sp.simplify(LER.subs(q, p2)), p2, 0, 2))
