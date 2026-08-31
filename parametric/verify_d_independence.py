"""Base cases for the distance-independence proposition: the exact
multi-channel closed form at d = 3, 5, 7, 9. Same pipeline as
exact_ler, with the branch scalars' common power-of-2 offset removed
before evaluation (only amplitude RATIOS enter p_L, and the raw
magnitudes underflow double precision beyond d = 5)."""
import sys, os, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
import sympy as sp
import tsim
from exact_ler import build_noisy, reduced_graph, forms_of, amp, D0

def exact_ler_stable(d):
    noisy_txt, channels, lw = build_noisy(d)
    cn = tsim.Circuit(noisy_txt)
    G = {b: reduced_graph(cn, noisy_txt, lw, b) for b in (0, 1)}
    k = min(G[0].scalar.power2, G[1].scalar.power2)
    for b in (0, 1):
        G[b].scalar.power2 -= k
    forms = sorted(forms_of(G[0]) | forms_of(G[1]), key=lambda f: sorted(f))
    m = len(forms)
    print(f'  d={d}: remnants {G[0].num_vertices()}/{G[1].num_vertices()} '
          f'vertices, {m} forms, common power2 offset {k}')
    p2, q = sp.symbols('p_2 q', positive=True)
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
    num = 0; den = 0
    for pv, (w, rep) in states.items():
        full_rep = D0({x: rep.get(x, 0) for f in forms for x in f})
        vals = []
        for b in (0, 1):
            a = abs(amp(G[b], full_rep)) ** 2
            r = sp.nsimplify(a, tolerance=1e-9, rational=True)
            assert r.is_finite, (d, b, a)
            vals.append(r)
        num += w * vals[1]
        den += w * (vals[0] + vals[1])
    return sp.simplify(num / den), p2, q

ref = None
for d in (3, 5, 7, 9):
    t0 = time.time()
    expr, p2, q = exact_ler_stable(d)
    dt = time.time() - t0
    if ref is None:
        ref = expr
        print(f'  d=3 formula ({dt:.1f}s):', expr)
        print('  series (q=p2):', sp.series(sp.simplify(
            expr.subs(q, p2)), p2, 0, 2))
    else:
        diff = sp.simplify(expr - ref)
        print(f'  d={d} - d=3 difference ({dt:.1f}s): {diff}')
        assert diff == 0, f'd={d} formula differs!'
print('ALL BASE CASES IDENTICAL')
