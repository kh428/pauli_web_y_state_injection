"""|T> distance-independence, airtight: assert the d=3 contraction
equals the paper's equation (4), then sweep every odd distance to 29."""
import sys, os, time
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import sympy as sp
from verify_d_independence import exact_ler_stable

expr, p2, q = exact_ler_stable(3)
paper_T = (65536*p2**4*q - 65536*p2**4 - 245760*p2**3*q
           + 245760*p2**3 + 345600*p2**2*q - 331200*p2**2
           - 216000*p2*q + 162000*p2 + 50625*q) / (450*(8*p2-15)**2)
diff = sp.simplify(expr - paper_T)
print('d=3 equals paper equation (4):', diff == 0, flush=True)
assert diff == 0
for d in (5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29):
    t0 = time.time()
    e2, _, _ = exact_ler_stable(d)
    ok = sp.simplify(e2 - paper_T) == 0
    print(f'd={d} ({time.time()-t0:.0f}s): '
          f'{"IDENTICAL" if ok else "DIFFERS"}', flush=True)
    assert ok
print('|T> closed form identical at every odd distance 3-29 (asserted)')
