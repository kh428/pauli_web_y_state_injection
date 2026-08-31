import os
"""Step 6b: decoded-convention per-gate malignant data for the flat
corner circuit (N/Z schedule, d=3), then regenerate fig_circuit_full
with the decoded violet numbers. Output stays in the rework folder."""
import sys, os, io, re, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..',
                   'figures_3d', 'src')
sys.path.insert(0, SRC)
import iter_lc_reconstruct as ilr
from physical import rect, injection_kind

d = 3
ck = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}

def nz_sched(pos, sup, typ):
    fi, fj = pos[0] - 0.5, pos[1] - 0.5
    NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
    SW = (fi, fj);     SE = (fi + 1, fj)
    order = ([NW, SW, NE, SE] if typ == 'X' else [NW, NE, SW, SE])
    ss = set(sup)
    return [c if c in ss else None for c in order]

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    n2, nI, nM = ilr.count(ck, d, nz_sched, 'corner/NZ/d3', verbose=True)
out = buf.getvalue()
print(f'decoded N/Z corner d=3: n2={n2}, nI={nI}, nM={nM}')

mal_cnot = {}
mal_init = []
gre = re.compile(r'^\s+(actrl|atgt)(p\d)\s+([XZ])\(([\d.,\s-]+)\) data '
                 r'\(([\d.,\s-]+)\) t=[\d.]+: (\d+) classes')
ire = re.compile(r'^\s+init flip \((\d+), (\d+)\)')
for line in out.splitlines():
    m = gre.match(line)
    if m:
        tag = {'p0': 'r', 'p1': 'v0'}[m.group(2)]
        typ = m.group(3)
        pos = tuple(float(x) for x in m.group(4).split(','))
        dc = tuple(float(x) for x in m.group(5).split(','))
        mal_cnot[(tag, typ, pos, dc)] = ['x'] * int(m.group(6))
        continue
    m = ire.match(line)
    if m:
        mal_init.append((int(m.group(1)), int(m.group(2))))
r1 = sum(len(v) for k, v in mal_cnot.items() if k[0] == 'r')
r2 = sum(len(v) for k, v in mal_cnot.items() if k[0] == 'v0')
print(f'parsed: round1={r1}, round2={r2}, inits={mal_init}')
assert r1 + r2 == n2 and len(mal_init) == nI

# regenerate the figure: reuse gen_circuit_full's drawing code with
gen_src = open(os.path.join(SRC, 'gen_circuit_full.py')).read()
gen_src = gen_src.replace(
    'from iter_li_pergate import pergate\n', '')
gen_src = gen_src.replace(
    'mal_cnot, mal_init = pergate(kinds, d)\n',
    'mal_cnot, mal_init = DECODED_CNOT, DECODED_INIT\n')
gen_src = gen_src.replace(
    "'..', '..', 'arXiv-2501.15566v5_draft_post_LC_read', "
    "'fig_circuit_full.tex'",
    "'..', 'fig_circuit_full_decoded.tex'")
assert 'DECODED_CNOT' in gen_src and 'fig_circuit_full_decoded' in gen_src
g = {'DECODED_CNOT': mal_cnot, 'DECODED_INIT': mal_init,
     '__file__': os.path.join(HERE, 'step6_flatfig_decoded.py'),
     '__name__': '__main__'}
exec(compile(gen_src, 'gen_circuit_full_decoded', 'exec'), g)
