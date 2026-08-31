"""3D spacetime ZX-diagram of the gate-level (CNOT-unrolled) corner
scheme at d=3: init layer, two N/Z-scheduled extraction rounds with
every ancilla an explicit wire, readout caps, open outputs. Emitted in
the paper 3D style with smaller nodes (dense diagram)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iter_li_counting import build
from physical import rect, injection_kind
from paper3d import emit

d = 3
kinds = {c: injection_kind(c, 0, 0, d) for c in rect(0, 0, d, d)}
g, meta = build(kinds, d)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'arXiv-2501.15566v5_draft_post_LC_read')
reps = {}
for v in g.vertices():
    r = meta[v]['role']
    if r.startswith('ancinit_'):
        reps.setdefault(r[-1] if not r.endswith('v0') else 'v0', v)
planes = list(reps.values())
emit(g, [], os.path.join(OUT, 'fig_circuit3d.tex'),
     zscale=2.2, node_size='0.4cm', planes=planes)
