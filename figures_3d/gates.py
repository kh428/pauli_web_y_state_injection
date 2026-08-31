"""Gate-level (unrolled CNOT-ladder) syndrome extraction for the web solver.

Each check gets its own ancilla WIRE: init spider, one 2q-tap per support
cell at its scheduled sub-tick, then a post-selected measurement cap.
X-check: ancilla |+> (green init), CNOT ancilla->data (green ctrl on the
ancilla wire, red tap on the data wire), measure X (green cap).
Z-check: ancilla |0> (red init), CNOT data->ancilla (green tap on data,
red target on ancilla), measure Z (red cap).
A hook rotation is a green pi/2 spider spliced into an ancilla wire between
two sub-ticks (role 'rot'); edges incident to it are 1q rotation slots."""
import sys, os
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'vendor'))

from pyzx import VertexType
from physical import SpacetimeBuilder, rect, region_checks

GREEN, RED, B = VertexType.Z, VertexType.X, VertexType.BOUNDARY


class GateBuilder(SpacetimeBuilder):
    def gate_round(self, cells, tag='', schedules=None, rot=None):
        """schedules: [(typ, anc_pos, [cells in CNOT order])]; default =
        region_checks with sorted supports. rot: (check_idx, after_k)."""
        if schedules is None:
            # standard N/Z interleaved scheduling of the rotated code:
            # X-checks visit their support in the "Z" pattern NW,NE,SW,SE;
            # Z-checks in the transposed "N" pattern NW,SW,NE,SE. Boundary
            # (weight-2) checks keep their ABSOLUTE slots and idle in
            # the others (matching the stim builder; collision-free).
            xs, zs = region_checks(cells)
            def sched(pos, sup, typ):
                fi, fj = pos[0] - 0.5, pos[1] - 0.5
                NW = (fi, fj + 1); NE = (fi + 1, fj + 1)
                SW = (fi, fj);     SE = (fi + 1, fj)
                order = ([NW, SW, NE, SE] if typ == 'X'
                         else [NW, NE, SW, SE])
                return [c if c in set(sup) else None for c in order]
            schedules = ([('X', pos, sched(pos, sup, 'X')) for pos, sup in xs] +
                         [('Z', pos, sched(pos, sup, 'Z')) for pos, sup in zs])
        t0 = self.t
        anc = {}
        for ci, (typ, pos, sup) in enumerate(schedules):
            ty = GREEN if typ == 'X' else RED
            self.t = t0
            v = self.V(ty, pos[0], pos[1], 0, f'ancinit_{typ}{tag}', ('anc', typ, pos))
            anc[ci] = v
        nmax = max(len(s[2]) for s in schedules)
        for k in range(nmax):
            # serialised collisions: when two checks tap the same data
            # cell in one slot, each later gate moves to its own
            # sub-tick, so no two spiders share a coordinate and every
            # CNOT stays level in time
            seen = {}
            for ci, (typ, pos, sup) in enumerate(schedules):
                if k >= len(sup):
                    continue
                cell = sup[k]
                if cell is None:
                    continue
                lvl = seen.get(cell, 0)
                seen[cell] = lvl + 1
                self.t = t0 + 1 + k + 0.45 * lvl
                if typ == 'X':
                    a = self.V(GREEN, pos[0], pos[1], 0, f'actrl{tag}', ('anc', typ, pos))
                    d_ = self.V(RED, *cell, 0, f'dtap{tag}', cell)
                else:
                    a = self.V(RED, pos[0], pos[1], 0, f'atgt{tag}', ('anc', typ, pos))
                    d_ = self.V(GREEN, *cell, 0, f'dtap{tag}', cell)
                self.g.add_edge((anc[ci], a))
                self.g.add_edge((self.frontier[cell], d_))
                self.g.add_edge((a, d_))
                anc[ci] = a
                self.frontier[cell] = d_
                if rot is not None and rot[0] == ci and rot[1] == k + 1:
                    # hook = X-rotation on the ancilla: conjugates through the
                    # REMAINING CNOTs onto the not-yet-touched support cells
                    self.t += 0.4
                    r = self.V(RED, pos[0] + 0.3, pos[1] + 0.3,
                               Fraction(1, 2), 'rot', ('anc', typ, pos))
                    self.g.add_edge((anc[ci], r))
                    anc[ci] = r
                    self.t -= 0.4
        self.t = t0 + 1 + nmax
        for ci, (typ, pos, sup) in enumerate(schedules):
            ty = GREEN if typ == 'X' else RED
            cap = self.V(ty, pos[0], pos[1], 0, f'ancmeas_{typ}{tag}', ('anc', typ, pos))
            self.g.add_edge((anc[ci], cap))
        self.t += 1
