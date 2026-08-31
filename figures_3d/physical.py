"""Physical-qubit surface-code spacetime builder, generalising the v4 injection model.

A rotated surface code on ANY axis-aligned rectangle of data qubits, with the same
global 2-colouring as the paper's d=5 tables (face at (i+.5, j+.5) is X-type iff i+j
is even; a weight-2 boundary face exists iff its type matches that side's boundary
type: N/S boundaries are X (smooth), E/W are Z (rough) in the standard orientation).
Because the colouring is global in (column,row), two patches placed an even number of
columns apart merge into one valid rectangle code -- which is exactly how the rough
merge is built here (seam column of fresh qubits, one round of the merged code).

Reproduces the v4 XCHECK/ZCHECK tables exactly for the 5x5 patch at origin (asserted
in phase_d1). All coordinates are GLOBAL (column, row) pairs.
"""
import sys, os
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vendor'))
import pyzx as zx
from pyzx import VertexType

GREEN, RED, B = VertexType.Z, VertexType.X, VertexType.BOUNDARY


def _keep_face(present, typ, xs, zs, fi, fj, bus):
    if len(present) == 4:
        (xs if typ == 'X' else zs).append(((fi + .5, fj + .5), present))
    elif len(present) == 2:
        a, b = present
        in_bus = a in bus and b in bus
        if a[0] == b[0]:                       # vertical pair: E or W exposure
            # rough for patches and for the bus ends/sides
            keep = (typ == 'Z')
        elif a[1] == b[1]:                     # horizontal pair: N or S exposure
            if not in_bus:
                keep = (typ == 'X')            # patch N/S: smooth
            else:
                # DIRECTION-AWARE bus boundary (the D2b resolution candidate):
                #   N exposure (gap segments facing fenced/spectator columns):
                #     SMOOTH -- otherwise Z-strings terminate there and fragment
                #     the joint product (the certified multi-contact failure);
                #   S exposure (the far side): ROUGH -- so individual X-strings
                #     entering through a window cannot terminate.
                north = (a[1] == fj)           # pair on the LOWER row -> face above = N
                keep = (typ == 'X') if north else (typ == 'Z')
        else:
            keep = False
        if keep:
            (xs if typ == 'X' else zs).append(((fi + .5, fj + .5), present))


def region_checks(cells, bus_cells=(), fences=()):
    """Checks of the rotated code on an arbitrary set of data cells (global colouring).
    Returns (xchecks, zchecks): lists of (face_pos, [cells]). Weight-4 checks appear
    wherever all 4 face cells are present; weight-2 boundary checks appear where
    exactly 2 adjacent cells are present AND the face type matches the exposed side:
    an X-type face is kept on a N/S exposure, a Z-type face on an E/W exposure."""
    cells = set(map(tuple, cells))
    bus = set(map(tuple, bus_cells))
    fence = {frozenset(p) for p in fences}     # adjacent cell-pairs with NO coupling
    xs, zs = [], []
    n3 = 0
    seen = set()
    for (c, r) in cells:
        for fi in (c - 1, c):
            for fj in (r - 1, r):
                if (fi, fj) in seen:
                    continue
                seen.add((fi, fj))
                quad = [(fi, fj), (fi, fj + 1), (fi + 1, fj), (fi + 1, fj + 1)]
                present = [q for q in quad if q in cells]
                typ = 'X' if (fi + fj) % 2 == 0 else 'Z'
                # a face straddling a fence splits into fence-connected components;
                # EACH component is evaluated as its own (boundary) face
                comps = []
                for q in present:
                    placed = False
                    for comp in comps:
                        if any(frozenset((q, a)) not in fence
                               and abs(q[0] - a[0]) + abs(q[1] - a[1]) == 1
                               for a in comp):
                            comp.append(q)
                            placed = True
                            break
                    if not placed:
                        comps.append([q])
                merged_comps = True
                while merged_comps:            # merge comps joined via a third cell
                    merged_comps = False
                    for i in range(len(comps)):
                        for j in range(i + 1, len(comps)):
                            if any(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1
                                   and frozenset((a, b)) not in fence
                                   for a in comps[i] for b in comps[j]):
                                comps[i] += comps[j]
                                del comps[j]
                                merged_comps = True
                                break
                        if merged_comps:
                            break
                for present in comps:
                    if len(present) == 3:
                        n3 += 1                # concave corner: hosts NO check
                        continue
                    _keep_face(present, typ, xs, zs, fi, fj, bus)
                continue
    region_checks.n3 = n3                     # dropped concave-corner faces
    return xs, zs


def rect(x0, y0, w, h):
    return [(x0 + i, y0 + j) for i in range(w) for j in range(h)]


def injection_kind(cell, x0, y0, d, corner='Y'):
    """Lao-Criger pattern on a d x d patch with origin (x0, y0) (v4 conventions):
    |Y> on the top-left corner, |+> on and below the anti-diagonal, |0> above."""
    c, r = cell[0] - x0, cell[1] - y0
    if (c, r) == (0, d - 1):
        return corner
    return '+' if c + r <= d - 1 else '0'


INIT = {'0': (RED, 0), '+': (GREEN, 0), 'Y': (GREEN, Fraction(1, 2))}
CAP = {'X': (GREEN, 0), 'Z': (RED, 0)}       # <+| green, <0| red (post-selected)


class SpacetimeBuilder:
    """Builds the physical spacetime ZX round by round.

    Convention (v4): each full round = X-check layer (red data spiders, green check
    hubs) then Z-check layer (green data spiders, red check hubs); measurement
    outcomes classical (no legs); a check enters a web iff covered on every leg."""

    def __init__(self):
        self.g = zx.Graph()
        self.meta = {}
        self.frontier = {}          # cell -> dangling vertex
        self.t = 0.0

    def V(self, ty, x, y, ph=0, role='', cell=None):
        v = self.g.add_vertex(ty, qubit=y, row=x, phase=ph)
        self.g.set_vdata(v, 'z', self.t)
        self.meta[v] = dict(cell=cell, role=role, t=self.t)
        return v

    def open_inputs(self, cells):
        """Start these worldlines as OPEN input boundaries (channel semantics)."""
        for cell in cells:
            assert cell not in self.frontier, cell
            self.frontier[cell] = self.V(B, *cell, 0, 'in', cell)
        self.t += 1

    def init_cells(self, kinds):
        """kinds: {cell: '0'|'+'|'Y'}"""
        for cell, k in kinds.items():
            ty, ph = INIT[k]
            assert cell not in self.frontier, cell
            self.frontier[cell] = self.V(ty, *cell, ph, f'init_{k}', cell)
        self.t += 1

    def round(self, cells, tag='', bus_cells=(), fences=()):
        """One full round (X layer then Z layer) of the code on `cells`."""
        xs, zs = region_checks(cells, bus_cells, fences)
        for layer, checks, dty, hty in (('x', xs, RED, GREEN), ('z', zs, GREEN, RED)):
            taps = {}
            for cell in cells:
                tap = self.V(dty, *cell, 0, f'data_{layer}{tag}', cell)
                self.g.add_edge((self.frontier[cell], tap))
                self.frontier[cell] = tap
                taps[cell] = tap
            for pos, sup in checks:
                hub = self.V(hty, pos[0], pos[1], 0, f'chk_{layer}{tag}', None)
                for cell in sup:
                    self.g.add_edge((taps[cell], hub))
            self.t += 1

    def measure_cells(self, cells, basis):
        """Destructive transversal measurement, post-selected (+1 caps)."""
        for cell in cells:
            ty, ph = CAP[basis]
            cap = self.V(ty, *cell, ph, f'meas_{basis}', cell)
            self.g.add_edge((self.frontier.pop(cell), cap))
        self.t += 1

    def open_outputs(self, cells):
        for cell in cells:
            b = self.V(B, *cell, 0, 'out', cell)
            self.g.add_edge((self.frontier.pop(cell), b))

    def finish(self):
        assert not self.frontier, f'dangling wires: {sorted(self.frontier)}'
        return self.g, self.meta
