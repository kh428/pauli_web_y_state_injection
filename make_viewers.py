"""Generate the interactive 3D HTML viewers for the injection Pauli webs.

Writes injection_webs_viewer.html: pick a graph (1 round / 2 rounds), then toggle any
combination of the 24 check webs on and off.
"""
import itertools, random, time
import numpy as np
from injection_webs import (build, web_system, nullspace, rref, vec_to_web, audit,
                            covers, detectors, XCHECK, ZCHECK, init_kind, versions)
from viewer import write_multi_viewer


class System:
    """web_system built once per graph, then solved repeatedly with different constraints."""

    def __init__(self, g, V):
        self.g, self.V = g, V
        self.A, self.edges, self.xv, self.zv, self.ncols = web_system(g)

    def out_edges(self):
        return [tuple(sorted((self.V[('out', i)],
                              next(iter(self.g.neighbors(self.V[('out', i)]))))))
                for i in range(25)]

    def no_outputs(self):
        f = {}
        for e in self.out_edges():
            f[self.xv[e]] = 0
            f[self.zv[e]] = 0
        return f

    def cover(self, kind, rd, idx, bit=1):
        v = self.V[('bx', rd, idx) if kind == 'X' else ('az', rd, idx)]
        var, other = (self.xv, self.zv) if kind == 'X' else (self.zv, self.xv)
        f = {}
        for w in self.g.neighbors(v):
            e = tuple(sorted((v, w)))
            f[var[e]] = bit
            f[other[e]] = 0
        return f

    def solve(self, fixed):
        rows, rhs = [self.A], [np.zeros(self.A.shape[0], np.uint8)]
        for col, bit in fixed.items():
            r = np.zeros(self.ncols, np.uint8)
            r[col] = 1
            rows.append(r[None, :])
            rhs.append(np.array([bit], np.uint8))
        M = np.vstack(rows)
        b = np.concatenate(rhs)
        R, piv = rref(np.hstack([M, b[:, None]]))
        if self.ncols in piv:
            return None
        part = np.zeros(self.ncols, np.uint8)
        for r, c in enumerate(piv):
            part[c] = R[r, self.ncols]
        return part, nullspace(M)

    def weight(self, v):
        return sum(1 for e in self.edges if v[self.xv[e]] or v[self.zv[e]])

    def small(self, part, basis, tries=3000, seed=0):
        best, bw = part.copy(), self.weight(part)
        for k in (1, 2):
            for combo in itertools.combinations(range(len(basis)), k):
                v = part.copy()
                for c in combo:
                    v = v ^ basis[c]
                w = self.weight(v)
                if w < bw:
                    best, bw = v, w
        rng = random.Random(seed)
        for _ in range(tries):
            if not len(basis):
                break
            v = best ^ basis[rng.randrange(len(basis))]
            w = self.weight(v)
            if w < bw:
                best, bw = v, w
        return best, bw

    def web(self, v):
        return vec_to_web(v, self.edges, self.xv, self.zv)


ALL = [('X', j) for j in sorted(XCHECK)] + [('Z', k) for k in sorted(ZCHECK)]


def wrong_basis(kind, qs):
    want = 'plus' if kind == 'X' else 'zero'
    return sum(1 for i in qs if init_kind(i) != want)


def main():
    print('versions:', versions())
    t0 = time.time()
    
    # ---------------------------------------------------------------- page 1: one round
    g1, V1, m1 = build(rounds=1)
    S1 = System(g1, V1)
    page1, n_det = [], 0
    for kind, j in ALL:
        qs = (XCHECK if kind == 'X' else ZCHECK)[j]
        f = dict(S1.no_outputs())
        f.update(S1.cover(kind, 1, j))
        res = S1.solve(f)                       # is it a detector?
        det = res is not None
        if not det:                             # else the forward-only web
            res = S1.solve(S1.cover(kind, 1, j))
        if res is None:
            continue
        v, bw = S1.small(*res, seed=j)
        w = S1.web(v)
        assert not audit(g1, w, m1), (kind, j)
        n_det += det
        tip = (f'{kind}-check {j} on qubits {qs}. '
               + (f'Round-1 detector: web reaches back to the initial states. {bw} edges.'
                  if det else
                  f'NOT a round-1 detector ({wrong_basis(kind, qs)} wrong-basis qubits): the web '
                  f'only runs forward to the outputs. {bw} edges.'))
        page1.append((f'{kind}{j}', w, det, tip))
    
    print(f'round 1: {n_det} detectors, {len(page1) - n_det} forward-only  '
          f'({time.time() - t0:.0f}s)')
    
    # ---------------------------------------------------------------- page 2: two rounds
    g2, V2, m2 = build(rounds=2)
    S2 = System(g2, V2)
    page2 = []
    for kind, j in ALL:
        qs = (XCHECK if kind == 'X' else ZCHECK)[j]
        f = dict(S2.no_outputs())
        f.update(S2.cover(kind, 1, j))
        f.update(S2.cover(kind, 2, j))
        res = S2.solve(f)
        if res is None:
            print(f'  !! {kind}{j}: no round-1 x round-2 detector')
            continue
        v, bw = S2.small(*res, seed=100 + j)
        w = S2.web(v)
        assert not audit(g2, w, m2), (kind, j)
        page2.append((f'{kind}{j}', w, True,
                      f'{kind}-check {j} on qubits {qs}. Round-1 x round-2 detector: closes '
                      f'between the two rounds, touching neither the initial states nor the '
                      f'outputs. {bw} edges.'))
    print(f'round 2: {len(page2)} round-1 x round-2 full cells  ({time.time() - t0:.0f}s)')
    
    # the init-anchored HALF cells also live on the 2-round graph: initial states -> round 1,
    # never reaching round 2. dim(detectors, 2 rounds) = 24 full + 10 half = 34, so together
    # these toggles span the whole detector space.
    page2h = []
    for kind, j in ALL:
        qs = (XCHECK if kind == 'X' else ZCHECK)[j]
        f = dict(S2.no_outputs())
        f.update(S2.cover(kind, 1, j))
        f.update(S2.cover(kind, 2, j, bit=0))       # forbid the round-2 copy -> a half cell
        res = S2.solve(f)
        if res is None:
            continue
        v, bw = S2.small(*res, seed=200 + j)
        w = S2.web(v)
        assert not audit(g2, w, m2), (kind, j)
        page2h.append((f'{kind}{j} init', w, True,
                       f'{kind}-check {j} on qubits {qs}. Init-anchored HALF cell: the web runs '
                       f'from the initial states up to round 1 only. {bw} edges.'))
    print(f'round 2: {len(page2h)} init-anchored half cells  ({time.time() - t0:.0f}s)')
    
    
    def split(items):
        return [('X-type checks (green ancillas, measured in X)',
                 [it for it in items if it[0][0] == 'X']),
                ('Z-type checks (red ancillas, measured in Z)',
                 [it for it in items if it[0][0] == 'Z'])]
    
    
    # preset selections. page1 is ordered X0..X11 then Z0..Z11, so index == position.
    nodet = [i for i, (_, _, det, _) in enumerate(page1) if not det]
    isdet = [i for i, (_, _, det, _) in enumerate(page1) if det]
    presets1 = [(f'all {len(nodet)} with NO round-1 detector', nodet),
                (f'the {len(isdet)} with one (= the post-selection set)', isdet),
                ('all 24', list(range(len(page1)))),
                ('none', [])]
    # page2 is [24 full cells | half cells]; the full cells share page1's ordering
    nfull = len(page2)
    half_ix = list(range(nfull, nfull + len(page2h)))
    presets2 = [(f'the {len(nodet)} full cells that had NO round-1 detector', nodet),
                (f'the {len(isdet)} full cells that already had one', isdet),
                (f'the {len(page2h)} init-anchored half cells', half_ix),
                (f'all {nfull} full cells', list(range(nfull))),
                (f'everything ({nfull + len(page2h)} = the whole detector space)',
                 list(range(nfull + len(page2h)))),
                ('none', [])]
    print(f'presets: {len(nodet)} non-detectors, {len(isdet)} detectors')
    
    rows = [['page', 'what a button shows'],
            ['1 round', 'the web of one check. Green outline = it has a round-1 detector '
                        '(10 of 24). Blue = it does not, so the web shown is the forward-only '
                        'one that runs to the outputs and touches no initial state.'],
            ['2 rounds', 'the round-1 x round-2 detector of one check. All 24 exist: this is '
                         'what the second round buys. The presets keep the same 14/10 split, so '
                         '"the 14" here is exactly the set of detectors that exist ONLY because '
                         'of the second round. The half cells (initial states -> round 1) are '
                         'listed separately: 24 full + 10 half = 34 = the full detector space.']]
    
    out = write_multi_viewer(
        'injection_webs_viewer.html',
        'Pauli webs of the |Y> state injection, d=5 rotated surface code',
        'Click any number of checks to overlay their webs. Green web = Z-type decoration, '
        'red = X-type. Measurement outcomes are classical, so no measurement legs are drawn: '
        'a check enters a web exactly when the web covers its ancilla on every leg. Time runs '
        'upward. Drag to orbit, scroll to zoom, drag a spider to move it.',
        [('r1', '1 round', g1, split(page1), presets1),
         ('r2', '2 rounds', g2,
          split(page2) + [('init-anchored half cells (initial states -> round 1)', page2h)],
          presets2)],
        rows)
    print('wrote', out, f'({time.time() - t0:.0f}s total)')


if __name__ == '__main__':
    main()
