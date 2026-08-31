"""Parse a (noiseless) stim/tsim circuit text into a PyZX graph with
states plugged in and measurement outcomes postselected, then contract
it exactly. Supported ops: R RX S T S_DAG T_DAG H X Y Z CX CZ M MX;
rec-conditioned gates are resolved against the chosen outcome string
(here: the all-zero record, so they are dropped). Returns amplitudes,
so conditional probabilities come from ratios and no global
normalisation bookkeeping is needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
import numpy as np
import pyzx as zx
from pyzx import VertexType, EdgeType

PHASE = {'S': Fraction(1, 2), 'T': Fraction(1, 4),
         'S_DAG': Fraction(3, 2), 'T_DAG': Fraction(7, 4),
         'Z': Fraction(1), 'X': Fraction(1), 'Y': None}


def parse(txt, outcomes=None):
    """outcomes: dict measurement-index -> bit (default all zero)."""
    outcomes = outcomes or {}
    g = zx.Graph()
    last = {}          # qubit -> (vertex, pending hadamard)
    row = [0]

    def V(ty, q, phase=0):
        row[0] += 1
        v = g.add_vertex(ty, qubit=q, row=row[0], phase=phase)
        if q in last:
            u, had = last[q]
            g.add_edge((u, v), EdgeType.HADAMARD if had
                       else EdgeType.SIMPLE)
        last[q] = (v, False)
        return v

    nmeas = 0
    for line in txt.split('\n'):
        parts = line.split()
        if not parts: continue
        op = parts[0].split('(')[0]
        if 'rec[' in line:
            continue                        # all-zero record: never fires
        if op in ('TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS'):
            continue
        args = [int(a) for a in parts[1:]] if op != 'MPP' else []
        if op == 'R':
            for q in args: last.pop(q, None); V(VertexType.X, q)
        elif op == 'RX':
            for q in args: last.pop(q, None); V(VertexType.Z, q)
        elif op in ('S', 'T', 'S_DAG', 'T_DAG', 'Z'):
            for q in args: V(VertexType.Z, q, PHASE[op])
        elif op == 'X':
            for q in args: V(VertexType.X, q, Fraction(1))
        elif op == 'Y':
            for q in args:
                V(VertexType.Z, q, Fraction(1))
                V(VertexType.X, q, Fraction(1))
        elif op == 'H':
            for q in args:
                u, had = last[q]
                last[q] = (u, not had)
        elif op == 'CX':
            c, t = args
            vc = V(VertexType.Z, c)
            vt = V(VertexType.X, t)
            g.add_edge((vc, vt), EdgeType.SIMPLE)
        elif op == 'CZ':
            a, b = args
            va = V(VertexType.Z, a)
            vb = V(VertexType.Z, b)
            g.add_edge((va, vb), EdgeType.HADAMARD)
        elif op == 'M':
            for q in args:
                b = outcomes.get(nmeas, 0); nmeas += 1
                V(VertexType.X, q, Fraction(b))
                del last[q]
        elif op == 'MX':
            for q in args:
                b = outcomes.get(nmeas, 0); nmeas += 1
                V(VertexType.Z, q, Fraction(b))
                del last[q]
        else:
            raise ValueError(f'unsupported op {op!r}')
    assert not last, f'open wires remain: {sorted(last)}'
    return g, nmeas


def amplitude(g):
    g2 = g.copy()
    zx.full_reduce(g2)
    if g2.num_vertices():
        return complex(zx.tensorfy(g2, preserve_scalar=True))
    return g2.scalar.to_number()


if __name__ == '__main__':
    # sanity: RX, T then T-basis measurement outcome 0 -> P(0)=1
    a0 = amplitude(parse('RX 0\nT 0\nT_DAG 0\nH 0\nM 0', {0: 0})[0])
    a1 = amplitude(parse('RX 0\nT 0\nT_DAG 0\nH 0\nM 0', {0: 1})[0])
    print('sanity P(1) =', abs(a1)**2 / (abs(a0)**2 + abs(a1)**2))
    # damage weights by pure contraction: X fault before T-basis readout
    for prep, name in [('RX 0\nS 0', '|Y>'), ('RX 0\nT 0', '|T>')]:
        und = 'S_DAG' if 'S 0' in prep else 'T_DAG'
        amps = []
        for b in (0, 1):
            txt = f'{prep}\nX 0\n{und} 0\nH 0\nM 0'
            amps.append(abs(amplitude(parse(txt, {0: b})[0]))**2)
        print(f'{name}: P(flip) under X fault = {amps[1]/sum(amps):.4f}')
