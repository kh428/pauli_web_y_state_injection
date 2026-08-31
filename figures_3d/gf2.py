"""Bit-packed GF(2) linear algebra for large web systems (10^4+ unknowns).

Rows are packed 64 columns per uint64 word. API mirrors injection_webs:
rref, nullspace, and the affine solve used by solve_with.
"""
import numpy as np

W = 64


def pack(rows, ncols):
    nw = (ncols + W - 1) // W
    M = np.zeros((len(rows), nw), dtype=np.uint64)
    for i, r in enumerate(rows):
        for c in np.nonzero(r)[0]:
            M[i, c // W] |= np.uint64(1) << np.uint64(c % W)
    return M


def bit(M, i, c):
    return int(M[i, c // W] >> np.uint64(c % W) & np.uint64(1))


def bit1(v, c):
    """bit c of a 1-D packed vector"""
    return int(v[c // W] >> np.uint64(c % W) & np.uint64(1))


def setbit(v, c):
    v[c // W] |= np.uint64(1) << np.uint64(c % W)


def rref_packed(M, ncols):
    """In-place packed rref. Returns (M_reduced_rows, pivots)."""
    M = M.copy()
    nrows = M.shape[0]
    piv = []
    r = 0
    for c in range(ncols):
        w, b = c // W, np.uint64(1) << np.uint64(c % W)
        col = (M[r:, w] & b) != 0
        hits = np.nonzero(col)[0]
        if len(hits) == 0:
            continue
        p = r + hits[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        mask = (M[:, w] & b) != 0
        mask[r] = False
        if mask.any():
            M[mask] ^= M[r]
        piv.append(c)
        r += 1
        if r == nrows:
            break
    return M[:r], piv


def nullspace_packed(M, ncols):
    """Basis of the null space, as packed rows."""
    R, piv = rref_packed(M, ncols)
    pivset = set(piv)
    free = [c for c in range(ncols) if c not in pivset]
    nw = (ncols + W - 1) // W
    basis = np.zeros((len(free), nw), dtype=np.uint64)
    for k, f in enumerate(free):
        setbit(basis[k], f)
        for r_i, c in enumerate(piv):
            if bit(R, r_i, f):
                setbit(basis[k], c)
    return basis


def solve_affine_packed(A_rows, ncols, fixed):
    """Solve A x = 0 with fixed {col: bit}. Returns (particular, basis) packed,
    or None if inconsistent. A_rows: packed matrix."""
    nw = (ncols + W - 1) // W
    extra = np.zeros((len(fixed), nw + 1), dtype=np.uint64)
    for i, (c, b) in enumerate(sorted(fixed.items())):
        setbit(extra[i, :nw], c)
        if b:
            extra[i, nw] |= np.uint64(1)
    Aug = np.zeros((A_rows.shape[0] + len(fixed), nw + 1), dtype=np.uint64)
    Aug[:A_rows.shape[0], :nw] = A_rows
    Aug[A_rows.shape[0]:] = extra
    # treat the rhs as column index ncols_aug-1 = nw*W (bit 0 of the last word)
    ncols_aug = nw * W + 1
    R, piv = rref_packed(Aug, ncols_aug)
    rhs_col = nw * W
    if rhs_col in piv:
        return None
    part = np.zeros(nw, dtype=np.uint64)
    for r_i, c in enumerate(piv):
        if c < ncols and bit(R, r_i, rhs_col):
            setbit(part, c)
    hom = Aug[:, :nw].copy()          # homogeneous part incl. the fixed rows
    basis = nullspace_packed(hom, ncols)
    return part, basis


def unpack_support(v, ncols):
    """Indices of set bits."""
    out = []
    for w in range(len(v)):
        word = int(v[w])
        while word:
            b = word & -word
            out.append(w * W + b.bit_length() - 1)
            word ^= b
    return [c for c in out if c < ncols]
