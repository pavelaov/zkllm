"""Multilinear-extension helpers over Goldilocks/Ext2 (shared math).

Index convention matches edgecert.sumfold: variables are bound LSB-first, so for a table v of length 2^k,  v~(r) = sum_i v[i] * eq(i, r)  with eq(i, r) = prod_t (r_t if bit_t(i) else 1 - r_t), and sumfold.fold(v, r_0) binds bit 0 first.

Matrices are row-major lists of rows of signed ints (real, unpadded dimensions); padding to powers of two is implicit zeros.  For a matrix M of shape (R x C) padded to (2^a x 2^b):  M~(x, y) = sum_{i,j} M[i][j] eq(i,x) eq(j,y).
"""
from __future__ import annotations

from typing import List, Sequence

from ..field import P, Ext, ONE, ZERO, eadd, emul, esub

def log2_ceil(n: int) -> int:
    return 0 if n <= 1 else (n - 1).bit_length()

def eq_table(r: Sequence[Ext]) -> List[Ext]:
    tbl: List[Ext] = [ONE]
    for rt in r:
        om = esub(ONE, rt)
        tbl = [emul(e, om) for e in tbl] + [emul(e, rt) for e in tbl]
    return tbl

def _acc_ext(pairs) -> Ext:
    """sum of (int m) * (Ext e) with a single final reduction."""
    a0 = 0
    a1 = 0
    for m, e in pairs:
        a0 += m * e[0]
        a1 += m * e[1]
    return (a0 % P, a1 % P)

def contract_rows(M: Sequence[Sequence[int]], eq_rows: Sequence[Ext], ncols_padded: int) -> List[Ext]:
    """f[j] = sum_i M[i][j] eq_rows[i]   (length ncols_padded, zero beyond real cols)."""
    ncols = len(M[0]) if M else 0
    out: List[Ext] = []
    for j in range(ncols):
        out.append(_acc_ext((M[i][j], eq_rows[i]) for i in range(len(M))))
    out += [ZERO] * (ncols_padded - ncols)
    return out

def contract_cols(M: Sequence[Sequence[int]], eq_cols: Sequence[Ext], nrows_padded: int) -> List[Ext]:
    """g[i] = sum_j M[i][j] eq_cols[j]   (length nrows_padded)."""
    out: List[Ext] = [_acc_ext(zip(row, eq_cols)) for row in M]
    out += [ZERO] * (nrows_padded - len(M))
    return out

def dot(u: Sequence[Ext], v: Sequence[Ext]) -> Ext:
    acc = ZERO
    for a, b in zip(u, v):
        acc = eadd(acc, emul(a, b))
    return acc

def mle_eval_vector(v: Sequence[Ext], r: Sequence[Ext]) -> Ext:
    return dot(v, eq_table(r))

def mle_eval_matrix(M: Sequence[Sequence[int]], r_rows: Sequence[Ext], r_cols: Sequence[Ext]) -> Ext:
    er = eq_table(r_rows)
    ec = eq_table(r_cols)
    g = contract_cols(M, ec, len(M))
    return dot(g, er[:len(M)])
