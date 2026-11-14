"""Exact integer operators of the EDGE-CERT authorized integer derivative.

Normative description: spec/numerical_semantics.md.  Every function here is exact numpy int64 arithmetic (no floating point touches proof-critical data). Each operator returns its output together with a ``GadgetRecord`` holding

* ``committed``: the witness tensors the prover commits for this gadget,
* ``checks``:    the raw values that must pass a range check; the contract
                 supplies ``(offset, bits)`` and the check is
                 ``0 <= value + offset < 2**bits``,
* ``lookups``:   number of table lookups per committed table,
* ``aux``:       values used only by the relation checker / tests.

Magnitude discipline: every intermediate is asserted to stay below 2**62 so that int64 never wraps (the static TileContract checker proves the same bound symbolically, see src/edgecert/compiler/tile_contract.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

I64 = np.int64
LIM62 = 1 << 62

def ubits(x: int) -> int:
    """Number of bits needed for the non-negative integer range [0, x]."""
    x = int(x)
    assert x >= 0
    return max(1, x.bit_length())

def _chk(a: np.ndarray, what: str) -> None:
    if a.size and int(np.max(np.abs(a))) >= LIM62:
        raise OverflowError(f"{what}: |value| >= 2^62, int64 reference would be unsafe")

def as_i64(a) -> np.ndarray:
    return np.asarray(a, dtype=I64)

@dataclass
class GadgetRecord:
    kind: str
    tag: str
    committed: Dict[str, np.ndarray] = field(default_factory=dict)
    checks: Dict[str, np.ndarray] = field(default_factory=dict)
    lookups: Dict[str, int] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)
    aux: Dict[str, np.ndarray] = field(default_factory=dict)

# ----------------------------------------------------------------- rounding
def rhaz_shift(n: np.ndarray, s: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round-half-away-from-zero of n / 2^s, exactly.

    Returns (y0, sigma, rem) with sigma = [n < 0] and
        n + 2^(s-1) - sigma = y0 * 2^s + rem,   0 <= rem < 2^s.
    """
    assert s >= 1
    n = as_i64(n)
    _chk(n, "rhaz_shift input")
    sigma = (n < 0).astype(I64)
    t = n + (1 << (s - 1)) - sigma
    y0 = t >> s                      # arithmetic shift == floor division by 2^s
    rem = t - (y0 << s)
    return y0, sigma, rem

def rhaz_int(n: int, s: int) -> int:
    """Scalar Python-int reference of RHAZ(n / 2^s) (independent formula for tests)."""
    q = (abs(n) + (1 << (s - 1))) >> s
    return q if n >= 0 else -q

def rhaz_float(x: np.ndarray) -> np.ndarray:
    """RHAZ for floats, used only to *build* committed tables and quantized weights."""
    x = np.asarray(x, dtype=np.float64)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(I64)

# ---------------------------------------------------------------- requant
def requant(v, M, s: int, lo: Optional[int] = None, hi: Optional[int] = None,
            add=None, tag: str = "rq") -> Tuple[np.ndarray, GadgetRecord]:
    """y = clamp(RHAZ(v * M / 2^s) + add, lo, hi)  (clamp omitted when lo is None).

    v: certified integer tensor, M >= 1 (scalar or per-channel on the last axis),
    add: optional integer tensor added after rounding (residual stream).
    """
    v = as_i64(v)
    M = as_i64(M)
    assert np.all(M >= 1), "requant multiplier must be a positive integer"
    n = v * M
    _chk(n, f"{tag}: v*M")
    y0, sigma, rem = rhaz_shift(n, s)
    if add is not None:
        y0 = y0 + as_i64(add)
    sgn = (1 - 2 * sigma) * v - sigma            # >= 0 iff sigma is the true sign bit
    rec = GadgetRecord("rq", tag, params={"s": s, "lo": lo, "hi": hi, "has_add": add is not None})
    rec.aux.update({"v": v, "y0": y0, "M": M, "add": as_i64(add) if add is not None else None})
    if lo is None:
        y = y0
        rec.committed.update({"sigma": sigma, "rem": rem, "y": y})
        rec.checks.update({"sgn": sgn, "rem": rem, "yr": y})
    else:
        c_hi = (y0 > hi).astype(I64)
        c_lo = (y0 < lo).astype(I64)
        y = np.clip(y0, lo, hi)
        sat = c_hi * (y0 - hi - 1) + c_lo * (lo - 1 - y0)
        rec.committed.update({"sigma": sigma, "rem": rem, "clo": c_lo, "chi": c_hi, "y": y})
        rec.checks.update({"sgn": sgn, "rem": rem, "out": y, "sat": sat})
    return y, rec

# ----------------------------------------------------------------- matmul
def matmul(x, w, b=None, tag: str = "mm") -> Tuple[np.ndarray, GadgetRecord]:
    """acc = x @ w.T + b over the integers (x: [..., K], w: [N, K])."""
    x = as_i64(x)
    w = as_i64(w)
    acc = x @ w.T
    if b is not None:
        acc = acc + as_i64(b)
    _chk(acc, tag)
    rec = GadgetRecord("mm", tag, committed={"acc": acc},
                       params={"K": int(x.shape[-1]), "N": int(w.shape[0])})
    return acc, rec

FP64_EXACT_LIMIT = 1 << 53

def matmul_fp64_exact(x, w_int, w_f64=None, b=None, B_x: int = 128, B_w: Optional[int] = None, gemm=None,
                      impl_id: str = "cpu_fp64_exact", spot_checks: int = 2, tag: str = "mm", seed: int = 0
                      ) -> Tuple[np.ndarray, GadgetRecord]:
    """acc = x @ w.T + b over the integers, computed on an *exact FP64 datapath*.

    Every product x_k w_k is an integer with |x_k w_k| <= B_x B_w and every partial sum
    of any summation order (blocked, pairwise, FMA, split-K on a GPU) is the sum of a
    subset of those products, so |partial| <= K B_x B_w.  If this static TileContract
    bound is <= 2^53, every partial sum is an integer representable in IEEE binary64 and
    the float GEMM computes the exact integer result (FMA rounds an exact integer to
    itself).  The bias is added afterwards in int64.  The function refuses to run when
    the bound is not certified, asserts integrality of the result and spot-checks
    ``spot_checks`` random entries against int64 dot products.
    """
    x = as_i64(x)
    w_int = np.asarray(w_int)                 # any integer dtype (int8/int16 storage is fine)
    assert np.issubdtype(w_int.dtype, np.integer)
    K = int(x.shape[-1])
    if B_w is None:
        B_w = int(np.abs(w_int.astype(I64)).max()) if w_int.size else 0
    bound = K * int(B_x) * int(B_w)
    if bound >= FP64_EXACT_LIMIT:
        raise OverflowError(f"{tag}: static bound K*B_x*B_w = {bound} >= 2^53, FP64 datapath not certified")
    if x.size and int(np.abs(x).max()) > B_x:
        raise OverflowError(f"{tag}: activation outside the certified operand range |x| <= {B_x}")
    wf = w_f64 if w_f64 is not None else w_int.astype(np.float64)
    xf = x.reshape(-1, K).astype(np.float64)
    accf = gemm(xf, wf) if gemm is not None else xf @ wf.T
    acci = np.rint(accf)
    if not np.array_equal(accf, acci):
        raise ArithmeticError(f"{tag}: FP64 datapath returned a non-integer")
    acc = acci.astype(I64).reshape(x.shape[:-1] + (w_int.shape[0],))
    nsc = 0
    if spot_checks and acc.size:
        rng = np.random.default_rng(seed + acc.size)
        x2 = x.reshape(-1, K)
        a2 = acc.reshape(-1, w_int.shape[0])
        for _ in range(spot_checks):
            i = int(rng.integers(0, x2.shape[0]))
            j = int(rng.integers(0, w_int.shape[0]))
            if int(np.dot(x2[i], w_int[j].astype(I64))) != int(a2[i, j]):
                raise ArithmeticError(f"{tag}: FP64 datapath spot check failed at ({i}, {j})")
            nsc += 1
    if b is not None:
        acc = acc + as_i64(b)
    _chk(acc, tag)
    rec = GadgetRecord("mm", tag, committed={"acc": acc},
                       params={"K": K, "N": int(w_int.shape[0]), "implementation_id": impl_id,
                               "static_bound": bound, "spot_checks": nsc})
    return acc, rec

# ------------------------------------------------------------------- isqrt
def isqrt64(X: np.ndarray) -> np.ndarray:
    """floor(sqrt(X)) for 0 <= X < 2^62, exact (float seed + integer fix-up)."""
    X = as_i64(X)
    assert np.all(X >= 0)
    z = np.floor(np.sqrt(X.astype(np.float64))).astype(I64)
    for _ in range(4):
        z = np.where(z * z > X, z - 1, z)
        z = np.where((z + 1) * (z + 1) <= X, z + 1, z)
    assert np.all(z * z <= X) and np.all((z + 1) * (z + 1) > X)
    return z

# -------------------------------------------------------------------- norm
def norm_int(r, kind: str, F: int, G: int, eps_int: int, tag: str = "norm"
             ) -> Tuple[np.ndarray, GadgetRecord]:
    """Integer LayerNorm ('ln') / RMSNorm ('rms') core over the last axis.

    Returns g = c * y where c = r - mu (LN) or r (RMS) and
    y = floor(2^(F+G) / z), z = floor(sqrt((v + eps_int) * 2^(2G))),
    v = floor(sum(c^2) / d), mu = floor(sum(r) / d).
    Real value of g / 2^F approximates (r - mean) / sqrt(var + eps).
    """
    assert eps_int >= 1 and kind in ("ln", "rms")
    r = as_i64(r)
    d = r.shape[-1]
    rec = GadgetRecord("norm", tag, params={"kind": kind, "d": d, "F": F, "G": G, "eps_int": eps_int})
    if kind == "ln":
        S = r.sum(axis=-1)
        mu = S // d                             # floor
        rho_mu = S - mu * d
        c = r - mu[..., None]
        rec.committed.update({"mu": mu, "rho_mu": rho_mu})
        rec.checks.update({"mu": mu, "rho_mu": rho_mu})
        if d & (d - 1):
            rec.checks["rho_mu_hi"] = d - 1 - rho_mu
    else:
        c = r
    Q = (c * c).sum(axis=-1)
    _chk(Q, tag + ": Q")
    v = Q // d
    rho_v = Q - v * d
    X = (v + eps_int) << (2 * G)
    _chk(X, tag + ": X")
    z = isqrt64(X)
    sa = X - z * z
    sb = (z + 1) * (z + 1) - 1 - X
    num = 1 << (F + G)
    y = num // z
    rho_y = num - y * z
    g = c * y[..., None]
    _chk(g, tag + ": g")
    rec.committed.update({"v": v, "rho_v": rho_v, "z": z, "y": y, "rho_y": rho_y, "g": g})
    rec.checks.update({"v": v, "rho_v": rho_v, "z": z, "sa": sa, "sb": sb, "y": y,
                       "rho_y": rho_y, "rho_y_hi": z - 1 - rho_y})
    if d & (d - 1):
        rec.checks["rho_v_hi"] = d - 1 - rho_v
    rec.aux.update({"r": r, "c": c, "Q": Q, "X": X})
    return g, rec

# ----------------------------------------------------------------- softmax
def exp_table(T: int, U_max: int, E_bits: int) -> np.ndarray:
    """Committed table E[u] = RHAZ(2^E_bits * exp(-u / 2^T)), u = 0..U_max."""
    u = np.arange(U_max + 1, dtype=np.float64)
    return rhaz_float((1 << E_bits) * np.exp(-u / (1 << T)))

def softmax_int(t, allowed, table: np.ndarray, Fp: int, tag: str = "softmax"
                ) -> Tuple[np.ndarray, GadgetRecord]:
    """Row-wise integer softmax over the last axis for allowed (unmasked) entries.

    m = max_j t_j (allowed), u = m - t, ub = min(u, U_max), E = table[ub],
    D = sum_j E_j, P = floor(E * 2^Fp / D).  Masked entries give P = 0 and no witness.
    """
    t = as_i64(t)
    allowed = np.asarray(allowed, dtype=bool)
    U_max = len(table) - 1
    big = np.iinfo(I64).min // 4
    m = np.where(allowed, t, big).max(axis=-1)
    assert np.all(allowed.any(axis=-1)), "every softmax row needs one allowed entry"
    u = m[..., None] - t
    c = (u > U_max).astype(I64)
    ub = np.minimum(u, U_max)
    e = np.where(c == 1, u - U_max - 1, U_max - u)
    E = np.where(allowed, table[np.clip(ub, 0, U_max)], 0).astype(I64)
    D = E.sum(axis=-1)
    num = E << Fp
    P = num // D[..., None]
    rho = num - P * D[..., None]
    P = np.where(allowed, P, 0)
    a = allowed
    rec = GadgetRecord("softmax", tag, params={"U_max": U_max, "Fp": Fp})
    rec.committed.update({"c": c[a], "E": E[a], "P": P[a], "rho": rho[a], "m": m, "D": D})
    rec.checks.update({"u": u[a], "e": e[a], "p": P[a], "rho": rho[a],
                       "rho_hi": (D[..., None] - 1 - rho)[a]})
    rec.lookups["T_exp"] = int(a.sum())
    # zero-product witness for the max: every row has some u == 0
    zp = np.where(a, u, 1)
    rec.aux.update({"t": t, "allowed": allowed, "u_rows_have_zero": np.all((zp == 0).any(axis=-1)),
                    "ub": ub[a]})
    return P, rec

# ------------------------------------------------------------ table lookup
def table_lookup(a, table: np.ndarray, lo: int, table_id: str, tag: str = "act"
                 ) -> Tuple[np.ndarray, GadgetRecord]:
    a = as_i64(a)
    idx = a - lo
    if np.any(idx < 0) or np.any(idx >= len(table)):
        raise ValueError(f"{tag}: lookup index outside committed table domain")
    out = table[idx]
    rec = GadgetRecord("act", tag, committed={"out": out}, params={"table": table_id, "lo": lo})
    rec.lookups[table_id] = int(a.size)
    rec.aux["a"] = a
    return out, rec

def act_table(fn, in_lo: int, in_hi: int, in_frac: int, out_scale: float,
              out_lo: int = -128, out_hi: int = 127) -> np.ndarray:
    """Committed activation table T[a - in_lo] = clamp(RHAZ(fn(a / 2^in_frac) / out_scale))."""
    a = np.arange(in_lo, in_hi + 1, dtype=np.float64)
    return np.clip(rhaz_float(fn(a / (1 << in_frac)) / out_scale), out_lo, out_hi).astype(I64)

def gelu_tanh(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

def silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))

# ------------------------------------------------------------------ argmax
def argmax_int(l, tag: str = "argmax") -> Tuple[np.ndarray, GadgetRecord]:
    """Greedy token = argmax with lowest-index tie break, per row of l ([rows, V])."""
    l = as_i64(np.atleast_2d(l))
    tau = np.argmax(l, axis=-1).astype(I64)       # numpy returns the first maximal index
    V = l.shape[-1]
    idx = np.arange(V, dtype=I64)
    delta = (idx[None, :] < tau[:, None]).astype(I64)
    ltau = l[np.arange(l.shape[0]), tau]
    d = ltau[:, None] - l - delta
    rec = GadgetRecord("argmax", tag, committed={"delta": delta, "tau": tau, "ltau": ltau},
                       checks={"d": d}, params={"V": V})
    rec.aux["l"] = l
    return tau, rec

# ---------------------------------------------------------------- RoPE tables
def rope_tables(ctx: int, dh: int, base: float, frac: int) -> Tuple[np.ndarray, np.ndarray]:
    """Public integer tables C[pos, i] = RHAZ(2^frac cos(pos*theta_i)), same for sin,
    theta_i = base^(-2i/dh), i < dh/2 (rotate-half layout)."""
    i = np.arange(dh // 2, dtype=np.float64)
    theta = base ** (-2.0 * i / dh)
    ang = np.arange(ctx, dtype=np.float64)[:, None] * theta[None, :]
    return rhaz_float((1 << frac) * np.cos(ang)), rhaz_float((1 << frac) * np.sin(ang))

def rope_int(q, C, S, M, s: int, lo=-128, hi=127, tag="rq") -> Tuple[np.ndarray, GadgetRecord]:
    """q: [n, heads, dh] int8-range; C,S: [n, dh/2].  Rotate-half:
    out[:h] = RQ(q[:h] C - q[h:] S), out[h:] = RQ(q[:h] S + q[h:] C)."""
    q = as_i64(q)
    h = q.shape[-1] // 2
    a, b = q[..., :h], q[..., h:]
    Cb, Sb = C[:, None, :], S[:, None, :]
    v = np.concatenate([a * Cb - b * Sb, a * Sb + b * Cb], axis=-1)
    return requant(v, M, s, lo, hi, tag=tag)
