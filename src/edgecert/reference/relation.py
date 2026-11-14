"""Field-level relation checker for the integer gadgets (spec/numerical_semantics.md).

This is what a proof system enforces: polynomial identities over F_p plus range checks (``0 <= canonical(x) < 2^bits``, i.e. lookups into a range table) and table membership.  It deliberately receives *claimed* witness values (possibly tampered) and re-derives nothing from the honest computation, so the negative tests can show which constraint rejects which attack.

All arithmetic is Python-int modulo p (default Goldilocks, from the read-only golden model ``edgecert.field``).  Intended for tests and small instances.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from ..field import P as GOLDILOCKS_P

BABYBEAR_P = (1 << 31) - (1 << 27) + 1          # 2013265921, a 31-bit field used as a counter-example
BN254_R = 21888242871839275222246405745257275088548364400416034343698204186575808495617

class RelationError(Exception):
    """A constraint of the relation does not hold for the claimed witness."""

def _ints(a) -> list:
    return [int(x) for x in np.asarray(a, dtype=object).ravel()]

def _rc(val: int, bits: int, p: int, what: str) -> None:
    x = val % p
    if not (0 <= x < (1 << bits)):
        raise RelationError(f"range check failed: {what} (canonical {x} not < 2^{bits})")

def _eq(a: int, b: int, p: int, what: str) -> None:
    if (a - b) % p != 0:
        raise RelationError(f"identity failed: {what}")

def _bool(b: int, p: int, what: str) -> None:
    _eq(b * (b - 1), 0, p, f"boolean {what}")

def lift(v: int, p: int) -> int:
    return v % p

def decode_signed(x: int, bound: int, p: int) -> int:
    """Unique integer in [-bound, bound] congruent to x (requires 2*bound < p)."""
    if 2 * bound >= p:
        raise RelationError("field too small for the certified bound: decoding is ambiguous")
    x %= p
    if x <= bound:
        return x
    if x >= p - bound:
        return x - p
    raise RelationError("field element outside the certified signed range")

def accept_integer_claim(claim: int, field_value: int, bound: int, p: int) -> bool:
    """Verifier-side acceptance of a public integer claim for a proven field value."""
    if abs(claim) > bound:
        return False
    return (claim - field_value) % p == 0 and 2 * bound < p

# ------------------------------------------------------------------ gadgets
def check_matmul(x, w, b, acc, p: int = GOLDILOCKS_P) -> None:
    """acc == x @ w^T + b  (mod p).  A real backend proves this with a sumcheck."""
    X = np.asarray(x, dtype=object)
    Wt = np.asarray(w, dtype=object).T
    ref = X.dot(Wt)
    if b is not None:
        ref = ref + np.asarray(b, dtype=object)
    for r, a in zip(_ints(ref), _ints(acc)):
        _eq(a, r, p, "matmul accumulator")

def check_requant(v, M, s: int, lo: Optional[int], hi: Optional[int], w: Dict[str, object],
                  bits: Dict[str, int], add=None, p: int = GOLDILOCKS_P) -> None:
    """RHAZ requantization with optional clamp (witness names as in intops.requant)."""
    vv = _ints(v)
    n_el = len(vv)
    MM = _ints(np.broadcast_to(np.asarray(M, dtype=object), np.shape(v)))
    sig, rem, y = _ints(w["sigma"]), _ints(w["rem"]), _ints(w["y"])
    adds = _ints(np.broadcast_to(np.asarray(add, dtype=object), np.shape(v))) if add is not None else [0] * n_el
    clamp = lo is not None
    if clamp:
        clo, chi = _ints(w["clo"]), _ints(w["chi"])
    inv2s = pow(1 << s, p - 2, p)
    half = 1 << (s - 1)
    for i in range(n_el):
        _bool(sig[i], p, "sigma")
        n = vv[i] * MM[i]
        y0 = ((n + half - sig[i] - rem[i]) * inv2s + adds[i]) % p
        _rc((1 - 2 * sig[i]) * vv[i] - sig[i], bits["sgn"], p, "sign slack")
        _rc(rem[i], s, p, "remainder")
        if clamp:
            _bool(clo[i], p, "c_lo")
            _bool(chi[i], p, "c_hi")
            _eq(clo[i] * chi[i], 0, p, "c_lo*c_hi")
            _eq(y[i], chi[i] * hi + clo[i] * lo + (1 - clo[i] - chi[i]) * y0, p, "clamp select")
            _rc(y[i] - lo, bits["out"], p, "output range")
            _rc(chi[i] * (y0 - hi - 1) + clo[i] * (lo - 1 - y0), bits["sat"], p, "saturation slack")
        else:
            _eq(y[i], y0, p, "requant output")
            _rc(y[i] + bits["yr_offset"], bits["yr"], p, "output range")

def table_set(table: Sequence[int], key_lo: int, p: int = GOLDILOCKS_P) -> set:
    return {((key_lo + i) % p, int(t) % p) for i, t in enumerate(table)}

def check_table(keys, vals, table: Sequence[int], key_lo: int, p: int = GOLDILOCKS_P,
                tset: Optional[set] = None) -> None:
    """(key, val) pairs must belong to the committed table {(key_lo + i, table[i])}."""
    T = tset if tset is not None else table_set(table, key_lo, p)
    for k, v in zip(_ints(keys), _ints(vals)):
        if (k % p, v % p) not in T:
            raise RelationError(f"lookup ({k}, {v}) not in committed table")

def check_softmax_row(t_row, w: Dict[str, object], table: Sequence[int], Fp: int, bits: Dict[str, int],
                      p: int = GOLDILOCKS_P, tset: Optional[set] = None) -> None:
    """One softmax row over its allowed entries (causal mask is public structure)."""
    tset = tset if tset is not None else table_set(table, 0, p)
    t = _ints(t_row)
    m, D = int(w["m"]), int(w["D"])
    c, E, Pr, rho = _ints(w["c"]), _ints(w["E"]), _ints(w["P"]), _ints(w["rho"])
    U = len(table) - 1
    prod = 1
    for j in range(len(t)):
        u = m - t[j]
        prod = prod * u % p
        _rc(u, bits["u"], p, "u = m - t >= 0")
        _bool(c[j], p, "clamp flag")
        _rc(c[j] * (u - U - 1) + (1 - c[j]) * (U - u), bits["e"], p, "exp-domain clamp slack")
        ub = (c[j] * U + (1 - c[j]) * u) % p
        check_table([ub], [E[j]], table, 0, p, tset=tset)
        _eq(E[j] << Fp, Pr[j] * D + rho[j], p, "division E*2^Fp = P*D + rho")
        _rc(Pr[j], bits["p"], p, "probability range")
        _rc(rho[j], bits["rho"], p, "remainder >= 0")
        _rc(D - 1 - rho[j], bits["rho_hi"], p, "remainder < D")
    _eq(prod, 0, p, "max is attained (zero product)")
    _eq(D, sum(E), p, "denominator sum")

def check_norm_row(r_row, kind: str, F: int, G: int, eps_int: int, w: Dict[str, object],
                   bits: Dict[str, int], p: int = GOLDILOCKS_P) -> None:
    """Integer LayerNorm/RMSNorm core for one row (witness names as intops.norm_int)."""
    r = _ints(r_row)
    d = len(r)
    if kind == "ln":
        mu, rho_mu = int(w["mu"]), int(w["rho_mu"])
        _eq(sum(r), d * mu + rho_mu, p, "mean division")
        _rc(mu + (1 << 15), bits["mu"], p, "mean range")
        _rc(rho_mu, bits["rho_mu"], p, "mean remainder")
        if d & (d - 1):
            _rc(d - 1 - rho_mu, bits["rho_mu"], p, "mean remainder < d")
        c = [x - mu for x in r]
    else:
        c = r
    v, rho_v, z, y, rho_y = (int(w[k]) for k in ("v", "rho_v", "z", "y", "rho_y"))
    _eq(sum(x * x for x in c), d * v + rho_v, p, "variance division")
    _rc(v, bits["v"], p, "variance range")
    _rc(rho_v, bits["rho_v"], p, "variance remainder")
    if d & (d - 1):
        _rc(d - 1 - rho_v, bits["rho_v"], p, "variance remainder < d")
    X = (v + eps_int) << (2 * G)
    _rc(z, bits["z"], p, "sqrt range")
    _rc(X - z * z, bits["sa"], p, "z^2 <= X")
    _rc((z + 1) * (z + 1) - 1 - X, bits["sb"], p, "X < (z+1)^2")
    _rc(y, bits["y"], p, "reciprocal range")
    _eq(1 << (F + G), y * z + rho_y, p, "reciprocal division")
    _rc(rho_y, bits["rho_y"], p, "reciprocal remainder")
    _rc(z - 1 - rho_y, bits["rho_y"], p, "reciprocal remainder < z")
    g = _ints(w["g"])
    for i in range(d):
        _eq(g[i], c[i] * y, p, "normalized product")

def check_argmax(l_row, w: Dict[str, object], bits_d: int, p: int = GOLDILOCKS_P) -> None:
    """tau = argmax with lowest-index tie break (witness names as intops.argmax_int)."""
    l = _ints(l_row)
    V = len(l)
    delta, tau, ltau = _ints(w["delta"]), int(w["tau"]), int(w["ltau"])
    prev = 1
    for v in range(V):
        _bool(delta[v], p, "delta")
        _bool(prev - delta[v], p, "monotone delta")
        prev = delta[v]
    _eq(delta[V - 1], 0, p, "tau < V")
    _eq(sum(delta), tau, p, "tau = sum(delta)")
    onehot = [(1 if v == 0 else delta[v - 1]) - delta[v] for v in range(V)]
    _eq(ltau, sum(e * x for e, x in zip(onehot, l)), p, "l_tau selection")
    for v in range(V):
        _rc(ltau - l[v] - delta[v], bits_d, p, "argmax slack")

# ------------------------------------------------------------- attack demo
def forge_requant_wrap(v: int, M: int, s: int, lo: int, hi: int, p: int, max_k: int = 8) -> dict:
    """Wrap-around forgery against the RHAZ requant relation in a field that is too small.

    Finds (y', rem') with y'*2^s + rem' = y0*2^s + rem + k*p for some small k, y' != y
    and y' inside the output range.  Such a witness satisfies every constraint of
    ``check_requant`` modulo p whenever p <= 2 * (hi+1) * 2^s, which is exactly what the
    static TileContract check forbids.  Returns honest and forged witnesses.
    """
    n = v * M
    sig = int(n < 0)
    t = n + (1 << (s - 1)) - sig
    y0 = t >> s
    rem = t - (y0 << s)
    y = min(max(y0, lo), hi)
    honest = {"sigma": [sig], "rem": [rem], "clo": [int(y0 < lo)], "chi": [int(y0 > hi)], "y": [y]}
    for k in [j * sgn for j in range(1, max_k + 1) for sgn in (1, -1)]:
        target = t + k * p
        yf = target >> s
        remf = target - (yf << s)
        if lo <= yf <= hi and yf != y:
            forged = {"sigma": [sig], "rem": [remf], "clo": [0], "chi": [0], "y": [yf]}
            return {"honest_y": y, "forged_y": yf, "k": k, "honest_witness": honest, "forged_witness": forged}
    raise ValueError("no in-range wrap-around witness for this field (field large enough)")
