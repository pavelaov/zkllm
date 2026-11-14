"""Goldilocks prime field and its quadratic extension (golden model).

This module is the single source of truth for field semantics shared by the Python reference, the C/CUDA kernels and the ZCU102 RTL (see spec/sumfold_kernel.md). Everything is exact integer arithmetic.

    p   = 2^64 - 2^32 + 1
    Ext = F_p[X] / (X^2 - W),  W = 7 (a quadratic non-residue mod p)

An extension element is a tuple (c0, c1) meaning c0 + c1*X.
"""
from __future__ import annotations

P = 0xFFFFFFFF00000001
W = 7
MASK64 = (1 << 64) - 1

Ext = tuple  # (c0, c1)

ZERO: Ext = (0, 0)
ONE: Ext = (1, 0)

def reduce128(x: int) -> int:
    """Goldilocks reduction of a 128-bit value, written exactly as the RTL does it.

    x = hi*2^64 + lo, hi = hh*2^32 + hl.  Since 2^64 = 2^32 - 1 (mod p) and
    2^96 = -1 (mod p):  x = lo - hh + hl*(2^32 - 1)  (mod p).
    """
    assert 0 <= x < (1 << 128)
    lo = x & MASK64
    hi = x >> 64
    hh = hi >> 32
    hl = hi & 0xFFFFFFFF
    t = lo - hh
    if t < 0:  # borrow: add p (equivalently subtract 2^32 - 1 from the wrapped value)
        t += P
    t += hl * 0xFFFFFFFF
    # t < 2^64 + 2^64, so at most two conditional subtractions are needed
    while t >= P:
        t -= P
    return t

def fadd(a: int, b: int) -> int:
    s = a + b
    return s - P if s >= P else s

def fsub(a: int, b: int) -> int:
    return a - b if a >= b else a - b + P

def fmul(a: int, b: int) -> int:
    return reduce128(a * b)

def fneg(a: int) -> int:
    return 0 if a == 0 else P - a

def finv(a: int) -> int:
    assert a % P != 0
    return pow(a, P - 2, P)

def canon(a: int) -> int:
    return a % P

# ---------------------------------------------------------------- extension

def eadd(a: Ext, b: Ext) -> Ext:
    return (fadd(a[0], b[0]), fadd(a[1], b[1]))

def esub(a: Ext, b: Ext) -> Ext:
    return (fsub(a[0], b[0]), fsub(a[1], b[1]))

def emul(a: Ext, b: Ext) -> Ext:
    """Karatsuba form used by all kernels: 3 base multiplications + mul-by-7."""
    t0 = fmul(a[0], b[0])
    t1 = fmul(a[1], b[1])
    t2 = fmul(fadd(a[0], a[1]), fadd(b[0], b[1]))
    c0 = fadd(t0, fmul(W, t1))
    c1 = fsub(fsub(t2, t0), t1)
    return (c0, c1)

def escale(a: Ext, k: int) -> Ext:
    return (fmul(a[0], k % P), fmul(a[1], k % P))

def edouble(a: Ext) -> Ext:
    return eadd(a, a)

def einv(a: Ext) -> Ext:
    # (c0 + c1 X)^-1 = (c0 - c1 X) / (c0^2 - W c1^2)
    n = fsub(fmul(a[0], a[0]), fmul(W, fmul(a[1], a[1])))
    ni = finv(n)
    return (fmul(a[0], ni), fmul(fneg(a[1]), ni))

def embed(a: int) -> Ext:
    return (a % P, 0)

def enc(a: Ext) -> bytes:
    """Canonical 16-byte little-endian encoding: c0 || c1."""
    return int(a[0]).to_bytes(8, "little") + int(a[1]).to_bytes(8, "little")

def ehex(a: Ext) -> str:
    """128-bit hex word as stored in RTL memories: {c1, c0}."""
    return f"{a[1]:016x}{a[0]:016x}"

def lift_signed(v: int) -> int:
    """Embed a signed integer into F_p (two's-complement style lift)."""
    return v % P

def unlift_signed(a: int, bound: int) -> int:
    """Inverse of lift_signed for |v| <= bound < p/2; raises on violation."""
    assert 2 * bound < P
    if a <= bound:
        return a
    if a >= P - bound:
        return a - P
    raise ValueError("field element outside certified signed range")

def _self_test() -> None:
    import random
    rng = random.Random(1)
    for _ in range(2000):
        a, b = rng.randrange(P), rng.randrange(P)
        assert fmul(a, b) == (a * b) % P
        x = rng.randrange(1 << 128)
        assert reduce128(x) == x % P
        ea = (rng.randrange(P), rng.randrange(P))
        eb = (rng.randrange(P), rng.randrange(P))
        c = emul(ea, eb)
        # schoolbook check
        c0 = (ea[0] * eb[0] + W * ea[1] * eb[1]) % P
        c1 = (ea[0] * eb[1] + ea[1] * eb[0]) % P
        assert c == (c0, c1)
        if ea != ZERO:
            assert emul(ea, einv(ea)) == ONE
    assert pow(W, (P - 1) // 2, P) == P - 1  # W is a non-residue -> Ext is a field

if __name__ == "__main__":
    _self_test()
    print("field self-test OK")
