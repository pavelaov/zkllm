"""SumFold golden model: product sumcheck over Goldilocks/Ext2 (spec/sumfold_kernel.md).

Proves  H = sum_{b in {0,1}^n} f~(b) * g~(b)  for two multilinear extensions given by their evaluation tables f, g of length N = 2^n.  Variables are bound LSB-first: round i pairs (v[2j], v[2j+1]).

This file defines, bit-exactly, what every accelerator (C, CUDA, RTL) must output.  It also contains the host-side Fiat-Shamir transcript and the verifier.  The verifier here has oracle access to f and g for the final check (it recomputes f~(r), g~(r)); in the full system that final check is an opening of the committed polynomials, provided by the proof backend.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dfield
from typing import List, Sequence, Tuple

from .field import (P, MASK64, Ext, ZERO, eadd, esub, emul, edouble, enc,
                    ehex, embed, escale, finv, fmul, fsub)

DOMAIN = b"EDGE-CERT/sumfold/v1"

# ------------------------------------------------------------------ PRNG
def xorshift64(state: int) -> int:
    """Marsaglia xorshift64 (13, 7, 17). The RTL generator is identical."""
    x = state & MASK64
    x ^= (x << 13) & MASK64
    x ^= x >> 7
    x ^= (x << 17) & MASK64
    return x

def gen_vector(seed: int, n_elems: int) -> List[Ext]:
    """v[i] = embed(x_{i+1} mod p) where x_0 = seed and x_{k+1} = xorshift64(x_k)."""
    assert seed & MASK64 != 0, "xorshift64 seed must be non-zero"
    out: List[Ext] = []
    x = seed & MASK64
    for _ in range(n_elems):
        x = xorshift64(x)
        v = x - P if x >= P else x
        out.append((v, 0))
    return out

# ------------------------------------------------------------ transcript
class Transcript:
    """SHA-256 chained transcript.  state_0 = H(DOMAIN || n || enc(claim)).

    challenge_i = decode(state_{i+1}) where
    state_{i+1} = H(state_i || enc(s0) || enc(s1) || enc(s2)),
    decode(d) = (u64le(d[0:8]) mod p, u64le(d[8:16]) mod p).
    """

    def __init__(self, n: int, claim: Ext):
        self.state = hashlib.sha256(DOMAIN + n.to_bytes(4, "little") + enc(claim)).digest()

    def round(self, s0: Ext, s1: Ext, s2: Ext) -> Ext:
        self.state = hashlib.sha256(self.state + enc(s0) + enc(s1) + enc(s2)).digest()
        c0 = int.from_bytes(self.state[0:8], "little") % P
        c1 = int.from_bytes(self.state[8:16], "little") % P
        return (c0, c1)

# ------------------------------------------------------------ kernel ops
def round_message(f: Sequence[Ext], g: Sequence[Ext]) -> Tuple[Ext, Ext, Ext]:
    s0 = s1 = s2 = ZERO
    for j in range(len(f) // 2):
        f0, f1 = f[2 * j], f[2 * j + 1]
        g0, g1 = g[2 * j], g[2 * j + 1]
        s0 = eadd(s0, emul(f0, g0))
        s1 = eadd(s1, emul(f1, g1))
        s2 = eadd(s2, emul(esub(edouble(f1), f0), esub(edouble(g1), g0)))
    return s0, s1, s2

def fold(v: Sequence[Ext], r: Ext) -> List[Ext]:
    return [eadd(v[2 * j], emul(r, esub(v[2 * j + 1], v[2 * j]))) for j in range(len(v) // 2)]

def inner(f: Sequence[Ext], g: Sequence[Ext]) -> Ext:
    acc = ZERO
    for a, b in zip(f, g):
        acc = eadd(acc, emul(a, b))
    return acc

def interp_quadratic(s0: Ext, s1: Ext, s2: Ext, r: Ext) -> Ext:
    """h(r) for the degree-2 h with h(0)=s0, h(1)=s1, h(2)=s2 (Lagrange)."""
    one = (1, 0)
    two = (2, 0)
    inv2 = finv(2)
    rm1 = esub(r, one)
    rm2 = esub(r, two)
    l0 = escale(emul(rm1, rm2), inv2)            # (r-1)(r-2)/2
    l1 = esub(ZERO, emul(r, rm2))                 # -r(r-2)
    l2 = escale(emul(r, rm1), inv2)               # r(r-1)/2
    return eadd(eadd(emul(s0, l0), emul(s1, l1)), emul(s2, l2))

# ------------------------------------------------------------ prover
@dataclass
class SumFoldProof:
    n: int
    claim: Ext
    msgs: List[Tuple[Ext, Ext, Ext]] = dfield(default_factory=list)
    challenges: List[Ext] = dfield(default_factory=list)
    final_f: Ext = ZERO
    final_g: Ext = ZERO

def prove(f: Sequence[Ext], g: Sequence[Ext]) -> SumFoldProof:
    N = len(f)
    n = N.bit_length() - 1
    assert N == 1 << n and len(g) == N and n >= 1
    claim = inner(f, g)
    tr = Transcript(n, claim)
    pf = SumFoldProof(n=n, claim=claim)
    cf, cg = list(f), list(g)
    for _ in range(n):
        m = round_message(cf, cg)
        r = tr.round(*m)
        pf.msgs.append(m)
        pf.challenges.append(r)
        cf, cg = fold(cf, r), fold(cg, r)
    pf.final_f, pf.final_g = cf[0], cg[0]
    return pf

def verify(pf: SumFoldProof, f_oracle: Sequence[Ext] | None = None,
           g_oracle: Sequence[Ext] | None = None) -> bool:
    tr = Transcript(pf.n, pf.claim)
    claim = pf.claim
    if len(pf.msgs) != pf.n:
        return False
    for i, (s0, s1, s2) in enumerate(pf.msgs):
        if eadd(s0, s1) != claim:
            return False
        r = tr.round(s0, s1, s2)
        if pf.challenges and pf.challenges[i] != r:
            return False
        claim = interp_quadratic(s0, s1, s2, r)
    if emul(pf.final_f, pf.final_g) != claim:
        return False
    if f_oracle is not None and g_oracle is not None:
        cf, cg = list(f_oracle), list(g_oracle)
        for r in (pf.challenges or []):
            cf, cg = fold(cf, r), fold(cg, r)
        if (cf[0], cg[0]) != (pf.final_f, pf.final_g):
            return False
    return True

# ------------------------------------------------ device (lane) semantics
def device_rounds(n: int, lanes: int) -> int:
    """Rounds executed on an L-lane device before the host tail: R = n - log2(L)."""
    lg = lanes.bit_length() - 1
    assert lanes == 1 << lg and n > lg
    return n - lg

def device_expected(f: Sequence[Ext], g: Sequence[Ext], lanes: int) -> dict:
    """What an L-lane device must report in AUTO mode.

    The device runs rounds 0..R-1 on lane-local contiguous blocks, summing the
    per-lane round messages, then applies the final fold with r_{R-1}, leaving
    one (f, g) pair per lane.  Challenges are the true transcript challenges,
    so device outputs equal the first R messages of prove(f, g).
    """
    pf = prove(f, g)
    R = device_rounds(pf.n, lanes)
    cf, cg = list(f), list(g)
    for i in range(R):
        cf, cg = fold(cf, pf.challenges[i]), fold(cg, pf.challenges[i])
    assert len(cf) == lanes
    return {
        "n": pf.n, "lanes": lanes, "device_rounds": R,
        "claim": ehex(pf.claim),
        "challenges": [ehex(r) for r in pf.challenges[:R]],
        "msgs": [[ehex(s) for s in m] for m in pf.msgs[:R]],
        "lane_final_f": [ehex(x) for x in cf],
        "lane_final_g": [ehex(x) for x in cg],
        "full_final_f": ehex(pf.final_f), "full_final_g": ehex(pf.final_g),
        "all_challenges": [ehex(r) for r in pf.challenges],
        "all_msgs": [[ehex(s) for s in m] for m in pf.msgs],
    }

def host_tail(lane_f: Sequence[Ext], lane_g: Sequence[Ext], challenges: Sequence[Ext]):
    """Host finishes the last log2(L) rounds from the per-lane residues."""
    cf, cg = list(lane_f), list(lane_g)
    msgs = []
    for r in challenges:
        msgs.append(round_message(cf, cg))
        cf, cg = fold(cf, r), fold(cg, r)
    return msgs, cf[0], cg[0]

def parse_ehex(h: str) -> Ext:
    v = int(h, 16)
    return (v & MASK64, v >> 64)

def golden(seed_f: int, seed_g: int, n: int, lanes: int) -> dict:
    f = gen_vector(seed_f, 1 << n)
    g = gen_vector(seed_g, 1 << n)
    d = device_expected(f, g, lanes)
    d.update({"seed_f": f"{seed_f:016x}", "seed_g": f"{seed_g:016x}"})
    return d

def _self_test() -> None:
    f = gen_vector(0x1234_5678_9ABC_DEF1, 64)
    g = gen_vector(0x0FED_CBA9_8765_4321, 64)
    pf = prove(f, g)
    assert verify(pf, f, g)
    bad = SumFoldProof(pf.n, eadd(pf.claim, (1, 0)), pf.msgs, [], pf.final_f, pf.final_g)
    assert not verify(bad)
    for L in (1, 2, 4, 8):
        d = device_expected(f, g, L)
        R = d["device_rounds"]
        lf = [parse_ehex(x) for x in d["lane_final_f"]]
        lg = [parse_ehex(x) for x in d["lane_final_g"]]
        msgs, ff, gg = host_tail(lf, lg, pf.challenges[R:])
        assert msgs == pf.msgs[R:] and (ff, gg) == (pf.final_f, pf.final_g)

if __name__ == "__main__":
    _self_test()
    print("sumfold self-test OK")
    print(json.dumps(golden(1, 2, 4, 2), indent=1)[:600])
