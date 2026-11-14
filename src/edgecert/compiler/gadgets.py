"""Gadget cost catalog: witnesses, range checks, lookups and certified bounds.

One function per gadget of spec/numerical_semantics.md.  Every function is pure integer arithmetic on *certified bounds* (never on data), so every number derived here is ANALYTIC.  Names match the witness names produced by the exact reference (``edgecert.reference.intops``) so that the dynamic checker and the count cross-check can compare them one to one.

Counting bases (``basis``): ``tok`` per token, ``score`` per unmasked (head, query, key) triple, ``qrow`` per (head, query) softmax row, ``emit`` per position whose token is emitted (LM head + argmax).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

def ubits(x: int) -> int:
    x = int(x)
    assert x >= 0
    return max(1, x.bit_length())

def isqrt(x: int) -> int:
    return math.isqrt(int(x))

@dataclass
class Check:
    name: str        # '<tag>.<check>' as recorded by the reference trace
    offset: int      # value + offset must lie in [0, 2^bits)
    bits: int
    basis: str
    count: int       # instances per basis unit

@dataclass
class Cost:
    committed: List[Tuple[str, str, int]] = field(default_factory=list)      # (name, basis, count)
    checks: List[Check] = field(default_factory=list)
    naive_extra: List[Tuple[str, str, int, int]] = field(default_factory=list)  # (name, basis, count, bits)
    lookups: List[Tuple[str, str, int]] = field(default_factory=list)          # (table, basis, count)
    terms: Dict[str, int] = field(default_factory=dict)   # max |term| in any constraint (no-wrap check)
    bounds: Dict[str, int] = field(default_factory=dict)  # certified |value| bounds of named tensors
    notes: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)  # gadget parameters (shift, multiplier, clamp)

    def extend(self, other: "Cost") -> "Cost":
        self.committed += other.committed
        self.checks += other.checks
        self.naive_extra += other.naive_extra
        self.lookups += other.lookups
        self.terms.update(other.terms)
        self.bounds.update(other.bounds)
        self.notes += other.notes
        self.meta.update(other.meta)
        return self

# ------------------------------------------------------------------ matmul
def matmul_bound(K: int, B_x: int, B_w: int, B_b: int = 0) -> int:
    """|s| <= K * B_w * B_x + |b| for every partial sum of a K-term dot product."""
    return K * B_x * B_w + B_b

def matmul_cost(basis: str, count: int, K: int, B_x: int, B_w: int, B_b: int = 0, tag: str = "mm") -> Cost:
    B = matmul_bound(K, B_x, B_w, B_b)
    c = Cost()
    c.committed.append((f"{tag}.acc", basis, count))
    c.naive_extra.append((f"{tag}.acc", basis, count, ubits(B) + 1))   # signed value
    c.terms[f"{tag}.acc"] = B
    c.bounds[f"{tag}.acc"] = B
    return c

# ----------------------------------------------------------------- requant
def rq_cost(basis: str, count: int, B_v: int, M_max: int, s: int, lo: int, hi: int,
            add_bound: int = 0, elide: bool = True, tag: str = "rq") -> Cost:
    """y = clamp(RHAZ(v*M/2^s) + add, lo, hi) with |v| <= B_v, 1 <= M <= M_max."""
    B_n = B_v * M_max
    B_y0 = ((B_n + (1 << (s - 1))) >> s) + add_bound
    sat_max = max(B_y0 - hi - 1, B_y0 + lo - 1)
    elided = elide and sat_max < 0
    c = Cost()
    c.committed += [(f"{tag}.sigma", basis, count), (f"{tag}.rem", basis, count), (f"{tag}.y", basis, count)]
    c.checks += [Check(f"{tag}.sgn", 0, ubits(B_v), basis, count),
                 Check(f"{tag}.rem", 0, s, basis, count),
                 Check(f"{tag}.out", -lo, ubits(hi - lo), basis, count)]
    if not elided:
        c.committed += [(f"{tag}.clo", basis, count), (f"{tag}.chi", basis, count)]
        sat_bits = ubits(max(0, sat_max))
        c.checks.append(Check(f"{tag}.sat", 0, sat_bits, basis, count))
        y0_rc = max(hi, -lo) + (1 << sat_bits)
    else:
        c.notes.append(f"{tag}: clamp elided (static |y0| <= {B_y0} inside [{lo},{hi}])")
        y0_rc = max(hi, -lo)
    c.terms.update({f"{tag}.n": B_n + (1 << (s - 1)) + 1,
                    f"{tag}.y0*2^s+rem": (y0_rc + add_bound + 1) << s,
                    f"{tag}.sgn": B_v + 1})
    c.bounds.update({f"{tag}.v": B_v, f"{tag}.y0": B_y0, f"{tag}.y": max(hi, -lo)})
    c.meta.update({f"{tag}.s": s, f"{tag}.M_max": M_max, f"{tag}.lo": lo, f"{tag}.hi": hi,
                   f"{tag}.elided": elided, f"{tag}.add_bound": add_bound, f"{tag}.rounding": "RHAZ"})
    return c

# -------------------------------------------------------------------- norm
def norm_bounds(d: int, kind: str, F: int, G: int, eps_int: int, in_bits: int = 16) -> Dict[str, int]:
    """Certified bounds of the integer LayerNorm/RMSNorm witnesses (input int<in_bits>)."""
    assert eps_int >= 1
    r_max = 1 << (in_bits - 1)
    C = (2 * r_max - 1) if kind == "ln" else r_max           # |c| (LN: r - mu with mu in int16)
    V_max = C * C                                          # v = floor(sum c^2 / d) <= C^2
    X_max = (V_max + eps_int) << (2 * G)
    Z_max = isqrt(X_max)
    z_min = isqrt(eps_int << (2 * G))
    Y_max = (1 << (F + G)) // z_min
    # derived bound |c_i * y| < sqrt(d) 2^F / (1 - 2^-G)   (see spec, Sec. "norm")
    g_derived = isqrt((d << (2 * F + 2 * G)) // (((1 << G) - 1) ** 2)) + 1
    g_interval = C * Y_max
    return {"C": C, "V_max": V_max, "X_max": X_max, "Z_max": Z_max, "Y_max": Y_max,
            "G_bound": min(g_derived, g_interval), "g_derived": g_derived, "g_interval": g_interval}

def norm_cost(rows_per_tok: int, d: int, kind: str, F: int, G: int, eps_int: int,
              in_bits: int = 16, tag: str = "nm", basis: str = "tok") -> Cost:
    nb = norm_bounds(d, kind, F, G, eps_int, in_bits)
    c = Cost()
    B = basis
    rpt = rows_per_tok
    pow2 = (d & (d - 1)) == 0
    if kind == "ln":
        c.committed += [(f"{tag}.mu", B, rpt), (f"{tag}.rho_mu", B, rpt)]
        c.checks += [Check(f"{tag}.mu", 1 << (in_bits - 1), in_bits, B, rpt),
                     Check(f"{tag}.rho_mu", 0, ubits(d - 1), B, rpt)]
        if not pow2:
            c.checks.append(Check(f"{tag}.rho_mu_hi", 0, ubits(d - 1), B, rpt))
        c.naive_extra.append((f"{tag}.S", B, rpt, ubits(d << (in_bits - 1)) + 1))
    c.committed += [(f"{tag}.{k}", B, rpt) for k in ("v", "rho_v", "z", "y", "rho_y")]
    c.committed.append((f"{tag}.g", B, rpt * d))
    zb = ubits(nb["Z_max"])
    c.checks += [Check(f"{tag}.v", 0, ubits(nb["V_max"]), B, rpt),
                 Check(f"{tag}.rho_v", 0, ubits(d - 1), B, rpt),
                 Check(f"{tag}.z", 0, zb, B, rpt),
                 Check(f"{tag}.sa", 0, ubits(2 * nb["Z_max"]), B, rpt),
                 Check(f"{tag}.sb", 0, ubits(2 * nb["Z_max"]), B, rpt),
                 Check(f"{tag}.y", 0, ubits(nb["Y_max"]), B, rpt),
                 Check(f"{tag}.rho_y", 0, ubits(nb["Z_max"] - 1), B, rpt),
                 Check(f"{tag}.rho_y_hi", 0, ubits(nb["Z_max"] - 1), B, rpt)]
    if not pow2:
        c.checks.append(Check(f"{tag}.rho_v_hi", 0, ubits(d - 1), B, rpt))
    c.naive_extra += [(f"{tag}.Q", B, rpt, ubits(d * nb["C"] ** 2)),
                      (f"{tag}.g", B, rpt * d, ubits(nb["G_bound"]) + 1)]
    # constraint terms use range-check-implied bounds (malicious witnesses)
    v_rc = (1 << ubits(nb["V_max"])) - 1
    z_rc = (1 << zb) - 1
    y_rc = (1 << ubits(nb["Y_max"])) - 1
    c_rc = (1 << in_bits) if kind == "ln" else (1 << (in_bits - 1))
    c.terms.update({f"{tag}.Q=d*v+rho": max(d * c_rc * c_rc, d * v_rc + d),
                    f"{tag}.X": (v_rc + eps_int) << (2 * G),
                    f"{tag}.(z+1)^2": (z_rc + 1) ** 2,
                    f"{tag}.y*z+rho": y_rc * z_rc + z_rc,
                    f"{tag}.g=c*y": c_rc * y_rc})
    c.bounds.update({f"{tag}.g": nb["G_bound"], f"{tag}.Q": d * nb["C"] ** 2})
    c.meta.update({f"{tag}.kind": kind, f"{tag}.d": d, f"{tag}.F": F, f"{tag}.G": G, f"{tag}.eps_int": eps_int,
                   f"{tag}.in_bits": in_bits})
    return c

# ------------------------------------------------------------------ softmax
def softmax_cost(t_lo: int, t_hi: int, U_max: int, E_bits: int, Fp: int, keys_max: int,
                 tag: str = "sm") -> Cost:
    u_max = t_hi - t_lo
    E_max = 1 << E_bits
    D_max = keys_max * E_max
    c = Cost()
    c.committed += [(f"{tag}.{k}", "score", 1) for k in ("c", "E", "P", "rho")]
    c.committed += [(f"{tag}.m", "qrow", 1), (f"{tag}.D", "qrow", 1)]
    c.checks += [Check(f"{tag}.u", 0, ubits(u_max), "score", 1),
                 Check(f"{tag}.e", 0, ubits(max(U_max, u_max - U_max - 1)), "score", 1),
                 Check(f"{tag}.p", 0, ubits(1 << Fp), "score", 1),
                 Check(f"{tag}.rho", 0, ubits(D_max - 1), "score", 1),
                 Check(f"{tag}.rho_hi", 0, ubits(D_max - 1), "score", 1)]
    c.lookups.append(("T_exp", "score", 1))
    c.naive_extra.append((f"{tag}.D", "qrow", 1, ubits(D_max)))
    p_rc = (1 << ubits(1 << Fp)) - 1
    c.terms.update({f"{tag}.E*2^Fp=P*D+rho": max(E_max << Fp, p_rc * D_max + D_max),
                    f"{tag}.D": D_max, f"{tag}.u": u_max})
    c.bounds.update({f"{tag}.P": 1 << Fp, f"{tag}.D": D_max})
    c.notes.append(f"{tag}: max certified by a zero-product (GKR product layer, no extra commitment)")
    c.meta.update({f"{tag}.U_max": U_max, f"{tag}.E_bits": E_bits, f"{tag}.Fp": Fp, f"{tag}.keys_max": keys_max,
                   f"{tag}.t_range": (t_lo, t_hi)})
    return c

# ------------------------------------------------------------------ argmax
def argmax_cost(V: int, B_l: int, tag: str = "am") -> Cost:
    c = Cost()
    c.committed += [(f"{tag}.delta", "emit", V), (f"{tag}.tau", "emit", 1), (f"{tag}.ltau", "emit", 1)]
    c.checks.append(Check(f"{tag}.d", 0, ubits(2 * B_l), "emit", V))
    c.terms[f"{tag}.d"] = 2 * B_l + 1
    c.bounds[f"{tag}.l"] = B_l
    return c

# ------------------------------------------------------------------- tables
def act_cost(count: int, table: str = "T_act", tag: str = "act") -> Cost:
    c = Cost()
    c.committed.append((f"{tag}.out", "tok", count))
    c.lookups.append((table, "tok", count))
    return c

def embed_cost(d: int, learned_pos: bool, tag: str = "emb") -> Cost:
    c = Cost()
    c.committed += [(f"{tag}.row", "tok", d), (f"{tag}.mult", "tok", 1)]
    c.lookups.append(("T_emb", "tok", 1))            # one row-tuple lookup (compressed)
    if learned_pos:
        c.committed.append((f"{tag}.pos_row", "tok", d))
        c.lookups.append(("T_pos", "tok", 1))
    return c
