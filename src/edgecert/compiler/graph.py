"""Graph description of a W8A8 / W4A8 decoder and its lowering into gadget costs.

``build_graph`` works from a ModelShape plus the quantizer policy (policy-level bounds: multipliers bounded by 2^m_max, eps_int = 1, bias and beta bounded by the policy).  ``build_t0_graph`` works from a calibrated T0 ``IntModel`` and uses the *committed* constants (exact multipliers, exact weight/bias maxima), which is what a publisher would certify at commitment time. Both produce the same ``Op`` list format and op ids match the reference trace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..reference.intref import IntModel, IntPolicy
from .gadgets import (Cost, act_cost, argmax_cost, embed_cost, matmul_bound, matmul_cost, norm_bounds,
                      norm_cost, rq_cost, softmax_cost)
from .model_shapes import ModelShape

TRIG_BOUND = lambda frac: math.ceil((1 << frac) * math.sqrt(2)) + 1   # max |C| + |S| of a RoPE table row

@dataclass
class Op:
    op_id: str
    kind: str
    layer: Optional[int]
    cost: Cost
    heads: int = 0                      # query heads (attention ops)
    window: Optional[int] = None        # sliding window of the layer (attention ops)
    mm: Optional[dict] = None           # matmul geometry for tiles / K-split
    params: Dict[str, object] = field(default_factory=dict)

def _mm(K, N, B_x, B_w, B_b, rows, x_bits, w_bits, keys_dependent=False):
    return {"K": K, "N": N, "B_x": B_x, "B_w": B_w, "B_b": B_b, "rows": rows,
            "x_bits": x_bits, "w_bits": w_bits, "keys_dependent": keys_dependent}

def _decoder_ops(*, name_d: int, L: int, H: int, Hkv: int, dh: int, ff: int, V: int, ctx: int,
                 norm_kind: str, learned_pos: bool, mlp: str, qk_norm: bool, sandwich: bool,
                 windows: List[Optional[int]], pol: IntPolicy, M: Dict[str, int], B_w: Dict[str, int],
                 B_b: Dict[str, int], B_beta: Dict[str, int], eps_int: int, elide: bool,
                 M_emb: int, M_pos: int, M_v: int) -> List[Op]:
    """Shared lowering for T0 and real shapes.  M[op_id] = max multiplier of that op's requant."""
    d = name_d
    sh = pol.shifts
    rlo, rhi = pol.res_lo, pol.res_hi
    alo, ahi = pol.act_lo, pol.act_hi
    ops: List[Op] = []
    Bx = 128

    def norm_op(op_id, layer, basis, rows, dim, in_bits, out_lo, out_hi, cls, add_bound=0, beta=0):
        kind = "rms" if op_id.endswith("qk_norm") else norm_kind
        c = norm_cost(rows, dim, kind, pol.F, pol.G, eps_int, in_bits, tag="nm", basis=basis)
        Gb = norm_bounds(dim, kind, pol.F, pol.G, eps_int, in_bits)["G_bound"]
        c.extend(rq_cost(basis, rows * dim, Gb * 127 + beta, M[op_id], sh[cls], out_lo, out_hi,
                         add_bound=add_bound, elide=elide))
        return Op(op_id, "norm", layer, c, params={"d": dim, "kind": kind, "rows_per_tok": rows})

    # embedding gather + requant into the residual stream
    c = embed_cost(d, learned_pos)
    c.extend(rq_cost("tok", d, 127 * M_emb + (127 * M_pos if learned_pos else 0), 1, sh["emb"], rlo, rhi,
                     elide=elide))
    ops.append(Op("embed", "embed", None, c))
    for l in range(L):
        p = f"L{l}."
        W = windows[l]
        keys_max = min(ctx, W) if W else ctx
        ops.append(norm_op(p + "ln1", l, "tok", 1, d, pol.res_bits, -128, 127, "norm", beta=B_beta[p + "ln1"]))
        Nqkv = (H + 2 * Hkv) * dh
        c = matmul_cost("tok", Nqkv, d, Bx, B_w[p + "qkv"], B_b[p + "qkv"])
        c.extend(rq_cost("tok", Nqkv, matmul_bound(d, Bx, B_w[p + "qkv"], B_b[p + "qkv"]), M[p + "qkv"],
                         sh["lin"], -128, 127, elide=elide))
        ops.append(Op(p + "qkv", "linear", l, c, mm=_mm(d, Nqkv, Bx, B_w[p + "qkv"], B_b[p + "qkv"], "tok", 8,
                                                          pol.wbits)))
        if qk_norm:
            ops.append(norm_op(p + "qk_norm", l, "tok", H + Hkv, dh, 8, -128, 127, "norm"))
        if not learned_pos:
            tb = TRIG_BOUND(pol.rope_frac)
            for nm, hh in (("rope_q", H), ("rope_k", Hkv)):
                c = rq_cost("tok", hh * dh, 128 * tb, M[p + "rope"], sh["rope"], -128, 127, elide=elide)
                ops.append(Op(p + nm, "rope", l, c))
        c = matmul_cost("score", 1, dh, Bx, Bx)
        c.extend(rq_cost("score", 1, matmul_bound(dh, Bx, Bx), M[p + "scores"], sh["score"], rlo, rhi,
                         elide=elide))
        ops.append(Op(p + "scores", "scores", l, c, heads=H, window=W,
                      mm=_mm(dh, None, Bx, Bx, 0, "score", 8, 8)))
        B_t = c.bounds["rq.y0"] if any("clamp elided" in n for n in c.notes) else None
        t_lo, t_hi = (-B_t, B_t) if B_t is not None else (rlo, rhi)
        ops.append(Op(p + "softmax", "softmax", l,
                      softmax_cost(t_lo, t_hi, pol.U_max, pol.E_bits, pol.Fp, keys_max), heads=H, window=W))
        Bp = 1 << pol.Fp
        c = matmul_cost("tok", H * dh, keys_max, Bp, Bx)
        c.extend(rq_cost("tok", H * dh, matmul_bound(keys_max, Bp, Bx), M[p + "pv"], sh["pv"], -128, 127,
                         elide=elide))
        ops.append(Op(p + "pv", "pv", l, c, heads=H, window=W,
                      mm=_mm(keys_max, dh, Bp, Bx, 0, "tok", pol.Fp + 1, 8, keys_dependent=True)))
        Bo = matmul_bound(H * dh, Bx, B_w[p + "o_proj"], B_b[p + "o_proj"])
        c = matmul_cost("tok", d, H * dh, Bx, B_w[p + "o_proj"], B_b[p + "o_proj"])
        if sandwich:
            c.extend(rq_cost("tok", d, Bo, M[p + "o_proj"], sh["res"], rlo, rhi, elide=elide))
            ops.append(Op(p + "o_proj", "linear", l, c, mm=_mm(H * dh, d, Bx, B_w[p + "o_proj"],
                                                                 B_b[p + "o_proj"], "tok", 8, pol.wbits)))
            ops.append(norm_op(p + "post_attn_norm", l, "tok", 1, d, pol.res_bits, rlo, rhi, "norm",
                               add_bound=1 << (pol.res_bits - 1)))
        else:
            c.extend(rq_cost("tok", d, Bo, M[p + "o_proj"], sh["res"], rlo, rhi,
                             add_bound=1 << (pol.res_bits - 1), elide=elide))
            ops.append(Op(p + "o_proj", "linear", l, c, mm=_mm(H * dh, d, Bx, B_w[p + "o_proj"],
                                                                 B_b[p + "o_proj"], "tok", 8, pol.wbits)))
        ops.append(norm_op(p + "ln2", l, "tok", 1, d, pol.res_bits, -128, 127, "norm", beta=B_beta[p + "ln2"]))
        if mlp == "gelu":
            Bu = matmul_bound(d, Bx, B_w[p + "ff_up"], B_b[p + "ff_up"])
            c = matmul_cost("tok", ff, d, Bx, B_w[p + "ff_up"], B_b[p + "ff_up"])
            c.extend(rq_cost("tok", ff, Bu, M[p + "ff_up"], sh["act"], alo, ahi, elide=elide))
            ops.append(Op(p + "ff_up", "linear", l, c, mm=_mm(d, ff, Bx, B_w[p + "ff_up"], B_b[p + "ff_up"],
                                                               "tok", 8, pol.wbits)))
            ops.append(Op(p + "act", "act", l, act_cost(ff)))
        else:
            Bg = matmul_bound(d, Bx, B_w[p + "ff_gate"], 0)
            c = matmul_cost("tok", ff, d, Bx, B_w[p + "ff_gate"])
            c.extend(rq_cost("tok", ff, Bg, M[p + "ff_gate"], sh["act"], alo, ahi, elide=elide))
            ops.append(Op(p + "ff_gate", "linear", l, c, mm=_mm(d, ff, Bx, B_w[p + "ff_gate"], 0, "tok", 8,
                                                                 pol.wbits)))
            ops.append(Op(p + "act", "act", l, act_cost(ff)))
            Bu = matmul_bound(d, Bx, B_w[p + "ff_up"], 0)
            c = matmul_cost("tok", ff, d, Bx, B_w[p + "ff_up"])
            c.extend(rq_cost("tok", ff, Bu, M[p + "ff_up"], sh["lin"], -128, 127, elide=elide))
            ops.append(Op(p + "ff_up", "linear", l, c, mm=_mm(d, ff, Bx, B_w[p + "ff_up"], 0, "tok", 8,
                                                               pol.wbits)))
            ops.append(Op(p + "glu", "glu", l, rq_cost("tok", ff, 128 * 128, M[p + "glu"], sh["glu"], -128, 127,
                                                        elide=elide)))
        Bd = matmul_bound(ff, Bx, B_w[p + "ff_down"], B_b[p + "ff_down"])
        c = matmul_cost("tok", d, ff, Bx, B_w[p + "ff_down"], B_b[p + "ff_down"])
        mmd = _mm(ff, d, Bx, B_w[p + "ff_down"], B_b[p + "ff_down"], "tok", 8, pol.wbits)
        if sandwich:
            c.extend(rq_cost("tok", d, Bd, M[p + "ff_down"], sh["res"], rlo, rhi, elide=elide))
            ops.append(Op(p + "ff_down", "linear", l, c, mm=mmd))
            ops.append(norm_op(p + "post_ffn_norm", l, "tok", 1, d, pol.res_bits, rlo, rhi, "norm",
                               add_bound=1 << (pol.res_bits - 1)))
        else:
            c.extend(rq_cost("tok", d, Bd, M[p + "ff_down"], sh["res"], rlo, rhi,
                             add_bound=1 << (pol.res_bits - 1), elide=elide))
            ops.append(Op(p + "ff_down", "linear", l, c, mm=mmd))
    ops.append(norm_op("lnf", None, "emit", 1, d, pol.res_bits, -128, 127, "norm", beta=B_beta["lnf"]))
    ops.append(Op("lm_head", "lm_head", None, matmul_cost("emit", V, d, Bx, 127),
                  mm=_mm(d, V, Bx, 127, 0, "emit", 8, 8)))
    ops.append(Op("argmax", "argmax", None, argmax_cost(V, matmul_bound(d, Bx, 127) * M_v)))
    return ops

def build_graph(shape: ModelShape, pol: Optional[IntPolicy] = None, elide: bool = True) -> List[Op]:
    """Policy-level lowering of a real architecture (no weights needed)."""
    pol = pol or IntPolicy()
    Mx = {k: (1 << v) - 1 for k, v in pol.m_max.items()}
    L = shape.n_layers
    Bb_pol = (1 << (pol.bias_bits - 1)) - 1 if shape.bias else 0
    M, Bw, Bb, Bbeta = {}, {}, {}, {}
    cls = {"qkv": "lin", "rope": "rope", "scores": "score", "pv": "pv", "o_proj": "res", "ff_up": "act",
           "ff_gate": "act", "glu": "glu", "ff_down": "res", "ln1": "norm", "ln2": "norm", "qk_norm": "norm",
           "post_attn_norm": "norm", "post_ffn_norm": "norm"}
    for l in range(L):
        for k, cl in cls.items():
            key = f"L{l}.{k}"
            M[key] = Mx[cl] if not (k == "ff_up" and shape.mlp != "gelu") else Mx["lin"]
            Bw[key] = pol.wmax
            Bb[key] = Bb_pol
            Bbeta[key] = (127 << pol.F) if shape.norm == "ln" else 0
    M["lnf"] = Mx["norm"]
    Bbeta["lnf"] = (127 << pol.F) if shape.norm == "ln" else 0
    windows = [shape.window(l) for l in range(L)]
    return _decoder_ops(name_d=shape.d_model, L=L, H=shape.n_heads, Hkv=shape.n_kv_heads, dh=shape.head_dim,
                        ff=shape.d_ff, V=shape.vocab, ctx=shape.ctx, norm_kind=shape.norm,
                        learned_pos=shape.pos == "learned", mlp=shape.mlp, qk_norm=shape.qk_norm,
                        sandwich=shape.sandwich_norm, windows=windows, pol=pol, M=M, B_w=Bw, B_b=Bb,
                        B_beta=Bbeta, eps_int=1, elide=elide, M_emb=Mx["emb"],
                        M_pos=Mx["emb"] if shape.pos == "learned" else 0, M_v=1 << pol.logit_m)

def build_t0_graph(im: IntModel, elide: bool = True) -> List[Op]:
    """Lowering of a calibrated T0 model using its committed constants."""
    cfg, pol = im.cfg, im.pol
    L = cfg.n_layers
    mx = lambda a: int(abs(int(a))) if not hasattr(a, "__len__") else int(max(abs(int(x)) for x in a))
    M, Bw, Bb, Bbeta = {}, {}, {}, {}
    for l, Ly in enumerate(im.layers):
        p = f"L{l}."
        M[p + "ln1"], M[p + "ln2"] = int(Ly["n1"]["M"]), int(Ly["n2"]["M"])
        Bbeta[p + "ln1"], Bbeta[p + "ln2"] = mx(Ly["n1"]["beta"]), mx(Ly["n2"]["beta"])
        M[p + "qkv"] = int(Ly["M_qkv"].max())
        Bw[p + "qkv"] = int(abs(Ly["Wqkv"]).max())
        Bb[p + "qkv"] = mx(Ly["b_qkv"]) if Ly["b_qkv"] is not None else 0
        M[p + "scores"], M[p + "pv"] = int(Ly["M_sc"]), int(Ly["M_pv"])
        M[p + "o_proj"] = int(Ly["M_o"].max())
        Bw[p + "o_proj"] = int(abs(Ly["Wo"]).max())
        Bb[p + "o_proj"] = mx(Ly["b_o"]) if Ly["b_o"] is not None else 0
        if cfg.family == "llama":
            M[p + "rope"] = int(Ly["M_rope"])
        if cfg.family == "gpt":
            M[p + "ff_up"], M[p + "ff_down"] = int(Ly["M_up"].max()), int(Ly["M_dn"].max())
            Bw[p + "ff_up"], Bw[p + "ff_down"] = int(abs(Ly["W1"]).max()), int(abs(Ly["W2"]).max())
            Bb[p + "ff_up"], Bb[p + "ff_down"] = mx(Ly["b_1"]), mx(Ly["b_2"])
        else:
            M[p + "ff_gate"], M[p + "ff_up"] = int(Ly["M_gate"].max()), int(Ly["M_up"].max())
            M[p + "glu"], M[p + "ff_down"] = int(Ly["M_glu"]), int(Ly["M_dn"].max())
            Bw[p + "ff_gate"], Bw[p + "ff_up"] = int(abs(Ly["Wg"]).max()), int(abs(Ly["Wu"]).max())
            Bw[p + "ff_down"] = int(abs(Ly["Wd"]).max())
            Bb[p + "ff_gate"] = Bb[p + "ff_up"] = Bb[p + "ff_down"] = 0
    M["lnf"] = int(im.P["nf"]["M"])
    Bbeta["lnf"] = mx(im.P["nf"]["beta"])
    M_emb = int(im.P["M_E"].max())
    M_pos = int(im.P["M_P"]) if cfg.family == "gpt" else 0
    return _decoder_ops(name_d=cfg.d, L=L, H=cfg.n_heads, Hkv=cfg.n_heads, dh=cfg.dh, ff=cfg.ff, V=cfg.vocab,
                        ctx=cfg.ctx, norm_kind="ln" if cfg.family == "gpt" else "rms",
                        learned_pos=cfg.family == "gpt", mlp="gelu" if cfg.family == "gpt" else "swiglu",
                        qk_norm=False, sandwich=False, windows=[None] * L, pol=pol, M=M, B_w=Bw, B_b=Bb,
                        B_beta=Bbeta, eps_int=im.eps_int, elide=elide, M_emb=M_emb, M_pos=M_pos,
                        M_v=int(im.P["M_v"].max()))

# ------------------------------------------------------------ workloads
def keys_sum(n: int, W: Optional[int]) -> int:
    """sum_{i<n} min(i+1, W): unmasked (query, key) pairs of a causal prefill."""
    if W is None or W >= n:
        return n * (n + 1) // 2
    return W * (W + 1) // 2 + (n - W) * W

def multiplicity(op: Op, basis: str, workload: dict) -> int:
    """How many basis units of ``op`` a workload contains.

    workload = {'kind': 'prefill', 'n': n}  (emits the token after position n-1)
             | {'kind': 'decode', 'pos': t}  (one new token at position t, keys t+1; KV of
               earlier positions was committed by earlier proofs)
    """
    W = op.window
    if workload["kind"] == "prefill":
        n = workload["n"]
        return {"tok": n, "emit": 1, "qrow": op.heads * n, "score": op.heads * keys_sum(n, W)}[basis]
    t = workload["pos"]
    keys = min(t + 1, W) if W else t + 1
    return {"tok": 1, "emit": 1, "qrow": op.heads, "score": op.heads * keys}[basis]
