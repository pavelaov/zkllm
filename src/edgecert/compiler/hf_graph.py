"""TileContract lowering of real-weight integer derivatives (GPT-2 124M, Qwen3-0.6B).

``build_hf_graph`` is the real-model analogue of ``graph.build_t0_graph``: it lowers an ``edgecert.reference.hf_models.HFIntModel`` with its *committed* constants (calibrated requant multipliers, the actual largest integer weight of every linear layer, the actual int32 biases and LayerNorm betas) through the shared ``graph._decoder_ops`` so that op ids, gadget costs and check names match the executed trace one to one.  Every bound is still a static worst case over all inputs (not a calibration statistic), so the contracts are sound for every prompt; ``tile_contract.dynamic_check`` confirms it on each run.

``fp64_exact_plan`` records the exact-FP64 datapath certificate: every matmul's static partial-sum bound K * B_x * B_w must be < 2^53.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List

from .gadgets import norm_bounds, norm_cost, rq_cost, ubits
from .graph import Op, _decoder_ops
from .tile_contract import IMPLEMENTATIONS

FP64_EXACT = 1 << 53

def _mx(a) -> int:
    if a is None:
        return 0
    if hasattr(a, "__len__") or hasattr(a, "shape"):
        import numpy as np
        a = np.asarray(a)
        return int(np.abs(a).max()) if a.size else 0
    return abs(int(a))

def committed_constants(im) -> dict:
    """Per-op committed constants of an HFIntModel (multiplier maxima, weight/bias bounds)."""
    cfg, P = im.cfg, im.P
    M: Dict[str, int] = {}
    Bw: Dict[str, int] = {}
    Bb: Dict[str, int] = {}
    Bbeta: Dict[str, int] = {}
    for l, Ly in enumerate(im.layers):
        p = f"L{l}."
        M[p + "ln1"], M[p + "ln2"] = _mx(Ly["n1"]["M"]), _mx(Ly["n2"]["M"])
        Bbeta[p + "ln1"], Bbeta[p + "ln2"] = _mx(Ly["n1"]["beta"]), _mx(Ly["n2"]["beta"])
        M[p + "qkv"] = _mx(Ly["M_qkv"])
        Bw[p + "qkv"] = im.wq[p + "qkv"]["B_w"]
        Bb[p + "qkv"] = _mx(Ly["b_qkv"])
        M[p + "scores"], M[p + "pv"] = _mx(Ly["M_sc"]), _mx(Ly["M_pv"])
        M[p + "o_proj"] = _mx(Ly["M_o"])
        Bw[p + "o_proj"] = im.wq[p + "o_proj"]["B_w"]
        Bb[p + "o_proj"] = _mx(Ly["b_o"])
        if cfg.family == "qwen3":
            M[p + "qk_norm"] = _mx(Ly["qk_M"])
            M[p + "rope"] = _mx(Ly["M_rope"])
            M[p + "ff_gate"], M[p + "ff_up"] = _mx(Ly["M_gate"]), _mx(Ly["M_up"])
            M[p + "glu"], M[p + "ff_down"] = _mx(Ly["M_glu"]), _mx(Ly["M_dn"])
            for k in ("ff_gate", "ff_up", "ff_down"):
                Bw[p + k] = im.wq[p + k]["B_w"]
                Bb[p + k] = 0
        else:
            M[p + "ff_up"], M[p + "ff_down"] = _mx(Ly["M_up"]), _mx(Ly["M_dn"])
            Bw[p + "ff_up"], Bw[p + "ff_down"] = im.wq[p + "ff_up"]["B_w"], im.wq[p + "ff_down"]["B_w"]
            Bb[p + "ff_up"], Bb[p + "ff_down"] = _mx(Ly["b_1"]), _mx(Ly["b_2"])
    M["lnf"] = _mx(P["nf"]["M"])
    Bbeta["lnf"] = _mx(P["nf"]["beta"])
    return {"M": M, "B_w": Bw, "B_b": Bb, "B_beta": Bbeta, "M_emb": _mx(P["M_E"]),
            "M_pos": _mx(P["M_P"]) if cfg.family == "gpt2" else 0, "M_v": _mx(P["M_v"])}

def build_hf_graph(im, elide: bool = True) -> List[Op]:
    """Lowering of a calibrated real-weight integer derivative with its committed constants."""
    cfg = im.cfg
    pol = copy.deepcopy(im.pol)
    cc = committed_constants(im)
        # the matmul operand width used for limb planning is that of the committed integer weights (int8 for W8A8; int4 for W4A8; int4 x group multiplier for W4A8g)
    pol.wbits = max(ubits(b) + 1 for b in cc["B_w"].values())
    ops = _decoder_ops(name_d=cfg.d, L=cfg.n_layers, H=cfg.n_heads, Hkv=cfg.n_kv_heads, dh=cfg.dh, ff=cfg.ff,
                        V=cfg.vocab, ctx=cfg.ctx, norm_kind="ln" if cfg.family == "gpt2" else "rms",
                        learned_pos=cfg.family == "gpt2", mlp="gelu" if cfg.family == "gpt2" else "swiglu",
                        qk_norm=cfg.family == "qwen3", sandwich=False, windows=[None] * cfg.n_layers, pol=pol,
                        M=cc["M"], B_w=cc["B_w"], B_b=cc["B_b"], B_beta=cc["B_beta"], eps_int=im.eps_int,
                        elide=elide, M_emb=cc["M_emb"], M_pos=cc["M_pos"], M_v=cc["M_v"])
    if cfg.family == "qwen3":
                # _decoder_ops takes one eps_int (that of the residual-stream norms); the per-head QK-norm has its own eps_int (eps / s_qk^2), so its cost is rebuilt here exactly as graph.norm_op does
        for op in ops:
            if op.op_id.endswith("qk_norm"):
                l = op.layer
                e = im.layers[l]["eps_qk"][0]
                rows, dim = cfg.n_heads + cfg.n_kv_heads, cfg.dh
                c = norm_cost(rows, dim, "rms", pol.F, pol.G, e, 8, tag="nm", basis="tok")
                Gb = norm_bounds(dim, "rms", pol.F, pol.G, e, 8)["G_bound"]
                c.extend(rq_cost("tok", rows * dim, Gb * 127, cc["M"][op.op_id], pol.shifts["norm"], -128, 127,
                                 elide=elide))
                op.cost = c
                op.params["eps_int"] = e
    return ops

def fp64_exact_plan(ops: List[Op]) -> dict:
    """Exact-FP64 datapath certificate: per matmul op, K * B_x * B_w (every partial sum) < 2^53."""
    rows = []
    for op in ops:
        mm = op.mm
        if mm is None:
            continue
        B = mm["K"] * mm["B_x"] * mm["B_w"]
        rows.append({"op": op.op_id, "K": mm["K"], "B_x": mm["B_x"], "B_w": mm["B_w"], "partial_sum_bound": B,
                     "bound_log2": round(math.log2(B), 3), "headroom_bits_vs_2^53": round(53 - math.log2(B), 3),
                     "certified_exact": B < FP64_EXACT})
    worst = max(rows, key=lambda r: r["partial_sum_bound"])
    return {"rule": "IEEE-754 binary64 represents every integer of magnitude <= 2^53; every partial sum of "
                    "any summation order is a subset sum of products bounded by K*B_x*B_w, so a GEMM whose "
                    "bound is < 2^53 returns the exact integer result (bias added in int64 afterwards)",
            "implementation_ids": [k for k in IMPLEMENTATIONS if k.endswith("fp64_exact")],
            "all_certified": all(r["certified_exact"] for r in rows), "matmul_ops": len(rows),
            "worst": worst, "per_op": rows}
