"""TileContract: range-certified integer lowering (EDGE-CERT contribution C2).

For every lowered op (tile family) the compiler emits

    TileContract = (shape, signed_ranges, limb_layout, accumulator_width,
                    padding_rule, reduction_rule, output_scale,
                    field_embedding, proof_gadget_id, implementation_id)

plus the list of range checks the proof must contain.  Three checkers:

* ``static_check``  : symbolic.  Every accumulator and partial sum satisfies B_s = K*B_w*B_x + |b| with p > 2*B_s, every constraint term of every gadget stays below p/2 (no wrap-around equality), the implementation accumulator (int32 IMMA / int32 SIMD / DSP48E2 48-bit) holds every partial sum, otherwise K is split into certified sub-tiles; operand limbs fit the datapath. * ``dynamic_check`` : runs on a reference trace and ABORTS (ContractViolation) if any observed value leaves its contract range or a witness is missing. * ``optimize_limbs``: chooses the range-check limb width b in {8,10,12,14,16} (table 2^b <= 2^16) minimising lookups + table size, and compares with naive decompositions (full 64-bit field element; generic 32-bit).

Everything here is exact integer arithmetic on bounds: results are ANALYTIC.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..field import P as GOLDILOCKS_P
from .gadgets import Check, ubits
from .graph import Op, multiplicity

BABYBEAR_P = (1 << 31) - (1 << 27) + 1
BN254_R = 21888242871839275222246405745257275088548364400416034343698204186575808495617
FIELDS = {"goldilocks": GOLDILOCKS_P, "babybear": BABYBEAR_P, "bn254_r": BN254_R}
RC_WIDTHS = (8, 10, 12, 14, 16)
TILE_M = TILE_N = 64

IMPLEMENTATIONS = {
    "gpu_imma_s8": {"acc_bits": 32, "limb_bits": 8, "k_align": 32,
                    "desc": "NVIDIA integer MMA (mma.sync .s8/.u8 operands, .s32 accumulator); wider "
                            "operands are split into 8-bit limbs (low limbs unsigned, top limb signed)"},
    "cpu_int32": {"acc_bits": 32, "limb_bits": None, "k_align": 1,
                  "desc": "generic int32 accumulation with direct operands (e.g. SIMD madd into s32)"},
    "fpga_dsp48e2": {"acc_bits": 48, "limb_bits": None, "a_bits": 27, "b_bits": 18, "k_align": 1,
                     "desc": "AMD DSP48E2 27x18 signed multiplier, 48-bit accumulator"},
    "cpu_int64": {"acc_bits": 64, "limb_bits": None, "k_align": 1, "desc": "int64 accumulation"},
        # exact FP64 datapaths (added for the real-weight reference, bench/runners/realmodel_eval.py): binary64 holds every integer of magnitude <= 2^53, so acc_bits = 54 gives acc_max = 2^53 - 1
    "cpu_fp64_exact": {"acc_bits": 54, "limb_bits": None, "k_align": 1,
                       "desc": "IEEE-754 binary64 GEMM (numpy/OpenBLAS DGEMM) on integer operands; exact because "
                               "every partial sum of any order is an integer bounded by K*B_x*B_w <= 2^53 - 1"},
    "gpu_fp64_exact": {"acc_bits": 54, "limb_bits": None, "k_align": 1,
                       "desc": "IEEE-754 binary64 GEMM on the GPU (CuPy/cuBLAS DGEMM) on integer operands; exact "
                               "under the same K*B_x*B_w <= 2^53 - 1 certificate (split-K order irrelevant)"},
}

class ContractError(ValueError):
    """Static contract check failed (the lowering is rejected before execution)."""

class ContractViolation(RuntimeError):
    """Dynamic check failed on a run: the session must ABORT (no certificate, no release)."""

def log2_half(p: int) -> float:
    return math.log2(p / 2)

# ------------------------------------------------------------ accumulation
def plan_accumulation(K: int, B_x: int, B_w: int, B_b: int, x_bits: int, w_bits: int, impl: str) -> dict:
    """Operand limb layout + K-split so that every partial sum fits the accumulator.

    Returns K_pad, limb split, number of limb passes, K_sub, n_sub, per-pass bound.
    Bias is added inside the accumulator only when no split is needed; otherwise it
    is added once in the wide (int64 / field) combine stage.
    """
    prof = IMPLEMENTATIONS[impl]
    A = prof["acc_bits"]
    amax = (1 << (A - 1)) - 1
    align = prof["k_align"]
    K_pad = -(-K // align) * align
    lb = prof.get("limb_bits")
    if lb:
        xl = -(-x_bits // lb)
        wl = -(-w_bits // lb)
        Bxl = B_x if xl == 1 else (1 << lb) - 1
        Bwl = B_w if wl == 1 else (1 << lb) - 1
        limbs = {"x": xl, "w": wl, "limb_bits": lb}
    elif "a_bits" in prof:
        # DSP48E2: put the wider operand on the 27-bit port
        big, small = max(x_bits, w_bits), min(x_bits, w_bits)
        if big > prof["a_bits"] or small > prof["b_bits"]:
            raise ContractError(f"{impl}: operands {x_bits}x{w_bits} bits exceed the 27x18 multiplier")
        xl = wl = 1
        Bxl, Bwl = B_x, B_w
        limbs = {"x": 1, "w": 1, "ports": "27x18"}
    else:
        xl = wl = 1
        Bxl, Bwl = B_x, B_w
        limbs = {"x": 1, "w": 1}
    per = Bxl * Bwl
    full = K * per + B_b           # padded operands are zero: only the K real terms contribute
    if full <= amax:
        K_sub, n_sub, bias_in_combine = K_pad, 1, False
    else:
        # bias moves to the wide combine stage; split K only if the products alone overflow
        K_sub_max = amax // per
        K_sub = min((K_sub_max // align) * align, K_pad)
        if K_sub <= 0:
            raise ContractError(f"{impl}: a single product ({per}) does not fit the accumulator")
        n_sub = -(-K_pad // K_sub)
        bias_in_combine = True
    return {"impl": impl, "acc_bits": A, "K": K, "K_pad": K_pad, "limbs": limbs, "passes": xl * wl,
            "K_sub": K_sub, "n_sub": n_sub, "needs_ksplit": n_sub > 1, "bias_in_combine": bias_in_combine,
            "subtile_bound": min(K_sub, K) * per + (0 if bias_in_combine else B_b), "acc_max": amax,
            "combined_bound": K * B_x * B_w + B_b}

# ---------------------------------------------------------------- contract
@dataclass
class TileContract:
    op_id: str
    shape: dict
    signed_ranges: dict
    limb_layout: dict
    accumulator_width: dict
    padding_rule: str
    reduction_rule: str
    output_scale: dict
    field_embedding: dict
    proof_gadget_id: str
    implementation_id: str
    checks: List[Check] = field(default_factory=list)
    terms: Dict[str, int] = field(default_factory=dict)
    kind: str = ""

    def signature(self) -> tuple:
        d = self.to_dict()
        d.pop("op_id")
        return tuple(sorted((k, repr(v)) for k, v in d.items()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [asdict(c) for c in self.checks]
        return d

GADGET_IDS = {"embed": "emb.rowlookup+rq.rhaz", "norm": "nm.isqrt_div+rq.rhaz", "linear": "mm.sumcheck+rq.rhaz",
              "rope": "rq.rhaz(public trig)", "scores": "mm.sumcheck+rq.rhaz", "softmax": "sm.exp_lookup+div",
              "pv": "mm.sumcheck+rq.rhaz", "act": "act.table_lookup", "glu": "rq.rhaz(product)",
              "lm_head": "mm.sumcheck", "argmax": "am.lowest_index"}

def lower(ops: Sequence[Op], impl: str = "gpu_imma_s8", field_name: str = "goldilocks", rc_bits: int = 16
          ) -> Dict[str, TileContract]:
    p = FIELDS[field_name]
    out: Dict[str, TileContract] = {}
    for op in ops:
        c = op.cost
        rng = {k: [-v, v] for k, v in c.bounds.items()}
        if "rq.lo" in c.meta:
            rng["rq.y"] = [c.meta["rq.lo"], c.meta["rq.hi"]]
        shape: dict = {"per_token_elements": sum(n for (_, b, n) in c.committed if b == "tok")}
        acc_w: dict = {"impl_acc_bits": None, "certified_acc_bits": None}
        red = "elementwise; no reduction"
        pad = "none (elementwise)"
        limb = {"range_check_limb_bits": rc_bits,
                "range_checks": [{"name": k.name, "bits": k.bits, "limbs": -(-k.bits // rc_bits)} for k in c.checks]}
        impl_id = "field"
        if op.mm is not None:
            mm = op.mm
            plan = plan_accumulation(mm["K"], mm["B_x"], mm["B_w"], mm["B_b"], mm["x_bits"], mm["w_bits"], impl)
            B = plan["combined_bound"]
            x_signed = mm["x_bits"] == 8
            rng.update({"x": [-mm["B_x"], mm["B_x"] - 1] if x_signed else [0, mm["B_x"]],
                        "w": [-mm["B_w"], mm["B_w"]], "b": [-mm["B_b"], mm["B_b"]], "mm.acc": [-B, B]})
            shape = {"tile_M": TILE_M, "tile_N": TILE_N, "K": mm["K"], "K_pad": plan["K_pad"],
                     "K_sub": plan["K_sub"], "n_sub": plan["n_sub"], "N": mm["N"],
                     "K_is_key_count": mm["keys_dependent"]}
            acc_w = {"impl_acc_bits": plan["acc_bits"], "certified_acc_bits": ubits(B) + 1,
                     "subtile_bound": plan["subtile_bound"], "combine": "int64/field" if plan["n_sub"] > 1 or
                     plan["passes"] > 1 or plan["bias_in_combine"] else "in-accumulator"}
            limb["operand_limbs"] = plan["limbs"]
            limb["passes"] = plan["passes"]
            pad = (f"zero-pad K={mm['K']} to K_pad={plan['K_pad']} (multiple of {IMPLEMENTATIONS[impl]['k_align']}) "
                   f"and the MLE domain to 2^{max(0, (plan['K_pad'] - 1).bit_length())}; padded operands are 0, "
                   "so padded products are 0 and the bound K*B_x*B_w+|b| is unchanged")
            red = ("exact integer sum, order-free (no partial sum can overflow); "
                   + (f"K split into {plan['n_sub']} sub-tiles of K_sub={plan['K_sub']}, each certified "
                      f"<= {plan['acc_max']}, bias added once in the wide combine" if plan["n_sub"] > 1 else
                      "single accumulator pass, bias added in the accumulator")
                   + (f"; {plan['passes']} limb passes recombined with shifts" if plan["passes"] > 1 else ""))
            impl_id = impl
        outs = {k.split(".", 1)[1]: v for k, v in c.meta.items() if k.startswith("rq.")}
        max_term = max(c.terms.values()) if c.terms else 0
        tb = ubits(max_term) + 1
        fe = {"field": field_name, "p_bits": p.bit_length(), "lift": "signed v -> v mod p; decode to the unique "
              "integer in [-B, B] (requires 2B < p)", "max_term": max_term, "max_term_bits_signed": tb,
              "headroom_bits": round(log2_half(p) - math.log2(max_term + 1), 3) if max_term else None}
        out[op.op_id] = TileContract(op.op_id, shape, rng, limb, acc_w, pad, red, outs, fe, GADGET_IDS[op.kind],
                                     impl_id, list(c.checks), dict(c.terms), op.kind)
    return out

# ------------------------------------------------------------ static check
def static_check(tc: TileContract, p: int = GOLDILOCKS_P, m_max_bits: int = 31) -> List[str]:
    """Return the list of violations (empty list = contract accepted)."""
    v: List[str] = []
    for name, B in tc.terms.items():
        if 2 * B >= p:
            v.append(f"{tc.op_id}: term {name} bound 2^{math.log2(B):.1f} >= p/2: wrap-around equality possible")
    if tc.implementation_id in IMPLEMENTATIONS:
        sh = tc.shape
        lo, hi = tc.signed_ranges["x"]
        Bx = max(-lo, hi)
        Bw = tc.signed_ranges["w"][1]
        Bb = tc.signed_ranges["b"][1]
        B = sh["K"] * Bx * Bw + Bb
        if tc.signed_ranges["mm.acc"][1] != B:
            v.append(f"{tc.op_id}: recorded accumulator bound differs from K*B_x*B_w+|b| = {B}")
        if 2 * B >= p:
            v.append(f"{tc.op_id}: accumulator bound {B} violates p > 2*B_s")
        align = IMPLEMENTATIONS[tc.implementation_id]["k_align"]
        if sh["K_pad"] < sh["K"] or sh["K_pad"] % align:
            v.append(f"{tc.op_id}: padding rule violated (K_pad={sh['K_pad']}, K={sh['K']}, align={align})")
        acc = tc.accumulator_width
        amax = (1 << (acc["impl_acc_bits"] - 1)) - 1
        if acc["subtile_bound"] > amax:
            v.append(f"{tc.op_id}: partial sum bound {acc['subtile_bound']} exceeds int{acc['impl_acc_bits']}")
        lb = tc.limb_layout.get("operand_limbs", {}).get("limb_bits")
        if lb and sh["n_sub"] * sh["K_sub"] < sh["K_pad"]:
            v.append(f"{tc.op_id}: K-split does not cover K_pad")
    osc = tc.output_scale
    if osc:
        if osc.get("M_max", 1) >= (1 << m_max_bits) or osc.get("M_max", 1) < 1:
            v.append(f"{tc.op_id}: requant multiplier out of range")
        if osc.get("s", 1) < 1:
            v.append(f"{tc.op_id}: requant shift must be >= 1")
    for k in tc.checks:
        if k.bits > 64 or k.bits < 1:
            v.append(f"{tc.op_id}: range check {k.name} has {k.bits} bits")
        if (1 << k.bits) >= p:
            v.append(f"{tc.op_id}: range check {k.name} ({k.bits} bits) does not fit the field")
    return v

def assert_static(contracts: Dict[str, TileContract], p: int = GOLDILOCKS_P) -> None:
    errs = [e for tc in contracts.values() for e in static_check(tc, p)]
    if errs:
        raise ContractError("; ".join(errs[:8]) + (f" (+{len(errs) - 8} more)" if len(errs) > 8 else ""))

# ----------------------------------------------------------- dynamic check
def dynamic_check(trace, contracts: Dict[str, TileContract]) -> Dict[str, dict]:
    """Check every recorded witness of a reference trace against its contract.

    Raises ContractViolation (ABORT) on the first violation; returns per-op stats
    (observed max bits vs certified bits) otherwise.
    """
    stats: Dict[str, dict] = {}
    for op in trace:
        tc = contracts.get(op.op_id)
        if tc is None:
            raise ContractViolation(f"{op.op_id}: no TileContract (unsupported operator): ABORT")
        chk = {f"{g.tag}.{k}": np.asarray(a) for g in op.gadgets for k, a in g.checks.items()}
        com = {f"{g.tag}.{k}": np.asarray(a) for g in op.gadgets for k, a in g.committed.items()}
        st = stats.setdefault(op.op_id, {})
        for k in tc.checks:
            if k.name not in chk:
                raise ContractViolation(f"{op.op_id}: witness for range check {k.name} missing: ABORT")
            a = chk[k.name]
            if a.size == 0:
                continue
            lo, hi = int(a.min()) + k.offset, int(a.max()) + k.offset
            if lo < 0 or hi >= (1 << k.bits):
                raise ContractViolation(f"{op.op_id}: {k.name} observed [{lo - k.offset}, {hi - k.offset}] "
                                        f"leaves [{-k.offset}, {(1 << k.bits) - 1 - k.offset}] ({k.bits} bits): ABORT")
            prev = st.get(k.name, {"observed_bits": 0})
            st[k.name] = {"certified_bits": k.bits,
                          "observed_bits": max(prev["observed_bits"], ubits(max(hi, 0)))}
        for name, (lo, hi) in tc.signed_ranges.items():
            if name in com and com[name].size:
                a = com[name]
                if int(a.min()) < lo or int(a.max()) > hi:
                    raise ContractViolation(f"{op.op_id}: {name} observed [{int(a.min())}, {int(a.max())}] "
                                            f"leaves certified [{lo}, {hi}]: ABORT")
                st["range:" + name] = {"certified_bits": ubits(max(-lo, hi)),
                                       "observed_bits": ubits(int(np.abs(a).max()))}
        if tc.output_scale.get("elided") and "rq.sat" in chk:
            if np.any(chk["rq.sat"] != 0) or np.any(com.get("rq.clo", 0)) or np.any(com.get("rq.chi", 0)):
                raise ContractViolation(f"{op.op_id}: clamp was statically elided but saturated: ABORT")
    return stats

def check_weights(im, contracts: Dict[str, TileContract]) -> None:
    """Commitment-time check that committed integer weights lie in the contract ranges."""
    names = {"qkv": "Wqkv", "o_proj": "Wo", "ff_up": ("W1", "Wu"), "ff_down": ("W2", "Wd"), "ff_gate": "Wg"}
    for l, Ly in enumerate(im.layers):
        for op, keys in names.items():
            tc = contracts.get(f"L{l}.{op}")
            if tc is None:
                continue
            for key in (keys if isinstance(keys, tuple) else (keys,)):
                if key in Ly:
                    lo, hi = tc.signed_ranges["w"]
                    if int(Ly[key].min()) < lo or int(Ly[key].max()) > hi:
                        raise ContractViolation(f"L{l}.{op}: committed weights leave [{lo},{hi}]")

# --------------------------------------------------------------- accounting
def check_histogram(ops: Sequence[Op], workload: dict, include_naive_extra: bool = False) -> Dict[int, int]:
    h: Dict[int, int] = defaultdict(int)
    for op in ops:
        for k in op.cost.checks:
            h[k.bits] += k.count * multiplicity(op, k.basis, workload)
        if include_naive_extra:
            for (_, b, n, bits) in op.cost.naive_extra:
                h[bits] += n * multiplicity(op, b, workload)
    return dict(h)

def lookups_for(hist: Dict[int, int], b: int) -> int:
    return sum(cnt * -(-bits // b) for bits, cnt in hist.items())

def limb_elements(hist: Dict[int, int], b: int) -> int:
    return sum(cnt * (-(-bits // b) if bits > b else 0) for bits, cnt in hist.items())

def naive_lookups(hist_all: Dict[int, int], mode: str) -> int:
    """Naive decompositions with 16-bit limbs: 'field64' = every checked value as a full
    64-bit field element (4 limbs); 'int32' = generic 32-bit (2 limbs; 4 if the value is wider)."""
    if mode == "field64":
        return sum(4 * c for c in hist_all.values())
    return sum((2 if bits <= 32 else 4) * c for bits, c in hist_all.items())

def naive_limb_elements(hist_all: Dict[int, int], mode: str) -> int:
    return naive_lookups(hist_all, mode)

def optimize_limbs(hist: Dict[int, int], widths: Iterable[int] = RC_WIDTHS) -> dict:
    """Pick b minimising lookups + 2^b (LogUp cost ~ #lookups + table size)."""
    rows = []
    for b in widths:
        lk = lookups_for(hist, b)
        rows.append({"b": b, "lookups": lk, "table_size": 1 << b, "cost": lk + (1 << b),
                     "limb_elements": limb_elements(hist, b)})
    best = min(rows, key=lambda r: (r["cost"], -r["b"]))
    return {"best_b": best["b"], "per_width": rows}

def table_sizes(ops: Sequence[Op], vocab: int, ctx: int, U_max: int, act_in_bits: int, learned_pos: bool) -> dict:
    n_act = sum(1 for op in ops if op.kind == "act")
    t = {"T_exp": U_max + 1, "T_act": n_act * (1 << act_in_bits), "T_emb": vocab}
    if learned_pos:
        t["T_pos"] = ctx
    return t

def account(ops: Sequence[Op], workload: dict, tables: Dict[str, int], rc_bits: Optional[int] = None) -> dict:
    """Witness / lookup accounting of one proof for ``workload``.

    Returns totals and a per-op-kind breakdown.  TC = TileContract lowering;
    naive_field64 / naive_int32 = same relation without certified bounds.
    """
    hist = check_histogram(ops, workload)
    hist_all = check_histogram(ops, workload, include_naive_extra=True)
    opt = optimize_limbs(hist)
    b = rc_bits or opt["best_b"]
    committed = 0
    by_kind: Dict[str, dict] = defaultdict(lambda: defaultdict(int))
    table_lk: Dict[str, int] = defaultdict(int)
    for op in ops:
        cm = sum(n * multiplicity(op, bs, workload) for (_, bs, n) in op.cost.committed)
        committed += cm
        h = {}
        for k in op.cost.checks:
            h[k.bits] = h.get(k.bits, 0) + k.count * multiplicity(op, k.basis, workload)
        ha = dict(h)
        for (_, bs, n, bits) in op.cost.naive_extra:
            ha[bits] = ha.get(bits, 0) + n * multiplicity(op, bs, workload)
        bk = by_kind[op.kind]
        bk["committed"] += cm
        bk["range_checked_values"] += sum(h.values())
        bk["rc_lookups_tc"] += lookups_for(h, b)
        bk["rc_lookups_naive_field64"] += naive_lookups(ha, "field64")
        bk["rc_lookups_naive_int32"] += naive_lookups(ha, "int32")
        for (t, bs, n) in op.cost.lookups:
            m = n * multiplicity(op, bs, workload)
            table_lk[t] += m
            bk["table_lookups"] += m
    mult_elems = sum(tables.values()) + (1 << b)
    tc_lookups = lookups_for(hist, b)
    witness_tc = committed + limb_elements(hist, b) + mult_elems
    witness_n64 = committed + naive_limb_elements(hist_all, "field64") + sum(tables.values()) + (1 << 16)
    witness_n32 = committed + naive_limb_elements(hist_all, "int32") + sum(tables.values()) + (1 << 16)
    return {
        "workload": workload, "rc_limb_bits": b, "limb_optimizer": opt,
        "range_checked_values": sum(hist.values()),
        "range_checked_values_naive": sum(hist_all.values()),
        "rc_lookups": {"tc": tc_lookups, "tc_b16": lookups_for(hist, 16),
                       "naive_field64": naive_lookups(hist_all, "field64"),
                       "naive_int32": naive_lookups(hist_all, "int32")},
        "table_lookups": dict(table_lk),
        "table_multiplicity_elements": mult_elems,
        "committed_elements": committed,
        "witness_elements": {"tc": witness_tc, "naive_field64": witness_n64, "naive_int32": witness_n32},
        "witness_bytes": {"goldilocks_8B": 8 * witness_tc, "field254_32B": 32 * witness_tc,
                          "naive_field64_goldilocks_8B": 8 * witness_n64},
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "check_histogram_bits": {str(k): v for k, v in sorted(hist.items())},
    }

# ------------------------------------------------------------------- tiles
def tile_census(ops: Sequence[Op], workload: dict, impl: str) -> dict:
    """Matmul tile instances (64x64 output tiles, full K) and how many need a K-split."""
    tiles = split = subtiles = 0
    per_kind: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    T = TILE_M
    for op in ops:
        mm = op.mm
        if mm is None:
            continue
        inst: List[Tuple[int, int]] = []   # (count, K_eff)
        if op.kind in ("linear", "lm_head"):
            rows = multiplicity(op, mm["rows"], workload)
            inst.append((-(-rows // T) * -(-mm["N"] // T), mm["K"]))
        elif op.kind == "scores":
            W = op.window
            if workload["kind"] == "prefill":
                n = workload["n"]
                nt = -(-n // T)
                cnt = 0
                for qt in range(nt):
                    q0, q1 = qt * T, min(n, (qt + 1) * T) - 1
                    k_lo = 0 if W is None else max(0, q0 - W + 1)
                    cnt += (q1 // T) - (k_lo // T) + 1
                inst.append((op.heads * cnt, mm["K"]))
            else:
                t = workload["pos"]
                keys = min(t + 1, W) if W else t + 1
                inst.append((op.heads * -(-keys // T), mm["K"]))
        elif op.kind == "pv":
            W = op.window
            ncol = -(-mm["N"] // T)
            if workload["kind"] == "prefill":
                n = workload["n"]
                for qt in range(-(-n // T)):
                    last = min(n, (qt + 1) * T)
                    keff = min(last, W) if W else last
                    inst.append((op.heads * ncol, keff))
            else:
                t = workload["pos"]
                inst.append((op.heads * ncol, min(t + 1, W) if W else t + 1))
        for cnt, K in inst:
            plan = plan_accumulation(K, mm["B_x"], mm["B_w"], mm["B_b"], mm["x_bits"], mm["w_bits"], impl)
            tiles += cnt
            subtiles += cnt * plan["n_sub"] * plan["passes"]
            per_kind[op.kind][0] += cnt
            if plan["needs_ksplit"]:
                split += cnt
                per_kind[op.kind][1] += cnt
    return {"impl": impl, "tile_instances": tiles, "tiles_needing_ksplit": split,
            "fraction_needing_ksplit": split / tiles if tiles else 0.0, "subtile_passes": subtiles,
            "per_kind": {k: {"tiles": v[0], "ksplit": v[1]} for k, v in per_kind.items()}}

def dedup(contracts: Dict[str, TileContract]) -> List[dict]:
    """Distinct contract signatures (ops sharing shape, ranges and rules share a contract)."""
    groups: Dict[tuple, List[str]] = {}
    for op_id, tc in contracts.items():
        groups.setdefault(tc.signature(), []).append(op_id)
    return [{"ops": v, "count": len(v)} for v in groups.values()]
