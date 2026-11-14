"""TileContract evaluation (EDGE-CERT C2): writes results/raw/contracts/contracts_<model>.json.

Real shapes (GPT-2 124M, Qwen3-0.6B, Gemma-3-1B, Llama-3-8B): policy-level lowering, all numbers (bounds and counts derived exactly from shapes and the quantizer policy; no weights, no execution).

T0 (tiny decoder, two families): calibrated synthetic model; contract numbers are from committed constants; dynamic range checks, relation replay, abort tests and quality metrics are NORMAL (executed here).

Usage (from the repository root):
    python bench/runners/contracts_eval.py            # everything
    python bench/runners/contracts_eval.py --quick    # fewer prompts (smoke run)
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edgecert.compiler import graph as G  # noqa: E402
from edgecert.compiler import tile_contract as TC  # noqa: E402
from edgecert.compiler.gadgets import ubits  # noqa: E402
from edgecert.compiler.model_shapes import MODELS  # noqa: E402
from edgecert.compiler.replay import replay_trace  # noqa: E402
from edgecert.field import P as GOLDILOCKS_P  # noqa: E402
from edgecert.reference import intref as R  # noqa: E402
from edgecert.reference import relation as REL  # noqa: E402

OUT = ROOT / "results" / "raw" / "contracts"
PREFILLS = (32, 128, 512)
DECODE_POS = (32, 128, 512)
LOG2_HALF_GL = math.log2(GOLDILOCKS_P / 2)
LOG2_HALF_BN = math.log2(TC.BN254_R / 2)
SOURCES = ["src/edgecert/reference/intops.py", "src/edgecert/reference/intref.py",
           "src/edgecert/reference/relation.py", "src/edgecert/compiler/gadgets.py",
           "src/edgecert/compiler/graph.py", "src/edgecert/compiler/tile_contract.py",
           "src/edgecert/compiler/model_shapes.py", "src/edgecert/compiler/replay.py",
           "bench/runners/contracts_eval.py"]

def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

def host_info() -> dict:
    return {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "python": sys.version.split()[0], "numpy": np.__version__,
            "note": "single-threaded numpy/Python; other Vivado jobs may run concurrently (timings not claimed)"}

def code_digest() -> dict:
    return {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:16] for s in SOURCES}

def base_record(name: str, evidence, status: str, argv) -> dict:
    return {"status": status, "evidence_class": evidence, "record_type": "tilecontract_eval", "model": name,
            "command": "python " + " ".join(argv), "cwd": str(ROOT), "timestamp_utc": now_utc(),
            "host": host_info(), "tool": {"script": "bench/runners/contracts_eval.py", "code_sha256_16": code_digest()}}

def policy_dict(pol: R.IntPolicy) -> dict:
    d = copy.deepcopy(pol.__dict__)
    d["weight_range"] = [-pol.wmax, pol.wmax]
    return d

# ------------------------------------------------------------------ helpers
def max_acc(contracts) -> dict:
    best = None
    for tc in contracts.values():
        if "mm.acc" in tc.signed_ranges:
            B = tc.signed_ranges["mm.acc"][1]
            if best is None or B > best[1]:
                best = (tc.op_id, B)
    return {"op": best[0], "bound": best[1], "bits_signed": ubits(best[1]) + 1,
            "headroom_bits_goldilocks": round(LOG2_HALF_GL - math.log2(best[1]), 2),
            "headroom_bits_bn254": round(LOG2_HALF_BN - math.log2(best[1]), 2)}

def max_term(contracts) -> dict:
    best = None
    for tc in contracts.values():
        for k, B in tc.terms.items():
            if best is None or B > best[2]:
                best = (tc.op_id, k, B)
    return {"op": best[0], "term": best[1], "bound": best[2], "bits_signed": ubits(best[2]) + 1,
            "headroom_bits_goldilocks": round(LOG2_HALF_GL - math.log2(best[2]), 2),
            "headroom_bits_bn254": round(LOG2_HALF_BN - math.log2(best[2]), 2)}

def per_token(acc: dict, tokens: int) -> dict:
    tabs = acc["table_multiplicity_elements"]
    wtc = acc["witness_elements"]["tc"] - tabs
    wn64 = acc["witness_elements"]["naive_field64"] - (tabs - (1 << acc["rc_limb_bits"]) + (1 << 16))
    lk = acc["rc_lookups"]
    b = acc["rc_limb_bits"]
    return {
        "tokens": tokens,
        "rc_lookups_per_token": {
            "naive_field64": lk["naive_field64"] / tokens, "naive_int32": lk["naive_int32"] / tokens,
            "tilecontract_b16": lk["tc_b16"] / tokens, "tilecontract_opt": lk["tc"] / tokens,
            "tilecontract_opt_b": b,
            "tilecontract_opt_incl_range_table": (lk["tc"] + (1 << b)) / tokens},
        "reduction_vs_naive_field64": lk["naive_field64"] / lk["tc"],
        "reduction_vs_naive_int32": lk["naive_int32"] / lk["tc"],
        "range_checked_values_per_token": {"tilecontract": acc["range_checked_values"] / tokens,
                                           "naive": acc["range_checked_values_naive"] / tokens},
        "table_lookups_per_token": {k: v / tokens for k, v in acc["table_lookups"].items()},
        "witness_elements_per_token_excl_tables": {"tilecontract": wtc / tokens, "naive_field64": wn64 / tokens},
        "witness_bytes_per_token_excl_tables": {"goldilocks_8B": 8 * wtc / tokens, "field254_32B": 32 * wtc / tokens,
                                                "naive_field64_goldilocks_8B": 8 * wn64 / tokens},
        "table_multiplicity_elements_per_proof": tabs,
        "rc_lookups_per_token_by_kind": {k: {"tilecontract": v["rc_lookups_tc"] / tokens,
                                             "naive_field64": v["rc_lookups_naive_field64"] / tokens,
                                             "naive_int32": v["rc_lookups_naive_int32"] / tokens}
                                         for k, v in acc["by_kind"].items()},
        "limb_optimizer": acc["limb_optimizer"],
        "check_histogram_bits_total": acc["check_histogram_bits"],
    }

def evaluate_ops(ops, contracts, vocab, ctx, pol, learned_pos) -> dict:
    tabs = TC.table_sizes(ops, vocab, ctx, pol.U_max, pol.act_in_bits, learned_pos)
    out = {"workloads": {}, "tiles": {}}
    wls = [(f"prefill_{n}", {"kind": "prefill", "n": n}, n) for n in PREFILLS if n <= ctx] + \
          [(f"decode_pos_{t}", {"kind": "decode", "pos": t}, 1) for t in DECODE_POS if t < ctx]
    for name, wl, toks in wls:
        acc = TC.account(ops, wl, tabs)
        out["workloads"][name] = per_token(acc, toks)
        out["tiles"][name] = {impl: TC.tile_census(ops, wl, impl) for impl in ("cpu_int32", "gpu_imma_s8",
                                                                                 "fpga_dsp48e2")}
    out["table_sizes"] = tabs
    return out

def contracts_block(ops, pol, vocab, ctx, learned_pos, include_all: bool) -> dict:
    cons = TC.lower(ops, impl="gpu_imma_s8", field_name="goldilocks")
    viol_gl = [e for tc in cons.values() for e in TC.static_check(tc, GOLDILOCKS_P)]
    viol_bb = [e for tc in cons.values() for e in TC.static_check(tc, TC.BABYBEAR_P)]
    viol_bn = [e for tc in cons.values() for e in TC.static_check(tc, TC.BN254_R)]
    dd = TC.dedup(cons)
    blk = {
        "num_ops": len(ops), "num_contracts": len(cons), "num_distinct_contracts": len(dd),
        "static_check": {"goldilocks": {"pass": not viol_gl, "violations": viol_gl[:20]},
                         "bn254_r": {"pass": not viol_bn, "violations": viol_bn[:5]},
                         "babybear_31bit": {"pass": not viol_bb, "violations_count": len(viol_bb),
                                            "violations_sample": viol_bb[:6]}},
        "max_certified_accumulator": max_acc(cons),
        "max_certified_constraint_term": max_term(cons),
        "clamps_elided": sum(1 for tc in cons.values() if tc.output_scale.get("elided")),
    }
    blk.update(evaluate_ops(ops, cons, vocab, ctx, pol, learned_pos))
    reps = [cons[g["ops"][0]].to_dict() | {"shared_by": g["ops"]} for g in dd]
    blk["contracts"] = [c.to_dict() for c in cons.values()] if include_all else reps
    return blk, cons

# --------------------------------------------------------------- real shapes
def run_real(name: str, argv) -> dict:
    shape = MODELS[name]
    rec = base_record(name, "SUCCESS", "COMPUTED", argv)
    rec["model_shape"] = shape.to_dict()
    rec["assumptions"] = [
        "Policy-level lowering: requant multipliers bounded by 2^m_max per class, eps_int = 1 (worst case), "
        "int32 bias |b| <= 2^23-1 when the model has biases, LayerNorm beta_q <= 127*2^F.",
        "Activations int8 in [-128,127]; residual stream int16; weights int8 [-127,127] (W8A8) or int4 [-7,7] "
        "(W4A8, linear layers only; embeddings and LM head stay int8).",
        "Decode workloads count only the new token's witness (KV of earlier positions was committed by earlier "
        "proofs); prefill workloads emit one token (LM head, final norm and argmax only at the last position).",
        "Tiles: 64x64 output tiles, full K; attention tiles restricted to the causal (and sliding-window) region.",
        "Naive baselines use the same relation but no certified bounds: every checked value and every "
        "accumulator is range-checked as a full 64-bit field element (4 x 16-bit limbs) or as a generic "
        "int32 (2 limbs; 4 if wider).",
    ]
    rec["variants"] = {}
    for vname, wb in (("W8A8", 8), ("W4A8", 4)):
        pol = R.IntPolicy(wbits=wb)
        ops = G.build_graph(shape, pol)
        blk, _ = contracts_block(ops, pol, shape.vocab, shape.ctx, shape.pos == "learned", include_all=False)
        blk["policy"] = policy_dict(pol)
        rec["variants"][vname] = blk
    return rec

# ------------------------------------------------------------------------ T0
def dynamic_campaign(im: R.IntModel, n_prompts: int, seed: int) -> dict:
    cons = TC.lower(G.build_t0_graph(im, elide=True))
    rng = np.random.default_rng(seed)
    agg: dict = {}
    checks = aborts = 0
    t0 = time.perf_counter()
    for _ in range(n_prompts):
        n = int(rng.integers(8, im.cfg.ctx + 1))
        toks = rng.integers(0, im.cfg.vocab, size=n)
        _, tr, _ = im.forward(toks, record=True)
        try:
            st = TC.dynamic_check(tr, cons)
        except TC.ContractViolation:
            aborts += 1
            continue
        for op_id, d in st.items():
            for k, v in d.items():
                key = op_id.split(".", 1)[1] if op_id.startswith("L") else op_id
                a = agg.setdefault(f"{key}:{k}", {"certified_bits": v["certified_bits"], "observed_max_bits": 0})
                a["observed_max_bits"] = max(a["observed_max_bits"], v["observed_bits"])
                a["certified_bits"] = max(a["certified_bits"], v["certified_bits"])
                checks += 1
    slack = [v["certified_bits"] - v["observed_max_bits"] for v in agg.values()]
    return {"evidence_class": "NORMAL", "prompts": n_prompts, "lengths": "uniform in [8, ctx]", "seed": seed,
            "aborts": aborts, "violations": aborts, "tensor_check_evaluations": checks,
            "per_check": agg, "certified_minus_observed_bits": {"min": min(slack), "max": max(slack),
                                                                 "mean": float(np.mean(slack))},
            "wall_s": round(time.perf_counter() - t0, 3)}

def abort_tests(im: R.IntModel) -> list:
    """Each case must ABORT (ContractViolation / ValueError / RelationError)."""
    cons = TC.lower(G.build_t0_graph(im, elide=True))
    toks = R.random_prompts(im.cfg, 1, 16, 77)[0]
    res = []

    def case(name, fn):
        try:
            fn()
            res.append({"case": name, "aborted": False})
        except (TC.ContractViolation, ValueError, REL.RelationError) as e:
            res.append({"case": name, "aborted": True, "exception": type(e).__name__, "reason": str(e)[:160]})

    def overflow_activation():
        _, tr, _ = im.forward(toks, record=True)
        g = next(op for op in tr if op.op_id == "L0.qkv").gadgets[1]
        g.checks["out"] = g.checks["out"].copy()
        g.checks["out"].reshape(-1)[5] = 200            # int8 activation 200 claimed
        TC.dynamic_check(tr, cons)

    def tight_calibrated_contract():
        c2 = copy.deepcopy(cons)
        k = next(k for k in c2["L0.qkv"].checks if k.name == "rq.sgn")
        k.bits = 10                                     # a calibration-only bound that real data exceeds
        _, tr, _ = im.forward(toks, record=True)
        TC.dynamic_check(tr, c2)

    def unsupported_op():
        c2 = dict(cons)
        c2.pop("L1.softmax")
        _, tr, _ = im.forward(toks, record=True)
        TC.dynamic_check(tr, c2)

    def token_out_of_vocab():
        im.forward(list(toks[:-1]) + [im.cfg.vocab], record=True)

    def context_overflow():
        im.forward(np.zeros(im.cfg.ctx + 1, dtype=np.int64), record=True)

    def tampered_act_table():
        _, tr, _ = im.forward(toks, record=True)
        cn = TC.lower(G.build_t0_graph(im, elide=False))
        g = next(op for op in tr if op.op_id == "L0.act").gadgets[0]
        g.committed["out"] = g.committed["out"].copy()
        g.committed["out"].reshape(-1)[0] += 1
        replay_trace(tr, cn, im)

    for nm, fn in (("int8 activation overflow injected", overflow_activation),
                   ("calibration-only contract tighter than data", tight_calibrated_contract),
                   ("op without TileContract (unsupported operator)", unsupported_op),
                   ("token id outside vocabulary", token_out_of_vocab),
                   ("sequence longer than certified context", context_overflow),
                   ("activation lookup inconsistent with committed table", tampered_act_table)):
        case(nm, fn)
    return res

def manifest_summary(im: R.IntModel, n: int) -> dict:
    toks = R.random_prompts(im.cfg, 1, n, 55)[0]
    _, tr, _ = im.forward(toks, record=True)
    man = R.witness_manifest(tr)
    per_layer: dict = {}
    for m in man:
        key = "global" if m["layer"] is None else f"L{m['layer']}"
        a = per_layer.setdefault(key, {"tensors": 0, "elements": 0, "bytes_native": 0, "bytes_goldilocks": 0,
                                       "bytes_254bit": 0})
        a["tensors"] += 1
        for k in ("elements", "bytes_native", "bytes_goldilocks", "bytes_254bit"):
            a[k] += m[k]
    tot = {k: sum(m[k] for m in man) for k in ("elements", "bytes_native", "bytes_goldilocks", "bytes_254bit")}
    return {"evidence_class": "NORMAL", "prefill_tokens": n, "committed_tensors": len(man), "totals": tot,
            "per_token": {k: v / n for k, v in tot.items()}, "per_layer": per_layer,
            "note": "committed gadget witnesses only (range-check limbs and table multiplicities are counted "
                    "in the contract accounting)",
            "layer0_tensors": [m for m in man if m["layer"] == 0]}

def run_t0(cfg: R.T0Config, argv, quick: bool) -> dict:
    t_start = time.perf_counter()
    rec = base_record(cfg.name, {"contract_stats": "SUCCESS", "dynamic_checks": "NORMAL",
                                 "relation_replay": "NORMAL", "abort_tests": "NORMAL",
                                 "quality": "NORMAL", "witness_manifest": "NORMAL"},
                      "SUCCESS", argv)
    rec["config"] = cfg.__dict__ | {"dh": cfg.dh}
    rec["weights"] = "synthetic float64, seeded (T0Config.seed); quantized to the authorized integer derivative"
    rec["calibration"] = {"prompts": 32, "length": cfg.ctx, "seed": 1001, "eval_seed": 2002,
                          "note": "calibration and evaluation prompt sets are disjoint seeds"}
    rec["variants"] = {}
    n_q = 16 if quick else 64
    for vname, wb in (("W8A8", 8), ("W4A8", 4)):
        pol = R.IntPolicy(wbits=wb)
        im = R.build_t0(cfg, pol)
        ops = G.build_t0_graph(im, elide=True)
        blk, cons = contracts_block(ops, pol, cfg.vocab, cfg.ctx, cfg.family == "gpt", include_all=(wb == 8))
        blk["policy"] = policy_dict(pol)
        blk["multipliers"] = im.mlog
        blk["bias_clipped_count"] = im.clip_log
        blk["eps_int"] = im.eps_int
        blk["residual_scale"] = im.s_r
        blk["table_digests_sha256"] = im.table_digests()
        TC.check_weights(im, cons)
        # count cross-check: analytic compiler vs executed trace (non-elided lowering)
        toks = R.random_prompts(cfg, 1, 24, 7)[0]
        _, tr, _ = im.forward(toks, record=True)
        got = R.trace_counts(tr)
        mism = []
        wl = {"kind": "prefill", "n": 24}
        for op in G.build_t0_graph(im, elide=False):
            exp = {"committed": {}, "checks": {}, "lookups": {}}
            for (nm, b, c) in op.cost.committed:
                exp["committed"][nm] = exp["committed"].get(nm, 0) + c * G.multiplicity(op, b, wl)
            for k in op.cost.checks:
                exp["checks"][k.name] = exp["checks"].get(k.name, 0) + k.count * G.multiplicity(op, k.basis, wl)
            for (t, b, c) in op.cost.lookups:
                exp["lookups"][t] = exp["lookups"].get(t, 0) + c * G.multiplicity(op, b, wl)
            if exp != got.get(op.op_id):
                mism.append(op.op_id)
        blk["count_crosscheck"] = {"evidence_class": "NORMAL", "prefill_tokens": 24, "ops": len(got),
                                   "mismatching_ops": mism, "match": not mism}
        blk["dynamic_check"] = dynamic_campaign(im, 16 if quick else 64, seed=3003)
        # relation replay over F_p (Goldilocks), non-elided contracts
        cn = TC.lower(G.build_t0_graph(im, elide=False))
        t0 = time.perf_counter()
        rp = {"evidence_class": "NORMAL", "prompts": 2 if quick else 8, "failures": 0, "counts": {}}
        for pr in R.random_prompts(cfg, rp["prompts"], 24, 4004):
            _, tr, _ = im.forward(pr, record=True)
            try:
                c = replay_trace(tr, cn, im)
                for k, v in c.items():
                    rp["counts"][k] = rp["counts"].get(k, 0) + v
            except REL.RelationError:
                rp["failures"] += 1
        rp["wall_s"] = round(time.perf_counter() - t0, 3)
        rp["field"] = "goldilocks"
        rp["note"] = "matmul accumulators are proven by the backend sumcheck and only range-checked here"
        blk["relation_replay"] = rp
        blk["abort_tests"] = abort_tests(im)
        blk["witness_manifest"] = manifest_summary(im, 32)
        t0 = time.perf_counter()
        q = R.quality_metrics(im, n_prompts=n_q, prompt_len=16, gen_len=16)
        q["evidence_class"] = "NORMAL"
        q["wall_s"] = round(time.perf_counter() - t0, 2)
        blk["quality"] = q
        rec["variants"][vname] = blk
    rec["runtime_s"] = round(time.perf_counter() - t_start, 2)
    return rec

# ------------------------------------------------------------ K-split demos
def run_ksplit(argv) -> dict:
    rec = base_record("ksplit_examples", "ANALYTIC", "ANALYTIC_COMPUTED", argv)
    ex = []

    def plans(label, K, Bx, Bw, Bb, xb, wb, note):
        e = {"case": label, "K": K, "B_x": Bx, "B_w": Bw, "B_b": Bb, "x_bits": xb, "w_bits": wb, "note": note,
             "bound": K * Bx * Bw + Bb, "bound_bits_signed": ubits(K * Bx * Bw + Bb) + 1, "plans": {}}
        for impl in ("cpu_int32", "gpu_imma_s8", "fpga_dsp48e2"):
            try:
                e["plans"][impl] = TC.plan_accumulation(K, Bx, Bw, Bb, xb, wb, impl)
            except TC.ContractError as err:
                e["plans"][impl] = {"rejected": str(err)}
        ex.append(e)

    plans("W8A8 linear, Llama-3-8B down-projection", 14336, 128, 127, 0, 8, 8, "no split needed anywhere")
    plans("W8A8 linear, hypothetical K = 2^18", 1 << 18, 128, 127, 0, 8, 8, "int32 overflows: K-split")
    plans("W8A16 linear (int16 activations), K = 4096", 4096, 1 << 15, 127, 0, 16, 8,
          "direct int32 needs K-split; IMMA uses two 8-bit activation limbs instead")
    for ctx in (512, 1024, 8192, 32768, 131072):
        plans(f"attention P.V, keys = {ctx} (P in [0,2^15], V int8)", ctx, 1 << 15, 128, 0, 16, 8,
              "probabilities are 16-bit: int32 direct splits beyond 511 keys")
    rec["accumulation_plans"] = ex
    # field-size rejection and a concrete wrap-around forgery
    pol = R.IntPolicy()
    ops = G.build_graph(MODELS["gpt2_124m"], pol)
    cons = TC.lower(ops)
    lin = cons["L0.qkv"]
    rec["babybear_rejection_gpt2_qkv"] = {"violations": TC.static_check(lin, TC.BABYBEAR_P),
                                          "goldilocks_violations": TC.static_check(lin, GOLDILOCKS_P)}
    v, M, s = 1000, 3000, 24
    forged = REL.forge_requant_wrap(v, M, s, -128, 127, TC.BABYBEAR_P)
    ok = {}
    for fname, p in (("babybear", TC.BABYBEAR_P), ("goldilocks", GOLDILOCKS_P)):
        try:
            REL.check_requant([v], [M], s, -128, 127, forged["forged_witness"],
                              {"sgn": ubits(v), "out": 8, "sat": 1}, p=p)
            ok[fname] = "ACCEPTED"
        except REL.RelationError as e:
            ok[fname] = f"REJECTED ({e})"
    rec["wraparound_forgery"] = {"v": v, "M": M, "s": s, "honest_y": forged["honest_y"],
                                 "forged_y": forged["forged_y"], "relation_result": ok,
                                 "static_check_babybear_on_this_requant": [
                                     e for e in TC.static_check(lin, TC.BABYBEAR_P) if "rq." in e][:3]}
    acc, B = 123456, 2 ** 25
    Bbig = (1 << 17) * 128 * 127           # W8A8 dot product with K = 2^17: bound > p_BabyBear / 2
    cong = lambda c, p, bnd: abs(c) <= bnd and (c - acc) % p == 0
    rec["integer_claim_decoding"] = {
        "goldilocks": {"bound": B, "accept_claim_acc": REL.accept_integer_claim(acc, acc, B, GOLDILOCKS_P),
                       "accept_claim_acc_plus_p": REL.accept_integer_claim(acc + GOLDILOCKS_P, acc, B, GOLDILOCKS_P)},
        "babybear_K_2^17": {"bound": Bbig, "bound_exceeds_p_over_2": 2 * Bbig >= TC.BABYBEAR_P,
                            "claim_acc_congruent_and_in_bound": cong(acc, TC.BABYBEAR_P, Bbig),
                            "claim_acc_minus_p_congruent_and_in_bound": cong(acc - TC.BABYBEAR_P, TC.BABYBEAR_P, Bbig),
                            "accept_after_static_check": REL.accept_integer_claim(acc - TC.BABYBEAR_P, acc, Bbig,
                                                                                  TC.BABYBEAR_P),
                            "meaning": "two different integers satisfy the field relation; the static check "
                                       "rejects this contract instead of choosing one"}}
    return rec

# --------------------------------------------------------------------- main
def write(rec: dict, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"contracts_{name}.json"
    path.write_text(json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")
    return path

def summary(recs: dict, argv) -> dict:
    rows = []
    for name, rec in recs.items():
        for vname, blk in rec.get("variants", {}).items():
            w = blk["workloads"]
            row = {"model": name, "variant": vname, "evidence_class": "ANALYTIC",
                   "num_ops": blk["num_ops"], "distinct_contracts": blk["num_distinct_contracts"],
                   "max_acc_bits": blk["max_certified_accumulator"]["bits_signed"],
                   "max_acc_op": blk["max_certified_accumulator"]["op"],
                   "acc_headroom_goldilocks": blk["max_certified_accumulator"]["headroom_bits_goldilocks"],
                   "acc_headroom_bn254": blk["max_certified_accumulator"]["headroom_bits_bn254"],
                   "max_term_bits": blk["max_certified_constraint_term"]["bits_signed"],
                   "term_headroom_goldilocks": blk["max_certified_constraint_term"]["headroom_bits_goldilocks"],
                   "static_goldilocks_pass": blk["static_check"]["goldilocks"]["pass"],
                   "static_babybear_pass": blk["static_check"]["babybear_31bit"]["pass"]}
            for wl, d in w.items():
                row[f"{wl}.lookups_naive64"] = round(d["rc_lookups_per_token"]["naive_field64"], 1)
                row[f"{wl}.lookups_naive32"] = round(d["rc_lookups_per_token"]["naive_int32"], 1)
                row[f"{wl}.lookups_tc"] = round(d["rc_lookups_per_token"]["tilecontract_opt"], 1)
                row[f"{wl}.tc_b"] = d["rc_lookups_per_token"]["tilecontract_opt_b"]
                row[f"{wl}.witness_MB_8B"] = round(d["witness_bytes_per_token_excl_tables"]["goldilocks_8B"] / 2**20, 3)
                row[f"{wl}.witness_MB_32B"] = round(d["witness_bytes_per_token_excl_tables"]["field254_32B"] / 2**20, 3)
                t = blk["tiles"][wl]["cpu_int32"]
                row[f"{wl}.tiles"] = t["tile_instances"]
                row[f"{wl}.ksplit_frac_int32"] = round(t["fraction_needing_ksplit"], 5)
            if "quality" in blk:
                q = blk["quality"]
                row["quality_evidence_class"] = "NORMAL"
                row["greedy_token_agreement"] = q["greedy_token_agreement"]
                row["teacher_forced_top1"] = q["teacher_forced_top1_agreement_vs_float"]
                row["logit_mean_abs_err"] = q["logit_mean_abs_err_vs_float"]
                row["dynamic_aborts"] = blk["dynamic_check"]["aborts"]
            rows.append(row)
    rec = base_record("summary", "ANALYTIC (quality/dynamic fields NORMAL)", "COMPUTED", argv)
    rec["rows"] = rows
    return rec

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="comma list of gpt2_124m,qwen3_0.6b,gemma3_1b,llama3_8b,"
                                                    "t0_gpt,t0_llama,ksplit or 'all'")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    argv = [str(Path(sys.argv[0]).as_posix())] + sys.argv[1:]
    want = set(a.models.split(",")) if a.models != "all" else \
        {"gpt2_124m", "qwen3_0.6b", "gemma3_1b", "llama3_8b", "t0_gpt", "t0_llama", "ksplit"}
    recs = {}
    for name in ("t0_gpt", "t0_llama"):
        if name in want:
            cfg = R.T0_GPT() if name == "t0_gpt" else R.T0_LLAMA()
            recs[name] = run_t0(cfg, argv, a.quick)
            print("wrote", write(recs[name], name))
    for name in ("gpt2_124m", "qwen3_0.6b", "gemma3_1b", "llama3_8b"):
        if name in want:
            recs[name] = run_real(name, argv)
            print("wrote", write(recs[name], name))
    if "ksplit" in want:
        print("wrote", write(run_ksplit(argv), "ksplit_examples"))
    if a.models == "all":
        print("wrote", write(summary(recs, argv), "summary"))

if __name__ == "__main__":
    main()
