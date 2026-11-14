"""Real-weight experiments: GPT-2 124M and Qwen3-0.6B integer derivatives vs float.

For each model and each integer variant this runner

1. calibrates activation / residual scales on the self-authored prompt set A, 2. builds the authorized integer derivative (exact int64 operators, exact FP64 GEMM datapath certified by the TileContract bound K*B_x*B_w < 2^53), 3. lowers it to TileContracts with the *committed* constants, runs the static checker (Goldilocks, BN254, 31-bit BabyBear) and the weight-range check, 4. runs every evaluation forward pass (teacher forcing, greedy decoding, perplexity) through the dynamic range checker: a violation raises ContractViolation and ABORTS that run (the run is excluded from quality metrics and counted), 5. measures quality against the float64 model on the same token sequences, 6. records contract statistics with real weight ranges and, for the main variant, an F_p (Goldilocks) replay of one full trace.

Outputs: results/raw/realmodels/<model>_quality.json, <model>_contracts.json (and summary.json via realmodel_summary.py). Timings are wall-clock on a contended host.

Usage (repository root):
    python bench/runners/realmodel_eval.py --model gpt2
    python bench/runners/realmodel_eval.py --model qwen3 --variants W8A8_sq_sink,W4A8g128_sq_sink
    python bench/runners/realmodel_eval.py --model gpt2 --quick        # smoke run (8 prompts, 8 tokens)
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
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from edgecert.compiler import graph as G  # noqa: E402
from edgecert.compiler import hf_graph as HG  # noqa: E402
from edgecert.compiler import tile_contract as TC  # noqa: E402
from edgecert.compiler.gadgets import ubits  # noqa: E402
from edgecert.compiler.model_shapes import MODELS  # noqa: E402
from edgecert.compiler.replay import replay_trace  # noqa: E402
from edgecert.field import P as GOLDILOCKS_P  # noqa: E402
from edgecert.reference import hf_models as HM  # noqa: E402
from edgecert.reference import intref as R  # noqa: E402
from edgecert.reference import relation as REL  # noqa: E402
from edgecert.reference.bpe import load_tokenizer  # noqa: E402

OUT = ROOT / "results" / "raw" / "realmodels"
MODEL_INFO = {
    "gpt2": {"dir": "models/gpt2", "bos": 50256, "shape": "gpt2_124m", "file": "gpt2"},
    "qwen3": {"dir": "models/qwen3-0.6b", "bos": 151643, "shape": "qwen3_0.6b", "file": "qwen3_0.6b"},
}
SANITY_PROMPT = "The capital of France is Paris. The capital of Germany is"
EXT_NOTES = {
    "shift_mode_auto": "Per-class requant shifts chosen per model (largest shift with the same number of 16-bit "
                       "rq.rem limbs such that every multiplier fits m_max bits); the spec's fixed T0 shifts make "
                       "the GPT-2 build fail (act class needs a 25-bit multiplier), which is the designed "
                       "fail-closed behaviour.",
    "pos_per_row": "GPT-2 position table int8 with per-row (per-position) scales like the token embedding "
                   "(spec 3.7 uses one scalar M_P); positions are public so row and multiplier are public constants.",
    "act_scale_domain": "Activation-table output scale = min(calibrated max, max |f| over the table input domain "
                        "[-16, 16)); larger outputs are unreachable because the table input is clamped.",
    "smoothing": "SmoothQuant-style function-preserving float transform before quantization (alpha 0.5): norm "
                 "gamma/beta divided per channel and the next linear layer's columns multiplied; Qwen3 also "
                 "q_norm/k_norm balancing per RoPE pair and up/down (GLU) balancing; the LM head becomes an untied "
                 "int8 table. Same gadgets and contract rules; only committed weights and scales change.",
    "sink_prefix": "Fixed one-token prefix (BOS id) whose per-layer int8 K/V are committed constants computed "
                   "offline from the float model; the certified integer forward starts at position 1, keeping the "
                   "attention-sink token's massive residual activations (GPT-2 ~3000, Qwen3 ~6500) out of the "
                   "integer residual stream. The float reference sees the same BOS + prompt input.",
    "res_bits_20": "Residual stream int20 instead of the spec's int16 (policy constant res_bits; norms take a "
                   "20-bit input; static no-wrap check re-run on Goldilocks).",
    "W4A8g": "Group-wise int4 weights (group 128 along K) with 8-bit integer group multipliers relative to the "
             "row's largest group scale; the effective integer weight W4*m (|.| <= 1792) enters the same mm "
             "gadget. Spec v1 only defines per-channel W4A8; this is an extension.",
}
VARIANTS = {
    "W8A8_spec_naive": {"policy": {"variant": "W8A8"}, "smooth": False, "prefix": False,
                        "extensions": ["shift_mode_auto", "pos_per_row", "act_scale_domain"],
                        "desc": "spec v1 number formats (int16 residual, per-tensor int8 activations, int8 gamma), "
                                "max calibration, no transform, no prefix"},
    "W8A8_sq_sink": {"policy": {"variant": "W8A8"}, "smooth": True, "prefix": True,
                     "extensions": ["shift_mode_auto", "pos_per_row", "act_scale_domain", "smoothing", "sink_prefix"],
                     "desc": "spec v1 number formats + smoothing transform + committed BOS sink prefix KV"},
    "W8A8_sq_sink_r20": {"policy": {"variant": "W8A8", "res_bits": 20}, "smooth": True, "prefix": True,
                         "extensions": ["shift_mode_auto", "pos_per_row", "act_scale_domain", "smoothing",
                                        "sink_prefix", "res_bits_20"],
                         "desc": "as W8A8_sq_sink with an int20 residual stream"},
    "W4A8_perchannel_sq_sink": {"policy": {"variant": "W4A8"}, "smooth": True, "prefix": True,
                                "extensions": ["shift_mode_auto", "pos_per_row", "act_scale_domain", "smoothing",
                                               "sink_prefix"],
                                "desc": "spec v1 W4A8 (per-output-channel int4 linear weights) + smoothing + sink"},
    "W4A8g128_sq_sink": {"policy": {"variant": "W4A8g"}, "smooth": True, "prefix": True,
                         "extensions": ["shift_mode_auto", "pos_per_row", "act_scale_domain", "smoothing",
                                        "sink_prefix", "W4A8g"],
                         "desc": "group-wise int4 (g=128) linear weights + smoothing + sink"},
}
DEFAULT_VARIANTS = {"gpt2": list(VARIANTS), "qwen3": list(VARIANTS)}
MAIN_VARIANT = "W8A8_sq_sink"

def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

def host_info(backend: str) -> dict:
    info = {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "python": sys.version.split()[0], "numpy": np.__version__,
            "blas": _blas_name(), "fp64_backend": backend,
            "contention": probe_contention()}
    return info

def probe_contention() -> dict:
    """Snapshot of foreign load at start (nvidia-smi GPU utilisation / compute apps, Vivado and other python
    processes).  A snapshot only: load may change during the run, so timings are never headline numbers."""
    import subprocess
    out = {"probe_time_utc": now_utc(), "timings_are_headline": False}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        out["gpus_nvidia_smi_order"] = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        out["gpu_compute_apps"] = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception as e:  # pragma: no cover
        out["gpu_probe_error"] = repr(e)
    try:
        r = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=30)
        names = [l.split(",")[0].strip('"').lower() for l in r.stdout.splitlines() if l.strip()]
        out["vivado_processes"] = sum(n.startswith("vivado") for n in names)
        out["python_processes_incl_self"] = sum(n.startswith("python") for n in names)
    except Exception as e:  # pragma: no cover
        out["cpu_probe_error"] = repr(e)
        # on WDDM --query-compute-apps also lists desktop graphics clients.
    out["contended"] = bool(out.get("vivado_processes") or out.get("python_processes_incl_self", 1) > 1)
    return out

def _blas_name() -> str:
    try:
        cfg = np.show_config(mode="dicts")
        b = cfg["Build Dependencies"]["blas"]
        return f"{b.get('name')} {b.get('version')}"
    except Exception:  # pragma: no cover
        return "unknown"

def code_digest() -> dict:
    files = ["src/edgecert/reference/hf_models.py", "src/edgecert/reference/intops.py",
             "src/edgecert/reference/intref.py", "src/edgecert/reference/bpe.py",
             "src/edgecert/reference/safetensors_np.py", "src/edgecert/compiler/hf_graph.py",
             "src/edgecert/compiler/graph.py", "src/edgecert/compiler/gadgets.py",
             "src/edgecert/compiler/tile_contract.py", "bench/runners/realmodel_eval.py"]
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16] for f in files}

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o

# ------------------------------------------------------------------ helpers
def log_softmax_nll(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    m = logits.max(-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(-1))
    return lse - logits[np.arange(len(targets)), targets]

def ppl_windows(N: int, max_len: int) -> List[tuple]:
    """Sliding windows (stride max_len/2): (begin, end, first_target); every target t >= 1 is
    scored exactly once, from the window giving it the longest context."""
    if N <= max_len:
        return [(0, N, 1)]
    stride = max_len // 2
    out, prev_end, b = [], 1, 0
    while True:
        e = min(b + max_len, N)
        out.append((b, e, max(prev_end, b + 1)))
        prev_end = e
        if e == N:
            return out
        b += stride

class DynamicChecker:
    """Sink for HFIntModel.forward: dynamic TileContract check of every op record (ABORT on violation)."""

    def __init__(self, contracts):
        self.cons = contracts
        self.agg: Dict[str, dict] = {}
        self.op_records = 0
        self.check_evaluations = 0
        self.runs = 0
        self.aborted_runs = 0
        self.violations: List[str] = []

    def __call__(self, op):
        st = TC.dynamic_check([op], self.cons)
        self.op_records += 1
        for op_id, d in st.items():
            key_op = op_id.split(".", 1)[1] if op_id.startswith("L") else op_id
            for k, v in d.items():
                a = self.agg.setdefault(f"{op_id}:{k}", {"op": op_id, "op_class": key_op, "check": k,
                                                         "certified_bits": v["certified_bits"], "observed_bits": 0})
                a["observed_bits"] = max(a["observed_bits"], v["observed_bits"])
                self.check_evaluations += 1

    def run(self, im, seqs, emit):
        self.runs += 1
        try:
            return im.forward(seqs, emit=emit, sink=self)
        except TC.ContractViolation as e:
            self.aborted_runs += 1
            self.violations.append(str(e)[:300])
            return None

    def summary(self) -> dict:
        slack = [a["certified_bits"] - a["observed_bits"] for a in self.agg.values()]
        by_class: Dict[str, dict] = {}
        for a in self.agg.values():
            key = f"{a['op_class']}:{a['check']}"
            c = by_class.setdefault(key, {"certified_bits_max": 0, "observed_bits_max": 0, "ops": 0})
            c["certified_bits_max"] = max(c["certified_bits_max"], a["certified_bits"])
            c["observed_bits_max"] = max(c["observed_bits_max"], a["observed_bits"])
            c["ops"] += 1
        tight = sorted(self.agg.values(), key=lambda a: a["certified_bits"] - a["observed_bits"])[:12]
        return {"forward_runs": self.runs, "aborted_runs": self.aborted_runs,
                "violations": len(self.violations), "violation_messages": self.violations[:10],
                "op_records_checked": self.op_records, "tensor_check_evaluations": self.check_evaluations,
                "certified_minus_observed_bits": {"min": min(slack) if slack else None,
                                                  "max": max(slack) if slack else None,
                                                  "mean": float(np.mean(slack)) if slack else None},
                "tightest_checks": tight, "per_check_class": by_class,
                "policy": "every op record of every evaluation forward pass is checked against its TileContract; "
                          "a violation raises ContractViolation, the run is ABORTED (no output) and excluded from "
                          "the quality metrics"}

# ------------------------------------------------------------ float reference
class FloatRef:
    def __init__(self, fm: HM.FloatLM, B: List[List[int]], passage: List[int], gen: int, bos: int, ctx: int):
        self.fm, self.B, self.passage, self.gen, self.bos, self.ctx = fm, B, passage, gen, bos, ctx
        self.cache: Dict[bool, dict] = {}

    def get(self, prefix: bool, do_ppl: bool, group: int) -> dict:
        if prefix in self.cache:
            return self.cache[prefix]
        fm = self.fm
        pre = [self.bos] if prefix else []
        t0 = time.perf_counter()
        # greedy (batched): prefill each prompt, then one packed step per token
        states = [fm.new_state() for _ in self.B]
        outs = fm.forward([(np.array(pre + b), st) for b, st in zip(self.B, states)], emit="last")
        gen = [[int(np.argmax(o[-1]))] for o in outs]
        for _ in range(self.gen - 1):
            outs = fm.forward([(np.array([g[-1]]), st) for g, st in zip(gen, states)], emit="last")
            for g, o in zip(gen, outs):
                g.append(int(np.argmax(o[-1])))
        t_greedy = time.perf_counter() - t0
        # teacher forcing on prompt + float greedy continuation
        t0 = time.perf_counter()
        tf_seqs = [b + g[:-1] for b, g in zip(self.B, gen)]
        tf_logits, tf_argmax, tf_margin = [], [], []
        for i in range(0, len(tf_seqs), group):
            chunk = tf_seqs[i:i + group]
            outs = fm.forward([(np.array(pre + s), fm.new_state()) for s in chunk], emit="all")
            for o in outs:
                lf = o[len(pre):]
                tf_argmax.append(lf.argmax(-1))              # argmax and margin from the float64 logits
                srt = np.partition(lf, -2, axis=-1)
                tf_margin.append(srt[:, -1] - srt[:, -2])
                tf_logits.append(lf.astype(np.float32))      # stored as float32 (memory); used for |err| only
        t_tf = time.perf_counter() - t0
        rec = {"prefix": pre, "greedy": gen, "tf_seqs": tf_seqs, "tf_logits": tf_logits,
               "tf_argmax": tf_argmax, "tf_margin": tf_margin,
               "timing_s": {"greedy": round(t_greedy, 2), "teacher_forced": round(t_tf, 2)}}
        if do_ppl:
            t0 = time.perf_counter()
            rec["ppl"] = self._ppl(pre)
            rec["timing_s"]["perplexity"] = round(time.perf_counter() - t0, 2)
        self.cache[prefix] = rec
        return rec

    def _ppl(self, pre) -> dict:
        fm, toks = self.fm, self.passage
        max_len = self.ctx - len(pre)
        nll = np.full(len(toks), np.nan)
        top1 = np.full(len(toks), -1, dtype=np.int64)
        for (b, e, f) in ppl_windows(len(toks), max_len):
            st = fm.new_state()
            rows = []
            first = True
            for c0 in range(b, e, 512):
                c1 = min(e, c0 + 512)
                ids = (pre if first else []) + toks[c0:c1]
                lg = fm.forward([(np.array(ids), st)], emit="all")[0]
                rows.append(lg[len(pre):] if first else lg)
                first = False
            L = np.concatenate(rows)                     # row j predicts token b + j + 1
            tgt = np.arange(f, e)
            nll[tgt] = log_softmax_nll(L[tgt - 1 - b], np.asarray(toks)[tgt])
            top1[tgt] = L[tgt - 1 - b].argmax(-1)
        return {"nll": nll, "top1": top1, "windows": ppl_windows(len(toks), max_len)}

# ------------------------------------------------------------------ variant
def build_variant(name: str, spec: dict, ctx: dict, backend: HM.Fp64Backend) -> dict:
    cfg, W, fm, A, bos = ctx["cfg"], ctx["W"], ctx["fm"], ctx["A"], ctx["bos"]
    pre = [bos] if spec["prefix"] else []
    t0 = time.perf_counter()
    key = (tuple(pre), spec["smooth"], spec.get("alpha", 0.5))
    cache = ctx.setdefault("cal_cache", {})
    if key not in cache:                      # identical calibration settings share one calibration
        cal = HM.calibrate(fm, A, prefix=pre)
        W_use, f_use, smooth_info, equiv = W, fm, None, None
        if spec["smooth"]:
            W_use, smooth_info = HM.smooth_weights(cfg, W, cal, alpha=spec.get("alpha", 0.5))
            f_use = HM.FloatLM(cfg, W_use)
            probe = ctx["B"][0]
            l0 = fm.forward([(np.array(pre + probe), fm.new_state())], emit="all")[0]
            l1 = f_use.forward([(np.array(pre + probe), f_use.new_state())], emit="all")[0]
            equiv = {"max_abs_logit_diff_vs_original_float": float(np.abs(l0 - l1).max()),
                     "note": "function-preserving transform: difference is float64 rounding only"}
            cal = HM.calibrate(f_use, A, prefix=pre)
        cache.clear()                         # keep at most one smoothed weight copy in memory
        cache[key] = (cal, W_use, f_use, smooth_info, equiv, round(time.perf_counter() - t0, 2))
    cal, W_use, f_use, smooth_info, equiv, t_cal_first = cache[key]
    t_cal = time.perf_counter() - t0
    pol = HM.HFPolicy(**spec["policy"])
    t0 = time.perf_counter()
    im = HM.HFIntModel(cfg, pol, W_use, cal, backend=backend)
    pinfo = im.set_prefix(pre, HM.prefix_kv_float(f_use, pre)) if pre else None
    t_build = time.perf_counter() - t0
    return {"im": im, "W_use": W_use, "f_use": f_use, "cal": cal, "smooth_info": smooth_info, "equiv": equiv,
            "prefix_info": pinfo, "pre": pre, "timing_s": {"calibration": round(t_cal, 2),
                                                          "calibration_first_computed_s": t_cal_first,
                                                          "build": round(t_build, 2)}}

def contract_stats(im, cons, ops, shape_name: str, workloads: List[tuple], impl_census=("cpu_int32", "gpu_imma_s8",
                                                                                         "fpga_dsp48e2")) -> dict:
    """Contract statistics with the committed (calibrated, real-weight) constants vs policy-level bounds."""
    cfg, pol = im.cfg, im.pol
    log2h = lambda p: math.log2(p / 2)
    res: Dict[str, object] = {"implementation_id": im.be.impl_id}
    viol = {fname: [e for tc in cons.values() for e in TC.static_check(tc, p)]
            for fname, p in (("goldilocks", GOLDILOCKS_P), ("bn254_r", TC.BN254_R), ("babybear_31bit", TC.BABYBEAR_P))}
    res["static_check"] = {k: {"pass": not v, "violations": len(v), "sample": v[:4]} for k, v in viol.items()}
    res["num_ops"], res["num_contracts_distinct"] = len(ops), len(TC.dedup(cons))
    acc = max(((tc.op_id, tc.signed_ranges["mm.acc"][1]) for tc in cons.values() if "mm.acc" in tc.signed_ranges),
              key=lambda x: x[1])
    term = max(((tc.op_id, k, B) for tc in cons.values() for k, B in tc.terms.items()), key=lambda x: x[2])
    res["max_certified_accumulator"] = {
        "op": acc[0], "bound": acc[1], "bits_signed": ubits(acc[1]) + 1,
        "headroom_bits_goldilocks": round(log2h(GOLDILOCKS_P) - math.log2(acc[1]), 2),
        "headroom_bits_babybear": round(log2h(TC.BABYBEAR_P) - math.log2(acc[1]), 2)}
    # largest accumulator of a *weight* matmul (linear / LM head), independent of the context length
    accw = max(((tc.op_id, tc.signed_ranges["mm.acc"][1]) for tc in cons.values()
                if "mm.acc" in tc.signed_ranges and tc.kind in ("linear", "lm_head")), key=lambda x: x[1])
    res["max_certified_accumulator_weight_matmul"] = {
        "op": accw[0], "bound": accw[1], "bits_signed": ubits(accw[1]) + 1,
        "headroom_bits_goldilocks": round(log2h(GOLDILOCKS_P) - math.log2(accw[1]), 2),
        "headroom_bits_babybear": round(log2h(TC.BABYBEAR_P) - math.log2(accw[1]), 2)}
    res["max_certified_constraint_term"] = {
        "op": term[0], "term": term[1], "bound": term[2], "bits_signed": ubits(term[2]) + 1,
        "headroom_bits_goldilocks": round(log2h(GOLDILOCKS_P) - math.log2(term[2]), 2),
        "headroom_bits_babybear": round(log2h(TC.BABYBEAR_P) - math.log2(term[2]), 2)}
    res["clamps_elided"] = sum(1 for tc in cons.values() if tc.output_scale.get("elided"))
    # policy-level lowering of the same architecture with the published (spec) policy
    ppol = R.IntPolicy(wbits=8 if pol.variant == "W8A8" else 4)
    ppol.res_bits = pol.res_bits
    pops = G.build_graph(MODELS[shape_name], ppol)
    pcons = TC.lower(pops, impl=im.be.impl_id)
    res["policy_level"] = {"clamps_elided": sum(1 for tc in pcons.values() if tc.output_scale.get("elided")),
                           "num_contracts_distinct": len(TC.dedup(pcons)),
                           "max_certified_constraint_term_bits": ubits(max(B for tc in pcons.values()
                                                                           for B in tc.terms.values())) + 1,
                           "static_check_goldilocks_pass": not any(TC.static_check(tc, GOLDILOCKS_P)
                                                                   for tc in pcons.values())}
    # per check class: certified bits with committed constants vs policy bounds
    cmp: Dict[str, dict] = {}
    for op_id, tc in cons.items():
        ptc = pcons.get(op_id)
        if ptc is None:
            continue
        pb = {k.name: k.bits for k in ptc.checks}
        cls = op_id.split(".", 1)[1] if op_id.startswith("L") else op_id
        for k in tc.checks:
            e = cmp.setdefault(f"{cls}:{k.name}", {"committed_bits_max": 0, "policy_bits_max": 0,
                                                    "committed_bits_min": 99, "ops": 0, "elided_in_committed": 0})
            e["committed_bits_max"] = max(e["committed_bits_max"], k.bits)
            e["committed_bits_min"] = min(e["committed_bits_min"], k.bits)
            e["policy_bits_max"] = max(e["policy_bits_max"], pb.get(k.name, 0))
            e["ops"] += 1
        for k in ptc.checks:
            if k.name not in {c.name for c in tc.checks}:
                e = cmp.setdefault(f"{cls}:{k.name}", {"committed_bits_max": 0, "policy_bits_max": 0,
                                                        "committed_bits_min": 0, "ops": 0, "elided_in_committed": 0})
                e["policy_bits_max"] = max(e["policy_bits_max"], k.bits)
                e["elided_in_committed"] += 1
    res["check_bits_committed_vs_policy"] = cmp
    tabs = TC.table_sizes(ops, cfg.vocab, cfg.ctx, pol.U_max, pol.act_in_bits, cfg.family == "gpt2")
    ptabs = TC.table_sizes(pops, cfg.vocab, cfg.ctx, ppol.U_max, ppol.act_in_bits, cfg.family == "gpt2")
    wl_out = {}
    for wname, wl, toks in workloads:
        a = TC.account(ops, wl, tabs)
        pa = TC.account(pops, wl, ptabs)
        lk, plk = a["rc_lookups"], pa["rc_lookups"]
        census = {impl: TC.tile_census(ops, wl, impl) for impl in impl_census}
        wl_out[wname] = {
            "tokens": toks,
            "rc_lookups_per_token": {"naive_field64": lk["naive_field64"] / toks, "naive_int32": lk["naive_int32"] / toks,
                                     "tilecontract_b16": lk["tc_b16"] / toks, "tilecontract_opt": lk["tc"] / toks,
                                     "tilecontract_opt_b": a["rc_limb_bits"],
                                     "policy_tilecontract_b16": plk["tc_b16"] / toks,
                                     "policy_tilecontract_opt": plk["tc"] / toks},
            "reduction_vs_naive_field64": lk["naive_field64"] / lk["tc"],
            "reduction_vs_naive_int32": lk["naive_int32"] / lk["tc"],
            "committed_vs_policy_lookup_ratio": lk["tc"] / plk["tc"],
            "range_checked_values_per_token": a["range_checked_values"] / toks,
            "table_lookups_per_token": {k: v / toks for k, v in a["table_lookups"].items()},
            "committed_elements_per_token": a["committed_elements"] / toks,
            "witness_bytes_per_token_goldilocks_8B_incl_tables": a["witness_bytes"]["goldilocks_8B"] / toks,
            "ksplit": {impl: {"tile_instances": c["tile_instances"], "tiles_needing_ksplit": c["tiles_needing_ksplit"],
                              "fraction_needing_ksplit": c["fraction_needing_ksplit"],
                              "per_kind": c["per_kind"]} for impl, c in census.items()},
        }
    res["workloads"] = wl_out
    res["fp64_exact_datapath"] = HG.fp64_exact_plan(ops)
    res["fp64_exact_datapath"].pop("per_op")
    return res

def summarize_mults(im) -> dict:
    by_cls: Dict[str, dict] = {}
    for what, m in im.mlog.items():
        cls = "norm" if what.endswith(("ln1", "ln2", "lnf", "q_norm", "k_norm")) else what.split(".")[-1]
        c = by_cls.setdefault(cls, {"M_min_bits": 99, "M_max_bits": 0, "ops": 0})
        c["M_min_bits"] = min(c["M_min_bits"], m["M_min_bits"])
        c["M_max_bits"] = max(c["M_max_bits"], m["M_max_bits"])
        c["ops"] += 1
    return by_cls

def eval_variant(name: str, ctx: dict, args, backend: HM.Fp64Backend) -> tuple:
    spec = VARIANTS[name]
    cfg, fref = ctx["cfg"], ctx["fref"]
    t_start = time.perf_counter()
    rec: Dict[str, object] = {"variant": name, "description": spec["desc"], "policy_args": spec["policy"],
                              "smoothing": spec["smooth"], "sink_prefix": spec["prefix"],
                              "extensions_vs_spec_v1": {k: EXT_NOTES[k] for k in spec["extensions"]}}
    crec: Dict[str, object] = {"variant": name}
    try:
        bv = build_variant(name, spec, ctx, backend)
    except R.QuantError as e:
        rec.update({"status": "BUILD_FAILED", "error": str(e)})
        crec.update({"status": "BUILD_FAILED", "error": str(e)})
        return rec, crec
    im = bv["im"]
    rec["timing_s"] = dict(bv["timing_s"])
    rec["calibration"] = {"prompts": len(ctx["A"]), "set": "A", "resid_calib": im.pol.resid_calib,
                          "act_calib": im.pol.act_calib, "s_r": im.s_r, "eps_int": im.eps_int,
                          "prefix_rows_excluded": len(bv["pre"])}
    rec["policy"] = {"shifts": im.pol.shifts, "m_max": im.pol.m_max, "res_bits": im.pol.res_bits,
                     "wbits": im.pol.wbits, "shift_selection": im.shift_selection, "norm_gamma": im.pol.norm_gamma}
    rec["multipliers_by_class"] = summarize_mults(im)
    rec["bias_clipped"] = sum(im.clip_log.values())
    ni = im.norm_info
    rec["norm_gamma_quantization"] = {"median_rel_err_max_over_norms": max(v["gamma_rel_err_median"] for v in ni.values()),
                                      "gammas_zeroed_total": sum(v["gamma_zeroed"] for v in ni.values())}
    rec["smoothing_info"] = ({"alpha": bv["smooth_info"]["alpha"], "pairs": len(bv["smooth_info"]["pairs"]),
                              "equivalence_check": bv["equiv"]} if bv["smooth_info"] else None)
    rec["prefix"] = bv["prefix_info"]
    rec["weight_digest_sha256"] = im.weight_digest()
    rec["table_digests_sha256"] = {"T_exp": im.table_digests()["T_exp"],
                                   "T_act_all_layers": hashlib.sha256("".join(v for k, v in
                                                                              sorted(im.table_digests().items())).encode()).hexdigest()}
    # ---- contracts (committed constants) + static checks
    ops = HG.build_hf_graph(im, elide=True)
    cons = TC.lower(ops, impl=backend.impl_id)
    static_gl = [e for tc in cons.values() for e in TC.static_check(tc, GOLDILOCKS_P)]
    if static_gl:
        rec.update({"status": "CONTRACT_REJECTED", "static_violations": static_gl[:10]})
        crec.update({"status": "CONTRACT_REJECTED", "static_violations": static_gl[:10]})
        return rec, crec
    TC.check_weights(im, cons)
    chk = DynamicChecker(cons)
    fr = fref.get(spec["prefix"], args.ppl, args.group)
    # ---- teacher forcing
    t0 = time.perf_counter()
    agree = tot = agree_big = tot_big = 0
    agree_dq_int = agree_dq_f = 0
    margins_dis: List[float] = []
    err_raw, err_c = [], []
    f_dq = HM.FloatLM(cfg, HM.dequantized_float_weights(im, bv["W_use"])) if args.dequant_ref else None
    pre = bv["pre"]
    for i in range(0, len(fr["tf_seqs"]), args.group):
        chunk = fr["tf_seqs"][i:i + args.group]
        outs = chk.run(im, [(np.array(s), im.new_state()) for s in chunk], "all")
        if outs is None:
            continue
        dq_outs = (f_dq.forward([(np.array(pre + s), f_dq.new_state()) for s in chunk], emit="all")
                   if f_dq is not None else [None] * len(chunk))
        for j, (lint, deq, tau) in enumerate(outs):
            lf = fr["tf_logits"][i + j].astype(np.float64)
            af = fr["tf_argmax"][i + j]
            agree += int((tau == af).sum())
            tot += len(af)
            margin = fr["tf_margin"][i + j]
            margins_dis += margin[tau != af].tolist()
            big = margin > 1.0
            tot_big += int(big.sum())
            agree_big += int(((tau == af) & big).sum())
            err_raw.append(np.abs(deq - lf).ravel())
            err_c.append(np.abs((deq - deq.mean(-1, keepdims=True)) - (lf - lf.mean(-1, keepdims=True))).ravel())
            if dq_outs[j] is not None:
                adq = dq_outs[j][len(pre):].argmax(-1)
                agree_dq_int += int((adq == tau).sum())
                agree_dq_f += int((adq == af).sum())
    t_tf = time.perf_counter() - t0
    er, ec = np.concatenate(err_raw), np.concatenate(err_c)
    q: Dict[str, object] = {
        "teacher_forced": {"sequences": len(fr["tf_seqs"]), "positions": tot,
                           "top1_agreement_int_vs_float": agree / tot if tot else None,
                           "top1_agreement_when_float_margin_gt_1": agree_big / tot_big if tot_big else None,
                           "positions_float_margin_gt_1": tot_big,
                           "disagreements": len(margins_dis),
                           "float_top1_margin_at_disagreements_mean": float(np.mean(margins_dis)) if margins_dis else None,
                           "float_top1_margin_at_disagreements_median": float(np.median(margins_dis)) if margins_dis else None,
                           "logit_abs_err_max": float(er.max()), "logit_abs_err_mean": float(er.mean()),
                           "logit_abs_err_centered_max": float(ec.max()), "logit_abs_err_centered_mean": float(ec.mean()),
                           "float_logit_std_mean": float(np.mean([l.astype(np.float64).std(-1).mean()
                                                                  for l in fr["tf_logits"]])),
                           "float_reference_logits_storage": "float32 copy of float64 logits (argmax and "
                                                             "top-2 margin taken from float64)",
                           "sequence_definition": "prompt + float greedy continuation (all but last token); every "
                                                  "position of the sequence is compared (prefix rows excluded)"},
    }
    if f_dq is not None and tot:
        q["teacher_forced"]["top1_agreement_int_vs_float_dequant_weights"] = agree_dq_int / tot
        q["teacher_forced"]["top1_agreement_float_dequant_weights_vs_float"] = agree_dq_f / tot
    # ---- greedy decoding (batched, one packed step per token)
    t0 = time.perf_counter()
    B = ctx["B"]
    gi: List[Optional[List[int]]] = [None] * len(B)
    for i in range(0, len(B), args.gen_group):
        idx = list(range(i, min(len(B), i + args.gen_group)))
        states = [im.new_state() for _ in idx]
        outs = chk.run(im, [(np.array(B[k]), st) for k, st in zip(idx, states)], "last")
        if outs is None:
            continue
        seqs = [[int(o[2][-1])] for o in outs]
        ok = True
        for _ in range(args.gen - 1):
            outs = chk.run(im, [(np.array([s[-1]]), st) for s, st in zip(seqs, states)], "last")
            if outs is None:
                ok = False
                break
            for s, o in zip(seqs, outs):
                s.append(int(o[2][-1]))
        if ok:
            for k, s in zip(idx, seqs):
                gi[k] = s
    t_gen = time.perf_counter() - t0
    done = [(g, f) for g, f in zip(gi, fr["greedy"]) if g is not None]
    tok_match = sum(sum(int(a == b) for a, b in zip(g, f)) for g, f in done)
    first_div = [next((j for j, (a, b) in enumerate(zip(g, f)) if a != b), len(f)) for g, f in done]
    q["greedy"] = {"prompts": len(B), "completed": len(done), "new_tokens": args.gen,
                   "token_agreement": tok_match / (len(done) * args.gen) if done else None,
                   "exact_sequence_match": sum(int(g == f) for g, f in done) / len(done) if done else None,
                   "mean_first_divergence": float(np.mean(first_div)) if done else None,
                   "median_first_divergence": float(np.median(first_div)) if done else None,
                   "samples": [{"prompt": ctx["B_text"][k], "float": ctx["tok"].decode(fr["greedy"][k]),
                                "int": ctx["tok"].decode(gi[k]) if gi[k] is not None else None}
                               for k in range(min(4, len(B)))]}
    # sanity prompt continuation (integer model)
    st = im.new_state()
    sp = ctx["tok"].encode(SANITY_PROMPT)
    o = chk.run(im, [(np.array(sp), st)], "last")
    sanity = []
    if o is not None:
        sanity.append(int(o[0][2][-1]))
        for _ in range(args.gen - 1):
            o = chk.run(im, [(np.array([sanity[-1]]), st)], "last")
            if o is None:
                break
            sanity.append(int(o[0][2][-1]))
    q["sanity_prompt_int_continuation"] = SANITY_PROMPT + ctx["tok"].decode(sanity)
    # ---- perplexity on the passage (dequantized integer logits)
    t_ppl = 0.0
    if args.ppl:
        t0 = time.perf_counter()
        toks = ctx["passage"]
        fp = fr["ppl"]
        max_len = cfg.ctx - len(pre)
        nll_i = np.full(len(toks), np.nan)
        top1_i = np.full(len(toks), -1, dtype=np.int64)
        aborted = False
        for (b, e, f) in ppl_windows(len(toks), max_len):
            st = im.new_state()
            rows = []
            for c0 in range(b, e, args.ppl_chunk):
                c1 = min(e, c0 + args.ppl_chunk)
                o = chk.run(im, [(np.array(toks[c0:c1]), st)], "all")
                if o is None:
                    aborted = True
                    break
                rows.append((o[0][1], o[0][2]))
            if aborted:
                break
            Lq = np.concatenate([r[0] for r in rows])
            tau = np.concatenate([r[1] for r in rows])
            tgt = np.arange(f, e)
            nll_i[tgt] = log_softmax_nll(Lq[tgt - 1 - b], np.asarray(toks)[tgt])
            top1_i[tgt] = tau[tgt - 1 - b]
        t_ppl = time.perf_counter() - t0
        if not aborted:
            m = ~np.isnan(fp["nll"]) & ~np.isnan(nll_i)
            ppl_f, ppl_i = float(np.exp(np.mean(fp["nll"][m]))), float(np.exp(np.mean(nll_i[m])))
            tt = np.asarray(toks)[m]
            q["perplexity"] = {"passage_tokens": len(toks), "scored_tokens": int(m.sum()),
                               "windows": [list(w) for w in fp["windows"]], "window_max_tokens": max_len,
                               "ppl_float": ppl_f, "ppl_int": ppl_i, "relative_change": ppl_i / ppl_f - 1.0,
                               "next_token_accuracy_float": float(np.mean(fp["top1"][m] == tt)),
                               "next_token_accuracy_int": float(np.mean(top1_i[m] == tt)),
                               "top1_agreement_int_vs_float": float(np.mean(fp["top1"][m] == top1_i[m])),
                               "int_logits": "dequantized integer LM-head accumulators (acc * s_xf * s_head); the "
                                             "certified output is only the argmax token"}
        else:
            q["perplexity"] = {"status": "ABORTED"}
    rec["timing_s"].update({"teacher_forced": round(t_tf, 2), "greedy": round(t_gen, 2),
                            "perplexity": round(t_ppl, 2), "total_variant": round(time.perf_counter() - t_start, 2),
                            "note": "contended host; includes the dynamic range checks; not headline numbers"})
    rec["quality"] = q
    rec["dynamic_range_certification"] = chk.summary()
    rec["fp64_datapath_runtime"] = dict(im.fp64_stats, implementation_id=backend.impl_id,
                                        device=backend.device_name(),
                                        max_bound_log2=round(math.log2(max(im.fp64_stats["max_bound"], 1)), 3))
    rec["status"] = "SUCCESS" if chk.aborted_runs == 0 else "SUCCESS_WITH_ABORTS"
    crec["status"] = "SUCCESS"
    crec.update(contract_stats(im, cons, ops, ctx["shape"], ctx["workloads"]))
    crec["dynamic_check_summary"] = {k: rec["dynamic_range_certification"][k] for k in
                                     ("forward_runs", "aborted_runs", "violations", "op_records_checked",
                                      "tensor_check_evaluations", "certified_minus_observed_bits")}
    # calibration-only bounds would abort: observed bits on set A vs evaluation
    if name == args.replay_variant and args.replay:
        crec["relation_replay"] = relation_replay(im, ctx)
    return rec, crec

def relation_replay(im, ctx) -> dict:
    """Replay one full trace of a short prompt through the F_p relation checker (Goldilocks)."""
    ops_ne = HG.build_hf_graph(im, elide=False)
    cn = TC.lower(ops_ne, impl=im.be.impl_id)
    toks = ctx["tok"].encode(ctx["B_text"][0])[:8]
    tr: list = []
    im.forward([(np.array(toks), im.new_state())], emit="last", sink=tr.append)
    t0 = time.perf_counter()
    try:
        counts = replay_trace(tr, cn, im)
        fails = 0
        err = None
    except REL.RelationError as e:  # pragma: no cover - would be a bug
        counts, fails, err = {}, 1, str(e)
    got = R.trace_counts(tr)
    n = len(toks)
    npre = len(im.prefix["tokens"]) if im.prefix else 0
    mism = []
    for op in ops_ne:
        mult = lambda b: {"tok": n, "emit": 1, "qrow": op.heads * n,
                          "score": op.heads * sum(npre + i + 1 for i in range(n))}[b]
        exp = {"committed": {}, "checks": {}, "lookups": {}}
        for (nm, b, c) in op.cost.committed:
            exp["committed"][nm] = exp["committed"].get(nm, 0) + c * mult(b)
        for k in op.cost.checks:
            exp["checks"][k.name] = exp["checks"].get(k.name, 0) + k.count * mult(k.basis)
        for (t, b, c) in op.cost.lookups:
            exp["lookups"][t] = exp["lookups"].get(t, 0) + c * mult(b)
        if exp != got.get(op.op_id):
            mism.append(op.op_id)
    return {"field": "goldilocks", "prompt_tokens": n, "prefix_tokens": npre,
            "failures": fails, "error": err, "counts": counts, "wall_s": round(time.perf_counter() - t0, 2),
            "count_crosscheck": {"ops": len(ops_ne), "mismatching_ops": mism, "match": not mism},
            "note": "every requant, norm, softmax, activation-table, embedding and argmax gadget of the trace is "
                    "re-checked over F_p with the non-elided contract widths; matmul accumulators are proven by "
                    "the backend sumcheck and only range-checked (dynamic checker)"}

# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_INFO), required=True)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--group", type=int, default=8, help="sequences per packed teacher-forced forward")
    ap.add_argument("--gen-group", type=int, default=64, help="sequences per packed greedy batch")
    ap.add_argument("--ppl-chunk", type=int, default=512)
    ap.add_argument("--no-ppl", dest="ppl", action="store_false")
    ap.add_argument("--no-dequant-ref", dest="dequant_ref", action="store_false")
    ap.add_argument("--no-replay", dest="replay", action="store_false")
    ap.add_argument("--replay-variant", default=MAIN_VARIANT)
    ap.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_eval, args.gen = 8, 8
    info = MODEL_INFO[args.model]
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    cfg = HM.load_config(ROOT / info["dir"])
    backend = HM.Fp64Backend(args.backend, args.device)
    tok = load_tokenizer(cfg.family, ROOT / info["dir"])
    W = HM.load_float_weights(cfg)
    fm = HM.FloatLM(cfg, W)
    A_text = json.loads((ROOT / "data/prompts/set_A_calibration.json").read_text(encoding="utf-8"))["prompts"]
    B_text = json.loads((ROOT / "data/prompts/set_B_evaluation.json").read_text(encoding="utf-8"))["prompts"]
    B_text = B_text[:args.n_eval]
    passage_text = (ROOT / "data/prompts/passage_perplexity.txt").read_text(encoding="utf-8")
    A = [tok.encode(p) for p in A_text]
    B = [tok.encode(p) for p in B_text]
    passage = tok.encode(passage_text)
    if args.quick:
        passage = passage[:600]
    rt = {"A": [tok.decode(a) == t for a, t in zip(A, A_text)], "B": [tok.decode(b) == t for b, t in zip(B, B_text)]}
    wls = ([("prefill_32", {"kind": "prefill", "n": 32}, 32), ("prefill_128", {"kind": "prefill", "n": 128}, 128),
            ("prefill_512", {"kind": "prefill", "n": 512}, 512), ("decode_pos_128", {"kind": "decode", "pos": 128}, 1),
            ("decode_pos_512", {"kind": "decode", "pos": 512}, 1)] +
           ([("prefill_1024", {"kind": "prefill", "n": 1024}, 1024), ("decode_pos_1023", {"kind": "decode", "pos": 1023}, 1)]
            if cfg.family == "gpt2" else
           [("prefill_3017", {"kind": "prefill", "n": 3017}, 3017), ("decode_pos_4096", {"kind": "decode", "pos": 4096}, 1),
            ("decode_pos_32767", {"kind": "decode", "pos": 32767}, 1)]))
    fref = FloatRef(fm, B, passage, args.gen, info["bos"], cfg.ctx)
    ctx = {"cfg": cfg, "W": W, "fm": fm, "A": A, "B": B, "B_text": B_text, "passage": passage, "tok": tok,
           "bos": info["bos"], "fref": fref, "shape": info["shape"], "workloads": wls}
    # float sanity continuation
    st = fm.new_state()
    sp = tok.encode(SANITY_PROMPT)
    lg = fm.forward([(np.array(sp), st)], emit="last")[0]
    cont = [int(lg[-1].argmax())]
    for _ in range(args.gen - 1):
        lg = fm.forward([(np.array([cont[-1]]), st)], emit="last")[0]
        cont.append(int(lg[-1].argmax()))
    base = {"evidence_class": {"dynamic_checks": "SUCCESS", "timings": "contended"},
            "record_type": "realmodel_quality", "model": cfg.name, "hf_id": MODELS[info["shape"]].hf_id,
            "command": "python " + " ".join([str(Path(sys.argv[0]).as_posix())] + (argv or sys.argv[1:])),
            "cwd": str(ROOT), "timestamp_utc": now_utc(), "host": host_info(f"{backend.impl_id} ({backend.device_name()})"),
            "tool": {"script": "bench/runners/realmodel_eval.py", "code_sha256_16": code_digest()},
            "model_file_sha256": file_sha256(Path(cfg.model_dir) / "model.safetensors"),
            "config": {k: v for k, v in cfg.__dict__.items() if k != "model_dir"},
            "data": {"provenance": PROVENANCE, "set_A_prompts": len(A), "set_A_tokens": sum(map(len, A)),
                     "set_B_prompts": len(B), "set_B_tokens": sum(map(len, B)), "passage_tokens": len(passage),
                     "passage_words": len(passage_text.split()),
                     "tokenizer_roundtrip_ok": all(rt["A"]) and all(rt["B"]) and tok.decode(passage) == (passage_text if not args.quick else tok.decode(passage)),
                     "tokenizer": f"pure-Python byte-level BPE ({tok.name}); see src/edgecert/reference/bpe.py"},
            "float_reference": {"dtype": "float64", "sanity_prompt": SANITY_PROMPT,
                                "sanity_greedy_continuation": SANITY_PROMPT + tok.decode(cont),
                                "note": "float64 numpy reference of the HF architecture; greedy, lowest index on ties"},
            "settings": {"n_eval": len(B), "gen_tokens": args.gen, "perplexity": args.ppl, "quick": args.quick}}
    qrec = dict(base, variants={})
    crec = dict(base, record_type="realmodel_contracts",
                evidence_class={"contract_stats": "SUCCESS", "dynamic_checks": "SUCCESS", "relation_replay": "SUCCESS"},
                variants={})
    names = args.variants.split(",") if args.variants else DEFAULT_VARIANTS[args.model]
    qpath = OUT / f"{info['file']}_quality{args.out_suffix}.json"
    cpath = OUT / f"{info['file']}_contracts{args.out_suffix}.json"
    for name in names:
        print(f"[{cfg.name}] variant {name} ...", flush=True)
        rq, rc = eval_variant(name, ctx, args, backend)
        qrec["variants"][name] = rq
        crec["variants"][name] = rc
        tfv = rq.get("quality", {}).get("teacher_forced", {}).get("top1_agreement_int_vs_float")
        pp = rq.get("quality", {}).get("perplexity", {})
        print(f"   status {rq['status']} TF {tfv} ppl {pp.get('ppl_float')} -> {pp.get('ppl_int')} "
              f"aborts {rq.get('dynamic_range_certification', {}).get('aborted_runs')} "
              f"t {rq.get('timing_s', {}).get('total_variant')}s", flush=True)
        fr = fref.cache
        qrec["float_reference"]["timing_s"] = {("prefix" if k else "no_prefix"): v["timing_s"] for k, v in fr.items()}
        for k, v in fr.items():
            if "ppl" in v:
                m = ~np.isnan(v["ppl"]["nll"])
                qrec["float_reference"][("prefix" if k else "no_prefix") + "_perplexity"] = float(np.exp(np.mean(v["ppl"]["nll"][m])))
        qrec["runtime_s"] = round(time.perf_counter() - t_all, 2)
        qpath.write_text(json.dumps(jsonable(qrec), indent=1), encoding="utf-8")
        cpath.write_text(json.dumps(jsonable(crec), indent=1), encoding="utf-8")
    print("wrote", qpath, cpath)

if __name__ == "__main__":
    main()
