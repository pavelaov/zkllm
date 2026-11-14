"""Why static contract bounds: what a calibration-only bound would do on real weights.

Builds the main integer derivative (W8A8_sq_sink, as in realmodel_eval.py), runs the exact integer forward on every calibration prompt (set A) and every evaluation prompt (set B) and records, per (op, witness check), the largest observed |value|.  A "calibration-only" contract would certify exactly the set-A maxima; the record counts how many set-B prompts would then ABORT (some value above its set-A maximum), and compares set-A-observed bits, set-B-observed bits, the committed static TileContract bits and the policy-level bits. The static contracts themselves never abort (checked on every run by the dynamic checker).

Usage: python bench/runners/realmodel_calibonly.py --model gpt2|qwen3
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "bench" / "runners"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import realmodel_eval as RE  # noqa: E402
from edgecert.compiler import hf_graph as HG  # noqa: E402
from edgecert.compiler import tile_contract as TC  # noqa: E402
from edgecert.compiler.gadgets import ubits  # noqa: E402
from edgecert.reference import hf_models as HM  # noqa: E402
from edgecert.reference.bpe import load_tokenizer  # noqa: E402

class MaxRecorder:
    def __init__(self, cons):
        self.cons = cons
        self.cur = {}

    def __call__(self, op):
        TC.dynamic_check([op], self.cons)          # the static contract is still enforced
        for g in op.gadgets:
            for k, a in list(g.checks.items()) + [("val:" + k, v) for k, v in g.committed.items() if k in ("acc", "y")]:
                a = np.asarray(a)
                if a.size:
                    key = f"{op.op_id}|{g.tag}.{k}"
                    self.cur[key] = max(self.cur.get(key, 0), int(np.abs(a).max()))

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt2", "qwen3"], required=True)
    ap.add_argument("--variant", default=RE.MAIN_VARIANT)
    args = ap.parse_args(argv)
    t_all = time.perf_counter()
    info = RE.MODEL_INFO[args.model]
    cfg = HM.load_config(ROOT / info["dir"])
    tok = load_tokenizer(cfg.family, ROOT / info["dir"])
    W = HM.load_float_weights(cfg)
    fm = HM.FloatLM(cfg, W)
    A = [tok.encode(p) for p in json.loads((ROOT / "data/prompts/set_A_calibration.json").read_text())["prompts"]]
    B = [tok.encode(p) for p in json.loads((ROOT / "data/prompts/set_B_evaluation.json").read_text())["prompts"]]
    ctx = {"cfg": cfg, "W": W, "fm": fm, "A": A, "B": B, "bos": info["bos"]}
    bv = RE.build_variant(args.variant, RE.VARIANTS[args.variant], ctx, HM.Fp64Backend("numpy"))
    im = bv["im"]
    ops = HG.build_hf_graph(im)
    cons = TC.lower(ops, impl=im.be.impl_id)
    rec_ = MaxRecorder(cons)
    per_prompt = {"A": [], "B": []}
    for name, prompts in (("A", A), ("B", B)):
        for p in prompts:
            rec_.cur = {}
            im.forward([(np.array(p), im.new_state())], emit="last", sink=rec_)
            per_prompt[name].append(rec_.cur)
    amax = {}
    for d in per_prompt["A"]:
        for k, v in d.items():
            amax[k] = max(amax.get(k, 0), v)
    bmax = {}
    for d in per_prompt["B"]:
        for k, v in d.items():
            bmax[k] = max(bmax.get(k, 0), v)
    exceed = [k for k in bmax if bmax[k] > amax.get(k, -1)]
    aborts = sum(1 for d in per_prompt["B"] if any(v > amax.get(k, -1) for k, v in d.items()))
    # bit-granular calibration-only contract: range check width = ubits(set-A max) per (op, check)
    aborts_bits = sum(1 for d in per_prompt["B"] if any(ubits(v) > ubits(amax.get(k, 0)) for k, v in d.items()))
    exceed_bits = [k for k in bmax if ubits(bmax[k]) > ubits(amax.get(k, 0))]
    cert = {}
    for tc in cons.values():
        for c in tc.checks:
            cert[f"{tc.op_id}|{c.name}"] = c.bits
    by_class = {}
    for k in bmax:
        op, chk = k.split("|")
        cls = (op.split(".", 1)[1] if op.startswith("L") else op) + ":" + chk
        e = by_class.setdefault(cls, {"bits_A_max": 0, "bits_B_max": 0, "certified_bits_max": 0, "keys": 0,
                                      "exceed_A": 0})
        e["bits_A_max"] = max(e["bits_A_max"], ubits(amax.get(k, 0)))
        e["bits_B_max"] = max(e["bits_B_max"], ubits(bmax[k]))
        e["certified_bits_max"] = max(e["certified_bits_max"], cert.get(k, 0))
        e["keys"] += 1
        e["exceed_A"] += int(bmax[k] > amax.get(k, -1))
    out = {"status": "MEASURED_SUCCESS", "evidence_class": "MEASURED", "record_type": "calibration_only_bounds",
           "model": cfg.name, "variant": args.variant,
           "command": "python " + " ".join([Path(sys.argv[0]).as_posix()] + (argv or sys.argv[1:])),
           "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "prompts_A": len(A), "prompts_B": len(B), "run": "prompt-only prefill, exact integer forward",
           "static_contract_aborts": 0,
           "calibration_only_contract": {
               "checked_values": len(bmax), "values_exceeding_set_A_max_on_set_B": len(exceed),
               "set_B_prompts_that_would_abort": aborts, "abort_fraction": aborts / len(B),
               "examples": sorted(exceed, key=lambda k: bmax[k] / max(amax.get(k, 1), 1), reverse=True)[:12]},
           "calibration_only_contract_bit_granular": {
               "rule": "each range check gets ubits(set-A max) bits", "values_needing_more_bits_on_set_B": len(exceed_bits),
               "set_B_prompts_that_would_abort": aborts_bits, "abort_fraction": aborts_bits / len(B),
               "examples": exceed_bits[:12]},
           "per_check_class": by_class,
           "data_provenance": "self-authored prompts (data/prompts), not a standard benchmark",
           "runtime_s": round(time.perf_counter() - t_all, 1), "contention": "CPU shared with Vivado and other jobs"}
    RE.OUT.mkdir(parents=True, exist_ok=True)
    path = RE.OUT / f"{info['file']}_calibration_only_bounds.json"
    path.write_text(json.dumps(RE.jsonable(out), indent=1), encoding="utf-8")
    print("wrote", path, "aborts", aborts, "bit-granular aborts", aborts_bits, "/", len(B))

if __name__ == "__main__":
    main()
