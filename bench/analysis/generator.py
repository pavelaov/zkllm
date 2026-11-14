"""Full-ZK latency for GPT-2 / Qwen3-0.6B / Gemma 3 1B."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.GEN / "generation.json"

# ------------------------------------------------------------------ REPORTED literature inputs
LIT = {
    "deepprove_gpt2": dict(system="DeepProve", cite="gailly2026deepprove", model="gpt2_124m",
                           basis="per_token", value_s=60.0 / 174.0,
                           reported="174 tokens/min for GPT-2", evidence="REPORTED",
                           where="ePrint 2026/1112 abstract (docs/literature_update.md, sec. 1)",
                           hardware="as reported by the authors (configuration of the abstract)"),
    "deepprove_gemma3": dict(system="DeepProve", cite="gailly2026deepprove", model="gemma3_1b",
                             basis="per_token", value_s=60.0 / 86.0,
                             reported="86 tokens/min for Gemma 3", evidence="REPORTED",
                             where="ePrint 2026/1112 abstract (docs/literature_update.md, sec. 1)",
                             hardware="as reported by the authors"),
    "deepprove_readme_gpt2": dict(system="DeepProve (README)", cite="deepprove-repo", model="gpt2_124m",
                                  basis="per_token", value_s=7.64 * 60.0 / 512.0,
                                  reported="512 tokens in 7.64 min, 24-core EPYC 9254", evidence="REPORTED",
                                  where="DeepProve README (docs/literature_update.md, sec. 1)",
                                  hardware="24-core AMD EPYC 9254"),
    "deepprove_readme_gemma3": dict(system="DeepProve (README)", cite="deepprove-repo", model="gemma3_1b",
                                    basis="per_token", value_s=18.95 * 60.0 / 512.0,
                                    reported="512 tokens in 18.95 min, 24-core EPYC 9254", evidence="REPORTED",
                                    where="DeepProve README (docs/literature_update.md, sec. 1)",
                                    hardware="24-core AMD EPYC 9254"),
    "zkgpt_gpt2": dict(system="zkGPT", cite="qu2025zkgpt", model="gpt2_124m", basis="per_forward",
                       value_s=25.0, bound="upper", reported="GPT-2 proved in < 25 s", evidence="REPORTED",
                       where="USENIX Security 2025 abstract (docs/literature_update.md, sec. 1)",
                       hardware="CPU (32 threads per planning report 01)",
                       secondary={"prove_s": 21.8, "verify_s": 0.35, "proof": "101 K",
                                  "where": "planning/01_Technical_Investigation_Report.md sec. 4.1 (from the PDF; "
                                           "not re-verified by the bib agent)"}),
    "joltatlas_gpt2": dict(system="Jolt Atlas", cite="benno2026joltatlas", model="gpt2_124m",
                           basis="per_forward", value_s=38.0, bound="approx",
                           reported="GPT-2 125M in about 38 s on an M3 MacBook, 16 GB", evidence="REPORTED",
                           where="arXiv 2602.17452 (docs/literature_update.md, sec. 1)",
                           hardware="Apple M3 laptop, 16 GB"),
    "zkllm_13b": dict(system="zkLLM", cite="sun2024zkllm", model="13B", params=13e9, basis="per_forward",
                      value_s=15 * 60.0, bound="upper", reported="13B-parameter model proved in < 15 min",
                      evidence="REPORTED", where="CCS 2024, arXiv 2404.16109 (docs/literature_update.md, sec. 1)",
                      hardware="A100 40 GB (planning report 01, sec. 4.1)"),
    "deepprove_verify": dict(system="DeepProve", cite="gailly2026deepprove", basis="verify",
                             low_s=1.0, high_s=3.7, reported="verification 1 to 3.7 s", evidence="REPORTED",
                             where="ePrint 2026/1112 abstract (docs/literature_update.md, sec. 1)"),
}

def _assumptions_from_doc() -> list[str]:
    """The A1..A6 paragraphs of the module docstring, one string each (single source of truth)."""
    body = __doc__.split('key "assumptions"):')[1].split("Sources of the REPORTED")[0]
    out, cur = [], []
    for ln in body.splitlines():
        s = ln.strip()
        if s[:2] in {f"A{i}" for i in range(1, 10)} and cur:
            out.append(" ".join(cur))
            cur = []
        if s:
            cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out

ASSUMPTIONS = _assumptions_from_doc()

DECODE_WL = "decode_pos_128"
PREFILL_WL = "prefill_128"

def work(key: str, workload: str) -> dict | None:
    obj, path, src = C.load_contracts(key)
    if obj is None:
        return None
    sh = C.shape_of(obj)
    row = C.contract_row(obj, "W8A8", workload)
    if sh is None or row is None:
        return None
    mac = C.macs_per_token(sh, workload)
    lk = (row["tc"] or 0.0) + (row["table_lookups"] or 0.0)
    return {"model": key, "workload": workload, "contracts_file": C.rel(path), "contracts_source": src,
            "macs": mac["total"], "macs_split": mac, "rc_lookups": row["tc"],
            "table_lookups": row["table_lookups"], "work": mac["total"] + lk,
            "params": sh.get("params"), "evidence_class": "ANALYTIC"}

def build() -> dict:
    models = ["gpt2_124m", "qwen3_0.6b", "gemma3_1b"]
    W = {(m, wl): work(m, wl) for m in models for wl in (DECODE_WL, PREFILL_WL)}
    missing = [f"{m}/{wl}" for (m, wl), v in W.items() if v is None]
    rec = {
        "record": "edgecert_fullzk_generation",
        "status": "MEASURED" if not missing else "PARTIAL",
        "evidence_class": "MEASURED",
        "command": "python bench/analysis/generation.py",
        "timestamp": C.now_iso(),
        "host": C.host_info(),
        "note": "ZK backend",
        "assumptions": ASSUMPTIONS,
        "literature_inputs": LIT,
        "work_per_token": {f"{m}/{wl}": v for (m, wl), v in W.items()},
        "missing_inputs": missing,
        "per_token": {}, "per_forward": {}, "validation": {},
    }
    if missing:
        C.warn("generation: missing contract inputs " + ", ".join(missing))

    def ratio(a, b, wl):
        wa, wb = W.get((a, wl)), W.get((b, wl))
        return wa["work"] / wb["work"] if wa and wb else None

    def pratio(a, b):
        wa, wb = W.get((a, DECODE_WL)), W.get((b, DECODE_WL))
        return wa["params"] / wb["params"] if wa and wb and wa["params"] and wb["params"] else None

    # ---------------- per generated token, DeepProve basis
    base = LIT["deepprove_gpt2"]["value_s"]
    g3 = LIT["deepprove_gemma3"]["value_s"]
    p_g2 = W[("gpt2_124m", DECODE_WL)]["params"] if W.get(("gpt2_124m", DECODE_WL)) else None
    p_g3 = W[("gemma3_1b", DECODE_WL)]["params"] if W.get(("gemma3_1b", DECODE_WL)) else None
    # empirical exponent from the two reported DeepProve points (A5)
    alpha = (math.log(g3 / base) / math.log(p_g3 / p_g2)) if (p_g2 and p_g3) else None
    for m in models:
        r_ops = ratio(m, "gpt2_124m", DECODE_WL)
        r_par = pratio(m, "gpt2_124m")
        if r_ops is None:
            continue
        opscaled = base * r_ops
        emp = base * (r_par ** alpha) if (r_par and alpha is not None) else None
        vals = [v for v in (opscaled, emp) if v is not None]
        rec["per_token"][m] = {
            "basis": "deepprove_gpt2", "workload": DECODE_WL,
            "work_ratio_vs_gpt2": r_ops, "param_ratio_vs_gpt2": r_par,
            "prove_s_per_token_est": opscaled,
            "prove_s_per_token_empirical_est": emp,
            "prove_s_per_token_low_est": min(vals), "prove_s_per_token_high_est": max(vals),
            "tokens_per_min_est": 60.0 / opscaled,
            "verify_s_low_est": LIT["deepprove_verify"]["low_s"], "verify_s_high_est": LIT["deepprove_verify"]["high_s"],
            "reported_same_model_s": (base if m == "gpt2_124m" else g3 if m == "gemma3_1b" else None),
            "evidence_class": "MEASURED",
        }
    if ("gemma3_1b" in rec["per_token"]):
        pred = rec["per_token"]["gemma3_1b"]["prove_s_per_token_est"]
        rec["validation"]["deepprove_gemma3"] = {
            "what": "op-count rule (A1) applied GPT-2 -> Gemma 3 vs the Gemma 3 number reported by DeepProve",
            "predicted_s_per_token_est": pred, "reported_s_per_token": g3,
            "predicted_over_reported_est": pred / g3,
            "empirical_param_exponent_est": alpha,
            "readme_ratio_gemma3_over_gpt2": LIT["deepprove_readme_gemma3"]["value_s"] / LIT["deepprove_readme_gpt2"]["value_s"],
            "abstract_ratio_gemma3_over_gpt2": g3 / base,
            "evidence_class": "MEASURED",
        }

    # ---------------- per forward pass, zkGPT / Jolt Atlas / zkLLM bases
    for key in ("zkgpt_gpt2", "joltatlas_gpt2"):
        lit = LIT[key]
        for m in models:
            r = ratio(m, "gpt2_124m", PREFILL_WL)
            if r is None:
                continue
            rec["per_forward"].setdefault(key, {})[m] = {
                "basis": key, "workload": PREFILL_WL, "work_ratio_vs_gpt2": r,
                "prove_s_est": lit["value_s"] * r, "bound": lit.get("bound"),
                "reported_same_model_s": lit["value_s"] if m == "gpt2_124m" else None,
                "evidence_class": "MEASURED"}
    lit = LIT["zkllm_13b"]
    for m in models:
        wm = W.get((m, DECODE_WL))
        if not wm or not wm["params"]:
            continue
        r = wm["params"] / lit["params"]
        rec["per_forward"].setdefault("zkllm_13b", {})[m] = {
            "basis": "zkllm_13b", "param_ratio_vs_13b": r, "prove_s_est": lit["value_s"] * r,
            "bound": "upper", "evidence_class": "MEASURED"}
    return rec

def main() -> int:
    rec = build()
    C.write_json(OUT, rec)
    print(f"wrote {C.rel(OUT)} (status {rec['status']})")
    for m, v in rec["per_token"].items():
        print(f"  per-token {m:12s} op-count {v['prove_s_per_token_est']:.3f} s  empirical "
              f"{(v['prove_s_per_token_empirical_est'] or float('nan')):.3f} s  (est.)")
    for b, d in rec["per_forward"].items():
        for m, v in d.items():
            print(f"  per-forward {b:15s} {m:12s} {v['prove_s_est']:.1f} s (est.)")
    if rec["validation"]:
        v = rec["validation"]["deepprove_gemma3"]
        print(f"  validation: predicted/reported Gemma 3 = {v['predicted_over_reported_est']:.2f} (est.)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
