"""Collect results/raw/realmodels/*_{quality,contracts}.json (+ fp64 datapath bench) into summary.json.

Usage: python bench/runners/realmodel_summary.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "raw" / "realmodels"
MODELS = ("gpt2", "qwen3_0.6b")

def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d

def main():
    summ = {"status": "SUMMARY", "record_type": "realmodel_summary",
            "command": "python bench/runners/realmodel_summary.py",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "evidence_classes": {"quality": "MEASURED", "dynamic_range_checks": "MEASURED",
                                 "relation_replay": "MEASURED", "contract_statistics": "ANALYTIC",
                                 "timings": "MEASURED, never headline (see contention_flags per model)",
                                 "diagnostic_ablation": "SIMULATION (float fake-quant, not the certified model)"},
            "data_provenance": None, "models": {}}
    for m in MODELS:
        qp, cp = OUT / f"{m}_quality.json", OUT / f"{m}_contracts.json"
        if not qp.exists():
            summ["models"][m] = {"status": "NOT_RUN"}
            continue
        q = json.loads(qp.read_text(encoding="utf-8"))
        c = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {"variants": {}}
        summ["data_provenance"] = g(q, "data", "provenance")
        ms = {"hf_id": q.get("hf_id"), "model_file_sha256": q.get("model_file_sha256"),
              "float_sanity_continuation": g(q, "float_reference", "sanity_greedy_continuation"),
              "float_perplexity": {k: v for k, v in q["float_reference"].items() if k.endswith("perplexity")},
              "eval": q.get("settings"), "status": q.get("status"), "source_command": q.get("command"),
              "source_timestamp_utc": q.get("timestamp_utc"), "contention_flags": g(q, "host", "contention"),
              "fp64_backend": g(q, "host", "fp64_backend"),
              "paper_primary": q.get("paper_primary"), "best_certified_variant": q.get("best_certified_variant"),
              "w8a8_quality_finding": q.get("w8a8_quality_finding"), "variants": {}}
        for vn, v in q["variants"].items():
            cv = c["variants"].get(vn, {})
            tf = g(v, "quality", "teacher_forced", default={})
            gr = g(v, "quality", "greedy", default={})
            pp = g(v, "quality", "perplexity", default={})
            dyn = g(v, "dynamic_range_certification", default={})
            wl = g(cv, "workloads", default={})
            wl_key = "prefill_128" if "prefill_128" in wl else (next(iter(wl)) if wl else None)
            ms["variants"][vn] = {
                "status": v.get("status"), "paper_primary": v.get("paper_primary"),
                "description": v.get("description"),
                "extensions_vs_spec_v1": sorted((v.get("extensions_vs_spec_v1") or {}).keys()),
                "tf_top1_agreement": tf.get("top1_agreement_int_vs_float"),
                "tf_top1_agreement_margin_gt_1": tf.get("top1_agreement_when_float_margin_gt_1"),
                "tf_positions": tf.get("positions"),
                "tf_float_dequant_weights_vs_float": tf.get("top1_agreement_float_dequant_weights_vs_float"),
                "logit_abs_err_mean": tf.get("logit_abs_err_mean"), "logit_abs_err_max": tf.get("logit_abs_err_max"),
                "logit_abs_err_centered_mean": tf.get("logit_abs_err_centered_mean"),
                "logit_abs_err_centered_max": tf.get("logit_abs_err_centered_max"),
                "greedy_token_agreement_32": gr.get("token_agreement"),
                "greedy_exact_match": gr.get("exact_sequence_match"),
                "greedy_mean_first_divergence": gr.get("mean_first_divergence"),
                "ppl_float": pp.get("ppl_float"), "ppl_int": pp.get("ppl_int"),
                "ppl_relative_change": pp.get("relative_change"),
                "sanity_int_continuation": g(v, "quality", "sanity_prompt_int_continuation"),
                "dynamic_checks": {"forward_runs": dyn.get("forward_runs"), "aborted_runs": dyn.get("aborted_runs"),
                                   "violations": dyn.get("violations"),
                                   "tensor_check_evaluations": dyn.get("tensor_check_evaluations"),
                                   "op_records_checked": dyn.get("op_records_checked")},
                "residual_scale": g(v, "calibration", "s_r"), "res_bits": g(v, "policy", "res_bits"),
                "shifts": g(v, "policy", "shifts"),
                "contracts": {
                    "static_goldilocks_pass": g(cv, "static_check", "goldilocks", "pass"),
                    "static_bn254_pass": g(cv, "static_check", "bn254_r", "pass"),
                    "static_babybear_pass": g(cv, "static_check", "babybear_31bit", "pass"),
                    "babybear_violations": g(cv, "static_check", "babybear_31bit", "violations"),
                    "max_acc": g(cv, "max_certified_accumulator"),
                    "max_acc_weight_matmul": g(cv, "max_certified_accumulator_weight_matmul"),
                    "max_term": g(cv, "max_certified_constraint_term"),
                    "distinct_contracts": g(cv, "num_contracts_distinct"), "ops": g(cv, "num_ops"),
                    "clamps_elided_committed": g(cv, "clamps_elided"),
                    "clamps_elided_policy": g(cv, "policy_level", "clamps_elided"),
                    "workload": wl_key,
                    "rc_lookups_per_token": g(wl, wl_key, "rc_lookups_per_token") if wl_key else None,
                    "reduction_vs_naive_field64": g(wl, wl_key, "reduction_vs_naive_field64") if wl_key else None,
                    "reduction_vs_naive_int32": g(wl, wl_key, "reduction_vs_naive_int32") if wl_key else None,
                    "committed_vs_policy_lookup_ratio": g(wl, wl_key, "committed_vs_policy_lookup_ratio") if wl_key else None,
                    "ksplit_fraction": {k: {impl: g(wv, "ksplit", impl, "fraction_needing_ksplit")
                                            for impl in ("cpu_int32", "gpu_imma_s8", "fpga_dsp48e2")}
                                        for k, wv in wl.items()},
                    "fp64_exact_all_certified": g(cv, "fp64_exact_datapath", "all_certified"),
                    "fp64_worst_bound_log2": g(cv, "fp64_exact_datapath", "worst", "bound_log2"),
                    "relation_replay": ({k: g(cv, "relation_replay", k) for k in ("failures", "counts", "wall_s",
                                                                                   "count_crosscheck")}
                                        if g(cv, "relation_replay") else None)},
                "timing_s": v.get("timing_s")}
        # certified (committed constants) vs policy-level vs observed bits, main variant
        mv = g(q, "paper_primary", "variant", default="W8A8_sq_sink")
        if mv in q["variants"] and mv in c["variants"]:
            cmp = g(c, "variants", mv, "check_bits_committed_vs_policy", default={})
            obs = g(q, "variants", mv, "dynamic_range_certification", "per_check_class", default={})
            rows = []
            for key, e in sorted(cmp.items()):
                o = obs.get(key, {})
                rows.append({"check": key, "committed_bits_max": e["committed_bits_max"],
                             "policy_bits_max": e["policy_bits_max"], "observed_bits_max": o.get("observed_bits_max"),
                             "elided_in_committed": e["elided_in_committed"]})
            for key, o in sorted(obs.items()):
                if key.split(":", 1)[1].startswith("range:"):
                    rows.append({"check": key, "committed_bits_max": o["certified_bits_max"], "policy_bits_max": None,
                                 "observed_bits_max": o["observed_bits_max"], "elided_in_committed": 0})
            tot_c = sum(r["committed_bits_max"] for r in rows if r["policy_bits_max"] is not None)
            tot_p = sum(r["policy_bits_max"] for r in rows if r["policy_bits_max"] is not None)
            ms["bits_certified_vs_policy_vs_observed"] = {
                "variant": mv, "evidence_class": "ANALYTIC (certified, policy) / MEASURED (observed)",
                "sum_over_check_classes_committed_bits": tot_c, "sum_over_check_classes_policy_bits": tot_p,
                "rows": rows,
                "note": "observed bits of offset checks (rq.out, nm.mu, rq.rem) include the offset/remainder "
                        "structure and naturally fill their width; sgn/sat/acc rows show value magnitudes"}
        compat = {}
        for alias, vn in (("W8A8", mv), ("W4A8", "W4A8g128_sq_sink")):
            vv = ms["variants"].get(vn)
            if vv:
                compat[alias] = {"source_variant": vn, "quality": {
                    "teacher_forced_top1_agreement_vs_float": vv["tf_top1_agreement"],
                    "perplexity_float": vv["ppl_float"], "perplexity_int": vv["ppl_int"],
                    "perplexity_delta": (vv["ppl_int"] - vv["ppl_float"]) if vv["ppl_int"] is not None else None}}
        ms["chart_compat"] = compat
        ab = OUT / f"{m}_actquant_ablation.json"
        if ab.exists():   # diagnostic float fake-quant: SIMULATION, never the certified model
            a = json.loads(ab.read_text(encoding="utf-8"))
            ms["diagnostic_simulation_actquant_ablation"] = {
                "evidence_class": a.get("evidence_class"), "status": a.get("status"), "file": ab.name,
                "prompts": a.get("prompts"), "positions": a.get("positions"), "results": a.get("results"),
                "leave_one_out": a.get("leave_one_out"), "note": a.get("method")}
        cb = OUT / f"{m}_calibration_only_bounds.json"
        if cb.exists():
            ms["calibration_only_bounds_file"] = cb.name
        summ["models"][m] = ms
    fb = OUT / "fp64_datapath_bench.json"
    if fb.exists():
        b = json.loads(fb.read_text(encoding="utf-8"))
        rows = []
        for r in b["results"]:
            row = {"shape": r["shape"], "M": r["M"], "K": r["K"], "N": r["N"],
                   "numpy_int64_s": g(r, "numpy_int64", "median_s"), "numpy_fp64_s": g(r, "numpy_fp64_exact", "median_s"),
                   "numpy_fp64_match": g(r, "numpy_fp64_exact", "match_int64")}
            for k, v in r.items():
                if k.startswith("cupy_fp64_exact_dev"):
                    row[k + "_s"] = v["median_s"]
                    row[k + "_kernel_s"] = v["kernel_only_median_s"]
                    row[k + "_match"] = v["match_int64"]
                    row[k + "_device"] = v["device"]
            rows.append(row)
        summ["fp64_datapath"] = {"evidence_class": "MEASURED (contended)", "rows": rows,
                                 "all_match": all(v for r in rows for k, v in r.items() if k.endswith("match") and v is not None),
                                 "end_to_end_gpt2": b.get("end_to_end_gpt2")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("wrote", OUT / "summary.json")

if __name__ == "__main__":
    sys.exit(main())
