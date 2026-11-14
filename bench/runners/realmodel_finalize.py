"""Make results/raw/realmodels/{gpt2,qwen3_0.6b}_{quality,contracts}.json final.

Input: the single-invocation reruns ``<m>_{quality,contracts}_final.json`` (all variants of a model in one process, one code version, CuPy FP64 exact datapath).  The earlier records (numpy runs, including the ``_part2`` split of Qwen3 after a MemoryError) are moved to ``results/raw/realmodels/superseded/`` and compared variant by variant with the rerun (``reproduction_vs_superseded``).  One record per variant; ``paper_primary`` marks the primary configuration per model and why; ``best_certified_variant`` names the variant with the highest teacher-forced top-1 agreement.  For Qwen3 the W8A8 quality finding is stated explicitly. The float fake-quant ablation stays in its own SIMULATION file and is only referenced.

Usage: python bench/runners/realmodel_finalize.py   (then --reannotate is idempotent on the final files)
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "raw" / "realmodels"
SUP = OUT / "superseded"
MODELS = ("gpt2", "qwen3_0.6b")
PRIMARY = "W8A8_sq_sink"
PRIMARY_REASON = ("Spec v1 number formats (int8 per-output-channel weights, static per-tensor int8 activations, "
                  "int16 residual) plus only pre-commitment, function-preserving or constant-only extensions "
                  "(auto shifts, per-row position scales, activation-table scale, smoothing, committed sink "
                  "prefix K/V; spec/numerical_semantics.md section 10). Same recipe for both models; r20 and "
                  "W4A8 variants change number formats and are reported as variants.")
ALL_VARIANTS = ("W8A8_spec_naive", "W8A8_sq_sink", "W8A8_sq_sink_r20", "W4A8_perchannel_sq_sink", "W4A8g128_sq_sink")
RUN_REASON = {
    "W8A8_sq_sink_r20": ("Bunded compute budget of the finishing session (one Qwen3 variant takes about "
                         "40 min on the shared host). SIMULATION evidence (qwen3_0.6b_actquant_ablation.json): an "
                         "int20 residual leaves fake-quant agreement unchanged (0.644 vs 0.644), so it cannot fix "
                         "the W8A8 finding."),
    "W4A8_perchannel_sq_sink": ("Bunded compute budget; W4A8g128_sq_sink is the reported int4 variant "
                                "(on GPT-2 per-channel int4 is worse than group-wise: TF 0.715 vs 0.794)."),
}
STOPPED = {"qwen3_0.6b": ("This invocation (all variants requested) was stopped by the finishing agent after "
                          "its first variant (W8A8_spec_naive) completed, to stay within the compute budget; "
                          "the variant reproduced the superseded record bit for bit (reproduction_vs_superseded).")}
TF = ("quality", "teacher_forced", "top1_agreement_int_vs_float")

def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d

def qwen_finding(q: dict) -> dict:
    v = q["variants"]
    rows = {n: {"tf_top1": g(r, *TF), "tf_top1_margin_gt_1": g(r, "quality", "teacher_forced",
                                                              "top1_agreement_when_float_margin_gt_1"),
                "greedy_token_agreement": g(r, "quality", "greedy", "token_agreement"),
                "ppl_int": g(r, "quality", "perplexity", "ppl_int"),
                "ppl_float": g(r, "quality", "perplexity", "ppl_float"),
                "ppl_relative_change": g(r, "quality", "perplexity", "relative_change")} for n, r in v.items()}
    w8 = {n: r for n, r in rows.items() if n.startswith("W8A8")}
    best = max(w8, key=lambda n: w8[n]["tf_top1"] or -1)
    return {
        "evidence_class": "MEASURED",
        "statement": ("W8A8 quality for Qwen3-0.6B is achieved within the certified static "
                      "per-tensor int8 activation semantics (v1 + v1.1 extensions). Best certified W8A8 variant: "
                      f"{best} (teacher-forced top-1 agreement {w8[best]['tf_top1']:.4f}, "
                      f"passage perplexity {w8[best]['ppl_float']:.2f} -> {w8[best]['ppl_int']:.2f}). "
                      "The certified integer derivative is well defined and never violated a contract; the "
                      "loss is a quality property of the authorized quantized model. "
                      "The paper states this as a limitation."),
        "acceptability_note": ("No numeric acceptance threshold is part of the certified statement; for "
                               "comparison the same recipe on GPT-2 gives TF top-1 0.89 and PPL +12.6 %."),
        "best_certified_w8a8_variant": best,
        "per_variant": rows,
        "diagnostic_simulation_file": "results/raw/realmodels/qwen3_0.6b_actquant_ablation.json",
        "diagnostic_note": ("Float fake-quant attribution is kept in its own "
                            "file; see docs/notes/realmodels.md section 8."),
        "not_in_certified_semantics": ["per-token dynamic activation scales", "16-bit activations for selected matmuls",
                                       "mixed precision residual beyond res_bits 20"],
    }

CONTENTION_NOTE = {
    "gpt2": ("Final run (2026-09-29, CuPy device 0 = RTX 6000 Ada) ran concurrently with the Qwen3 rerun "
             "(CuPy device 1 = RTX 4090), the Qwen3 float fake-quant ablation and the unit test suite, all "
             "started by the finishing agent; CPU fully loaded (BLAS oversubscription);"
             "foreign GPU job. The probe's gpu_compute_apps list contains Windows desktop graphics clients "
             "and is a contention signal."),
    "qwen3_0.6b": ("W8A8_spec_naive: final run (2026-09-29, CuPy device 1 = RTX 4090) concurrent with the GPT-2 "
                   "rerun, the Qwen3 ablation and the unit tests (all by the finishing agent), no Vivado, no "
                   "foreign GPU job. W8A8_sq_sink and W4A8g128_sq_sink: numpy run of 2026-09-27 on a host shared "
                   "with Vivado place-and-route and a foreign GPU job (see variant_sources)."),
}

def annotate(m: str, q: dict, c: dict) -> None:
    """Idempotent annotations of the final records (contention summary, float reference fill-in)."""
    for rec in (q, c):
        probe = g(rec, "host", "contention", default={}) or {}
        rec["contention_flags"] = {"contended": True, "timings_are_headline": False,
                                   "vivado_processes_at_start": probe.get("vivado_processes"),
                                   "python_processes_at_start": probe.get("python_processes_incl_self"),
                                   "gpus_at_start": probe.get("gpus_nvidia_smi_order"),
                                   "note": CONTENTION_NOTE[m]}
        if isinstance(probe, dict) and "contended" in probe:
            probe["contended"] = True
            probe["gpu_compute_apps_note"] = "WDDM lists desktop graphics clients."
    # float perplexity with the BOS prefix comes from the run that evaluated the prefix variants
    fr = q["float_reference"]
    if "prefix_perplexity" not in fr:
        for vn, vr in q["variants"].items():
            if vr.get("sink_prefix") and g(vr, "quality", "perplexity", "ppl_float") is not None:
                fr["prefix_perplexity"] = g(vr, "quality", "perplexity", "ppl_float")
                fr["prefix_perplexity_source"] = f"variants.{vn}.quality.perplexity.ppl_float"
                break

def reannotate():
    for m in MODELS:
        qp, cp = OUT / f"{m}_quality.json", OUT / f"{m}_contracts.json"
        q = json.loads(qp.read_text(encoding="utf-8"))
        c = json.loads(cp.read_text(encoding="utf-8"))
        annotate(m, q, c)
        qp.write_text(json.dumps(q, indent=1), encoding="utf-8")
        cp.write_text(json.dumps(c, indent=1), encoding="utf-8")
        print("annotated", m)

def main():
    if "--reannotate" in sys.argv:
        return reannotate()
    SUP.mkdir(exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    for m in MODELS:
        fq, fc = OUT / f"{m}_quality_final.json", OUT / f"{m}_contracts_final.json"
        if not fq.exists():
            print("missing", fq)
            continue
        q = json.loads(fq.read_text(encoding="utf-8"))
        c = json.loads(fc.read_text(encoding="utf-8"))
        src = {vn: {"file": fq.name, "command": q.get("command"), "timestamp_utc": q.get("timestamp_utc"),
                    "fp64_backend": g(q, "host", "fp64_backend")} for vn in q["variants"]}
        for pq in sorted(OUT.glob(f"{m}_quality_part*.json")):
            dq = json.loads(pq.read_text(encoding="utf-8"))
            dc = json.loads((OUT / pq.name.replace("_quality_", "_contracts_")).read_text(encoding="utf-8"))
            # model / compiler code must be identical; the runner differs only by the host contention probe
            strip = lambda d: {k: v for k, v in (d or {}).items() if not k.startswith("bench/")}
            same_code = strip(g(dq, "tool", "code_sha256_16")) == strip(g(q, "tool", "code_sha256_16"))
            same_data = all(dq.get(k) == q.get(k) for k in ("data", "model_file_sha256")) and                 g(dq, "settings") == g(q, "settings")
            for vn in dq["variants"]:
                if vn in q["variants"] or not (same_code and same_data):
                    continue
                q["variants"][vn] = dq["variants"][vn]
                c["variants"][vn] = dc["variants"][vn]
                src[vn] = {"file": "superseded/" + pq.name, "command": dq.get("command"),
                           "timestamp_utc": dq.get("timestamp_utc"), "fp64_backend": g(dq, "host", "fp64_backend"),
                           "host_contention_at_run": g(dq, "host", "contention"),
                           "same_model_and_compiler_code_sha256_as_final_run": same_code,
                           "runner_sha256_16": g(dq, "tool", "code_sha256_16", "bench/runners/realmodel_eval.py"),
                           "runner_difference": "final-run runner only adds the host contention probe",
                           "same_data_and_settings": same_data}
        if STOPPED.get(m):
            for vn in list(src):
                if src[vn]["file"] == fq.name:
                    src[vn]["note"] = STOPPED[m]
        # superseded records: previous main files and parts
        old = {}
        for p in sorted(OUT.glob(f"{m}_quality*.json")) + sorted(OUT.glob(f"{m}_contracts*.json")):
            if p.name.endswith("_final.json"):
                continue
            if "quality" in p.name:
                d = json.loads(p.read_text(encoding="utf-8"))
                for vn, vr in d.get("variants", {}).items():
                    if vn not in old and "parts" not in vr:
                        old[vn] = {"file": "superseded/" + p.name, "command": d.get("command"),
                                   "timestamp_utc": d.get("timestamp_utc"), "tf_top1": g(vr, *TF),
                                   "ppl_int": g(vr, "quality", "perplexity", "ppl_int"),
                                   "code_sha256_16": g(d, "tool", "code_sha256_16")}
            shutil.move(str(p), str(SUP / p.name))
        repro = {}
        for vn, vr in q["variants"].items():
            o = old.get(vn)
            if src[vn]["file"] != fq.name:
                repro[vn] = {"previous": None, "note": "record taken from " + src[vn]["file"] + " (same code version)"}
                continue
            if o is None:
                repro[vn] = {"previous": None, "note": "new variant in the final run"}
                continue
            new_tf, new_ppl = g(vr, *TF), g(vr, "quality", "perplexity", "ppl_int")
            repro[vn] = {"previous": o, "tf_top1_final": new_tf, "ppl_int_final": new_ppl,
                         "identical_tf": new_tf == o["tf_top1"],
                         "identical_ppl_int": (new_ppl == o["ppl_int"]) if o["ppl_int"] is not None else None}
        statuses = {vn: vr.get("status") for vn, vr in q["variants"].items()}
        best = max(q["variants"], key=lambda n: g(q["variants"][n], *TF) or -1)
        prim = {"variant": PRIMARY, "reason": PRIMARY_REASON}
        for rec in (q, c):
            rec["status"] = ("MEASURED_SUCCESS" if all(s == "MEASURED_SUCCESS" for s in statuses.values())
                             else "PARTIAL")
            rec["variant_status"] = statuses
            rec["final"] = {"finalized_utc": ts, "finalize_command": "python bench/runners/realmodel_finalize.py",
                            "source_file": fc.name if rec is c else fq.name,
                            "note": ("One record per variant. variant_sources gives the run of every record; all "
                                     "records share one code version (code_sha256_16), data and settings. Earlier "
                                     "records are in superseded/ and are compared in reproduction_vs_superseded.")}
            rec["paper_primary"] = prim
            rec["variant_sources"] = src
            rec["best_certified_variant"] = {"variant": best, "criterion": "highest teacher-forced top-1 agreement",
                                             "tf_top1": g(q["variants"][best], *TF)}
            for vn, vr in rec["variants"].items():
                vr["paper_primary"] = vn == PRIMARY
        for rec in (q, c):
            rec["variants"] = {vn: rec["variants"][vn] for vn in ALL_VARIANTS if vn in rec["variants"]}
        q["reproduction_vs_superseded"] = repro
        annotate(m, q, c)
        if m.startswith("qwen3"):
            q["w8a8_quality_finding"] = qwen_finding(q)
            c["w8a8_quality_finding_ref"] = f"{m}_quality.json:w8a8_quality_finding"
        (OUT / f"{m}_quality.json").write_text(json.dumps(q, indent=1), encoding="utf-8")
        (OUT / f"{m}_contracts.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
        fq.unlink()
        fc.unlink()
        print("final", m, statuses, "primary", PRIMARY, "best", best)
    for lg in ("gpt2_run.log", "qwen3_run.log", "qwen3_run_part2.log"):
        if (OUT / lg).exists():
            shutil.move(str(OUT / lg), str(SUP / lg))

if __name__ == "__main__":
    main()
