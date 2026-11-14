"""Generate paper/generated/numbers.tex (LaTeX macros) and paper/generated/NUMBERS.md from results/.

Run from the repository root:  python bench/analysis/make_numbers.py
"""
from __future__ import annotations

import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT_TEX = C.GEN / "numbers.tex"
OUT_MD = C.GEN / "NUMBERS.md"
MACROS: list[dict] = []

def _fmt(v, kind: str, nd: int) -> str:
    if isinstance(v, str):
        return v
    if kind == "int":
        return C.fmt_int(v)
    if kind == "pct":
        return f"{100 * v:.{nd}f}"
    if kind == "sci":
        e = int(math.floor(math.log10(abs(v))))
        m = v / 10 ** e
        return f"{m:.{nd}f}\\times10^{{{e}}}"
    s = f"{v:.{nd}f}"
    if abs(v) >= 1000 and nd == 0:
        s = C.fmt_int(abs(v))
        s = ("-" + s) if v < 0 else s
    if s.startswith("-"):
        s = "\\ensuremath{-}" + s[1:]
    return s

def N(name: str, value, src: str, field: str, ev: str, kind: str = "float", nd: int = 1, note: str = ""):
    """Register one macro. value None -> pending (source missing or field absent)."""
    assert re.fullmatch(r"Num[A-Za-z]+", name), name
    assert not any(m["name"] == name for m in MACROS), "duplicate " + name
    if value is None:
        C.pending(f"numbers.tex \\{name}", f"{src} ({field})")
        body = "\{pending}"
        shown = "pending"
    else:
        shown = _fmt(value, kind, nd)
        if kind == "sci":
    MACROS.append({"name": name, "body": body, "value": shown, "src": src, "field": field, "ev": ev,
                   "note": note, "pending": value is None})

def g(obj, path, default=None):
    return C.get(obj, path, default) if obj is not None else default

# ============================================================ attacks / security (C1)
def attacks():
    src = "results/raw/attacks/matrix.json"
    o = C.load_json(src)
    s = g(o, "summary") or {}
    N("NumAttackCases", s.get("n_rows"), src, "summary.n_rows", "SUCCESS", "int")
    N("NumAttackNegatives", s.get("n_negative"), src, "summary.n_negative", "SUCCESS", "int")
    N("NumAttackPositives", s.get("n_positive"), src, "summary.n_positive", "SUCCESS", "int")
    N("NumAttackPassed", s.get("n_pass"), src, "summary.n_pass", "SUCCESS", "int")
    N("NumAttackFailed", s.get("n_fail"), src, "summary.n_fail", "SUCCESS", "int")
    N("NumAttackReleasedNegatives", s.get("negatives_released_to_actuator"), src,
      "summary.negatives_released_to_actuator", "SUCCESS", "int")
    for k, nm in [("REJECT", "Reject"), ("ABORT", "Abort"), ("UNCERTIFIED", "Uncertified"), ("ACCEPT", "Accept")]:
        N(f"NumAttackExpected{nm}", g(s, f"by_expected.{k}.n"), src, f"summary.by_expected.{k}.n", "SUCCESS", "int")
    cats = sorted({r["category"] for r in (o or {}).get("rows", []) if r["category"] != "positive control"})
    N("NumAttackCategories", len(cats) if o else None, src, "distinct rows[].category except 'positive control'",
      "SUCCESS", "int")
    checks = sorted({r["detecting_check"] for r in (o or {}).get("rows", []) if r.get("detecting_check")})
    N("NumAttackDetectingChecks", len(checks) if o else None, src, "distinct rows[].detecting_check", "SUCCESS", "int")
    N("NumAttackSuiteWallS", s.get("suite_wall_s"), src, "summary.suite_wall_s", "SUCCESS", nd=1)
    b = g(o, "honest_timing.backends") or {}
    host, emu = b.get("python-host", {}), b.get("sumfold-lane-emulator-L4", {})
    ms = lambda d, k: (g(d, f"{k}.median") * 1e3) if g(d, f"{k}.median") is not None else None  # noqa: E731
    N("NumToyProveMs", ms(host, "prove_s"), src, "honest_timing.backends.python-host.prove_s.median x1e3", "SUCCESS", nd=0)
    N("NumToyVerifyMs", ms(host, "verify_s"), src, "honest_timing.backends.python-host.verify_s.median x1e3", "SUCCESS", nd=0)
    N("NumToySumFoldProveMs", ms(emu, "prove_s"), src, "honest_timing.backends.sumfold-lane-emulator-L4.prove_s.median x1e3",
      "SUCCESS", nd=0)
    N("NumToySumFoldVerifyMs", ms(emu, "verify_s"), src,
      "honest_timing.backends.sumfold-lane-emulator-L4.verify_s.median x1e3", "SUCCESS", nd=0)
    pb = g(host, "proof_bytes_canonical_json.median")
    N("NumToyProofKB", pb / 1e3 if pb else None, src, "python-host.proof_bytes_canonical_json.median / 1e3", "SUCCESS", nd=0)
    N("NumToyMaxTokens", g(host, "output_tokens.max"), src, "python-host.output_tokens.max", "SUCCESS", "int")
    N("NumToyReps", g(o, "honest_timing.reps"), src, "honest_timing.reps", "SUCCESS", "int")
    N("NumToySoundnessLogTwo", g(o, "config.interactive_soundness_bound_T8.log2_bound"), src,
      "config.interactive_soundness_bound_T8.log2_bound (sketch, pending review)", "ANALYTIC", nd=1)
    N("NumToySoundnessNumerator", g(o, "config.interactive_soundness_bound_T8.numerator"), src,
      "config.interactive_soundness_bound_T8.numerator (over p^2)", "ANALYTIC", "int")
    src2 = "results/raw/attacks/transparent_scaling.json"
    t = C.load_json(src2)
    pts = (t or {}).get("points", [])
    N("NumTranspProveMinS", min(p["prove_s_median"] for p in pts) if pts else None, src2, "min points[].prove_s_median",
      "SUCCESS", nd=3)
    N("NumTranspProveMaxS", max(p["prove_s_median"] for p in pts) if pts else None, src2, "max points[].prove_s_median",
      "SUCCESS", nd=3)
    N("NumTranspProofMinKB", min(p["proof_bytes_canonical_json"] for p in pts) / 1e3 if pts else None, src2,
      "min points[].proof_bytes_canonical_json / 1e3", "SUCCESS", nd=0)
    N("NumTranspProofMaxKB", max(p["proof_bytes_canonical_json"] for p in pts) / 1e3 if pts else None, src2,
      "max points[].proof_bytes_canonical_json / 1e3", "SUCCESS", nd=0)
    N("NumTranspSumcheckBytesMin", min(p["sumcheck_part_bytes"] for p in pts) if pts else None, src2,
      "min points[].sumcheck_part_bytes", "ANALYTIC", "int")
    N("NumTranspSumcheckBytesMax", max(p["sumcheck_part_bytes"] for p in pts) if pts else None, src2,
      "max points[].sumcheck_part_bytes", "ANALYTIC", "int")

# ============================================================ contracts (C2)
MODEL_TAG = {"gpt2_124m": "GptTwo", "qwen3_0.6b": "Qwen", "gemma3_1b": "Gemma", "llama3_8b": "LlamaEight"}

def contracts():
    all64, all32, heads = [], [], []
    for key, tag in MODEL_TAG.items():
        obj, path, source = C.load_contracts(key)
        src = C.rel(path) if path else f"results/raw/contracts/contracts_{key}.json"
        r = C.contract_row(obj, "W8A8", "prefill_128") if obj else None
        rr = r or {}
        v = g(obj, "variants.W8A8") or {}
        N(f"Num{tag}Ops", v.get("num_ops"), src, "variants.W8A8.num_ops", "ANALYTIC", "int")
        N(f"Num{tag}DistinctContracts", v.get("num_distinct_contracts"), src, "variants.W8A8.num_distinct_contracts",
          "ANALYTIC", "int")
        N(f"Num{tag}HeadroomGoldilocks", rr.get("term_head_gl"), src,
          "variants.W8A8.max_certified_constraint_term.headroom_bits_goldilocks", "ANALYTIC", nd=1)
        N(f"Num{tag}HeadroomBabyBear", rr.get("term_head_bb"), src,
          "log2((p_BB-1)/2) - log2(max_certified_constraint_term.bound)", "ANALYTIC", nd=1)
        N(f"Num{tag}AccHeadroomGoldilocks", rr.get("acc_head_gl"), src,
          "variants.W8A8.max_certified_accumulator.headroom_bits_goldilocks", "ANALYTIC", nd=1)
        N(f"Num{tag}AccBits", rr.get("acc_bits"), src, "log2(max_certified_accumulator.bound)", "ANALYTIC", nd=0)
        N(f"Num{tag}TermBits", rr.get("term_bits"), src, "log2(max_certified_constraint_term.bound)", "ANALYTIC", nd=1)
        N(f"Num{tag}LookupsNaiveSixtyFour", rr.get("naive64"), src,
          "workloads.prefill_128.rc_lookups_per_token.naive_field64", "ANALYTIC", "int")
        N(f"Num{tag}LookupsIntThirtyTwo", rr.get("naive32"), src,
          "workloads.prefill_128.rc_lookups_per_token.naive_int32", "ANALYTIC", "int")
        N(f"Num{tag}LookupsTileContract", rr.get("tc"), src,
          "workloads.prefill_128.rc_lookups_per_token.tilecontract_opt", "ANALYTIC", "int")
        N(f"Num{tag}LookupReduction", rr.get("red64"), src, "workloads.prefill_128.reduction_vs_naive_field64",
          "ANALYTIC", nd=2)
        N(f"Num{tag}LookupReductionInt", rr.get("red32"), src, "workloads.prefill_128.reduction_vs_naive_int32",
          "ANALYTIC", nd=2)
        N(f"Num{tag}WitnessMBGoldilocks", rr["wit_gl_B"] / 1e6 if r else None, src,
          "workloads.prefill_128.witness_bytes_per_token_excl_tables.goldilocks_8B / 1e6", "ANALYTIC", nd=1)
        N(f"Num{tag}WitnessMBBigField", rr["wit_254_B"] / 1e6 if r else None, src,
          "workloads.prefill_128.witness_bytes_per_token_excl_tables.field254_32B / 1e6", "ANALYTIC", nd=1)
        N(f"Num{tag}WitnessMBNaive", rr["wit_naive_gl_B"] / 1e6 if r else None, src,
          "workloads.prefill_128.witness_bytes_per_token_excl_tables.naive_field64_goldilocks_8B / 1e6", "ANALYTIC", nd=1)
        if obj:
            heads.append(rr.get("term_head_gl"))
            for wl in (g(v, "workloads") or {}):
                w = g(v, f"workloads.{wl}")
                all64.append(w.get("reduction_vs_naive_field64"))
                all32.append(w.get("reduction_vs_naive_int32"))
        if key == "gpt2_124m":
            N("NumGptTwoBabyBearViolations", g(v, "static_check.babybear_31bit.violations_count"), src,
              "variants.W8A8.static_check.babybear_31bit.violations_count", "ANALYTIC", "int")
            ks = g(v, "tiles.decode_pos_512.cpu_int32.fraction_needing_ksplit")
            N("NumGptTwoKsplitPctDecodeFiveTwelve", ks, src, "tiles.decode_pos_512.cpu_int32.fraction_needing_ksplit",
              "ANALYTIC", "pct", nd=1)
            N("NumWitnessFieldRatio", (rr["wit_254_B"] / rr["wit_gl_B"]) if r else None, src,
              "field254_32B / goldilocks_8B (prefill_128)", "ANALYTIC", nd=1)
            N("NumGptTwoLayers", g(obj, "model_shape.n_layers"), src, "model_shape.n_layers", "ANALYTIC", "int")
    all64 = [x for x in all64 if x]
    all32 = [x for x in all32 if x]
    srcs = "results/raw/contracts/contracts_{gpt2_124m,qwen3_0.6b,gemma3_1b,llama3_8b}.json"
    N("NumLookupReductionMin", min(all64) if all64 else None, srcs, "min W8A8 workloads.*.reduction_vs_naive_field64",
      "ANALYTIC", nd=2)
    N("NumLookupReductionMax", max(all64) if all64 else None, srcs, "max W8A8 workloads.*.reduction_vs_naive_field64",
      "ANALYTIC", nd=2)
    N("NumLookupReductionIntMin", min(all32) if all32 else None, srcs, "min W8A8 workloads.*.reduction_vs_naive_int32",
      "ANALYTIC", nd=2)
    N("NumLookupReductionIntMax", max(all32) if all32 else None, srcs, "max W8A8 workloads.*.reduction_vs_naive_int32",
      "ANALYTIC", nd=2)
    heads = [h for h in heads if h is not None]
    N("NumMinHeadroomGoldilocks", min(heads) if heads else None, srcs,
      "min over models of max_certified_constraint_term.headroom_bits_goldilocks", "ANALYTIC", nd=1)
    # T0 (SUCCESS quality, ANALYTIC counts)
    for key, tag in (("t0_gpt", "Gpt"), ("t0_llama", "Llama")):
        src = f"results/raw/contracts/contracts_{key}.json"
        o = C.load_json(src)
        for var, vt in (("W8A8", "EightEight"), ("W4A8", "FourEight")):
            q = g(o, f"variants.{var}.quality") or {}
            N(f"NumTzero{tag}TopOne{vt}", q.get("teacher_forced_top1_agreement_vs_float"), src,
              f"variants.{var}.quality.teacher_forced_top1_agreement_vs_float", "SUCCESS", "pct", nd=1)
            N(f"NumTzero{tag}Greedy{vt}", q.get("greedy_token_agreement"), src,
              f"variants.{var}.quality.greedy_token_agreement", "SUCCESS", "pct", nd=1)
        N(f"NumTzero{tag}DynamicAborts", g(o, "variants.W8A8.dynamic_check.aborts"), src,
          "variants.W8A8.dynamic_check.aborts", "SUCCESS", "int")
        N(f"NumTzero{tag}DynamicPrompts", g(o, "variants.W8A8.dynamic_check.prompts"), src,
          "variants.W8A8.dynamic_check.prompts", "SUCCESS", "int")
        N(f"NumTzero{tag}ReplayTraces", g(o, "variants.W8A8.relation_replay.prompts"), src,
          "variants.W8A8.relation_replay.prompts", "SUCCESS", "int")
        N(f"NumTzero{tag}ReplayFailures", g(o, "variants.W8A8.relation_replay.failures"), src,
          "variants.W8A8.relation_replay.failures", "SUCCESS", "int")
        at = g(o, "variants.W8A8.abort_tests") or []
        N(f"NumTzero{tag}AbortTests", len(at) if o else None, src, "len(variants.W8A8.abort_tests)", "SUCCESS", "int")
        N(f"NumTzero{tag}AbortTestsAborted", sum(1 for a in at if a.get("aborted")) if o else None, src,
          "count(variants.W8A8.abort_tests[].aborted)", "SUCCESS", "int")
        if key == "t0_gpt":
            wm = g(o, "variants.W8A8.witness_manifest.per_token") or {}
            N("NumTzeroWitnessElemsPerToken", wm.get("elements"), src,
              "variants.W8A8.witness_manifest.per_token.elements", "SUCCESS", "int")
            N("NumTzeroWitnessKBGoldilocks", wm["bytes_goldilocks"] / 1e3 if wm else None, src,
              "witness_manifest.per_token.bytes_goldilocks / 1e3", "SUCCESS", nd=1)
            N("NumTzeroWitnessKBBigField", wm["bytes_254bit"] / 1e3 if wm else None, src,
              "witness_manifest.per_token.bytes_254bit / 1e3", "SUCCESS", nd=1)
            N("NumTzeroReplayRequantElems", g(o, "variants.W8A8.relation_replay.counts.rq_elements"), src,
              "variants.W8A8.relation_replay.counts.rq_elements", "SUCCESS", "int")

# ============================================================ kernels (C4, CPU/GPU)
def kernels():
    src = "results/raw/kernels/cpu_bench.json"
    cpu = C.load_json(src)
    cont = " (contended host)" if (cpu or {}).get("contended") else ""
    f, u = C.cpu_records(cpu, "gcc", "fused"), C.cpu_records(cpu, "gcc", "unfused")
    r1 = f.get((24, 1))
    N("NumCpuOneThreadNsPerEntry", r1["ns_per_elem_median"] if r1 else None, src,
      "results[gcc,fused,n=24,threads=1].ns_per_elem_median" + cont, "SUCCESS", nd=1)
    N("NumCpuOneThreadMs", r1["total_stats_s"]["median"] * 1e3 if r1 else None, src,
      "results[gcc,fused,n=24,threads=1].total_stats_s.median x1e3" + cont, "SUCCESS", nd=0)
    best = min(((r["total_stats_s"]["median"], th) for (n, th), r in f.items() if n == 24), default=None)
    N("NumCpuBestMs", best[0] * 1e3 if best else None, src, "min over threads results[gcc,fused,n=24].total median" + cont,
      "SUCCESS", nd=1)
    N("NumCpuBestThreads", best[1] if best else None, src, "argmin threads of the above", "SUCCESS", "int")
    for th, tag in ((1, "OneThread"), (32, "ThirtyTwo")):
        a, b = f.get((24, th)), u.get((24, th))
        N(f"NumCpuFusionGain{tag}", (b["total_stats_s"]["median"] / a["total_stats_s"]["median"]) if a and b else None,
          src, f"unfused/fused total median, gcc, n=24, {th} threads" + cont, "SUCCESS", nd=2)
    N("NumCpuConfigs", len((cpu or {}).get("results", [])) if cpu else None, src, "len(results)", "SUCCESS", "int")
    srcg = "results/raw/kernels/gpu_bench.json"
    gpu = C.load_json(srcg)
    devs = {}
    for dname, cont_d, recs in C.gpu_devices(gpu):
        tag = "Ada" if "6000" in dname else "FourNinety" if "4090" in dname else None
        if tag:
            devs[tag] = (dname, cont_d, recs)
    for tag in ("FourNinety", "Ada"):      # fixed macro set: missing devices become 'pending'
        dname, cont_d, recs = devs.get(tag, ("RTX 4090" if tag == "FourNinety" else "RTX 6000 Ada", False, {}))
        a, b = recs.get((24, "fused")), recs.get((24, "unfused"))
        cs = " (contended GPU)" if (cont_d and not C.idle_verified("kernels")[0]) else ""
        N(f"NumGpu{tag}FusedMs", a["kernel_stats_s"]["median"] * 1e3 if a else None, srcg,
          f"devices[{dname}].results[n=24,fused].kernel_stats_s.median x1e3" + cs, "SUCCESS", nd=2)
        N(f"NumGpu{tag}FusedWallMs", a["wall_stats_s"]["median"] * 1e3 if a else None, srcg,
          f"devices[{dname}].results[n=24,fused].wall_stats_s.median x1e3" + cs, "SUCCESS", nd=2)
        N(f"NumGpu{tag}FusionGain", (b["kernel_stats_s"]["median"] / a["kernel_stats_s"]["median"]) if a and b else None,
          srcg, f"devices[{dname}] unfused/fused kernel median, n=24" + cs, "SUCCESS", nd=2)
        N(f"NumGpu{tag}NsPerEntry", a["wall_stats_s"]["median"] / 2 ** 24 * 1e9 if a else None, srcg,
          f"devices[{dname}].results[n=24,fused].wall median / 2^24" + cs, "SUCCESS", nd=2)
        if tag == "FourNinety":
            dev = next((d for d in (gpu or {}).get("devices", []) if "4090" in C.get(d, "device.name", "")), {})
            mb = next((m for m in dev.get("pass_microbench", []) if m.get("n") == 24), None)
            N("NumGpuFourNinetyPassFusionGain", mb.get("speedup_unfused_over_fused_at_min") if mb else None, srcg,
              "devices[RTX 4090].pass_microbench[n=24].speedup_unfused_over_fused_at_min" + cs, "SUCCESS", nd=2)
    nver = sum(1 for d in (gpu or {}).get("devices", []) for r in d.get("results", []) if r.get("verify_ok"))
    N("NumGpuConfigsVerified", nver if gpu else None, srcg, "count devices[].results[].verify_ok", "SUCCESS", "int")
    srco = "results/raw/kernels/opcount.json"
    op = C.load_json(srco)
    v = g(op, "values.24") or {}
    Nn = v.get("N")
    N("NumBaseMulPerEntry", v.get("base_mul_per_index"), srco, "values.24.base_mul_per_index", "ANALYTIC", nd=0)
    N("NumBytesPerEntryFused", g(v, "fused.bytes_total") / Nn if Nn else None, srco, "values.24.fused.bytes_total / N",
      "ANALYTIC", nd=0)
    N("NumBytesPerEntryUnfused", g(v, "unfused.bytes_total") / Nn if Nn else None, srco,
      "values.24.unfused.bytes_total / N", "ANALYTIC", nd=0)
    N("NumResidentBytesFused", g(v, "fused.resident_bytes") / Nn if Nn else None, srco,
      "values.24.fused.resident_bytes / N", "ANALYTIC", nd=0)
    N("NumResidentBytesPingPong", g(v, "unfused.resident_bytes") / Nn if Nn else None, srco,
      "values.24.unfused.resident_bytes / N", "ANALYTIC", nd=0)
    N("NumTrafficRatio", v.get("traffic_ratio_unfused_over_fused"), srco,
      "values.24.traffic_ratio_unfused_over_fused", "ANALYTIC", nd=2)

# ============================================================ FPGA (C4)
def fpga():
    src = "results/raw/fpga_sim/sim_results.json"
    sim = C.load_json(src)
    cases = (sim or {}).get("cases", [])
    N("NumRtlCases", len(cases) if sim else None, src, "len(cases)", "SIMULATION", "int")
    N("NumRtlCasesMatch", sum(1 for c in cases if c.get("match")) if sim else None, src, "count(cases[].match)",
      "SIMULATION", "int")
    fu = (sim or {}).get("field_units", [])
    N("NumRtlFieldVectors", fu[0]["vectors"] if fu else None, src, "field_units[0].vectors", "SIMULATION", "int")
    lat = {u["unit"]: u["latency_cycles"] for u in fu}
    N("NumGlMulLatency", lat.get("gl_mul"), src, "field_units[gl_mul].latency_cycles", "SIMULATION", "int")
    N("NumExtMulLatency", lat.get("ext_mul"), src, "field_units[ext_mul].latency_cycles", "SIMULATION", "int")
    for L, tag in ((1, "One"), (2, "Two"), (4, "Four"), (8, "Eight")):
        cyc = C.sim_cycles(sim, 16, L)
        N(f"NumRtlCyclesL{tag}", cyc, src, f"cases[lanes={L}, n=16, auto].cycles_total", "SIMULATION", "int")
    c8 = C.sim_cycles(sim, 16, 8)
    N("NumRtlOverheadPctLEight", (c8 - 1.5 * 2 ** 16 / 8) / c8 if c8 else None, src,
      "(cycles_total(L=8,n=16) - 1.5 N/L) / cycles_total", "SIMULATION", "pct", nd=1)
    gen = next((c["cycles_gen"] for c in cases if c.get("n") == 16 and c.get("src") == "gen"), None)
    N("NumRtlGenCycles", gen, src, "cases[n=16, src=gen].cycles_gen", "SIMULATION", "int")
    N("NumRtlResidentMiB", 2 * 2 ** 16 * 128 / 8 / 2 ** 20, "spec/sumfold_kernel.md sec. 3.1", "2 N_max x 128 bit, N_max = 2^16",
      "SUCCESS", nd=0)
    srco = "results/raw/fpga_sim/ooc_units.json"
    ooc = C.load_json(srco)
    units = {u["config"]: u for u in (ooc or {}).get("units", [])}
    dev = {"dsp": 2520, "lut": 274080, "bram": 912}
    gm = units.get("gl_mul")
    N("NumOocGlMulDsp", g(gm, "utilization_report.dsps"), srco, "units[gl_mul].utilization_report.dsps", "POST_SYNTH", "int")
    ln = next((u for k, u in units.items() if k.startswith("sumfold_lane") and "split" not in k), None)
    N("NumOocLaneDsp", g(ln, "utilization_report.dsps"), srco, "units[sumfold_lane*].utilization_report.dsps",
      "POST_SYNTH", "int")
    c8u = next((u for k, u in units.items() if "core" in k and ("L8" in k or "8" in k.split("_")[-1])), None)
    if c8u is None:
        c8u = next((u for u in units.values() if u.get("top") == "sumfold_core" and "8" in str(u.get("tcl_args"))), None)
    ur = g(c8u, "utilization_report") or {}
    N("NumOocCoreEightDsp", ur.get("dsps"), srco, "units[core L=8].utilization_report.dsps", "POST_SYNTH", "int")
    N("NumOocCoreEightDspPct", ur["dsps"] / dev["dsp"] if ur.get("dsps") else None, srco,
      "dsps / 2520 (XCZU9EG)", "POST_SYNTH", "pct", nd=1)
    N("NumOocCoreEightLutPct", ur["clb_luts"] / dev["lut"] if ur.get("clb_luts") else None, srco,
      "clb_luts / 274080", "POST_SYNTH", "pct", nd=1)
    N("NumOocCoreEightBramPct", ur["bram_tiles"] / dev["bram"] if ur.get("bram_tiles") else None, srco,
      "bram_tiles / 912", "POST_SYNTH", "pct", nd=1)
    N("NumOocCoreEightWns", g(c8u, "wns_ns"), srco, "units[core L=8].wns_ns at 3.333 ns", "POST_SYNTH", nd=3)
    # post-route implementation runs (POST_ROUTE): v2 RTL only (v1 is functionally incorrect on the board)
    srcr = "results/raw/fpga_impl/runs.json"
    runs = C.load_json(srcr)
    v2 = C.impl_runs(runs, "v2")
    v1 = C.impl_runs(runs, "v1")
    met = [r for r in v2.values() if r.get("status") == "POST_ROUTE_TIMING_MET"]
    ok = [r for r in met if r.get("board_status") == "SUCCESS_SUCCESS"] or met
    best = max(ok, key=lambda r: (r["lanes"] * r["target_mhz"], C.get(r, "timing.design.wns_ns", 0))) if ok else None
    N("NumFpgaRoutedRuns", len([r for r in v2.values() if r.get("evidence_class") == "POST_ROUTE"]) if runs else None,
      srcr, "count(v2 runs with evidence_class == POST_ROUTE)", "POST_ROUTE", "int")
    N("NumFpgaLanes", best["lanes"] if best else None, srcr, "best v2 timing-met, board-verified run .lanes", "POST_ROUTE", "int")
    N("NumFpgaClockMHz", best["target_mhz"] if best else None, srcr, "best v2 run .target_mhz", "POST_ROUTE", "int")
    N("NumFpgaWns", g(best, "timing.design.wns_ns"), srcr, "best v2 run .timing.design.wns_ns", "POST_ROUTE", nd=3)
    for k, tag in (("dsp", "Dsp"), ("lut", "Lut"), ("bram36_tiles", "Bram"), ("ff", "Ff")):
        N(f"NumFpga{tag}Pct", g(best, f"utilization.{k}.util_pct"), srcr, f"best v2 run .utilization.{k}.util_pct",
          "POST_ROUTE", nd=1)
    N("NumFpgaFmaxEst", g(best, "fmax_SUCCESS_mhz"), srcr, "best v2 run .fmax_SUCCESS_mhz (1000/(T-WNS))", "SUCCESS", nd=0)
    N("NumFpgaPowerW", g(best, "power.total_w"), srcr, "best v2 run .power.total_w (vectorless)", "SUCCESS", nd=2)
    srcb = "results/raw/fpga_board/summary.json"
    board, camp = C.load_board()
    bauto = {(c["lanes"], c["kclk_mhz"], int(c["n"])): c for c in C.board_configs(board, "auto")}
    bb = bauto.get((best["lanes"], best["target_mhz"], 16)) if best else None
    N("NumFpgaKernelUs", g(bb, "device_us_at_nominal_clock"), srcb,
      "configs[best L, MHz, auto, n=16].device_us_at_nominal_clock (on-chip cycles / nominal clock)", "SUCCESS", nd=1)
    # per v2 configuration (L = 1, 2, 4, 8 at 250 MHz; L = 8 at 300 MHz)
    wtag = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
    for (L, mhz), r in sorted(v2.items()):
        if r.get("status") != "POST_ROUTE_TIMING_MET":
            continue
        t = f"L{wtag.get(L, str(L))}" + ("ThreeHundred" if mhz == 300 else "TwoFifty")
        N(f"NumFpgaDspPct{t}", g(r, "utilization.dsp.util_pct"), srcr, f"v2 {r['run']} utilization.dsp.util_pct",
          "POST_ROUTE", nd=1)
        N(f"NumFpgaLutPct{t}", g(r, "utilization.lut.util_pct"), srcr, f"v2 {r['run']} utilization.lut.util_pct",
          "POST_ROUTE", nd=1)
        N(f"NumFpgaBramPct{t}", g(r, "utilization.bram36_tiles.util_pct"), srcr,
          f"v2 {r['run']} utilization.bram36_tiles.util_pct", "POST_ROUTE", nd=1)
        N(f"NumFpgaWns{t}", g(r, "timing.design.wns_ns"), srcr, f"v2 {r['run']} timing.design.wns_ns", "POST_ROUTE", nd=3)
        N(f"NumFpgaFmaxEst{t}", r.get("fmax_SUCCESS_mhz"), srcr, f"v2 {r['run']} fmax_SUCCESS_mhz", "SUCCESS", nd=0)
        N(f"NumFpgaPowerW{t}", g(r, "power.total_w"), srcr, f"v2 {r['run']} power.total_w (vectorless)", "SUCCESS", nd=2)
    wns = [g(r, "timing.design.wns_ns") for r in met if g(r, "timing.design.wns_ns") is not None]
    N("NumFpgaWnsMin", min(wns) if wns else None, srcr, "min WNS over v2 timing-met runs", "POST_ROUTE", nd=3)
    N("NumFpgaWnsMax", max(wns) if wns else None, srcr, "max WNS over v2 timing-met runs", "POST_ROUTE", nd=3)
    # v1 vs v2 multiplier (DSP48E2 per GL multiplier from the RTL record; DSP use POST_ROUTE)
    d1 = next((C.dsp_per_multiplier(r) for r in v1.values() if C.dsp_per_multiplier(r)), None)
    d2 = next((C.dsp_per_multiplier(r) for r in v2.values() if C.dsp_per_multiplier(r)), None)
    N("NumDspPerMulVOne", d1, srcr, "v1 runs rtl.gl_mul ('<k>-DSP48E2')", "POST_ROUTE", "int")
    N("NumDspPerMulVTwo", d2, srcr, "v2 runs rtl.gl_mul ('<k>-DSP48E2')", "POST_ROUTE", "int")
    a8, b8 = v1.get((8, 250)), v2.get((8, 250))
    N("NumFpgaDspPctLEightVOne", g(a8, "utilization.dsp.util_pct"), srcr, "v1 L8_250MHz utilization.dsp.util_pct",
      "POST_ROUTE", nd=1)
    N("NumFpgaDspUsedLEightVOne", g(a8, "utilization.dsp.used"), srcr, "v1 L8_250MHz utilization.dsp.used", "POST_ROUTE", "int")
    N("NumFpgaDspUsedLEightVTwo", g(b8, "utilization.dsp.used"), srcr, "v2 L8_250MHz_v2 utilization.dsp.used", "POST_ROUTE", "int")
    v1b = C.v1_board_outcomes(runs) if runs else {}
    N("NumFpgaVOneRuns", len([k for k, v in v1b.items() if v == "VERIFY_FAILED"]) if runs else None,
      "results/raw/fpga_board/{bringup_L8_250_auto_n10,v1_*}.json",
      "count routed v1 runs whose bitstream has a board record with status VERIFY_FAILED", "SUCCESS", "int")
    N("NumFpgaVOneNotRun", len([k for k, v in v1b.items() if v == "ON_BOARD"]) if runs else None,
      "results/raw/fpga_board/*.json + results/raw/fpga_impl/runs.json",
      "count routed v1 runs without any board record", "SUCCESS", "int")

# ============================================================ FPGA on-board campaign (SUCCESS)
def board():
    src = "results/raw/fpga_board/summary.json"
    srcc = "results/raw/fpga_board/campaign.json"
    s, camp = C.load_board()
    corr = g(s, "correctness") or {}
    N("NumBoardJobs", corr.get("test_runs"), src, "correctness.test_runs", "SUCCESS", "int")
    N("NumBoardJobsMatch", corr.get("match"), src, "correctness.match (bit-exact vs golden)", "SUCCESS", "int")
    N("NumBoardJobsVerified", corr.get("transcript_verified"), src, "correctness.transcript_verified", "SUCCESS", "int")
    N("NumBoardSteps", g(camp, "steps_recorded"), srcc, "steps_recorded", "SUCCESS", "int")
    auto = {(c["lanes"], c["kclk_mhz"], int(c["n"])): c for c in C.board_configs(s, "auto")}
    step = {(c["lanes"], c["kclk_mhz"], int(c["n"])): c for c in C.board_configs(s, "step")}
    lt = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
    nt = {10: "Ten", 12: "Twelve", 14: "Fourteen", 16: "Sixteen"}
    for (L, mhz, n), c in sorted(auto.items()):
        t = f"L{lt[L]}{'ThreeHundred' if mhz == 300 else 'TwoFifty'}N{nt.get(n, str(n))}"
        N(f"NumBoardCycles{t}", c["cyc_total"], src, f"configs[L{L}_{mhz} auto n={n}].cyc_total", "SUCCESS", "int")
        N(f"NumBoardDeviceUs{t}", c["device_us_at_nominal_clock"], src,
          f"configs[L{L}_{mhz} auto n={n}].device_us_at_nominal_clock", "SUCCESS", nd=2 if c["device_us_at_nominal_clock"] < 100 else 1)
        N(f"NumBoardHostJobMs{t}", (g(c, "job_host_us.median") or 0) / 1e3 if g(c, "job_host_us.median") else None, src,
          f"configs[L{L}_{mhz} auto n={n}].job_host_us.median / 1e3", "SUCCESS", nd=1)
    c8 = auto.get((8, 300, 16))
    N("NumBoardNsPerEntryBest", c8["device_us_at_nominal_clock"] * 1e3 / 2 ** 16 if c8 else None, src,
      "L8_300 auto n=16 device_us x 1e3 / 2^16", "SUCCESS", nd=2)
    N("NumBoardGenCyclesSixteen", g(c8, "cyc_gen"), src, "L8_300 auto n=16 cyc_gen", "SUCCESS", "int")
    N("NumBoardGenUsSixteen", g(c8, "cyc_gen") / 300 if c8 else None, src, "cyc_gen / 300 MHz", "SUCCESS", nd=0)
    hj = [g(c, "job_host_us.median") for c in auto.values() if g(c, "job_host_us.median")]
    hj_ok = sorted(hj)
    N("NumBoardHostJobMsMin", hj_ok[0] / 1e3 if hj_ok else None, src, "min over AUTO configs of job_host_us.median", "SUCCESS", nd=1)
    N("NumBoardHostJobMsMax", hj_ok[-1] / 1e3 if hj_ok else None, src, "max over AUTO configs of job_host_us.median", "SUCCESS", nd=1)
    for (L, mhz, n), c in sorted(step.items()):
        t = f"N{nt.get(n, str(n))}"
        N(f"NumStepRoundMs{t}", g(c, "step_round_trip_us.median") / 1e3 if g(c, "step_round_trip_us.median") else None, src,
          f"configs[L{L}_{mhz} step n={n}].step_round_trip_us.median / 1e3", "SUCCESS", nd=1)
        N(f"NumStepRoundMinMs{t}", g(c, "step_round_trip_us.min") / 1e3 if g(c, "step_round_trip_us.min") else None, src,
          f"... step_round_trip_us.min / 1e3", "SUCCESS", nd=1)
        N(f"NumStepRoundMaxMs{t}", g(c, "step_round_trip_us.max") / 1e3 if g(c, "step_round_trip_us.max") else None, src,
          f"... step_round_trip_us.max / 1e3", "SUCCESS", nd=1)
        N(f"NumStepServiceMs{t}", g(c, "step_round_service_us.median") / 1e3 if g(c, "step_round_service_us.median") else None,
          src, "... step_round_service_us.median / 1e3", "SUCCESS", nd=1)
        N(f"NumStepPollMs{t}", g(c, "step_round_poll_us.median") / 1e3 if g(c, "step_round_poll_us.median") else None,
          src, "... step_round_poll_us.median / 1e3", "SUCCESS", nd=2)
        N(f"NumStepJobMs{t}", g(c, "job_host_us.median") / 1e3 if g(c, "job_host_us.median") else None, src,
          "... job_host_us.median / 1e3", "SUCCESS", nd=1)
        N(f"NumStepRounds{t}", g(c, "step_round_trip_us.n") // max(1, c.get("runs", 1)) if g(c, "step_round_trip_us.n") else None,
          src, "step_round_trip_us.n / runs (device rounds per job)", "SUCCESS", "int")
    prim, lruns = C.board_link(s)
    lat = [g(prim, f"{m}.us_per_op") for m in C.LINK_METHODS if g(prim, f"{m}.us_per_op")] if prim else []
    N("NumJtagTxnMsMin", min(lat) / 1e3 if lat else None, src, "link_rate primary: min us_per_op over 6 methods / 1e3",
      "SUCCESS", nd=2)
    N("NumJtagTxnMsMax", max(lat) / 1e3 if lat else None, src, "link_rate primary: max us_per_op / 1e3", "SUCCESS", nd=2)
    N("NumJtagTxnPerSMin", 1e6 / max(lat) if lat else None, src, "1e6 / max us_per_op", "SUCCESS", nd=0)
    N("NumJtagTxnPerSMax", 1e6 / min(lat) if lat else None, src, "1e6 / min us_per_op", "SUCCESS", nd=0)
    N("NumJtagMemElemMs", g(prim, "mem_element_write.us_per_op") / 1e3 if g(prim, "mem_element_write.us_per_op") else None,
      src, "link_rate primary mem_element_write.us_per_op / 1e3 (one 128-bit element, 7 writes)", "SUCCESS", nd=1)
    N("NumJtagHostLoadPct", g(prim, "contention_flags.host_cpu_load_pct"), src,
      "link_rate primary contention_flags.host_cpu_load_pct", "SUCCESS", nd=0)
    busy = next((r for r in lruns if r is not prim), None)
    blat = [g(busy, f"{m}.us_per_op") for m in C.LINK_METHODS if g(busy, f"{m}.us_per_op")] if busy else []
    N("NumJtagTxnMsBusyMax", max(blat) / 1e3 if blat else None, src, "link_rate loaded run: max us_per_op / 1e3",
      "SUCCESS", nd=1)
    tle = g(prim, "table_load_extrapolation") or {}
    # load from the per-write rate (no table was loaded over JTAG)
    N("NumTableLoadFullS", tle.get("full_128bit_elements_s"), src,
      "table_load_extrapolation.full_128bit_elements_s (extrapolated from the SUCCESS write rate)", "SUCCESS", nd=0)
    N("NumTableLoadMinS", tle.get("base_field_min_s"), src,
      "table_load_extrapolation.base_field_min_s (extrapolated)", "SUCCESS", nd=0)
    N("NumTableLoadQueuedS", tle.get("base_field_min_queued_s"), src,
      "table_load_extrapolation.base_field_min_queued_s (extrapolated)", "SUCCESS", nd=0)
    sm = C.sysmon_values(camp)
    for q, tag, nd in (("TEMPERATURE", "Temp", 1), ("VCCINT", "Vccint", 3), ("VCCAUX", "Vccaux", 3), ("VCCBRAM", "Vccbram", 3)):
        v = sm.get(q) or []
        N(f"NumSysmon{tag}Min", min(v) if v else None, srcc, f"min over SYSMON samples ({q})", "SUCCESS", nd=nd)
        N(f"NumSysmon{tag}Max", max(v) if v else None, srcc, f"max over SYSMON samples ({q})", "SUCCESS", nd=nd)
    N("NumSysmonSamples", len(sm.get("TEMPERATURE") or []) if camp else None, srcc, "count SYSMON temperature samples",
      "SUCCESS", "int")
    lp = next((l_ for l_ in (g(s, "sysmon_loops") or []) if l_.get("step") == "L8_300_sysmon_n16"), None)
    N("NumSysmonLoopJobs", lp.get("jobs") if lp else None, src, "sysmon_loops[L8_300_sysmon_n16].jobs", "SUCCESS", "int")
    N("NumSysmonLoopS", lp.get("loop_s") if lp else None, src, "sysmon_loops[L8_300_sysmon_n16].loop_s", "SUCCESS", nd=1)
    N("NumSysmonBusyPct", lp.get("kernel_busy_fraction") if lp else None, src,
      "sysmon_loops[L8_300_sysmon_n16].kernel_busy_fraction", "SUCCESS", "pct", nd=0)
    neg = g(s, "negative_tests") or []
    N("NumBoardNegTests", len(neg) if s else None, src, "len(negative_tests)", "SUCCESS", "int")
    N("NumBoardNegTestsPass", sum(1 for t in neg if t.get("pass")) if s else None, src, "count negative_tests[].pass",
      "SUCCESS", "int")
    loads = [g(r, "contention_flags.host_cpu_load_pct") for r in (g(camp, "runs") or [])
             if g(r, "contention_flags.host_cpu_load_pct") is not None]
    N("NumBoardHostLoadMaxPct", max(loads) if loads else None, srcc, "max runs[].contention_flags.host_cpu_load_pct",
      "SUCCESS", nd=0)
    tm = C.load_json("results/raw/fpga_impl/throughput_model.json")
    r8 = next((r for r in (tm or {}).get("rows", []) if r.get("lanes") == 8 and r.get("clock_mhz") == 300), None)
    N("NumFpgaEnergyUjEst", g(r8, "energy_per_sumcheck_uj"), "results/raw/fpga_impl/throughput_model.json",
      "rows[L8, 300].energy_per_sumcheck (vectorless power x kernel time)", "SUCCESS", nd=0)

# ============================================================ kernels, final idle-host extras
def kernels_final():
    src = "results/raw/kernels/cpu_bench.json"
    src2 = "results/raw/kernels/cpu_bench_final_rep2.json"
    a, b = C.load_json(src), C.load_json(src2)
    idle, probes = C.idle_verified("kernels")
    N("NumKernelIdleProbes", len(probes) if probes else None, "results/raw/kernels/idle_probes/*.json",
      "count of idle probes (all idle)" if idle else "count of idle probes (NOT all idle)", "SUCCESS", "int")
    fa, fb = C.cpu_records(a, "gcc", "fused"), C.cpu_records(b, "gcc", "fused")
    r = fb.get((24, 1))
    N("NumCpuOneThreadMsRunB", r["total_stats_s"]["median"] * 1e3 if r else None, src2,
      "results[gcc,fused,n=24,threads=1].total_stats_s.median x1e3 (run B)", "SUCCESS", nd=1)
    for th, tag in ((8, "Eight"), (24, "TwentyFour"), (32, "ThirtyTwo")):
        vals = [x[(24, th)]["total_stats_s"]["median"] * 1e3 for x in (fa, fb) if (24, th) in x]
        N(f"NumCpu{tag}ThrMsMin", min(vals) if vals else None, f"{src} + {src2}",
          f"min over runs A/B of gcc fused n=24 {th} thr. median (ms)", "SUCCESS", nd=1)
        N(f"NumCpu{tag}ThrMsMax", max(vals) if vals else None, f"{src} + {src2}",
          f"max over runs A/B of gcc fused n=24 {th} thr. median (ms)", "SUCCESS", nd=1)
    multi = []
    for comp in ("gcc", "msvc"):
        for x in (a, b):
            fr = C.cpu_records(x, comp, "fused")
            multi += [fr[(24, t)]["total_stats_s"]["median"] * 1e3 for t in (24, 32) if (24, t) in fr]
    N("NumCpuMultiMsMin", min(multi) if multi else None, f"{src} + {src2}",
      "min over gcc/MSVC, runs A/B, 24/32 thr. of fused n=24 median (ms)", "SUCCESS", nd=1)
    N("NumCpuMultiMsMax", max(multi) if multi else None, f"{src} + {src2}",
      "max over gcc/MSVC, runs A/B, 24/32 thr. of fused n=24 median (ms)", "SUCCESS", nd=1)
    # best of 8/24/32 threads per run (gcc, fused), n=24: the definition of the multi-thread row of tab:kernels
    mb = []
    for x in (fa, fb):
        cand = [x[(24, t)]["total_stats_s"]["median"] * 1e3 for t in (8, 24, 32) if (24, t) in x]
        if cand:
            mb.append(min(cand))
    N("NumCpuMultiBestMsMin", min(mb) if mb else None, f"{src} + {src2}",
      "min over runs A/B of (best of 8/24/32 thr., gcc fused, n=24) median (ms)", "SUCCESS", nd=1)
    N("NumCpuMultiBestMsMax", max(mb) if mb else None, f"{src} + {src2}",
      "max over runs A/B of (best of 8/24/32 thr., gcc fused, n=24) median (ms)", "SUCCESS", nd=1)
    # n=16: best gcc fused configuration (any thread count) and best of all compilers/variants/threads, per run
    for tag, combos in (("Gcc", (("gcc", "fused"),)),
                        ("Any", (("gcc", "fused"), ("gcc", "unfused"), ("msvc", "fused"), ("msvc", "unfused")))):
        per_run = []
        for x in (a, b):
            cand = []
            for comp, var in combos:
                fr = C.cpu_records(x, comp, var)
                cand += [r["total_stats_s"]["median"] * 1e3 for (n, _t), r in fr.items() if n == 16]
            if cand:
                per_run.append(min(cand))
        what = "gcc fused, any thread count" if tag == "Gcc" else "any compiler, variant and thread count"
        N(f"NumCpuBest{tag}MsSixteenMin", min(per_run) if per_run else None, f"{src} + {src2}",
          f"min over runs A/B of best ({what}) n=16 total median (ms)", "SUCCESS", nd=2)
        N(f"NumCpuBest{tag}MsSixteenMax", max(per_run) if per_run else None, f"{src} + {src2}",
          f"max over runs A/B of best ({what}) n=16 total median (ms)", "SUCCESS", nd=2)
    rats = []
    for x in (a, b):
        fr, ur = C.cpu_records(x, "gcc", "fused"), C.cpu_records(x, "gcc", "unfused")
        rats += [ur[(24, t)]["total_stats_s"]["median"] / fr[(24, t)]["total_stats_s"]["median"]
                 for t in (24, 32) if (24, t) in fr and (24, t) in ur]
    N("NumCpuFusionGainMultiMin", min(rats) if rats else None, f"{src} + {src2}",
      "min unfused/fused gcc n=24 at 24/32 thr. over runs A/B", "SUCCESS", nd=2)
    N("NumCpuFusionGainMultiMax", max(rats) if rats else None, f"{src} + {src2}",
      "max unfused/fused gcc n=24 at 24/32 thr. over runs A/B", "SUCCESS", nd=2)
    srcg = "results/raw/kernels/gpu_bench.json"
    gpu = C.load_json(srcg)
    for d in (gpu or {}).get("devices", []):
        nm = C.get(d, "device.name", "")
        tag = "Ada" if "6000" in nm else "FourNinety" if "4090" in nm else None
        if tag != "Ada":
            continue
        mb = next((m for m in d.get("pass_microbench", []) if m.get("n") == 24), None)
        N("NumGpuAdaPassFusionGain", mb.get("speedup_unfused_over_fused_at_min") if mb else None, srcg,
          "devices[RTX 6000 Ada].pass_microbench[n=24].speedup_unfused_over_fused_at_min", "SUCCESS", nd=2)
    srcp = "results/raw/kernels/gpu_power_final.json"
    pw = C.load_json(srcp)
    got = {}
    for d in (pw or {}).get("devices", []):
        nm = d.get("name", "")
        tag = "Ada" if "6000" in nm else "FourNinety" if "4090" in nm else None
        row = next((x for x in d.get("rows", []) if x.get("n") == 24 and x.get("variant") == "fused"), None)
        if tag:
            got[tag] = (row, d.get("idle_W"))
    for tag in ("FourNinety", "Ada"):
        row, idle_w = got.get(tag, (None, None))
        N(f"NumGpu{tag}EnergyJ", g(row, "net_energy_per_proof_J"), srcp,
          f"devices[{tag}].rows[n=24,fused].net_energy_per_proof_J (GPU board power minus idle)", "SUCCESS", nd=2)
        N(f"NumGpu{tag}PowerW", g(row, "mean_W"), srcp, f"devices[{tag}].rows[n=24,fused].mean_W", "SUCCESS", nd=0)
        N(f"NumGpu{tag}IdleW", idle_w, srcp, f"devices[{tag}].idle_W", "SUCCESS", nd=1)

# ============================================================ GPT-2 transparent linear-layer prover (SUCCESS)
GPT2_BE = {"cpu1": "CpuOne", "cpu32": "CpuThirtyTwo", "gpu0": "GpuZero", "gpu1": "GpuOne"}

def _be_tag(be, r):
    """Macro tag of a GPT-2 backend: GPU tags follow the recorded device name (never the CuPy index)."""
    dev = str((r or {}).get("device", ""))
    if be.startswith("gpu"):
        return "FourNinety" if "4090" in dev else "Ada" if "6000" in dev else GPT2_BE[be]
    return GPT2_BE[be]

def gpt2prove():
    data, files = C.load_gpt2prove()
    src = files[0] if files else "results/raw/gpt2prove/summary_final.json"
    tt = {32: "ThirtyTwo", 128: "OneTwentyEight", 512: "FiveTwelve"}
    for T, tag in tt.items():
        for be in GPT2_BE:
            r = (data.get(T) or {}).get(be)
            bt = _be_tag(be, r)
            N(f"NumGptProve{bt}T{tag}S", r["wall"] if r else None, src,
              f"rows[{T}].configs.{be}.prover_wall_s_median ({r['device'] if r else be})", "SUCCESS", nd=3)
        for be in ("cpu1", "cpu32"):
            r = (data.get(T) or {}).get(be)
            N(f"NumGptVerify{GPT2_BE[be]}T{tag}S", r["verify"] if r else None, src,
              f"rows[{T}].configs.{be}.verifier_wall_s_median", "SUCCESS", nd=3)
        r = (data.get(T) or {}).get("cpu1")
        N(f"NumGptProofKBT{tag}", r["proof_bytes"] / 1e3 if r and r["proof_bytes"] else None, src,
          f"rows[{T}].configs.cpu1.proof_bytes_total / 1e3", "SUCCESS", nd=1)
        rg = (data.get(T) or {}).get("gpu0")
        N(f"NumGptPerTokenMs{_be_tag('gpu0', rg)}T{tag}", rg["wall_per_token"] * 1e3 if rg and rg.get("wall_per_token") else None, src,
          f"rows[{T}].configs.gpu0.wall_per_token_s x1e3", "SUCCESS", nd=2)
        N(f"NumGptBindShare{_be_tag('gpu0', rg)}T{tag}", (rg["bind"] / rg["wall"]) if rg and rg.get("bind") and rg.get("wall") else None,
          src, f"rows[{T}].configs.gpu0 bind / wall", "SUCCESS", "pct", nd=0)
    # share of the prover wall time spent in the product sumcheck itself (the SumFold kernel), over T = 32/128/512
    sf_src = "results/raw/gpt2prove/summary_final.json"
    sf = C.load_json(sf_src)
    for grp, bes, tag in (("cpu1", ("cpu1",), "CpuOne"), ("cpu32", ("cpu32",), "CpuThirtyTwo"),
                          ("gpu", ("gpu0", "gpu1"), "Gpu")):
        sh = [c["prover_sumcheck_s_median"] / c["prover_wall_s_median"]
              for r in ((sf or {}).get("rows") or {}).values() for be, c in (r.get("configs") or {}).items()
              if be in bes and c.get("prover_sumcheck_s_median") and c.get("prover_wall_s_median")]
        N(f"NumGptSumcheckShare{tag}Min", min(sh) if sh else None, sf_src,
          f"min over T of rows[T].configs.{'/'.join(bes)}.prover_sumcheck_s_median / prover_wall_s_median",
          "SUCCESS", "pct", nd=1)
        N(f"NumGptSumcheckShare{tag}Max", max(sh) if sh else None, sf_src,
          f"max over T of rows[T].configs.{'/'.join(bes)}.prover_sumcheck_s_median / prover_wall_s_median",
          "SUCCESS", "pct", nd=1)
    gp = [c.get("gpu_resident_bytes", 0) + c.get("gpu_prover_scratch_peak_bytes", 0)
          for be, c in ((((sf or {}).get("rows") or {}).get("128") or {}).get("configs") or {}).items()
          if be.startswith("gpu") and c.get("gpu_resident_bytes")]
    N("NumGptGpuPoolGiBTOneTwentyEight", max(gp) / 2 ** 30 if gp else None, sf_src,
      "max over GPUs of rows[128].configs.gpu*.(gpu_resident_bytes + gpu_prover_scratch_peak_bytes) / 2^30",
      "SUCCESS", nd=2)
    tf = C.load_json("results/raw/gpt2prove/gpt2_prove_T128_final.json")
    N("NumGptProofs", C.get(tf, "configs.cpu1.counts.proofs") if tf else None, "results/raw/gpt2prove/gpt2_prove_T128_final.json",
      "configs.cpu1.counts.proofs", "SUCCESS", "int")
    ts = C.load_json("results/raw/gpt2prove/tests_final.json")
    N("NumGptTests", C.get(ts, "counts.tests"), "results/raw/gpt2prove/tests_final.json", "counts.tests", "SUCCESS", "int")
    N("NumGptTestsOk", C.get(ts, "counts.ok"), "results/raw/gpt2prove/tests_final.json", "counts.ok", "SUCCESS", "int")
    fe = C.load_json("results/raw/gpt2prove/fold_vs_eq_final.json")
    for th, tag in ((1, "OneThr"), (32, "ThirtyTwoThr")):
        rs = [r["ratio_fold_over_eq_median"] for r in (fe or {}).get("rows", []) if r.get("threads") == th]
        N(f"NumFoldEqRatio{tag}Min", min(rs) if rs else None, "results/raw/gpt2prove/fold_vs_eq_final.json",
          f"min rows[threads={th}].ratio_fold_over_eq_median", "SUCCESS", nd=1)
        N(f"NumFoldEqRatio{tag}Max", max(rs) if rs else None, "results/raw/gpt2prove/fold_vs_eq_final.json",
          f"max rows[threads={th}].ratio_fold_over_eq_median", "SUCCESS", nd=1)

# ============================================================ real models (GPT-2, Qwen3-0.6B)
def realmodels():
    rm, src, s = C.load_realmodels()
    src = src or "results/raw/realmodels/summary.json"
    for key, tag in (("gpt2_124m", "GptTwo"), ("qwen3_0.6b", "Qwen")):
        m = rm.get(key) or {}
        prim = g(m, "paper_primary.variant")
        v = g(m, f"variants.{prim}") if prim else None
        N(f"NumReal{tag}TopOne", g(v, "tf_top1_agreement"), src, f"models.{key}.variants.{prim}.tf_top1_agreement",
          "SUCCESS", nd=3)
        N(f"NumReal{tag}TopOneMargin", g(v, "tf_top1_agreement_margin_gt_1"), src,
          f"...{prim}.tf_top1_agreement_margin_gt_1", "SUCCESS", nd=3)
        N(f"NumReal{tag}PplFloat", g(v, "ppl_float"), src, f"...{prim}.ppl_float", "SUCCESS", nd=2)
        N(f"NumReal{tag}PplInt", g(v, "ppl_int"), src, f"...{prim}.ppl_int", "SUCCESS", nd=2)
        N(f"NumReal{tag}PplRelPct", g(v, "ppl_relative_change"), src, f"...{prim}.ppl_relative_change", "SUCCESS",
          "pct", nd=1)
        N(f"NumReal{tag}Greedy", g(v, "greedy_token_agreement_32"), src, f"...{prim}.greedy_token_agreement_32",
          "SUCCESS", "pct", nd=1)
        N(f"NumReal{tag}Violations", g(v, "dynamic_checks.violations"), src, f"...{prim}.dynamic_checks.violations",
          "SUCCESS", "int")
        N(f"NumReal{tag}ForwardRuns", g(v, "dynamic_checks.forward_runs"), src, f"...{prim}.dynamic_checks.forward_runs",
          "SUCCESS", "int")
        N(f"NumReal{tag}CheckEvals", g(v, "dynamic_checks.tensor_check_evaluations"), src,
          f"...{prim}.dynamic_checks.tensor_check_evaluations", "SUCCESS", "int")
        N(f"NumReal{tag}ReplayFailures", g(v, "contracts.relation_replay.failures"), src,
          f"...{prim}.contracts.relation_replay.failures", "SUCCESS", "int")
        N(f"NumReal{tag}ReplayRqElems", g(v, "contracts.relation_replay.counts.rq_elements"), src,
          f"...{prim}.contracts.relation_replay.counts.rq_elements", "SUCCESS", "int")
        N(f"NumReal{tag}HeadroomGl", g(v, "contracts.max_term.headroom_bits_goldilocks"), src,
          f"...{prim}.contracts.max_term.headroom_bits_goldilocks", "ANALYTIC", nd=2 if tag == "GptTwo" else 1)
        N(f"NumReal{tag}HeadroomBb", g(v, "contracts.max_term.headroom_bits_babybear"), src,
          f"...{prim}.contracts.max_term.headroom_bits_babybear", "ANALYTIC", nd=1)
        N(f"NumReal{tag}BbViolations", g(v, "contracts.babybear_violations"), src, f"...{prim}.contracts.babybear_violations",
          "ANALYTIC", "int")
        N(f"NumReal{tag}LookupReduction", g(v, "contracts.reduction_vs_naive_field64"), src,
          f"...{prim}.contracts.reduction_vs_naive_field64 (prefill 128)", "ANALYTIC", nd=2)
        N(f"NumReal{tag}CommittedVsPolicy", g(v, "contracts.committed_vs_policy_lookup_ratio"), src,
          f"...{prim}.contracts.committed_vs_policy_lookup_ratio", "ANALYTIC", nd=3)
        w4 = g(m, "variants.W4A8g128_sq_sink")
        N(f"NumReal{tag}WFourTopOne", g(w4, "tf_top1_agreement"), src, "...W4A8g128_sq_sink.tf_top1_agreement",
          "SUCCESS", nd=3)
        N(f"NumReal{tag}WFourPplRelPct", g(w4, "ppl_relative_change"), src, "...W4A8g128_sq_sink.ppl_relative_change",
          "SUCCESS", "pct", nd=0)
        nv = g(m, "variants.W8A8_spec_naive")
        N(f"NumReal{tag}NaiveTopOne", g(nv, "tf_top1_agreement"), src, "...W8A8_spec_naive.tf_top1_agreement",
          "SUCCESS", nd=4)
        N(f"NumReal{tag}Variants", len([x for x in (m.get("variants") or {}).values()
                                         if x.get("status") == "SUCCESS"]) if m else None, src,
          f"models.{key}.variants", "SUCCESS", "int")
    r20 = C.get(rm.get("gpt2_124m") or {}, "variants.W8A8_sq_sink_r20")
    N("NumRealGptTwoRTwentyTopOne", g(r20, "tf_top1_agreement"), src, "gpt2 W8A8_sq_sink_r20.tf_top1_agreement",
      "SUCCESS", nd=3)
    d = C.get(rm.get("qwen3_0.6b") or {}, "diagnostic_simulation_actquant_ablation") or {}
    res = d.get("results") or {}
    N("NumQwenSimWeightsOnly", res.get("weights_only"), src, "qwen3 diagnostic_simulation_actquant_ablation.results.weights_only",
      "SIMULATION", nd=3)
    N("NumQwenSimActsOnly", res.get("activations_only_R16"), src, "... results.activations_only_R16", "SIMULATION", nd=3)
    N("NumQwenSimAllQuant", res.get("all_quantized_R16"), src, "... results.all_quantized_R16", "SIMULATION", nd=3)
    N("NumQwenSimAllQuantRTwenty", res.get("all_quantized_R20"), src, "... results.all_quantized_R20", "SIMULATION", nd=3)
    N("NumQwenSimPrompts", d.get("prompts"), src, "... prompts", "SIMULATION", "int")
    loo = d.get("leave_one_out") or {}
    N("NumQwenSimLeaveOneOutMax", max(loo.values()) if loo else None, src, "max ... leave_one_out", "SIMULATION", nd=3)
    ev = C.get(C.load_json("results/raw/realmodels/gpt2_quality.json") or {}, "settings.n_eval")
    N("NumRealEvalSequences", ev, "results/raw/realmodels/gpt2_quality.json", "settings.n_eval", "SUCCESS", "int")

# ============================================================ scheduling (C3)
def sched():
    src = "results/raw/sched/summary.json"
    s = C.load_json(src)
    N("NumLegalityCases", g(s, "legality.cases"), src, "legality.cases", "SIMULATION", "int")
    N("NumMutationsRejected", len(g(s, "legality.mutations_rejected") or []) if s else None, src,
      "len(legality.mutations_rejected)", "SIMULATION", "int")
    N("NumMutationsAccepted", len(g(s, "legality.mutations_accepted") or []) if s else None, src,
      "len(legality.mutations_accepted)", "SIMULATION", "int")
    leg = C.load_csv("results/processed/sched_legality.csv")
    nv = None
    for r in leg or []:
        if r["model"] == "gpt2-124m" and r["protocol"] == "GLOBAL" and r["case"] == "LAYER_STREAM_NAIVE":
            import json as _j
            nv = sum(_j.loads(r["rules"]).values())
    N("NumNaiveStreamViolations", nv, "results/processed/sched_legality.csv",
      "rules of gpt2-124m GLOBAL LAYER_STREAM_NAIVE", "SIMULATION", "int")
    pm = g(s, "peak_memory_GiB.by_protocol_model_policy_ctx.GLOBAL") or {}
    for mk, tag in (("gpt2-124m", "GptTwo"), ("qwen3-0.6b", "Qwen"), ("gemma3-1b", "Gemma"), ("llama3-8b", "LlamaEight")):
        for ctx, ct in (("32", "ThirtyTwo"), ("512", "FiveTwelve"), ("2048", "TwoK")):
            N(f"NumKeepAll{tag}{ct}GiB", g(pm, f"{mk}.KEEP_ALL.{ctx}"), src,
              f"peak_memory_GiB.by_protocol_model_policy_ctx.GLOBAL.{mk}.KEEP_ALL.{ctx}", "ANALYTIC",
              nd=1 if (g(pm, f"{mk}.KEEP_ALL.{ctx}") or 0) >= 10 else 2)
        N(f"NumNaiveStream{tag}ThirtyTwoGiB", g(pm, f"{mk}.LAYER_STREAM_NAIVE.32"), src,
          f"...GLOBAL.{mk}.LAYER_STREAM_NAIVE.32 (illegal policy)", "ANALYTIC", nd=2)
    rows = C.load_csv("results/processed/sched_memory_vs_ctx.csv")
    q = next((r for r in rows or [] if r["protocol"] == "GLOBAL" and r["model"] == "qwen3-0.6b" and r["ctx"] == "512"
              and r["policy"] == "CERTSCHED" and C.fnum(r["cap_GiB"]) == 8), None)
    srcm = "results/processed/sched_memory_vs_ctx.csv"
    N("NumQwenFiveTwelveWitnessGiB", C.fnum(q["witness_GiB"]) if q else None, srcm,
      "GLOBAL qwen3-0.6b ctx 512 CERTSCHED cap 8: witness_GiB", "ANALYTIC", nd=1)
    N("NumQwenFiveTwelveCertSchedPeakGiB", C.fnum(q["peak_effective_GiB"]) if q else None, srcm,
      "... peak_effective_GiB", "ANALYTIC", nd=1)
    N("NumQwenFiveTwelveCertSchedAddedS", C.fnum(q["added_s_est"]) if q else None, srcm, "... added_s_est", "SUCCESS", nd=1)
    # the best uniform policy that also fits 8 GiB (RECOMPUTE): CertSched's benefit is time, not feasibility
    qr = next((r for r in rows or [] if r["protocol"] == "GLOBAL" and r["model"] == "qwen3-0.6b" and r["ctx"] == "512"
               and r["policy"] == "RECOMPUTE" and C.fnum(r["cap_GiB"]) == 8), None)
    N("NumQwenFiveTwelveRecomputePeakGiB", C.fnum(qr["peak_effective_GiB"]) if qr else None, srcm,
      "GLOBAL qwen3-0.6b ctx 512 RECOMPUTE cap 8: peak_effective_GiB", "ANALYTIC", nd=2)
    N("NumQwenFiveTwelveRecomputeAddedS", C.fnum(qr["added_s_est"]) if qr else None, srcm,
      "GLOBAL qwen3-0.6b ctx 512 RECOMPUTE cap 8: added_s_est", "SUCCESS", nd=1)
    cu = g(s, "certsched_vs_uniform") or {}
    N("NumCertSchedCells", cu.get("cells"), src, "certsched_vs_uniform.cells", "SIMULATION", "int")
    N("NumCertSchedFits", cu.get("certsched_fits"), src, "certsched_vs_uniform.certsched_fits", "SIMULATION", "int")
    N("NumUniformFits", cu.get("some_uniform_fits"), src, "certsched_vs_uniform.some_uniform_fits", "SIMULATION", "int")
    N("NumOnlyCertSchedFits", cu.get("only_certsched_fits"), src, "certsched_vs_uniform.only_certsched_fits",
      "SIMULATION", "int")
    ar = cu.get("added_time_ratio_vs_best_uniform_est") or {}
    N("NumCertSchedRatioCells", ar.get("n"), src, "certsched_vs_uniform.added_time_ratio_vs_best_uniform_est.n",
      "SIMULATION", "int")
    N("NumCertSchedRatioMedian", ar.get("median"), src, "...added_time_ratio_vs_best_uniform_est.median", "SUCCESS", nd=2)
    N("NumCertSchedRatioMin", ar.get("min"), src, "...added_time_ratio_vs_best_uniform_est.min", "SUCCESS", nd=2)
    og = g(s, "optimality_gap") or {}
    N("NumGapSyntheticMeanPct", og.get("synthetic_mean_gap"), src, "optimality_gap.synthetic_mean_gap", "SIMULATION",
      "pct", nd=2)
    N("NumGapSyntheticMaxPct", og.get("synthetic_max_gap"), src, "optimality_gap.synthetic_max_gap", "SIMULATION",
      "pct", nd=1)
    N("NumGapSyntheticMinOptimalPct", og.get("synthetic_min_frac_optimal"), src,
      "optimality_gap.synthetic_min_frac_optimal", "SIMULATION", "pct", nd=1)
    N("NumGapRatioOnlyMaxPct", og.get("synthetic_ratio_only_max_gap"), src, "optimality_gap.synthetic_ratio_only_max_gap",
      "SIMULATION", "pct", nd=0)
    ms = og.get("model_small_gap_est") or {}
    N("NumGapModelMeanPct", ms.get("mean"), src, "optimality_gap.model_small_gap_est.mean", "SUCCESS", "pct", nd=2)
    N("NumGapModelMaxPct", ms.get("max"), src, "optimality_gap.model_small_gap_est.max", "SUCCESS", "pct", nd=1)
    N("NumGapModelInstances", ms.get("n"), src, "optimality_gap.model_small_gap_est.n", "SIMULATION", "int")
    lp = og.get("model_full_gap_vs_lp_upper_bound_est") or {}
    N("NumGapLpMax", max(lp.values()) if lp else None, src, "max optimality_gap.model_full_gap_vs_lp_upper_bound_est",
      "SUCCESS", "sci", nd=1)
    # DES
    des = g(s, "des") or {}
    snap = des.get("snapshots") or {}
    N("NumDesDebtBound", des.get("debt_bound_B"), src, "des.debt_bound_B", "SIMULATION", "int")
    N("NumDesMaxDebtCertSchedFull", des.get("certsched_full_max_debt_over_all_configs"), src,
      "des.certsched_full_max_debt_over_all_configs", "SIMULATION", "int")
    for var, tag in (("KEEP_ALL", "KeepAll"), ("RECOMPUTE", "Recompute"), ("SPILL", "Spill"), ("HOST_STAGE", "HostStage"),
                     ("CERTSCHED_MEM", "CertSchedMem"), ("CERTSCHED_FULL", "CertSchedFull")):
        sn = snap.get(f"ws_4090/qwen3-0.6b/rho0.7/{var}") or {}
        N(f"NumDesQwen{tag}CertPNinetyFive", sn.get("cert_p95_est"), src,
          f"des.snapshots['ws_4090/qwen3-0.6b/rho0.7/{var}'].cert_p95_est", "SUCCESS", nd=0 if (sn.get("cert_p95_est") or 0) >= 100 else 1)
        N(f"NumDesQwen{tag}AbortPct", sn.get("abort_rate"), src,
          f"des.snapshots['ws_4090/qwen3-0.6b/rho0.7/{var}'].abort_rate", "SIMULATION", "pct", nd=0)
        N(f"NumDesQwen{tag}MaxDebt", sn.get("max_debt_tokens"), src,
          f"des.snapshots['ws_4090/qwen3-0.6b/rho0.7/{var}'].max_debt_tokens", "SIMULATION", "int")
    sn = snap.get("ws_4090/qwen3-0.6b/rho0.7/CERTSCHED_FULL") or {}
    N("NumDesQwenCertSchedFullProvPNinetyFive", sn.get("prov_p95_est"), src,
      "des.snapshots['ws_4090/qwen3-0.6b/rho0.7/CERTSCHED_FULL'].prov_p95_est", "SUCCESS", nd=1)
    lsw = C.load_json("results/raw/sched/des_load_sweep.json")
    N("NumDesSeeds", len(g(lsw, "design.seeds") or []) if lsw else None, "results/raw/sched/des_load_sweep.json",
      "len(design.seeds)", "SIMULATION", "int")
    N("NumDesRequestsPerRun", g(lsw, "design.requests_per_run"), "results/raw/sched/des_load_sweep.json",
      "design.requests_per_run", "SIMULATION", "int")
    ov = g(s, "overload") or {}
    nb = ov.get("ws_4090/qwen3-0.6b/CERTSCHED_noBP") or {}
    b128 = ov.get("ws_4090/qwen3-0.6b/CERTSCHED_B128") or {}
    b64 = ov.get("ws_4090/qwen3-0.6b/CERTSCHED_B64") or {}
    base = "overload['ws_4090/qwen3-0.6b/"
    N("NumOverloadNoBoundPeakDebt", nb.get("peak_median_debt"), src, base + "CERTSCHED_noBP'].peak_median_debt",
      "SIMULATION", "int")
    N("NumOverloadNoBoundMaxDebt", nb.get("max_debt_over_seeds"), src, base + "CERTSCHED_noBP'].max_debt_over_seeds",
      "SIMULATION", "int")
    N("NumOverloadNoBoundAbortPct", nb.get("mean_abort_rate"), src, base + "CERTSCHED_noBP'].mean_abort_rate",
      "SIMULATION", "pct", nd=1)
    N("NumOverloadNoBoundAbortsPerRun", nb.get("mean_oom_aborts_per_run"), src,
      base + "CERTSCHED_noBP'].mean_oom_aborts_per_run", "SIMULATION", nd=1)
    N("NumOverloadNoBoundParkedGiB", nb.get("peak_median_parked_GiB"), src, base + "CERTSCHED_noBP'].peak_median_parked_GiB",
      "SIMULATION", nd=0)
    N("NumOverloadBoundMaxDebt", b128.get("max_debt_over_seeds"), src, base + "CERTSCHED_B128'].max_debt_over_seeds",
      "SIMULATION", "int")
    N("NumOverloadBoundParkedGiB", b128.get("peak_median_parked_GiB"), src,
      base + "CERTSCHED_B128'].peak_median_parked_GiB", "SIMULATION", nd=1)
    N("NumOverloadBoundSixtyFourMaxDebt", b64.get("max_debt_over_seeds"), src,
      base + "CERTSCHED_B64'].max_debt_over_seeds", "SIMULATION", "int")
    N("NumOverloadBoundSixtyFourParkedGiB", b64.get("peak_median_parked_GiB"), src,
      base + "CERTSCHED_B64'].peak_median_parked_GiB", "SIMULATION", nd=1)
    N("NumOverloadBoundPNinetyNine", b128.get("mean_cert_p99_s_est"), src, base + "CERTSCHED_B128'].mean_cert_p99_s_est",
      "SUCCESS", nd=0)
    N("NumOverloadNoBoundPNinetyNine", nb.get("mean_cert_p99_s_est"), src, base + "CERTSCHED_noBP'].mean_cert_p99_s_est",
      "SUCCESS", nd=0)
    # offload admission (scheduler view)
    srca = "results/raw/sched/offload_admission.json"
    adm = C.load_json(srca)
    rows = (adm or {}).get("rows", [])
    N("NumOffloadEligibleTasks", rows[0]["eligible_tasks"] if rows else None, srca, "rows[0].eligible_tasks",
      "SIMULATION", "int")
    N("NumOffloadAdmittedTasks", sum(r["admitted_tasks"] for r in rows) if rows else None, srca,
      "sum rows[].admitted_tasks (all links and local devices)", "SIMULATION", "int")
    N("NumOffloadBestSlowdown", min(r["best_T_offload_over_T_local_est"] for r in rows) if rows else None, srca,
      "min rows[].best_T_offload_over_T_local_est", "SUCCESS", nd=0)

# ============================================================ offload model, charts
def derived():
    import offload_model as OM
    M = OM.Model()
    src = "bench/analysis/offload_model.py (reads kernels, fpga_sim, sched/offload_admission)"
    if M.ok:
        oc = max(M.speedup(n, 2 ** 12, OM.LINKS["2.5 GbE"], gpu=False) for n in range(12, 17))
        N("NumOffloadOnchipMaxSpeedup", oc, src, "max over n<=16, R=4096, 2.5 GbE of T_local_cpu/T_offload", "SUCCESS", nd=2)
        for n, tag in ((20, "Twenty"), (24, "TwentyFour")):
            r = M.min_reuse(n, OM.LINKS["2.5 GbE"], gpu=False)
            N(f"NumOffloadMinReuse{tag}", r, src, f"smallest R with speed-up >= 1 vs best CPU, n={n}, 2.5 GbE",
              "SUCCESS", nd=0)
        N("NumOffloadRttUs", M.rtt * 1e6, M.rtt_source, "Ethernet host round trip per sumcheck round", "SUCCESS", nd=0)
        if M.jtag_bps and M.step_rtt:
            oj = max(M.speedup(n, 2 ** 12, M.jtag_bps, gpu=False, rtt=M.step_rtt) for n in range(12, 17))
            N("NumOffloadOnchipMaxSpeedupJtag", oj, src,
              "max over n<=16, R=4096 of T_local_cpu/T_offload over the SUCCESS JTAG link and STEP round trip",
              "SUCCESS", "sci", nd=0)
            N("NumOffloadJtagUploadSixteenS", 32 * 2 ** 16 / M.jtag_bps, src,
              "32 N bytes / (16 B per MEM element write), n=16 (extrapolation)", "SUCCESS", nd=0)
        N("NumOffloadEthUploadSixteenMs", 32 * 2 ** 16 / OM.LINKS["2.5 GbE"] * 1e3, src, "32 N bytes / 2.5 Gb/s, n=16",
          "ANALYTIC", nd=1)
        N("NumOffloadOnchipMaxSpeedupGpu", max(M.speedup(n, 2 ** 12, OM.LINKS["2.5 GbE"], gpu=True) for n in range(12, 17)),
          src, "max over n<=16, R=4096, 2.5 GbE of T_local_best(CPU or GPU)/T_offload", "SUCCESS", nd=2)
        N("NumOffloadMaxSpeedupGpu", max(M.speedup(n / 4, 2 ** 12, OM.LINKS["2.5 GbE"], gpu=True) for n in range(48, 97)),
          src, "max over 12<=n<=24, R=4096, 2.5 GbE of T_local_best(CPU or GPU)/T_offload", "SUCCESS", nd=2)
        N("NumOffloadModelKernelUs", M.cycles(16) / M.f_mhz, src,
          f"cycles(n=16, L={M.L}) / {M.f_mhz} MHz ({M.clock_evidence} clock)",
          "SUCCESS" if M.clock_evidence == "SUCCESS" else "SUCCESS", nd=1)
        N("NumOffloadSyncMsSixteen", (16 - M.lg) * M.rtt * 1e3, src, "(16 - log2 L) round trips x RTT", "SUCCESS", nd=1)
        cpu16 = M.cpu_best.get(16)
        N("NumCpuBestMsSixteen", cpu16[0] * 1e3 if cpu16 else None, "results/raw/kernels/cpu_bench.json",
          "min over compilers/variants/threads total median, n=16", "SUCCESS", nd=2)
    else:
        for nm in ("NumOffloadOnchipMaxSpeedup", "NumOffloadMinReuseTwenty", "NumOffloadMinReuseTwentyFour",
                   "NumOffloadRttUs", "NumOffloadModelKernelUs", "NumOffloadSyncMsSixteen", "NumCpuBestMsSixteen"):
            N(nm, None, "results/raw/kernels/cpu_bench.json", "offload model", "SUCCESS")
    p = C.load_json(C.GEN / "generator.json")
    pt = g(p, "per_token") or {}
    pf = g(p, "per_forward") or {}
    lit = g(p, "literature_inputs") or {}
    N("NumLitDeepProveGptTwoTokMin", 60 / lit["deepprove_gpt2"]["value_s"] if lit else None, srcp,
      "literature_inputs.deepprove_gpt2 (174 tokens/min, gailly2026deepprove)", "REPORTED", nd=0)
    N("NumLitDeepProveGemmaTokMin", 60 / lit["deepprove_gemma3"]["value_s"] if lit else None, srcp,
      "literature_inputs.deepprove_gemma3 (gailly2026deepprove)", "REPORTED", nd=0)
    N("NumLitZkgptS", g(lit, "zkgpt_gpt2.value_s"), srcp, "literature_inputs.zkgpt_gpt2.value_s (qu2025zkgpt, upper bound)",
      "REPORTED", nd=0)
    N("NumLitJoltS", g(lit, "joltatlas_gpt2.value_s"), srcp, "literature_inputs.joltatlas_gpt2.value_s (benno2026joltatlas)",
      "REPORTED", nd=0)
    N("NumLitZkllmMin", (g(lit, "zkllm_13b.value_s") or 0) / 60 if lit else None, srcp,
      "literature_inputs.zkllm_13b.value_s / 60 (sun2024zkllm)", "REPORTED", nd=0)
    q = pt.get("qwen3_0.6b") or {}
    N("NumProjQwenPerTokenS", q.get("prove_s_per_token_est"), srcp, "per_token.qwen3_0.6b.prove_s_per_token_est",
      "SUCCESS", nd=2)
    N("NumProjQwenPerTokenEmpiricalS", q.get("prove_s_per_token_empirical_est"), srcp,
      "per_token.qwen3_0.6b.prove_s_per_token_empirical_est", "SUCCESS", nd=2)
    N("NumProjQwenTokensPerMin", q.get("tokens_per_min_est"), srcp, "per_token.qwen3_0.6b.tokens_per_min_est",
      "SUCCESS", nd=0)
    N("NumProjQwenWorkRatio", q.get("work_ratio_vs_gpt2"), srcp, "per_token.qwen3_0.6b.work_ratio_vs_gpt2 (ANALYTIC counts)",
      "ANALYTIC", nd=1)
    gm = pt.get("gemma3_1b") or {}
    N("NumProjGemmaPerTokenS", gm.get("prove_s_per_token_est"), srcp, "per_token.gemma3_1b.prove_s_per_token_est",
      "SUCCESS", nd=2)
    N("NumProjGemmaOverpredict", g(p, "validation.deepprove_gemma3.predicted_over_reported_est"), srcp,
      "validation.deepprove_gemma3.predicted_over_reported_est", "SUCCESS", nd=1)
    N("NumProjVerifyLowS", q.get("verify_s_low_est"), srcp, "per_token.*.verify_s_low_est (DeepProve reported range)",
      "SUCCESS", nd=1)
    N("NumProjVerifyHighS", q.get("verify_s_high_est"), srcp, "per_token.*.verify_s_high_est", "SUCCESS", nd=1)
    for base, tag in (("zkgpt_gpt2", "Zkgpt"), ("joltatlas_gpt2", "Jolt"), ("zkllm_13b", "Zkllm")):
        N(f"NumProj{tag}QwenS", g(pf, f"{base}.qwen3_0.6b.prove_s_est"), srcp, f"per_forward.{base}.qwen3_0.6b.prove_s_est",
          "SUCCESS", nd=0)
    N("NumProjZkllmGptTwoS", g(pf, "zkllm_13b.gpt2_124m.prove_s_est"), srcp, "per_forward.zkllm_13b.gpt2_124m.prove_s_est",
      "SUCCESS", nd=1)
    # latency CDF: never-certified fraction of KEEP_ALL (ws_4090, Qwen3, rho 0.7)
    cdf = C.load_csv("results/processed/sched_latency_cdf.csv")
    kf = next((C.fnum(r["completed_fraction"]) for r in cdf or [] if r["scenario"] == "ws_4090"
               and r["model"] == "qwen3-0.6b" and r["variant"] == "KEEP_ALL"), None)
    N("NumCdfKeepAllNeverCertifiedPct", (1 - kf) if kf is not None else None, "results/processed/sched_latency_cdf.csv",
      "1 - completed_fraction (ws_4090, qwen3-0.6b, KEEP_ALL, rho 0.7)", "SIMULATION", "pct", nd=0)

# ============================================================ prototype size (static line count)
def loc():
    src = "paper/generated/loc.json"
    o = C.load_json(C.GEN / "loc.json")
    N("NumLocFiles", g(o, "total_files"), src, "total_files (bench/analysis/loc_count.py)", "SUCCESS", "int")
    N("NumLocLines", g(o, "total_lines"), src, "total_lines (non-blank, non-comment; static count)", "SUCCESS", "int")
    gold = next((r for r in (o or {}).get("rows", []) if r["component"] == "Golden model"), None)
    N("NumLocGolden", g(gold, "lines"), src, "rows[Golden model].lines", "SUCCESS", "int")

def write():
    ts = C.now_iso()
    lines = ["% GENERATED by bench/analysis/make_numbers.py.",
             f"% generated {ts}; every macro names its source file, field and evidence class.",
             "% SUCCESS and POST_SYNTH values expand to",]
    for m in MACROS:
        tail = f"% {m['ev']}; {m['src']} :: {m['field']}" + (f"; {m['note']}" if m["note"] else "")
        if m["pending"]:
            tail = "% input missing; " + tail[2:]
        lines.append(f"\\newcommand{{\\{m['name']}}}{{{m['body']}}}{tail}")
    C.write_text(OUT_TEX, "\n".join(lines) + "\n")
    md = ["# Paper numbers (generated)", "",
          f"Generated by `python bench/analysis/make_numbers.py` at {ts}. Do not edit: re-run the generator.",
          "Macros are defined in `paper/generated/numbers.tex` (input by `paper/main.tex`). SUCCESS and",
          "POST_SYNTH values expand to `\\est{...}`; `pending` means the input file does not exist yet.", "",
          f"Total macros: {len(MACROS)}; pending: {sum(m['pending'] for m in MACROS)}.", "",
          "| Macro | Value | Evidence | Source | Field |", "| :-- | --: | :-- | :-- | :-- |"]
    for m in MACROS:
        val = m["value"].replace("{,}", ",").replace("|", "\\|")
        md.append(f"| `\\{m['name']}` | {val} | {m['ev']} | `{m['src']}` | {m['field'].replace('|', '/')} |")
    C.write_text(OUT_MD, "\n".join(md) + "\n")
    print(f"wrote {C.rel(OUT_TEX)} and {C.rel(OUT_MD)}: {len(MACROS)} macros")
def main() -> int:
    for fn in (attacks, contracts, kernels, kernels_final, fpga, board, gpt2prove, realmodels, sched, derived, loc):
        try:
            fn()
        except Exception as e:  # never leave numbers.tex half-written because one input changed schema
            import traceback
            traceback.print_exc()
            C.warn(f"make_numbers.{fn.__name__} failed: {e!r}")
    write()
    return 0

if __name__ == "__main__":
    sys.exit(main())
