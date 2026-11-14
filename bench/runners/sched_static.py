"""CertSched static experiments (C3).

  E1  peak memory vs context length (32/128/512/2048) per policy per model, both
      protocol modes, CertSched under 8/16/20/40 GiB device caps;
  E5  legality-checker outcomes per policy and for adversarial schedule mutations;
  F2  a small DAG export for the legal proof/memory DAG figure;
  OA  scheduler-side FPGA offload admission view (T_offload vs local) per task.

Outputs: results/raw/sched/{memory_vs_ctx,legality,dag_example,offload_admission}.json
         results/processed/{sched_memory_vs_ctx,sched_legality,sched_offload_admission}.csv

Usage: python bench/runners/sched_static.py [--jobs 6] [--quick]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from sched_common import GiB, inputs, meta, write_csv, write_json  # noqa: E402

from edgecert.runtime.dag import MODEL_ORDER, Encoding, ModelShape, build_dag, dag_example_json  # noqa: E402
from edgecert.runtime.scheduler import (CHALLENGE, COMMIT, DERIVE, FREE, MODIFY, OPEN, PROVE,  # noqa: E402
                                        REBIND, RECOMPUTE, Event, action_list, check_schedule,
                                        emit_schedule, plan_certsched, run_policy,
                                        schedule_time, uniform_decisions)
from edgecert.runtime.simulate import load_cost_model, make_profile  # noqa: E402

CTX = [32, 128, 512, 2048]
TG = 32
UNIFORM = ["KEEP_ALL", "LAYER_STREAM_NAIVE", "HOST_STAGE", "SPILL", "RECOMPUTE"]
CAPS = [(8, "edge_unified_8g"), (16, "edge_unified_16g"), (20, "gpu_rtx4090"), (40, "gpu_rtx6000ada")]
HOST_CAP = 128 * GiB
SSD_CAP = 500 * 10 ** 9

def _job(args):
    model, ctx, proto = args
    cm = load_cost_model()
    t0 = time.time()
    dag = build_dag(model, ctx, TG, Encoding(protocol=proto))
    acts = action_list(dag)
    rows = []
    ref = make_profile(cm, "gpu_rtx4090")
    keep_total = None
    uniform_runs = {}
    for pol in UNIFORM:
        r = run_policy(dag, pol, ref, acts=acts)
        uniform_runs[pol] = r
        c = r.check
        if pol == "KEEP_ALL":
            keep_total = r.time["total_s"]
        row = {
            "model": model, "ctx": ctx, "generated": TG, "protocol": proto, "policy": pol,
            "cap_GiB": None, "profile": ref.name,
            "legal": c.legal, "violations": c.rule_counts,
            "peak_device_GiB": c.peak_device / GiB, "peak_host_GiB": c.peak_host / GiB,
            "peak_ssd_GiB": c.peak_ssd / GiB, "peak_unified_GiB": c.peak_unified / GiB,
            "fits_ws20": c.legal and c.peak_device <= 20 * GiB and c.peak_host <= HOST_CAP and c.peak_ssd <= SSD_CAP,
            "fits_ws40": c.legal and c.peak_device <= 40 * GiB and c.peak_host <= HOST_CAP and c.peak_ssd <= SSD_CAP,
            "fits_edge16": c.legal and c.peak_unified <= 16 * GiB and c.peak_ssd <= SSD_CAP,
            "fits_edge8": c.legal and c.peak_unified <= 8 * GiB and c.peak_ssd <= SSD_CAP,
            "total_s_est": r.time["total_s"] if c.legal else None,
            "added_s_est": (r.time["total_s"] - keep_total) if c.legal else None,
            "overhead_s_est": r.time["overhead_s"] if c.legal else None,
            "n_events": c.n_events,
            "witness_GiB": dag.witness_bytes() / GiB, "weights_GiB": dag.static_bytes() / GiB,
            "beyond_native_ctx": dag.summary()["beyond_native_ctx"],
        }
        rows.append(row)
    for cap, dev in CAPS:
        prof = make_profile(cm, dev)
        kt = schedule_time(dag, uniform_runs["KEEP_ALL"].events, prof)["total_s"]
        # uniform policies re-timed on this cap's profile (their events are profile-independent)
        for pol in ("KEEP_ALL", "HOST_STAGE", "SPILL", "RECOMPUTE"):
            u = uniform_runs[pol]
            c = u.check
            tt = schedule_time(dag, u.events, prof)
            peak_eff = c.peak_unified if prof.unified else c.peak_device
            fits = (c.legal and peak_eff <= cap * GiB and c.peak_ssd <= SSD_CAP
                    and (prof.unified or c.peak_host <= HOST_CAP))
            rows.append({
                "model": model, "ctx": ctx, "generated": TG, "protocol": proto, "policy": pol,
                "cap_GiB": cap, "profile": dev, "legal": c.legal, "violations": c.rule_counts,
                "peak_device_GiB": c.peak_device / GiB, "peak_host_GiB": c.peak_host / GiB,
                "peak_ssd_GiB": c.peak_ssd / GiB, "peak_unified_GiB": c.peak_unified / GiB,
                "peak_effective_GiB": peak_eff / GiB, "fits_cap": bool(fits), "feasible": bool(fits),
                "total_s_est": tt["total_s"], "added_s_est": tt["total_s"] - kt,
                "overhead_s_est": tt["overhead_s"], "n_events": c.n_events,
                "witness_GiB": dag.witness_bytes() / GiB, "weights_GiB": dag.static_bytes() / GiB,
                "beyond_native_ctx": dag.summary()["beyond_native_ctx"], "selected": "uniform"})
        r = plan_certsched(dag, prof, cap * GiB, None if prof.unified else HOST_CAP, SSD_CAP, acts=acts)
        c = r.check
        dec = {}
        for v in r.decisions.values():
            dec[v] = dec.get(v, 0) + 1
        peak_eff = c.peak_unified if prof.unified else c.peak_device
        rows.append({
            "model": model, "ctx": ctx, "generated": TG, "protocol": proto, "policy": "CERTSCHED",
            "cap_GiB": cap, "profile": dev, "legal": c.legal, "violations": c.rule_counts,
            "peak_device_GiB": c.peak_device / GiB, "peak_host_GiB": c.peak_host / GiB,
            "peak_ssd_GiB": c.peak_ssd / GiB, "peak_unified_GiB": c.peak_unified / GiB,
            "peak_effective_GiB": peak_eff / GiB,
            "fits_cap": bool(c.fits), "feasible": bool(r.feasible),
            "total_s_est": r.time["total_s"], "added_s_est": r.time["total_s"] - kt,
            "overhead_s_est": r.time["overhead_s"], "predicted_added_s_est": r.predicted_added_s,
            "iterations": r.iterations, "decisions": dec, "n_events": c.n_events,
            "selected": r.selected,
            "alternatives_added_s_est": {k: (v - kt if v != float("inf") else None)
                                         for k, v in r.alternatives.items()},
            "witness_GiB": dag.witness_bytes() / GiB, "weights_GiB": dag.static_bytes() / GiB,
            "beyond_native_ctx": dag.summary()["beyond_native_ctx"],
        })
    return {"rows": rows, "elapsed_s": time.time() - t0, "key": [model, ctx, proto]}

# ------------------------------------------------------------------ E5
TINY_GPT = ModelShape("tiny-gpt2", n_layers=2, d_model=32, n_heads=2, n_kv_heads=2, head_dim=16,
                      d_ff=128, vocab=128, gated_mlp=False, qk_norm=False, tied_embeddings=True,
                      native_ctx=128, learned_pos=True)

def _mutations(dag, keep, rec, spill_plain):
    """Adversarial schedules derived from legal ones (name, events, expected rule)."""
    T = dag.tensors
    idx = lambda ev, k, t=None: next(i for i, e in enumerate(ev) if e.kind == k and (t is None or e.target == t))  # noqa: E731
    chal = [i for i, e in enumerate(keep) if e.kind == CHALLENGE]
    first_ch, last_ch = chal[0], chal[-1]
    ti = next(i for i, e in enumerate(keep) if i > last_ch and e.kind == PROVE
              and any(not T[r].static for r in dag.tasks[e.target].reads))
    victim = next(r for r in dag.tasks[keep[ti].target].reads if not T[r].static)
    out = []
    ev = list(keep)
    ev.insert(last_ch + 1, Event(FREE, victim))
    out.append(("free_before_proof", ev, "A_UNAVAILABLE"))
    ev = list(keep)
    j = idx(ev, COMMIT, victim)
    c = ev.pop(j)
    binding = {ch.id for ch in dag.challenges.values() if victim in ch.depends}
    ev.insert(next(i for i, e in enumerate(ev) if e.kind == CHALLENGE and e.target in binding) + 1, c)
    out.append(("commit_after_challenge", ev, "C_CHALLENGE_BEFORE_COMMIT"))
    ev = list(keep)
    ev.insert(last_ch + 1, Event(MODIFY, victim))
    out.append(("adapt_after_challenge", ev, "B_ADAPT_AFTER_CHALLENGE"))
    ev = list(keep)
    ev[last_ch + 1:last_ch + 1] = [Event(MODIFY, victim), Event(COMMIT, victim)]
    out.append(("adapt_and_recommit", ev, "B_RECOMMIT"))
    ev = list(keep)
    ev.insert(last_ch + 1, Event(MODIFY, "W:0.q"))
    out.append(("modify_model_weight", ev, "B_MODIFIED_AFTER_COMMIT"))
    ev = list(keep)
    d = ev.pop(idx(ev, DERIVE))
    ev.insert(first_ch, d)
    out.append(("derive_before_challenge", ev, "C_TASK_BEFORE_CHALLENGE"))
    ev = list(keep)
    p = ev.pop(ti)
    ev.insert(first_ch, p)
    out.append(("prove_before_challenge", ev, "C_TASK_BEFORE_CHALLENGE"))
    ev = list(keep)
    j2 = next(k for k in range(ti + 1, len(ev)) if ev[k].kind in (PROVE, OPEN))
    ev[ti], ev[j2] = ev[j2], ev[ti]
    out.append(("transcript_reorder", ev, "D_TRANSCRIPT_ORDER"))
    ev = [e for k, e in enumerate(keep) if k != ti]
    out.append(("drop_proof_obligation", ev, "E_MISSING_OBLIGATION"))
    ev = list(keep)
    ev.insert(ti + 1, ev[ti])
    out.append(("duplicate_proof_task", ev, "E_DUPLICATE_TASK"))
        # the remaining mutations need spill / recompute events in the base schedule
    if any(e.kind == "SPILL" for e in spill_plain):
        out.append(("unencrypted_spill", spill_plain, "A_UNENCRYPTED_SPILL"))
    if any(e.kind == REBIND for e in rec):
        j = idx(rec, REBIND)
        out.append(("recompute_without_rebind", rec[:j] + rec[j + 1:], "A_UNBOUND_RECOMPUTE"))
        j = idx(rec, RECOMPUTE)
        op = dag.ops.get(rec[j].target) or dag.derive_ops[rec[j].target]
        inp = next(u for u in op.inputs if not T[u].static)
        ev = list(rec)
        ev.insert(j, Event(MODIFY, inp))
        out.append(("recompute_from_tampered_input", ev, "B_REBIND_MISMATCH"))
    return out

def legality(cm):
    ref = make_profile(cm, "gpu_rtx4090")
    rows = []
    for shape_name, shape, ctx in (("gpt2-124m", "gpt2-124m", 128), ("tiny-gpt2", TINY_GPT, 16)):
        for proto in ("GLOBAL", "LAYERWISE"):
            dag = build_dag(shape, ctx, 8, Encoding(protocol=proto))
            acts = action_list(dag)
            keep_peak = None
            for pol in UNIFORM + ["CERTSCHED"]:
                if pol == "CERTSCHED":
                    cap = dag.static_bytes() + (keep_peak - dag.static_bytes()) // 4
                    r = plan_certsched(dag, ref, cap, HOST_CAP, SSD_CAP, acts=acts)
                else:
                    r = run_policy(dag, pol, ref, acts=acts)
                if pol == "KEEP_ALL":
                    keep_peak = r.check.peak_device
                rows.append({"model": shape_name, "ctx": ctx, "protocol": proto, "case": pol,
                             "kind": "policy", "expected": "reject" if pol == "LAYER_STREAM_NAIVE" else "accept",
                             "legal": r.check.legal, "rules": r.check.rule_counts,
                             "outcome": "accepted" if r.check.legal else "rejected",
                             "correct": r.check.legal == (pol != "LAYER_STREAM_NAIVE"),
                             "n_events": r.check.n_events})
            keep = emit_schedule(dag, uniform_decisions(dag, "KEEP_ALL"), acts)
            rec = emit_schedule(dag, uniform_decisions(dag, "RECOMPUTE"), acts)
            sp = emit_schedule(dag, uniform_decisions(dag, "SPILL"), acts, encrypt_spill=False)
            for name, ev, exp_rule in _mutations(dag, keep, rec, sp):
                c = check_schedule(dag, ev)
                rows.append({"model": shape_name, "ctx": ctx, "protocol": proto, "case": name,
                             "kind": "mutation", "expected": "reject", "expected_rule": exp_rule,
                             "legal": c.legal, "rules": c.rule_counts,
                             "outcome": "accepted" if c.legal else "rejected",
                             "correct": (not c.legal) and exp_rule in c.rule_counts,
                             "n_events": c.n_events})
    return rows

# ------------------------------------------------------------------ OA
def offload_admission(cm):
    """T_offload = T_h2b + T_fpga + T_b2h + T_sync  vs  T_local, per product-sumcheck task."""
    f = cm["devices"]["fpga_zcu102_sumfold"]
    links = cm["links"]
    fpga_rate = f["sc_elems_per_s"]
    nmax = 1 << int(f["onchip_max_log2_n"])
    rows = []
    dag = build_dag("gpt2-124m", 128, 32)
    for link in ("eth_1g", "eth_2g5", "jtag_axi"):
        bw = links[link]["nominal_GBps"] * 1e9
        rtt = {"eth_1g": 100e-6, "eth_2g5": 100e-6, "jtag_axi": 2e-3}[link]
        for local in ("cpu_i9_14900k", "gpu_rtx4090", "edge_unified_16g"):
            prof = make_profile(cm, local)
            n_elig = n_adm = 0
            best_ratio = math.inf
            for t in dag.tasks.values():
                if t.kind not in ("sc_hadamard", "lookup"):
                    continue
                N = t.work // 2 if t.kind == "sc_hadamard" else t.work
                if N < 2 or N > nmax:
                    continue
                n_elig += 1
                rounds = max(1, math.ceil(math.log2(N)))
                c = prof.c_hadamard if t.kind == "sc_hadamard" else prof.c_lookup
                t_local = t.work * c / prof.sc_elems_per_s
                t_off = (2 * N * 16) / bw + t.work * c / fpga_rate + rounds * 48 / bw + rounds * rtt
                best_ratio = min(best_ratio, t_off / t_local)
                if t_off < t_local:
                    n_adm += 1
            rows.append({"link": link, "local_device": local, "eligible_tasks": n_elig,
                         "admitted_tasks": n_adm, "best_T_offload_over_T_local_est": best_ratio,
                         "fpga_elems_per_s": fpga_rate, "fpga_rate_evidence": f["evidence"]["sc_elems_per_s"],
                         "rtt_s_assumed": rtt})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cmd = "python bench/runners/sched_static.py" + (" --quick" if a.quick else "") + f" --jobs {a.jobs}"
    cm = load_cost_model()
    est = inputs(cm)

    # F2 DAG example
    small = build_dag(TINY_GPT, 4, 2)
    ex = dag_example_json(small, max_nodes=600)
    ex_lw = dag_example_json(build_dag(TINY_GPT, 4, 2, Encoding(protocol="LAYERWISE")), max_nodes=600)
    write_json("dag_example.json", {"status": "ANALYTIC", "evidence_class": "ANALYTIC", **meta(cmd),
                                    "shape": TINY_GPT.__dict__, "global": ex, "layerwise": ex_lw})

    # E5
    t0 = time.time()
    leg = legality(cm)
    write_json("legality.json", {"status": "SIMULATION_ONLY", "evidence_class": "SIMULATION",
                                 **meta(cmd, cm), "inputs": [],
                                 "all_correct": all(r["correct"] for r in leg), "rows": leg,
                                 "elapsed_s": time.time() - t0})
    write_csv("sched_legality.csv", leg, ["model", "ctx", "protocol", "kind", "case", "expected",
                                          "expected_rule", "outcome", "correct", "rules", "n_events"])
    print("legality: all_correct =", all(r["correct"] for r in leg), f"({time.time() - t0:.1f}s)")

    # OA
    oa = offload_admission(cm)
    write_json("offload_admission.json", {"status": "SIMULATION_ONLY", "evidence_class": "SIMULATION",
                                          **meta(cmd, cm), 
                                          "note": "ratios derive from constants",
                                          "rows": oa})
    write_csv("sched_offload_admission.csv", oa)

    # E1
    models = MODEL_ORDER if not a.quick else ["gpt2-124m", "qwen3-0.6b"]
    ctxs = CTX if not a.quick else [32, 128]
    jobs = [(m, c, p) for p in ("GLOBAL", "LAYERWISE") for m in models for c in ctxs]
    jobs.sort(key=lambda j: -(MODEL_ORDER.index(j[0]) * 10 + CTX.index(j[1])))   # big first
    t0 = time.time()
    res = []
    with Pool(a.jobs) as pool:
        for out in pool.imap_unordered(_job, jobs):
            res.append(out)
            print("done", out["key"], f"{out['elapsed_s']:.1f}s", flush=True)
    rows = [r for o in res for r in o["rows"]]
    rows.sort(key=lambda r: (r["protocol"], MODEL_ORDER.index(r["model"]), r["ctx"], r["policy"], r["cap_GiB"] or 0))
    write_json("memory_vs_ctx.json", {
        "status": "SIMULATION_ONLY", "evidence_class": "SIMULATION", **meta(cmd, cm),
        "generated_tokens": TG,
        "encoding": Encoding().__dict__, "rows": rows, "elapsed_s": time.time() - t0})
    write_csv("sched_memory_vs_ctx.csv", rows, [
        "protocol", "model", "ctx", "generated", "policy", "cap_GiB", "profile", "legal",
        "peak_device_GiB", "peak_host_GiB", "peak_ssd_GiB", "peak_unified_GiB", "peak_effective_GiB",
        "fits_ws20", "fits_ws40", "fits_edge16", "fits_edge8", "fits_cap", "feasible",
        "added_s_est", "overhead_s_est", "total_s_est", "witness_GiB", "weights_GiB",
        "beyond_native_ctx", "selected", "violations", "decisions", "alternatives_added_s_est"])
    print("memory_vs_ctx rows:", len(rows), f"({time.time() - t0:.1f}s)")

if __name__ == "__main__":
    main()
