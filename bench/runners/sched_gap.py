"""CertSched optimality gap (C3).

  (a) synthetic min-cost eviction instances, n = 8..20 items: greedy (with and
      without the completion step) vs exact brute force. Pure algorithmic result,
      no hardware constants involved (SIMULATION).
  (b) model-derived small instances: one-layer GPT-2 / Qwen3 shapes, one item per
      op bundle (y + limbs + helper), costs from configs/cost_model.yaml, exact
      brute force. Costs derive from constants, so gaps are *_est.
  (c) full-size instances (thousands of tensors): greedy vs the LP (fractional)
      lower bound, an upper bound on the true gap; *_est for the same reason.

Outputs: results/raw/sched/optimality_gap.json, results/processed/sched_optimality_gap.csv Usage:   python bench/runners/sched_gap.py [--instances 200]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from sched_common import GiB, inputs, meta, write_csv, write_json  # noqa: E402

import numpy as np  # noqa: E402

from edgecert.runtime.dag import Encoding, ModelShape, build_dag, get_shape  # noqa: E402
from edgecert.runtime.scheduler import (HOST, RECOMP, SPILLD, Item, action_list, candidate_items,  # noqa: E402
                                        check_schedule, emit_schedule, exact_cover, greedy_cover,
                                        uniform_decisions)
from edgecert.runtime.simulate import load_cost_model, make_profile  # noqa: E402

def ratio_greedy(items, need):
    """Plain ratio greedy with clean-up."""
    if need <= 0:
        return 0.0
    ranked = sorted(items, key=lambda it: -(it.nbytes / max(it.best()[1], 1e-15)))
    ch, saved = [], 0
    for it in ranked:
        if saved >= need:
            break
        ch.append(it)
        saved += it.nbytes
    for it in sorted(ch, key=lambda it: -it.best()[1]):
        if saved - it.nbytes >= need:
            ch.remove(it)
            saved -= it.nbytes
    return sum(it.best()[1] for it in ch)

def lp_bound(items, need):
    """Fractional (LP) lower bound on the min-cost eviction."""
    if need <= 0:
        return 0.0
    ranked = sorted(items, key=lambda it: it.best()[1] / it.nbytes)
    got, cost = 0.0, 0.0
    for it in ranked:
        take = min(it.nbytes, need - got)
        cost += it.best()[1] * take / it.nbytes
        got += take
        if got >= need:
            break
    return cost

def synth(n, regime, rng):
    items = []
    for i in range(n):
        b = int(rng.lognormvariate(16, 1.2)) + 1
        if regime == "correlated":
            c = b * 1e-9 * rng.lognormvariate(0, 0.5)
            costs = {SPILLD: c}
        elif regime == "uncorrelated":
            costs = {SPILLD: rng.uniform(0.01, 1.0)}
        else:  # three options with independent multipliers (host / spill / recompute)
            costs = {HOST: b * 1e-9 * rng.uniform(0.5, 2), SPILLD: b * 1e-9 * rng.uniform(2, 8),
                     RECOMP: rng.uniform(0.001, 0.5)}
        items.append(Item(f"t{i}", b, costs))
    return items

def summarize(gaps):
    g = np.array(gaps, dtype=float)
    if g.size == 0:
        return {}
    return {"n": int(g.size), "mean": float(g.mean()), "p50": float(np.quantile(g, 0.5)),
            "p95": float(np.quantile(g, 0.95)), "max": float(g.max()),
            "frac_optimal": float(np.mean(g <= 1e-9))}

def bundle_items(dag, prof, acts):
    """Aggregate tensor items per producing op (one decision per op bundle)."""
    its = candidate_items(dag, prof, acts, allow_host=not prof.unified)
    agg = defaultdict(lambda: [0, defaultdict(float), set()])
    for it in its:
        p = dag.tensors[it.tid].producer or it.tid
        p = p.replace("d:", "op:")          # helper goes with its column bundle
        a = agg[p]
        a[0] += it.nbytes
        a[2].add(frozenset(it.costs))
        o, c = it.best()
        a[1]["best"] += c
    return [Item(k, v[0], {"BUNDLE": v[1]["best"]}) for k, v in agg.items()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=200)
    a = ap.parse_args()
    cmd = f"python bench/runners/sched_gap.py --instances {a.instances}"
    cm = load_cost_model()
    rows = []
    t0 = time.time()
    # ---------------- (a) synthetic
    rng = random.Random(2026)
    synth_summary = {}
    for regime in ("correlated", "uncorrelated", "three_options"):
        for n in (8, 12, 16, 20):
            for frac in (0.25, 0.5, 0.75):
                gg, gr = [], []
                for _ in range(a.instances):
                    items = synth(n, regime, rng)
                    need = int(frac * sum(it.nbytes for it in items))
                    _, gcost, _ = greedy_cover(items, need)
                    opt, _ = exact_cover(items, need)
                    rc = ratio_greedy(items, need)
                    gg.append(gcost / opt - 1 if opt > 0 else 0.0)
                    gr.append(rc / opt - 1 if opt > 0 else 0.0)
                s1, s2 = summarize(gg), summarize(gr)
                synth_summary[f"{regime}/n{n}/f{frac}"] = {"certsched_greedy": s1, "ratio_greedy": s2}
                rows.append({"family": "synthetic", "regime": regime, "n_items": n, "evict_fraction": frac,
                             "instances": a.instances, "gap_mean": s1["mean"], "gap_p95": s1["p95"],
                             "gap_max": s1["max"], "frac_optimal": s1["frac_optimal"],
                             "ratio_only_gap_mean": s2["mean"], "ratio_only_gap_p95": s2["p95"],
                             "ratio_only_gap_max": s2["max"], "evidence": "SIMULATION"})
    print(f"synthetic done {time.time() - t0:.1f}s", flush=True)

    # ---------------- (b) model-derived small instances (exact)
    small = []
    for base in ("gpt2-124m", "qwen3-0.6b"):
        s = get_shape(base)
        one = ModelShape(**{**s.__dict__, "name": s.name + "-1layer", "n_layers": 1})
        for ctx in (128, 512):
            dag = build_dag(one, ctx, 1, Encoding(head_split=False))
            acts = action_list(dag)
            for dev in ("gpu_rtx4090", "edge_unified_8g"):
                prof = make_profile(cm, dev)
                items = bundle_items(dag, prof, acts)
                total = sum(it.nbytes for it in items)
                for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                    need = int(frac * total)
                    _, gcost, _ = greedy_cover(items, need)
                    opt, _ = exact_cover(items, need, max_items=24)
                    rc = ratio_greedy(items, need)
                    r = {"family": "model_small", "regime": f"{one.name}@{ctx}/{dev}",
                         "n_items": len(items), "evict_fraction": frac, "instances": 1,
                         "greedy_s_est": gcost, "opt_s_est": opt, "ratio_only_s_est": rc,
                         "gap_mean": (gcost / opt - 1) if opt else 0.0,
                         "ratio_only_gap_mean": (rc / opt - 1) if opt else 0.0}
                    rows.append(r)
                    small.append(r)
    print(f"model-small done {time.time() - t0:.1f}s", flush=True)

    # ---------------- (c) full-size instances vs LP bound
    full = []
    for model, ctx, dev, cap in (("gpt2-124m", 2048, "gpu_rtx4090", 20), ("gpt2-124m", 2048, "edge_unified_8g", 8),
                                 ("qwen3-0.6b", 512, "gpu_rtx4090", 8), ("qwen3-0.6b", 512, "edge_unified_16g", 16),
                                 ("qwen3-0.6b", 2048, "gpu_rtx6000ada", 40), ("gemma3-1b", 512, "edge_unified_8g", 8)):
        dag = build_dag(model, ctx, 32)
        acts = action_list(dag)
        prof = make_profile(cm, dev)
        c0 = check_schedule(dag, emit_schedule(dag, uniform_decisions(dag, "KEEP_ALL"), acts))
        peak0 = c0.peak_unified if prof.unified else c0.peak_device
        need = peak0 - cap * GiB
        items = candidate_items(dag, prof, acts, allow_host=not prof.unified)
        _, gcost, saved = greedy_cover(items, need)
        lb = lp_bound(items, need)
        rc = ratio_greedy(items, need)
        r = {"family": "model_full", "regime": f"{model}@{ctx}/{dev}/{cap}GiB", "n_items": len(items),
             "evict_fraction": need / max(sum(it.nbytes for it in items), 1), "instances": 1,
             "greedy_s_est": gcost, "lp_bound_s_est": lb, "ratio_only_s_est": rc,
             "gap_vs_lp_upper_bound": (gcost / lb - 1) if lb > 0 else 0.0,
             "ratio_only_gap_vs_lp": (rc / lb - 1) if lb > 0 else 0.0
        rows.append(r)
        full.append(r)
        print("full", r["regime"], f"{r['gap_vs_lp_upper_bound']:.4f}", flush=True)

    sm = [r["gap_mean"] for r in small]
    out = {
        "status": "SUCCESS", **meta(cmd, cm),
        "inputs": inputs(cm)
        "synthetic": synth_summary,
        "model_small": small, "model_small_gap_summary_est": summarize(sm),
        "model_full": full,
        "elapsed_s": time.time() - t0,
    }
    write_json("optimality_gap.json", out)
    write_csv("sched_optimality_gap.csv", rows, [
        "family", "regime", "n_items", "evict_fraction", "instances", "gap_mean", "gap_p95", "gap_max",
        "frac_optimal", "ratio_only_gap_mean", "ratio_only_gap_p95", "ratio_only_gap_max",
        "greedy_s_est", "opt_s_est", "ratio_only_s_est", "lp_bound_s_est", "gap_vs_lp_upper_bound",
        "ratio_only_gap_vs_lp", "evidence"])
    print("done", f"{time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
