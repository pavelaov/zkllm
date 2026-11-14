"""Discrete-event simulation of certify-before-release serving under CertSched (C3).

A Poisson request stream passes admission control, inference (provisional tokens), optional witness transfer, proving and verification (certified tokens). Per-request service times, device peaks and parked witness bytes come from the legality-checked schedules of ``edgecert.runtime.scheduler`` evaluated with the cost model in ``configs/cost_model.yaml``.

Proof debt = tokens generated but not yet certified. With a debt bound B, admission of a request (Tg tokens) waits while reserved debt + Tg > B (backpressure: requests are deferred, proof obligations are never dropped).
"""
from __future__ import annotations

import heapq
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .dag import build_dag, Encoding
from .scheduler import (DeviceProfile, action_list, run_policy, op_time, schedule_time,
                        check_schedule, emit_schedule, uniform_decisions)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GiB = 1 << 30

# ------------------------------------------------------------ YAML subset
def _parse_scalar(s: str):
    s = s.strip()
    if s == "" or s in ("null", "~", "None"):
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s.startswith('"') and s.endswith('"'):
        return bytes(s[1:-1], "utf-8").decode("unicode_escape")
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [] if not inner else [_parse_scalar(x) for x in _split_flow(inner)]
    try:
        if any(c in s for c in ".eE") and not s.lstrip("-").isdigit():
            return float(s)
        return int(s)
    except ValueError:
        return s

def _split_flow(s: str) -> List[str]:
    out, cur, q = [], "", None
    for ch in s:
        if q:
            cur += ch
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            cur += ch
        elif ch == ",":
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out

def _strip_comment(line: str) -> str:
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line

def load_yaml_subset(text: str) -> dict:
    """Block mappings with 2-space indentation, scalars, inline lists, comments."""
    root: dict = {}
    stack: List[Tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip(" "))
        key, _, rest = line.strip().partition(":")
        while stack and ind <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest.strip() == "":
            d: dict = {}
            parent[key.strip()] = d
            stack.append((ind, d))
        else:
            parent[key.strip()] = _parse_scalar(rest)
    return root

def load_cost_model(path: Optional[str] = None) -> dict:
    path = path or os.path.join(ROOT, "configs", "cost_model.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return load_yaml_subset(f.read())

def make_profile(cm: dict, device: str) -> DeviceProfile:
    d = cm["devices"][device]
    links = cm["links"]
    wf = cm["work_factors"]
    link = d.get("link")
    link_Bps = None
    if link and not d.get("unified"):
        L = links[link]
        link_Bps = L["nominal_GBps"] * L.get("sustained_fraction", 1.0) * 1e9
    ssd = links[d.get("ssd") or "nvme_gen4_x4"]
    aes = d.get("aes_gcm_GBps") or cm["devices"]["cpu_i9_14900k"]["aes_gcm_GBps"]
    ev = d.get("evidence", {})
    measured = {k: v for k, v in ev.items() if isinstance(v, str) and v.startswith("MEASURED")}
    return DeviceProfile(
        name=device,
        int8_ops_per_s=d["int8_tops_nominal"] * 1e12 * d["int8_efficiency"],
        mem_Bps=d["mem_bw_GBps_nominal"] * 1e9 * d["mem_efficiency"],
        sc_elems_per_s=float(d["sc_elems_per_s"]),
        commit_Bps=d["commit_GBps"] * 1e9,
        hash_Bps=d["hash_GBps"] * 1e9,
        link_Bps=link_Bps,
        ssd_read_Bps=ssd["read_GBps"] * 1e9,
        ssd_write_Bps=ssd["write_GBps"] * 1e9,
        aes_Bps=aes * 1e9,
        unified=bool(d.get("unified")),
        c_matmul=wf["c_matmul"], c_hadamard=wf["c_hadamard"], c_lookup=wf["c_lookup"],
        c_open=wf["c_open"], c_derive=wf["c_derive"],
        evidence={"measured_constants": ",".join(sorted(measured)) or "none",
                  "time_evidence": "MEASURED"},
    )

# --------------------------------------------------------------- scenarios
SCENARIOS: Dict[str, dict] = {
    # single RTX 4090 runs inference and proving (time-shared), host staging over PCIe
    "ws_4090": {"proof_dev": "gpu_rtx4090", "infer_dev": "gpu_rtx4090", "shared": True,
                "cap_GiB": 20},
    # inference on RTX 4090, proving on RTX 6000 Ada; witnesses cross PCIe via host
    "ws_split": {"proof_dev": "gpu_rtx6000ada", "infer_dev": "gpu_rtx4090", "shared": False,
                 "cap_GiB": 40},
    # edge-class unified-memory SoC budgets (host staging saves nothing there)
    "edge16": {"proof_dev": "edge_unified_16g", "infer_dev": "edge_unified_16g", "shared": True,
               "cap_GiB": 16},
    "edge8": {"proof_dev": "edge_unified_8g", "infer_dev": "edge_unified_8g", "shared": True,
              "cap_GiB": 8},
}

def verify_time(dag, cm) -> float:
    v = cm["verifier"]
    rounds = 0
    opens = 0
    for t in dag.tasks.values():
        if t.kind == "open":
            opens += 1
        else:
            rounds += max(1, int(math.ceil(math.log2(max(t.work, 2)))))
    return rounds * v["per_round_us"] * 1e-6 + opens * v["per_opening_us"] * 1e-6

_NAIVE_CACHE: Dict[tuple, int] = {}

def inference_working_set(dag, key) -> int:
    """Peak device bytes of plain streaming inference (weights + KV + live tensors)."""
    if key not in _NAIVE_CACHE:
        dec = uniform_decisions(dag, "LAYER_STREAM_NAIVE")
        ev = emit_schedule(dag, dec)
        # only the inference prefix matters: stop at the first challenge
        cut = next((i for i, e in enumerate(ev) if e.kind == "CHALLENGE"), len(ev))
        chk = check_schedule(dag, ev[:cut])
        _NAIVE_CACHE[key] = chk.peak_device
    return _NAIVE_CACHE[key]

def inference_reserve(model: str, classes: Sequence[Tuple[int, int]], protocol: str = "GLOBAL") -> int:
    """Largest non-weight inference working set over request classes (bytes)."""
    best = 0
    for Tp, Tg in classes:
        dag = build_dag(model, Tp, Tg, Encoding(protocol=protocol))
        best = max(best, inference_working_set(dag, (model, Tp, Tg, protocol)) - dag.static_bytes())
    return best

def request_plan(model: str, Tp: int, Tg: int, policy: str, scenario: str, cm: dict,
                 protocol: str = "GLOBAL", reserve_bytes: int = 0) -> dict:
    """Per-request plan. ``reserve_bytes`` (CertSched on a shared device): memory kept
    free for one concurrent inference working set, so inference can interleave with a
    proof in progress."""
    sc = SCENARIOS[scenario]
    prof = make_profile(cm, sc["proof_dev"])
    iprof = make_profile(cm, sc["infer_dev"])
    cap = int(sc["cap_GiB"] * GiB)
    host_cap = None if prof.unified else int(cm["host"]["app_budget_GiB"] * GiB)
    ssd_cap = int(cm["storage"]["spill_budget_GB"] * 1e9)
    dag = build_dag(model, Tp, Tg, Encoding(protocol=protocol))
    plan_cap = cap - reserve_bytes if policy == "CERTSCHED" else cap
    r = run_policy(dag, policy, prof, plan_cap, host_cap, ssd_cap)
    if policy == "CERTSCHED" and reserve_bytes and not (r.check.legal and r.check.fits):
        r = run_policy(dag, policy, prof, cap, host_cap, ssd_cap)   # no room to reserve
        plan_cap = cap
    chk = r.check
    static = dag.static_bytes()
    t_inf_proofdev = r.time["by_category_s"]["inference"]
    t_inf = sum(op_time(dag, op, iprof, True) for op in dag.ops.values())
    t_proof = r.time["total_s"] - t_inf_proofdev
    base_w = dag.witness_bytes(("y", "aux", "lk", "token"))
    t_xfer = 0.0 if sc["shared"] else 2.0 * base_w / (iprof.link_Bps or iprof.mem_Bps)
    b = chk.boundary_bytes or {"device": chk.peak_device, "host": 0, "ssd": 0}
    peak_eff = chk.peak_unified if prof.unified else chk.peak_device
    return {
        "model": model, "Tp": Tp, "Tg": Tg, "policy": policy, "scenario": scenario,
        "protocol": protocol, "legal": chk.legal,
        "fits": bool(chk.legal and peak_eff <= cap and (host_cap is None or chk.peak_host <= host_cap)
                     and chk.peak_ssd <= ssd_cap),
        "rules": chk.rule_counts, "plan_cap_bytes": plan_cap, "reserve_bytes": cap - plan_cap,
        "selected": getattr(r, "selected", None),
        "peak_device_bytes": peak_eff, "peak_host_bytes": chk.peak_host, "peak_ssd_bytes": chk.peak_ssd,
        "boundary_device_bytes": b["device"], "boundary_host_bytes": b["host"],
        "boundary_ssd_bytes": b["ssd"], "static_bytes": static,
        "inf_ws_bytes": inference_working_set(dag, (model, Tp, Tg, protocol)),
        "witness_bytes": dag.witness_bytes(), "base_witness_bytes": base_w,
        "t_inf_s_est": t_inf, "t_xfer_s_est": t_xfer, "t_proof_s_est": t_proof,
        "t_overhead_s_est": r.time["overhead_s"], "t_verify_s_est": verify_time(dag, cm),
        "by_category_s_est": r.time["by_category_s"],
        "n_chunks": 8, "unified": prof.unified,
        "spill_w_Bps": prof.spill_w_Bps(), "spill_r_Bps": prof.spill_r_Bps(),
        "link_Bps": prof.link_Bps,
    }

# -------------------------------------------------------------------- DES
@dataclass
class Req:
    rid: int
    arr: float
    cls: int
    Tg: int
    admit: float = math.nan
    inf_start: float = math.nan
    inf_end: float = math.nan
    ready: float = math.nan
    proof_start: float = math.nan
    proof_end: float = math.nan
    cert: float = math.nan
    aborted: str = ""
    blocked: bool = False
    deadline: float = math.inf
    chunks_left: int = 0
    chunk_s: float = 0.0
    host_b: float = 0.0
    ssd_b: float = 0.0
    uni_b: float = 0.0

@dataclass
class SimConfig:
    shared: bool
    unified: bool
    cap: float
    host_cap: float
    ssd_cap: float
    B: float = math.inf                 # proof-debt bound (tokens); inf = no backpressure
    proof_order: str = "FIFO"           # FIFO | EDF
    shared_priority: str = "infer_first"  # infer_first | debt_aware
    slo_factor: float = 3.0

def arrivals(rate_fn: Callable[[float], float], rate_max: float, horizon: float, rng) -> List[float]:
    """Non-homogeneous Poisson arrivals by thinning."""
    t, out = 0.0, []
    while True:
        t += rng.expovariate(rate_max)
        if t > horizon:
            return out
        if rng.random() * rate_max <= rate_fn(t):
            out.append(t)

def simulate(plans: Sequence[dict], arr_times: Sequence[float], cfg: SimConfig, rng,
             series_dt: Optional[float] = None, horizon: Optional[float] = None) -> dict:
    """Run one replication. ``plans`` are request classes drawn uniformly at random."""
    reqs = [Req(i, t, rng.randrange(len(plans)), 0) for i, t in enumerate(arr_times)]
    for r in reqs:
        p = plans[r.cls]
        r.Tg = p["Tg"]
        r.deadline = r.arr + cfg.slo_factor * p["_S_ref"]
    ev: List[Tuple[float, int, str, int]] = []
    seq = [0]

    def push(t, kind, rid):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, rid))

    for r in reqs:
        push(r.arr, "arrive", r.rid)

    admit_q: List[int] = []           # blocked by backpressure (FIFO)
    inf_q: List[int] = []             # admitted, waiting for inference
    xfer_q: List[int] = []
    proof_q: List[Tuple[float, int]] = []   # (key, rid) heap
    reserved = 0.0
    debt = 0.0
    host_used = ssd_used = uni_parked = 0.0
    max_debt = 0.0
    bp_events = 0
    oom = 0
    srv = {"inf": None, "proof": None, "xfer": None}   # rid being served (or ("chunk", rid))
    cur_proof: Optional[int] = None   # proof job in progress (possibly paused on shared server)
    series: List[Tuple[float, float, float, float, int]] = []
    next_sample = [0.0]
    now = 0.0
    max_active_peak = max(p["peak_device_bytes"] for p in plans)

    def sample(t):
        if series_dt is None:
            return
        while next_sample[0] <= t:
            series.append((next_sample[0], debt, host_used + uni_parked, ssd_used,
                           len(admit_q)))
            next_sample[0] += series_dt

    def try_admit():
        nonlocal reserved
        while admit_q:
            r = reqs[admit_q[0]]
            if reserved + r.Tg > cfg.B:
                return
            admit_q.pop(0)
            admit(r)

    def admit(r: Req):
        nonlocal reserved
        p = plans[r.cls]
        if not p["legal"] or not p["fits"]:
            r.aborted = "PLAN_INFEASIBLE" if p["legal"] else "ILLEGAL"
            return
        r.admit = now
        reserved += r.Tg
        inf_q.append(r.rid)

    def park(r: Req):
        """Account witness bytes held between inference end and proof end."""
        nonlocal host_used, ssd_used, uni_parked, oom, reserved, debt
        p = plans[r.cls]
        extra = 0.0
        if cfg.unified:
            want = max(0.0, p["boundary_device_bytes"] + p["boundary_host_bytes"] - p["static_bytes"])
            room = max(0.0, cfg.cap - max_active_peak - uni_parked)
            keep = min(want, room)
            over = want - keep
            r.uni_b = keep
            uni_parked += keep
            ssd_extra = over + p["boundary_ssd_bytes"]
            if over > 0:
                extra += over / p["spill_w_Bps"] + over / p["spill_r_Bps"]
        else:
            want = max(0.0, p["boundary_device_bytes"] - p["static_bytes"]) + p["peak_host_bytes"]
            room = max(0.0, cfg.host_cap - host_used)
            keep = min(want, room)
            over = want - keep
            r.host_b = keep
            host_used += keep
            ssd_extra = over + p["peak_ssd_bytes"]
            if over > 0:
                extra += over / p["spill_w_Bps"] + over / p["spill_r_Bps"]
        if ssd_used + ssd_extra > cfg.ssd_cap:
            host_used -= r.host_b
            uni_parked -= r.uni_b
            r.host_b = r.uni_b = 0.0
            r.aborted = "OOM_SPILL_BUDGET"
            oom += 1
            return False
        r.ssd_b = ssd_extra
        ssd_used += ssd_extra
        total = p["t_proof_s_est"] + extra
        r.chunks_left = p["n_chunks"]
        r.chunk_s = total / p["n_chunks"]
        return True

    def unpark(r: Req):
        nonlocal host_used, ssd_used, uni_parked
        host_used -= r.host_b
        ssd_used -= r.ssd_b
        uni_parked -= r.uni_b
        r.host_b = r.ssd_b = r.uni_b = 0.0

    def proof_key(r: Req):
        return r.deadline if cfg.proof_order == "EDF" else r.ready

    def dispatch():
        nonlocal cur_proof
        # transfer engine (split only)
        if not cfg.shared and srv["xfer"] is None and xfer_q:
            rid = xfer_q.pop(0)
            srv["xfer"] = rid
            push(now + plans[reqs[rid].cls]["t_xfer_s_est"], "xfer_done", rid)
        if cfg.shared:
            if srv["inf"] is not None:
                return
            proof_pending = cur_proof is not None or bool(proof_q)
            # debt-aware: once half the debt bound is outstanding, proofs go first
            thresh = cfg.B / 2 if cfg.B < math.inf else 32.0
            want_proof_first = cfg.shared_priority == "debt_aware" and debt >= thresh
            can_inf = bool(inf_q)
            if can_inf and cur_proof is not None:
                p_new = plans[reqs[inf_q[0]].cls]
                p_cur = plans[reqs[cur_proof].cls]
                if p_cur["peak_device_bytes"] + p_new["inf_ws_bytes"] - p_new["static_bytes"] > cfg.cap:
                    can_inf = False       # memory-aware interleaving
            if can_inf and not (want_proof_first and proof_pending):
                start_inf()
            elif proof_pending:
                run_chunk()
            elif can_inf:
                start_inf()
        else:
            if srv["inf"] is None and inf_q:
                start_inf()
            if srv["proof"] is None and (cur_proof is not None or proof_q):
                run_chunk()

    def start_inf():
        rid = inf_q.pop(0)
        r = reqs[rid]
        r.inf_start = now
        srv["inf"] = rid
        push(now + plans[r.cls]["t_inf_s_est"], "inf_done", rid)

    def run_chunk():
        nonlocal cur_proof
        if cur_proof is None:
            _, rid = heapq.heappop(proof_q)
            cur_proof = rid
            reqs[rid].proof_start = now
        r = reqs[cur_proof]
        key = "inf" if cfg.shared else "proof"
        srv[key] = ("chunk", cur_proof)
        push(now + r.chunk_s, "chunk_done", cur_proof)

    while ev:
        t, _, kind, rid = heapq.heappop(ev)
        sample(t)
        now = t
        r = reqs[rid]
        if kind == "arrive":
            if cfg.B < math.inf and (admit_q or reserved + r.Tg > cfg.B):
                r.blocked = True
                bp_events += 1
                admit_q.append(rid)
            else:
                admit(r)
        elif kind == "inf_done":
            srv["inf"] = None
            r.inf_end = now
            debt += r.Tg
            max_debt = max(max_debt, debt)
            if not park(r):
                debt -= r.Tg
                reserved -= r.Tg
                try_admit()
            elif cfg.shared:
                r.ready = now
                heapq.heappush(proof_q, (proof_key(r), rid))
            else:
                xfer_q.append(rid)
        elif kind == "xfer_done":
            srv["xfer"] = None
            r.ready = now
            heapq.heappush(proof_q, (proof_key(r), rid))
        elif kind == "chunk_done":
            key = "inf" if cfg.shared else "proof"
            srv[key] = None
            r.chunks_left -= 1
            if r.chunks_left <= 0:
                cur_proof = None
                r.proof_end = now
                unpark(r)
                push(now + plans[r.cls]["t_verify_s_est"], "verified", rid)
        elif kind == "verified":
            r.cert = now
            debt -= r.Tg
            reserved -= r.Tg
            try_admit()
        dispatch()
    sample(horizon if horizon is not None else now)
    return {"reqs": reqs, "max_debt": max_debt, "bp_events": bp_events, "oom": oom,
            "series": series, "end": now}

def _pct(xs, q):
    if not xs:
        return math.nan
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def run_metrics(out: dict, warmup_frac: float = 0.1) -> dict:
    reqs = out["reqs"]
    n0 = int(len(reqs) * warmup_frac)
    rs = reqs[n0:]
    done = [r for r in rs if not r.aborted and not math.isnan(r.cert)]
    prov = [r.inf_end - r.arr for r in rs if not r.aborted and not math.isnan(r.inf_end)]
    certl = [r.cert - r.arr for r in done]
    gap = [r.cert - r.inf_end for r in done]
    aborted = sum(1 for r in rs if r.aborted)
    unfinished = sum(1 for r in rs if not r.aborted and math.isnan(r.cert))
    miss = sum(1 for r in done if r.cert > r.deadline)
    t0 = rs[0].arr if rs else 0.0
    t1 = max((r.cert for r in done), default=t0)
    toks = sum(r.Tg for r in done)
    return {
        "n": len(rs), "completed": len(done), "aborted": aborted, "unfinished": unfinished,
        "abort_rate": aborted / max(len(rs), 1),
        "prov_p50": _pct(prov, 50), "prov_p95": _pct(prov, 95), "prov_p99": _pct(prov, 99),
        "cert_p50": _pct(certl, 50), "cert_p95": _pct(certl, 95), "cert_p99": _pct(certl, 99),
        "uncert_gap_p95": _pct(gap, 95),
        "deadline_miss_rate": miss / max(len(done), 1),
        "throughput_req_s": len(done) / max(t1 - t0, 1e-9),
        "throughput_tok_s": toks / max(t1 - t0, 1e-9),
        "max_debt_tokens": out["max_debt"], "bp_events": out["bp_events"], "oom_events": out["oom"],
        "bp_fraction": sum(1 for r in rs if r.blocked) / max(len(rs), 1),
    }

def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 12345,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap CI over replications (resampling unit: seed)."""
    import numpy as np
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))],
                 dtype=float)
    if v.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
