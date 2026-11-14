"""Shared helpers for the EDGE-CERT figure / table / number pipeline (bench/analysis).
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
RAW = RES / "raw"
PROC = RES / "processed"
CHARTS = REPO / "paper" / "figures" / "charts"
GEN = REPO / "paper" / "generated"
TABLES = REPO / "paper" / "tables"
TEMPLATES = REPO / "paper" / "templates"

if str(TEMPLATES) not in sys.path:
    sys.path.insert(0, str(TEMPLATES))

# ------------------------------------------------------------------ constants (ANALYTIC)
P_GOLDILOCKS = 2**64 - 2**32 + 1
P_BABYBEAR = 2**31 - 2**27 + 1
# log2 of the largest |v| that a signed lift into F_p decodes uniquely (2B < p)
HALF_BITS_GL = math.log2((P_GOLDILOCKS - 1) / 2)
HALF_BITS_BB = math.log2((P_BABYBEAR - 1) / 2)

EVIDENCE = ("SUCCESS", "POST_ROUTE", "ABORT", "ANALYTIC", "REPORTED")

MODELS = [
    dict(key="gpt2_124m", sched="gpt2-124m", label="GPT-2 124M", short="GPT-2",
         aliases=("gpt2_124m", "gpt2-124m", "gpt2")),
    dict(key="qwen3_0.6b", sched="qwen3-0.6b", label="Qwen3-0.6B", short="Qwen3",
         aliases=("qwen3_0.6b", "qwen3-0.6b", "qwen3_0_6b", "qwen3")),
    dict(key="gemma3_1b", sched="gemma3-1b", label="Gemma 3 1B", short="Gemma 3",
         aliases=("gemma3_1b", "gemma3-1b", "gemma3", "gemma")),
    dict(key="llama3_8b", sched="llama3-8b", label="Llama-3 8B", short="8B",
         aliases=("llama3_8b", "llama3-8b", "llama3")),
]
T0_MODELS = [dict(key="t0_gpt", label="T0-gpt", short="T0-gpt"),
             dict(key="t0_llama", label="T0-llama", short="T0-llama")]
MODEL_BY_KEY = {m["key"]: m for m in MODELS + T0_MODELS}
MODEL_BY_SCHED = {m["sched"]: m for m in MODELS}

# ------------------------------------------------------------------ run bookkeeping
PENDING: list[dict] = []
INPUTS: dict[str, dict] = {}
WARNINGS: list[str] = []

def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")

def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)

def host_info() -> dict:
    return {"platform": platform.platform(), "python": sys.version.split()[0],
            "node": platform.node(), "cwd": rel(os.getcwd())}

def pending(name: str, missing, why: str | None = None) -> None:
    """Print the canonical PENDING line and record it (never raises)."""
    miss = missing if isinstance(missing, str) else ", ".join(missing)
    line = f"PENDING {name}: missing {miss}" + (f" ({why})" if why else "")
    print(line)
    PENDING.append({"name": name, "missing": miss, "why": why})

def warn(msg: str) -> None:
    print("WARN " + msg)
    WARNINGS.append(msg)

def _meta_of(obj) -> dict:
    meta = {}
    if isinstance(obj, dict):
        for k in ("status", "evidence_class", "timestamp_utc", "timestamp", "timestamp_utc_start",
                  "command", "contended"):
            if k in obj and not isinstance(obj[k], (dict, list)):
                meta[k] = obj[k]
    return meta

def load_json(path, required_by: str | None = None):
    """Load results JSON (path relative to the repo or absolute).  Missing -> None (+PENDING)."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    key = rel(p)
    if not p.exists():
        INPUTS.setdefault(key, {"exists": False})
        if required_by:
            pending(required_by, key)
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError) as e:  # half-written file from a concurrent run
        INPUTS[key] = {"exists": True, "error": repr(e)}
        if required_by:
            pending(required_by, key, "unreadable: " + repr(e)[:80])
        return None
    meta = {"exists": True, "mtime": _dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone()
            .isoformat(timespec="seconds"), "bytes": p.stat().st_size}
    meta.update(_meta_of(obj))
    INPUTS[key] = meta
    return obj

def load_csv(path, required_by: str | None = None):
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    key = rel(p)
    if not p.exists():
        INPUTS.setdefault(key, {"exists": False})
        if required_by:
            pending(required_by, key)
        return None
    with open(p, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    INPUTS[key] = {"exists": True, "rows": len(rows),
                   "mtime": _dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone()
                   .isoformat(timespec="seconds")}
    return rows

def glob_results(pattern: str) -> list[Path]:
    return sorted(RAW.glob(pattern))

def fnum(s, default=None):
    if s is None:
        return default
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if s == "" or s.lower() in ("none", "nan", "null"):
        return default
    try:
        return float(s)
    except ValueError:
        return default

def get(d, dotted: str, default=None):
    """Nested dict access with a dotted path ('a.b.c'); keys containing dots are tried whole."""
    cur = d
    parts = dotted.split(".")
    i = 0
    while i < len(parts):
        if not isinstance(cur, dict):
            return default
        # greedy: allow keys that contain dots (e.g. 'qwen3_0.6b')
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if k in cur:
                cur = cur[k]
                i = j
                break
        else:
            return default
    return cur

# ------------------------------------------------------------------ contracts (C2)
def _contract_schema_ok(obj) -> bool:
    return bool(get(obj, "variants.W8A8.workloads")) and bool(get(obj, "variants.W8A8.max_certified_accumulator"))

def _adapt_realmodel_contracts(obj, policy):
    """Present a results/raw/realmodels/<model>_contracts.json record (realmodel_contracts schema,
    variants named e.g. 'W8A8_sq_sink') in the contracts_eval schema used by contract_row():
    variants.W8A8 := the paper_primary variant (W8A8 number formats).  Real-weight fields: lookups,
    reductions, table lookups, accumulator / constraint-term bounds, static checks, K-split tiles,
    op counts.  The real record has no 'witness bytes excluding tables' breakdown, so those three
    fields are taken from the policy-bound record of the same model (flagged 'witness_source')."""
    prim = get(obj, "paper_primary.variant") if isinstance(obj.get("paper_primary"), dict) else None
    if not prim:
        prim = next((k for k, v in (obj.get("variants") or {}).items()
                     if isinstance(v, dict) and v.get("paper_primary")), None)
    v = get(obj, f"variants.{prim}") if prim else None
    if not v or not v.get("workloads"):
        return None
    wls, tiles = {}, {}
    for wl, w in v["workloads"].items():
        w2 = dict(w)
        pw = get(policy, f"variants.W8A8.workloads.{wl}.witness_bytes_per_token_excl_tables") if policy else None
        if pw:
            w2["witness_bytes_per_token_excl_tables"] = pw
        wls[wl] = w2
        ks = get(w, "ksplit.cpu_int32")
        if ks:
            tiles[wl] = {"cpu_int32": ks}
    sc = v.get("static_check") or {}
    bb = dict(sc.get("babybear_31bit") or {})
    if "violations" in bb and "violations_count" not in bb:
        bb["violations_count"] = bb["violations"]
    ad = {"num_ops": v.get("num_ops"), "num_distinct_contracts": v.get("num_contracts_distinct"),
          "static_check": {"goldilocks": sc.get("goldilocks"), "babybear_31bit": bb},
          "max_certified_accumulator": v.get("max_certified_accumulator"),
          "max_certified_constraint_term": v.get("max_certified_constraint_term"),
          "workloads": wls, "tiles": tiles, "source_variant": prim,
          "relation_replay": v.get("relation_replay"), "dynamic_check_summary": v.get("dynamic_check_summary"),
          "witness_source": "policy_bounds (results/raw/contracts)" if policy else None}
    out = {"status": obj.get("status"), "evidence_class": "ANALYTIC", "record_type": "realmodel_contracts_adapted",
           "model": obj.get("model"), "config": obj.get("config"), "variants": {"W8A8": ad},
           "source_variant": prim}
    if policy and policy.get("model_shape"):
        out["model_shape"] = policy["model_shape"]
    return out

def load_contracts(key: str, prefer_real: bool = True):
    """Contract record for one model.

    Preference: results/raw/realmodels/<alias>_contracts.json (real weights; the paper_primary W8A8
    variant, adapted to the contracts_eval schema), else results/raw/contracts/contracts_<key>.json
    (policy bounds).  Returns (obj, path, source) with source in {'real_weights', 'policy_bounds'};
    (None, None, None) when neither exists.
    """
    m = MODEL_BY_KEY.get(key, {"aliases": (key,)})
    pp = RAW / "contracts" / f"contracts_{key}.json"
    policy = load_json(pp)
    if prefer_real:
        cands = []
        for a in m.get("aliases", (key,)):
            cands += [p for p in glob_results(f"realmodels/*{a}*contracts*.json") if p not in cands]
        for p in cands:
            obj = load_json(p)
            if obj is None:
                continue
            if _contract_schema_ok(obj):
                return obj, p, "real_weights"
            ad = _adapt_realmodel_contracts(obj, policy)
            if ad is not None:
                return ad, p, "real_weights"
            warn(f"{rel(p)}: neither contracts_eval nor realmodel_contracts schema; "
                 f"falling back to results/raw/contracts for {key}")
    if policy is None:
        return None, None, None
    return policy, pp, "policy_bounds"

def shape_of(obj) -> dict | None:
    """Normalise model_shape (real shapes) or config (T0) into one dict."""
    if obj is None:
        return None
    s = obj.get("model_shape")
    if s:
        mlp = str(s.get("mlp", "gelu")).lower()
        return dict(name=s.get("name"), L=int(s["n_layers"]), d=int(s["d_model"]), H=int(s["n_heads"]),
                    Hkv=int(s.get("n_kv_heads") or s["n_heads"]), dh=int(s["head_dim"]), ff=int(s["d_ff"]),
                    V=int(s["vocab"]), gated=mlp in ("swiglu", "geglu", "glu", "gated"),
                    window=s.get("sliding_window"), global_every=s.get("global_every"),
                    params=get(s, "computed_params.total") or s.get("reported_params"))
    c = obj.get("config")
    if c and "d" in c:
        fam = str(c.get("family", "gpt")).lower()
        dh = int(c.get("dh") or c["d"] // c["n_heads"])
        # gated (SwiGLU / GeGLU) MLP families: Llama, Qwen, Gemma, Mistral; GPT-2 style is not gated
        gated = fam.startswith(("llama", "qwen", "gemma", "mistral"))
        return dict(name=c.get("name"), L=int(c["n_layers"]), d=int(c["d"]), H=int(c["n_heads"]),
                    Hkv=int(c.get("n_kv_heads") or c["n_heads"]), dh=dh, ff=int(c["ff"]), V=int(c["vocab"]),
                    gated=gated, window=None, global_every=None, params=None)
    return None

def macs_per_token(sh: dict, workload: str) -> dict:
    """ANALYTIC multiply-accumulate count per token of the integer forward pass.

    Linear layers: Q,K,V,O projections + MLP (2 matrices, or 3 when gated).
    Attention: scores + P.V = 2 H dh keys per token and layer (causal; Gemma-style sliding-window
    layers see at most `window` keys; one global layer every `global_every`).
    LM head: V d at emitting positions only (prefill: once per prompt; decode: every token).
    workload: 'prefill_<T>' (average over the T prompt tokens) or 'decode_pos_<p>' (one new token
    at position p).  Norms, softmax, activations and requantisation are counted as lookups /
    range checks by the contract accounting, not here.
    """
    L, d, H, Hkv, dh, ff, V = sh["L"], sh["d"], sh["H"], sh["Hkv"], sh["dh"], sh["ff"], sh["V"]
    lin = d * (H * dh + 2 * Hkv * dh) + H * dh * d + (3 if sh["gated"] else 2) * d * ff
    kind, _, n = workload.rpartition("_")
    n = int(n)

    def keys(i, layer):
        k = i + 1
        w = sh.get("window")
        ge = sh.get("global_every")
        if w and ge and (layer + 1) % int(ge) != 0:
            k = min(k, int(w))
        elif w and not ge:
            k = min(k, int(w))
        return k

    if kind.startswith("prefill"):
        T = n
        attn = sum(2 * H * dh * keys(i, l) for l in range(L) for i in range(T)) / T
        head = V * d / T
    else:
        p = n
        attn = sum(2 * H * dh * keys(p, l) for l in range(L))
        head = V * d
    return {"linear": L * lin, "attention": attn, "lm_head": head, "total": L * lin + attn + head}

def contract_row(obj, variant: str, workload: str) -> dict | None:
    """Flattened per-(variant, workload) contract metrics used by figures, tables and numbers."""
    v = get(obj, f"variants.{variant}")
    if not v:
        return None
    w = get(v, f"workloads.{workload}")
    if not w:
        return None
    rc = w.get("rc_lookups_per_token", {})
    wb = w.get("witness_bytes_per_token_excl_tables", {})
    tl = w.get("table_lookups_per_token", {})
    acc = v.get("max_certified_accumulator", {})
    term = v.get("max_certified_constraint_term", {})
    tiles = get(v, f"tiles.{workload}.cpu_int32") or {}
    acc_bits = math.log2(acc["bound"]) if acc.get("bound") else None
    term_bits = math.log2(term["bound"]) if term.get("bound") else None
    return {
        "naive64": rc.get("naive_field64"), "naive32": rc.get("naive_int32"),
        "tc": rc.get("tilecontract_opt"), "tc_b": rc.get("tilecontract_opt_b"),
        "table_lookups": sum(x for x in tl.values() if isinstance(x, (int, float))) if tl else None,
        "wit_gl_B": wb.get("goldilocks_8B"), "wit_254_B": wb.get("field254_32B"),
        "wit_naive_gl_B": wb.get("naive_field64_goldilocks_8B"),
        "red64": w.get("reduction_vs_naive_field64"), "red32": w.get("reduction_vs_naive_int32"),
        "acc_bound": acc.get("bound"), "acc_bits": acc_bits, "acc_op": acc.get("op"),
        "acc_head_gl": acc.get("headroom_bits_goldilocks"),
        "acc_head_bb": (HALF_BITS_BB - acc_bits) if acc_bits is not None else None,
        "term_bound": term.get("bound"), "term_bits": term_bits, "term_op": term.get("op"),
        "term_head_gl": term.get("headroom_bits_goldilocks"),
        "term_head_bb": (HALF_BITS_BB - term_bits) if term_bits is not None else None,
        "tiles": tiles.get("tile_instances"), "ksplit_frac_int32": tiles.get("fraction_needing_ksplit"),
        "num_ops": v.get("num_ops"), "distinct": v.get("num_distinct_contracts"),
        "gl_pass": get(v, "static_check.goldilocks.pass"), "bb_pass": get(v, "static_check.babybear_31bit.pass"),
        "bb_violations": get(v, "static_check.babybear_31bit.violations_count"),
    }

# ------------------------------------------------------------------ kernels (C4)
def cpu_records(obj, compiler="gcc", variant="fused"):
    """{(n, threads): record} of verified CPU kernel runs."""
    out = {}
    for r in (obj or {}).get("results", []):
        if r.get("compiler") != compiler or r.get("variant") != variant:
            continue
        if r.get("status") != "SUCCESS" or not r.get("verify_ok"):
            continue
        if r.get("match") is False:
            continue
        out[(int(r["n"]), int(r["threads"]))] = r
    return out

def gpu_devices(obj):
    """[(device_name, contended, {(n, variant): record})] of verified GPU kernel runs."""
    out = []
    for dev in (obj or {}).get("devices", []):
        recs = {}
        for r in dev.get("results", []):
            if not r.get("verify_ok") or r.get("match") is False:
                continue
            recs[(int(r["n"]), r["variant"])] = r
        name = get(dev, "device.name", "GPU").replace("NVIDIA ", "").replace("GeForce ", "")
        name = name.replace(" Generation", "")
        out.append((name, bool(dev.get("contended", obj.get("contended"))), recs))
    return out

def sumfold_cycles(n: int, L: int) -> int:
    """ANALYTIC closed form of the RTL AUTO job (docs/notes/fpga_rtl.md sec. 4; exact in xsim):
    1.5 N/L - 2 + R (60 + log2 L) + 30 + 2L, R = n - log2 L."""
    N = 1 << n
    lg = int(round(math.log2(L)))
    R = n - lg
    return int(1.5 * N / L - 2 + R * (60 + lg) + 30 + 2 * L)

def sim_cycles(sim, n: int, L: int):
    """Simulated CYC_TOTAL for an AUTO case (bit-exact run), else None."""
    for c in (sim or {}).get("cases", []):
        if c.get("n") == n and c.get("lanes") == L and c.get("mode") == "auto" and c.get("match"):
            return int(c["cycles_total"])
    return None

# ------------------------------------------------------------------ GPT-2 transparent linear-layer prover
def _gpt2_row_from_part(d: dict, src: str) -> dict | None:
    if not isinstance(d, dict) or d.get("status") != "SUCCESS" or not d.get("total_prefill"):
        return None
    tot = d["total_prefill"]
    med = lambda k: (tot.get(k) or {}).get("median")  # noqa: E731
    fold = med("fold_s")
    if fold is None and med("fold_a_s") is not None:
        fold = med("fold_a_s") + (med("fold_w_s") or 0.0)
    be = d.get("backend") or {}
    dev = (get(be, "device.name") or "GPU") if be.get("backend") == "gpu" else f"CPU {be.get('threads')} thr."
    dev = dev.replace("NVIDIA ", "").replace("GeForce ", "").replace(" Generation", "")
    ver = get(d, "verifier.total_prefill.v_total_s.median")
    return {"device": dev, "backend_config": d.get("backend_config"), "T": d.get("T"),
            "wall": med("wall_s"), "bind": med("bind_s"), "fold": fold, "sumcheck": med("sumcheck_s"),
            "core": med("core_s"), "verify": ver, "verify_threads": get(d, "verifier.threads"),
            "proof_bytes": tot.get("proof_bytes"), "peak_ws": get(d, "memory_peak.peak_working_set_bytes"),
            "contended": bool(d.get("contended")), "verifier_all_accept": d.get("verifier_all_accept"),
            "proofs": get(d, "counts.proofs"), "source": src, "evidence_class": d.get("evidence_class", "SUCCESS")}

def _gpt2_row_from_compact(c: dict, T, be, src) -> dict | None:
    if not isinstance(c, dict) or c.get("status") != "SUCCESS":
        return None
    dev = str(c.get("device") or be).replace("NVIDIA ", "").replace("GeForce ", "").replace(" Generation", "")
    dev = dev.replace(" thread(s)", " thr.")
    wall = c.get("prover_wall_s_median")
    parts = [c.get("prover_bind_s_median"), c.get("prover_fold_s_median"), c.get("prover_sumcheck_s_median")]
        # multi-threaded CPU runs report bind (and at large T also fold/core) summed over concurrent workers: the stage split then exceeds the wall time and must not be drawn as a stacked wall-time split
    summed = wall is not None and all(p is not None for p in parts) and sum(parts) > 1.02 * wall
    ver = c.get("verifier_total_s_median")
    if ver is None:
        ver = c.get("verifier_wall_s_median")
    return {"device": dev, "backend_config": be, "T": int(T), "wall": wall,
            "bind": parts[0], "fold": parts[1], "sumcheck": parts[2], "core": c.get("prover_core_s_median"),
            "stage_split_summed": summed,
            "verify": ver, "verify_threads": c.get("verifier_threads"),
            "proof_bytes": c.get("proof_bytes_total"), "peak_ws": c.get("peak_working_set_bytes"),
            "contended": bool(c.get("contended")), "verifier_all_accept": c.get("verifier_all_accept"),
            "wall_per_token": c.get("wall_per_token_s"),
            "proofs": None, "source": src, "evidence_class": "SUCCESS"}

def load_gpt2prove() -> tuple[dict, list]:
    """{T: {backend_config: row}} of GPT-2 linear-layer transparent proofs.

    Preference: summary_final.json > summary.json > gpt2_prove_T*.json > parts/T*_*.json (a run in progress
    writes parts first).  Each row: prover wall and its bind / fold / sumcheck split, verifier time, proof
    bytes, peak working set, contention flag and source file.
    """
    out, files = {}, []
    base = RAW / "gpt2prove"
    for name in ("summary_final.json", "summary.json"):
        s = load_json(base / name)
        if s and s.get("rows"):
            for T, r in s["rows"].items():
                for be, c in (r.get("configs") or {}).items():
                    row = _gpt2_row_from_compact(c, T, be, rel(base / name))
                    if row:
                        out.setdefault(int(T), {})[be] = row
            files.append(rel(base / name))
            if out:
                return out, files
    for p in sorted(base.glob("gpt2_prove_T*.json")):
        if "_quick" in p.name:
            continue
        s = load_json(p)
        for be, d in ((s or {}).get("configs") or {}).items():
            row = _gpt2_row_from_part(d, rel(p))
            if row:
                out.setdefault(int(row["T"]), {})[be] = row
                if rel(p) not in files:
                    files.append(rel(p))
    if out:
        return out, files
    for p in sorted((base / "parts").glob("T*_*.json")):
        if "_quick" in p.name:
            continue
        d = load_json(p)
        row = _gpt2_row_from_part(d, rel(p))
        if row:
            out.setdefault(int(row["T"]), {})[row["backend_config"] or p.stem.split("_", 1)[1]] = row
            files.append(rel(p))
    return out, files

# ------------------------------------------------------------------ idleness evidence
def idle_verified(subdir: str) -> tuple[bool, list[str]]:
    """True when results/raw/<subdir>/idle_probes/*.json exist and every probe reports idle and not contended.

    The kernel and GPT-2 runners set `contended` / `host_busy` from snapshots that also catch the runner's
    own load (docs/notes/kernels.md sec. 9.1, gpt2prove.md sec. 11).  The independent idle probes taken
    before and after each final benchmark are the idleness evidence; the runners' raw flags are still
    reported (charts_report.json, table captions).
    """
    files = sorted((RAW / subdir / "idle_probes").glob("*.json"))
    ok = bool(files)
    for p in files:
        o = load_json(p)
        if not o or not o.get("idle") or o.get("contended"):
            ok = False
    return ok, [rel(p) for p in files]

# ------------------------------------------------------------------ FPGA on-board campaign 
def load_board():
    """(summary, campaign) of results/raw/fpga_board/{summary,campaign}.json (v2 campaign) or (None, None)."""
    s = load_json(RAW / "fpga_board" / "summary.json")
    c = load_json(RAW / "fpga_board" / "campaign.json")
    if s and s.get("rtl_version") and "v1" in str(s.get("rtl_version")):
        warn("fpga_board/summary.json is a v1 record: ignored (v1 is functionally incorrect on board)")
        return None, c
    return s, c

def board_configs(summary, mode: str | None = None) -> list[dict]:
    """Bit-exact, verified on-board configurations of the summary (optionally one mode)."""
    out = []
    for c in (summary or {}).get("configs", []):
        if c.get("evidence_class") != "SUCCESS" or not c.get("all_match") or not c.get("all_transcripts_verified"):
            continue
        if mode and c.get("mode") != mode:
            continue
        out.append(c)
    return out

def board_link(summary):
    """Primary JTAG-to-AXI link-rate run (lower host load) and all runs."""
    lr = (summary or {}).get("link_rate") or {}
    runs = lr.get("runs") or []
    prim = next((r for r in runs if r.get("step") == lr.get("primary")), runs[-1] if runs else None)
    return prim, runs

LINK_METHODS = ("write_create_run", "read_create_run", "write_run_precreated", "read_run_precreated",
                "write_queued", "read_queued")

def sysmon_values(campaign) -> dict:
    """{'TEMPERATURE': [..], 'VCCINT': [..], ...} over every SYSMON sample of the campaign (step records and
    thermal loops)."""
    vals: dict[str, list[float]] = {}
    seen = set()
    recs = [(r.get("step"), r) for r in (campaign or {}).get("sysmon_samples", [])] + \
        [(l_.get("step"), l_.get("sysmon") or {}) for l_ in (campaign or {}).get("sysmon_loops", [])]
    for step, r in recs:
        for k, v in r.items():
            q = k.rsplit("_", 1)[-1]
            if q not in ("TEMPERATURE", "VCCINT", "VCCAUX", "VCCBRAM"):
                continue
            if (step, k) in seen:          # a loop's START/END sample also appears in sysmon_samples
                continue
            seen.add((step, k))
            x = fnum(v)
            if x is not None:
                vals.setdefault(q, []).append(x)
    return vals

def impl_runs(runs, version: str = "v2"):
    """Routed implementation runs of one RTL version, keyed by (lanes, MHz)."""
    by = {}
    for r in runs or []:
        if version not in str(r.get("rtl_version", "")):
            continue
        k = (r["lanes"], r["target_mhz"])
        routed = r.get("evidence_class") == "POST_ROUTE" and r.get("utilization")
        if k not in by or (routed and not by[k].get("utilization")):
            by[k] = r
    return by

def board_runs_by_build() -> dict:
    """{build-directory name: [single-run board records]} from results/raw/fpga_board/*.json.

    The on-board status of an implementation run is taken from the board records themselves (the bitstream
    each one programmed, from its `bitstream` field or the `--bit fpga/build/<run>/...` argument of its
    invocation), not from the `board_status` field of fpga_impl/runs.json: a bitstream that never ran on the
    board has no record here and is reported as not run (fixer, 2026-09-29).
    """
    import re
    out = {}
    for p in sorted((RAW / "fpga_board").glob("*.json")):
        o = load_json(p)
        if not isinstance(o, dict) or o.get("record") != "sumfold_board_test":
            continue
        bit = str(o.get("bitstream") or "")
        if not bit:
            m = re.search(r"--bit\s+(\S+)", str(o.get("invocation") or ""))
            bit = m.group(1) if m else ""
        m = re.search(r"fpga[/\\]build[/\\]([^/\\]+)[/\\]", bit.replace('"', ""))
        if m:
            out.setdefault(m.group(1), []).append(o)
    return out

def v1_board_outcomes(runs) -> dict:
    """{(lanes, MHz): 'VERIFY_FAILED' | 'VERIFIED' | 'RUN_ON_BOARD'} for the routed v1 runs."""
    recs = board_runs_by_build()
    out = {}
    for k, r in impl_runs(runs, "v1").items():
        rs = recs.get(r.get("run"), [])
        if not rs:
            out[k] = "RUN_ON_BOARD"
        elif any(x.get("status") == "VERIFY_FAILED" or x.get("transcript_verified") is False for x in rs):
            out[k] = "VERIFY_FAILED"
        else:
            out[k] = "VERIFIED"
        if (out[k] == "RUN_ON_BOARD") != (r.get("board_status") in (None, "RUN_ON_BOARD")):
            warn(f"fpga_impl/runs.json {r.get('run')}: board_status={r.get('board_status')!r} but board records "
                 f"say {out[k]} (the board records are used)")
    return out

def dsp_per_multiplier(run) -> int | None:
    """DSP48E2 per GL multiplier as stated in the run's RTL description ('16-DSP48E2', '12-DSP48E2')."""
    import re
    m = re.search(r"(\d+)-DSP48E2", str(get(run, "rtl.gl_mul") or ""))
    return int(m.group(1)) if m else None

# ------------------------------------------------------------------ real models (results/raw/realmodels)
REALMODEL_KEYS = {"gpt2": "gpt2_124m", "qwen3_0.6b": "qwen3_0.6b"}

def load_realmodels():
    """results/raw/realmodels/summary.json -> {model_key: summary block}, path; ({}, None) if missing."""
    p = RAW / "realmodels" / "summary.json"
    s = load_json(p)
    out = {}
    for k, v in ((s or {}).get("models") or {}).items():
        out[REALMODEL_KEYS.get(k, k)] = v
    return out, (rel(p) if s else None), s

# ------------------------------------------------------------------ formatting
def fmt_int(x) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")

def fmt(x, nd=1) -> str:
    if x is None:
        return "--"
    return f"{x:.{nd}f}"

def human_bytes(b: float) -> str:
    for unit, k in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if b >= k:
            return f"{b / k:.3g} {unit}"
    return f"{b:.0f} B"

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=str)
        f.write("\n")

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
