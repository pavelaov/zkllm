"""Shared helpers for the CertSched (C3) runners: paths, run metadata, writers."""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

RAW = os.path.join(ROOT, "results", "raw", "sched")
PROC = os.path.join(ROOT, "results", "processed")
GiB = 1 << 30

def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

def meta(command: str, cost_model: dict | None = None) -> dict:
    m = {
        "command": command,
        "timestamp_utc": now_utc(),
        "host": {"node": platform.node(), "os": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "tools": {"python": sys.version.split()[0]},
        "code": ["src/edgecert/runtime/dag.py", "src/edgecert/runtime/scheduler.py",
                 "src/edgecert/runtime/simulate.py"],
    }
    try:
        import numpy
        m["tools"]["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover
        pass
    if cost_model is not None:
        m["cost_model"] = {"path": "configs/cost_model.yaml",
                           "generated_utc": cost_model.get("meta", {}).get("generated_utc"),
                           "inputs_found": cost_model.get("meta", {}).get("inputs_found")}
    return m

def inputs(cm: dict) -> list:
    """Which cost-model inputs are listed."""
    out = []
    for dev, d in cm.get("devices", {}).items():
        ev = d.get("evidence", {})
        for k, v in ev.items():
            if isinstance(v, str) and v.startswith("SUCCESS"):
                out.append(f"devices.{dev}.{k}")
    for name, L in cm.get("links", {}).items():
        if "SUCCESS" in str(L.get("evidence", "")):
            out.append(f"links.{name} (sustained part)")
    out += ["work_factors (all)", "verifier (all)"]
    return out

def write_json(name: str, obj: dict) -> str:
    os.makedirs(RAW, exist_ok=True)
    p = os.path.join(RAW, name)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_clean(obj), f, indent=1, default=_default, allow_nan=False)
    return p

def _clean(o):
    """NaN/inf -> None (strict JSON); tuples -> lists."""
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o

def _default(o):
    if isinstance(o, float) and (o != o):
        return None
    if isinstance(o, set):
        return sorted(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)

def write_csv(name: str, rows: list, fields: list | None = None) -> str:
    os.makedirs(PROC, exist_ok=True)
    p = os.path.join(PROC, name)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})
    return p

def _fmt(v):
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.6g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True)
    return v
