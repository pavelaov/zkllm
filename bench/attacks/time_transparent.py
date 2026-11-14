#!/usr/bin/env python
"""Timing sweep of the TRANSPARENT_REFERENCE prover and independent verifier (MEASURED, pure Python).

Sweeps hidden width and generated length on the toy window-MLP LM (EOS disabled so every run generates exactly T tokens).  Reports wall time of device.prove and Verifier.verify, the canonical-JSON proof size, and the exact byte count of the sumcheck part (round messages + final values, 16 B per Ext element), which is the only part a succinct backend would keep; the rest is revealed witness.

Usage (repository root, PYTHONPATH=src):
    python bench/attacks/time_transparent.py [--reps 5] [--warmup 1]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edgecert.crypto.ed25519 import SigningKey  # noqa: E402
from edgecert.runtime.manifest import TrustPolicy, canonical_json, sign_manifest  # noqa: E402
from edgecert.runtime.session import CertificationSession, State, new_request  # noqa: E402
from edgecert.verifier.transparent_verifier import Verifier  # noqa: E402
from edgecert.prover_adapters.toy_model import ToyConfig, build_model, manifest_body  # noqa: E402
from edgecert.prover_adapters.transparent import SumFoldLaneEmulator, TransparentProver  # noqa: E402

def run(hidden: int, T: int, reps: int, warmup: int):
    cfg = ToyConfig(hidden=(hidden, hidden), eos_bias=-10 ** 6)
    key = SigningKey()
    m = build_model(cfg)
    env = sign_manifest(manifest_body(m, key, epoch=1), key)
    v = Verifier(TrustPolicy({key.key_id: key.public}))
    v.authorize_manifest(env)
    ps, vs, sz, scb = [], [], [], []
    for k in range(warmup + reps):
        dev = TransparentProver(m, env, backend=SumFoldLaneEmulator(4))
        s = CertificationSession(v, dev, env, new_request([2 + k % 20, 5, 9], T, m.tokenizer["id"]))
        s.authorize()
        s.run_inference()
        t0 = time.perf_counter()
        s.await_proof()
        tp = time.perf_counter() - t0
        s.verify()
        if s.state != State.VERIFIED or len(s.provisional) != T:
            raise RuntimeError(f"run failed: {s.abort_reason} {s.failed_check}")
        if k < warmup:
            continue
        ps.append(tp)
        vs.append(s.verification.verify_s)
        sz.append(len(canonical_json(s.proof)))
        scb.append(sum(16 * (3 * len(sc["msgs"]) + 3) for sc in s.proof["sumchecks"]))  # msgs + claim + 2 finals
    rounds = [sc["n"] for sc in s.proof["sumchecks"]]
    return {"hidden": hidden, "T": T, "layers": [[L["in"], L["out"]] for L in m.layers], "sumcheck_rounds": rounds,
            "prove_s_median": statistics.median(ps), "prove_s_min": min(ps), "prove_s_max": max(ps),
            "verify_s_median": statistics.median(vs), "verify_s_min": min(vs), "verify_s_max": max(vs),
            "proof_bytes_canonical_json": sz[0], "sumcheck_part_bytes": scb[0], "n": len(ps)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "results" / "raw" / "attacks" / "transparent_scaling.json"))
    a = ap.parse_args()
    pts = []
    for hidden in (32, 64, 128):
        for T in (4, 8, 16):
            r = run(hidden, T, a.reps, a.warmup)
            print(json.dumps(r))
            pts.append(r)
    rec = {"status": "MEASURED_SUCCESS", "evidence_class": "MEASURED", "experiment_kind": "FULL_INFERENCE_PROOF",
           "assurance_mode": "NON_ZK_ARGUMENT", "certificate_kind": "TRANSPARENT_REFERENCE",
           "note": "Pure-Python transparent (non-ZK, non-succinct) reference prover on a toy integer LM; "
                   "not comparable with real zkML backends. sumcheck_part_bytes is an exact byte count "
                   "(ANALYTIC) of round messages, claims and final values.",
           "command": "python bench/attacks/time_transparent.py " + " ".join(sys.argv[1:]) + "  (PYTHONPATH=src)",
           "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "host": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
                    "python": sys.version.split()[0],
                    "load_note": "concurrent Vivado builds were running on this host"},
           "config": {"vocab": 32, "d_emb": 16, "window": 4, "lanes": 4, "warmup": a.warmup, "reps": a.reps},
           "points": pts}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
