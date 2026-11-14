#!/usr/bin/env python
"""EDGE-CERT attack / enforcement suite (C1, C5).

Runs positive controls and negative tests against the prototype certification runtime (runtime/session.py) with the TRANSPARENT_REFERENCE prover (prover_adapters/transparent.py) and the independent verifier (verifier/transparent_verifier.py), plus audit-log rollback/fork tests.

Usage (repository root):
    set PYTHONPATH=src
    python bench/attacks/run_attacks.py [--reps 10] [--warmup 3] [--out results/raw/attacks]
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edgecert.field import P  # noqa: E402
from edgecert.verifier.mle import log2_ceil  # noqa: E402
from edgecert.crypto.ed25519 import SigningKey  # noqa: E402
from edgecert.runtime.manifest import TrustPolicy, canonical_json, sign_manifest  # noqa: E402
from edgecert.runtime.session import (CertificationSession, ReleaseBlocked, ReleasePolicy, State, Timeouts,  # noqa: E402
                                      new_request, UNCERTIFIED, UNCERTIFIED_FAILED)
from edgecert.verifier import instance as I  # noqa: E402
from edgecert.verifier.transparent_verifier import Expectation, Verifier  # noqa: E402
from edgecert.audit import receipts as RC  # noqa: E402
from edgecert.audit.log import AuditLog, Witness, verify_presented_log  # noqa: E402
from edgecert.prover_adapters.toy_model import (PAD, ToyConfig, act_lut_relu, adapter_commitment,  # noqa: E402
                                                adapter_delta, apply_adapter, build_model, combined_commitment,
                                                manifest_body)
from edgecert.prover_adapters.transparent import (HostSumcheck, SumFoldLaneEmulator, Tamper,  # noqa: E402
                                                  TransparentProver)

OBS_OF_ABORT = {
    "VERIFY_REJECTED": "REJECT", "MANIFEST_OR_POLICY_REJECTED": "REJECT",
    "PROOF_MISSING": "UNCERTIFIED",
    "PROOF_TIMEOUT": "ABORT", "PROVER_OOM": "ABORT", "ACCELERATOR_FAULT": "ABORT", "PROVER_ERROR": "ABORT",
    "INFERENCE_ERROR": "ABORT", "INFERENCE_TIMEOUT": "ABORT", "VERIFY_TIMEOUT": "ABORT",
    "ILLEGAL_TRANSITION": "ABORT",
}

LANES = 4

# stable names for checks that fire in the runtime rather than in the verifier
RUNTIME_CHECK = {
    "PROOF_MISSING": "runtime.proof_presence", "PROOF_TIMEOUT": "runtime.proof_deadline",
    "PROVER_OOM": "runtime.memory_budget", "ACCELERATOR_FAULT": "runtime.host_round_self_check",
    "PROVER_ERROR": "runtime.prover_error", "INFERENCE_ERROR": "runtime.inference_error",
    "INFERENCE_TIMEOUT": "runtime.inference_deadline", "VERIFY_TIMEOUT": "runtime.verify_deadline",
    "ILLEGAL_TRANSITION": "runtime.state_machine",
}

def flip_int8_bit(v: int, bit: int) -> int:
    u = (v & 0xFF) ^ (1 << bit)
    return u - 256 if u >= 128 else u

class World:
    """Publisher, verifier, operator log, witnesses and the authorized model."""

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.pub = SigningKey()
        self.m = build_model()
        self.tok_id = self.m.tokenizer["id"]
        base_mc = self.m.commitment()
        # one authorized adapter (positive control) on L1.w
        self.adapter = {"adapter_id": "adapter-a", "target": "L1.w",
                        "delta": adapter_delta(self.m, "L1.w", seed=11)}
        acom = adapter_commitment("L1.w", self.adapter["delta"])
        self.adapter_rec = {"adapter_id": "adapter-a", "adapter_commitment": acom,
                            "combined_model_commitment": combined_commitment(base_mc, acom), "target": "L1.w"}
        self.env_old = sign_manifest(manifest_body(self.m, self.pub, epoch=1), self.pub)
        self.env = sign_manifest(manifest_body(self.m, self.pub, epoch=2, adapters=[self.adapter_rec]), self.pub)
        self.trust = TrustPolicy({self.pub.key_id: self.pub.public})
        self.verifier = Verifier(self.trust)
        self.verifier.authorize_manifest(self.env_old)
        self.mid = self.verifier.authorize_manifest(self.env)       # epoch floor is now 2
        self.op_key = SigningKey()
        self.log = AuditLog("edge-gw-01", self.op_key)
        self.w1 = Witness("witness-A", SigningKey(), {self.op_key.key_id: self.op_key.public})
        self.w2 = Witness("witness-B", SigningKey(), {self.op_key.key_id: self.op_key.public})
        self.w1_seen = 0
        self.w2_seen = 0

    def device(self, tamper=None, backend=None, adapter=None, mem=None, env=None):
        return TransparentProver(self.m, env or self.env, backend=backend or SumFoldLaneEmulator(LANES),
                                 adapter=adapter, mem_budget_bytes=mem, tamper=tamper)

    def request(self, prompt, max_new=8, **kw):
        return new_request(prompt, max_new, self.tok_id, **kw)

    def random_prompt(self, lo=1, hi=6):
        return [self.rng.randrange(2, self.m.cfg.vocab) for _ in range(self.rng.randint(lo, hi))]

    def honest_out(self, prompt, max_new, state=None, tensors=None):
        sw, sp = (state["window"], state["position"]) if state else ([PAD] * self.m.cfg.window, 0)
        return self.m.generate(sw, sp, prompt, max_new, tensors=tensors)

def tampered_out(w: World, tamper: Tamper, prompt, max_new=8):
    """Output the tampering device would claim for `prompt` (no proof)."""
    from edgecert.runtime.session import PrivateRequest, SessionContext
    dev = TransparentProver(w.m, w.env, tamper=tamper)
    ctx = SessionContext(w.mid, "", "", "", I.GENESIS_RECEIPT_ID, I.GENESIS_ANCHOR, 2)
    return dev.infer(ctx, PrivateRequest(list(prompt), max_new, bytes(32)))

def find_distinguishing_prompt(w: World, tamper: Tamper, max_new=8, lo=1, hi=6, tries=3000, seed=17):
    """Search public random prompts until the tampered claim differs from the honest output."""
    rng = random.Random(seed)
    for k in range(tries):
        pr = [rng.randrange(2, w.m.cfg.vocab) for _ in range(rng.randint(lo, hi))]
        if tampered_out(w, tamper, pr, max_new) != w.honest_out(pr, max_new):
            return pr, k + 1
    raise RuntimeError("no distinguishing input found")

def run_session(w: World, device, req, policy=ReleasePolicy.STRICT, timeouts=None, env=None, verifier=None):
    s = CertificationSession(verifier or w.verifier, device, env or w.env, req, policy=policy,
                             timeouts=timeouts or Timeouts(), audit_log=w.log)
    s.run()
    return s

def released(s) -> bool:
    try:
        s.release_to_actuator()
        return True
    except ReleaseBlocked:
        return False

def session_row(s, expected, honest_tokens=None, honest_witness=None):
    if s.certified:
        observed = "ACCEPT"
    else:
        observed = OBS_OF_ABORT.get(s.abort_reason, "ABORT")
    rel = released(s)
    row = {
        "expected": expected, "observed": observed,
        "detecting_check": None if observed == "ACCEPT" else RUNTIME_CHECK.get(s.abort_reason, s.failed_check),
        "abort_detail": s.failed_check,
        "session_state": s.state.value, "abort_reason": s.abort_reason, "label": s.label,
        "released_to_actuator": rel,
        "verifier_reason": (s.verification.reason if s.verification is not None and not s.verification.accepted
                            else None),
    }
    if honest_tokens is not None and s.provisional is not None:
        row["output_differs_from_honest"] = s.provisional != honest_tokens
    if honest_witness is not None and s.proof is not None:
        row["witness_differs_from_honest"] = s.proof.get("witness") != honest_witness
    row["pass"] = (observed == expected) and (rel == (expected == "ACCEPT")) and \
        ((s.label == "CERTIFIED") == (expected == "ACCEPT"))
    return row

def honest_witness(w: World, req, adapter=None):
    """Witness an honest prover would produce for the same request (for 'distinguishing' flags)."""
    s = CertificationSession(Verifier(TrustPolicy({w.pub.key_id: w.pub.public})),
                             w.device(adapter=adapter, backend=HostSumcheck()), w.env, req)
    s.authorize()
    s.run_inference()
    s.await_proof()
    return s.provisional, (s.proof or {}).get("witness")

# ============================================================ attack catalogue
def attacks(w: World):
    rows = []

    def add(aid, name, category, description, fn):
        t0 = time.perf_counter()
        try:
            r = fn()
        except Exception as e:  # a crashing test is a FAIL, never silently skipped
            r = {"expected": "?", "observed": "HARNESS_ERROR", "pass": False,
                 "detecting_check": None, "error": f"{type(e).__name__}: {e}"}
        r.update({"id": aid, "name": name, "category": category, "description": description,
                  "elapsed_s": round(time.perf_counter() - t0, 4)})
        rows.append(r)
        flag = "PASS" if r["pass"] else "FAIL"
        print(f"[{flag}] {aid} {name:38s} expected={r['expected']:<11} observed={r['observed']:<11} "
              f"check={r.get('detecting_check')}", flush=True)

    # ---------------------------------------------------------------- positive controls
    receipts = {}

    def pc_honest():
        req = w.request([5, 9, 3], 8)
        s = run_session(w, w.device(), req)
        receipts["pc1"] = (s, req)
        r = session_row(s, "ACCEPT")
        r["receipt_id"] = s.receipt["receipt_id"] if s.receipt else None
        return r
    add("P01", "positive_control_honest", "positive control",
        "Honest device, authorized base model, SumFold 4-lane emulator backend, strict release.", pc_honest)

    def pc_adapter():
        req = w.request([7, 7, 2, 30], 8)
        s = run_session(w, w.device(adapter=w.adapter), req)
        return session_row(s, "ACCEPT")
    add("P02", "positive_control_authorized_adapter", "positive control",
        "Honest device using the manifest-authorized adapter (combined model commitment).", pc_adapter)

    def pc_continuation():
        s1, _ = receipts["pc1"]
        es = s1.proof["end_state"]
        st = {"window": es["window"], "position": es["position"], "randomness": bytes.fromhex(es["randomness"])}
        req = w.request([11], 6, previous_receipt=s1.receipt, previous_state=st)
        s = run_session(w, w.device(), req)
        receipts["pc3"] = (s, req, st)
        return session_row(s, "ACCEPT")
    add("P03", "positive_control_chunk_continuation", "positive control",
        "Second chunk anchored to the first chunk's receipt end-state commitment.", pc_continuation)

    # ---------------------------------------------------------------- manifest / model binding
    m2 = build_model(ToyConfig(seed=4242))
    fake_pub = SigningKey()

    def unauthorized_model():
        req = w.request([5, 9, 3], 8)
        dev = w.device(Tamper(compute_tensors=m2.tensors, open_tensors=m2.tensors,
                              declared_model_commitment=m2.commitment(),
                              scales={l: (L["multiplier"], L["shift"]) for l, L in enumerate(m2.layers)}))
        s = run_session(w, dev, req)
        return session_row(s, "REJECT", w.honest_out([5, 9, 3], 8))
    add("A01", "unauthorized_model_commitment", "model binding",
        "Counterfeit model (different weights) proven honestly under its own commitment.", unauthorized_model)

    def forged_manifest_sig():
        env = copy.deepcopy(w.env)
        env["body"]["model"]["model_commitment"] = m2.commitment()
        s = run_session(w, w.device(Tamper(compute_tensors=m2.tensors, open_tensors=m2.tensors), env=env),
                        w.request([5, 9, 3], 8), env=env)
        return session_row(s, "REJECT")
    add("A02", "forged_manifest_signature", "model binding",
        "Manifest body edited to authorize the counterfeit commitment, original signature kept.", forged_manifest_sig)

    def untrusted_publisher():
        env = sign_manifest(manifest_body(m2, fake_pub, epoch=9), fake_pub)
        s = run_session(w, TransparentProver(m2, env, backend=SumFoldLaneEmulator(LANES)),
                        w.request([5, 9, 3], 8), env=env)
        return session_row(s, "REJECT")
    add("A03", "self_signed_counterfeit_manifest", "model binding",
        "Operator signs its own manifest for the counterfeit model with an untrusted key.", untrusted_publisher)

    def manifest_rollback():
        s = run_session(w, w.device(env=w.env_old), w.request([5, 9, 3], 8), env=w.env_old)
        return session_row(s, "REJECT")
    add("A04", "manifest_rollback_stale_epoch", "session binding",
        "Superseded manifest (epoch 1) presented after epoch 2 was accepted.", manifest_rollback)

    def stale_epoch_instance():
        def mut(inst):
            inst = dict(inst)
            inst["policy_epoch"] = 1
            return inst
        s = run_session(w, w.device(Tamper(instance_mutator=mut)), w.request([5, 9, 3], 8))
        return session_row(s, "REJECT")
    add("A05", "stale_policy_epoch_in_instance", "session binding",
        "Public instance claims the old policy epoch (digest recomputed by the attacker).", stale_epoch_instance)

    # single-bit weight modification with a distinguishing input
    flip = {}

    def find_flip():
        rng = random.Random(99)
        for _ in range(4000):
            name = rng.choice(["L2.w", "L1.w", "L0.w"])
            Wt = w.m.tensors[name]
            i, j = rng.randrange(len(Wt)), rng.randrange(len(Wt[0]))
            bit = rng.choice([6, 5, 4])
            mod = copy.deepcopy(w.m.tensors)
            mod[name][i][j] = flip_int8_bit(mod[name][i][j], bit)
            for _ in range(6):
                pr = w.random_prompt()
                if w.honest_out(pr, 8) != w.honest_out(pr, 8, tensors=mod):
                    flip.update(name=name, i=i, j=j, bit=bit, tensors=mod, prompt=pr)
                    return
        raise RuntimeError("no distinguishing single-bit flip found")

    def bitflip_stealth():
        find_flip()
        req = w.request(flip["prompt"], 8)
        s = run_session(w, w.device(Tamper(compute_tensors=flip["tensors"])), req)
        r = session_row(s, "REJECT", w.honest_out(flip["prompt"], 8))
        r["details"] = {k: flip[k] for k in ("name", "i", "j", "bit", "prompt")}
        r["details"]["opened"] = "authorized weights"
        return r
    add("A06", "single_bit_weight_flip_stealth", "model binding",
        "One weight bit flipped (distinguishing input found by search); proof opens the AUTHORIZED weights.",
        bitflip_stealth)

    def bitflip_opened():
        req = w.request(flip["prompt"], 8)
        s = run_session(w, w.device(Tamper(compute_tensors=flip["tensors"], open_tensors=flip["tensors"])), req)
        r = session_row(s, "REJECT", w.honest_out(flip["prompt"], 8))
        r["details"] = {"opened": "modified weights"}
        return r
    add("A07", "single_bit_weight_flip_opened", "model binding",
        "Same flipped weight, proof opens the MODIFIED weights under the authorized commitment.", bitflip_opened)

    bad_delta = adapter_delta(w.m, "L0.w", seed=777, density=4, mag=8)

    adp_prompt = {}

    def adapter_prompt():
        if not adp_prompt:
            pr, k = find_distinguishing_prompt(
                w, Tamper(compute_tensors=apply_adapter(w.m.tensors, "L0.w", bad_delta)))
            adp_prompt.update(prompt=pr, tries=k)
        return adp_prompt["prompt"]

    def adapter_declared():
        pr = adapter_prompt()
        acom = adapter_commitment("L0.w", bad_delta)
        dev = w.device(Tamper(declared_adapter={"adapter_id": "evil", "target": "L0.w", "delta": bad_delta},
                              declared_model_commitment=combined_commitment(w.m.commitment(), acom)))
        s = run_session(w, dev, w.request(pr, 8))
        r = session_row(s, "REJECT", w.honest_out(pr, 8))
        r["details"] = {"distinguishing_prompt": pr, "search_tries": adp_prompt["tries"]}
        return r
    add("A08", "unauthorized_adapter_declared", "model binding",
        "Unauthorized adapter applied and declared with its combined commitment.", adapter_declared)

    def adapter_stealth():
        pr = adapter_prompt()
        dev = w.device(Tamper(compute_tensors=apply_adapter(w.m.tensors, "L0.w", bad_delta)))
        s = run_session(w, dev, w.request(pr, 8))
        r = session_row(s, "REJECT", w.honest_out(pr, 8))
        r["details"] = {"distinguishing_prompt": pr, "search_tries": adp_prompt["tries"]}
        return r
    add("A09", "unauthorized_adapter_stealth", "model binding",
        "Unauthorized adapter applied silently; proof claims the base commitment.", adapter_stealth)

    # ---------------------------------------------------------------- numerical semantics
    def numeric(tamper, lo=1, hi=6):
        prompt, k = find_distinguishing_prompt(w, tamper, lo=lo, hi=hi)
        req = w.request(prompt, 8)
        ho, hw = honest_witness(w, req)
        s = run_session(w, w.device(tamper), req)
        r = session_row(s, "REJECT", ho, hw)
        r["details"] = {"distinguishing_prompt": prompt, "search_tries": k}
        return r

    L1 = w.m.layers[1]
    add("A10", "wrong_scale", "numerical semantics",
        "Layer-1 requant multiplier doubled; remainders forged so acc*M+h = q*2^s+rem holds with the manifest M.",
        lambda: numeric(Tamper(scales={1: (L1["multiplier"] * 2, L1["shift"])}, forge_remainders=True)))
    add("A11", "wrong_rounding", "numerical semantics",
        "Truncation (floor) instead of RHAZ; remainders forged against the manifest rule.",
        lambda: numeric(Tamper(rounding="TRUNC_FLOOR", forge_remainders=True)))
    add("A12", "wrong_lookup_table", "numerical semantics",
        "ReLU table used instead of the authorized leaky table; authorized table opened.",
        lambda: numeric(Tamper(lut={"act0": act_lut_relu()})))
    add("A13", "wrong_lookup_table_opened", "numerical semantics",
        "ReLU table used and opened as act0.",
        lambda: numeric(Tamper(lut={"act0": act_lut_relu()}, open_luts={"act0": act_lut_relu()})))

    def wraparound():
        prompt = [9, 9, 2]
        req = w.request(prompt, 6)
        honest = w.honest_out(prompt, 6)
        target = 0 if honest[0] != 0 else 2
        last = len(w.m.layers) - 1
        Lq = w.env["body"]["quantizer"]["layers"][last]
        M, s_ = Lq["multiplier"], Lq["shift"]
        info = {}

        def outmut(o):
            o = list(o)
            o[0] = target
            return o

        def wmut(W):
            A = W[f"L{last}.ACC"][target][0]
            sg = W[f"L{last}.SIG"][target][0]
            q_new = 127
            lhs = A * M + (1 << (s_ - 1)) - sg
            rem = (lhs - q_new * (1 << s_)) % P     # field element ("negative" remainder p - k)
            W[f"L{last}.Q"][target][0] = q_new
            W[f"L{last}.REM"][target][0] = rem
            W[f"L{last}.OUT"][target][0] = q_new
            info.update(acc=A, q_forged=q_new, rem_field=rem, rem_as_signed=rem - P,
                        field_equation_holds=((lhs - q_new * (1 << s_) - rem) % P == 0),
                        integer_equation_holds=(lhs == q_new * (1 << s_) + rem))
        s = run_session(w, w.device(Tamper(output_mutator=outmut, witness_mutator=wmut)), req)
        r = session_row(s, "REJECT", honest)
        r["details"] = {"target_token": target, **info}
        return r
    add("A14", "field_wraparound_claim", "numerical semantics",
        "Logit forged to 127 with a remainder that satisfies the equation only modulo p (p - k).", wraparound)

    # ---------------------------------------------------------------- input / context binding
    add("A15", "wrong_mask", "input/context binding",
        "PAD slots before the context start replaced by a real token embedding (mask not applied).",
        lambda: numeric(Tamper(unmask=True), lo=1, hi=3))
    add("A16", "wrong_position", "input/context binding",
        "Positional index shifted by +1.", lambda: numeric(Tamper(position_offset=1)))

    def input_mismatch():
        req = w.request([5, 9, 3], 8)
        s = run_session(w, w.device(Tamper(compute_prompt=[5, 9, 4])), req)
        return session_row(s, "REJECT", w.honest_out([5, 9, 3], 8))
    add("A17", "input_commitment_mismatch", "input/context binding",
        "Device computes on a substituted prompt and opens the commitment with it.", input_mismatch)

    # ---------------------------------------------------------------- session binding
    def nonce_replay_resubmit():
        s1, req1 = receipts["pc1"]
        s = run_session(w, w.device(Tamper(replay_proof=s1.proof)), req1)
        return session_row(s, "REJECT")
    add("A18", "nonce_replay_resubmission", "session binding",
        "An accepted proof is resubmitted for the same session/nonce.", nonce_replay_resubmit)

    def nonce_replay_new_session():
        s1, _ = receipts["pc1"]
        s = run_session(w, w.device(Tamper(replay_proof=s1.proof)), w.request([5, 9, 3], 8))
        return session_row(s, "REJECT")
    add("A19", "stale_proof_new_nonce", "session binding",
        "Old valid proof for the same prompt replayed to a new session with a fresh nonce.",
        nonce_replay_new_session)

    def output_swap():
        sa = run_session(w, w.device(), w.request([20, 21, 22], 8))
        a_tokens = sa.proof["trace"]["output_tokens"]

        def pm(pf):
            pf = copy.deepcopy(pf)
            pf["instance"]["output_tokens_or_commitment"] = I.output_field_tokens(a_tokens)
            pf["instance"]["output_length"] = len(a_tokens)
            pf["instance_digest"] = I.instance_digest(pf["instance"]).hex()
            return pf
        req_b = w.request([2, 3, 5, 7], 8)
        s = run_session(w, w.device(Tamper(proof_mutator=pm)), req_b)
        r = session_row(s, "REJECT")
        r["details"] = {"swapped_in_tokens": a_tokens, "honest_tokens_b": w.honest_out([2, 3, 5, 7], 8)}
        return r
    add("A20", "cross_session_output_swap", "session binding",
        "Session A's output placed in session B's public instance (digest recomputed).", output_swap)

    # ---------------------------------------------------------------- sequence coverage
    def find_prompt(pred, max_new):
        rng = random.Random(5)
        for _ in range(2000):
            pr = [rng.randrange(2, w.m.cfg.vocab) for _ in range(rng.randint(1, 5))]
            o = w.honest_out(pr, max_new)
            if pred(o):
                return pr, o
        raise RuntimeError("no prompt found")

    def missing_token():
        pr, o = find_prompt(lambda o: len(o) == 8 and o[-1] != w.m.cfg.eos_id, 8)
        s = run_session(w, w.device(Tamper(output_mutator=lambda x: x[:-1])), w.request(pr, 8))
        r = session_row(s, "REJECT", o)
        r["details"] = {"prompt": pr, "honest": o}
        return r
    add("A21", "missing_token", "sequence coverage",
        "Final generated token dropped (no EOS, below the length budget).", missing_token)

    def reordered():
        pr, o = find_prompt(lambda o: len(o) >= 3 and o[0] != o[1] and w.m.cfg.eos_id not in o[:2], 8)

        def mut(x):
            x = list(x)
            x[0], x[1] = x[1], x[0]
            return x
        s = run_session(w, w.device(Tamper(output_mutator=mut)), w.request(pr, 8))
        r = session_row(s, "REJECT", o)
        r["details"] = {"prompt": pr, "honest": o}
        return r
    add("A22", "reordered_tokens", "sequence coverage",
        "First two generated tokens swapped; trace recomputed teacher-forced on the swapped sequence.", reordered)

    def modified_eos():
        eos = w.m.cfg.eos_id
        pr, o = find_prompt(lambda o: o[-1] == eos and len(o) < 8, 8)

        def mut(x):
            x = list(x[:-1]) + [(eos + 1) % w.m.cfg.vocab]
            full = x + w.m.generate([PAD] * w.m.cfg.window, 0, pr + x, 8 - len(x))[: 8 - len(x)]
            return [t if t != eos else (eos + 2) % w.m.cfg.vocab for t in full][:8]
        s = run_session(w, w.device(Tamper(output_mutator=mut)), w.request(pr, 8))
        r = session_row(s, "REJECT", o)
        r["details"] = {"prompt": pr, "honest": o, "claimed": s.provisional}
        return r
    add("A23", "modified_eos", "sequence coverage",
        "EOS replaced and generation continued to the budget (output extended past EOS).", modified_eos)

    # ---------------------------------------------------------------- state integrity
    def state_corruption():
        s1, _ = receipts["pc1"]
        es = s1.proof["end_state"]
        st = {"window": es["window"], "position": es["position"], "randomness": bytes.fromhex(es["randomness"])}
        bad = list(es["window"])
        bad[-1] = (bad[-1] + 1) % w.m.cfg.vocab
        req = w.request([11], 6, previous_receipt=s1.receipt, previous_state=st)
        s = run_session(w, w.device(Tamper(state_window=bad)), req)
        r = session_row(s, "REJECT", w.honest_out([11], 6, state=st))
        r["details"] = {"true_window": es["window"], "corrupted_window": bad}
        return r
    add("A24", "state_boundary_corruption", "state integrity",
        "Carried context window altered at the chunk boundary (opening no longer matches the anchor).",
        state_corruption)

    def forged_receipt():
        s1, _ = receipts["pc1"]
        es = s1.proof["end_state"]
        evil_key = SigningKey()
        window = [2, 3, 4, 5]
        c, o = I.commit_state(window, 40)
        body = dict(s1.receipt["body"])
        body.update(end_state_commitment=c.hex(), verifier_key_id=evil_key.key_id)
        forged = RC.issue_receipt(body, evil_key)
        st = {"window": window, "position": 40, "randomness": o.randomness}
        req = w.request([11], 6, previous_receipt=forged, previous_state=st)
        s = run_session(w, w.device(), req)
        r = session_row(s, "REJECT")
        # auditor mode: the verifier itself is handed the chain (no relying-party import step)
        from edgecert.runtime.session import SessionContext, PrivateRequest
        dev = w.device()
        ctx = SessionContext(w.mid, req.session_id, req.requester_nonce, req.input_commitment, forged["receipt_id"],
                             c.hex(), 2)
        priv = PrivateRequest([11], 6, req.input_randomness, st)
        pf = dev.prove(ctx, priv, dev.infer(ctx, priv))
        vr = w.verifier.verify(pf, Expectation(w.mid, req.session_id, req.requester_nonce, req.input_commitment, None))
        r["details"] = {"auditor_mode_verifier_check": vr.failed_check, "auditor_mode_accepted": vr.accepted}
        r["pass"] = r["pass"] and not vr.accepted
        return r
    add("A25", "forged_previous_receipt", "state integrity",
        "Continuation anchored to a receipt signed by an untrusted key (relying party and auditor mode).",
        forged_receipt)

    # ---------------------------------------------------------------- hardware faults (SumFold device)
    def accel(fault, self_check=False, expected="REJECT", policy=ReleasePolicy.STRICT):
        emu = SumFoldLaneEmulator(LANES, fault=fault, self_check=self_check)
        req = w.request([5, 9, 3], 8)
        s = run_session(w, w.device(backend=emu), req, policy=policy)
        r = session_row(s, expected, w.honest_out([5, 9, 3], 8))
        r["details"] = {"fault": fault, "fault_fired": emu.fault_fired, "host_self_check": self_check}
        r["pass"] = r["pass"] and emu.fault_fired
        return r
    add("A26", "accelerator_round_message_bitflip", "hardware fault",
        "Bit 5 of s0 flipped in device round 1 of the layer-1 SumFold job.",
        lambda: accel({"kind": "msg_bitflip", "job": 1, "round": 1, "comp": 0, "bit": 5}))
    add("A27", "accelerator_lane_final_bitflip", "hardware fault",
        "Bit 9 of lane 2's final f value flipped (host tail consumes it).",
        lambda: accel({"kind": "lane_final_bitflip", "job": 0, "lane": 2, "bit": 9}))
    add("A28", "stale_accelerator_buffer", "hardware fault",
        "g-table load skipped for the layer-1 job; previous job's buffer used.",
        lambda: accel({"kind": "stale_buffer", "job": 1}))
    add("A29", "dropped_dma_block", "hardware fault",
        "One 8-element DMA burst of the layer-2 f table never lands (reads as zero).",
        lambda: accel({"kind": "dropped_dma_block", "job": 2, "block": 3, "block_elems": 8}))
    add("A30", "accelerator_fault_host_selfcheck", "hardware fault",
        "Same bit flip as A26 with the host round self-check enabled: prover aborts before submitting.",
        lambda: accel({"kind": "msg_bitflip", "job": 1, "round": 1, "comp": 0, "bit": 5}, self_check=True,
                      expected="ABORT"))

    # ---------------------------------------------------------------- enforcement
    def missing_proof():
        s = run_session(w, w.device(Tamper(return_none=True)), w.request([5, 9, 3], 8))
        return session_row(s, "UNCERTIFIED")
    add("A31", "missing_proof", "enforcement", "Device returns no proof.", missing_proof)

    def timeout():
        s = run_session(w, w.device(Tamper(delay_s=1.0)), w.request([5, 9, 3], 8),
                        timeouts=Timeouts(inference_s=30, proof_s=0.3, verify_s=30))
        return session_row(s, "ABORT")
    add("A32", "proof_timeout", "enforcement", "Prover exceeds the 0.3 s proof deadline (1.0 s injected delay).",
        timeout)

    def oom():
        s = run_session(w, w.device(mem=64 * 1024), w.request([5, 9, 3], 8))
        r = session_row(s, "ABORT")
        r["details"] = {"memory_budget_bytes": 64 * 1024, "note": "budget-enforced (injected) OOM, not an OS OOM"}
        return r
    add("A33", "oom_during_proving", "enforcement", "Prover working-set exceeds a 64 KiB budget.", oom)

    def preview_failed():
        emu = SumFoldLaneEmulator(LANES, fault={"kind": "stale_buffer", "job": 1})
        s = CertificationSession(w.verifier, w.device(backend=emu), w.env, w.request([5, 9, 3], 8),
                                 policy=ReleasePolicy.PREVIEW, audit_log=w.log)
        s.authorize()
        s.run_inference()
        pv_before = s.preview()
        s.await_proof()
        s.verify()
        pv_after = s.preview()
        r = session_row(s, "REJECT")
        r["details"] = {"preview_label_before_verify": pv_before.label if pv_before else None,
                        "preview_label_after_verify": pv_after.label if pv_after else None}
        r["pass"] = r["pass"] and pv_before is not None and pv_before.label == UNCERTIFIED and \
            pv_after is not None and pv_after.label == UNCERTIFIED_FAILED
        return r
    add("A34", "preview_mode_failed_proof", "enforcement",
        "PREVIEW policy with a stale-buffer fault: tokens shown only as UNCERTIFIED, never released.",
        preview_failed)

    def unsupported_op():
        body = manifest_body(w.m, w.pub, epoch=2)
        gs = copy.deepcopy(body["graph"]["spec"])
        gs["layers"][0]["activation"] = "gelu_fp32"
        from edgecert.runtime.manifest import digest_obj
        body["graph"] = {"format": body["graph"]["format"], "graph_digest": digest_obj("EDGE-CERT/graph/v1", gs),
                         "spec": gs}
        body["quantizer"]["layers"][0]["activation"] = "gelu_fp32"
        env = sign_manifest(body, w.pub)
        s = run_session(w, w.device(env=env), w.request([5, 9, 3], 8), env=env)
        return session_row(s, "REJECT")
    add("A35", "unsupported_operator_fail_closed", "enforcement",
        "Validly signed manifest names an operator the verifier does not support (gelu_fp32).", unsupported_op)

    # ---------------------------------------------------------------- audit
    def audit_positive():
        n = len(w.log.entries)
        ok1, _ = w.w1.observe(w.log.checkpoint(), w.log.entries_between(w.w1_seen, n))
        ok2, _ = w.w2.observe(w.log.checkpoint(), w.log.entries_between(w.w2_seen, n))
        w.w1_seen = w.w2_seen = n
        ok, why = verify_presented_log(w.log.log_id, w.log.entries, [w.w1, w.w2],
                                       receipt_id=receipts["pc1"][0].receipt["receipt_id"])
        return {"expected": "ACCEPT", "observed": "ACCEPT" if (ok and ok1 and ok2) else "REJECT",
                "detecting_check": None if ok else why, "pass": ok and ok1 and ok2,
                "details": {"log_entries": n}}
    add("P04", "positive_control_audit_log", "positive control",
        "Honest log of all receipts issued so far, witnessed by two witnesses, receipt inclusion checked.",
        audit_positive)

    def audit_rollback():
        n = len(w.log.entries)
        rolled = w.log.fork()
        rolled.truncate(n - 2)
        ok, why = verify_presented_log(w.log.log_id, rolled.entries, [w.w1, w.w2])
                # an old checkpoint that is a prefix of what a witness saw is consistent (not an alarm by itself); the rollback is detected when the relying party checks the PRESENTED log against the witnesses.
        okc, alarm = w.w1.observe(rolled.checkpoint(), [])
        return {"expected": "REJECT", "observed": "REJECT" if not ok else "ACCEPT",
                "detecting_check": why.split(" ")[0] if not ok else None, "pass": not ok,
                "details": {"presented_len": len(rolled.entries), "witnessed_len": n, "reason": why,
                            "witness_accepts_old_prefix_checkpoint": okc,
                            "witness_alarm": (alarm or {}).get("kind")}}
    add("A36", "audit_rollback", "audit",
        "Operator presents the log rolled back by two receipts after witnesses checkpointed it.", audit_rollback)

    def audit_fork():
        n = len(w.log.entries)
        branch_a, branch_b = w.log.fork(), w.log.fork()
        branch_a.append("aa" * 32)
        branch_b.append("bb" * 32)
        wa = Witness("witness-C", SigningKey(), {w.op_key.key_id: w.op_key.public})
        wb = Witness("witness-D", SigningKey(), {w.op_key.key_id: w.op_key.public})
        wa.observe(branch_a.checkpoint(), branch_a.entries_between(0, n + 1))
        wb.observe(branch_b.checkpoint(), branch_b.entries_between(0, n + 1))
        gossip = wa.gossip(wb)
        ok, why = verify_presented_log(w.log.log_id, branch_b.entries, [wa, wb])
        okq, alarm = wa.observe(branch_b.checkpoint(), [])
        detected = bool(gossip) and not ok and not okq
        return {"expected": "REJECT", "observed": "REJECT" if detected else "ACCEPT",
                "detecting_check": "audit.fork" if detected else None, "pass": detected,
                "details": {"gossip_alarm": gossip[0]["kind"] if gossip else None, "presented_check": why,
                            "equivocation_alarm": (alarm or {}).get("kind")}}
    add("A37", "audit_fork", "audit",
        "Operator shows two different extensions of the same log to two witnesses (split view).", audit_fork)

    return rows

# ============================================================ honest timing
def honest_timing(w: World, reps: int, warmup: int):
    out = {}
    prompts = [[5, 9, 3], [7, 7, 2, 30], [12, 4], [3, 1, 4, 1, 5], [20, 21, 22]]
    for bname, mk in (("python-host", HostSumcheck), ("sumfold-lane-emulator-L4", lambda: SumFoldLaneEmulator(LANES))):
        prove_s, verify_s, sizes, Ts, wbytes, stages = [], [], [], [], [], []
        v = Verifier(TrustPolicy({w.pub.key_id: w.pub.public}))
        v.authorize_manifest(w.env)
        for k in range(warmup + reps):
            req = w.request(prompts[k % len(prompts)], 8)
            dev = TransparentProver(w.m, w.env, backend=mk())
            s = CertificationSession(v, dev, w.env, req)
            s.authorize()
            s.run_inference()
            t0 = time.perf_counter()
            s.await_proof()
            tp = time.perf_counter() - t0
            s.verify()
            if s.state != State.VERIFIED:
                raise RuntimeError(f"honest run not verified: {s.abort_reason} {s.failed_check}")
            if k < warmup:
                continue
            prove_s.append(tp)
            verify_s.append(s.verification.verify_s)
            sizes.append(len(canonical_json(s.proof)))
            Ts.append(len(s.provisional))
            wbytes.append(dev.last_stats["witness_bytes_est"])
            stages.append({k2: dev.last_stats[k2] for k2 in ("witness_s", "commit_s", "sumcheck_s", "open_s")})

        def st(xs):
            return {"median": statistics.median(xs), "mean": statistics.fmean(xs), "min": min(xs), "max": max(xs),
                    "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}
        sumcheck_bytes = None
        out[bname] = {
            "prove_s": st(prove_s), "verify_s": st(verify_s), "proof_bytes_canonical_json": st(sizes),
            "output_tokens": st(Ts), "witness_bytes_est": st(wbytes),
            "prover_stage_median_s": {k2: statistics.median(d[k2] for d in stages) for k2 in stages[0]},
        }
    return out

def bound(w: World, T: int):
    """spec/security_obligations.md: interactive per-layer error (a + t + 2b) / p^2."""
    import math
    terms = []
    for L in w.m.layers:
        a, t, b = log2_ceil(L["out"]), log2_ceil(T), max(1, log2_ceil(L["in"] + 1))
        terms.append(a + t + 2 * b)
    num = sum(terms)
    return {"evidence_class": "SUCCESS", "status": "SKETCH_PENDING_REVIEW", "per_layer_numerators": terms,
            "numerator": num, "denominator": "p^2", "log2_bound": math.log2(num) - 2 * math.log2(P),
            "excludes": "Fiat-Shamir query loss (about Q times 2/p^2) and hash-collision terms; not a security level"}

def model_summary(w: World):
    c = w.m.cfg
    return {"graph": "window-mlp-lm/v1", "vocab": c.vocab, "d_emb": c.d_emb, "window": c.window,
            "max_position": c.max_position, "hidden": list(c.hidden),
            "layers": [{k: L[k] for k in ("name", "in", "out", "activation", "multiplier", "shift")} for L in w.m.layers],
            "sumcheck_rounds_per_layer": [max(1, log2_ceil(L["in"] + 1)) for L in w.m.layers],
            "interactive_soundness_bound_T8": bound(w, 8),
            "manifest_id": w.mid, "model_commitment": w.m.commitment(),
            "relation_id": w.env["body"]["proof_system"]["relation_id"],
            "proof_config_id": w.env["body"]["proof_system"]["proof_config_id"], "sumfold_lanes": LANES}

def host_info():
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "load_note": "Concurrent Vivado builds by another team were running on this host; timings are "
                         "single-threaded pure-Python and include that interference."}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "raw" / "attacks"))
    args = ap.parse_args()
    t_start = time.perf_counter()
    w = World()
    rows = attacks(w)
    timing = honest_timing(w, args.reps, args.warmup)
    neg = [r for r in rows if r["category"] != "positive control"]
    pos = [r for r in rows if r["category"] == "positive control"]
    by_exp = {}
    for r in rows:
        by_exp.setdefault(r["expected"], {"n": 0, "pass": 0})
        by_exp[r["expected"]]["n"] += 1
        by_exp[r["expected"]]["pass"] += int(r["pass"])
    all_pass = all(r["pass"] for r in rows)
    rec = {
        "run_id": "attacks-" + _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "NORMAL" if all_pass else "VERIFY_FAILED",
        "evidence_class": "NORMAL",
        "experiment_kind": "ATTACK_TEST",
        "assurance_mode": "NON_ZK_ARGUMENT",
        "certificate_kind": "TRANSPARENT_REFERENCE",
        "note": NOTE,
        "command": "python " + " ".join([str(Path(sys.argv[0]).as_posix())] + sys.argv[1:]) + "  (PYTHONPATH=src)",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": host_info(),
        "tool": {"python": sys.version.split()[0], "edgecert_modules": [
            "edgecert.runtime.session", "edgecert.verifier.transparent_verifier",
            "edgecert.prover_adapters.transparent", "edgecert.audit.log", "edgecert.crypto.ed25519"]},
        "config": model_summary(w),
        "summary": {"n_rows": len(rows), "n_negative": len(neg), "n_positive": len(pos),
                    "n_pass": sum(r["pass"] for r in rows), "n_fail": sum(not r["pass"] for r in rows),
                    "negatives_released_to_actuator": sum(bool(r.get("released_to_actuator")) for r in neg),
                    "by_expected": by_exp, "suite_wall_s": round(time.perf_counter() - t_start, 3)},
        "honest_timing": {"evidence_class": "NORMAL", "warmup": args.warmup, "reps": args.reps,
                          "what": "prove_s = device.prove wall time (witness + commitments + sumchecks + openings); "
                                  "verify_s = independent verifier wall time; pure Python, single thread",
                          "backends": timing},
        "rows": rows,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rec["summary"], indent=1))
    print("honest timing (median s):", {k: (v["prove_s"]["median"], v["verify_s"]["median"]) for k, v in timing.items()})
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
