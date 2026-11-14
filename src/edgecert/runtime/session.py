"""Certify-before-release session state machine (C1 enforcement layer).

    CREATED -> AUTHORIZED -> RUNNING -> PROOF_PENDING -> VERIFIED -> CLOSED
    any error ---------------------------------------------------> ABORTED

Invariants enforced here (spec/security_obligations.md, O7):
  E1  Only the independent verifier's acceptance moves a session to VERIFIED;
      the session never sets VERIFIED on its own judgement.
  E2  STRICT release (default): nothing is released to an actuator unless the
      session is VERIFIED (or CLOSED after VERIFIED) and holds a signed receipt.
  E3  PREVIEW: provisional tokens may be shown, always labelled UNCERTIFIED;
      after a failure they are labelled UNCERTIFIED_FAILED.  PREVIEW never
      releases to an actuator either.
  E4  Timeouts, OOM, missing proof, prover/accelerator errors, verifier
      rejection and illegal transitions all end in ABORTED; an ABORTED
      session never becomes VERIFIED (terminal state).
  E5  There is no silent fallback: an uncertified output never carries the
      CERTIFIED label.

The session runs on the relying-party side (release gate).  The device (prover) is untrusted and is called through ``infer``/``prove``; the verifier is independent.  Worker-thread timeouts abandon a late prover result (Python threads cannot be killed; the cooperative deadline is passed to the prover).
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field as dfield
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..verifier import instance as I

class State(str, Enum):
    CREATED = "CREATED"
    AUTHORIZED = "AUTHORIZED"
    RUNNING = "RUNNING"
    PROOF_PENDING = "PROOF_PENDING"
    VERIFIED = "VERIFIED"
    CLOSED = "CLOSED"
    ABORTED = "ABORTED"

TRANSITIONS = {
    State.CREATED: {State.AUTHORIZED, State.ABORTED},
    State.AUTHORIZED: {State.RUNNING, State.ABORTED},
    State.RUNNING: {State.PROOF_PENDING, State.ABORTED},
    State.PROOF_PENDING: {State.VERIFIED, State.ABORTED},
    State.VERIFIED: {State.CLOSED},
    State.CLOSED: set(),
    State.ABORTED: set(),
}

class ReleasePolicy(str, Enum):
    STRICT = "STRICT"
    PREVIEW = "PREVIEW"

CERTIFIED = "CERTIFIED"
UNCERTIFIED = "UNCERTIFIED"
UNCERTIFIED_FAILED = "UNCERTIFIED_FAILED"

class IllegalTransition(RuntimeError):
    pass

class ReleaseBlocked(RuntimeError):
    pass

@dataclass
class Timeouts:
    inference_s: float = 60.0
    proof_s: float = 300.0
    verify_s: float = 300.0

@dataclass
class Request:
    """Requester-side request.  Private fields go to the device only."""
    prompt: List[int]
    max_new_tokens: int
    tokenizer_id: str
    session_id: str
    requester_nonce: str
    input_commitment: str
    input_randomness: bytes
    previous_receipt: Optional[dict] = None
    previous_state: Optional[dict] = None    # device-held state opening {"window","position","randomness"}
    output_mode: str = "tokens"

    @property
    def previous_receipt_id(self) -> str:
        return self.previous_receipt["receipt_id"] if self.previous_receipt else I.GENESIS_RECEIPT_ID

    @property
    def context_anchor(self) -> str:
        return self.previous_receipt["body"]["end_state_commitment"] if self.previous_receipt else I.GENESIS_ANCHOR

def new_request(prompt: List[int], max_new_tokens: int, tokenizer_id: str, previous_receipt: Optional[dict] = None,
                previous_state: Optional[dict] = None, output_mode: str = "tokens") -> Request:
    """Fresh CSPRNG nonce, session id and input-commitment randomness."""
    ic, io = I.commit_input(prompt, max_new_tokens, tokenizer_id)
    return Request(list(prompt), max_new_tokens, tokenizer_id, secrets.token_hex(16), secrets.token_hex(32),
                   ic.hex(), io.randomness, previous_receipt, previous_state, output_mode)

@dataclass
class SessionContext:
    """Public session data the device receives (mirrors verifier Expectation + anchor/epoch)."""
    manifest_id: str
    session_id: str
    requester_nonce: str
    input_commitment: str
    previous_receipt_id: str
    context_anchor: str
    policy_epoch: int
    output_mode: str = "tokens"

@dataclass
class PrivateRequest:
    """Witness-side request data sent to the device over an authenticated channel."""
    prompt: List[int]
    max_new_tokens: int
    input_randomness: bytes
    state: Optional[dict] = None       # {"window": [...], "position": int, "randomness": bytes | hex}

@dataclass
class Released:
    tokens: List[int]
    label: str
    receipt_id: Optional[str]
    session_id: str

def _call_with_timeout(fn: Callable[[], Any], timeout: float):
    box: Dict[str, Any] = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # propagate to caller thread
            box["error"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError("worker exceeded its timeout (result abandoned)")
    if "error" in box:
        raise box["error"]
    return box.get("value")

class CertificationSession:
    def __init__(self, verifier, device, manifest_env: dict, request: Request,
                 policy: ReleasePolicy = ReleasePolicy.STRICT, timeouts: Timeouts = Timeouts(),
                 audit_log=None, clock: Callable[[], float] = time.monotonic):
        self.verifier = verifier
        self.device = device
        self.manifest_env = manifest_env
        self.req = request
        self.policy = policy
        self.timeouts = timeouts
        self.audit_log = audit_log
        self.clock = clock
        self.state = State.CREATED
        self.abort_reason: Optional[str] = None
        self.failed_check: Optional[str] = None
        self.manifest_id: Optional[str] = None
        self.provisional: Optional[List[int]] = None
        self.proof: Optional[dict] = None
        self.receipt: Optional[dict] = None
        self.verification = None
        self.events: List[Dict[str, Any]] = []
        self.t_created = clock()
        self.t_provisional: Optional[float] = None
        self.t_verified: Optional[float] = None
        self._log(State.CREATED, "session created")

    # ------------------------------------------------------------ transitions
    def _log(self, st: State, note: str) -> None:
        self.events.append({"t": self.clock() - self.t_created, "state": st.value, "note": note})

    def _to(self, st: State, note: str = "") -> None:
        if st not in TRANSITIONS[self.state]:
            raise IllegalTransition(f"{self.state.value} -> {st.value}")
        self.state = st
        self._log(st, note)

    def abort(self, reason: str, failed_check: Optional[str] = None) -> None:
        if self.state in (State.ABORTED, State.CLOSED):
            return
        if self.state == State.VERIFIED:
            # a verified session can only be closed; record the error but keep the certificate
            self._log(self.state, f"post-verification error ignored for release decision: {reason}")
            return
        self.abort_reason = reason
        self.failed_check = failed_check
        self.state = State.ABORTED
        self._log(State.ABORTED, reason)

    def _ctx(self):
        ctx = SessionContext(self.manifest_id, self.req.session_id, self.req.requester_nonce,
                             self.req.input_commitment, self.req.previous_receipt_id, self.req.context_anchor,
                             self.manifest_env["body"]["policy"]["policy_epoch"], self.req.output_mode)
        priv = PrivateRequest(self.req.prompt, self.req.max_new_tokens, self.req.input_randomness,
                              self.req.previous_state)
        return ctx, priv

    # ------------------------------------------------------------ steps
    def authorize(self) -> None:
        try:
            ok, why, mid = self.verifier.trust.check(self.manifest_env)
            if mid in self.verifier.trust.accepted:
                cur, why2 = self.verifier.trust.current(mid)
                if not cur:
                    raise ValueError(why2)
            elif ok:
                mid = self.verifier.authorize_manifest(self.manifest_env)
            else:
                raise ValueError(why)
            if self.req.previous_receipt is not None:
                rok, rwhy = self.verifier.import_receipt(self.req.previous_receipt)
                if not rok:
                    raise ValueError(rwhy)
            self.manifest_id = mid
            self._to(State.AUTHORIZED, f"manifest {mid[:12]} accepted")
        except IllegalTransition:
            self.abort("ILLEGAL_TRANSITION")
        except Exception as e:
            self.abort("MANIFEST_OR_POLICY_REJECTED", str(e))

    def run_inference(self) -> None:
        try:
            self._to(State.RUNNING, "inference started")
            ctx, priv = self._ctx()
            out = _call_with_timeout(lambda: self.device.infer(ctx, priv), self.timeouts.inference_s)
            self.provisional = list(out)
            self.t_provisional = self.clock()
            self._to(State.PROOF_PENDING, f"{len(out)} provisional tokens (UNCERTIFIED)")
        except IllegalTransition:
            self.abort("ILLEGAL_TRANSITION")
        except TimeoutError:
            self.abort("INFERENCE_TIMEOUT")
        except Exception as e:
            self.abort("INFERENCE_ERROR", f"{type(e).__name__}: {e}")

    def await_proof(self) -> None:
        if self.state != State.PROOF_PENDING:
            self.abort("ILLEGAL_TRANSITION")
            return
        ctx, priv = self._ctx()
        deadline = self.clock() + self.timeouts.proof_s
        try:
            pf = _call_with_timeout(lambda: self.device.prove(ctx, priv, self.provisional, deadline),
                                    self.timeouts.proof_s)
        except TimeoutError as e:
            self.abort("PROOF_TIMEOUT", type(e).__name__)
            return
        except MemoryError as e:
            self.abort("PROVER_OOM", str(e))
            return
        except Exception as e:
            name = type(e).__name__
            self.abort("ACCELERATOR_FAULT" if name == "AcceleratorFault" else "PROVER_ERROR", f"{name}: {e}")
            return
        if pf is None:
            self.abort("PROOF_MISSING")
            return
        self.proof = pf

    def verify(self) -> None:
        if self.state != State.PROOF_PENDING or self.proof is None:
            if self.state not in (State.ABORTED,):
                self.abort("PROOF_MISSING" if self.proof is None else "ILLEGAL_TRANSITION")
            return
        from ..verifier.transparent_verifier import Expectation
        exp = Expectation(self.manifest_id, self.req.session_id, self.req.requester_nonce,
                          self.req.input_commitment, self.req.previous_receipt_id, self.req.output_mode)
        try:
            res = _call_with_timeout(lambda: self.verifier.verify(self.proof, exp), self.timeouts.verify_s)
        except TimeoutError:
            self.abort("VERIFY_TIMEOUT")
            return
        self.verification = res
        if not res.accepted or res.receipt is None:
            self.abort("VERIFY_REJECTED", res.failed_check)
            return
        # E1: VERIFIED only on independent acceptance, and the certified tokens are the proven ones
        proven = self.proof["trace"]["output_tokens"]
        if proven != self.provisional:
            self.abort("VERIFY_REJECTED", "session.provisional_output_mismatch")
            return
        self.receipt = res.receipt
        self.t_verified = self.clock()
        self._to(State.VERIFIED, f"receipt {res.receipt['receipt_id'][:12]}")
        if self.audit_log is not None:
            self.audit_log.append(res.receipt["receipt_id"])

    def close(self) -> None:
        if self.state == State.VERIFIED:
            self._to(State.CLOSED, "closed")

    def run(self) -> "CertificationSession":
        self.authorize()
        if self.state == State.AUTHORIZED:
            self.run_inference()
        if self.state == State.PROOF_PENDING:
            self.await_proof()
        if self.state == State.PROOF_PENDING:
            self.verify()
        return self

    # ------------------------------------------------------------ release
    @property
    def certified(self) -> bool:
        verified_seen = any(e["state"] == State.VERIFIED.value for e in self.events)
        return self.state in (State.VERIFIED, State.CLOSED) and verified_seen and self.receipt is not None

    def release_to_actuator(self) -> Released:
        """E2: actionable release only for certified sessions (both policies)."""
        if not self.certified:
            raise ReleaseBlocked(f"session {self.state.value}: output UNCERTIFIED, not released")
        return Released(list(self.proof["trace"]["output_tokens"]), CERTIFIED, self.receipt["receipt_id"],
                        self.req.session_id)

    def preview(self) -> Optional[Released]:
        """E3: display-only view.  STRICT shows nothing until certified."""
        if self.certified:
            return self.release_to_actuator()
        if self.policy != ReleasePolicy.PREVIEW or self.provisional is None:
            return None
        label = UNCERTIFIED_FAILED if self.state == State.ABORTED else UNCERTIFIED
        return Released(list(self.provisional), label, None, self.req.session_id)

    @property
    def label(self) -> str:
        if self.certified:
            return CERTIFIED
        return UNCERTIFIED_FAILED if self.state == State.ABORTED else UNCERTIFIED

    def metrics(self) -> Dict[str, Any]:
        prov = None if self.t_provisional is None else self.t_provisional - self.t_created
        cert = None if self.t_verified is None else self.t_verified - self.t_created
        unc = None
        if self.t_provisional is not None:
            unc = (self.t_verified if self.t_verified is not None else self.clock()) - self.t_provisional
        return {"state": self.state.value, "label": self.label, "abort_reason": self.abort_reason,
                "failed_check": self.failed_check, "provisional_latency_s": prov, "certified_latency_s": cert,
                "uncertified_duration_s": unc,
                "max_uncertified_tokens": len(self.provisional) if self.provisional is not None else 0}
