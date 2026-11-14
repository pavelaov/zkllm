"""TRANSPARENT_REFERENCE prover for R_EDGE on the toy integer LM (NOT zero knowledge, NOT succinct).

What it is
    A reference implementation of the *shape* of a sumcheck-based inference
    proof: every matrix multiplication of the integer MLP chain
        y_{l+1} = act(requant(W_l y_l + b_l))
    is reduced to one product sumcheck (edgecert.sumfold semantics)
        ACC~(r1, r2) = sum_k Waug~(r1, k) * Y~(k, r2),
    with the evaluation point (r1, r2) and every SumFold challenge drawn from
    ONE Fiat-Shamir transcript whose initial state is the public-instance
    digest and which absorbs all witness commitments before any challenge.

What it is not
    * Not zero knowledge: the proof reveals the whole witness (weights,
      activations, quotients, remainders, prompt and state openings).
    * Not succinct: the verifier re-evaluates multilinear extensions from the
      revealed tables instead of checking polynomial-commitment openings, so
      its cost is linear in the witness.
    * Not an inference certificate in the sense of the paper.  It exists to
      exercise the certification contract (C1) end to end and to run the
      attack/enforcement suite.  Label: TRANSPARENT_REFERENCE.

Requantization, lookups, gather, argmax and termination are checked by the verifier directly on the revealed integers, in constraint form (quotient / remainder range checks, table membership, difference range checks).  A real backend (prover_adapters/deepprove_adapter.py) replaces these with its own range/lookup arguments.

The ``Tamper`` hooks and the ``SumFoldLaneEmulator`` faults exist only for bench/attacks/run_attacks.py; an honest prover never sets them.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field as dfield
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..field import P, Ext, ZERO, eadd, ehex
from ..sumfold import fold, interp_quadratic, round_message
from ..crypto import merkle
from ..crypto.transcript import BoundSumcheckTranscript, ProofTranscript
from ..verifier import instance as I
from ..verifier.mle import contract_cols, contract_rows, dot, eq_table, log2_ceil
from ..runtime.session import PrivateRequest, SessionContext  # plain data classes (no logic)
from .toy_model import PAD, ToyModel, adapter_commitment, apply_adapter, combined_commitment, requant

LABEL = "TRANSPARENT_REFERENCE"
PROOF_FORMAT = "edgecert-transparent-proof/v1"

class ProverError(RuntimeError):
    pass

class ProverOutOfMemory(MemoryError):
    pass

class ProofDeadlineExceeded(TimeoutError):
    pass

class AcceleratorFault(RuntimeError):
    pass

# ================================================================ sumcheck backends
class HostSumcheck:
    """Plain host implementation of the SumFold protocol (golden-model ops)."""
    name = "python-host"

    def run(self, f: List[Ext], g: List[Ext], tr: BoundSumcheckTranscript, claim: Ext) -> dict:
        msgs, chals = [], []
        cf, cg = list(f), list(g)
        n = len(f).bit_length() - 1
        for _ in range(n):
            m = round_message(cf, cg)
            r = tr.round(*m)
            msgs.append(m)
            chals.append(r)
            cf, cg = fold(cf, r), fold(cg, r)
        return {"msgs": msgs, "challenges": chals, "final_f": cf[0], "final_g": cg[0], "device": None}

def _flip(e: Ext, bit: int) -> Ext:
    return (e[0] ^ (1 << bit), e[1])

class SumFoldLaneEmulator:
    """Software model of the L-lane SumFold device in STEP mode (spec/sumfold_kernel.md 3.2, 4).

    Rounds 0..R-1 run on lane-local contiguous blocks (per-lane partial
    messages summed by an adder tree), the host derives each challenge from
    the bound transcript, and the host finishes the last log2(L) rounds from
    the lane finals.  ``fault`` injects one accelerator/DMA error into job
    number ``fault['job']`` (0-based count of sumchecks run by this object):

        {"kind": "msg_bitflip", "round": i, "comp": k, "bit": b}
        {"kind": "lane_final_bitflip", "lane": l, "bit": b}
        {"kind": "stale_buffer"}                      # g-table load skipped, previous job's data used
        {"kind": "dropped_dma_block", "block": j, "block_elems": e}   # a DMA burst of f never lands

    With ``self_check=True`` the host checks s0 + s1 against its running claim
    after every device round and raises AcceleratorFault (the session then
    ABORTs instead of submitting a bad proof).
    """
    name = "sumfold-lane-emulator"

    def __init__(self, lanes: int = 4, fault: Optional[dict] = None, self_check: bool = False):
        assert lanes >= 1 and lanes & (lanes - 1) == 0
        self.lanes = lanes
        self.fault = fault
        self.self_check = self_check
        self.jobs = 0
        self._prev_g: Optional[List[Ext]] = None
        self.fault_fired = False

    def _check(self, m, cur):
        if self.self_check and eadd(m[0], m[1]) != cur:
            raise AcceleratorFault("host self-check: s0 + s1 != running claim")

    def run(self, f: List[Ext], g: List[Ext], tr: BoundSumcheckTranscript, claim: Ext) -> dict:
        job = self.jobs
        self.jobs += 1
        flt = self.fault if (self.fault and self.fault.get("job", 0) == job) else None
        N = len(f)
        n = N.bit_length() - 1
        L = max(1, min(self.lanes, N // 2))
        R = n - (L.bit_length() - 1)
        mem_f, mem_g = list(f), list(g)
        if flt and flt["kind"] == "dropped_dma_block":
            e = flt.get("block_elems", 8)
            lo = flt.get("block", 0) * e
            for k in range(lo, min(lo + e, N)):
                mem_f[k] = ZERO
            self.fault_fired = True
        if flt and flt["kind"] == "stale_buffer" and self._prev_g is not None:
            prev = self._prev_g
            mem_g = [prev[k] if k < len(prev) else ZERO for k in range(N)]
            self.fault_fired = True
        self._prev_g = list(g)
        msgs, chals = [], []
        cur = claim
        for i in range(R):
            blk = len(mem_f) // L
            parts = [round_message(mem_f[l * blk:(l + 1) * blk], mem_g[l * blk:(l + 1) * blk]) for l in range(L)]
            s = [ZERO, ZERO, ZERO]
            for pm in parts:  # adder tree (order irrelevant in the field)
                s = [eadd(s[k], pm[k]) for k in range(3)]
            m = tuple(s)
            if flt and flt["kind"] == "msg_bitflip" and flt["round"] == i:
                m = tuple(_flip(x, flt["bit"]) if k == flt["comp"] else x for k, x in enumerate(m))
                self.fault_fired = True
            self._check(m, cur)
            r = tr.round(*m)
            msgs.append(m)
            chals.append(r)
            cur = interp_quadratic(m[0], m[1], m[2], r)
            mem_f = [x for l in range(L) for x in fold(mem_f[l * blk:(l + 1) * blk], r)]
            mem_g = [x for l in range(L) for x in fold(mem_g[l * blk:(l + 1) * blk], r)]
        lane_f, lane_g = mem_f, mem_g            # one (f, g) pair per lane
        if flt and flt["kind"] == "lane_final_bitflip":
            lane_f = list(lane_f)
            lane_f[flt["lane"] % len(lane_f)] = _flip(lane_f[flt["lane"] % len(lane_f)], flt["bit"])
            self.fault_fired = True
        cf, cg = list(lane_f), list(lane_g)
        while len(cf) > 1:                       # host tail
            m = round_message(cf, cg)
            self._check(m, cur)
            r = tr.round(*m)
            msgs.append(m)
            chals.append(r)
            cur = interp_quadratic(m[0], m[1], m[2], r)
            cf, cg = fold(cf, r), fold(cg, r)
        return {"msgs": msgs, "challenges": chals, "final_f": cf[0], "final_g": cg[0],
                "device": {"lanes": L, "device_rounds": R, "host_tail_rounds": n - R}}

# ================================================================ tamper hooks
@dataclass
class Tamper:
    """Adversarial prover behaviours used by the attack suite (all default to honest)."""
    compute_tensors: Optional[Dict[str, list]] = None      # tensors used to COMPUTE (weights may differ)
    open_tensors: Optional[Dict[str, list]] = None         # tensors REVEALED in the model opening
    declared_model_commitment: Optional[str] = None       # instance model_commitment override
    declared_adapter: Optional[dict] = None               # {"adapter_id", "target", "delta"} revealed in proof
    lut: Optional[Dict[str, List[int]]] = None            # activation tables used to compute
    open_luts: Optional[Dict[str, List[int]]] = None      # tables revealed
    scales: Optional[Dict[int, Tuple[int, int]]] = None   # layer -> (M, s) used to compute
    rounding: Optional[str] = None                        # e.g. "TRUNC_FLOOR"
    forge_remainders: bool = False                        # recompute rem to satisfy the equality with manifest M,s
    witness_mutator: Optional[Callable[[dict], None]] = None
    unmask: bool = False
    position_offset: int = 0
    compute_prompt: Optional[List[int]] = None            # prompt actually used (commitment opened with this)
    output_mutator: Optional[Callable[[List[int]], List[int]]] = None  # teacher-forced trace on these tokens
    state_window: Optional[List[int]] = None              # corrupted carried state (and its opening)
    instance_mutator: Optional[Callable[[dict], dict]] = None
    proof_mutator: Optional[Callable[[dict], dict]] = None
    replay_proof: Optional[dict] = None                   # return this proof instead of proving
    return_none: bool = False                             # missing proof
    delay_s: float = 0.0                                  # slow prover (timeouts)

# ================================================================ helpers
def _T(cols: List[List[int]]) -> List[List[int]]:
    """columns (T x R) -> matrix (R x T)."""
    return [list(r) for r in zip(*cols)] if cols else []

def _ones_row(T: int) -> List[int]:
    return [1] * T

def _ext_list(xs) -> List[str]:
    return [ehex(x) for x in xs]

def table_order(n_layers: int) -> List[str]:
    names = ["Y0"]
    for l in range(n_layers):
        names += [f"L{l}.ACC", f"L{l}.Q", f"L{l}.REM", f"L{l}.SIG", f"L{l}.OUT"]
    return names

# ================================================================ prover
class TransparentProver:
    """Untrusted device: runs inference and produces a TRANSPARENT_REFERENCE proof."""

    def __init__(self, model: ToyModel, manifest_env: dict, backend=None, adapter: Optional[dict] = None,
                 mem_budget_bytes: Optional[int] = None, tamper: Optional[Tamper] = None):
        self.model = model
        self.manifest = manifest_env["body"]
        self.backend = backend or HostSumcheck()
        self.adapter = adapter           # authorized adapter {"adapter_id", "target", "delta"}
        self.mem_budget = mem_budget_bytes
        self.t = tamper or Tamper()
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------ tensors
    def _compute_tensors(self) -> Dict[str, list]:
        if self.t.compute_tensors is not None:
            return self.t.compute_tensors
        if self.adapter is not None:
            return apply_adapter(self.model.tensors, self.adapter["target"], self.adapter["delta"])
        if self.t.declared_adapter is not None:
            a = self.t.declared_adapter
            return apply_adapter(self.model.tensors, a["target"], a["delta"])
        return self.model.tensors

    def _state(self, priv: PrivateRequest) -> Tuple[List[int], int]:
        if priv.state is None:
            return [PAD] * self.model.cfg.window, 0
        w = self.t.state_window if self.t.state_window is not None else list(priv.state["window"])
        return w, int(priv.state["position"])

    # ------------------------------------------------------------ inference
    def infer(self, ctx: SessionContext, priv: PrivateRequest) -> List[int]:
        m = self.model
        sw, spos = self._state(priv)
        prompt = self.t.compute_prompt if self.t.compute_prompt is not None else priv.prompt
        tensors = self._compute_tensors()
        out = self._generate(sw, spos + self.t.position_offset, prompt, priv.max_new_tokens, tensors)
        if self.t.output_mutator is not None:
            out = self.t.output_mutator(list(out))
        return out

    def _generate(self, sw, spos, prompt, max_new, tensors) -> List[int]:
        m = self.model
        ctx = list(sw) + list(prompt)
        out: List[int] = []
        for i in range(max_new):
            win = (ctx + out)[-m.cfg.window:]
            x = m.gather(win, spos + len(prompt) + i, unmask=self.t.unmask)
            st = m.step(x, tensors, self.t.lut, self.t.scales, self.t.rounding)
            logits = st["logits"]
            tok = max(range(len(logits)), key=lambda j: (logits[j], -j))
            out.append(tok)
            if tok == m.cfg.eos_id:
                break
        return out

    # ------------------------------------------------------------ proving
    def prove(self, ctx: SessionContext, priv: PrivateRequest, out_tokens: List[int],
              deadline: Optional[float] = None) -> Optional[dict]:
        t = self.t
        if t.delay_s:
            time.sleep(t.delay_s)
        if t.return_none:
            return None
        if t.replay_proof is not None:
            return copy.deepcopy(t.replay_proof)
        proof = self._prove(ctx, priv, list(out_tokens), deadline)
        if t.proof_mutator is not None:
            proof = t.proof_mutator(proof)
        return proof

    def _check_deadline(self, deadline):
        if deadline is not None and time.monotonic() > deadline:
            raise ProofDeadlineExceeded("prover passed its proof deadline")

    def _prove(self, ctx: SessionContext, priv: PrivateRequest, out: List[int], deadline) -> dict:
        t0 = time.perf_counter()
        m, t, body = self.model, self.t, self.manifest
        tensors = self._compute_tensors()
        sw, spos = self._state(priv)
        prompt = t.compute_prompt if t.compute_prompt is not None else list(priv.prompt)
        T = len(out)
        if T == 0:
            raise ProverError("empty output")
        # ---------------- witness generation (teacher-forced on `out`)
        cols_x, layers_rec = [], []
        wins = m.windows(sw, spos + t.position_offset, prompt, out)
        for (win, pos) in wins:
            x = m.gather(win, pos, unmask=t.unmask)
            st = m.step(x, tensors, t.lut, t.scales, t.rounding)
            cols_x.append(x)
            layers_rec.append(st["layers"])
        nl = len(m.layers)
        W = {"Y0": _T(cols_x)}
        for l in range(nl):
            W[f"L{l}.ACC"] = _T([rec[l]["acc"] for rec in layers_rec])
            W[f"L{l}.Q"] = _T([rec[l]["q"] for rec in layers_rec])
            W[f"L{l}.REM"] = _T([rec[l]["rem"] for rec in layers_rec])
            W[f"L{l}.SIG"] = _T([rec[l]["sig"] for rec in layers_rec])
            W[f"L{l}.OUT"] = _T([rec[l]["out"] for rec in layers_rec])
        qlayers = body["quantizer"]["layers"]
        if t.forge_remainders:   # make acc*M + h - sigma = q*2^s + rem hold with the AUTHORIZED M, s (RHAZ)
            for l in range(nl):
                M, s = qlayers[l]["multiplier"], qlayers[l]["shift"]
                h = 1 << (s - 1)
                A, Q = W[f"L{l}.ACC"], W[f"L{l}.Q"]
                SG = [[1 if A[i][j] * M < 0 else 0 for j in range(T)] for i in range(len(A))]
                W[f"L{l}.SIG"] = SG
                W[f"L{l}.REM"] = [[A[i][j] * M + h - SG[i][j] - (Q[i][j] << s) for j in range(T)]
                                  for i in range(len(A))]
        if t.witness_mutator is not None:
            t.witness_mutator(W)
        t_witness = time.perf_counter()
        # ---------------- memory admission (prototype, bytes)
        K_pads = [1 << max(1, log2_ceil(L_["in"] + 1)) for L_ in m.layers]
        est = 8 * sum(len(v) * len(v[0]) for v in W.values() if v) + sum(32 * k for k in K_pads) \
            + 8 * sum(len(merkle.flatten(v)[0]) for v in tensors.values())
        self.last_stats["witness_bytes_est"] = est
        if self.mem_budget is not None and est > self.mem_budget:
            raise ProverOutOfMemory(f"working set B exceeds budget {self.mem_budget} B")
        self._check_deadline(deadline)
        # ---------------- end state + instance
        ew, epos = m.end_state(sw, spos, prompt, out)
        ecomm, eopen = I.commit_state(ew, epos)
        if ctx.output_mode == "tokens":
            ofield, oopen = I.output_field_tokens(out), None
        else:
            oc, oo = I.commit_output(out)
            ofield, oopen = I.output_field_commitment(oc.hex()), {"randomness": oo.randomness.hex()}
        if self.adapter is not None:
            mc = combined_commitment(body["model"]["model_commitment"],
                                     adapter_commitment(self.adapter["target"], self.adapter["delta"]))
        else:
            mc = body["model"]["model_commitment"]
        if t.declared_model_commitment is not None:
            mc = t.declared_model_commitment
        inst = I.make_instance(
            manifest_id=ctx.manifest_id, model_commitment=mc,
            relation_id=body["proof_system"]["relation_id"],
            proof_config_id=body["proof_system"]["proof_config_id"],
            session_id=ctx.session_id, requester_nonce=ctx.requester_nonce,
            input_commitment=ctx.input_commitment, output_tokens_or_commitment=ofield,
            context_anchor=ctx.context_anchor, output_length=T,
            previous_receipt_id=ctx.previous_receipt_id, policy_epoch=ctx.policy_epoch)
        if t.instance_mutator is not None:
            inst = t.instance_mutator(inst)
        digest = I.instance_digest(inst)
        # ---------------- commit witness tables, then derive challenges
        names = table_order(nl)
        roots = {nm: merkle.commit_tensor(W[nm]).root.hex() for nm in names}
        G = ProofTranscript(digest)
        G.absorb("end_state_commitment", ecomm)
        for nm in names:
            G.absorb("root:" + nm, bytes.fromhex(roots[nm]))
        t_commit = time.perf_counter()
        sumchecks = []
        a_t = log2_ceil(T)
        Y = W["Y0"]
        for l, L_ in enumerate(m.layers):
            self._check_deadline(deadline)
            Wa = m.weight_aug(L_, tensors)
            a = log2_ceil(L_["out"])
            b = max(1, log2_ceil(L_["in"] + 1))
            r1 = G.challenge_vec(f"L{l}.r_row", a)
            r2 = G.challenge_vec(f"L{l}.r_col", a_t)
            f = contract_rows(Wa, eq_table(r1), 1 << b)
            g = contract_cols(Y, eq_table(r2), 1 << b)
            claim = dot(f, g)
            tr = BoundSumcheckTranscript(b, claim, G.state, f"L{l}.matmul")
            res = self.backend.run(f, g, tr, claim)
            G.absorb(f"L{l}.sumcheck_state", tr.state)
            G.absorb_ext(f"L{l}.final_f", res["final_f"])
            G.absorb_ext(f"L{l}.final_g", res["final_g"])
            sumchecks.append({"layer": l, "n": b, "claim": ehex(claim),
                              "msgs": [_ext_list(mm) for mm in res["msgs"]],
                              "final_f": ehex(res["final_f"]), "final_g": ehex(res["final_g"]),
                              "device": res.get("device")})
            Y = W[f"L{l}.OUT"] + [_ones_row(T)]
        t_sc = time.perf_counter()
        # ---------------- openings
        open_tensors = t.open_tensors if t.open_tensors is not None else self.model.tensors
        mcom = merkle.commit_model(open_tensors)
        model_opening = {"tensors": {nm: {"values": open_tensors[nm], "opening": mcom.open(nm)}
                                     for nm in mcom.names},
                         "adapters": [], "luts": dict(t.open_luts if t.open_luts is not None else self.model.luts)}
        adp = self.adapter or t.declared_adapter
        if adp is not None:
            model_opening["adapters"].append({"adapter_id": adp["adapter_id"], "target": adp["target"],
                                              "delta": adp["delta"]})
        proof = {
            "format": PROOF_FORMAT, "label": LABEL, "zero_knowledge": False, "succinct": False,
            "instance": inst, "instance_digest": digest.hex(),
            "model_opening": model_opening,
            "input_opening": {"tokens": list(prompt), "max_new_tokens": priv.max_new_tokens,
                              "tokenizer_id": body["tokenizer"]["id"],
                              "randomness": priv.input_randomness.hex()},
            "state_opening": None if priv.state is None else {
                "window": list(sw), "position": spos,
                "randomness": priv.state["randomness"].hex() if isinstance(priv.state["randomness"], bytes)
                else priv.state["randomness"]},
            "end_state": {"commitment": ecomm.hex(), "window": ew, "position": epos,
                          "randomness": eopen.randomness.hex()},
            "output_opening": oopen,
            "trace": {"output_tokens": list(out)},
            "witness": {nm: W[nm] for nm in names},
            "roots": roots,
            "sumchecks": sumchecks,
            "meta": {"backend": self.backend.name, "transcript_labels": len(G.log)},
        }
        t_end = time.perf_counter()
        self.last_stats.update({"witness_s": t_witness - t0, "commit_s": t_commit - t_witness,
                                "sumcheck_s": t_sc - t_commit, "open_s": t_end - t_sc, "total_s": t_end - t0})
        return proof
