"""Independent verifier for EDGE-CERT TRANSPARENT_REFERENCE proofs.

Independence rule: this module imports only shared math and data formats (edgecert.field, edgecert.sumfold verifier-side functions, edgecert.crypto, edgecert.verifier.{instance, mle}, edgecert.runtime.manifest, edgecert.audit.receipts). It never imports edgecert.prover_adapters.  Gather, requantization, lookup, argmax and termination semantics are re-implemented here from the signed manifest, not taken from the prover's model code.

The verifier is the enforcement point of the certification contract (C1): it accepts a proof only if EVERY check passes, it fails closed on any exception, and it issues a signed receipt only on acceptance.  It runs outside the prover's administrative boundary (spec/threat_model.md).

Check order (the first failing check is reported as the detecting check): format.* -> manifest.* -> instance.digest -> instance.session_binding -> instance.relation_config -> policy.epoch -> policy.model_commitment_authorized -> session.nonce_fresh -> receipt.previous_valid -> context.anchor_opening -> input.commitment_opening -> model.graph_supported -> model.tensor_openings -> model.adapter_authorized -> model.lut_digest -> range.static_certificate -> output.binding -> decode.termination_eos_length -> witness.shapes -> witness.commitments -> gather.embedding_position_mask -> per layer: sumcheck.claim / sumcheck.shape / sumcheck.round_consistency / sumcheck.final_product / sumcheck.final_opening_weights / sumcheck.final_opening_activations / requant.* / lookup.membership -> decode.argmax_tiebreak -> state.end_state_opening
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field as dfield
from typing import Any, Dict, List, Optional, Tuple

from ..field import P, Ext, eadd, emul
from ..sumfold import interp_quadratic
from ..crypto import ed25519, merkle
from ..crypto.transcript import BoundSumcheckTranscript, ProofTranscript
from ..runtime.manifest import TrustPolicy, authorized_model_commitments, digest_obj
from ..audit import receipts as R
from . import instance as I
from .mle import contract_cols, contract_rows, dot, eq_table, log2_ceil, mle_eval_matrix

SUPPORTED_PROOF_CONFIGS = {"TRANSPARENT_REFERENCE/sumcheck-goldilocks-ext2/v1"}
SUPPORTED_GRAPHS = {"window-mlp-lm/v1"}
PROOF_FORMAT = "edgecert-transparent-proof/v1"
LABEL = "TRANSPARENT_REFERENCE"

class _Reject(Exception):
    def __init__(self, check: str, reason: str = ""):
        super().__init__(f"{check}: {reason}")
        self.check = check
        self.reason = reason

def _need(cond: bool, check: str, reason: str = "") -> None:
    if not cond:
        raise _Reject(check, reason)

@dataclass
class Expectation:
    """What the relying party itself knows about the session (never taken from the prover)."""
    manifest_id: str
    session_id: str
    requester_nonce: str
    input_commitment: str
    previous_receipt_id: Optional[str]      # None = auditor mode: take from instance, validate against store
    output_mode: str = "tokens"

@dataclass
class VerificationResult:
    accepted: bool
    failed_check: Optional[str]
    reason: str
    checks_passed: List[str] = dfield(default_factory=list)
    receipt: Optional[dict] = None
    verify_s: float = 0.0

def _parse_ext(h: Any, check: str) -> Ext:
    _need(isinstance(h, str) and len(h) == 32, check, "malformed extension element")
    v = int(h, 16)
    c0, c1 = v & ((1 << 64) - 1), v >> 64
    _need(c0 < P and c1 < P, check, "non-canonical field element")
    return (c0, c1)

def _is_int_matrix(M: Any, rows: int, cols: int) -> bool:
    if not isinstance(M, list) or len(M) != rows:
        return False
    for r in M:
        if not isinstance(r, list) or len(r) != cols:
            return False
        for v in r:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
    return True

def _unlift(v: int, bound: int) -> Optional[int]:
    """Field element -> signed integer in [-bound, bound], else None."""
    a = v % P
    if a <= bound:
        return a
    if a >= P - bound:
        return a - P
    return None

class Verifier:
    def __init__(self, trust: TrustPolicy, key: Optional[ed25519.SigningKey] = None,
                 trusted_verifier_keys: Optional[Dict[str, bytes]] = None):
        self.trust = trust
        self.key = key or ed25519.SigningKey()
        self.trusted_verifier_keys = dict(trusted_verifier_keys or {})
        self.trusted_verifier_keys[self.key.key_id] = self.key.public
        self.receipts: Dict[str, dict] = {}
        self.consumed_nonces: set = set()

    # ------------------------------------------------------------ policy
    def authorize_manifest(self, envelope: dict, now: Optional[int] = None) -> str:
        return self.trust.accept(envelope, now)

    def import_receipt(self, receipt: dict) -> Tuple[bool, str]:
        ok, why = R.check_receipt(receipt, self.trusted_verifier_keys)
        if ok:
            self.receipts[receipt["receipt_id"]] = receipt
        return ok, why

    # ------------------------------------------------------------ verify
    def verify(self, proof: Any, exp: Expectation) -> VerificationResult:
        t0 = time.perf_counter()
        passed: List[str] = []
        try:
            receipt = self._verify(proof, exp, passed)
            return VerificationResult(True, None, "ok", passed, receipt, time.perf_counter() - t0)
        except _Reject as r:
            return VerificationResult(False, r.check, r.reason, passed, None, time.perf_counter() - t0)
        except Exception as e:  # fail closed on anything unexpected
            return VerificationResult(False, "format.malformed", f"{type(e).__name__}: {e}", passed, None,
                                      time.perf_counter() - t0)

    def _verify(self, pf: dict, exp: Expectation, passed: List[str]) -> dict:
        ok = passed.append
        # ---------------------------------------------------------- format
        _need(isinstance(pf, dict), "format.missing_proof", "no proof object")
        _need(pf.get("format") == PROOF_FORMAT and pf.get("label") == LABEL, "format.label",
              "unsupported proof format/label")
        for k in ("instance", "instance_digest", "model_opening", "input_opening", "end_state", "trace",
                  "witness", "roots", "sumchecks"):
            _need(k in pf, "format.label", f"missing {k}")
        ok("format.label")
        inst = pf["instance"]
        # ---------------------------------------------------------- manifest
        mid = inst.get("manifest_id")
        _need(mid == exp.manifest_id, "manifest.id_match", "instance names a different manifest")
        cur, why = self.trust.current(mid)
        _need(cur, "manifest.current", why)
        body = self.trust.accepted[mid].body
        ok("manifest.current")
        # ---------------------------------------------------------- instance
        _need(set(inst) == set(I.INSTANCE_FIELDS), "instance.digest", "wrong field set")
        digest = I.instance_digest(inst)
        _need(digest.hex() == pf["instance_digest"], "instance.digest", "digest mismatch")
        ok("instance.digest")
        _need(inst["session_id"] == exp.session_id and inst["requester_nonce"] == exp.requester_nonce
              and inst["input_commitment"] == exp.input_commitment, "instance.session_binding",
              "session id / nonce / input commitment differ from the requester's")
        if exp.previous_receipt_id is not None:
            _need(inst["previous_receipt_id"] == exp.previous_receipt_id, "instance.session_binding",
                  "previous receipt differs from the requester's")
        _need(inst["output_tokens_or_commitment"].get("mode") == exp.output_mode, "instance.session_binding",
              "output mode")
        ok("instance.session_binding")
        ps = body["proof_system"]
        _need(inst["relation_id"] == ps["relation_id"] and inst["proof_config_id"] == ps["proof_config_id"],
              "instance.relation_config", "relation/proof config not the manifest's")
        _need(inst["proof_config_id"] in SUPPORTED_PROOF_CONFIGS, "instance.relation_config",
              "verifier does not support this proof configuration (fail closed)")
        ok("instance.relation_config")
        floor = self.trust.epoch_floor.get((body["publisher"]["key_id"], body["model"]["family"]), 0)
        _need(inst["policy_epoch"] == body["policy"]["policy_epoch"] and inst["policy_epoch"] >= floor,
              "policy.epoch", "policy epoch mismatch or rolled back")
        ok("policy.epoch")
        auth = authorized_model_commitments(body)
        _need(inst["model_commitment"] in auth, "policy.model_commitment_authorized",
              "model commitment not authorized by the manifest")
        adapter_rec = auth[inst["model_commitment"]]
        ok("policy.model_commitment_authorized")
        _need(inst["requester_nonce"] not in self.consumed_nonces, "session.nonce_fresh",
              "nonce already consumed (replay)")
        ok("session.nonce_fresh")
        # ---------------------------------------------------------- previous receipt / context anchor
        so = pf.get("state_opening")
        gs = body["graph"]["spec"]
        Wn, V, pad = gs["window"], gs["vocab"], gs["pad_id"]
        if inst["previous_receipt_id"] == I.GENESIS_RECEIPT_ID:
            _need(inst["context_anchor"] == I.GENESIS_ANCHOR and so is None, "receipt.previous_valid",
                  "genesis session must use the genesis anchor")
            s_window, s_pos = [pad] * Wn, 0
            ok("receipt.previous_valid")
        else:
            prev = self.receipts.get(inst["previous_receipt_id"])
            _need(prev is not None, "receipt.previous_valid", "previous receipt unknown or forged")
            rok, rwhy = R.check_receipt(prev, self.trusted_verifier_keys)
            _need(rok, "receipt.previous_valid", rwhy)
            _need(prev["body"]["manifest_id"] == mid and prev["body"]["end_state_commitment"] == inst["context_anchor"],
                  "receipt.previous_valid", "context anchor is not the previous receipt's end state")
            ok("receipt.previous_valid")
            _need(isinstance(so, dict), "context.anchor_opening", "missing state opening")
            _need(isinstance(so.get("window"), list) and len(so["window"]) == Wn
                  and all(isinstance(x, int) and (x == pad or 0 <= x < V) for x in so["window"])
                  and isinstance(so.get("position"), int), "context.anchor_opening", "malformed state")
            _need(I.check_state_opening(inst["context_anchor"], so["window"], so["position"], so["randomness"]),
                  "context.anchor_opening", "state opening does not match the context anchor")
            s_window, s_pos = list(so["window"]), so["position"]
            ok("context.anchor_opening")
        # ---------------------------------------------------------- input opening
        io = pf["input_opening"]
        tok = body["tokenizer"]
        _need(io.get("tokenizer_id") == tok["id"], "input.commitment_opening", "tokenizer mismatch")
        _need(I.check_input_opening(inst["input_commitment"], io["tokens"], io["max_new_tokens"], io["tokenizer_id"],
                                    io["randomness"]), "input.commitment_opening", "input opening invalid")
        prompt = io["tokens"]
        max_new = io["max_new_tokens"]
        _need(all(isinstance(x, int) and 0 <= x < V for x in prompt), "input.commitment_opening", "token range")
        _need(isinstance(max_new, int) and 1 <= max_new <= body["decoding"]["max_new_tokens_cap"],
              "input.commitment_opening", "generation budget outside manifest cap")
        ok("input.commitment_opening")
        # ---------------------------------------------------------- graph / model opening
        _need(digest_obj("EDGE-CERT/graph/v1", gs) == body["graph"]["graph_digest"], "model.graph_supported",
              "graph digest")
        _need(gs.get("type") in SUPPORTED_GRAPHS, "model.graph_supported", "unsupported graph type")
        lut_names = {t["name"] for t in body["lookup_tables"]}
        qlayers = body["quantizer"]["layers"]
        _need(len(qlayers) == len(gs["layers"]), "model.graph_supported", "quantizer/graph layer mismatch")
        _need(body["quantizer"]["rounding"] == "RHAZ", "model.graph_supported", "unsupported rounding rule")
        for L_, q in zip(gs["layers"], qlayers):
            _need(L_["activation"] is None or L_["activation"] in lut_names, "model.graph_supported",
                  f"unsupported operator {L_['activation']!r} (fail closed)")
            _need(q["name"] == L_["name"] and q.get("activation") == L_["activation"], "model.graph_supported",
                  "quantizer entry does not match layer")
        for a, b in zip(gs["layers"], gs["layers"][1:]):
            _need(b["in"] == a["out"], "model.graph_supported", "layer chaining dims")
        _need(gs["layers"][0]["in"] == Wn * gs["d_emb"] + gs["d_emb"], "model.graph_supported", "input dim")
        _need(gs["layers"][-1]["out"] >= V, "model.graph_supported", "logit rows")
        ok("model.graph_supported")
        base_mc = body["model"]["model_commitment"]
        mo = pf["model_opening"]
        shapes = gs["tensor_shapes"]
        _need(set(mo["tensors"]) == set(shapes), "model.tensor_openings", "tensor set differs from graph")
        tensors: Dict[str, list] = {}
        for nm, shp in shapes.items():
            ent = mo["tensors"][nm]
            vals = ent["values"]
            flat, shape = merkle.flatten(vals)
            _need(list(shape) == list(shp), "model.tensor_openings", f"shape of {nm}")
            _need(ent["opening"]["name"] == nm and ent["opening"]["count"] == len(shapes), "model.tensor_openings",
                  "opening metadata")
            _need(merkle.verify_model_opening(bytes.fromhex(base_mc), ent["opening"], vals),
                  "model.tensor_openings", f"opening of {nm} does not match the model commitment")
            tensors[nm] = vals
        ok("model.tensor_openings")
        adapters = mo.get("adapters", [])
        if adapter_rec is None:
            _need(len(adapters) == 0, "model.adapter_authorized", "adapter present but not authorized")
        else:
            _need(len(adapters) == 1, "model.adapter_authorized", "authorized adapter missing")
            ad = adapters[0]
            _need(ad["adapter_id"] == adapter_rec["adapter_id"] and ad["target"] == adapter_rec["target"]
                  and ad["target"] in shapes, "model.adapter_authorized", "adapter id/target")
            dflat, dshape = merkle.flatten(ad["delta"])
            _need(list(dshape) == list(shapes[ad["target"]]), "model.adapter_authorized", "adapter shape")
            acom = merkle.commit_model({ad["target"] + ".delta": ad["delta"]}).commitment.hex()
            _need(acom == adapter_rec["adapter_commitment"], "model.adapter_authorized", "adapter commitment")
            comb = hashlib.sha256(b"EDGE-CERT/model+adapter/v1" + bytes.fromhex(base_mc) + bytes.fromhex(acom)).hexdigest()
            _need(comb == inst["model_commitment"], "model.adapter_authorized", "combined commitment")
            tensors[ad["target"]] = [[a + d for a, d in zip(ra, rd)] for ra, rd in zip(tensors[ad["target"]], ad["delta"])]
        ok("model.adapter_authorized")
        luts: Dict[str, List[int]] = {}
        lo_d, hi_d = gs["activation_domain"]
        for tdesc in body["lookup_tables"]:
            tbl = mo["luts"].get(tdesc["name"])
            _need(isinstance(tbl, list) and len(tbl) == hi_d - lo_d + 1, "model.lut_digest", "table shape")
            _need(digest_obj("EDGE-CERT/lut/v1", {"name": tdesc["name"], "domain": tdesc["domain"], "table": tbl})
                  == tdesc["digest"], "model.lut_digest", f"table {tdesc['name']} does not match the manifest")
            luts[tdesc["name"]] = tbl
        ok("model.lut_digest")
        # ---------------------------------------------------------- static range certificate (TileContract-style)
        cl_lo, cl_hi = gs["clamp"]
        Bx = max(abs(lo_d), abs(hi_d), abs(cl_lo), abs(cl_hi))
        bounds = []
        for L_, q in zip(gs["layers"], qlayers):
            Wt, bt = tensors[L_["weight"]], tensors[L_["bias"]]
            Bw = max(abs(v) for r in Wt for v in r)
            Bb = max(abs(v) for v in bt)
            Bx_l = max(Bx, max(abs(v) for r in tensors["emb"] for v in r),
                       max(abs(v) for r in tensors["pos"] for v in r))
            B_acc = L_["in"] * Bw * Bx_l + Bb
            M, s = q["multiplier"], q["shift"]
            h = 1 << (s - 1) if s > 0 else 0
            B_num = B_acc * M + h                    # bounds |n| and |n + h - sigma|
            B_q = (B_num >> s) + 1
            _need(2 * max(B_num, B_q * (1 << s) + (1 << s), B_acc) < P, "range.static_certificate",
                  "p <= 2B: wrap-around equalities not excluded")
            bounds.append((B_acc, B_q, M, s, h))
        ok("range.static_certificate")
        # ---------------------------------------------------------- output binding + termination
        T = inst["output_length"]
        out = pf["trace"]["output_tokens"]
        _need(isinstance(T, int) and 1 <= T <= max_new and isinstance(out, list) and len(out) == T,
              "output.binding", "output length")
        _need(all(isinstance(x, int) and 0 <= x < V for x in out), "output.binding", "token range")
        of = inst["output_tokens_or_commitment"]
        if of["mode"] == "tokens":
            _need(of.get("tokens") == out, "output.binding", "instance tokens differ from the proven trace")
        else:
            oo = pf.get("output_opening") or {}
            _need(I.check_output_opening(of.get("commitment", ""), out, oo.get("randomness", "")), "output.binding",
                  "output commitment opening")
        ok("output.binding")
        eos = body["decoding"]["eos_id"]
        _need(eos not in out[:-1], "decode.termination_eos_length", "tokens after EOS")
        _need(out[-1] == eos or T == max_new, "decode.termination_eos_length",
              "generation stopped before EOS and before the length budget")
        ok("decode.termination_eos_length")
        # ---------------------------------------------------------- witness shapes and commitments
        wit = pf["witness"]
        layers = gs["layers"]
        K0 = layers[0]["in"] + 1
        _need(_is_int_matrix(wit.get("Y0"), K0, T), "witness.shapes", "Y0")
        names = ["Y0"]
        for l, L_ in enumerate(layers):
            for kind in ("ACC", "Q", "REM", "SIG", "OUT"):
                nm = f"L{l}.{kind}"
                _need(_is_int_matrix(wit.get(nm), L_["out"], T), "witness.shapes", nm)
                names.append(nm)
        ok("witness.shapes")
        roots = pf["roots"]
        _need(set(roots) == set(names), "witness.commitments", "root set")
        for nm in names:
            _need(merkle.commit_tensor(wit[nm]).root.hex() == roots[nm], "witness.commitments",
                  f"root of {nm} does not match the revealed table")
        ok("witness.commitments")
        # ---------------------------------------------------------- gather (embedding, positional, mask)
        emb, pos_t = tensors[gs["embedding"]], tensors[gs["positional"]]
        ctx_tokens = s_window + list(prompt)
        Y0 = wit["Y0"]
        for i in range(T):
            win = (ctx_tokens + out[:i])[-Wn:]
            p_i = s_pos + len(prompt) + i
            _need(0 <= p_i < gs["max_position"], "gather.embedding_position_mask", "position beyond max_position")
            col: List[int] = []
            for tkn in win:
                col += [0] * gs["d_emb"] if tkn == pad else list(emb[tkn])
            col += list(pos_t[p_i]) + [1]
            _need(all(Y0[k][i] == col[k] for k in range(K0)), "gather.embedding_position_mask",
                  f"input column {i} is not the committed-context gather")
        ok("gather.embedding_position_mask")
        # ---------------------------------------------------------- transcript + per-layer checks
        G = ProofTranscript(digest)
        es = pf["end_state"]
        G.absorb("end_state_commitment", bytes.fromhex(es["commitment"]))
        for nm in names:
            G.absorb("root:" + nm, bytes.fromhex(roots[nm]))
        scs = pf["sumchecks"]
        _need(isinstance(scs, list) and len(scs) == len(layers), "sumcheck.shape", "sumcheck count")
        a_t = log2_ceil(T)
        Y = Y0
        for l, (L_, (B_acc, B_q, M, s, h)) in enumerate(zip(layers, bounds)):
            sc = scs[l]
            Wa = [list(tensors[L_["weight"]][i]) + [tensors[L_["bias"]][i]] for i in range(L_["out"])]
            a = log2_ceil(L_["out"])
            b = max(1, log2_ceil(L_["in"] + 1))
            r1 = G.challenge_vec(f"L{l}.r_row", a)
            r2 = G.challenge_vec(f"L{l}.r_col", a_t)
            ACC = wit[f"L{l}.ACC"]
            claim_v = mle_eval_matrix(ACC, r1, r2)
            claim_p = _parse_ext(sc.get("claim"), "sumcheck.claim")
            _need(claim_p == claim_v, "sumcheck.claim", f"layer {l}: claim is not ACC~(r1, r2)")
            _need(sc.get("n") == b and isinstance(sc.get("msgs"), list) and len(sc["msgs"]) == b, "sumcheck.shape",
                  f"layer {l}: wrong number of rounds")
            tr = BoundSumcheckTranscript(b, claim_v, G.state, f"L{l}.matmul")
            cur = claim_v
            rk: List[Ext] = []
            for i, msg in enumerate(sc["msgs"]):
                _need(isinstance(msg, list) and len(msg) == 3, "sumcheck.shape", "message arity")
                s0, s1, s2 = (_parse_ext(x, "sumcheck.round_consistency") for x in msg)
                _need(eadd(s0, s1) == cur, "sumcheck.round_consistency", f"layer {l} round {i}: s0 + s1 != claim")
                r = tr.round(s0, s1, s2)
                rk.append(r)
                cur = interp_quadratic(s0, s1, s2, r)
            ff = _parse_ext(sc.get("final_f"), "sumcheck.final_product")
            fg = _parse_ext(sc.get("final_g"), "sumcheck.final_product")
            _need(emul(ff, fg) == cur, "sumcheck.final_product", f"layer {l}: final product != reduced claim")
            ek = eq_table(rk)
            f_eval = dot(contract_rows(Wa, eq_table(r1), 1 << b), ek)
            _need(f_eval == ff, "sumcheck.final_opening_weights",
                  f"layer {l}: final_f is not Waug~(r1, r_k) of the committed weights")
            g_eval = dot(contract_cols(Y, eq_table(r2), 1 << b), ek)
            _need(g_eval == fg, "sumcheck.final_opening_activations",
                  f"layer {l}: final_g is not Y~(r_k, r2) of the committed activations")
            G.absorb(f"L{l}.sumcheck_state", tr.state)
            G.absorb_ext(f"L{l}.final_f", ff)
            G.absorb_ext(f"L{l}.final_g", fg)
            # requantization (RHAZ) + activation, in constraint form over F_p:
            #   sigma*(sigma-1) = 0,  (1-2 sigma)*n - sigma in [0, B_num],  rem in [0, 2^s),
            #   n + h - sigma = q*2^s + rem,  q in [-B_q, B_q],  out = clamp(q) or LUT[clamp(q)]
            Q, REM, SIG, OUT = wit[f"L{l}.Q"], wit[f"L{l}.REM"], wit[f"L{l}.SIG"], wit[f"L{l}.OUT"]
            two_s = 1 << s
            act = L_["activation"]
            for i in range(L_["out"]):
                for j in range(T):
                    acc_i = _unlift(ACC[i][j], B_acc)
                    _need(acc_i is not None, "requant.accumulator_range", f"layer {l} ({i},{j})")
                    q_i = _unlift(Q[i][j], B_q)
                    _need(q_i is not None, "requant.quotient_range", f"layer {l} ({i},{j})")
                    sg = SIG[i][j] % P
                    _need(sg * (sg - 1) % P == 0, "requant.sign_bit", f"layer {l} ({i},{j}): sigma not boolean")
                    n_i = acc_i * M
                    sgn = _unlift(((1 - 2 * sg) * n_i - sg) % P, B_num)
                    _need(sgn is not None and sgn >= 0, "requant.sign_bit",
                          f"layer {l} ({i},{j}): sigma is not the sign of acc*M")
                    rem_f = REM[i][j] % P
                    _need(rem_f < two_s, "requant.remainder_range", f"layer {l} ({i},{j}): remainder not in [0, 2^s)")
                    _need((n_i + h - sg - q_i * two_s - rem_f) % P == 0, "requant.equality",
                          f"layer {l} ({i},{j}): acc*M + h - sigma != q*2^s + rem")
                    z = cl_lo if q_i < cl_lo else cl_hi if q_i > cl_hi else q_i
                    if act is None:
                        _need(OUT[i][j] == z, "requant.clamp", f"layer {l} ({i},{j})")
                    else:
                        _need(OUT[i][j] == luts[act][z - lo_d], "lookup.membership",
                              f"layer {l} ({i},{j}): (z, out) not in table {act}")
            Y = OUT + [[1] * T]
            ok(f"layer{l}")
        # ---------------------------------------------------------- argmax / tie-break
        LOG = wit[f"L{len(layers) - 1}.OUT"]
        for j in range(T):
            tk = out[j]
            for v in range(V):
                d = LOG[tk][j] - LOG[v][j] - (1 if v < tk else 0)
                _need(0 <= d <= 2 * 256, "decode.argmax_tiebreak",
                      f"step {j}: token {tk} is not the lowest-index argmax")
        ok("decode.argmax_tiebreak")
        # ---------------------------------------------------------- end state
        full = s_window + list(prompt) + out
        ew, epos = full[-Wn:], s_pos + len(prompt) + T
        _need(es.get("window") == ew and es.get("position") == epos and
              I.check_state_opening(es["commitment"], ew, epos, es["randomness"]), "state.end_state_opening",
              "end-state commitment does not open to the certified context")
        ok("state.end_state_opening")
        # ---------------------------------------------------------- accept: consume nonce, issue receipt
        self.consumed_nonces.add(inst["requester_nonce"])
        rbody = {"receipt_version": R.RECEIPT_VERSION, "session_id": inst["session_id"],
                 "instance_digest": digest.hex(), "manifest_id": mid, "model_commitment": inst["model_commitment"],
                 "relation_id": inst["relation_id"], "proof_config_id": inst["proof_config_id"],
                 "certificate_kind": LABEL, "verdict": "VERIFIED", "end_state_commitment": es["commitment"],
                 "previous_receipt_id": inst["previous_receipt_id"], "output_length": T,
                 "policy_epoch": inst["policy_epoch"], "verifier_key_id": self.key.key_id}
        rc = R.issue_receipt(rbody, self.key)
        self.receipts[rc["receipt_id"]] = rc
        return rc
