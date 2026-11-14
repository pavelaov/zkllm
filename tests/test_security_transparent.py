"""Unit tests: transparent reference prover, SumFold lane emulator fidelity, independent verifier."""
import ast
import copy
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgecert import sumfold  # noqa: E402
from edgecert.field import P, ehex  # noqa: E402
from edgecert.crypto.ed25519 import SigningKey  # noqa: E402
from edgecert.runtime.manifest import TrustPolicy, sign_manifest  # noqa: E402
from edgecert.runtime.session import PrivateRequest, SessionContext  # noqa: E402
from edgecert.verifier import instance as I  # noqa: E402
from edgecert.verifier.mle import contract_cols, contract_rows, dot, eq_table, mle_eval_matrix  # noqa: E402
from edgecert.verifier.transparent_verifier import Expectation, Verifier  # noqa: E402
from edgecert.prover_adapters.toy_model import ToyConfig, build_model, manifest_body, requant  # noqa: E402
from edgecert.prover_adapters.transparent import (HostSumcheck, SumFoldLaneEmulator, Tamper,  # noqa: E402
                                                  TransparentProver)
from edgecert.prover_adapters.deepprove_adapter import ADAPTERS  # noqa: E402
from edgecert.prover_adapters.base import BackendNotAvailable  # noqa: E402

CFG = ToyConfig(d_emb=8, hidden=(16,), vocab=16, max_position=32, eos_bias=0)

def setup(cfg=CFG, output_mode="tokens"):
    key = SigningKey()
    m = build_model(cfg)
    env = sign_manifest(manifest_body(m, key, epoch=1), key)
    v = Verifier(TrustPolicy({key.key_id: key.public}))
    mid = v.authorize_manifest(env)
    prompt, max_new = [3, 5, 7], 5
    ic, io = I.commit_input(prompt, max_new, m.tokenizer["id"])
    ctx = SessionContext(mid, "ab" * 8, "cd" * 32, ic.hex(), I.GENESIS_RECEIPT_ID, I.GENESIS_ANCHOR, 1, output_mode)
    priv = PrivateRequest(prompt, max_new, io.randomness)
    exp = Expectation(mid, ctx.session_id, ctx.requester_nonce, ctx.input_commitment, I.GENESIS_RECEIPT_ID,
                      output_mode)
    return m, env, v, ctx, priv, exp

class TestMLE(unittest.TestCase):
    def test_matmul_identity(self):
        rng = random.Random(3)
        A = [[rng.randint(-50, 50) for _ in range(5)] for _ in range(3)]      # 3 x 5
        B = [[rng.randint(-50, 50) for _ in range(6)] for _ in range(5)]      # 5 x 6
        C = [[sum(A[i][k] * B[k][j] for k in range(5)) for j in range(6)] for i in range(3)]
        r1 = [(rng.randrange(P), rng.randrange(P)) for _ in range(2)]
        r2 = [(rng.randrange(P), rng.randrange(P)) for _ in range(3)]
        f = contract_rows(A, eq_table(r1), 8)
        g = contract_cols(B, eq_table(r2), 8)
        self.assertEqual(dot(f, g), mle_eval_matrix(C, r1, r2))

    def test_eq_table_matches_fold(self):
        rng = random.Random(4)
        v = [(rng.randrange(P), 0) for _ in range(8)]
        r = [(rng.randrange(P), rng.randrange(P)) for _ in range(3)]
        cur = v
        for ri in r:
            cur = sumfold.fold(cur, ri)
        self.assertEqual(cur[0], dot(v, eq_table(r)))

class TestLaneEmulatorFidelity(unittest.TestCase):
    def test_matches_golden_device_expected(self):
        f = sumfold.gen_vector(0x1234_5678_9ABC_DEF1, 32)
        g = sumfold.gen_vector(0x0FED_CBA9_8765_4321, 32)
        claim = sumfold.inner(f, g)
        for L in (1, 2, 4, 8):
            gold = sumfold.device_expected(f, g, L)
            res = SumFoldLaneEmulator(L).run(f, g, sumfold.Transcript(5, claim), claim)
            self.assertEqual([[ehex(x) for x in m] for m in res["msgs"]], gold["all_msgs"])
            self.assertEqual((ehex(res["final_f"]), ehex(res["final_g"])),
                             (gold["full_final_f"], gold["full_final_g"]))
            host = HostSumcheck().run(f, g, sumfold.Transcript(5, claim), claim)
            self.assertEqual(host["msgs"], res["msgs"])

class TestRequant(unittest.TestCase):
    def test_rhaz_constraint(self):
        rng = random.Random(5)
        for _ in range(2000):
            acc, M, s = rng.randint(-10 ** 6, 10 ** 6), rng.randint(1, 5000), rng.randint(1, 20)
            q, rem, sg = requant(acc, M, s)
            n = acc * M
            self.assertTrue(0 <= rem < (1 << s))
            self.assertEqual(sg, int(n < 0))
            self.assertEqual(n + (1 << (s - 1)) - sg, q * (1 << s) + rem)
            ref = (abs(n) + (1 << (s - 1))) >> s          # independent RHAZ formula
            self.assertEqual(q, ref if n >= 0 else -ref)

    def test_rhaz_ties(self):
        self.assertEqual(requant(-3, 1, 1)[0], -2)      # -1.5 -> -2 (away from zero)
        self.assertEqual(requant(3, 1, 1)[0], 2)        # +1.5 -> +2
        self.assertEqual(requant(-3, 1, 1, "HALF_UP")[0], -1)

    def test_matches_lowering_reference_if_present(self):
        try:
            from edgecert.reference.intops import rhaz_int
        except Exception:
            self.skipTest("reference.intops not available")
        rng = random.Random(6)
        for _ in range(500):
            acc, M, s = rng.randint(-10 ** 6, 10 ** 6), rng.randint(1, 5000), rng.randint(1, 20)
            self.assertEqual(requant(acc, M, s)[0], rhaz_int(acc * M, s))

class TestProveVerify(unittest.TestCase):
    def test_honest_accept_both_backends_and_output_modes(self):
        for backend in (HostSumcheck(), SumFoldLaneEmulator(2)):
            for mode in ("tokens", "commitment"):
                m, env, v, ctx, priv, exp = setup(output_mode=mode)
                pr = TransparentProver(m, env, backend=backend)
                out = pr.infer(ctx, priv)
                pf = pr.prove(ctx, priv, out)
                self.assertEqual(pf["label"], "TRANSPARENT_REFERENCE")
                self.assertFalse(pf["zero_knowledge"])
                r = v.verify(pf, exp)
                self.assertTrue(r.accepted, (r.failed_check, r.reason))
                self.assertIsNotNone(r.receipt)
                # replay of the same proof is refused (nonce consumed)
                r2 = v.verify(pf, exp)
                self.assertEqual((r2.accepted, r2.failed_check), (False, "session.nonce_fresh"))

    def _run(self, mutate):
        m, env, v, ctx, priv, exp = setup()
        pr = TransparentProver(m, env)
        pf = pr.prove(ctx, priv, pr.infer(ctx, priv))
        pf = copy.deepcopy(pf)
        mutate(pf)
        return v.verify(pf, exp)

    def test_tampered_sumcheck_message(self):
        def mut(pf):
            s0 = pf["sumchecks"][0]["msgs"][2]
            v = int(s0[1], 16) ^ 1
            s0[1] = f"{v:032x}"
        r = self._run(mut)
        self.assertFalse(r.accepted)
        self.assertTrue(r.failed_check.startswith("sumcheck"))

    def test_tampered_accumulator_caught_by_commitment(self):
        def mut(pf):
            pf["witness"]["L0.ACC"][0][0] += 1
        r = self._run(mut)
        self.assertEqual(r.failed_check, "witness.commitments")

    def test_noncanonical_field_element(self):
        def mut(pf):
            pf["sumchecks"][1]["final_f"] = f"{0:016x}{P:016x}"
        r = self._run(mut)
        self.assertFalse(r.accepted)

    def test_missing_and_garbage_proofs_fail_closed(self):
        m, env, v, ctx, priv, exp = setup()
        self.assertFalse(v.verify(None, exp).accepted)
        self.assertFalse(v.verify({"format": "x"}, exp).accepted)
        r = v.verify({"format": "edgecert-transparent-proof/v1", "label": "TRANSPARENT_REFERENCE"}, exp)
        self.assertFalse(r.accepted)

    def test_forged_sign_bit_detected(self):
        m, env, v, ctx, priv, exp = setup()

        def mut(W):
            W["L0.SIG"][0][0] ^= 1          # claim the opposite sign for one element (roots recomputed)
        pr = TransparentProver(m, env, tamper=Tamper(witness_mutator=mut))
        pf = pr.prove(ctx, priv, pr.infer(ctx, priv))
        r = v.verify(pf, exp)
        self.assertFalse(r.accepted)
        self.assertEqual(r.failed_check, "requant.sign_bit")

    def test_wrong_rounding_detected(self):
        m, env, v, ctx, priv, exp = setup()
        pr = TransparentProver(m, env, tamper=Tamper(rounding="TRUNC_FLOOR", forge_remainders=True))
        pf = pr.prove(ctx, priv, pr.infer(ctx, priv))
        r = v.verify(pf, exp)
        self.assertFalse(r.accepted)
        self.assertTrue(r.failed_check.startswith("requant"))

class TestIndependence(unittest.TestCase):
    def test_verifier_never_imports_prover(self):
        for path in (ROOT / "src" / "edgecert" / "verifier").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = (node.module or "")
                    self.assertNotIn("prover_adapters", mod, f"{path.name} imports {mod}")
                    self.assertNotIn("toy_model", mod)
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        self.assertNotIn("prover_adapters", a.name)

if __name__ == "__main__":
    unittest.main()
