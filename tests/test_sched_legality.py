"""CertSched legality checker: legal policies accepted, illegal schedules rejected.

Run:  set PYTHONPATH=src && python -m unittest tests.test_sched_legality -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edgecert.runtime.dag import Encoding, ModelShape, build_dag  # noqa: E402
from edgecert.runtime.scheduler import (  # noqa: E402
    CHALLENGE, COMMIT, DERIVE, FREE, MODIFY, OPEN, PROVE, REBIND, RECOMPUTE, Event, DeviceProfile,
    action_list, check_schedule, emit_schedule, plan_certsched, run_policy, uniform_decisions)

TINY = ModelShape("tiny", n_layers=2, d_model=16, n_heads=2, n_kv_heads=1, head_dim=8, d_ff=32,
                  vocab=64, gated_mlp=True, qk_norm=True, tied_embeddings=True, native_ctx=4096)
TINY_GPT = ModelShape("tiny-gpt", n_layers=2, d_model=16, n_heads=2, n_kv_heads=2, head_dim=8,
                      d_ff=64, vocab=64, gated_mlp=False, qk_norm=False, tied_embeddings=True,
                      native_ctx=64, learned_pos=True)
PROF = DeviceProfile("test", int8_ops_per_s=1e14, mem_Bps=5e11, sc_elems_per_s=1e10,
                     commit_Bps=4e9, hash_Bps=2e10, link_Bps=2.5e10, ssd_read_Bps=5e9,
                     ssd_write_Bps=3e9, aes_Bps=1e10)

def keep_all(dag):
    return emit_schedule(dag, uniform_decisions(dag, "KEEP_ALL"))

def first_index(events, kind, target=None):
    for i, e in enumerate(events):
        if e.kind == kind and (target is None or e.target == target):
            return i
    raise AssertionError(f"{kind} {target} not found")

class TestLegalPolicies(unittest.TestCase):
    def test_legal_policies_accepted_both_protocols(self):
        for shape in (TINY, TINY_GPT):
            for proto in ("GLOBAL", "LAYERWISE"):
                dag = build_dag(shape, 6, 4, Encoding(protocol=proto))
                acts = action_list(dag)
                for pol in ("KEEP_ALL", "HOST_STAGE", "SPILL", "RECOMPUTE"):
                    r = run_policy(dag, pol, PROF, acts=acts)
                    self.assertTrue(r.check.legal, f"{shape.name} {proto} {pol}: {r.check.violations[:3]}")

    def test_naive_layer_streaming_rejected(self):
        for proto in ("GLOBAL", "LAYERWISE"):
            dag = build_dag(TINY, 6, 4, Encoding(protocol=proto))
            r = run_policy(dag, "LAYER_STREAM_NAIVE", PROF)
            self.assertFalse(r.check.legal)
            self.assertIn("A_UNAVAILABLE", r.check.rules)

    def test_certsched_legal_and_under_cap(self):
        dag = build_dag(TINY, 16, 4)
        keep = run_policy(dag, "KEEP_ALL", PROF)
        cap = dag.static_bytes() + (keep.check.peak_device - dag.static_bytes()) // 3
        r = plan_certsched(dag, PROF, cap)
        self.assertTrue(r.check.legal, r.check.violations[:3])
        self.assertTrue(r.check.fits)
        self.assertLessEqual(r.check.peak_device, cap)
        # never worse than the cheapest uniform legal policy that also fits the cap
        best_uniform = min(run_policy(dag, p, PROF, cap).time["overhead_s"]
                           for p in ("HOST_STAGE", "SPILL", "RECOMPUTE")
                           if run_policy(dag, p, PROF, cap).check.fits)
        self.assertLessEqual(r.time["overhead_s"], best_uniform * 1.0001)

    def test_keep_all_boundary_is_exact_witness_sum(self):
        dag = build_dag(TINY, 8, 3)
        chk = check_schedule(dag, keep_all(dag))
        base = sum(t.nbytes for t in dag.tensors.values() if t.committed and t.role != "helper")
        self.assertEqual(chk.boundary_bytes["device"], dag.static_bytes() + base)
        self.assertEqual(chk.peak_device, dag.static_bytes() + dag.witness_bytes())

class TestIllegalSchedules(unittest.TestCase):
    def setUp(self):
        self.dag = build_dag(TINY, 6, 4)
        self.ev = keep_all(self.dag)
        self.assertTrue(check_schedule(self.dag, self.ev).legal)
        self.gamma = first_index(self.ev, CHALLENGE, "gamma")
        self.beta = first_index(self.ev, CHALLENGE, "beta")
        # a committed witness that a later proof task reads
        i = next(i for i, e in enumerate(self.ev) if i > self.gamma and e.kind == PROVE
                 and any(not self.dag.tensors[r].static for r in self.dag.tasks[e.target].reads))
        self.task_idx = i
        self.victim = next(r for r in self.dag.tasks[self.ev[i].target].reads
                           if not self.dag.tensors[r].static)

    def rules(self, ev):
        return check_schedule(self.dag, ev).rules

    def test_free_before_proof(self):
        ev = list(self.ev)
        ev.insert(self.gamma + 1, Event(FREE, self.victim))
        self.assertIn("A_UNAVAILABLE", self.rules(ev))

    def test_commit_after_challenge(self):
        ev = list(self.ev)
        j = first_index(ev, COMMIT, self.victim)
        c = ev.pop(j)
        ev.insert(first_index(ev, CHALLENGE, "beta") + 1, c)
        r = self.rules(ev)
        self.assertIn("C_CHALLENGE_BEFORE_COMMIT", r)
        self.assertIn("C_COMMIT_AFTER_CHALLENGE", r)
        self.assertIn("D_TRANSCRIPT_ORDER", r)

    def test_adapt_after_challenge(self):
        ev = list(self.ev)
        ev.insert(self.gamma + 1, Event(MODIFY, self.victim))
        r = self.rules(ev)
        self.assertIn("B_MODIFIED_AFTER_COMMIT", r)
        self.assertIn("B_ADAPT_AFTER_CHALLENGE", r)

    def test_adapt_and_recommit_after_challenge(self):
        ev = list(self.ev)
        ev[self.gamma + 1:self.gamma + 1] = [Event(MODIFY, self.victim), Event(COMMIT, self.victim)]
        r = self.rules(ev)
        self.assertIn("B_ADAPT_AFTER_CHALLENGE", r)
        self.assertIn("B_RECOMMIT", r)

    def test_modify_model_weight_rejected(self):
        ev = list(self.ev)
        ev.insert(self.gamma + 1, Event(MODIFY, "W:0.q"))
        self.assertIn("B_MODIFIED_AFTER_COMMIT", self.rules(ev))

    def test_derive_before_challenge(self):
        ev = list(self.ev)
        j = first_index(ev, DERIVE)
        d = ev.pop(j)
        ev.insert(self.beta, d)     # just before beta
        self.assertIn("C_TASK_BEFORE_CHALLENGE", self.rules(ev))

    def test_proof_before_challenge(self):
        ev = list(self.ev)
        p = ev.pop(self.task_idx)
        ev.insert(self.gamma, p)
        r = self.rules(ev)
        self.assertIn("C_TASK_BEFORE_CHALLENGE", r)

    def test_transcript_reorder(self):
        ev = list(self.ev)
        i = self.task_idx
        j = next(k for k in range(i + 1, len(ev)) if ev[k].kind in (PROVE, OPEN))
        ev[i], ev[j] = ev[j], ev[i]
        self.assertEqual(self.rules(ev), ["D_TRANSCRIPT_ORDER"])

    def test_missing_obligation(self):
        ev = [e for k, e in enumerate(self.ev) if k != self.task_idx]
        self.assertIn("E_MISSING_OBLIGATION", self.rules(ev))

    def test_duplicate_task(self):
        ev = list(self.ev)
        ev.insert(self.task_idx + 1, ev[self.task_idx])
        self.assertIn("E_DUPLICATE_TASK", self.rules(ev))

    def test_unencrypted_spill(self):
        dec = uniform_decisions(self.dag, "SPILL")
        ev = emit_schedule(self.dag, dec, encrypt_spill=False)
        self.assertIn("A_UNENCRYPTED_SPILL", check_schedule(self.dag, ev).rules)

    def test_recompute_without_rebind(self):
        dec = uniform_decisions(self.dag, "RECOMPUTE")
        ev = emit_schedule(self.dag, dec)
        self.assertTrue(check_schedule(self.dag, ev).legal)
        j = first_index(ev, REBIND)
        ev = ev[:j] + ev[j + 1:]
        self.assertIn("A_UNBOUND_RECOMPUTE", check_schedule(self.dag, ev).rules)

    def test_recompute_mismatch_detected(self):
        dec = uniform_decisions(self.dag, "RECOMPUTE")
        ev = emit_schedule(self.dag, dec)
        j = first_index(ev, RECOMPUTE)
        op = self.dag.ops.get(ev[j].target) or self.dag.derive_ops[ev[j].target]
        # tamper with a resident input right before the recomputation
        inp = next(u for u in op.inputs if not self.dag.tensors[u].static)
        ev.insert(j, Event(MODIFY, inp))
        r = check_schedule(self.dag, ev).rules
        self.assertIn("B_REBIND_MISMATCH", r)

if __name__ == "__main__":
    unittest.main()
