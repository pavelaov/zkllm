"""TileContract compiler: shapes, static bounds, K-split, accounting, trace cross-check.

Run from the repository root:
    set PYTHONPATH=src && python -m unittest discover -s tests -p "test_compiler*.py"
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edgecert.compiler import graph as G  # noqa: E402
from edgecert.compiler import tile_contract as TC  # noqa: E402
from edgecert.compiler.gadgets import matmul_bound  # noqa: E402
from edgecert.compiler.model_shapes import MODELS  # noqa: E402
from edgecert.field import P  # noqa: E402
from edgecert.reference import intref as R  # noqa: E402

def expected_counts(op, wl):
    exp = {"committed": {}, "checks": {}, "lookups": {}}
    for (nm, b, c) in op.cost.committed:
        exp["committed"][nm] = exp["committed"].get(nm, 0) + c * G.multiplicity(op, b, wl)
    for k in op.cost.checks:
        exp["checks"][k.name] = exp["checks"].get(k.name, 0) + k.count * G.multiplicity(op, k.basis, wl)
    for (t, b, c) in op.cost.lookups:
        exp["lookups"][t] = exp["lookups"].get(t, 0) + c * G.multiplicity(op, b, wl)
    return exp

class TestShapes(unittest.TestCase):
    def test_param_counts(self):
        self.assertEqual(MODELS["gpt2_124m"].param_count()["total"], 124_439_808)
        for name, tol in (("qwen3_0.6b", 0.02), ("gemma3_1b", 0.02), ("llama3_8b", 0.002)):
            m = MODELS[name]
            got = m.param_count()["total"]
            self.assertLess(abs(got - m.reported_params) / m.reported_params, tol, name)
        self.assertEqual(MODELS["gemma3_1b"].param_count()["embedding"], 262144 * 1152)
        self.assertTrue(MODELS["gpt2_124m"].verified)
        self.assertFalse(MODELS["gemma3_1b"].verified)

    def test_gemma_window_pattern(self):
        g = MODELS["gemma3_1b"]
        self.assertEqual([l for l in range(g.n_layers) if g.is_global(l)], [5, 11, 17, 23])

class TestStatic(unittest.TestCase):
    def test_matmul_bound(self):
        self.assertEqual(matmul_bound(16384, 128, 8), 16_777_216)            # design report example
        self.assertEqual(matmul_bound(16384, 128, 128), 268_435_456)

    def test_real_models_pass_goldilocks_fail_babybear(self):
        for name, shape in MODELS.items():
            cons = TC.lower(G.build_graph(shape))
            TC.assert_static(cons, P)
            self.assertTrue(any(TC.static_check(tc, TC.BABYBEAR_P) for tc in cons.values()), name)
            self.assertFalse(any(TC.static_check(tc, TC.BN254_R) for tc in cons.values()), name)

    def test_ksplit_w8a16(self):
        p = TC.plan_accumulation(4096, 1 << 15, 127, 0, 16, 8, "cpu_int32")
        self.assertTrue(p["needs_ksplit"])
        self.assertEqual(p["K_sub"], ((1 << 31) - 1) // ((1 << 15) * 127))
        self.assertLessEqual(p["subtile_bound"], (1 << 31) - 1)
        self.assertGreaterEqual(p["n_sub"] * p["K_sub"], 4096)
        q = TC.plan_accumulation(4096, 1 << 15, 127, 0, 16, 8, "gpu_imma_s8")
        self.assertEqual((q["passes"], q["n_sub"]), (2, 1))
        d = TC.plan_accumulation(4096, 1 << 15, 127, 0, 16, 8, "fpga_dsp48e2")
        self.assertFalse(d["needs_ksplit"])
        big = TC.plan_accumulation(1 << 18, 128, 127, 0, 8, 8, "gpu_imma_s8")
        self.assertEqual(big["n_sub"], 2)
        self.assertEqual(big["K_sub"] % 32, 0)
        ok = TC.plan_accumulation(14336, 128, 127, (1 << 23) - 1, 8, 8, "gpu_imma_s8")
        self.assertFalse(ok["needs_ksplit"])

    def test_bias_moves_to_combine_stage(self):
        p = TC.plan_accumulation(132000, 128, 127, (1 << 23) - 1, 8, 8, "cpu_int32")
        self.assertFalse(p["needs_ksplit"])         # K*B alone fits int32 ...
        self.assertTrue(p["bias_in_combine"])       # ... but K*B + |b| does not: bias added in the wide combine
        self.assertLessEqual(p["subtile_bound"], (1 << 31) - 1)
        self.assertGreater(p["combined_bound"], (1 << 31) - 1)

    def test_pv_split_threshold(self):
        ops = G.build_graph(MODELS["gpt2_124m"])
        c511 = TC.tile_census(ops, {"kind": "decode", "pos": 510}, "cpu_int32")
        c512 = TC.tile_census(ops, {"kind": "decode", "pos": 511}, "cpu_int32")
        self.assertEqual(c511["tiles_needing_ksplit"], 0)
        self.assertGreater(c512["tiles_needing_ksplit"], 0)
        self.assertEqual(TC.tile_census(ops, {"kind": "decode", "pos": 511}, "gpu_imma_s8")["tiles_needing_ksplit"], 0)

class TestAccounting(unittest.TestCase):
    def test_limb_optimizer_and_naive_order(self):
        ops = G.build_graph(MODELS["gpt2_124m"])
        tabs = TC.table_sizes(ops, 50257, 1024, 1023, 12, True)
        acc = TC.account(ops, {"kind": "prefill", "n": 128}, tabs)
        lk = acc["rc_lookups"]
        self.assertLess(lk["tc"], lk["naive_int32"])
        self.assertLess(lk["naive_int32"], lk["naive_field64"])
        per = acc["limb_optimizer"]["per_width"]
        self.assertEqual([r["b"] for r in per], [8, 10, 12, 14, 16])
        self.assertTrue(all(per[i]["lookups"] >= per[i + 1]["lookups"] for i in range(len(per) - 1)))
        self.assertIn(acc["rc_limb_bits"], TC.RC_WIDTHS)
        small = TC.optimize_limbs({20: 10})
        self.assertLess(small["best_b"], 16)        # tiny proofs prefer small tables

    def test_distinct_contracts_small(self):
        cons = TC.lower(G.build_graph(MODELS["llama3_8b"]))
        self.assertLess(len(TC.dedup(cons)), 25)

    def test_t0_counts_match_trace_and_dynamic_check_passes(self):
        for cfg in (R.T0_GPT(), R.T0_LLAMA()):
            im = R.build_t0(cfg)
            toks = R.random_prompts(cfg, 1, 21, 8)[0]
            _, tr, _ = im.forward(toks, record=True)
            got = R.trace_counts(tr)
            wl = {"kind": "prefill", "n": 21}
            ops = G.build_t0_graph(im, elide=False)
            self.assertEqual(len(ops), len(got))
            for op in ops:
                self.assertEqual(expected_counts(op, wl), got[op.op_id], op.op_id)
            for elide in (False, True):
                cons = TC.lower(G.build_t0_graph(im, elide=elide))
                TC.assert_static(cons)
                TC.check_weights(im, cons)
                TC.dynamic_check(tr, cons)

if __name__ == "__main__":
    unittest.main()
