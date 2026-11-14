"""Bit-exactness of the CPU SumFold kernel (kernels/cpu) against the golden model.

Run:  python -m unittest tests.test_kernels_cpu -v     (from the repository root)

Builds kernels/cpu/build/sumfold_cpu.exe with MinGW-w64 gcc if missing or stale, then checks every transcript field (claim, all round messages, all challenges, final f~(r), g~(r)) against edgecert.sumfold.prove() for n = 1..12, threads in {1, 4, 32}, both variants; plus lane / scheduling overrides, the verifier, and the numpy generator.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "runners"))
sys.path.insert(0, str(ROOT / "src"))

import kernel_common as kc  # noqa: E402
from edgecert import sumfold as sf  # noqa: E402

SEEDS = [(kc.DEFAULT_SEED_F, kc.DEFAULT_SEED_G), (1, 2)]

class CpuKernelBitExact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = kc.build_cpu()
        cls.exe = cls.build["exe"]

    def test_selftest(self):
        import subprocess
        r = subprocess.run([self.exe, "--selftest"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('"OK"', r.stdout)

    def test_bit_exact_grid(self):
        seedf, seedg = SEEDS[0]
        for n in range(1, 13):
            gold = kc.golden_record(seedf, seedg, n)
            for threads in (1, 4, 32):
                for variant in ("fused", "unfused"):
                    with self.subTest(n=n, threads=threads, variant=variant):
                        rec = kc.run_cpu(self.exe, n, threads, variant, seedf, seedg, reps=2, warmup=1)
                        self.assertTrue(rec["consistent_across_runs"])
                        self.assertEqual(kc.compare_records(rec, gold), [], rec["_command"])

    def test_second_seed_pair(self):
        seedf, seedg = SEEDS[1]
        for n in (1, 5, 9):
            gold = kc.golden_record(seedf, seedg, n)
            for variant in ("fused", "unfused"):
                rec = kc.run_cpu(self.exe, n, 4, variant, seedf, seedg)
                self.assertEqual(kc.compare_records(rec, gold), [], rec["_command"])

    def test_lane_and_schedule_overrides(self):
        seedf, seedg = SEEDS[0]
        n = 10
        gold = kc.golden_record(seedf, seedg, n)
        for lanes in (1, 2, 8, 64, 512):
            for threads in (1, 3, 24):
                for sched in ("static", "dynamic"):
                    with self.subTest(lanes=lanes, threads=threads, sched=sched):
                        rec = kc.run_cpu(self.exe, n, threads, "fused", seedf, seedg, lanes=lanes, sched=sched)
                        self.assertEqual(rec["lanes"], lanes)
                        self.assertEqual(kc.compare_records(rec, gold), [], rec["_command"])

    def test_invalid_lanes_rejected(self):
        with self.assertRaises(RuntimeError):
            kc.run_cpu(self.exe, 4, 2, "fused", lanes=16)  # lanes > N/2
        with self.assertRaises(RuntimeError):
            kc.run_cpu(self.exe, 6, 2, "fused", lanes=3)   # not a power of two

    def test_pass_structure(self):
        # fused, L = 1: R + 1 = n + 1 passes (E, F x (n-1), L); unfused: 2n passes (E, U)*n
        rec = kc.run_cpu(self.exe, 8, 1, "fused")
        self.assertEqual(rec["pass_kinds"], "E" + "F" * 7 + "L")
        rec = kc.run_cpu(self.exe, 8, 4, "unfused")
        self.assertEqual(rec["pass_kinds"], "EU" * 8)
        rec = kc.run_cpu(self.exe, 8, 4, "fused", lanes=4)  # R = 6 device rounds, tail of 2
        self.assertEqual(rec["pass_kinds"], "E" + "F" * 5 + "L" + "g" + "e" + "f" + "l")

    def test_verifier_accepts_kernel_output(self):
        rec = kc.run_cpu(self.exe, 14, 8, "fused")
        self.assertTrue(kc.verify_record(rec))
        rec2 = kc.run_cpu(self.exe, 14, 8, "unfused")
        self.assertEqual(kc.compare_records(rec, rec2), [])
        bad = dict(rec)
        bad["msgs"] = [list(m) for m in rec["msgs"]]
        bad["msgs"][3][2] = "0" * 31 + "1"
        self.assertFalse(kc.verify_record(bad))

    def test_fast_generator_matches_golden(self):
        for n_elems in (1, 7, 64, 1000, 4096):
            v = kc.gen_vector_fast(kc.DEFAULT_SEED_F, n_elems)
            ref = [c0 for (c0, _) in sf.gen_vector(kc.DEFAULT_SEED_F, n_elems)]
            self.assertEqual([int(x) for x in v], ref)

if __name__ == "__main__":
    unittest.main()
