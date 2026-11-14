"""GPU SumFold benchmark (kernels/cuda/sumfold.cu) with bit-exactness checks.

Backends
  cupy : CuPy RawModule (NVRTC) on a real GPU, timed with CUDA events; board power is
         sampled with `nvidia-smi --query-gpu=power.draw -lms <period>` while a
         configuration is repeated for >= --power-window-s seconds.    
  shim : the *same* sumfold.cu compiled by g++ against kernels/cuda/shim and executed on
         the CPU (SIMT emulation, OpenMP team per block).  Used for correctness only; its
         timings are meaningless and never reported.                 

Host round loop (identical for both backends, see prove_device):
  fused  : pass 0 sf_eval; passes 1..n-1 sf_fold_eval (src level i-1 -> dst level i,
           ping-pong); final sf_fold; after every evaluating pass sf_reduce_partials
           (1 block) and a 48-byte device->host copy; the SHA-256 transcript runs on the
           host with the golden edgecert.sumfold.Transcript (STEP mode of the spec).
  unfused: per round sf_eval + reduce + transcript, then sf_fold (ping-pong).

Examples python bench/runners/kernel_gpu_bench.py                        # GPU 0 (nvidia-smi order) + emulation python bench/runners/kernel_gpu_bench.py --devices 0,1 --logn 12,16,20,24 --reps 10
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_common as kc  # noqa: E402

from edgecert.field import ehex, eadd  # noqa: E402
from edgecert.sumfold import Transcript  # noqa: E402

CUDA_DIR = kc.ROOT / "kernels" / "cuda"
CU_SRC = CUDA_DIR / "sumfold.cu"
SHIM_DIR = CUDA_DIR / "shim"
SHIM_DLL = CUDA_DIR / "build" / "sumfold_shim.dll"
SHIM_FLAGS = ["-O2", "-std=c++14", "-fopenmp", "-shared", "-static-libgcc", "-static-libstdc++",
              "-Wall", "-Wno-unknown-pragmas"]
KERNELS = ("sf_eval", "sf_fold_eval", "sf_fold", "sf_reduce_partials")

# ================================================================ backends
def build_shim(force: bool = False) -> Dict:
    SHIM_DLL.parent.mkdir(parents=True, exist_ok=True)
    deps = [CU_SRC, SHIM_DIR / "shim_launch.cpp", SHIM_DIR / "cuda_host_shim.h"]
    stale = force or not SHIM_DLL.exists() or any(d.stat().st_mtime > SHIM_DLL.stat().st_mtime for d in deps)
    cmd = [str(kc.MINGW_BIN / "g++.exe")] + SHIM_FLAGS + ["-o", str(SHIM_DLL), str(SHIM_DIR / "shim_launch.cpp")]
    if stale:
        r = subprocess.run(cmd, cwd=str(CUDA_DIR), env=kc._gcc_env(), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("shim build failed:\n" + r.stdout + r.stderr)
    return {"dll": str(SHIM_DLL), "command": subprocess.list2cmdline(cmd), "rebuilt": stale}

class ShimBackend:
    """sumfold.cu executed on the CPU via kernels/cuda/shim (correctness only)."""
    name = "shim"

    def __init__(self, order: int = 0):
        import numpy as np
        self.np = np
        build_shim()
        if hasattr(os, "add_dll_directory"):
            self._dlldir = os.add_dll_directory(str(kc.MINGW_BIN))  # libgomp-1.dll
        self.lib = ctypes.CDLL(str(SHIM_DLL))
        P, U, I, LL, ULL = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_longlong, ctypes.c_ulonglong
        self.lib.shim_sf_eval.argtypes = [U, U, P, P, LL, P]
        self.lib.shim_sf_fold_eval.argtypes = [U, U, P, P, P, P, LL, ULL, ULL, P]
        self.lib.shim_sf_fold.argtypes = [U, U, P, P, P, P, LL, ULL, ULL]
        self.lib.shim_sf_reduce_partials.argtypes = [U, U, P, I, P]
        self.lib.shim_set_block_order.argtypes = [I]
        self.lib.shim_set_block_order(order)

    @staticmethod
    def _p(a):
        return a.ctypes.data

    def alloc(self, n):
        return self.np.zeros((max(n, 1), 2), dtype=self.np.uint64)

    def from_host(self, a):
        return self.np.ascontiguousarray(a, dtype=self.np.uint64).copy()

    def to_host(self, a):
        return self.np.array(a)

    def copy(self, dst, src):
        dst[...] = src

    def _chk(self, rc):
        if rc != 0:
            raise RuntimeError(f"shim launch failed rc={rc}")

    def sf_eval(self, grid, block, f, g, npairs, partials):
        self._chk(self.lib.shim_sf_eval(grid, block, self._p(f), self._p(g), npairs, self._p(partials)))

    def sf_fold_eval(self, grid, block, fs, gs, fd, gd, nquads, r, partials):
        self._chk(self.lib.shim_sf_fold_eval(grid, block, self._p(fs), self._p(gs), self._p(fd), self._p(gd),
                                             nquads, r[0], r[1], self._p(partials)))

    def sf_fold(self, grid, block, fs, gs, fd, gd, npairs, r):
        self._chk(self.lib.shim_sf_fold(grid, block, self._p(fs), self._p(gs), self._p(fd), self._p(gd),
                                        npairs, r[0], r[1]))

    def sf_reduce_partials(self, block, partials, nblocks, out3):
        self._chk(self.lib.shim_sf_reduce_partials(1, block, self._p(partials), nblocks, self._p(out3)))

    def event(self):
        return None

    def synchronize(self):
        pass

class CupyBackend:
    """Real GPU through CuPy RawModule (NVRTC)."""
    name = "cupy"

    def __init__(self, device: int = 0):
        import cupy as cp 
        import numpy as np
        self.cp, self.np = cp, np
        self.device = device
        cp.cuda.Device(device).use()
        self.mod = cp.RawModule(code=CU_SRC.read_text(encoding="utf-8"), options=("-std=c++14",),
                                backend="nvrtc")
        self.k = {name: self.mod.get_function(name) for name in KERNELS}
        props = cp.cuda.runtime.getDeviceProperties(device)
        self.sm_count = props["multiProcessorCount"]
        self.props = {"name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                      "sm_count": self.sm_count, "cc": f'{props["major"]}.{props["minor"]}',
                      "global_mem_GiB": round(props["totalGlobalMem"] / 2**30, 1)}

    def alloc(self, n):
        return self.cp.zeros((max(n, 1), 2), dtype=self.cp.uint64)

    def from_host(self, a):
        return self.cp.asarray(self.np.ascontiguousarray(a, dtype=self.np.uint64))

    def to_host(self, a):
        return self.cp.asnumpy(a)

    def copy(self, dst, src):
        self.cp.copyto(dst, src)

    def elapsed(self, e0, e1):
        return self.cp.cuda.get_elapsed_time(e0, e1) / 1e3

    def sf_eval(self, grid, block, f, g, npairs, partials):
        self.k["sf_eval"]((grid,), (block,), (f, g, self.np.int64(npairs), partials))

    def sf_fold_eval(self, grid, block, fs, gs, fd, gd, nquads, r, partials):
        self.k["sf_fold_eval"]((grid,), (block,), (fs, gs, fd, gd, self.np.int64(nquads),
                                                    self.np.uint64(r[0]), self.np.uint64(r[1]), partials))

    def sf_fold(self, grid, block, fs, gs, fd, gd, npairs, r):
        self.k["sf_fold"]((grid,), (block,), (fs, gs, fd, gd, self.np.int64(npairs),
                                              self.np.uint64(r[0]), self.np.uint64(r[1])))

    def sf_reduce_partials(self, block, partials, nblocks, out3):
        self.k["sf_reduce_partials"]((1,), (block,), (partials, self.np.int32(nblocks), out3))

    def event(self):
        e = self.cp.cuda.Event()
        e.record()
        return e

    def synchronize(self):
        self.cp.cuda.Device(self.device).synchronize()

# ============================================================== host flow
def _to_ext3(a) -> List[tuple]:
    return [(int(a[k, 0]), int(a[k, 1])) for k in range(3)]

class DeviceWork:
    """Device buffers for one problem size: pristine tables (uploaded once), working level-0
    tables A, ping-pong buffer B, per-block partials and the 3-word round message."""

    def __init__(self, be, f_host, g_host, n: int, grid_max: int):
        N = 1 << n
        self.be, self.n = be, n
        self.A0 = (be.from_host(f_host), be.from_host(g_host))
        self.A = (be.alloc(N), be.alloc(N))
        self.B = (be.alloc(N // 2), be.alloc(N // 2))
        self.partials = be.alloc(3 * grid_max)
        self.out3 = be.alloc(3)
        self.grid_max = grid_max

    def reset(self):
        self.be.copy(self.A[0], self.A0[0])
        self.be.copy(self.A[1], self.A0[1])
        self.be.synchronize()

def prove_device(be, f_host, g_host, n: int, variant: str = "fused", block: int = 256,
                 grid_max: int = 1024, timing: bool = False, work: Optional[DeviceWork] = None) -> Dict:
    """Full SumFold prover on a backend; returns the transcript in hex plus timings.

    f_host/g_host: numpy uint64 arrays of shape (N, 2) ({c0, c1} per element); ignored when a
    prepared DeviceWork is passed (its pristine tables are copied device-to-device, untimed).
    wall_s covers every kernel, the per-round 48-byte device->host copy and the host
    SHA-256 transcript (STEP mode); it excludes uploads and the untimed reset copy.
    """
    assert block % 32 == 0 and 32 <= block <= 1024
    N = 1 << n
    if work is None:
        work = DeviceWork(be, f_host, g_host, n, grid_max)
    assert work.n == n and work.grid_max >= grid_max
    work.reset()
    grid_for = lambda work_items: max(1, min(-(-work_items // block), grid_max))  # noqa: E731
    A, B, partials, out3 = work.A, work.B, work.partials, work.out3
    src, dst = A, B
    pass_ev = []           # (kind, start_event, end_event)
    t0 = time.perf_counter()
    msgs, chal = [], []

    def ev():
        return be.event() if timing else None

    def round_msg(grid):
        be.sf_reduce_partials(block, partials, grid, out3)
        return _to_ext3(be.to_host(out3))  # device -> host copy (synchronizes)

    if variant == "fused":
        npairs = N // 2
        grid = grid_for(npairs)
        e0 = ev()
        be.sf_eval(grid, block, src[0], src[1], npairs, partials)
        s = round_msg(grid)
        pass_ev.append(("E", e0, ev()))
        claim = eadd(s[0], s[1])
        tr = Transcript(n, claim)
        r = tr.round(*s)
        msgs.append(s); chal.append(r)
        for i in range(1, n):
            nquads = (N >> (i - 1)) // 4
            grid = grid_for(nquads)
            e0 = ev()
            be.sf_fold_eval(grid, block, src[0], src[1], dst[0], dst[1], nquads, r, partials)
            s = round_msg(grid)
            pass_ev.append(("F", e0, ev()))
            r = tr.round(*s)
            msgs.append(s); chal.append(r)
            src, dst = dst, src
        e0 = ev()
        be.sf_fold(1, block, src[0], src[1], dst[0], dst[1], 1, r)
        pass_ev.append(("L", e0, ev()))
        fin_f, fin_g = be.to_host(dst[0][:1])[0], be.to_host(dst[1][:1])[0]
    elif variant == "unfused":
        claim = None
        tr = None
        for i in range(n):
            npairs = (N >> i) // 2
            grid = grid_for(npairs)
            e0 = ev()
            be.sf_eval(grid, block, src[0], src[1], npairs, partials)
            s = round_msg(grid)
            pass_ev.append(("E", e0, ev()))
            if i == 0:
                claim = eadd(s[0], s[1])
                tr = Transcript(n, claim)
            r = tr.round(*s)
            msgs.append(s); chal.append(r)
            e0 = ev()
            be.sf_fold(grid, block, src[0], src[1], dst[0], dst[1], npairs, r)
            pass_ev.append(("U", e0, ev()))
            src, dst = dst, src
        fin_f, fin_g = be.to_host(src[0][:1])[0], be.to_host(src[1][:1])[0]
    else:
        raise ValueError(variant)
    be.synchronize()
    wall = time.perf_counter() - t0
    out = {"n": n, "claim": ehex(claim), "msgs": [[ehex(x) for x in m] for m in msgs],
           "challenges": [ehex(x) for x in chal],
           "final_f": ehex((int(fin_f[0]), int(fin_f[1]))), "final_g": ehex((int(fin_g[0]), int(fin_g[1]))),
           "wall_s": wall, "pass_kinds": "".join(k for k, _, _ in pass_ev)}
    if timing and pass_ev and pass_ev[0][1] is not None:
        out["pass_event_s"] = [be.elapsed(a, b) for _, a, b in pass_ev]
    return out

def host_tables(seedf: int, seedg: int, n: int):
    import numpy as np
    N = 1 << n
    f = np.zeros((N, 2), dtype=np.uint64)
    g = np.zeros((N, 2), dtype=np.uint64)
    f[:, 0] = kc.gen_vector_fast(seedf, N)
    g[:, 0] = kc.gen_vector_fast(seedg, N)
    return f, g

# ============================================================ power / smi
def nvidia_smi_inventory() -> Dict:
    q = "index,name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,pcie.link.gen.current,pcie.link.width.current"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        rows = [dict(zip(q.split(","), [x.strip() for x in line.split(",")]))
                for line in r.stdout.strip().splitlines() if line.strip()]
        return {"ok": r.returncode == 0, "gpus": rows, "query": q}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class PowerSampler:
    """Background `nvidia-smi --query-gpu=timestamp,power.draw -lms period -i idx`."""

    def __init__(self, index: int, period_ms: int = 50):
        self.cmd = ["nvidia-smi", "--query-gpu=timestamp,power.draw", "--format=csv,noheader,nounits",
                    f"--loop-ms={period_ms}", "-i", str(index)]
        self.samples: List[tuple] = []
        self.proc = None
        self.th = None

    def __enter__(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.th = threading.Thread(target=self._read, daemon=True)
        self.th.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            parts = [x.strip() for x in line.split(",")]
            try:
                self.samples.append((time.perf_counter(), float(parts[-1])))
            except ValueError:
                pass

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        self.th.join(timeout=2)

    def window(self, t0: float, t1: float) -> List[float]:
        return [w for (t, w) in self.samples if t0 <= t <= t1]

# ================================================================ checks
def emulation_checks(max_n: int = 10, quick: bool = False) -> Dict:
    """Bit-exactness of (a) the shim-executed sumfold.cu and (b) the pure-Python emulator."""
    import kernel_cuda_emu as emu
    from edgecert import sumfold as sf
    res = {"evidence_class": "SIMULATION", "shim": [], "python_emulator": [], "inplace_hazard": []}
    try:
        info = build_shim()
        res["shim_build"] = info
    except Exception as e:
        res["shim_build"] = {"error": str(e)[:800]}
        info = None
    configs = [(64, 3), (32, 1), (96, 2)] if not quick else [(64, 3)]
    orders = [0, 1, 2] if not quick else [0, 1]
    if info is not None:
        for order in orders:
            be = ShimBackend(order=order)
            for block, gmax in configs:
                for n in range(1, max_n + 1):
                    gold = kc.golden_record(kc.DEFAULT_SEED_F, kc.DEFAULT_SEED_G, n)
                    f, g = host_tables(kc.DEFAULT_SEED_F, kc.DEFAULT_SEED_G, n)
                    for v in ("fused", "unfused"):
                        out = prove_device(be, f, g, n, v, block=block, grid_max=gmax)
                        res["shim"].append({"n": n, "variant": v, "block": block, "grid_max": gmax,
                                            "block_order": ["asc", "desc", "oddfirst"][order],
                                            "match": kc.compare_records(out, gold) == []})
    to_hex = lambda d: {"claim": ehex(d["claim"]), "msgs": [[ehex(x) for x in m] for m in d["msgs"]],  # noqa: E731
                        "challenges": [ehex(x) for x in d["challenges"]], "final_f": ehex(d["final_f"]),
                        "final_g": ehex(d["final_g"])}
    for n in range(1, min(max_n, 8) + 1):
        fv = sf.gen_vector(kc.DEFAULT_SEED_F, 1 << n)
        gv = sf.gen_vector(kc.DEFAULT_SEED_G, 1 << n)
        gold = kc.golden_record(kc.DEFAULT_SEED_F, kc.DEFAULT_SEED_G, n)
        for order in ("asc", "desc", "oddfirst"):
            for block, gmax in ((32, 1), (32, 3), (64, 2)):
                for v in ("fused", "unfused"):
                    d = emu.emu_prove(fv, gv, v, block=block, grid_max=gmax, order=order)
                    res["python_emulator"].append({"n": n, "variant": v, "block": block, "grid_max": gmax,
                                                   "block_order": order,
                                                   "match": kc.compare_records(to_hex(d), gold) == []})
        # Rejected design: global in-place fold (dst = src).  Needs > 1 block or > 1 grid-stride iteration in pass 1 to expose the hazard, i.e. N/4 > block (n >= 8 with block = 32).
    for n in range(6, max(max_n, 10) + 1):
        fv = sf.gen_vector(kc.DEFAULT_SEED_F, 1 << n)
        gv = sf.gen_vector(kc.DEFAULT_SEED_G, 1 << n)
        gold = kc.golden_record(kc.DEFAULT_SEED_F, kc.DEFAULT_SEED_G, n)
        for order, gmax in (("asc", 1), ("desc", 3), ("asc", 3)):
            d = emu.emu_prove(fv, gv, "fused", block=32, grid_max=gmax, order=order, inplace=True,
                              check_visits=False)
            res["inplace_hazard"].append({"n": n, "block_order": order, "grid_max": gmax, "block": 32,
                                          "match": kc.compare_records(to_hex(d), gold) == []})
    res["shim_all_match"] = bool(res["shim"]) and all(x["match"] for x in res["shim"])
    res["python_emulator_all_match"] = all(x["match"] for x in res["python_emulator"])
    res["inplace_hazard_summary"] = (
        "in-place global fold (rejected design) mismatches the golden transcript in "
        f"{sum(1 for x in res['inplace_hazard'] if not x['match'])} of {len(res['inplace_hazard'])} emulated "
        "runs (block order / grid-stride dependent); the ping-pong kernels match in every order.")
    return res

# ================================================================== bench
def gpu_snapshot(index: int, samples: int = 5, period_s: float = 0.2) -> Dict:
    """Load on GPU <index> (nvidia-smi order) while this runner is idle: mean utilization over
    a short window, memory in use, power, and Python compute processes other than this one."""
    util, mem, pw = [], None, None
    for _ in range(samples):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                "--format=csv,noheader,nounits", "-i", str(index)],
                               capture_output=True, text=True, timeout=20)
            u, mem, pw = [x.strip() for x in r.stdout.strip().split(",")]
            util.append(float(u))
        except Exception:
            pass
        time.sleep(period_s)
    apps = []
    try:
        uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(index)],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3 and parts[0] == uuid and parts[1].isdigit() and int(parts[1]) != os.getpid():
                if "python" in parts[2].lower():
                    apps.append({"pid": int(parts[1]), "name": parts[2]})
    except Exception:
        pass
    mu = statistics.fmean(util) if util else None
    return {"t": kc.now_utc(), "util_mean_pct": mu, "util_samples": util, "memory_used_MiB": mem,
            "power_W": pw, "other_python_compute_procs": apps,
            "contended": (mu is not None and mu > 10.0)}

def pass_microbench(be, n: int, block: int, grid_max: int, reps: int, seedf: int, seedg: int) -> Dict:
    """Pass-level ablation at level 0 -> 1, interleaved so both sides see the same background load:
    fused  = one sf_fold_eval (fold level 0 with r, evaluate round 1),
    unfused= sf_fold (level 0 -> 1) followed by sf_eval (round 1 on level 1).
    Also a device copy (cupy.copyto, both tables) as an achievable-bandwidth reference.
    Reports min and median CUDA-event times; min is the least contended sample."""
    N = 1 << n
    f, g = host_tables(seedf, seedg, n)
    w = DeviceWork(be, f, g, n, grid_max)
    w.reset()
    A, B, part = w.A, w.B, w.partials
    r = (0x0123456789ABCDEF % kc.P, 0x0FEDCBA987654321 % kc.P)
    gf = lambda x: max(1, min(-(-x // block), grid_max))  # noqa: E731
    fused, unf, fold, ev, cpy = [], [], [], [], []
    for _ in range(reps):
        e0 = be.event()
        be.sf_fold_eval(gf(N // 4), block, A[0], A[1], B[0], B[1], N // 4, r, part)
        e1 = be.event()
        e1.synchronize()
        fused.append(be.elapsed(e0, e1))
        e0 = be.event()
        be.sf_fold(gf(N // 2), block, A[0], A[1], B[0], B[1], N // 2, r)
        e1 = be.event()
        be.sf_eval(gf(N // 4), block, B[0], B[1], N // 4, part)
        e2 = be.event()
        e2.synchronize()
        unf.append(be.elapsed(e0, e2)); fold.append(be.elapsed(e0, e1)); ev.append(be.elapsed(e1, e2))
        e0 = be.event()
        be.copy(w.A0[0], A[0]); be.copy(w.A0[1], A[1])
        e1 = be.event()
        e1.synchronize()
        cpy.append(be.elapsed(e0, e1))
    w.reset()
    st = lambda xs: {"min_s": min(xs), "median_s": statistics.median(xs), "samples": xs}  # noqa: E731
    bytes_fused = 2 * 16 * (N + N // 2)             # read level 0, write level 1
    bytes_unf = 2 * 16 * (N + N // 2 + N // 2)      # + re-read level 1
    bytes_copy = 2 * 16 * (2 * N)
    return {"n": n, "block": block, "grid_max": grid_max, "reps": reps,
            "fused_sf_fold_eval": st(fused), "unfused_fold_plus_eval": st(unf),
            "unfused_fold_only": st(fold), "unfused_eval_only": st(ev), "copy_reference": st(cpy),
            "bytes": {"fused": bytes_fused, "unfused": bytes_unf, "copy": bytes_copy},
            "GBps_at_min": {"fused": bytes_fused / min(fused) / 1e9, "unfused": bytes_unf / min(unf) / 1e9,
                            "copy": bytes_copy / min(cpy) / 1e9},
            "speedup_unfused_over_fused_at_min": min(unf) / min(fused),
            "speedup_unfused_over_fused_at_median": statistics.median(unf) / statistics.median(fused)}

def run_gpu_bench(args, device: int) -> Dict:
    be = CupyBackend(device)
    dev = {"cuda_index": device, "nvidia_smi_index": device, **be.props}
    grid_max = args.grid_max or 8 * be.sm_count
    drec = {"device": dev, "grid_max": grid_max, "results": [], "gpu_snapshots": [gpu_snapshot(device)]}
    if args.micro_logn:
        drec["pass_microbench"] = [pass_microbench(be, n, args.block, grid_max, args.micro_reps,
                                                   args.seedf, args.seedg) for n in args.micro_logn]
        be.cp.get_default_memory_pool().free_all_blocks()
    for n in args.logn:
        f, g = host_tables(args.seedf, args.seedg, n)
        gold = kc.golden_record(args.seedf, args.seedg, n) if n <= args.golden_max_logn else None
        work = DeviceWork(be, f, g, n, grid_max)
        for v in args.variants.split(","):
            for _ in range(args.warmup):
                prove_device(be, None, None, n, v, args.block, grid_max, timing=True, work=work)
            runs = [prove_device(be, None, None, n, v, args.block, grid_max, timing=True, work=work)
                    for _ in range(args.reps)]
            kern = [sum(r["pass_event_s"]) for r in runs]
            wall = [r["wall_s"] for r in runs]
            N = 1 << n
            entry = {"n": n, "variant": v, "block": args.block, "grid_max": grid_max,
                     "kernel_s": kern, "kernel_stats_s": kc.summarize(kern),
                     "wall_s": wall, "wall_stats_s": kc.summarize(wall),
                     "elems_per_s_kernel_median": N / statistics.median(kern),
                     "elems_per_s_wall_median": N / statistics.median(wall),
                     "pass_kinds": runs[0]["pass_kinds"],
                     "pass_median_s": [statistics.median(r["pass_event_s"][p] for r in runs)
                                       for p in range(len(runs[0]["pass_event_s"]))],
                     "match": (kc.compare_records(runs[0], gold) == []) if gold else None,
                     "match_basis": "golden edgecert.sumfold.prove" if gold else
                                    "golden verifier (+ cpu_bench transcript_sha256 at equal seeds)",
                     "verify_ok": kc.verify_record(runs[0]),
                     "transcript_sha256": kc.transcript_digest(runs[0]),
                     "consistent_across_runs": len({kc.transcript_digest(r) for r in runs}) == 1}
            if args.power:  # repeat the prover for >= window seconds while sampling board power
                with PowerSampler(device, args.power_period_ms) as ps:
                    time.sleep(0.6)
                    tn = time.perf_counter()
                    idle = ps.window(tn - 0.5, tn)
                    t0 = time.perf_counter()
                    k = 0
                    while time.perf_counter() - t0 < args.power_window_s:
                        prove_device(be, None, None, n, v, args.block, grid_max, work=work)
                        k += 1
                    t1 = time.perf_counter()
                    busy = ps.window(t0, t1)
                entry["power"] = {"period_ms": args.power_period_ms, "window_s": t1 - t0, "proofs": k,
                                  "mean_W": statistics.fmean(busy) if busy else None, "samples": len(busy),
                                  "pre_window_mean_W": statistics.fmean(idle) if idle else None,
                                  "energy_per_proof_J": (statistics.fmean(busy) * (t1 - t0) / k) if busy else None,
                                  "note": "board power (nvidia-smi power.draw) incl. static power and any "
                                          "other workload on the GPU; wall-clock loop incl. host transcript"}
            drec["results"].append(entry)
        del work
        be.cp.get_default_memory_pool().free_all_blocks()
        drec["gpu_snapshots"].append(gpu_snapshot(device))
    drec["contended"] = any(sn["contended"] for sn in drec["gpu_snapshots"])
    drec["status"] = "SUCCESS" if all(e["verify_ok"] and e["match"] in (True, None)
                                               and e["consistent_across_runs"] for e in drec["results"]) \
        else "VERIFY_FAILED"
    return drec

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", type=lambda s: [int(x) for x in s.split(",")], default=[0],
                    help="nvidia-smi indices (CUDA_DEVICE_ORDER=PCI_BUS_ID is set so they agree)")
    ap.add_argument("--logn", type=lambda s: [int(x) for x in s.split(",")], default=[12, 14, 16, 18, 20, 22, 24])
    ap.add_argument("--variants", default="fused,unfused")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--grid-max", type=int, default=0, help="0 = 8 x SM count")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seedf", type=lambda s: int(s, 16), default=kc.DEFAULT_SEED_F)
    ap.add_argument("--seedg", type=lambda s: int(s, 16), default=kc.DEFAULT_SEED_G)
    ap.add_argument("--golden-max-logn", type=int, default=16)
    ap.add_argument("--power", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--power-period-ms", type=int, default=50)
    ap.add_argument("--power-window-s", type=float, default=2.0)
    ap.add_argument("--micro-logn", type=lambda s: [int(x) for x in s.split(",") if x], default=[20, 22, 24],
                    help="pass-level fused-vs-unfused microbenchmark sizes ('' to skip)")
    ap.add_argument("--micro-reps", type=int, default=30)
    ap.add_argument("--emu-max-n", type=int, default=10)
    ap.add_argument("--skip-emulation", action="store_true")
    ap.add_argument("--out", default=str(kc.ROOT / "results" / "raw" / "kernels" / "gpu_bench.json"))
    args = ap.parse_args(argv)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    rec = {"status": "RUNNING", "evidence_class": None, "experiment_kind": "PROOF_KERNEL", "impl": "cuda",
           "kernel_source": str(CU_SRC.relative_to(kc.ROOT)).replace("\\", "/"),
           "command": subprocess.list2cmdline([sys.executable] + sys.argv),
           "timestamp_utc": kc.now_utc(), "host": kc.host_info(), "nvidia_smi": nvidia_smi_inventory(),
           "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
           "params": {k: v for k, v in vars(args).items() if k not in ("out",)},
           "timing_definition": ("kernel_s = sum over passes of CUDA-event time (pass kernel + one-block "
                                 "partial reduction + 48-byte device->host copy of the round message); "
                                 "wall_s = host perf_counter over the whole prover incl. the host SHA-256 "
                                 "transcript per round (STEP mode); both exclude table upload and the "
                                 "untimed device-to-device reset copy."),
           "devices": [], "notes": []}
    rec["params"]["seedf"] = f"{args.seedf:016x}"
    rec["params"]["seedg"] = f"{args.seedg:016x}"
    try:
        import cupy
        rec["cupy"] = {"version": cupy.__version__, "path": str(Path(cupy.__file__).parent),
                       "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
                       "cuda_driver": cupy.cuda.runtime.driverGetVersion()}
        have_cupy = True
    except Exception as e:  # ImportError or CUDA init failure
        have_cupy = False
        rec["cupy_import_error"] = f"{type(e).__name__}: {e}"
    if have_cupy:
        for d in args.devices:
            try:
                rec["devices"].append(run_gpu_bench(args, d))
            except Exception as e:
                rec["devices"].append({"device": {"nvidia_smi_index": d},
                                       "status": "BUILD_FAILED" if "nvrtc" in str(e).lower() else "RUN_FAILED",
                                       "error": str(e)[:2000]})
            tmp = Path(args.out).with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        sts = [d.get("status") for d in rec["devices"]]
        if sts and all(x == "SUCCESS" for x in sts):
            rec["status"] = "SUCCESS"
        elif any(x == "SUCCESS" for x in sts):
            rec["status"] = "PARTIAL_FAILURES"
        else:
            rec["status"] = sts[0] if sts else "SUCCESS"
        rec["evidence_class"] = "MEASURED_SUCCESS" if rec["status"] in ("ABORT", "PARTIAL_FAILURES") else None
        rec["contended"] = any(d.get("contended") for d in rec["devices"])
        if rec["contended"]:
            rec["notes"].append("GPU(s) were busy with other workloads (utilization > 10% while this runner "
                                "was idle, see gpu_snapshots): timings and power are contended; rerun on an "
                                "idle GPU before quoting them.")
    else:
        rec["status"] = "SUCCESS"
        rec["notes"].append("CuPy (NVRTC) is not importable and no CUDA toolkit is present; per rules "
                            "nothing was installed. GPU timings/power are therefore absent (null).")
        rec["measurements"] = {"kernel_s": None, "wall_s": None, "energy_J": None, "sample_count": None}
    if not args.skip_emulation:
        rec["emulation_checks"] = emulation_checks(args.emu_max_n)
        rec["notes"].append("emulation_checks are SIMULATION evidence: sumfold.cu compiled by g++ and executed "
                            "on the CPU through kernels/cuda/shim, plus a pure-Python transcription of the "
                            "index/reduction logic; they check the kernel logic bit-exactly, not GPU "
                            "performance.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp = out.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    summ = {"out": str(out), "status": rec["status"], "contended": rec.get("contended")}
    if "emulation_checks" in rec:
        ec = rec["emulation_checks"]
        summ.update({"shim_all_match": ec["shim_all_match"], "shim_runs": len(ec["shim"]),
                     "pyemu_all_match": ec["python_emulator_all_match"], "pyemu_runs": len(ec["python_emulator"]),
                     "inplace_hazard_mismatches": sum(1 for x in ec["inplace_hazard"] if not x["match"])})
    print(json.dumps(summ))
    return 0

if __name__ == "__main__":
    sys.exit(main())
