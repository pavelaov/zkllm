"""GPT-2 124M: TRANSPARENT_REFERENCE sumcheck proofs for every matrix multiplication of a prefill.

Label TRANSPARENT_REFERENCE: no zero knowledge, no polynomial commitment, not succinct (the verifier reads A, W and C).  This is the arithmetic core of certifying the linear part of a real model (MLE partial evaluation + product sumcheck), NOT an inference certificate.

Workload per prefill of T tokens (T in {32, 128, 512} by default):
  per layer l = 0..11
    attn.c_attn   A: T x 768   W: 768 x 2304   (real GPT-2 weights, int8 per output channel)
    attn.qk  x12  A: Q_h T x 64   W: K_h^T 64 x T        (activation x activation, per head)
    attn.pv  x12  A: P_h T x T (uint16, [0, 2^15], causal zeros)   W: V_h T x 64
    attn.c_proj   A: T x 768   W: 768 x 768
    mlp.c_fc      A: T x 768   W: 768 x 3072
    mlp.c_proj    A: T x 3072  W: 3072 x 768
  lm_head.emit    A: 1 x 768   W: wte^T 768 x 50257 (tied, int8 per token row); spec/numerical_
                  semantics.md 3.8: only the emitting position evaluates the LM head in prefill
  lm_head.all     same with all T rows (reported separately, not in the total)
Weights: models/gpt2/model.safetensors quantised with the spec rule
  W_q = clamp(RHAZ(W / s_w), -127, 127), s_w = max |W| / 127 per output channel (per token row for
  the tied embedding / LM head) using edgecert.reference.intref.quantize_rows.
Activations: seeded random int8 (uint16 for P) of the right shapes (the integer reference exists
  only for the synthetic T0 models; values do not affect the cost).  C = A W exactly (float64 BLAS
  on integers < 2^53, checked by the proof itself).

Backends (each in its own worker process, sequentially): cpu1 (C library, 1 thread), cpu32 (C library, 32 OpenMP threads; attention heads of a layer run concurrently, 1 thread each), gpu<i> (CuPy/NVRTC, device i in nvidia-smi order via CUDA_DEVICE_ORDER=PCI_BUS_ID; heads of a layer are one batched launch sequence).  All backends must produce bit-identical proofs (checked).

Timing per group (one linear matmul, or the 12 heads of one attention matmul):
  wall_s      host perf_counter of the whole group (bind + prover + result unpacking)
  bind_s      instance digests (SHA-256 of A and C bytes; digest(W) of weights is precomputed once
              as the model-commitment stand-in) + Fiat-Shamir challenges r1, r2 (host)
  fold_a_s, fold_w_s, sumcheck_s, core_s
              CPU: C-internal QueryPerformanceCounter per proof (summed over concurrent heads, i.e.
              CPU-seconds for the cpu32 attention groups); GPU: CUDA events (device time)

Contention: every record stores Get-Process Vivado/xsim snapshots, machine load and nvidia-smi utilization of BOTH GPUs before and after each worker.  Records are flagged contended = true if any of them shows foreign load.  --final reruns everything with the full repetition counts for an idle host and writes *_final.json.

Examples python bench/runners/gpt2_linear_prove.py                       # T 32,128,512; cpu1,cpu32,gpu0,gpu1 python bench/runners/gpt2_linear_prove.py --quick               # T 32 only, 1 warm-up + 2 reps python bench/runners/gpt2_linear_prove.py --final               # idle-host rerun (orchestrator) python bench/runners/gpt2_linear_prove.py --opcount-only        # ANALYTIC file only
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import ctypes
import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_common as kc  # noqa: E402  (host info, contention probe, stats)

ROOT = kc.ROOT
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from edgecert.prover_adapters import matmul_sumcheck as ms  # noqa: E402

MODEL_DIR = ROOT / "models" / "gpt2"
MODEL_FILE = MODEL_DIR / "model.safetensors"
OUT_DIR = ROOT / "results" / "raw" / "gpt2prove"
N_LAYER, D, H, DH, FF, VOCAB = 12, 768, 12, 64, 3072, 50257
SEED = 20260927
TYPES = ["attn_c_attn", "attn_qk", "attn_pv", "attn_c_proj", "mlp_c_fc", "mlp_c_proj", "lm_head_emit"]
EXTRA_TYPES = ["lm_head_all"]

# ====================================================================== helpers
def now() -> str:
    return kc.now_utc()

def proc_memory() -> Dict:
    """Current / peak working set and peak commit of this process (Windows psapi)."""
    if os.name != "nt":
        return {}

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    ok = k32.K32GetProcessMemoryInfo(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.byref(c), c.cb)
    if not ok:
        return {}
    return {"working_set_bytes": int(c.WorkingSetSize), "peak_working_set_bytes": int(c.PeakWorkingSetSize),
            "commit_bytes": int(c.PagefileUsage), "peak_commit_bytes": int(c.PeakPagefileUsage)}

def gpu_snapshot_all() -> List[Dict]:
    try:
        import kernel_gpu_bench as kg
        return [dict(kg.gpu_snapshot(i, samples=5, period_s=0.2), nvidia_smi_index=i) for i in (0, 1)]
    except Exception as e:  # pragma: no cover
        return [{"error": str(e)[:300]}]

def contention_snapshot() -> Dict:
    c = kc.contended_now(1.0)
    g = gpu_snapshot_all()
    return {"t": now(), "vivado_xsim": c, "gpus": g,
            "cpu_contended": bool(c.get("contended")) or bool(c.get("host_busy")),
            "gpu_contended": {str(x.get("nvidia_smi_index")): x.get("contended") for x in g}}

def nvidia_smi_inventory() -> Dict:
    try:
        import kernel_gpu_bench as kg
        return kg.nvidia_smi_inventory()
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}

def code_digests() -> Dict[str, str]:
    files = ["kernels/cpu/mle.c", "kernels/cpu/mle.h", "kernels/cpu/sumfold.c", "kernels/cpu/gl_field.h",
             "kernels/cuda/mle.cu", "kernels/cuda/sumfold.cu", "src/edgecert/prover_adapters/matmul_sumcheck.py",
             "bench/runners/gpt2_linear_prove.py", "src/edgecert/field.py", "src/edgecert/sumfold.py"]
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16] for f in files}

# ====================================================================== model
def read_safetensors(path: Path, names: List[str]) -> Dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    base = 8 + n
    out = {}
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for nm in names:
        e = hdr[nm]
        assert e["dtype"] == "F32", e
        a, b = e["data_offsets"]
        out[nm] = np.frombuffer(mm[base + a:base + b].tobytes(), dtype=np.float32).reshape(e["shape"])
    return out

def cache_dir(args) -> Path:
    d = Path(args.cache_dir) if args.cache_dir else Path(tempfile.gettempdir()) / "edgecert_gpt2prove_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def prepare_model(args) -> Dict:
    """Quantise GPT-2 weights (spec rule) once; cache int8 arrays + weight digests in a temp dir."""
    cd = cache_dir(args)
    st = MODEL_FILE.stat()
    key = f"{st.st_size}-{int(st.st_mtime)}-v1"
    meta_p = cd / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("key") == key and all((cd / f"{k}.npy").exists() for k in meta["digests"]):
            meta["cache_hit"] = True
            return meta
    from edgecert.reference.intref import quantize_rows
    t0 = time.perf_counter()
    h = hashlib.sha256()
    with open(MODEL_FILE, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    file_sha = h.hexdigest()
    names = ["wte.weight"] + [f"h.{l}.{m}.weight" for l in range(N_LAYER)
                              for m in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")]
    Wf = read_safetensors(MODEL_FILE, names)
    digests, qstats = {}, {}
    for nm in names:
        w = Wf[nm].astype(np.float64)
        if nm == "wte.weight":            # (V x d), per row (token)
            q, s = quantize_rows(w, 127)
            key_nm = "wte_q"
        else:                             # Conv1D (in x out) = (K x N), per output channel = column
            qT, s = quantize_rows(w.T, 127)
            q = qT.T
            key_nm = nm.replace("h.", "L").replace(".weight", "").replace(".", "_")
        q8 = np.ascontiguousarray(q.astype(np.int8))
        assert int(np.abs(q).max()) <= 127
        np.save(cd / f"{key_nm}.npy", q8)
        logical = q8.T if key_nm == "wte_q" else q8     # logical W (K x N) of the matmul
        digests[key_nm] = ms.mat_digest(logical).hex()
        qstats[key_nm] = {"shape": list(q8.shape), "scale_min": float(s.min()), "scale_max": float(s.max()),
                          "zero_fraction": float(np.mean(q8 == 0)), "abs_max": int(np.abs(q8).max())}
    meta = {"key": key, "model_file": str(MODEL_FILE.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": file_sha, "digests": digests, "quant_stats": qstats,
            "quant_rule": "W_q = clamp(RHAZ(W / s_w), -127, 127), s_w = max|W|/127 per output channel "
                          "(Conv1D column); tied wte/LM head per token row (spec/numerical_semantics.md 1)",
            "quantize_s": time.perf_counter() - t0, "cache_dir": str(cd), "cache_hit": False}
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta

def load_weights(meta: Dict) -> Dict[str, np.ndarray]:
    cd = Path(meta["cache_dir"])
    return {k: np.load(cd / f"{k}.npy") for k in meta["digests"]}

# ====================================================================== workload
def rng_for(T: int, layer: int, name: str) -> np.random.Generator:
    return np.random.default_rng([SEED, T, layer, int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)])

def i8(rng, shape):
    return rng.integers(-128, 128, size=shape, dtype=np.int8)

def exact_mm(A: np.ndarray, W: np.ndarray, out_dtype=np.int32) -> np.ndarray:
    """C = A W on integers via float64 BLAS; exact because every |partial sum| < 2^53."""
    C = np.rint(A.astype(np.float64) @ W.astype(np.float64))
    assert np.abs(C).max() < 2**31
    return C.astype(out_dtype)

ACT_SOURCE = ("seeded random int8 (P: uint16 in [0, 2^15] with causal zeros), numpy default_rng([%d, T, layer, "
              "int(sha256(name)[:8], 16)]); the integer reference exists only for the synthetic T0 models, not for "
              "real GPT-2, and values do not affect the cost" % SEED)
WL_VERSION = "wl-v1"

def make_tensors(T: int, Wq: Dict[str, np.ndarray], with_lm_all: bool = True) -> Dict[str, np.ndarray]:
    """Activations (seeded random int8 / uint16) and exact outputs C = A W for one prefill of T tokens."""
    t = {}
    causal = np.tril(np.ones((T, T), dtype=bool))
    for l in range(N_LAYER):
        pre = f"L{l}_"
        X1 = i8(rng_for(T, l, "x_attn"), (T, D))
        Q, Kx, V = (i8(rng_for(T, l, nm), (T, D)) for nm in ("q", "k", "v"))
        Pm = rng_for(T, l, "p").integers(0, 2**15 + 1, size=(H, T, T)).astype(np.uint16)
        Pm[:, ~causal] = 0
        X2 = i8(rng_for(T, l, "x_proj"), (T, D))
        X3 = i8(rng_for(T, l, "x_fc"), (T, D))
        X4 = i8(rng_for(T, l, "x_mproj"), (T, FF))
        Qh = Q.reshape(T, H, DH).transpose(1, 0, 2).astype(np.float64)
        Kh = Kx.reshape(T, H, DH).transpose(1, 0, 2).astype(np.float64)
        Vh = V.reshape(T, H, DH).transpose(1, 0, 2).astype(np.float64)
        S = np.rint(Qh @ Kh.transpose(0, 2, 1)).astype(np.int32)                 # (H, T, T)
        Ctx = np.rint(Pm.astype(np.float64) @ Vh).astype(np.int64)             # |ctx| <= T * 2^15 * 128
        if Ctx.min() >= -2**31 and Ctx.max() < 2**31:
            Ctx = Ctx.astype(np.int32)
        t.update({pre + "X1": X1, pre + "Q": Q, pre + "K": Kx, pre + "V": V, pre + "P": Pm, pre + "X2": X2,
                  pre + "X3": X3, pre + "X4": X4, pre + "S": S, pre + "Ctx": np.ascontiguousarray(Ctx),
                  pre + "C_attn": exact_mm(X1, Wq[pre + "attn_c_attn"]),
                  pre + "C_proj": exact_mm(X2, Wq[pre + "attn_c_proj"]),
                  pre + "C_fc": exact_mm(X3, Wq[pre + "mlp_c_fc"]),
                  pre + "C_mproj": exact_mm(X4, Wq[pre + "mlp_c_proj"])})
    wte = Wq["wte_q"]                   # (V x d); logical LM-head W = wte^T (d x V), a strided view
    t["Xf"] = i8(rng_for(T, 99, "x_final"), (1, D))
    t["Cf"] = exact_mm(t["Xf"], wte.T)
    if with_lm_all:
        t["Xa"] = i8(rng_for(T, 99, "x_final_all"), (T, D))
        t["Ca"] = exact_mm(t["Xa"], wte.T)
    return t

def workload_dir(meta: Dict, T: int, with_lm_all: bool) -> Path:
    return Path(meta["cache_dir"]) / f"workload_T{T}_{WL_VERSION}_{SEED}{'' if with_lm_all else '_nolm'}"

def save_workload(meta: Dict, T: int, Wq: Dict[str, np.ndarray], with_lm_all: bool = True) -> Dict:
    """Parent side: generate once per T and store as .npy in the temp cache (the worker only loads, so
    its peak working set is model + workload + prover, without float64 generation temporaries)."""
    d = workload_dir(meta, T, with_lm_all)
    done = d / "done.json"
    if done.exists() and json.loads(done.read_text()).get("model_key") == meta["key"]:
        return json.loads(done.read_text())
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tens = make_tensors(T, Wq, with_lm_all)
    for k, v in tens.items():
        np.save(d / f"{k}.npy", v)
    info = {"model_key": meta["key"], "names": sorted(tens), "bytes": int(sum(v.nbytes for v in tens.values())),
            "generate_s": time.perf_counter() - t0}
    done.write_text(json.dumps(info))
    return info

def load_workload(meta: Dict, T: int, with_lm_all: bool = True) -> Dict[str, np.ndarray]:
    d = workload_dir(meta, T, with_lm_all)
    info = json.loads((d / "done.json").read_text())
    return {k: np.load(d / f"{k}.npy") for k in info["names"]}

def build_workload(T: int, Wq: Dict[str, np.ndarray], meta: Dict, tens: Dict[str, np.ndarray],
                   with_lm_all: bool = True) -> Dict:
    """Instance groups (numpy views into the loaded tensors)."""
    dig = {k: bytes.fromhex(v) for k, v in meta["digests"].items()}
    mk = ms.MatmulInstance
    groups = []
    for l in range(N_LAYER):
        pre = f"L{l}_"
        g = lambda k: tens[pre + k]  # noqa: E731
        Q, Kx, V, Pm, S, Ctx = g("Q"), g("K"), g("V"), g("P"), g("S"), g("Ctx")
        groups.append({"name": f"L{l}.attn_c_attn", "type": "attn_c_attn", "layer": l, "dev": ("X1", "attn_c_attn"),
                       "insts": [mk(f"gpt2.L{l}.attn.c_attn", g("X1"), Wq[pre + "attn_c_attn"], g("C_attn"),
                                    dig[pre + "attn_c_attn"])]})
        groups.append({"name": f"L{l}.attn_qk", "type": "attn_qk", "layer": l, "dev": ("Q", "K"),
                       "insts": [mk(f"gpt2.L{l}.attn.qk.h{h}", Q[:, h * DH:(h + 1) * DH],
                                    Kx[:, h * DH:(h + 1) * DH].T, S[h]) for h in range(H)]})
        groups.append({"name": f"L{l}.attn_pv", "type": "attn_pv", "layer": l, "dev": ("P", "V"),
                       "insts": [mk(f"gpt2.L{l}.attn.pv.h{h}", Pm[h], V[:, h * DH:(h + 1) * DH], Ctx[h])
                                 for h in range(H)]})
        groups.append({"name": f"L{l}.attn_c_proj", "type": "attn_c_proj", "layer": l, "dev": ("X2", "attn_c_proj"),
                       "insts": [mk(f"gpt2.L{l}.attn.c_proj", g("X2"), Wq[pre + "attn_c_proj"], g("C_proj"),
                                    dig[pre + "attn_c_proj"])]})
        groups.append({"name": f"L{l}.mlp_c_fc", "type": "mlp_c_fc", "layer": l, "dev": ("X3", "mlp_c_fc"),
                       "insts": [mk(f"gpt2.L{l}.mlp.c_fc", g("X3"), Wq[pre + "mlp_c_fc"], g("C_fc"),
                                    dig[pre + "mlp_c_fc"])]})
        groups.append({"name": f"L{l}.mlp_c_proj", "type": "mlp_c_proj", "layer": l, "dev": ("X4", "mlp_c_proj"),
                       "insts": [mk(f"gpt2.L{l}.mlp.c_proj", g("X4"), Wq[pre + "mlp_c_proj"], g("C_mproj"),
                                    dig[pre + "mlp_c_proj"])]})
    wte = Wq["wte_q"]
    groups.append({"name": "lm_head_emit", "type": "lm_head_emit", "layer": None, "dev": ("Xf", "wte"),
                   "insts": [mk("gpt2.lm_head.emit", tens["Xf"], wte.T, tens["Cf"], dig["wte_q"])]})
    if with_lm_all:
        groups.append({"name": "lm_head_all", "type": "lm_head_all", "layer": None, "dev": ("Xa", "wte"),
                       "insts": [mk("gpt2.lm_head.all", tens["Xa"], wte.T, tens["Ca"], dig["wte_q"])]})
    dev_tensors = {k: v for k, v in tens.items() if not (k.startswith("C") or "_C" in k or k.endswith("_S"))}
    return {"T": T, "groups": groups, "tensors": dev_tensors,
            "workload_bytes": int(sum(v.nbytes for v in tens.values())), "activation_source": ACT_SOURCE}

# ====================================================================== backends
class CpuRunner:
    def __init__(self, threads: int):
        self.threads = threads
        self.be = ms.CpuBackend()
        self.pool = cf.ThreadPoolExecutor(max_workers=min(H, threads)) if threads > 1 else None
        self.name = f"cpu{threads}"

    def info(self) -> Dict:
        return {"backend": "cpu", "threads": self.threads, "dll": self.be.lib.build_info,
                "dll_flags": self.be.lib.flags,
                "attention_heads": "concurrent (thread pool, 1 C thread per head)" if self.pool else "sequential"}

    def run_group(self, g) -> (List, Dict):
        insts = g["insts"]
        t0 = time.perf_counter()
        if len(insts) == 1:
            res = [self.be.prove(insts[0], threads=self.threads)]
        elif self.pool is not None:
            res = list(self.pool.map(lambda x: self.be.prove(x, threads=1), insts))
        else:
            res = [self.be.prove(x, threads=1) for x in insts]
        wall = time.perf_counter() - t0
        tm = {"wall_s": wall}
        for k in ("bind_s", "fold_a_s", "fold_w_s", "sumcheck_s", "core_s"):
            tm[k] = sum(r[1][k] for r in res)
        return [r[0] for r in res], tm

    def close(self):
        if self.pool:
            self.pool.shutdown()

class GpuRunner:
    def __init__(self, device: int, wl: Dict, Wq: Dict):
        self.be = ms.GpuBackend(device)
        self.name = f"gpu{device}"
        cp = self.be.cp
        pool = cp.get_default_memory_pool()
        t0 = time.perf_counter()
        self.dev = {}
        for k, v in Wq.items():                       # resident model
            self.dev["W:" + k] = self.be.to_device(v)
        cp.cuda.Device(device).synchronize()
        self.model_bytes = int(pool.total_bytes())
        for k, v in wl["tensors"].items():            # resident activations of this prefill
            self.dev["A:" + k] = self.be.to_device(v)
        cp.cuda.Device(device).synchronize()
        self.upload_s = time.perf_counter() - t0
        self.resident_bytes = int(pool.total_bytes())
        self.hpool = cf.ThreadPoolExecutor(max_workers=H)

    def info(self) -> Dict:
        return {"backend": "gpu", "device": self.be.props, "kernel_attrs": self.be.kernel_attrs,
                "resident_model_bytes": self.model_bytes, "resident_total_bytes": self.resident_bytes,
                "upload_s_untimed": self.upload_s, "block": self.be.BLOCK, "sumcheck_block": self.be.SC_BLOCK}

    def specs(self, g):
        a_key, w_key = g["dev"]
        t = g["type"]
        T = g["insts"][0].shape[0]
        if t == "attn_qk":
            l = g["layer"]
            return (ms.DevMat(self.dev[f"A:L{l}_Q"], ms.MLE_I8, 0, DH, D, 1),
                    ms.DevMat(self.dev[f"A:L{l}_K"], ms.MLE_I8, 0, DH, 1, D))
        if t == "attn_pv":
            l = g["layer"]
            return (ms.DevMat(self.dev[f"A:L{l}_P"], ms.MLE_U16, 0, T * T, T, 1),
                    ms.DevMat(self.dev[f"A:L{l}_V"], ms.MLE_I8, 0, DH, D, 1))
        inst = g["insts"][0]
        A = self.dev[f"A:L{g['layer']}_{a_key}" if g["layer"] is not None else f"A:{a_key}"]
        if w_key == "wte":
            W = ms.DevMat(self.dev["W:wte_q"], ms.MLE_I8, 0, 0, 1, D)          # W[k][j] = wte[j][k]
        else:
            Wd = self.dev[f"W:L{g['layer']}_{w_key}"]
            W = ms.DevMat(Wd, ms.MLE_I8, 0, 0, inst.W.shape[1], 1)
        return ms.DevMat(A, ms.MLE_I8, 0, 0, inst.A.shape[1], 1), W

    def run_group(self, g):
        dA, dW = self.specs(g)
        proofs, tm = self.be.prove_batch(g["insts"], dA, dW, pool=self.hpool)
        return proofs, tm

    def close(self):
        self.hpool.shutdown()

# ====================================================================== worker
def summarize(xs: List[float]) -> Optional[Dict]:
    return kc.summarize(xs) if xs else None

def group_digest(proofs) -> str:
    h = hashlib.sha256()
    for p in proofs:
        h.update(p.layer_id.encode() + b"|" + p.to_bytes())
    return h.hexdigest()

def run_worker(args) -> Dict:
    t_start = time.perf_counter()
    rec = {"status": "RUNNING", "evidence_class": "MEASURED", "label": ms.LABEL, "backend_config": args.backend,
           "T": args.T[0], "timestamp_utc": now(), "pid": os.getpid()}
    rec["contention_before"] = contention_snapshot()
    meta = prepare_model(args)
    Wq = load_weights(meta)
    rec["memory_after_model_load"] = proc_memory()
    T = args.T[0]
    t0 = time.perf_counter()
    tens = load_workload(meta, T, not args.no_lm_all)
    wl = build_workload(T, Wq, meta, tens, with_lm_all=not args.no_lm_all)
    rec["workload_load_s_untimed"] = time.perf_counter() - t0
    rec["activation_source"] = wl["activation_source"]
    rec["resident_bytes"] = {"model_int8": int(sum(v.nbytes for v in Wq.values())), "workload": wl["workload_bytes"]}
    rec["memory_after_workload"] = proc_memory()
    if args.backend.startswith("cpu"):
        runner = CpuRunner(int(args.backend[3:]))
    else:
        runner = GpuRunner(int(args.backend[3:]), wl, Wq)
    rec["backend"] = runner.info()
    groups = wl["groups"]
    lib = ms.get_lib()
    lib.scratch_reset()
    reps_rec = {g["name"]: {k: [] for k in ("wall_s", "bind_s", "fold_a_s", "fold_w_s", "sumcheck_s", "core_s")}
                for g in groups}
    digests = {g["name"]: set() for g in groups}
    rep_wall = []
    last_proofs = {}
    nw, nr = args.warmup, args.reps
    for it in range(nw + nr):
        tr0 = time.perf_counter()
        for g in groups:
            proofs, tm = runner.run_group(g)
            digests[g["name"]].add(group_digest(proofs))
            if it >= nw:
                for k in reps_rec[g["name"]]:
                    reps_rec[g["name"]][k].append(tm[k])
            if it == nw + nr - 1:
                last_proofs[g["name"]] = proofs
        if it >= nw:
            rep_wall.append(time.perf_counter() - tr0)
    rec["memory_after_prover"] = proc_memory()
    rec["prover_scratch_peak_bytes_cpu_lib"] = lib.scratch_peak()
    if isinstance(runner, GpuRunner):
        rec["gpu_pool_peak_bytes"] = runner.be.peak_pool_bytes
        rec["gpu_prover_scratch_peak_bytes"] = runner.be.peak_pool_bytes - runner.resident_bytes
    rec["counts"] = {"warmup": nw, "reps": nr, "groups": len(groups),
                     "proofs": sum(len(g["insts"]) for g in groups if g["type"] in TYPES),
                     "proofs_extra": sum(len(g["insts"]) for g in groups if g["type"] in EXTRA_TYPES)}
    rec["deterministic_across_reps"] = all(len(v) == 1 for v in digests.values())
    rec["group_proof_digests"] = {k: sorted(v)[0] for k, v in digests.items()}
    rec["groups"] = []
    for g in groups:
        pf0 = last_proofs[g["name"]][0]
        rec["groups"].append({"name": g["name"], "type": g["type"], "layer": g["layer"], "batch": len(g["insts"]),
                              "shape_TKN": list(g["insts"][0].shape), "kappa": pf0.kappa,
                              "proof_bytes_each": pf0.n_bytes, "reps": reps_rec[g["name"]]})

    # per-type aggregation (sum over the layers of each rep)
    def agg(types: List[str]) -> Dict:
        out = {}
        for ty in types:
            gs = [g for g in rec["groups"] if g["type"] == ty]
            if not gs:
                continue
            d = {"groups": len(gs), "proofs": sum(g["batch"] for g in gs),
                 "proof_bytes": sum(g["batch"] * g["proof_bytes_each"] for g in gs)}
            for k in ("wall_s", "bind_s", "fold_a_s", "fold_w_s", "sumcheck_s", "core_s"):
                per_rep = [sum(g["reps"][k][r] for g in gs) for r in range(nr)]
                d[k] = summarize(per_rep)
            fold_rep = [sum(g["reps"]["fold_a_s"][r] + g["reps"]["fold_w_s"][r] for g in gs) for r in range(nr)]
            d["fold_s"] = summarize(fold_rep)
            out[ty] = d
        return out

    rec["per_type"] = agg(TYPES + EXTRA_TYPES)
    main = [g for g in rec["groups"] if g["type"] in TYPES]
    tot = {}
    for k in ("wall_s", "bind_s", "fold_a_s", "fold_w_s", "sumcheck_s", "core_s"):
        tot[k] = summarize([sum(g["reps"][k][r] for g in main) for r in range(nr)])
    tot["fold_s"] = summarize([sum(g["reps"]["fold_a_s"][r] + g["reps"]["fold_w_s"][r] for g in main)
                               for r in range(nr)])
    tot["proof_bytes"] = sum(g["batch"] * g["proof_bytes_each"] for g in main)
    tot["wall_per_token_s_median"] = tot["wall_s"]["median"] / T
    tot["core_per_token_s_median"] = tot["core_s"]["median"] / T
    rec["total_prefill"] = tot
    rec["rep_wall_s_incl_lm_head_all"] = summarize(rep_wall)

    # ---------------------------------------------------------------- verifier
    vthreads = runner.threads if isinstance(runner, CpuRunner) else 32
    all_proofs = [(g, pf, inst) for g in groups for pf, inst in zip(last_proofs[g["name"]], g["insts"])]
    ok = [ms.verify_matmul(pf, inst, "c", threads=vthreads, lib=lib) for g, pf, inst in all_proofs]
    rec["verifier_all_accept"] = all(ok)
    rec["verifier_rejects"] = [pf.layer_id for (g, pf, inst), o in zip(all_proofs, ok) if not o]
    vrec = None
    if isinstance(runner, CpuRunner) and args.verify_reps > 0:
        vw, vr = args.verify_warmup, args.verify_reps
        vkeys = ("v_wall_s", "v_total_s", "v_bind_s", "v_claim_s", "v_sumcheck_s", "v_final_s")
        per = {ty: {k: [] for k in vkeys} for ty in TYPES + EXTRA_TYPES}

        def v_one(pf, inst, th):
            tm = {}
            ms._verify(pf, inst, "c", th, lib, tm)
            return tm

        for it in range(vw + vr):
            acc = {ty: {k: 0.0 for k in vkeys} for ty in per}
            for g in groups:
                pairs = list(zip(last_proofs[g["name"]], g["insts"]))
                tg = time.perf_counter()
                if len(pairs) > 1 and runner.pool is not None:     # heads verified concurrently, 1 thread each
                    tms = list(runner.pool.map(lambda x: v_one(x[0], x[1], 1), pairs))
                else:
                    tms = [v_one(pf, inst, vthreads) for pf, inst in pairs]
                acc[g["type"]]["v_wall_s"] += time.perf_counter() - tg
                for tm in tms:
                    for k in vkeys[1:]:
                        acc[g["type"]][k] += tm[k]
            if it >= vw:
                for ty in per:
                    for k in per[ty]:
                        per[ty][k].append(acc[ty][k])
        present = {g["type"] for g in groups}
        vrec = {"threads": vthreads, "warmup": vw, "reps": vr, "engine": "C eq-table contraction (mlev_*) + Python "
                "transcript/sumcheck checks (golden interp_quadratic, BoundSumcheckTranscript)",
                "timing_semantics": "v_wall_s = wall per group summed over groups; v_* components are summed per "
                                    "proof (CPU-seconds when heads are verified concurrently)",
                "per_type": {ty: {k: summarize(v) for k, v in per[ty].items()} for ty in per if ty in present}}
        vrec["total_prefill"] = {k: summarize([sum(per[ty][k][r] for ty in TYPES if ty in present)
                                               for r in range(vr)]) for k in vkeys}
    rec["verifier"] = vrec

    # ---------------------------------------------------------------- golden spot check
    if args.golden_check and isinstance(runner, CpuRunner):
        gc = []
        py = ms.PythonBackend()
        picks = [("L0.attn_c_proj", 0), ("L0.attn_qk", 0), ("L0.attn_pv", 0), ("L11.attn_pv", 11)]
        if T <= 32:
            picks.append(("L0.attn_c_attn", 0))
        for gname, hidx in picks:
            g = next(x for x in groups if x["name"] == gname)
            inst = g["insts"][hidx if len(g["insts"]) > 1 else 0]
            t0 = time.perf_counter()
            ref, _ = py.prove(inst)
            dt = time.perf_counter() - t0
            mine = next(p for p in last_proofs[gname] if p.layer_id == inst.layer_id)
            gc.append({"layer_id": inst.layer_id, "shape_TKN": list(inst.shape), "match": ref.to_bytes() == mine.to_bytes(),
                       "python_prove_s": dt, "python_verify_ok": ms.verify_matmul(ref, inst, "c", lib=lib)})
        rec["golden_spot_check"] = {"evidence_class": "MEASURED", "reference": "PythonBackend (edgecert.sumfold.fold, "
                                    "round_message, BoundSumcheckTranscript)", "checks": gc,
                                    "all_match": all(x["match"] for x in gc)}
    runner.close()
    rec["contention_after"] = contention_snapshot()
    cb, ca = rec["contention_before"], rec["contention_after"]
    if isinstance(runner, GpuRunner):
        idx = str(runner.be.device)
        gpu_c = bool(cb["gpu_contended"].get(idx)) or bool(ca["gpu_contended"].get(idx))
    else:
        gpu_c = False
    rec["contended"] = bool(cb["cpu_contended"] or ca["cpu_contended"] or gpu_c)
    rec["contention_flags"] = {"cpu_vivado_or_busy": bool(cb["cpu_contended"] or ca["cpu_contended"]),
                               "gpu_in_use_busy": gpu_c,
                               "any_gpu_busy": any(bool(v) for s in (cb, ca) for v in s["gpu_contended"].values())}
    rec["memory_peak"] = proc_memory()
    rec["worker_wall_s"] = time.perf_counter() - t_start
    good = rec["verifier_all_accept"] and rec["deterministic_across_reps"] and \
        rec.get("golden_spot_check", {"all_match": True})["all_match"]
    rec["status"] = "MEASURED_SUCCESS" if good else "VERIFY_FAILED"
    return rec

# ====================================================================== opcount (ANALYTIC)
def opcount_record(Ts: List[int]) -> Dict:
    rows = {}
    for T in Ts:
        per = {}
        shapes = {"attn_c_attn": (T, D, 3 * D, N_LAYER), "attn_qk": (T, DH, T, N_LAYER * H),
                  "attn_pv": (T, T, DH, N_LAYER * H), "attn_c_proj": (T, D, D, N_LAYER),
                  "mlp_c_fc": (T, D, FF, N_LAYER), "mlp_c_proj": (T, FF, D, N_LAYER),
                  "lm_head_emit": (1, D, VOCAB, 1), "lm_head_all": (T, D, VOCAB, 1)}
        for ty, (t_, k_, n_, mult) in shapes.items():
            oc = ms.matmul_opcount(t_, k_, n_)
            per[ty] = {"multiplicity": mult, "per_proof": oc,
                       "total": {k: v * mult for k, v in oc.items() if isinstance(v, int) and k not in
                                 ("T", "K", "N", "t", "nb", "kappa")}}
        keys = per["attn_c_attn"]["total"].keys()
        tot = {k: sum(per[ty]["total"][k] for ty in TYPES) for k in keys}
        rows[str(T)] = {"per_type": per, "total_prefill": tot,
                        "per_token": {k: v / T for k, v in tot.items()},
                        "weight_fold_elements": sum(per[ty]["per_proof"]["K"] * per[ty]["per_proof"]["N"] *
                                                    per[ty]["multiplicity"] for ty in
                                                    ("attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj",
                                                     "lm_head_emit")),
                        "activation_fold_elements": tot["fold_elements"] - sum(
                            per[ty]["per_proof"]["K"] * per[ty]["per_proof"]["N"] * per[ty]["multiplicity"]
                            for ty in ("attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj", "lm_head_emit"))}
    # one decode step (T = 1) with a KV cache of ctx keys: every weight is folded again per token
    dec = {}
    for ctx in Ts:
        shapes = {"attn_c_attn": (1, D, 3 * D, N_LAYER), "attn_qk": (1, DH, ctx, N_LAYER * H),
                  "attn_pv": (1, ctx, DH, N_LAYER * H), "attn_c_proj": (1, D, D, N_LAYER),
                  "mlp_c_fc": (1, D, FF, N_LAYER), "mlp_c_proj": (1, FF, D, N_LAYER),
                  "lm_head_emit": (1, D, VOCAB, 1)}
        tot = {}
        for ty, (t_, k_, n_, mult) in shapes.items():
            oc = ms.matmul_opcount(t_, k_, n_)
            for k, v in oc.items():
                if isinstance(v, int) and k not in ("T", "K", "N", "t", "nb", "kappa"):
                    tot[k] = tot.get(k, 0) + v * mult
        dec[str(ctx)] = tot
    return {"status": "ANALYTIC_COMPUTED", "evidence_class": "ANALYTIC", "label": ms.LABEL,
            "timestamp_utc": now(), "command": subprocess.list2cmdline([sys.executable] + sys.argv),
            "definitions": {
                "fold": "fold(v, r)[j] = v[2j] + r (v[2j+1] - v[2j]); ideal level-by-level count of one zero-padded "
                        "vector with m real entries: level 1 has ceil(m/2) base x ext folds (2 base 64x64 products), "
                        "each later level ceil(prev/2) ext x ext folds (3 products + mul-by-7)",
                "fold_elements": "T*K + K*N entries read by the two partial evaluations",
                "sumcheck": "SumFold over 2^kappa entries: 5(2^kappa - 1) Ext mults = 15(2^kappa - 1) base products "
                            "(results/raw/kernels/opcount.json)",
                "sha256_msgs": "hash invocations (short messages, 1 to 3 compressions each); digest_bytes = bytes "
                               "hashed for digest(A) and digest(C) per proof (int8 A, int32 C; PV's A is uint16 "
                               "(2 bytes) and attention W are activations, see notes)",
                "verifier": "C~(r1,r2), A~(r1,r3), W~(r3,r2) by eq-table contraction: T*N + T*K + K*N integer x Ext "
                            "multiply-adds (2 products each, lazily reduced) plus eq tables",
                "per_token": "total prefill / T",
                "decode_step": "one generated token with a KV cache of ctx keys (T = 1): every weight matrix is "
                               "folded again, so the prover cost per token approaches the parameter count"},
            "prefill": rows, "decode_step_by_ctx": dec,
            "note": "The implementation's streaming folds compute exactly these folds (plus zero-partner folds "
                    "inside a zero-padded tail chunk on the CPU fast path / GPU segments)."}

# ====================================================================== tests / ablation
def run_tests_record(command: str) -> Dict:
    """Run the bit-exactness / soundness test modules and record every test outcome (MEASURED)."""
    import re
    suites = [("tests.test_matmul_sumcheck", {}), ("tests.test_matmul_sumcheck_gpu", {"EDGECERT_GPU": "0"}),
              ("tests.test_matmul_sumcheck_gpu", {"EDGECERT_GPU": "1"})]
    rows = []
    for mod, extra in suites:
        env = dict(os.environ)
        env.update(extra)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, "-m", "unittest", mod, "-v"]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=3600)
        tests, cur = [], None
        for line in r.stderr.splitlines():
            m = re.match(r"^(test_\w+) \(", line)
            if m:
                cur = m.group(1)
            if " ... " in line and cur:
                tests.append({"test": cur, "outcome": line.rsplit(" ... ", 1)[1].strip()})
                cur = None
        dev = re.search(r"\[gpu tests\] device (\d+): (.+)", r.stderr)
        lines = r.stderr.splitlines()
        rows.append({"module": mod, "env": extra, "command": subprocess.list2cmdline(cmd), "returncode": r.returncode,
                     "wall_s": time.perf_counter() - t0, "device": dev.group(2).strip() if dev else None,
                     "tests": tests, "summary_lines": [x for x in lines if x.startswith("Ran ")][-1:]
                     + [x for x in lines if x.startswith(("OK", "FAILED"))][-1:]})
    n_ok = sum(1 for x in rows for t in x["tests"] if t["outcome"] == "ok")
    n_all = sum(len(x["tests"]) for x in rows)
    return {"status": "PASSED" if all(x["returncode"] == 0 for x in rows) and n_ok == n_all else "FAILED",
            "evidence_class": "MEASURED", "label": ms.LABEL, "record_type": "test_run", "command": command,
            "timestamp_utc": now(), "host": kc.host_info(), "code_sha256_16": code_digests(),
            "counts": {"tests": n_all, "ok": n_ok}, "suites": rows,
            "what": "bit-exactness of kernels/cpu/mle.c and kernels/cuda/mle.cu against the pure-Python golden "
                    "implementation (edgecert.sumfold / edgecert.field), sumcheck engines vs sumfold.prove and "
                    "sf_prove, and soundness smoke tests (tampered C, tampered W, tampered a_k, tampered messages, "
                    "wrong layer id => reject)"}

def ablation_record(args, command: str) -> Dict:
    """Same field value W~(r3, r2) of real GPT-2 weights by (a) the prover's fold path
    (mle_fold_axis: w_k = W~(k, r2), then sum_k eq(k, r3) w_k) and (b) the verifier's eq-table
    contraction (mlev_eq_table + mlev_bilinear, lazy 128-bit integer x Ext accumulation)."""
    from edgecert.field import emul, eadd
    lib = ms.get_lib()
    meta = prepare_model(args)
    Wq = load_weights(meta)
    cb = contention_snapshot()
    rng = np.random.default_rng(SEED + 7)

    def rext(k):
        return [(int(rng.integers(0, 2**63)) % ms.P, int(rng.integers(0, 2**63)) % ms.P) for _ in range(k)]

    cases = [("L0.attn_c_attn", Wq["L0_attn_c_attn"]), ("L0.mlp_c_fc", Wq["L0_mlp_c_fc"]),
             ("L0.mlp_c_proj", Wq["L0_mlp_c_proj"]), ("lm_head (wte^T view)", Wq["wte_q"].T)]
    rows = []
    nw, nr = 3, 10
    for name, W in cases:
        K, N = W.shape
        nb, kappa = ms.log2_ceil(N), ms.log2_ceil(K)
        r2, r3 = rext(nb), rext(kappa)
        eq3 = lib.eq_table(r3)
        for th in (1, 32):
            tf, te, same = [], [], []
            for it in range(nw + nr):
                t0 = time.perf_counter()
                w = lib.fold_axis(W.T, r2, nb, th)
                t1 = time.perf_counter()
                eq2 = lib.eq_table(r2)
                v = lib.bilinear(W, eq3, eq2, th)
                t2 = time.perf_counter()
                if it >= nw:
                    tf.append(t1 - t0)
                    te.append(t2 - t1)
                acc = (0, 0)
                for k in range(K):
                    acc = eadd(acc, emul((int(eq3[k, 0]), int(eq3[k, 1])), (int(w[k, 0]), int(w[k, 1]))))
                same.append(acc == v)
            rows.append({"matrix": name, "K": K, "N": N, "threads": th, "warmup": nw, "reps": nr,
                         "fold_path_s": kc.summarize(tf), "eq_contraction_s": kc.summarize(te),
                         "ratio_fold_over_eq_median": statistics.median(tf) / statistics.median(te),
                         "ns_per_weight_fold_median": statistics.median(tf) / (K * N) * 1e9,
                         "ns_per_weight_eq_median": statistics.median(te) / (K * N) * 1e9,
                         "same_value": all(same)})
    ca = contention_snapshot()
    return {"status": "MEASURED_SUCCESS" if all(r["same_value"] for r in rows) else "VERIFY_FAILED",
            "evidence_class": "MEASURED", "label": ms.LABEL, "record_type": "ablation", "command": command,
            "timestamp_utc": now(), "host": kc.host_info(), "dll_flags": lib.flags,
            "contention_before": cb, "contention_after": ca,
            "contended": bool(cb["cpu_contended"] or ca["cpu_contended"]),
            "what": "Evaluating an int8 weight matrix at a point: (a) repeated folds (prover path; first fold base x "
                    "ext, then ext x ext, reduction after every product) vs (b) eq-table contraction with lazy exact "
                    "128-bit accumulation of int8 x Ext products (verifier path). Same field value (checked). "
                    "Timing of (a) excludes the final K-term dot product; (b) includes building the eq table.",
            "rows": rows}

# ====================================================================== parent
def spawn_worker(args, T: int, backend: str, part: Path) -> Dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--T", str(T), "--backend", backend,
           "--warmup", str(args.warmup), "--reps", str(args.reps), "--verify-warmup", str(args.verify_warmup),
           "--verify-reps", str(args.verify_reps), "--part-out", str(part)]
    if args.cache_dir:
        cmd += ["--cache-dir", args.cache_dir]
    if args.golden_check and T == min(args.T):
        cmd += ["--golden-check"]
    if args.no_lm_all:
        cmd += ["--no-lm-all"]
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=args.worker_timeout)
    if r.returncode != 0 or not part.exists():
        return {"status": "RUN_FAILED", "evidence_class": None, "backend_config": backend, "T": T,
                "command": subprocess.list2cmdline(cmd), "returncode": r.returncode,
                "stderr_tail": r.stderr[-3000:], "stdout_tail": r.stdout[-1000:]}
    d = json.loads(part.read_text())
    d["command"] = subprocess.list2cmdline(cmd)
    d["spawn_wall_s"] = time.perf_counter() - t0
    return d

def compact(d: Dict) -> Dict:
    """Small summary row for summary.json."""
    if d.get("status") != "MEASURED_SUCCESS":
        return {"status": d.get("status")}
    tot = d["total_prefill"]
    row = {"status": d["status"], "contended": d["contended"],
           "device": d["backend"].get("device", {}).get("name") if d["backend"]["backend"] == "gpu" else
           f"CPU {d['backend']['threads']} thread(s)",
           "prover_wall_s_median": tot["wall_s"]["median"], "prover_core_s_median": tot["core_s"]["median"],
           "prover_bind_s_median": tot["bind_s"]["median"], "prover_fold_s_median": tot["fold_s"]["median"],
           "prover_sumcheck_s_median": tot["sumcheck_s"]["median"],
           "proof_bytes_total": tot["proof_bytes"], "wall_per_token_s": tot["wall_per_token_s_median"],
           "core_per_token_s": tot["core_per_token_s_median"],
           "per_type_wall_s_median": {k: v["wall_s"]["median"] for k, v in d["per_type"].items()},
           "per_type_core_s_median": {k: v["core_s"]["median"] for k, v in d["per_type"].items()},
           "peak_working_set_bytes": d["memory_peak"].get("peak_working_set_bytes"),
           "verifier_all_accept": d["verifier_all_accept"]}
    if d.get("verifier"):
        row["verifier_wall_s_median"] = d["verifier"]["total_prefill"]["v_wall_s"]["median"]
        row["verifier_threads"] = d["verifier"]["threads"]
    if "gpu_prover_scratch_peak_bytes" in d:
        row["gpu_prover_scratch_peak_bytes"] = d["gpu_prover_scratch_peak_bytes"]
        row["gpu_resident_bytes"] = d["backend"]["resident_total_bytes"]
    return row

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=lambda s: [int(x) for x in s.split(",")], default=[32, 128, 512])
    ap.add_argument("--backends", default="cpu1,cpu32,gpu0,gpu1",
                    help="cpu<threads>, gpu<nvidia-smi index> (comma separated)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--verify-warmup", type=int, default=3)
    ap.add_argument("--verify-reps", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="T=32 only, 1 warm-up, 2 reps, 1 verifier rep")
    ap.add_argument("--final", action="store_true",
                    help="idle-host rerun: 3 warm-ups + 10 reps everywhere (verifier too), *_final.json")
    ap.add_argument("--golden-check", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-lm-all", action="store_true")
    ap.add_argument("--opcount-only", action="store_true")
    ap.add_argument("--run-tests", action="store_true",
                    help="run tests/test_matmul_sumcheck*.py (GPU tests on nvidia-smi 0 and 1) -> tests.json")
    ap.add_argument("--ablation", action="store_true",
                    help="fold-based vs eq-table evaluation of W~(r3, r2) for real GPT-2 weights -> fold_vs_eq.json")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--worker-timeout", type=float, default=7200)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", default="", help=argparse.SUPPRESS)
    ap.add_argument("--part-out", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if args.worker:
        try:
            rec = run_worker(args)
        except Exception as e:
            rec = {"status": "RUN_FAILED", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:],
                   "backend_config": args.backend, "T": args.T[0]}
        Path(args.part_out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        return 0 if rec.get("status") == "MEASURED_SUCCESS" else 1

    if args.quick:
        args.T, args.warmup, args.reps, args.verify_warmup, args.verify_reps = [32], 1, 2, 0, 1
    if args.final:
        args.warmup, args.reps, args.verify_warmup, args.verify_reps = 3, 10, 3, 10
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_final" if args.final else ("_quick" if args.quick else "")
    command = subprocess.list2cmdline([sys.executable] + (argv if argv is not None else sys.argv))

    if args.run_tests:
        d = run_tests_record(command)
        (out_dir / f"tests{suffix}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(json.dumps({"tests": d["status"], "counts": d["counts"]}))
        return 0 if d["status"] == "PASSED" else 1
    if args.ablation:
        d = ablation_record(args, command)
        (out_dir / f"fold_vs_eq{suffix}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(json.dumps({"ablation": d["status"], "contended": d["contended"]}))
        return 0
    op = opcount_record(args.T if not args.quick else [32, 128, 512])
    (out_dir / f"opcount{suffix}.json").write_text(json.dumps(op, indent=1), encoding="utf-8")
    if args.opcount_only:
        print(json.dumps({"opcount": str(out_dir / f"opcount{suffix}.json")}))
        return 0

    t0 = time.perf_counter()
    build = ms.build_mle_lib()
    meta = prepare_model(args)
    header = {"label": ms.LABEL, "evidence_class": "MEASURED", "command": command, "timestamp_utc": now(),
              "host": kc.host_info(), "nvidia_smi": nvidia_smi_inventory(), "code_sha256_16": code_digests(),
              "dll_build": build, "model": {k: meta[k] for k in ("model_file", "model_sha256", "quant_rule",
                                                                  "quant_stats", "digests")},
              "final_run": bool(args.final), "params": {k: v for k, v in vars(args).items()
                                                        if k not in ("worker", "backend", "part_out")},
              "protocol": ms.__doc__.split("Backends")[0].strip(),
              "not_an_inference_certificate": "TRANSPARENT_REFERENCE: no ZK, no commitment, verifier non-succinct; "
                                              "linear layers only (no requant, softmax, norm, lookup or bias "
                                              "constraints)"}
    try:
        import cupy
        header["cupy"] = {"version": cupy.__version__, "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
                          "cuda_driver": cupy.cuda.runtime.driverGetVersion()}
    except Exception as e:
        header["cupy"] = {"error": str(e)[:300]}
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = {}
    Wq = load_weights(meta)
    for T in args.T:
        wl_info = save_workload(meta, T, Wq, not args.no_lm_all)
        recs = {}
        for be in backends:
            part = parts_dir / f"T{T}_{be}{suffix}.json"
            if part.exists():
                part.unlink()
            print(f"[gpt2prove] T={T} backend={be} ...", flush=True)
            d = spawn_worker(args, T, be, part)
            recs[be] = d
            print(f"[gpt2prove]   -> {d.get('status')} contended={d.get('contended')} "
                  f"wall={d.get('total_prefill', {}).get('wall_s', {}).get('median') if d.get('total_prefill') else None}",
                  flush=True)
        digs = {be: d.get("group_proof_digests") for be, d in recs.items() if d.get("status") == "MEASURED_SUCCESS"}
        ident = len({json.dumps(v, sort_keys=True) for v in digs.values()}) == 1 if digs else None
        statuses = [d.get("status") for d in recs.values()]
        rec = dict(header)
        rec.update({"status": "MEASURED_SUCCESS" if all(s == "MEASURED_SUCCESS" for s in statuses) else
                    ("PARTIAL_FAILURES" if any(s == "MEASURED_SUCCESS" for s in statuses) else "RUN_FAILED"),
                    "T": T, "contended": any(d.get("contended") for d in recs.values()),
                    "cross_backend_identical_proofs": ident, "backends_compared": sorted(digs),
                    "workload": {"activation_source": ACT_SOURCE, "bytes": wl_info["bytes"],
                                 "generated_by_parent_s": wl_info.get("generate_s"),
                                 "note": "generated once per T by the parent (untimed) and loaded by every worker"},
                    "configs": recs})
        (out_dir / f"gpt2_prove_T{T}{suffix}.json").write_text(json.dumps(rec, indent=1, default=str),
                                                               encoding="utf-8")
        results[T] = rec
    summ = {"label": ms.LABEL, "evidence_class": "MEASURED", "command": command, "timestamp_utc": now(),
            "final_run": bool(args.final), "contended": any(r["contended"] for r in results.values()),
            "note": "Medians over reps; see gpt2_prove_T*.json for stats, counts, contention snapshots. Timings on a "
                    "contended host are not final (rerun with --final on an idle host).",
            "rows": {str(T): {"cross_backend_identical_proofs": r["cross_backend_identical_proofs"],
                              "configs": {be: compact(d) for be, d in r["configs"].items()}}
                     for T, r in results.items()},
            "runner_wall_s": time.perf_counter() - t0}
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / f"summary{suffix}.json"),
                      "status": {T: r["status"] for T, r in results.items()}}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
