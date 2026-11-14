"""Real-weight GPT-2 124M and Qwen3-0.6B: float references and authorized integer derivatives.

* ``load_config`` / ``load_float_weights``: Hugging Face config.json + safetensors (via ``safetensors_np``) mapped to canonical names (row-major [out, in] linear weights). * ``FloatLM``: numpy float64 reference (GPT-2: learned positions, LayerNorm, GELU tanh, tied LM head; Qwen3: RMSNorm eps 1e-6, per-head q_norm/k_norm, RoPE theta 1e6, GQA, SiLU-gated MLP, tied embeddings).  Packed multi-sequence forward with a KV cache. * ``HFIntModel``: the authorized integer derivative of spec/numerical_semantics.md extended from the T0 families to the two real architectures (W8A8; W4A8 per-channel; W4A8 with group-wise int4 scales as a documented extension, see ``quantize_groups``). Every non-matmul operator is the exact int64 operator of ``intops``.  Matmuls use the *exact FP64 datapath* (``intops.matmul_fp64_exact``): IEEE binary64 GEMM is exact because the TileContract static bound certifies every partial sum <= 2^53 in magnitude; the result is asserted integral and spot-checked against int64 dot products.

Calibration (activation and residual scales) happens on prompt set A before commitment and is not part of the certified relation; the integer forward never touches floats except inside the certified-exact GEMM.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .intops import (I64, GadgetRecord, act_table, argmax_int, exp_table, gelu_tanh, matmul_fp64_exact,
                     norm_int, requant, rhaz_float, rope_int, rope_tables, silu, softmax_int, table_lookup,
                     ubits)
from .intref import DEFAULT_SHIFTS, IntPolicy, OpRecord, QuantError, _mask_record, _mult, quantize_rows
from .safetensors_np import SafeTensors

# ===================================================================== config
@dataclass
class HFConfig:
    name: str
    family: str                 # 'gpt2' | 'qwen3'
    model_dir: str
    n_layers: int
    d: int
    n_heads: int
    n_kv_heads: int
    dh: int
    ff: int
    vocab: int
    ctx: int                    # certified context (positions)
    norm_eps: float
    rope_base: Optional[float]
    tied: bool

    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def qdim(self) -> int:
        return self.n_heads * self.dh

    @property
    def kvdim(self) -> int:
        return self.n_kv_heads * self.dh

def load_config(model_dir, ctx: Optional[int] = None) -> HFConfig:
    d = Path(model_dir)
    cj = json.loads((d / "config.json").read_text(encoding="utf-8"))
    if cj["model_type"] == "gpt2":
        assert cj.get("activation_function") == "gelu_new"
        return HFConfig("gpt2_124m", "gpt2", str(d), cj["n_layer"], cj["n_embd"], cj["n_head"], cj["n_head"],
                        cj["n_embd"] // cj["n_head"], 4 * cj["n_embd"], cj["vocab_size"],
                        ctx or cj["n_positions"], cj["layer_norm_epsilon"], None, True)
    if cj["model_type"] == "qwen3":
        assert cj["hidden_act"] == "silu" and not cj.get("attention_bias") and not cj.get("use_sliding_window")
        # certified context: 32,768 (native context per the model card; config allows 40,960)
        return HFConfig("qwen3_0.6b", "qwen3", str(d), cj["num_hidden_layers"], cj["hidden_size"],
                        cj["num_attention_heads"], cj["num_key_value_heads"], cj["head_dim"],
                        cj["intermediate_size"], cj["vocab_size"], ctx or 32768, cj["rms_norm_eps"],
                        float(cj["rope_theta"]), bool(cj.get("tie_word_embeddings")))
    raise ValueError(f"unsupported model_type {cj['model_type']}")

def load_float_weights(cfg: HFConfig, dtype=np.float64) -> Dict[str, np.ndarray]:
    """Canonical float weights; linear weights as [out, in]."""
    st = SafeTensors(Path(cfg.model_dir) / "model.safetensors")
    f = lambda k: np.asarray(st[k], dtype=dtype)
    W: Dict[str, np.ndarray] = {}
    if cfg.family == "gpt2":
        W["emb"], W["pos"] = f("wte.weight"), f("wpe.weight")
        for l in range(cfg.n_layers):
            s, p = f"h.{l}.", f"L{l}."
            W[p + "n1_g"], W[p + "n1_b"] = f(s + "ln_1.weight"), f(s + "ln_1.bias")
            W[p + "wqkv"], W[p + "bqkv"] = f(s + "attn.c_attn.weight").T.copy(), f(s + "attn.c_attn.bias")
            W[p + "wo"], W[p + "bo"] = f(s + "attn.c_proj.weight").T.copy(), f(s + "attn.c_proj.bias")
            W[p + "n2_g"], W[p + "n2_b"] = f(s + "ln_2.weight"), f(s + "ln_2.bias")
            W[p + "w1"], W[p + "b1"] = f(s + "mlp.c_fc.weight").T.copy(), f(s + "mlp.c_fc.bias")
            W[p + "w2"], W[p + "b2"] = f(s + "mlp.c_proj.weight").T.copy(), f(s + "mlp.c_proj.bias")
        W["nf_g"], W["nf_b"] = f("ln_f.weight"), f("ln_f.bias")
    else:
        W["emb"] = f("model.embed_tokens.weight")
        if "lm_head.weight" in st:
            lm = f("lm_head.weight")
            if cfg.tied and not np.array_equal(lm, W["emb"]):
                raise ValueError("config says tied embeddings but lm_head.weight differs from embed_tokens")
            if not cfg.tied:
                W["lm"] = lm
        for l in range(cfg.n_layers):
            s, p = f"model.layers.{l}.", f"L{l}."
            W[p + "n1_g"] = f(s + "input_layernorm.weight")
            W[p + "wq"], W[p + "wk"] = f(s + "self_attn.q_proj.weight"), f(s + "self_attn.k_proj.weight")
            W[p + "wv"], W[p + "wo"] = f(s + "self_attn.v_proj.weight"), f(s + "self_attn.o_proj.weight")
            W[p + "qn_g"], W[p + "kn_g"] = f(s + "self_attn.q_norm.weight"), f(s + "self_attn.k_norm.weight")
            W[p + "n2_g"] = f(s + "post_attention_layernorm.weight")
            W[p + "wg"], W[p + "wu"] = f(s + "mlp.gate_proj.weight"), f(s + "mlp.up_proj.weight")
            W[p + "wd"] = f(s + "mlp.down_proj.weight")
        W["nf_g"] = f("model.norm.weight")
    return W

def weights_digest(cfg: HFConfig) -> str:
    """SHA-256 of the safetensors file (model identity recorded in results)."""
    h = hashlib.sha256()
    with open(Path(cfg.model_dir) / "model.safetensors", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

# ================================================================ KV buffer
class KVBuf:
    """Growable [n, heads, dh] buffer (capacity doubling)."""

    def __init__(self, dtype):
        self.buf: Optional[np.ndarray] = None
        self.n = 0
        self.dtype = dtype

    def append(self, x: np.ndarray) -> np.ndarray:
        need = self.n + x.shape[0]
        if self.buf is None or need > self.buf.shape[0]:
            cap = max(need, 64, 2 * (self.buf.shape[0] if self.buf is not None else 0))
            nb = np.empty((cap,) + x.shape[1:], dtype=self.dtype)
            if self.buf is not None:
                nb[:self.n] = self.buf[:self.n]
            self.buf = nb
        self.buf[self.n:need] = x
        self.n = need
        return self.buf[:self.n]

def new_state(cfg: HFConfig, dtype) -> dict:
    return {"n": 0, "k": [KVBuf(dtype) for _ in range(cfg.n_layers)],
            "v": [KVBuf(dtype) for _ in range(cfg.n_layers)]}

# ============================================================ float model
class Capture:
    """Calibration statistics per named tensor: max |x|, per-call |x| quantiles (the
    aggregate of a quantile is its maximum over calls, i.e. over prompts) and, for names
    ending in one of ``CHANNEL_KEYS``, the per-channel (last axis) max |x|."""
    QS = (0.5, 0.9, 0.99, 0.999, 0.9999)
    CHANNEL_KEYS = (".x1", ".x2", "xf", ".q_n", ".k_n", ".glu", ".v", ".a")

    KEEP_PREFIX_KEYS = (".k", ".v")   # the constant prefix K/V is quantized with these scales

    def __init__(self, quantiles: bool = True, skip_rows: int = 0):
        self.max: Dict[str, float] = {}
        self.q: Dict[str, List[np.ndarray]] = {}
        self.ch: Dict[str, np.ndarray] = {}
        self.quantiles = quantiles
        self.skip_rows = skip_rows      # rows (positions) of a fixed prefix that the integer model never computes

    def __call__(self, name: str, x: np.ndarray) -> None:
        if self.skip_rows and not name.endswith(self.KEEP_PREFIX_KEYS):
            x = x[self.skip_rows:]
        a = np.abs(x)
        m = float(a.max()) if a.size else 0.0
        self.max[name] = max(self.max.get(name, 0.0), m)
        if self.quantiles and a.size:
            self.q.setdefault(name, []).append(np.quantile(a.ravel(), self.QS))
        if a.size and name.endswith(self.CHANNEL_KEYS):
            c = a.reshape(-1, a.shape[-1]).max(axis=0)
            self.ch[name] = np.maximum(self.ch[name], c) if name in self.ch else c

    def summary(self) -> Dict[str, dict]:
        out = {}
        for k, m in self.max.items():
            d: Dict[str, object] = {"max": m}
            if k in self.q:
                qq = np.max(np.stack(self.q[k]), axis=0)
                d.update({f"q{q}": float(v) for q, v in zip(self.QS, qq)})
            if k in self.ch:
                d["channel_max"] = self.ch[k].tolist()
            out[k] = d
        return out

def _ln(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def _rms(x, g, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g

def _rope_f(x, pos, base):
    """Rotate-half RoPE (HF Llama/Qwen convention), x: [n, heads, dh], pos: [n]."""
    dh = x.shape[-1]
    h = dh // 2
    inv = base ** (-2.0 * np.arange(h, dtype=np.float64) / dh)
    ang = pos.astype(np.float64)[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :].astype(x.dtype), np.sin(ang)[:, None, :].astype(x.dtype)
    a, b = x[..., :h], x[..., h:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)

def _seq_meta(cfg: HFConfig, seqs):
    toks = np.concatenate([np.asarray(t, dtype=np.int64) for t, _ in seqs])
    pos = np.concatenate([st["n"] + np.arange(len(t), dtype=np.int64) for t, st in seqs])
    offs = np.cumsum([0] + [len(t) for t, _ in seqs])
    if toks.size and (toks.min() < 0 or toks.max() >= cfg.vocab):
        raise ValueError("token id outside vocabulary")
    if pos.size and pos.max() >= cfg.ctx:
        raise ValueError("sequence longer than the certified context")
    return toks, pos, offs

class FloatLM:
    def __init__(self, cfg: HFConfig, W: Dict[str, np.ndarray], dtype=np.float64):
        self.cfg, self.dtype = cfg, dtype
        self.W = {k: np.ascontiguousarray(v, dtype=dtype) for k, v in W.items()}
        # an explicit 'lm' entry means an untied head (e.g. after a smoothing transform)
        self.head = self.W["lm"] if "lm" in self.W else self.W["emb"]

    def new_state(self) -> dict:
        return new_state(self.cfg, self.dtype)

    def forward(self, seqs, emit: str = "last", capture: Optional[Callable] = None) -> List[np.ndarray]:
        """seqs: list of (new_tokens, state).  Returns logits per sequence ('last' row or 'all' rows)."""
        cfg, W = self.cfg, self.W
        cap = capture or (lambda k, v: None)
        toks, pos, offs = _seq_meta(cfg, seqs)
        H, Hkv, dh, rep = cfg.n_heads, cfg.n_kv_heads, cfg.dh, cfg.n_rep
        h = W["emb"][toks]
        if cfg.family == "gpt2":
            h = h + W["pos"][pos]
        cap("resid", h)
        for l in range(cfg.n_layers):
            p = f"L{l}."
            if cfg.family == "gpt2":
                x = _ln(h, W[p + "n1_g"], W[p + "n1_b"], cfg.norm_eps)
                cap(p + "x1", x)
                qkv = x @ W[p + "wqkv"].T + W[p + "bqkv"]
                d = cfg.d
                q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
                q, k, v = q.reshape(-1, H, dh), k.reshape(-1, Hkv, dh), v.reshape(-1, Hkv, dh)
            else:
                x = _rms(h, W[p + "n1_g"], cfg.norm_eps)
                cap(p + "x1", x)
                q = (x @ W[p + "wq"].T).reshape(-1, H, dh)
                k = (x @ W[p + "wk"].T).reshape(-1, Hkv, dh)
                v = (x @ W[p + "wv"].T).reshape(-1, Hkv, dh)
                cap(p + "q_pre", q)
                cap(p + "k_pre", k)
                q = _rms(q, W[p + "qn_g"], cfg.norm_eps)
                k = _rms(k, W[p + "kn_g"], cfg.norm_eps)
                cap(p + "q_n", q)
                cap(p + "k_n", k)
                q, k = _rope_f(q, pos, cfg.rope_base), _rope_f(k, pos, cfg.rope_base)
            cap(p + "q", q)
            cap(p + "k", k)
            cap(p + "v", v)
            a = np.empty((len(toks), H, dh), dtype=self.dtype)
            for i, (_, st) in enumerate(seqs):
                s0, s1 = offs[i], offs[i + 1]
                m = s1 - s0
                if m == 0:
                    continue
                K = st["k"][l].append(k[s0:s1])
                V = st["v"][l].append(v[s0:s1])
                npast = st["n"]
                qg = q[s0:s1].reshape(m, Hkv, rep, dh)
                sc = np.einsum("mgrd,ngd->grmn", qg, K) / math.sqrt(dh)
                allowed = np.arange(K.shape[0])[None, :] <= (npast + np.arange(m))[:, None]
                sc = np.where(allowed[None, None], sc, -np.inf)
                sc = sc - sc.max(-1, keepdims=True)
                e = np.exp(sc)
                att = e / e.sum(-1, keepdims=True)
                a[s0:s1] = np.einsum("grmn,ngd->mgrd", att, V).reshape(m, H, dh)
            a = a.reshape(len(toks), H * dh)
            cap(p + "a", a)
            o = a @ W[p + "wo"].T
            if cfg.family == "gpt2":
                o = o + W[p + "bo"]
            h = h + o
            cap("resid", h)
            if cfg.family == "gpt2":
                x2 = _ln(h, W[p + "n2_g"], W[p + "n2_b"], cfg.norm_eps)
                cap(p + "x2", x2)
                u = x2 @ W[p + "w1"].T + W[p + "b1"]
                cap(p + "pre_act", u)
                g = gelu_tanh(u).astype(self.dtype)
                cap(p + "act", g)
                h = h + g @ W[p + "w2"].T + W[p + "b2"]
            else:
                x2 = _rms(h, W[p + "n2_g"], cfg.norm_eps)
                cap(p + "x2", x2)
                gt = x2 @ W[p + "wg"].T
                cap(p + "pre_act", gt)
                sg = silu(gt).astype(self.dtype)
                up = x2 @ W[p + "wu"].T
                hh = sg * up
                cap(p + "act", sg)
                cap(p + "up", up)
                cap(p + "glu", hh)
                h = h + hh @ W[p + "wd"].T
            cap("resid", h)
            cap(f"resid.L{l}", h)
        for (t, st) in seqs:
            st["n"] += len(t)
        rows = np.arange(len(toks)) if emit == "all" else offs[1:] - 1
        hf = h[rows]
        xf = _ln(hf, W["nf_g"], W["nf_b"], cfg.norm_eps) if cfg.family == "gpt2" else _rms(hf, W["nf_g"], cfg.norm_eps)
        cap("xf", xf)
        logits = xf @ self.head.T
        if emit == "all":
            return [logits[offs[i]:offs[i + 1]] for i in range(len(seqs))]
        return [logits[i:i + 1] for i in range(len(seqs))]

# ======================================================= integer policy
@dataclass
class HFPolicy(IntPolicy):
    """IntPolicy + real-model options.

    variant: 'W8A8' (spec v1), 'W4A8' (spec v1, per-output-channel int4), 'W4A8g'
    (extension: group-wise int4 scales, group ``group_size`` along K, each group scale
    encoded as an integer multiplier m_g in [0, 2^group_mult_bits] relative to the row's
    largest group scale; the effective integer weight is W4 * m_g, |.| <= 7 * 2^t).
    resid_calib: 'max' (T0 rule: s_r = 1.1 max|resid| / (2^15 - 1)) or 'q<quantile>'
    (e.g. 'q0.9999': the largest per-prompt quantile of |resid|, times resid_margin;
    larger values saturate in the certified int16 clamp).
    act_calib: 'max' or 'q<quantile>' for the int8 activation scales (clamps certified).
    """
    variant: str = "W8A8"
    group_size: int = 128
    group_mult_bits: int = 8
    resid_calib: str = "max"
    resid_margin: float = 1.1
    act_calib: str = "max"
        # 'spec': the fixed spec shifts (the build fails if a multiplier needs > m_max bits); 'auto': per class, the largest shift s <= 16*ceil(s_spec/16) (same number of 16-bit limbs for the rq.rem range check) such that every multiplier of the class is < 2^m_max. Shifts are manifest constants either way; RoPE keeps its exact power-of-two multiplier.
    shift_mode: str = "auto"
        # 'int8': spec 3.3 (gamma int8 per tensor, one scalar multiplier M per norm); 'perchannel': gamma_q = sign(gamma) in {-1, 0, 1} (still int8) and a per-channel multiplier M_c = RHAZ(|gamma_c| / s_out * 2^(s - F)) on the last axis, i.e. the same rq gadget as a linear layer's per-output-channel requant; removes the int8 gamma rounding (real norm weights span up to 6 orders of magnitude).  beta_q = RHAZ(beta 2^F / |gamma|).
    norm_gamma: str = "int8"

    def __post_init__(self):
        if self.variant == "W4A8g":
            self.wbits = 4
            if self.shifts == DEFAULT_SHIFTS:
                self.shifts = dict(self.shifts)
                for k in ("lin", "res", "act"):
                    self.shifts[k] += self.group_mult_bits - 4
            return
        if self.variant == "W4A8":
            self.wbits = 4
        super().__post_init__()

    def label(self) -> str:
        return f"{self.variant}[resid={self.resid_calib},act={self.act_calib}]"

def quantize_groups(Wf: np.ndarray, qmax: int, gsize: int, tbits: int):
    """Group-wise symmetric int quantization along K with integer group multipliers.

    s_g[o, j] = max_{k in group j} |W[o, k]| / qmax,  Wq = clamp(RHAZ(W / s_g), -qmax, qmax),
    m[o, j] = RHAZ(s_g / max_j s_g * 2^t) in [0, 2^t].  Effective integer weight
    W'[o, k] = Wq[o, k] * m[o, k // gsize] with scale s_ref[o] / 2^t.
    """
    N, K = Wf.shape
    assert K % gsize == 0, "K must be a multiple of the group size"
    Wg = Wf.reshape(N, K // gsize, gsize)
    sg = np.abs(Wg).max(axis=2) / qmax
    sg_safe = np.where(sg == 0, 1.0, sg)
    Wq = np.clip(rhaz_float(Wg / sg_safe[:, :, None]), -qmax, qmax).astype(I64)
    sref = sg.max(axis=1)
    sref = np.where(sref == 0, 1.0, sref)
    m = rhaz_float(sg / sref[:, None] * (1 << tbits)).astype(I64)
    Weff = (Wq * m[:, :, None]).reshape(N, K)
    return Weff, sref / (1 << tbits), Wq.reshape(N, K), m

# ===================================================== FP64 exact backend
class Fp64Backend:
    """Exact FP64 GEMM backend: numpy (BLAS) or CuPy (cuBLAS DGEMM)."""

    def __init__(self, kind: str = "numpy", device: int = 0):
        self.kind = kind
        self.device = device
        self.cp = None
        if kind == "cupy":
            import cupy as cp  # noqa: WPS433 (optional dependency)
            self.cp = cp
            self.dev = cp.cuda.Device(device)
        self.impl_id = "cpu_fp64_exact" if kind == "numpy" else "gpu_fp64_exact"

    def weight(self, w_int: np.ndarray):
        wf = np.ascontiguousarray(w_int, dtype=np.float64)
        if self.cp is None:
            return wf
        with self.dev:
            return self.cp.asarray(wf)

    def gemm(self, xf: np.ndarray, wf) -> np.ndarray:
        if self.cp is None:
            return xf @ wf.T
        cp = self.cp
        with self.dev:
            return cp.asnumpy(cp.asarray(xf) @ wf.T)

    def device_name(self) -> str:
        if self.cp is None:
            return "cpu"
        props = self.cp.cuda.runtime.getDeviceProperties(self.device)
        n = props["name"]
        return n.decode() if isinstance(n, bytes) else str(n)

# ========================================================= integer model
def _cal_value(cal: Dict[str, dict], key: str, mode: str) -> float:
    e = cal[key]
    if mode == "max":
        return e["max"]
    if mode.startswith("q"):
        return e[mode] if mode in e else e["max"]
    raise ValueError(mode)

class HFIntModel:
    """Authorized integer derivative of a real HF model (see module docstring)."""

    def __init__(self, cfg: HFConfig, pol: HFPolicy, W: Dict[str, np.ndarray], cal: Dict[str, dict],
                 backend: Optional[Fp64Backend] = None, spot_checks: int = 2, upload: bool = True):
        self.cfg, self.cal = cfg, cal
        self.pol = pol = copy.deepcopy(pol)
        self.be = backend or Fp64Backend("numpy")
        self.spot_checks = spot_checks
        self.fp64_stats = {"gemms": 0, "spot_checks": 0, "max_bound": 0, "max_abs_acc": 0}
        self._qcache: Dict[str, tuple] = {}
        self.shift_selection: Dict[str, dict] = {}
        self.prefix: Optional[dict] = None
        self.prefix_info: Optional[dict] = None
        if pol.shift_mode == "auto":
            req: Dict[str, List[float]] = {}

            def collect(R, cls, what):
                R = np.asarray(R, dtype=np.float64)
                req.setdefault(cls, []).append(float(R.max()))
                return np.ones(R.shape, dtype=I64) if R.ndim else 1

            self._build(W, cal, collect)
            for cls, rs in req.items():
                cap = 16 * math.ceil(DEFAULT_SHIFTS[cls] / 16)
                Rmax = max(rs)
                s = cap
                while s > 1 and int(rhaz_float(Rmax * 2.0 ** s)) >= (1 << pol.m_max[cls]):
                    s -= 1
                self.shift_selection[cls] = {"policy_shift": pol.shifts[cls], "cap": cap, "chosen": s,
                                             "R_max": Rmax, "m_max_bits": pol.m_max[cls]}
                pol.shifts[cls] = s
        elif pol.shift_mode != "spec":
            raise ValueError(pol.shift_mode)
        self.mlog: Dict[str, dict] = {}
        self._build(W, cal, lambda R, cls, what: _mult(R, pol.shifts[cls], pol.m_max[cls], what, self.mlog))
        del self._qcache
        if upload:
            self.upload()

    # ----------------------------------------------------------- build
    def _build(self, W: Dict[str, np.ndarray], cal: Dict[str, dict], mult) -> None:
        cfg, pol = self.cfg, self.pol
        d, H, Hkv, dh, L = cfg.d, cfg.n_heads, cfg.n_kv_heads, cfg.dh, cfg.n_layers
        self.clip_log: Dict[str, int] = {}
        Bb = (1 << (pol.bias_bits - 1)) - 1
        res_top = _cal_value(cal, "resid", pol.resid_calib)
        self.s_r = res_top * pol.resid_margin / pol.res_hi
        self.eps_int = max(1, int(rhaz_float(cfg.norm_eps / self.s_r ** 2)))
        act = lambda key: _cal_value(cal, key, pol.act_calib) / 127.0
        P: Dict[str, object] = {}
        self.wq: Dict[str, dict] = {}       # quantized linear weights: int, fp64 (backend), scale, meta
        self.norm_info: Dict[str, dict] = {}

        def qlin(name, Wf):
            if name in self._qcache:
                Wi, s, meta = self._qcache[name]
            else:
                if pol.variant == "W4A8g":
                    Wi, s, W4, m = quantize_groups(Wf, 7, pol.group_size, pol.group_mult_bits)
                    meta = {"int4_range": [int(W4.min()), int(W4.max())], "group_mult_max": int(m.max()),
                            "group_mult_min": int(m.min())}
                else:
                    Wi, s = quantize_rows(Wf, pol.wmax)
                    meta = {}
                Wi = Wi.astype(np.int16)          # compact storage (|W| <= 1792); exact
                self._qcache[name] = (Wi, s, meta)
            self.wq[name] = {"int": Wi, "f64": None, "B_w": int(np.abs(Wi).max()), "scale": s, "meta": meta}
            return Wi, s

        def qbias(b, sx, sw, name):
            bq = rhaz_float(b / (sx * sw))
            self.clip_log[name] = int(np.sum(np.abs(bq) > Bb))
            return np.clip(bq, -Bb, Bb).astype(I64)

        def qnorm(g, b, s_out, name):
            if pol.norm_gamma == "perchannel":
                absg = np.abs(g)
                if b is not None and np.any((absg == 0) & (b != 0)):
                    raise QuantError(f"{name}: zero gamma with non-zero beta")
                safe = np.where(absg == 0, 1.0, absg)
                gq = np.sign(g).astype(I64)
                bq = rhaz_float(b * (1 << pol.F) / safe).astype(I64) if b is not None else np.zeros(len(g), I64)
                M = mult(safe / s_out * 2.0 ** -pol.F, "norm", name)
                geff = gq * (M if np.ndim(M) else np.full(len(g), M)) * s_out * 2.0 ** (pol.F - pol.shifts["norm"])
                s_g = None
            elif pol.norm_gamma == "int8":
                s_g = np.abs(g).max() / 127.0
                gq = np.clip(rhaz_float(g / s_g), -127, 127).astype(I64)
                bq = rhaz_float(b * (1 << pol.F) / s_g).astype(I64) if b is not None else np.zeros(len(g), I64)
                M = mult(s_g / s_out * 2.0 ** -pol.F, "norm", name)
                geff = gq * s_g
            else:
                raise ValueError(pol.norm_gamma)
            rel = np.abs(geff - g) / np.maximum(np.abs(g), 1e-30)
            self.norm_info[name] = {"mode": pol.norm_gamma, "s_gamma": s_g, "gamma_max": float(np.abs(g).max()),
                                    "gamma_min_abs": float(np.abs(g).min()),
                                    "gamma_rel_err_max": float(rel.max()),
                                    "gamma_rel_err_median": float(np.median(rel)),
                                    "gamma_zeroed": int(np.sum((gq == 0) & (g != 0)))}
            return {"gamma": gq, "beta": bq, "M": M, "s_g": s_g}

        def act_scale(key, fn):
            # the table input is clamped to [act_lo, act_hi] / 2^7, so larger outputs are unreachable
            dom = np.arange(pol.act_lo, pol.act_hi + 1, dtype=np.float64) / (1 << pol.act_in_frac)
            return min(_cal_value(cal, key, pol.act_calib), float(np.abs(fn(dom)).max())) / 127.0

        # embedding (int8 per row), also the tied LM head
        Eq, sE = quantize_rows(W["emb"], 127)
        P["E"], P["sE"] = Eq, sE
        P["M_E"] = mult(sE / self.s_r, "emb", "embed.tok")
        if cfg.family == "gpt2":
                        # per-row (per-position) int8 scales, like the token embedding; the position index is public, so the row and its multiplier are public constants (extension of spec 3.7, whose M_P is one scalar: GPT-2's position 0 row is 4.5x larger than the median row)
            P["Pos"], sPos = quantize_rows(W["pos"], 127)
            P["M_P"] = mult(sPos / self.s_r, "emb", "embed.pos")
        self.layers: List[dict] = []
        fa = 1 << pol.act_in_frac
        for l in range(L):
            p = f"L{l}."
            Ly: Dict[str, object] = {}
            sx1, sx2, sv, sa = act(p + "x1"), act(p + "x2"), act(p + "v"), act(p + "a")
            if cfg.family == "gpt2":
                sq, sk = act(p + "q"), act(p + "k")
                Ly["n1"] = qnorm(W[p + "n1_g"], W[p + "n1_b"], sx1, p + "ln1")
                Ly["n2"] = qnorm(W[p + "n2_g"], W[p + "n2_b"], sx2, p + "ln2")
                Wqkv_q, swqkv = qlin(p + "qkv", W[p + "wqkv"])
                s_out = np.concatenate([np.full(d, sq), np.full(d, sk), np.full(d, sv)])
                Ly["b_qkv"] = qbias(W[p + "bqkv"], sx1, swqkv, p + "qkv.bias")
            else:
                sq_pre, sk_pre = act(p + "q_pre"), act(p + "k_pre")
                # q, k after QK-norm and after RoPE share one scale (RoPE keeps the scale)
                sq = max(_cal_value(cal, p + "q", pol.act_calib), _cal_value(cal, p + "q_n", pol.act_calib)) / 127
                sk = max(_cal_value(cal, p + "k", pol.act_calib), _cal_value(cal, p + "k_n", pol.act_calib)) / 127
                Ly["n1"] = qnorm(W[p + "n1_g"], None, sx1, p + "ln1")
                Ly["n2"] = qnorm(W[p + "n2_g"], None, sx2, p + "ln2")
                Wqkv = np.concatenate([W[p + "wq"], W[p + "wk"], W[p + "wv"]], axis=0)
                Wqkv_q, swqkv = qlin(p + "qkv", Wqkv)
                s_out = np.concatenate([np.full(cfg.qdim, sq_pre), np.full(cfg.kvdim, sk_pre),
                                        np.full(cfg.kvdim, sv)])
                Ly["b_qkv"] = None
                qn = qnorm(W[p + "qn_g"], None, sq, p + "q_norm")
                kn = qnorm(W[p + "kn_g"], None, sk, p + "k_norm")
                Ly["qk_gamma"] = np.concatenate([np.tile(qn["gamma"], (H, 1)), np.tile(kn["gamma"], (Hkv, 1))])
                qM = np.broadcast_to(np.asarray(qn["M"], dtype=I64), (dh,))
                kM = np.broadcast_to(np.asarray(kn["M"], dtype=I64), (dh,))
                Ly["qk_M"] = np.concatenate([np.tile(qM, (H, 1)), np.tile(kM, (Hkv, 1))]).astype(I64)
                Ly["eps_qk"] = (max(1, int(rhaz_float(cfg.norm_eps / sq_pre ** 2))),
                                max(1, int(rhaz_float(cfg.norm_eps / sk_pre ** 2))))
                if Ly["eps_qk"][0] != Ly["eps_qk"][1]:
                    raise QuantError(f"{p}qk_norm: q and k need different eps_int {Ly['eps_qk']}")
                # RoPE keeps the exact power-of-two multiplier 2^(s - rope_frac) of the policy
                Ly["M_rope"] = _mult(2.0 ** -pol.rope_frac, pol.shifts["rope"], pol.m_max["rope"], p + "rope", {})
            Ly["Wqkv"] = Wqkv_q
            Ly["M_qkv"] = mult(sx1 * swqkv / s_out, "lin", p + "qkv")
            Ly["s_q"], Ly["s_k"], Ly["s_v"], Ly["s_a"] = sq, sk, sv, sa
            Ly["M_sc"] = mult(sq * sk * (1 << pol.T_score) / math.sqrt(dh), "score", p + "scores")
            Ly["M_pv"] = mult(2.0 ** -pol.Fp * sv / sa, "pv", p + "pv")
            Wo_q, swo = qlin(p + "o_proj", W[p + "wo"])
            Ly["Wo"] = Wo_q
            Ly["M_o"] = mult(sa * swo / self.s_r, "res", p + "o_proj")
            Ly["b_o"] = qbias(W[p + "bo"], sa, swo, p + "o_proj.bias") if cfg.family == "gpt2" else None
            if cfg.family == "gpt2":
                s_act = act_scale(p + "act", gelu_tanh)
                W1q, sw1 = qlin(p + "ff_up", W[p + "w1"])
                W2q, sw2 = qlin(p + "ff_down", W[p + "w2"])
                Ly.update({"W1": W1q, "W2": W2q,
                           "M_up": mult(sx2 * sw1 * fa, "act", p + "ff_up"),
                           "b_1": qbias(W[p + "b1"], sx2, sw1, p + "ff_up.bias"),
                           "T_act": act_table(gelu_tanh, pol.act_lo, pol.act_hi, pol.act_in_frac, s_act),
                           "M_dn": mult(s_act * sw2 / self.s_r, "res", p + "ff_down"),
                           "b_2": qbias(W[p + "b2"], s_act, sw2, p + "ff_down.bias")})
            else:
                s_act = act_scale(p + "act", silu)
                s_up, s_h = act(p + "up"), act(p + "glu")
                Wgq, swg = qlin(p + "ff_gate", W[p + "wg"])
                Wuq, swu = qlin(p + "ff_up", W[p + "wu"])
                Wdq, swd = qlin(p + "ff_down", W[p + "wd"])
                Ly.update({"Wg": Wgq, "Wu": Wuq, "Wd": Wdq,
                           "M_gate": mult(sx2 * swg * fa, "act", p + "ff_gate"),
                           "T_act": act_table(silu, pol.act_lo, pol.act_hi, pol.act_in_frac, s_act),
                           "M_up": mult(sx2 * swu / s_up, "lin", p + "ff_up"),
                           "M_glu": mult(s_act * s_up / s_h, "glu", p + "glu"),
                           "M_dn": mult(s_h * swd / self.s_r, "res", p + "ff_down")})
            Ly["s_act"] = s_act
            self.layers.append(Ly)
        sxf = act("xf")
        P["nf"] = qnorm(W["nf_g"], W.get("nf_b"), sxf, "lnf")
        if "lm" in W:                                   # untied head (e.g. after smoothing)
            P["Head"], sH = quantize_rows(W["lm"], 127)
            P["head_tied"] = False
        else:
            P["Head"], sH = Eq, sE
            P["head_tied"] = True
        P["sH"] = sH
        P["M_v"] = np.maximum(rhaz_float(sH / sH.max() * (1 << pol.logit_m)), 1).astype(I64)
        P["s_xf"] = sxf
        self.P = P
        self.T_exp = exp_table(pol.T_score, pol.U_max, pol.E_bits)
        if cfg.family == "qwen3":
            self.C, self.S = rope_tables(cfg.ctx, dh, cfg.rope_base, pol.rope_frac)

    def upload(self) -> None:
        """Place FP64 copies of the committed integer weights on the backend (numpy or GPU)."""
        for w in self.wq.values():
            w["f64"] = self.be.weight(w["int"])
        self.P["Head_f64"] = self.be.weight(self.P["Head"])

    # --------------------------------------------------------- helpers
    def table_digests(self) -> Dict[str, str]:
        h = lambda a: hashlib.sha256(np.ascontiguousarray(a, dtype=I64).tobytes()).hexdigest()
        out = {"T_exp": h(self.T_exp)}
        for l, Ly in enumerate(self.layers):
            out[f"L{l}.T_act"] = h(Ly["T_act"])
        return out

    def weight_digest(self) -> str:
        """SHA-256 over all committed integer weights (the integer model identity)."""
        hsh = hashlib.sha256()
        for k in sorted(self.wq):
            hsh.update(k.encode())
            hsh.update(np.ascontiguousarray(self.wq[k]["int"], dtype=I64).tobytes())
        hsh.update(np.ascontiguousarray(self.P["E"], dtype=I64).tobytes())
        return hsh.hexdigest()

    def new_state(self) -> dict:
        st = new_state(self.cfg, I64)
        if self.prefix is not None:
            for l in range(self.cfg.n_layers):
                st["k"][l].append(self.prefix["k"][l])
                st["v"][l].append(self.prefix["v"][l])
            st["n"] = len(self.prefix["tokens"])
        return st

    def set_prefix(self, tokens: Sequence[int], kv_float: Dict[str, List[np.ndarray]]) -> dict:
        """Commit a fixed prompt prefix (e.g. BOS) as constant int8 K/V per layer.

        The prefix K/V are quantized from the float model with the layer's k/v scales
        (post-RoPE k for Qwen3) and become public constants of the manifest; the integer
        forward then starts at position len(tokens) and never computes the prefix rows
        (this keeps the attention-sink token's massive activations out of the certified
        integer residual stream).  Returns clamp statistics.
        """
        info = {"tokens": [int(t) for t in tokens], "clamped_k": 0, "clamped_v": 0, "elements": 0}
        ks, vs = [], []
        for l, Ly in enumerate(self.layers):
            kq = rhaz_float(kv_float["k"][l] / Ly["s_k"])
            vq = rhaz_float(kv_float["v"][l] / Ly["s_v"])
            info["clamped_k"] += int(np.sum((kq < -128) | (kq > 127)))
            info["clamped_v"] += int(np.sum((vq < -128) | (vq > 127)))
            info["elements"] += int(kq.size + vq.size)
            ks.append(np.clip(kq, -128, 127).astype(I64))
            vs.append(np.clip(vq, -128, 127).astype(I64))
        self.prefix = {"tokens": info["tokens"], "k": ks, "v": vs}
        h = hashlib.sha256()
        for a in ks + vs:
            h.update(np.ascontiguousarray(a, dtype=I64).tobytes())
        info["sha256"] = h.hexdigest()
        self.prefix_info = info
        return info

    def _mm(self, x, name: str, b=None, tag="mm"):
        wq = self.wq[name] if name in self.wq else None
        if wq is None:  # LM head
            w_int, wf, B_w = self.P["Head"], self.P["Head_f64"], 127
        else:
            w_int, wf, B_w = wq["int"], wq["f64"], wq["B_w"]
        acc, rec = matmul_fp64_exact(x, w_int, wf, b=b, B_x=128, B_w=B_w, gemm=self.be.gemm,
                                     impl_id=self.be.impl_id, spot_checks=self.spot_checks, tag=tag)
        st = self.fp64_stats
        st["gemms"] += 1
        st["spot_checks"] += rec.params["spot_checks"]
        st["max_bound"] = max(st["max_bound"], rec.params["static_bound"])
        if acc.size:
            st["max_abs_acc"] = max(st["max_abs_acc"], int(np.abs(acc).max()))
        return acc, rec

    def _norm_q(self, r, prm, kind):
        pol = self.pol
        g, gn = norm_int(r, kind, pol.F, pol.G, self.eps_int, tag="nm")
        x, gq = requant(g * prm["gamma"] + prm["beta"], prm["M"], pol.shifts["norm"], -128, 127)
        return x, gn, gq

    @staticmethod
    def _batched_exact(a: np.ndarray, b: np.ndarray, bound: int) -> np.ndarray:
        """Exact integer batched matmul a @ b via float64 (bound certified <= 2^53)."""
        if bound >= (1 << 53):
            raise OverflowError("attention matmul bound exceeds 2^53 - 1: exact FP64 datapath not certified")
        out = np.matmul(a.astype(np.float64), b.astype(np.float64))
        ri = np.rint(out)
        if not np.array_equal(out, ri):
            raise ArithmeticError("FP64 datapath produced a non-integer (bound violated)")
        return ri.astype(I64)

    # --------------------------------------------------------- forward
    def forward(self, seqs, emit: str = "last", sink: Optional[Callable[[OpRecord], None]] = None):
        """Exact integer forward over packed sequences with a KV cache.

        seqs: list of (new_tokens, state).  emit: 'last' (one emitting row per sequence) or
        'all' (every new position).  sink(OpRecord) receives every op record (dynamic range
        checks, counting); it may raise ContractViolation to ABORT.
        Returns per sequence (l_int [rows, V] int64, deq_logits [rows, V] float64, tau [rows]).
        """
        cfg, pol, P = self.cfg, self.pol, self.P
        sh = pol.shifts
        toks, pos, offs = _seq_meta(cfg, seqs)
        n = len(toks)
        H, Hkv, dh, rep = cfg.n_heads, cfg.n_kv_heads, cfg.dh, cfg.n_rep
        rlo, rhi = pol.res_lo, pol.res_hi

        def rec(op_id, kind, *gadgets):
            if sink is not None:
                sink(OpRecord(op_id, kind, list(gadgets)))

        # ---- embedding gather + requant into the int16 residual stream
        E = P["E"][toks]
        v = E * P["M_E"][toks][:, None]
        g_emb = GadgetRecord("embed", "emb", committed={"row": E, "mult": P["M_E"][toks]}, lookups={"T_emb": n})
        if cfg.family == "gpt2":
            Pr = P["Pos"][pos]
            v = v + Pr * P["M_P"][pos][:, None]
            g_emb.committed["pos_row"] = Pr
            g_emb.lookups["T_pos"] = n
        r, g_rq = requant(v, 1, sh["emb"], rlo, rhi, tag="rq")
        rec("embed", "embed", g_emb, g_rq)
        kind = "ln" if cfg.family == "gpt2" else "rms"
        for l, Ly in enumerate(self.layers):
            p = f"L{l}."
            x, g1, g2 = self._norm_q(r, Ly["n1"], kind)
            rec(p + "ln1", "norm", g1, g2)
            acc, gm = self._mm(x, p + "qkv", Ly["b_qkv"])
            qkv, gr = requant(acc, Ly["M_qkv"], sh["lin"], -128, 127)
            rec(p + "qkv", "linear", gm, gr)
            qd, kd = cfg.qdim, cfg.kvdim
            q = qkv[:, :qd].reshape(n, H, dh)
            k = qkv[:, qd:qd + kd].reshape(n, Hkv, dh)
            vv = qkv[:, qd + kd:].reshape(n, Hkv, dh)
            if cfg.family == "qwen3":
                # per-head RMS QK-norm: one norm row per (token, head), rows = n * (H + Hkv)
                qk = np.concatenate([q, k], axis=1).reshape(n * (H + Hkv), dh)
                g, gn = norm_int(qk, "rms", pol.F, pol.G, Ly["eps_qk"][0], tag="nm")
                qk, gq = requant(g * np.tile(Ly["qk_gamma"], (n, 1)), np.tile(Ly["qk_M"], (n, 1)), sh["norm"],
                                 -128, 127)
                rec(p + "qk_norm", "norm", gn, gq)
                qk = qk.reshape(n, H + Hkv, dh)
                q, k = qk[:, :H], qk[:, H:]
                Cp, Sp = self.C[pos], self.S[pos]
                q, g_rq_q = rope_int(q, Cp, Sp, Ly["M_rope"], sh["rope"])
                rec(p + "rope_q", "rope", g_rq_q)
                k, g_rq_k = rope_int(k, Cp, Sp, Ly["M_rope"], sh["rope"])
                rec(p + "rope_k", "rope", g_rq_k)
            a = np.empty((n, H, dh), dtype=I64)
            for i, (_, st) in enumerate(seqs):
                s0, s1 = int(offs[i]), int(offs[i + 1])
                m = s1 - s0
                if m == 0:
                    continue
                K = st["k"][l].append(k[s0:s1])
                V = st["v"][l].append(vv[s0:s1])
                nk = K.shape[0]
                npast = st["n"]
                # scores S[h, i, j] = q[i, h] . k[j, h // rep]  (GQA), exact
                qg = q[s0:s1].reshape(m, Hkv, rep, dh).transpose(1, 2, 0, 3).reshape(Hkv, rep * m, dh)
                Kt = K.transpose(1, 2, 0)                                   # [Hkv, dh, nk]
                S = self._batched_exact(qg, Kt, dh * 128 * 128).reshape(Hkv, rep, m, nk).reshape(H, m, nk)
                allowed = np.broadcast_to(np.arange(nk)[None, :] <= (npast + np.arange(m))[:, None], (H, m, nk))
                g_sm_mm = GadgetRecord("mm", "mm", committed={"acc": S[allowed]}, params={"K": dh})
                t, g_t = requant(S, Ly["M_sc"], sh["score"], rlo, rhi)
                if sink is not None:
                    g_t = _mask_record(g_t, allowed)
                rec(p + "scores", "scores", g_sm_mm, g_t)
                Pm, g_sm = softmax_int(t, allowed, self.T_exp, pol.Fp, tag="sm")
                rec(p + "softmax", "softmax", g_sm)
                Pg = Pm.reshape(Hkv, rep * m, nk)
                Vt = V.transpose(1, 0, 2)                                   # [Hkv, nk, dh]
                acc = self._batched_exact(Pg, Vt, nk * (1 << pol.Fp) * 128)
                acc = acc.reshape(Hkv, rep, m, dh).transpose(2, 0, 1, 3).reshape(m, H, dh)
                ai, g_pv = requant(acc, Ly["M_pv"], sh["pv"], -128, 127)
                rec(p + "pv", "pv", GadgetRecord("mm", "mm", committed={"acc": acc}, params={"K": nk}), g_pv)
                a[s0:s1] = ai
            a = a.reshape(n, H * dh)
            acc, gm = self._mm(a, p + "o_proj", Ly["b_o"])
            r, gr = requant(acc, Ly["M_o"], sh["res"], rlo, rhi, add=r)
            rec(p + "o_proj", "linear", gm, gr)
            x2, g1, g2 = self._norm_q(r, Ly["n2"], kind)
            rec(p + "ln2", "norm", g1, g2)
            if cfg.family == "gpt2":
                acc, gm = self._mm(x2, p + "ff_up", Ly["b_1"])
                ain, gr = requant(acc, Ly["M_up"], sh["act"], pol.act_lo, pol.act_hi)
                rec(p + "ff_up", "linear", gm, gr)
                hh, ga = table_lookup(ain, Ly["T_act"], pol.act_lo, "T_act", tag="act")
                rec(p + "act", "act", ga)
                acc, gm = self._mm(hh, p + "ff_down", Ly["b_2"])
            else:
                acc, gm = self._mm(x2, p + "ff_gate")
                ain, gr = requant(acc, Ly["M_gate"], sh["act"], pol.act_lo, pol.act_hi)
                rec(p + "ff_gate", "linear", gm, gr)
                sg, ga = table_lookup(ain, Ly["T_act"], pol.act_lo, "T_act", tag="act")
                rec(p + "act", "act", ga)
                acc, gm = self._mm(x2, p + "ff_up")
                up, gr = requant(acc, Ly["M_up"], sh["lin"], -128, 127)
                rec(p + "ff_up", "linear", gm, gr)
                hh, gg = requant(sg * up, Ly["M_glu"], sh["glu"], -128, 127)
                rec(p + "glu", "glu", gg)
                acc, gm = self._mm(hh, p + "ff_down")
            r, gr = requant(acc, Ly["M_dn"], sh["res"], rlo, rhi, add=r)
            rec(p + "ff_down", "linear", gm, gr)
        for (t_, st) in seqs:
            st["n"] += len(t_)
        rows = np.arange(n) if emit == "all" else offs[1:] - 1
        xf, g1, g2 = self._norm_q(r[rows], P["nf"], kind)
        rec("lnf", "norm", g1, g2)
        acc, gm = self._mm(xf, "lm_head")
        rec("lm_head", "lm_head", gm)
        lint = acc * P["M_v"][None, :]
        tau, gam = argmax_int(lint, tag="am")
        rec("argmax", "argmax", gam)
        deq = acc.astype(np.float64) * P["s_xf"] * P["sH"][None, :]
        if emit == "all":
            return [(lint[offs[i]:offs[i + 1]], deq[offs[i]:offs[i + 1]], tau[offs[i]:offs[i + 1]])
                    for i in range(len(seqs))]
        return [(lint[i:i + 1], deq[i:i + 1], tau[i:i + 1]) for i in range(len(seqs))]

# ================================================================ helpers
def calibrate(fm: FloatLM, prompts: Sequence[Sequence[int]], quantiles: bool = True,
              prefix: Sequence[int] = ()) -> Dict[str, dict]:
    """Activation statistics of the float model on ``prefix + prompt`` for every prompt;
    rows of the fixed prefix are excluded except for k and v (see Capture)."""
    cap = Capture(quantiles=quantiles, skip_rows=len(prefix))
    for pr in prompts:
        fm.forward([(np.asarray(list(prefix) + list(pr)), fm.new_state())], emit="all", capture=cap)
    return cap.summary()

def prefix_kv_float(fm: FloatLM, prefix: Sequence[int]) -> Dict[str, List[np.ndarray]]:
    """Float K (post-RoPE for Qwen3) and V of a fixed prefix, per layer."""
    st = fm.new_state()
    fm.forward([(np.asarray(prefix), st)], emit="last")
    return {"k": [st["k"][l].buf[:len(prefix)].copy() for l in range(fm.cfg.n_layers)],
            "v": [st["v"][l].buf[:len(prefix)].copy() for l in range(fm.cfg.n_layers)]}

def dequantized_float_weights(im: HFIntModel, W: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Float weights with every linear weight (and the embedding / head) replaced by its
    dequantized integer value; used to separate weight-quantization error from the rest."""
    cfg = im.cfg
    Wd = dict(W)
    for l in range(cfg.n_layers):
        p = f"L{l}."
        deq = lambda name: im.wq[p + name]["int"] * im.wq[p + name]["scale"][:, None]
        if cfg.family == "gpt2":
            Wd[p + "wqkv"], Wd[p + "wo"] = deq("qkv"), deq("o_proj")
            Wd[p + "w1"], Wd[p + "w2"] = deq("ff_up"), deq("ff_down")
        else:
            wqkv = deq("qkv")
            Wd[p + "wq"], Wd[p + "wk"], Wd[p + "wv"] = (wqkv[:cfg.qdim], wqkv[cfg.qdim:cfg.qdim + cfg.kvdim],
                                                        wqkv[cfg.qdim + cfg.kvdim:])
            Wd[p + "wo"], Wd[p + "wg"] = deq("o_proj"), deq("ff_gate")
            Wd[p + "wu"], Wd[p + "wd"] = deq("ff_up"), deq("ff_down")
    Wd["emb"] = im.P["E"] * im.P["sE"][:, None]
    if not im.P["head_tied"]:
        Wd["lm"] = im.P["Head"] * im.P["sH"][:, None]
    return Wd

# ============================================ pre-commitment equivalent transforms
def smooth_weights(cfg: HFConfig, W: Dict[str, np.ndarray], cal: Dict[str, dict], alpha: float = 0.5,
                   qk_balance: bool = True, glu_smooth: bool = True, head: bool = True
                   ) -> Tuple[Dict[str, np.ndarray], dict]:
    """SmoothQuant-style *function-preserving* float transforms applied before quantization.

    In exact real arithmetic the transformed float model computes the same function, so the
    integer derivative built from it is still an instance of the spec semantics (same
    gadgets, same TileContract rules); only the committed integer weights and scales differ.

    * norm -> linear: x' = x / s (norm gamma and beta divided by s per channel), following
      weight columns multiplied by s, with s_c = a_c^alpha / w_c^(1 - alpha) (a_c: calibrated
      channel max of the norm output, w_c: max |W[:, c]|).  Pairs: ln1 -> qkv, ln2 -> ff_up
      (GPT-2) or gate/up (Qwen3), and, if ``head``, lnf -> LM head (the head becomes an
      untied int8 table; the input embedding is unchanged).
    * Qwen3 QK balance: q_norm gamma * t, k_norm gamma / t with t equal on each RoPE rotation
      pair (i, i + dh/2), so RoPE commutes and every score q.k is unchanged.
    * Qwen3 GLU -> down: up_proj rows / s, down_proj columns * s (the gate product is linear
      in up).
    """
    Ws = {k: v.copy() for k, v in W.items()}
    info: Dict[str, object] = {"alpha": alpha, "pairs": []}

    def factors(a, w):
        a = np.maximum(np.asarray(a, dtype=np.float64), 1e-6)
        w = np.maximum(np.asarray(w, dtype=np.float64), 1e-6)
        s = a ** alpha / w ** (1 - alpha)
        return np.clip(s, 1e-4, 1e4)

    def norm_pair(key_cal, g, b, weights, name):
        a = np.asarray(cal[key_cal]["channel_max"])
        wmax = np.max(np.stack([np.abs(Ws[k]).max(axis=0) for k in weights]), axis=0)
        s = factors(a, wmax)
        Ws[g] = Ws[g] / s
        if b is not None and b in Ws:
            Ws[b] = Ws[b] / s
        for k in weights:
            Ws[k] = Ws[k] * s[None, :]
        info["pairs"].append({"pair": name, "s_min": float(s.min()), "s_max": float(s.max()),
                              "act_ratio_before": float(a.max() / np.median(a)),
                              "act_ratio_after": float((a / s).max() / np.median(a / s))})

    for l in range(cfg.n_layers):
        p = f"L{l}."
        if cfg.family == "gpt2":
            norm_pair(p + "x1", p + "n1_g", p + "n1_b", [p + "wqkv"], p + "ln1->qkv")
            norm_pair(p + "x2", p + "n2_g", p + "n2_b", [p + "w1"], p + "ln2->ff_up")
        else:
            norm_pair(p + "x1", p + "n1_g", None, [p + "wq", p + "wk", p + "wv"], p + "ln1->qkv")
            norm_pair(p + "x2", p + "n2_g", None, [p + "wg", p + "wu"], p + "ln2->gate,up")
            if qk_balance:
                h = cfg.dh // 2
                qa = np.asarray(cal[p + "q_n"]["channel_max"])
                ka = np.asarray(cal[p + "k_n"]["channel_max"])
                qp = np.maximum(np.maximum(qa[:h], qa[h:]), 1e-6)
                kp = np.maximum(np.maximum(ka[:h], ka[h:]), 1e-6)
                t = np.sqrt(kp / qp)
                t = np.concatenate([t, t])
                Ws[p + "qn_g"] = Ws[p + "qn_g"] * t
                Ws[p + "kn_g"] = Ws[p + "kn_g"] / t
                info["pairs"].append({"pair": p + "q_norm<->k_norm", "t_min": float(t.min()),
                                      "t_max": float(t.max())})
            if glu_smooth:
                a = np.asarray(cal[p + "glu"]["channel_max"])
                s = factors(a, np.abs(Ws[p + "wd"]).max(axis=0))
                Ws[p + "wu"] = Ws[p + "wu"] / s[:, None]
                Ws[p + "wd"] = Ws[p + "wd"] * s[None, :]
                info["pairs"].append({"pair": p + "glu->down", "s_min": float(s.min()), "s_max": float(s.max())})
    if head:
        a = np.asarray(cal["xf"]["channel_max"])
        base = Ws["lm"] if "lm" in Ws else Ws["emb"]
        s = factors(a, np.abs(base).max(axis=0))
        Ws["nf_g"] = Ws["nf_g"] / s
        if "nf_b" in Ws:
            Ws["nf_b"] = Ws["nf_b"] / s
        Ws["lm"] = base * s[None, :]
        info["pairs"].append({"pair": "lnf->lm_head (untied)", "s_min": float(s.min()), "s_max": float(s.max())})
    return Ws, info
