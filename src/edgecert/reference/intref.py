"""EDGE-CERT exact integer reference (W8A8 / W4A8 authorized integer derivative).

* Re-exports the exact operators of ``intops`` (spec/numerical_semantics.md).
* Tiny decoder-only model **T0** in two families that together cover every
  operator of the spec:
    - ``gpt``   : LayerNorm, learned positions, GELU(tanh) table, biases, tied embedding
    - ``llama`` : RMSNorm, integer RoPE, SiLU-gated MLP (SwiGLU), untied LM head
  default size: 2 layers, d=64, 4 heads, ff=256, vocab=256, ctx=64.
* Deterministic synthetic float weights (no downloads), float64 reference of the
  same architecture, calibration on a prompt set disjoint from evaluation,
  quantization into the authorized integer derivative, exact int64 forward with
  a full witness trace, greedy decoding with lowest-index tie break, and quality
  metrics of the integer derivative against float.

Everything proof-critical (the integer forward) is exact int64; floats are used only to generate synthetic weights, to calibrate scales and to build committed tables, all of which happen before the model is committed.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .intops import (I64, GadgetRecord, act_table, argmax_int, as_i64, exp_table, gelu_tanh,  # noqa: F401
                     isqrt64, matmul, norm_int, requant, rhaz_float, rhaz_int, rhaz_shift,
                     rope_int, rope_tables, silu, softmax_int, table_lookup, ubits)

# ------------------------------------------------------------------ configs
DEFAULT_SHIFTS = {"lin": 24, "res": 20, "act": 24, "score": 24, "pv": 30, "norm": 40,
                  "rope": 30, "glu": 20, "emb": 16}
DEFAULT_MMAX = {"lin": 20, "res": 22, "act": 22, "score": 24, "pv": 20, "norm": 20,
                "rope": 17, "glu": 22, "emb": 24}

@dataclass
class T0Config:
    name: str = "t0_gpt"
    family: str = "gpt"            # 'gpt' | 'llama'
    n_layers: int = 2
    d: int = 64
    n_heads: int = 4
    ff: int = 256
    vocab: int = 256
    ctx: int = 64
    seed: int = 20260927
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def dh(self) -> int:
        return self.d // self.n_heads

@dataclass
class IntPolicy:
    """Authorized-derivative quantizer policy (all values are manifest constants)."""
    wbits: int = 8                  # 8 -> W8A8 (weights in [-127,127]); 4 -> W4A8 ([-7,7])
    F: int = 24                     # norm reciprocal fraction bits
    G: int = 8                      # norm sqrt fraction bits
    T_score: int = 6                # softmax logit resolution 2^-T
    U_max: int = 1023               # exp-table domain [0, U_max]
    E_bits: int = 16                # exp table scale 2^E_bits
    Fp: int = 15                    # probability scale 2^Fp
    act_in_bits: int = 12           # activation-table input width (signed)
    act_in_frac: int = 7            # activation-table input resolution 2^-7
    rope_frac: int = 14             # RoPE table scale 2^14
    res_bits: int = 16              # residual stream int16
    logit_m: int = 16               # per-channel logit multiplier bits
    bias_bits: int = 24             # int32 bias certified range |b| <= 2^23 - 1
    shifts: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SHIFTS))
    m_max: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MMAX))

    def __post_init__(self):
                # int4 weight scales are ~2^4 larger (127/7): lower the shift of every requant whose input scale contains a weight scale so that multipliers keep the same width.
        if self.wbits == 4 and self.shifts == DEFAULT_SHIFTS:
            self.shifts = dict(self.shifts)
            for k in ("lin", "res", "act"):
                self.shifts[k] -= 4

    @property
    def wmax(self) -> int:
        return 127 if self.wbits == 8 else (1 << (self.wbits - 1)) - 1

    @property
    def res_lo(self) -> int:
        return -(1 << (self.res_bits - 1))

    @property
    def res_hi(self) -> int:
        return (1 << (self.res_bits - 1)) - 1

    @property
    def act_lo(self) -> int:
        return -(1 << (self.act_in_bits - 1))

    @property
    def act_hi(self) -> int:
        return (1 << (self.act_in_bits - 1)) - 1

def T0_GPT(**kw) -> T0Config:
    return T0Config(name="t0_gpt", family="gpt", **kw)

def T0_LLAMA(**kw) -> T0Config:
    return T0Config(name="t0_llama", family="llama", **kw)

# ------------------------------------------------------------ float weights
def synth_weights(cfg: T0Config) -> Dict[str, np.ndarray]:
    """Deterministic synthetic float64 weights (seeded; no downloads)."""
    rng = np.random.default_rng(cfg.seed + (0 if cfg.family == "gpt" else 1))
    d, ff, V, L = cfg.d, cfg.ff, cfg.vocab, cfg.n_layers
    N = lambda *s, std=1.0: rng.standard_normal(s) * std
    W: Dict[str, np.ndarray] = {}
    W["emb"] = N(V, d)
    if cfg.family == "gpt":
        W["pos"] = N(cfg.ctx, d, std=0.3)
    for l in range(L):
        p = f"L{l}."
        W[p + "n1_g"] = 1.0 + N(d, std=0.1)
        W[p + "n2_g"] = 1.0 + N(d, std=0.1)
        W[p + "wq"] = N(d, d, std=1.5 / math.sqrt(d))
        W[p + "wk"] = N(d, d, std=1.5 / math.sqrt(d))
        W[p + "wv"] = N(d, d, std=1.0 / math.sqrt(d))
        W[p + "wo"] = N(d, d, std=1.0 / math.sqrt(d))
        if cfg.family == "gpt":
            W[p + "n1_b"] = N(d, std=0.05)
            W[p + "n2_b"] = N(d, std=0.05)
            W[p + "bqkv"] = N(3 * d, std=0.02)
            W[p + "bo"] = N(d, std=0.02)
            W[p + "w1"] = N(ff, d, std=1.0 / math.sqrt(d))
            W[p + "b1"] = N(ff, std=0.02)
            W[p + "w2"] = N(d, ff, std=1.0 / math.sqrt(ff))
            W[p + "b2"] = N(d, std=0.02)
        else:
            W[p + "wg"] = N(ff, d, std=1.0 / math.sqrt(d))
            W[p + "wu"] = N(ff, d, std=1.0 / math.sqrt(d))
            W[p + "wd"] = N(d, ff, std=1.0 / math.sqrt(ff))
    W["nf_g"] = 1.0 + N(d, std=0.1)
    if cfg.family == "gpt":
        W["nf_b"] = N(d, std=0.05)
    else:
        W["lm"] = N(V, d)
    return W

# ---------------------------------------------------------- float forward
def _fnorm(x, g, b, eps, kind):
    if kind == "ln":
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        y = (x - mu) / np.sqrt(var + eps)
    else:
        y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    y = y * g
    return y + b if b is not None else y

def _frope(q, pos, base):
    dh = q.shape[-1]
    h = dh // 2
    i = np.arange(h, dtype=np.float64)
    theta = base ** (-2.0 * i / dh)
    ang = pos[:, None] * theta[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = q[..., :h], q[..., h:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)

def float_forward(cfg: T0Config, W: Dict[str, np.ndarray], tokens: Sequence[int],
                  capture: Optional[Dict[str, List[np.ndarray]]] = None) -> np.ndarray:
    """float64 reference of the same architecture; returns logits [n, V]."""
    tokens = np.asarray(tokens)
    n, d, H, dh = len(tokens), cfg.d, cfg.n_heads, cfg.dh
    kind = "ln" if cfg.family == "gpt" else "rms"
    cap = (lambda k, v: capture.setdefault(k, []).append(np.abs(v).max())) if capture is not None \
        else (lambda k, v: None)
    h = W["emb"][tokens].copy()
    if cfg.family == "gpt":
        h = h + W["pos"][:n]
    cap("resid", h)
    mask = np.tril(np.ones((n, n), dtype=bool))
    pos = np.arange(n, dtype=np.float64)
    for l in range(cfg.n_layers):
        p = f"L{l}."
        x = _fnorm(h, W[p + "n1_g"], W.get(p + "n1_b"), cfg.norm_eps, kind)
        cap(p + "x1", x)
        q, k, v = x @ W[p + "wq"].T, x @ W[p + "wk"].T, x @ W[p + "wv"].T
        if cfg.family == "gpt":
            bq = W[p + "bqkv"]
            q, k, v = q + bq[:d], k + bq[d:2 * d], v + bq[2 * d:]
        q, k, v = q.reshape(n, H, dh), k.reshape(n, H, dh), v.reshape(n, H, dh)
        if cfg.family == "llama":
            cap(p + "q_pre", q)
            cap(p + "k_pre", k)
            q, k = _frope(q, pos, cfg.rope_base), _frope(k, pos, cfg.rope_base)
        cap(p + "q", q)
        cap(p + "k", k)
        cap(p + "v", v)
        s = np.einsum("ihd,jhd->hij", q, k) / math.sqrt(dh)
        s = np.where(mask[None], s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        att = e / e.sum(-1, keepdims=True)
        a = np.einsum("hij,jhd->ihd", att, v).reshape(n, d)
        cap(p + "a", a)
        o = a @ W[p + "wo"].T
        if cfg.family == "gpt":
            o = o + W[p + "bo"]
        h = h + o
        cap("resid", h)
        x2 = _fnorm(h, W[p + "n2_g"], W.get(p + "n2_b"), cfg.norm_eps, kind)
        cap(p + "x2", x2)
        if cfg.family == "gpt":
            u = x2 @ W[p + "w1"].T + W[p + "b1"]
            g = gelu_tanh(u)
            cap(p + "act", g)
            h = h + g @ W[p + "w2"].T + W[p + "b2"]
        else:
            gt = x2 @ W[p + "wg"].T
            sg = silu(gt)
            up = x2 @ W[p + "wu"].T
            hh = sg * up
            cap(p + "act", sg)
            cap(p + "up", up)
            cap(p + "glu", hh)
            h = h + hh @ W[p + "wd"].T
        cap("resid", h)
    xf = _fnorm(h, W["nf_g"], W.get("nf_b"), cfg.norm_eps, kind)
    cap("xf", xf)
    head = W["emb"] if cfg.family == "gpt" else W["lm"]
    return xf @ head.T

# ------------------------------------------------------------- calibration
def random_prompts(cfg: T0Config, count: int, length: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, cfg.vocab, size=length) for _ in range(count)]

def calibrate(cfg: T0Config, W, prompts) -> Dict[str, float]:
    cap: Dict[str, List[float]] = {}
    for pr in prompts:
        float_forward(cfg, W, pr, capture=cap)
    return {k: float(max(v)) for k, v in cap.items()}

# ------------------------------------------------------------ quantization
def quantize_rows(Wf: np.ndarray, qmax: int):
    """Symmetric per-output-channel (per-row) quantization: W ~= Wq * s[:, None]."""
    s = np.abs(Wf).max(axis=1) / qmax
    s = np.where(s == 0, 1.0, s)
    Wq = np.clip(rhaz_float(Wf / s[:, None]), -qmax, qmax)
    return Wq.astype(I64), s

class QuantError(ValueError):
    pass

def _mult(R, s: int, m_max: int, what: str, log: Dict[str, dict]):
    R = np.asarray(R, dtype=np.float64)
    M = rhaz_float(R * (1 << s))
    M = np.maximum(M, 1)
    if int(M.max()) >= (1 << m_max):
        raise QuantError(f"{what}: multiplier {int(M.max())} needs more than {m_max} bits "
                         f"(policy shift {s} too large for ratio {float(R.max()):.3g})")
    log[what] = {"s": s, "M_min": int(M.min()), "M_max": int(M.max()),
                 "M_min_bits": ubits(int(M.min())), "M_max_bits": ubits(int(M.max()))}
    return M.astype(I64) if M.ndim else int(M)

class IntModel:
    """The authorized integer derivative of a T0 float model."""

    def __init__(self, cfg: T0Config, pol: IntPolicy, W: Dict[str, np.ndarray], cal: Dict[str, float]):
        self.cfg, self.pol, self.Wf, self.cal = cfg, pol, W, cal
        d, H, dh, L = cfg.d, cfg.n_heads, cfg.dh, cfg.n_layers
        sh, mm = pol.shifts, pol.m_max
        self.mlog: Dict[str, dict] = {}
        self.clip_log: Dict[str, int] = {}
        wq = pol.wmax
        Bb = (1 << (pol.bias_bits - 1)) - 1
        self.s_r = cal["resid"] * 1.1 / pol.res_hi
        self.eps_int = max(1, int(rhaz_float(cfg.norm_eps / self.s_r ** 2)))
        s_ = lambda key: cal[key] / 127.0
        P: Dict[str, object] = {}

        def qbias(b, sx, sw, name):
            bq = rhaz_float(b / (sx * sw))
            self.clip_log[name] = int(np.sum(np.abs(bq) > Bb))
            return np.clip(bq, -Bb, Bb).astype(I64)

        def qnorm(prefix, s_in_x, name):
            g = W[prefix + "_g"]
            s_g = np.abs(g).max() / 127.0
            gq = np.clip(rhaz_float(g / s_g), -127, 127).astype(I64)
            b = W.get(prefix + "_b")
            bq = rhaz_float(b * (1 << pol.F) / s_g).astype(I64) if b is not None else np.zeros(d, I64)
            M = _mult(s_g / s_in_x * 2.0 ** -pol.F, sh["norm"], mm["norm"], name, self.mlog)
            return {"gamma": gq, "beta": bq, "M": M}

        # embeddings (always int8, per-row scales; also the tied LM head for gpt)
        Eq, sE = quantize_rows(W["emb"], 127)
        P["E"], P["sE"] = Eq, sE
        P["M_E"] = _mult(sE / self.s_r, sh["emb"], mm["emb"], "embed.tok", self.mlog)
        if cfg.family == "gpt":
            sP = np.abs(W["pos"]).max() / 127.0
            P["Pos"] = np.clip(rhaz_float(W["pos"] / sP), -127, 127).astype(I64)
            P["M_P"] = _mult(sP / self.s_r, sh["emb"], mm["emb"], "embed.pos", self.mlog)
        self.layers = []
        for l in range(L):
            p = f"L{l}."
            Ly: Dict[str, object] = {}
            sx1, sx2 = s_(p + "x1"), s_(p + "x2")
            sq, sk, sv, sa = s_(p + "q"), s_(p + "k"), s_(p + "v"), s_(p + "a")
            if cfg.family == "llama":
                sq = max(cal[p + "q"], cal[p + "q_pre"]) / 127.0
                sk = max(cal[p + "k"], cal[p + "k_pre"]) / 127.0
            Ly["n1"] = qnorm(p + "n1", sx1, p + "ln1")
            Ly["n2"] = qnorm(p + "n2", sx2, p + "ln2")
            Wqkv = np.concatenate([W[p + "wq"], W[p + "wk"], W[p + "wv"]], axis=0)
            Wqkv_q, swqkv = quantize_rows(Wqkv, wq)
            s_out = np.concatenate([np.full(d, sq), np.full(d, sk), np.full(d, sv)])
            Ly["Wqkv"] = Wqkv_q
            Ly["M_qkv"] = _mult(sx1 * swqkv / s_out, sh["lin"], mm["lin"], p + "qkv", self.mlog)
            Ly["b_qkv"] = qbias(W[p + "bqkv"], sx1, swqkv, p + "qkv.bias") if cfg.family == "gpt" else None
            if cfg.family == "llama":
                Ly["M_rope"] = _mult(2.0 ** -pol.rope_frac, sh["rope"], mm["rope"], p + "rope", self.mlog)
            Ly["M_sc"] = _mult(sq * sk * (1 << pol.T_score) / math.sqrt(dh), sh["score"], mm["score"],
                               p + "scores", self.mlog)
            Ly["M_pv"] = _mult(2.0 ** -pol.Fp * sv / sa, sh["pv"], mm["pv"], p + "pv", self.mlog)
            Wo_q, swo = quantize_rows(W[p + "wo"], wq)
            Ly["Wo"] = Wo_q
            Ly["M_o"] = _mult(sa * swo / self.s_r, sh["res"], mm["res"], p + "o_proj", self.mlog)
            Ly["b_o"] = qbias(W[p + "bo"], sa, swo, p + "o_proj.bias") if cfg.family == "gpt" else None
            fa = 1 << pol.act_in_frac
            if cfg.family == "gpt":
                W1q, sw1 = quantize_rows(W[p + "w1"], wq)
                W2q, sw2 = quantize_rows(W[p + "w2"], wq)
                s_act = s_(p + "act")
                Ly.update({"W1": W1q, "W2": W2q,
                           "M_up": _mult(sx2 * sw1 * fa, sh["act"], mm["act"], p + "ff_up", self.mlog),
                           "b_1": qbias(W[p + "b1"], sx2, sw1, p + "ff_up.bias"),
                           "T_act": act_table(gelu_tanh, pol.act_lo, pol.act_hi, pol.act_in_frac, s_act),
                           "M_dn": _mult(s_act * sw2 / self.s_r, sh["res"], mm["res"], p + "ff_down",
                                         self.mlog),
                           "b_2": qbias(W[p + "b2"], s_act, sw2, p + "ff_down.bias")})
            else:
                Wgq, swg = quantize_rows(W[p + "wg"], wq)
                Wuq, swu = quantize_rows(W[p + "wu"], wq)
                Wdq, swd = quantize_rows(W[p + "wd"], wq)
                s_act, s_up, s_h = s_(p + "act"), s_(p + "up"), s_(p + "glu")
                Ly.update({"Wg": Wgq, "Wu": Wuq, "Wd": Wdq,
                           "M_gate": _mult(sx2 * swg * fa, sh["act"], mm["act"], p + "ff_gate", self.mlog),
                           "T_act": act_table(silu, pol.act_lo, pol.act_hi, pol.act_in_frac, s_act),
                           "M_up": _mult(sx2 * swu / s_up, sh["lin"], mm["lin"], p + "ff_up", self.mlog),
                           "M_glu": _mult(s_act * s_up / s_h, sh["glu"], mm["glu"], p + "glu", self.mlog),
                           "M_dn": _mult(s_h * swd / self.s_r, sh["res"], mm["res"], p + "ff_down",
                                         self.mlog)})
            self.layers.append(Ly)
        sxf = s_("xf")
        P["nf"] = qnorm("nf", sxf, "lnf")
        if cfg.family == "gpt":
            P["Head"], sH = Eq, sE
        else:
            P["Head"], sH = quantize_rows(W["lm"], 127)
        P["sH"] = sH
        P["M_v"] = np.maximum(rhaz_float(sH / sH.max() * (1 << pol.logit_m)), 1).astype(I64)
        P["s_xf"] = sxf
        self.P = P
        self.T_exp = exp_table(pol.T_score, pol.U_max, pol.E_bits)
        if cfg.family == "llama":
            self.C, self.S = rope_tables(cfg.ctx, dh, cfg.rope_base, pol.rope_frac)

    # ----------------------------------------------------------- digests
    def table_digests(self) -> Dict[str, str]:
        h = lambda a: hashlib.sha256(np.ascontiguousarray(a, dtype=I64).tobytes()).hexdigest()
        out = {"T_exp": h(self.T_exp)}
        for l, Ly in enumerate(self.layers):
            out[f"L{l}.T_act"] = h(Ly["T_act"])
        return out

    # ---------------------------------------------------------- forward
    def forward(self, tokens: Sequence[int], record: bool = False,
                emit: Optional[Sequence[int]] = None):
        """Exact integer forward.

        Returns (l_int [rows, V] int64 logits-for-argmax, trace or None, deq_logits [rows, V]).
        With record=True the LM head / final norm / argmax are evaluated only on
        the emitting rows ``emit`` (default: the last position), exactly as the
        certified relation does; otherwise on every position.
        """
        cfg, pol, P = self.cfg, self.pol, self.P
        sh = pol.shifts
        tokens = np.asarray(tokens, dtype=I64)
        n, d, H, dh = len(tokens), cfg.d, cfg.n_heads, cfg.dh
        if n > cfg.ctx:
            raise ValueError("sequence longer than the certified context")
        if np.any(tokens < 0) or np.any(tokens >= cfg.vocab):
            raise ValueError("token id outside vocabulary")
        tr: Optional[List[OpRecord]] = [] if record else None

        def rec(op_id, kind, *gadgets):
            if tr is not None:
                tr.append(OpRecord(op_id, kind, list(gadgets)))

        # ---- embedding gather + requant into the int16 residual stream
        E = P["E"][tokens]
        v = E * P["M_E"][tokens][:, None]
        g_emb = GadgetRecord("embed", "emb", committed={"row": E, "mult": P["M_E"][tokens]},
                             lookups={"T_emb": n})
        if cfg.family == "gpt":
            Pr = P["Pos"][:n]
            v = v + Pr * P["M_P"]
            g_emb.committed["pos_row"] = Pr
            g_emb.lookups["T_pos"] = n
        r, g_rq = requant(v, 1, sh["emb"], pol.res_lo, pol.res_hi, tag="rq")
        rec("embed", "embed", g_emb, g_rq)
        causal = np.tril(np.ones((n, n), dtype=bool))
        allowed = np.broadcast_to(causal, (H, n, n))
        kind = "ln" if cfg.family == "gpt" else "rms"
        for l, Ly in enumerate(self.layers):
            p = f"L{l}."
            x, g1, g2 = self._norm_q(r, Ly["n1"], kind)
            rec(p + "ln1", "norm", g1, g2)
            acc, gm = matmul(x, Ly["Wqkv"], Ly["b_qkv"])
            qkv, gr = requant(acc, Ly["M_qkv"], sh["lin"], -128, 127)
            rec(p + "qkv", "linear", gm, gr)
            q = qkv[:, :d].reshape(n, H, dh)
            k = qkv[:, d:2 * d].reshape(n, H, dh)
            vv = qkv[:, 2 * d:].reshape(n, H, dh)
            if cfg.family == "llama":
                q, gq = rope_int(q, self.C[:n], self.S[:n], Ly["M_rope"], sh["rope"])
                rec(p + "rope_q", "rope", gq)
                k, gk = rope_int(k, self.C[:n], self.S[:n], Ly["M_rope"], sh["rope"])
                rec(p + "rope_k", "rope", gk)
            S = np.einsum("ihd,jhd->hij", q, k)
            g_sm_mm = GadgetRecord("mm", "mm", committed={"acc": S[allowed]}, params={"K": dh})
            t, g_t = requant(S, Ly["M_sc"], sh["score"], pol.res_lo, pol.res_hi)
            if tr is not None:
                g_t = _mask_record(g_t, allowed)
            rec(p + "scores", "scores", g_sm_mm, g_t)
            Pm, g_sm = softmax_int(t, allowed, self.T_exp, pol.Fp, tag="sm")
            rec(p + "softmax", "softmax", g_sm)
            acc = np.einsum("hij,jhd->ihd", Pm, vv)
            a, g_pv = requant(acc, Ly["M_pv"], sh["pv"], -128, 127)
            rec(p + "pv", "pv", GadgetRecord("mm", "mm", committed={"acc": acc}, params={"K": n}), g_pv)
            a = a.reshape(n, d)
            acc, gm = matmul(a, Ly["Wo"], Ly["b_o"])
            r, gr = requant(acc, Ly["M_o"], sh["res"], pol.res_lo, pol.res_hi, add=r)
            rec(p + "o_proj", "linear", gm, gr)
            x2, g1, g2 = self._norm_q(r, Ly["n2"], kind)
            rec(p + "ln2", "norm", g1, g2)
            if cfg.family == "gpt":
                acc, gm = matmul(x2, Ly["W1"], Ly["b_1"])
                ain, gr = requant(acc, Ly["M_up"], sh["act"], pol.act_lo, pol.act_hi)
                rec(p + "ff_up", "linear", gm, gr)
                hh, ga = table_lookup(ain, Ly["T_act"], pol.act_lo, "T_act", tag="act")
                rec(p + "act", "act", ga)
                acc, gm = matmul(hh, Ly["W2"], Ly["b_2"])
            else:
                acc, gm = matmul(x2, Ly["Wg"])
                ain, gr = requant(acc, Ly["M_gate"], sh["act"], pol.act_lo, pol.act_hi)
                rec(p + "ff_gate", "linear", gm, gr)
                sg, ga = table_lookup(ain, Ly["T_act"], pol.act_lo, "T_act", tag="act")
                rec(p + "act", "act", ga)
                acc, gm = matmul(x2, Ly["Wu"])
                up, gr = requant(acc, Ly["M_up"], sh["lin"], -128, 127)
                rec(p + "ff_up", "linear", gm, gr)
                hh, gg = requant(sg * up, Ly["M_glu"], sh["glu"], -128, 127)
                rec(p + "glu", "glu", gg)
                acc, gm = matmul(hh, Ly["Wd"])
            r, gr = requant(acc, Ly["M_dn"], sh["res"], pol.res_lo, pol.res_hi, add=r)
            rec(p + "ff_down", "linear", gm, gr)
        rows = (np.asarray(emit, dtype=I64) if emit is not None else np.array([n - 1], dtype=I64)) \
            if record else np.arange(n)
        xf, g1, g2 = self._norm_q(r[rows], P["nf"], kind)
        rec("lnf", "norm", g1, g2)
        acc, gm = matmul(xf, P["Head"])
        rec("lm_head", "lm_head", gm)
        lint = acc * P["M_v"][None, :]
        tau, gam = argmax_int(lint, tag="am")
        rec("argmax", "argmax", gam)
        deq = acc.astype(np.float64) * P["s_xf"] * P["sH"][None, :]
        return lint, tr, deq

    def _norm_q(self, r, prm, kind):
        pol = self.pol
        g, gn = norm_int(r, kind, pol.F, pol.G, self.eps_int, tag="nm")
        x, gq = requant(g * prm["gamma"] + prm["beta"], prm["M"], pol.shifts["norm"], -128, 127)
        return x, gn, gq

    # ------------------------------------------------------------ decoding
    def greedy(self, prompt: Sequence[int], steps: int) -> List[int]:
        seq = list(int(t) for t in prompt)
        for _ in range(steps):
            lint, _, _ = self.forward(seq)
            seq.append(int(np.argmax(lint[-1])))   # lowest-index tie break
        return seq[len(prompt):]

def _mask_record(g: GadgetRecord, mask: np.ndarray) -> GadgetRecord:
    sel = lambda dct: {k: (v[mask] if getattr(v, "shape", None) == mask.shape else v) for k, v in dct.items()}
    return GadgetRecord(g.kind, g.tag, sel(g.committed), sel(g.checks), dict(g.lookups), dict(g.params),
                        sel(g.aux))

@dataclass
class OpRecord:
    op_id: str
    kind: str
    gadgets: List[GadgetRecord]

def greedy_float(cfg: T0Config, W, prompt, steps: int) -> List[int]:
    seq = list(int(t) for t in prompt)
    for _ in range(steps):
        seq.append(int(np.argmax(float_forward(cfg, W, seq)[-1])))
    return seq[len(prompt):]

def dequantized_weights(im: IntModel) -> Dict[str, np.ndarray]:
    """Float weights with every linear weight replaced by its dequantized int value."""
    W = {k: v.copy() for k, v in im.Wf.items()}
    wq = im.pol.wmax
    names = ["wq", "wk", "wv", "wo"] + (["w1", "w2"] if im.cfg.family == "gpt" else ["wg", "wu", "wd"])
    for l in range(im.cfg.n_layers):
        for nm in names:
            q, s = quantize_rows(W[f"L{l}.{nm}"], wq)
            W[f"L{l}.{nm}"] = q * s[:, None]
    q, s = quantize_rows(W["emb"], 127)
    W["emb"] = q * s[:, None]
    if im.cfg.family == "llama":
        q, s = quantize_rows(W["lm"], 127)
        W["lm"] = q * s[:, None]
    return W

# ---------------------------------------------------------- witness manifest
def _dtype_for(a: np.ndarray) -> str:
    if a.size == 0:
        return "int8"
    lo, hi = int(a.min()), int(a.max())
    if lo >= 0 and hi <= 1:
        return "bool"
    for nm, b in (("int8", 8), ("int16", 16), ("int32", 32)):
        if -(1 << (b - 1)) <= lo and hi < (1 << (b - 1)):
            return nm
    return "int64"

_DT_BYTES = {"bool": 1, "int8": 1, "int16": 2, "int32": 4, "int64": 8}

def witness_manifest(trace: List[OpRecord]) -> List[dict]:
    """Per committed witness tensor: layer/op, name, shape, minimal dtype, bytes."""
    out = []
    for op in trace:
        layer = int(op.op_id[1:op.op_id.index(".")]) if op.op_id.startswith("L") else None
        for g in op.gadgets:
            for nm, a in g.committed.items():
                a = np.asarray(a)
                dt = _dtype_for(a)
                out.append({"op_id": op.op_id, "layer": layer, "gadget": g.tag, "name": nm,
                            "shape": list(a.shape), "dtype": dt, "elements": int(a.size),
                            "bytes_native": int(a.size) * _DT_BYTES[dt],
                            "bytes_goldilocks": 8 * int(a.size), "bytes_254bit": 32 * int(a.size)})
    return out

def trace_counts(trace: List[OpRecord]) -> Dict[str, dict]:
    """Element counts per op / gadget witness / check / table (for the compiler cross-check)."""
    out: Dict[str, dict] = {}
    for op in trace:
        c = out.setdefault(op.op_id, {"committed": {}, "checks": {}, "lookups": {}})
        for g in op.gadgets:
            for nm, a in g.committed.items():
                c["committed"][f"{g.tag}.{nm}"] = c["committed"].get(f"{g.tag}.{nm}", 0) + int(np.size(a))
            for nm, a in g.checks.items():
                c["checks"][f"{g.tag}.{nm}"] = c["checks"].get(f"{g.tag}.{nm}", 0) + int(np.size(a))
            for t, k in g.lookups.items():
                c["lookups"][t] = c["lookups"].get(t, 0) + int(k)
    return out

def trace_maxima(trace: List[OpRecord]) -> Dict[str, Dict[str, int]]:
    """Observed max |value| per op / gadget check and committed tensor."""
    out: Dict[str, Dict[str, int]] = {}
    for op in trace:
        o = out.setdefault(op.op_id, {})
        for g in op.gadgets:
            for nm, a in list(g.committed.items()) + [("chk." + k, v) for k, v in g.checks.items()]:
                a = np.asarray(a)
                if a.size:
                    key = f"{g.tag}.{nm}"
                    o[key] = max(o.get(key, 0), int(np.abs(a).max()))
    return out

# ---------------------------------------------------------------- quality
def build_t0(cfg: T0Config, pol: Optional[IntPolicy] = None, calib_prompts: int = 32,
             calib_seed: int = 1001) -> IntModel:
    pol = pol or IntPolicy()
    W = synth_weights(cfg)
    cal = calibrate(cfg, W, random_prompts(cfg, calib_prompts, cfg.ctx, calib_seed))
    return IntModel(cfg, pol, W, cal)

def quality_metrics(im: IntModel, n_prompts: int = 64, prompt_len: int = 16, gen_len: int = 16,
                    eval_seed: int = 2002) -> dict:
    """Integer derivative vs float64 reference (and vs float with dequantized weights).

    * greedy: both models decode gen_len tokens from the same prompt independently;
    * teacher-forced: on the float greedy sequence, per-position top-1 agreement
      and logit errors (integer logits dequantized with s_xf * s_head).
    """
    cfg = im.cfg
    Wdq = dequantized_weights(im)
    prompts = random_prompts(cfg, n_prompts, prompt_len, eval_seed)
    exact_seq = tok_match = tok_total = 0
    first_div = []
    tf_agree = tf_total = 0
    tf_agree_dq = 0
    dq_vs_f_agree = 0
    abs_err, abs_err_dq, fl_std, margins_disagree = [], [], [], []
    tf_big_total = tf_big_agree = 0
    for pr in prompts:
        gf = greedy_float(cfg, im.Wf, pr, gen_len)
        gi = im.greedy(pr, gen_len)
        exact_seq += int(gf == gi)
        tok_match += sum(int(a == b) for a, b in zip(gf, gi))
        tok_total += gen_len
        fd = next((i for i, (a, b) in enumerate(zip(gf, gi)) if a != b), gen_len)
        first_div.append(fd)
        seq = list(pr) + gf[:-1]
        lf = float_forward(cfg, im.Wf, seq)
        ldq = float_forward(cfg, Wdq, seq)
        lint, _, deq = im.forward(seq)
        ai, af, adq = np.argmax(lint, -1), np.argmax(lf, -1), np.argmax(ldq, -1)
        tf_agree += int((ai == af).sum())
        tf_agree_dq += int((ai == adq).sum())
        dq_vs_f_agree += int((adq == af).sum())
        tf_total += len(seq)
        abs_err.append(np.abs(deq - lf))
        abs_err_dq.append(np.abs(deq - ldq))
        fl_std.append(lf.std(axis=-1))
        srt = np.sort(lf, axis=-1)
        margin = srt[:, -1] - srt[:, -2]
        margins_disagree.extend(margin[ai != af].tolist())
        big = margin > 1.0
        tf_big_total += int(big.sum())
        tf_big_agree += int(((ai == af) & big).sum())
    ae, aedq = np.concatenate([a.ravel() for a in abs_err]), np.concatenate([a.ravel() for a in abs_err_dq])
    return {
        "n_prompts": n_prompts, "prompt_len": prompt_len, "gen_len": gen_len, "eval_seed": eval_seed,
        "greedy_exact_sequence_match": exact_seq / n_prompts,
        "greedy_token_agreement": tok_match / tok_total,
        "greedy_mean_first_divergence": float(np.mean(first_div)),
        "teacher_forced_top1_agreement_vs_float": tf_agree / tf_total,
        "teacher_forced_top1_agreement_vs_float_dequant_weights": tf_agree_dq / tf_total,
        "float_dequant_vs_float_top1_agreement": dq_vs_f_agree / tf_total,
        "teacher_forced_positions": tf_total,
        "logit_max_abs_err_vs_float": float(ae.max()),
        "logit_mean_abs_err_vs_float": float(ae.mean()),
        "logit_max_abs_err_vs_float_dequant_weights": float(aedq.max()),
        "logit_mean_abs_err_vs_float_dequant_weights": float(aedq.mean()),
        "float_logit_std_mean": float(np.mean(np.concatenate(fl_std))),
        "float_top1_margin_at_disagreements_mean": float(np.mean(margins_disagree)) if margins_disagree else None,
        "teacher_forced_disagreements": len(margins_disagree),
        "teacher_forced_top1_agreement_when_float_margin_gt_1": tf_big_agree / tf_big_total if tf_big_total else None,
        "positions_with_float_margin_gt_1": tf_big_total,
    }
