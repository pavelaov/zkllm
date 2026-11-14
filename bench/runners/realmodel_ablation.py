"""Where does the W8A8 quality loss come from?  Float fake-quantization ablation.

A float64 simulation (evidence class SIMULATION, not the certified integer model) of the W8A8_sq_sink recipe: smoothing transform + BOS sink prefix, per-output-channel int8 weights, per-tensor int8 activations with the same calibrated maxima (set A), int16 (or int20) residual, activation-table input clamp to [-16, 16).  Quantization points can be switched off one tensor type at a time (leave-one-out) to attribute the teacher-forced top-1 loss. Top-1 agreement is against the unquantized float model on the prompt tokens of the first ``--n`` prompts of set B.

Usage: python bench/runners/realmodel_ablation.py --model qwen3 [--n 16]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from edgecert.reference import hf_models as HM  # noqa: E402
from edgecert.reference.bpe import load_tokenizer  # noqa: E402
from edgecert.reference.hf_models import _ln, _rms, _rope_f  # noqa: E402
from edgecert.reference.intops import gelu_tanh, silu  # noqa: E402

OUT = ROOT / "results" / "raw" / "realmodels"
INFO = {"gpt2": ("models/gpt2", 50256, "gpt2"), "qwen3": ("models/qwen3-0.6b", 151643, "qwen3_0.6b")}
def _probe():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from realmodel_eval import probe_contention  # noqa: E402
    return probe_contention()

LIN = ("wq", "wk", "wv", "wo", "wg", "wu", "wd", "wqkv", "w1", "w2")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(INFO), required=True)
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args(argv)
    d, bos, fname = INFO[args.model]
    t_all = time.perf_counter()
    cfg = HM.load_config(ROOT / d)
    W = HM.load_float_weights(cfg)
    fm = HM.FloatLM(cfg, W)
    tok = load_tokenizer(cfg.family, ROOT / d)
    A = [tok.encode(p) for p in json.loads((ROOT / "data/prompts/set_A_calibration.json").read_text())["prompts"]]
    B = [tok.encode(p) for p in json.loads((ROOT / "data/prompts/set_B_evaluation.json").read_text())["prompts"][:args.n]]
    pre = [bos]
    cal0 = HM.calibrate(fm, A, prefix=pre)
    Ws, _ = HM.smooth_weights(cfg, W, cal0)
    cal = HM.calibrate(HM.FloatLM(cfg, Ws), A, prefix=pre)
    ref = [fm.forward([(np.array(pre + b), fm.new_state())], emit="all")[0][1:] for b in B]

    def qw(Wf):
        s = np.abs(Wf).max(1, keepdims=True) / 127
        s = np.where(s == 0, 1.0, s)
        return np.clip(np.round(Wf / s), -127, 127) * s
    Wq = {k: (qw(v) if k.split(".")[-1] in LIN else v) for k, v in Ws.items()}
    Wq["emb"] = qw(Ws["emb"])
    if "lm" in Ws:
        Wq["lm"] = qw(Ws["lm"])

    def fq(x, key, bits, skip):
        if key in skip or key not in cal:
            return x
        qmax = (1 << (bits - 1)) - 1
        s = cal[key]["max"] / 127 if bits == 8 else cal[key]["max"] * 1.1 / qmax
        y = np.clip(np.round(x / s), -qmax - 1, qmax) * s
        y[:1] = x[:1]                        # the committed prefix row is a constant, not recomputed
        return y

    def run(tokens, skip, wq=True, resbits=16):
        Wt = Wq if wq else Ws
        n = len(tokens)
        pos = np.arange(n)
        H, Hkv, dh, rep = cfg.n_heads, cfg.n_kv_heads, cfg.dh, cfg.n_rep
        R = (lambda h: h) if "resid" in skip else (lambda h: fq(h, "resid", resbits, skip))
        h = Wt["emb"][tokens] + (Wt["pos"][pos] if cfg.family == "gpt2" else 0)
        h = R(h)
        for l in range(cfg.n_layers):
            p = f"L{l}."
            if cfg.family == "gpt2":
                x = fq(_ln(h, Wt[p + "n1_g"], Wt[p + "n1_b"], cfg.norm_eps), p + "x1", 8, skip)
                qkv = x @ Wt[p + "wqkv"].T + Wt[p + "bqkv"]
                D = cfg.d
                q = fq(qkv[:, :D].reshape(n, H, dh), p + "q", 8, skip)
                k = fq(qkv[:, D:2 * D].reshape(n, H, dh), p + "k", 8, skip)
                v = fq(qkv[:, 2 * D:].reshape(n, H, dh), p + "v", 8, skip)
            else:
                x = fq(_rms(h, Wt[p + "n1_g"], cfg.norm_eps), p + "x1", 8, skip)
                q = fq((x @ Wt[p + "wq"].T).reshape(n, H, dh), p + "q_pre", 8, skip)
                k = fq((x @ Wt[p + "wk"].T).reshape(n, Hkv, dh), p + "k_pre", 8, skip)
                v = fq((x @ Wt[p + "wv"].T).reshape(n, Hkv, dh), p + "v", 8, skip)
                q = _rope_f(fq(_rms(q, Wt[p + "qn_g"], cfg.norm_eps), p + "q", 8, skip), pos, cfg.rope_base)
                k = _rope_f(fq(_rms(k, Wt[p + "kn_g"], cfg.norm_eps), p + "k", 8, skip), pos, cfg.rope_base)
            sc = np.einsum("mgrd,ngd->grmn", q.reshape(n, Hkv, rep, dh), k) / math.sqrt(dh)
            sc = np.where(np.tril(np.ones((n, n), bool))[None, None], sc, -np.inf)
            e = np.exp(sc - sc.max(-1, keepdims=True))
            a = fq(np.einsum("grmn,ngd->mgrd", e / e.sum(-1, keepdims=True), v).reshape(n, H * dh), p + "a", 8, skip)
            h = R(h + a @ Wt[p + "wo"].T + (Wt[p + "bo"] if cfg.family == "gpt2" else 0))
            clip = (lambda u: u) if "clip16" in skip else (lambda u: np.clip(u, -16, 16))
            if cfg.family == "gpt2":
                x2 = fq(_ln(h, Wt[p + "n2_g"], Wt[p + "n2_b"], cfg.norm_eps), p + "x2", 8, skip)
                g = fq(gelu_tanh(clip(x2 @ Wt[p + "w1"].T + Wt[p + "b1"])), p + "act", 8, skip)
                h = R(h + g @ Wt[p + "w2"].T + Wt[p + "b2"])
            else:
                x2 = fq(_rms(h, Wt[p + "n2_g"], cfg.norm_eps), p + "x2", 8, skip)
                sg = fq(silu(clip(x2 @ Wt[p + "wg"].T)), p + "act", 8, skip)
                up = fq(x2 @ Wt[p + "wu"].T, p + "up", 8, skip)
                h = R(h + fq(sg * up, p + "glu", 8, skip) @ Wt[p + "wd"].T)
        hf = _ln(h, Wt["nf_g"], Wt["nf_b"], cfg.norm_eps) if cfg.family == "gpt2" else _rms(h, Wt["nf_g"], cfg.norm_eps)
        xf = fq(hf, "xf", 8, skip)
        return (xf @ (Wt["lm"] if "lm" in Wt else Wt["emb"]).T)[1:]

    types = ["resid", "x1", "q", "k", "v", "a", "x2", "act", "xf"] + \
            (["q_pre", "k_pre", "up", "glu"] if cfg.family == "qwen3" else [])

    def keys(t):
        return {t} if t in ("resid", "xf") else {f"L{l}.{t}" for l in range(cfg.n_layers)}
    all_keys = set().union(*[keys(t) for t in types])

    def tf(skip, wq=True, resbits=16):
        ag = tot = 0
        for b, lf in zip(B, ref):
            lq = run(np.array(pre + b), skip, wq, resbits)
            ag += int((lq.argmax(-1) == lf.argmax(-1)).sum())
            tot += len(b)
        return ag / tot, tot
    res = {}
    for name, (skip, wq, rb) in {"all_quantized_R16": (set(), True, 16), "all_quantized_R20": (set(), True, 20),
                                 "weights_only": (all_keys, True, 16), "activations_only_R16": (set(), False, 16),
                                 "no_act_input_clamp": ({"clip16"}, True, 16)}.items():
        res[name], npos = tf(skip, wq, rb)
        print(name, res[name], flush=True)
    loo = {}
    for t in types:
        loo[t], _ = tf(keys(t))
        print("leave out", t, loo[t], flush=True)
    rec = {"status": "SUCCESS", "evidence_class": "SIMULATION", "record_type": "actquant_ablation",
           "model": cfg.name, "command": "python " + " ".join([Path(sys.argv[0]).as_posix()] + (argv or sys.argv[1:])),
           "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "method": "float64 fake quantization (round-to-nearest, per-tensor int8 activations with the calibrated "
                     "maxima, per-channel int8 weights) of the W8A8_sq_sink recipe; NOT the certified integer model",
           "prompts": len(B), "positions": npos, "metric": "teacher-forced top-1 agreement with the float model on "
                                                          "the prompt tokens (set B)",
           "results": res, "leave_one_out": loo,
           "data_provenance": "self-authored prompts (data/prompts), not a standard benchmark",
           "contention": _probe(), "runtime_s": round(time.perf_counter() - t_all, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{fname}_actquant_ablation.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("wrote", OUT / f"{fname}_actquant_ablation.json")

if __name__ == "__main__":
    main()
