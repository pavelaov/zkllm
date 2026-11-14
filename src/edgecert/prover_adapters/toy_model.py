"""Toy integer autoregressive LM used by the transparent reference prover.

"window-mlp-lm/v1": a deterministic, fully integer next-token model that is small enough to prove in pure Python yet exercises every relation component of R_EDGE that the attack suite needs (embedding gather, positional encoding, padding mask, matmul + bias, requantization, lookup-table activation, greedy argmax with tie-break, EOS/length, chunk state).

For generation step i (predicting output token i):
    window_i = last W tokens of (state.window ++ prompt ++ out[:i])   (PAD = -1 masked -> zero vector)
    pos_i    = state.position + len(prompt) + i                     (< max_position)
    x_i      = concat(emb[window_i[0]], ..., emb[window_i[W-1]], pos[pos_i], 1)
    y_0 = x_i ; for l: acc_l = W_l y_l + b_l  (bias as augmented column)
                       n     = acc_l * M_l,  sigma = [n < 0]
                       q_l   = floor((n + 2^(s_l-1) - sigma) / 2^s_l)      (RHAZ: round half away from zero)
                       z_l   = clamp(q_l, -128, 127)
                       y_{l+1} = LUT_act[z_l]  (hidden layers),  logits = z_L (last layer)
    out[i] = argmin{ j : logits[j] = max(logits) }   (greedy, lowest-index tie-break)
Generation stops after EOS or after max_new_tokens.

The requantization rule is RHAZ, the same rule as the lowering team's integer reference (src/edgecert/reference/intops.py: rhaz_shift / requant), so the toy relation follows spec/numerical_semantics.md once that file is fixed (spec/inference_relation.md Section 5 restates the rule).  All randomness here is a public, seeded synthetic model (no real weights).
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field as dfield
from typing import Dict, List, Optional, Sequence, Tuple

from ..crypto import merkle
from ..runtime.manifest import digest_obj

PAD = -1
INT8 = (-128, 127)

@dataclass
class ToyConfig:
    vocab: int = 32
    d_emb: int = 16
    window: int = 4
    max_position: int = 64
    hidden: Tuple[int, ...] = (64, 64)
    eos_id: int = 1
    weight_range: int = 64       # weights uniform in [-weight_range, weight_range]
    bias_range: int = 2048
    shift: int = 16
    target_std: float = 40.0
    eos_bias: int = 10000        # extra bias on the EOS logit row so that some prompts terminate early
    seed: int = 20260927

    @property
    def d_in(self) -> int:
        return self.window * self.d_emb + self.d_emb

def act_lut_leaky() -> List[int]:
    """Authorized activation table act0: int8 -> int8, v>=0 -> v, v<0 -> -floor(-v/4)."""
    return [v if v >= 0 else -((-v) >> 2) for v in range(-128, 128)]

def act_lut_relu() -> List[int]:
    """A DIFFERENT table (used by the 'wrong lookup table' attack)."""
    return [max(v, 0) for v in range(-128, 128)]

def lut_digest(name: str, table: Sequence[int], domain=INT8) -> str:
    return digest_obj("EDGE-CERT/lut/v1", {"name": name, "domain": list(domain), "table": list(table)})

def requant(acc: int, M: int, s: int, rounding: str = "RHAZ") -> Tuple[int, int, int]:
    """Return (q, rem, sigma) with acc*M + h - sigma = q*2^s + rem, 0 <= rem < 2^s.

    RHAZ (authorized): h = 2^(s-1), sigma = [acc*M < 0]  -> round half away from zero.
    HALF_UP:           h = 2^(s-1), sigma = 0.
    TRUNC_FLOOR:       h = 0,       sigma = 0            (used only by the attack suite).
    """
    n = acc * M
    h = (1 << (s - 1)) if (rounding in ("RHAZ", "HALF_UP") and s > 0) else 0
    sigma = 1 if (rounding == "RHAZ" and n < 0) else 0
    t = n + h - sigma
    q = t >> s  # floor division for negative numbers too
    rem = t - (q << s)
    return q, rem, sigma

def clamp(v: int, lo: int = INT8[0], hi: int = INT8[1]) -> int:
    return lo if v < lo else hi if v > hi else v

def argmax_lowest(v: Sequence[int]) -> int:
    best = 0
    for j in range(1, len(v)):
        if v[j] > v[best]:
            best = j
    return best

@dataclass
class ToyModel:
    cfg: ToyConfig
    tensors: Dict[str, list]                     # name -> nested int lists
    luts: Dict[str, List[int]]
    layers: List[dict]                           # [{name, in, out, weight, bias, activation, multiplier, shift}]
    graph_spec: dict = dfield(default_factory=dict)
    tokenizer: dict = dfield(default_factory=dict)

    def clone(self) -> "ToyModel":
        return copy.deepcopy(self)

    # ------------------------------------------------------------ forward
    def gather(self, window: Sequence[int], position: int, unmask: bool = False) -> List[int]:
        x: List[int] = []
        emb = self.tensors["emb"]
        for t in window:
            if t == PAD:
                x += [0] * self.cfg.d_emb if not unmask else list(emb[0])
            else:
                x += list(emb[t])
        if not 0 <= position < self.cfg.max_position:
            raise ValueError("position outside max_position")
        x += list(self.tensors["pos"][position])
        x.append(1)  # bias lane
        return x

    def weight_aug(self, layer: dict, tensors: Optional[Dict[str, list]] = None) -> List[List[int]]:
        t = tensors if tensors is not None else self.tensors
        W, b = t[layer["weight"]], t[layer["bias"]]
        return [list(W[i]) + [b[i]] for i in range(layer["out"])]

    def step(self, x: List[int], tensors=None, lut=None, scales=None, rounding=None) -> dict:
        """One generation step; returns per-layer witness + logits."""
        y = x
        rec = []
        for li, L in enumerate(self.layers):
            Wa = self.weight_aug(L, tensors)
            M, s = (scales or {}).get(li, (L["multiplier"], L["shift"]))
            acc = [sum(w * v for w, v in zip(row, y)) for row in Wa]
            qr = [requant(a, M, s, rounding or "RHAZ") for a in acc]
            q = [a for a, _, _ in qr]
            rem = [b for _, b, _ in qr]
            sig = [c for _, _, c in qr]
            z = [clamp(v) for v in q]
            if L["activation"] is not None:
                table = (lut if lut is not None else self.luts)[L["activation"]]
                out = [table[v + 128] for v in z]
            else:
                out = z
            rec.append({"acc": acc, "q": q, "rem": rem, "sig": sig, "out": out})
            y = out + [1]
        return {"layers": rec, "logits": rec[-1]["out"][: self.cfg.vocab]}

    def windows(self, state_window: Sequence[int], state_pos: int, prompt: Sequence[int],
                out: Sequence[int]) -> List[Tuple[List[int], int]]:
        ctx = list(state_window) + list(prompt)
        res = []
        for i in range(len(out)):
            full = ctx + list(out[:i])
            res.append((full[-self.cfg.window:], state_pos + len(prompt) + i))
        return res

    def generate(self, state_window, state_pos, prompt, max_new, **kw) -> List[int]:
        ctx = list(state_window) + list(prompt)
        out: List[int] = []
        for i in range(max_new):
            win = (ctx + out)[-self.cfg.window:]
            x = self.gather(win, state_pos + len(prompt) + i, unmask=kw.get("unmask", False))
            st = self.step(x, kw.get("tensors"), kw.get("lut"), kw.get("scales"), kw.get("rounding"))
            tok = argmax_lowest(st["logits"])
            out.append(tok)
            if tok == self.cfg.eos_id:
                break
        return out

    def end_state(self, state_window, state_pos, prompt, out) -> Tuple[List[int], int]:
        full = list(state_window) + list(prompt) + list(out)
        return full[-self.cfg.window:], state_pos + len(prompt) + len(out)

    def commitment(self, extra_delta: Optional[Tuple[str, list]] = None) -> str:
        return merkle.commit_model(self.tensors).commitment.hex()

def _rand_matrix(rng: random.Random, rows: int, cols: int, r: int) -> List[List[int]]:
    return [[rng.randint(-r, r) for _ in range(cols)] for _ in range(rows)]

def build_model(cfg: ToyConfig = ToyConfig()) -> ToyModel:
    rng = random.Random(cfg.seed)
    tensors: Dict[str, list] = {
        "emb": _rand_matrix(rng, cfg.vocab, cfg.d_emb, 127),
        "pos": _rand_matrix(rng, cfg.max_position, cfg.d_emb, 127),
    }
    dims = [cfg.d_in] + list(cfg.hidden) + [cfg.vocab]
    layers = []
    for li in range(len(dims) - 1):
        name = f"L{li}"
        tensors[f"{name}.w"] = _rand_matrix(rng, dims[li + 1], dims[li], cfg.weight_range)
        tensors[f"{name}.b"] = [rng.randint(-cfg.bias_range, cfg.bias_range) for _ in range(dims[li + 1])]
        layers.append({"name": name, "in": dims[li], "out": dims[li + 1], "weight": f"{name}.w",
                       "bias": f"{name}.b", "activation": "act0" if li < len(dims) - 2 else None,
                       "multiplier": 1, "shift": cfg.shift})
    m = ToyModel(cfg, tensors, {"act0": act_lut_leaky()}, layers)
    _calibrate(m, rng)
    tensors[f"L{len(layers) - 1}.b"][cfg.eos_id] += cfg.eos_bias
    m.graph_spec = graph_spec(m)
    m.tokenizer = {"id": f"toy-vocab{cfg.vocab}", "version": "1",
                   "digest": digest_obj("EDGE-CERT/tokenizer/v1", {"vocab": [f"<t{i}>" for i in range(cfg.vocab)]}),
                   "vocab_size": cfg.vocab, "eos_id": cfg.eos_id, "pad_id": PAD}
    return m

def _calibrate(m: ToyModel, rng: random.Random) -> None:
    """Choose per-layer integer multipliers so requantized outputs are well spread.

    Calibration only picks the (public) quantizer constants; it proves nothing
    about ranges.  Ranges are enforced by the relation's range constraints.
    """
    samples = []
    for _ in range(24):
        win = [rng.randrange(m.cfg.vocab) for _ in range(m.cfg.window)]
        samples.append(m.gather(win, rng.randrange(m.cfg.max_position)))
    ys = samples
    for li, L in enumerate(m.layers):
        Wa = m.weight_aug(L)
        accs = [sum(w * v for w, v in zip(row, y)) for y in ys for row in Wa]
        mean = sum(accs) / len(accs)
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5 or 1.0
        L["multiplier"] = max(1, int(round(m.cfg.target_std * (1 << L["shift"]) / std)))
        new = []
        for y in ys:
            acc = [sum(w * v for w, v in zip(row, y)) for row in Wa]
            z = [clamp(requant(a, L["multiplier"], L["shift"])[0]) for a in acc]
            out = [m.luts["act0"][v + 128] for v in z] if L["activation"] else z
            new.append(out + [1])
        ys = new

def graph_spec(m: ToyModel) -> dict:
    c = m.cfg
    return {
        "type": "window-mlp-lm/v1", "vocab": c.vocab, "d_emb": c.d_emb, "window": c.window,
        "max_position": c.max_position, "pad_id": PAD, "embedding": "emb", "positional": "pos",
        "layers": [{k: L[k] for k in ("name", "in", "out", "weight", "bias", "activation")} for L in m.layers],
        "tensor_shapes": {n: list(merkle.flatten(v)[1]) for n, v in sorted(m.tensors.items())},
        "activation_domain": list(INT8), "clamp": list(INT8),
    }

# ------------------------------------------------------------ adapters
def adapter_delta(m: ToyModel, target: str, seed: int, density: int = 16, mag: int = 4) -> List[List[int]]:
    rng = random.Random(seed)
    W = m.tensors[target]
    return [[(rng.randint(-mag, mag) if rng.randrange(density) == 0 else 0) for _ in row] for row in W]

def adapter_commitment(target: str, delta) -> str:
    return merkle.commit_model({target + ".delta": delta}).commitment.hex()

def combined_commitment(base_commitment_hex: str, adapter_commitment_hex: str) -> str:
    import hashlib
    return hashlib.sha256(b"EDGE-CERT/model+adapter/v1" + bytes.fromhex(base_commitment_hex)
                          + bytes.fromhex(adapter_commitment_hex)).hexdigest()

def apply_adapter(tensors: Dict[str, list], target: str, delta) -> Dict[str, list]:
    t = copy.deepcopy(tensors)
    t[target] = [[a + d for a, d in zip(ra, rd)] for ra, rd in zip(t[target], delta)]
    return t

# ------------------------------------------------------------ manifest (publisher helper)
RELATION_ID = "R_EDGE/window-mlp-lm/v1"
PROOF_CONFIG_ID = "TRANSPARENT_REFERENCE/sumcheck-goldilocks-ext2/v1"

def manifest_body(m: ToyModel, publisher_key, epoch: int = 1, adapters: Sequence[dict] = (),
                  issued_at: int = 1790000000, not_after: Optional[int] = None,
                  revokes: Sequence[str] = (), graph_override: Optional[dict] = None) -> dict:
    gs = graph_override if graph_override is not None else m.graph_spec
    return {
        "manifest_version": "edgecert-manifest/v1",
        "publisher": {"name": "toy-publisher", "key_id": publisher_key.key_id, "alg": "Ed25519",
                      "public_key": publisher_key.public.hex()},
        "model": {"family": "toy-window-mlp-lm", "version": "1", "checkpoint_digest": None,
                  "model_commitment": m.commitment(), "commitment_scheme": "merkle-sha256/v1",
                  "numeric_tier": "AUTHORIZED_DERIVED", "weights_visibility": "PUBLIC"},
        "graph": {"format": "edgecert-window-mlp-lm/v1",
                  "graph_digest": digest_obj("EDGE-CERT/graph/v1", gs), "spec": gs},
        "tokenizer": dict(m.tokenizer),
        "quantizer": {"scheme": "W8A8-int-requant", "weight_bits": 8, "activation_bits": 8,
                      "rounding": "RHAZ", "clamp": list(INT8),
                      "layers": [{"name": L["name"], "multiplier": L["multiplier"], "shift": L["shift"],
                                  "activation": L["activation"]} for L in m.layers]},
        "lookup_tables": [{"name": n, "digest": lut_digest(n, t), "domain": list(INT8)}
                          for n, t in sorted(m.luts.items())],
        "positional_encoding": {"kind": "LEARNED_ABSOLUTE", "max_position": m.cfg.max_position, "tensor": "pos"},
        "attention_mask": {"kind": "CAUSAL_WINDOW_PAD", "window": m.cfg.window},
        "decoding": {"rule": "GREEDY", "tie_break": "LOWEST_INDEX", "eos_id": m.cfg.eos_id,
                     "max_new_tokens_cap": m.cfg.max_position},
        "adapters": list(adapters),
        "proof_system": {"relation_id": RELATION_ID, "proof_config_id": PROOF_CONFIG_ID,
                         "backend": "edgecert-transparent-reference", "backend_version": "0.1",
                         "field": "goldilocks p=2^64-2^32+1", "challenge_field": "Fp[X]/(X^2-7)",
                         "hash": "SHA-256", "zero_knowledge": False, "succinct": False,
                         "claimed_security_bits": None, "verifier_key_id": None,
                         "setup": "transparent (hash-based Fiat-Shamir, no trusted setup)"},
        "policy": {"policy_epoch": epoch, "rollback_protection": "monotonic-epoch", "issued_at": issued_at,
                   "not_after": not_after, "revokes": list(revokes)},
    }
