"""Architecture hyperparameters of the evaluated models.

Only values we are confident about are marked verified.  Any field we could not double-check against a primary source in this environment (no internet access during the build) is listed in ``unverified_fields`` and makes ``verified`` False. Parameter counts computed from these shapes are compared with the publicly reported totals (``reported_params``, evidence class REPORTED) as a sanity check. Weights used anywhere in EDGE-CERT for these shapes are synthetic int8.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

@dataclass(frozen=True)
class ModelShape:
    name: str
    hf_id: str
    family: str                  # gpt2 | qwen3 | gemma3 | llama3
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    ctx: int                     # context length used for certified bounds
    norm: str                    # 'ln' (LayerNorm) | 'rms' (RMSNorm)
    pos: str                     # 'learned' | 'rope'
    mlp: str                     # 'gelu' | 'swiglu' | 'geglu'
    bias: bool
    tied_embeddings: bool
    qk_norm: bool
    sandwich_norm: bool          # extra post-attention / post-FFN norms (Gemma 3)
    sliding_window: Optional[int]
    global_every: Optional[int]  # layer l is global iff (l + 1) % global_every == 0
    rope_theta: Optional[float]
    norm_eps: float
    unverified_fields: Tuple[str, ...] = ()
    reported_params: Optional[int] = None   # publicly reported total (REPORTED)
    notes: str = ""

    @property
    def verified(self) -> bool:
        return not self.unverified_fields

    def is_global(self, layer: int) -> bool:
        if self.sliding_window is None or self.global_every is None:
            return True
        return (layer + 1) % self.global_every == 0

    def window(self, layer: int) -> Optional[int]:
        return None if self.is_global(layer) else self.sliding_window

    def param_count(self) -> Dict[str, int]:
        d, L, V = self.d_model, self.n_layers, self.vocab
        q = self.n_heads * self.head_dim
        kv = self.n_kv_heads * self.head_dim
        attn = d * q + 2 * d * kv + q * d
        if self.bias:
            attn += q + 2 * kv + d
        if self.mlp == "gelu":
            mlp = 2 * d * self.d_ff + (self.d_ff + d if self.bias else 0)
        else:
            mlp = 3 * d * self.d_ff
        norms_per_layer = (4 if self.sandwich_norm else 2) * d * (2 if self.norm == "ln" else 1)
        if self.qk_norm:
            norms_per_layer += 2 * self.head_dim
        per_layer = attn + mlp + norms_per_layer
        emb = V * d
        pos = self.ctx * d if self.pos == "learned" else 0
        head = 0 if self.tied_embeddings else V * d
        final_norm = d * (2 if self.norm == "ln" else 1)
        total = L * per_layer + emb + pos + head + final_norm
        return {"per_layer": per_layer, "non_embedding": L * per_layer + final_norm,
                "embedding": emb + pos, "lm_head_untied": head, "total": total}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verified"] = self.verified
        d["computed_params"] = self.param_count()
        return d

GPT2_124M = ModelShape(
    name="gpt2_124m", hf_id="openai-community/gpt2", family="gpt2",
    n_layers=12, d_model=768, n_heads=12, n_kv_heads=12, head_dim=64, d_ff=3072,
    vocab=50257, ctx=1024, norm="ln", pos="learned", mlp="gelu", bias=True,
    tied_embeddings=True, qk_norm=False, sandwich_norm=False, sliding_window=None,
    global_every=None, rope_theta=None, norm_eps=1e-5,
    reported_params=124_439_808,
    notes="GPT-2 small; activation gelu_new (tanh form); fused c_attn qkv with bias.")

QWEN3_0_6B = ModelShape(
    name="qwen3_0.6b", hf_id="Qwen/Qwen3-0.6B", family="qwen3",
    n_layers=28, d_model=1024, n_heads=16, n_kv_heads=8, head_dim=128, d_ff=3072,
    vocab=151936, ctx=32768, norm="rms", pos="rope", mlp="swiglu", bias=False,
    tied_embeddings=True, qk_norm=True, sandwich_norm=False, sliding_window=None,
    global_every=None, rope_theta=1_000_000.0, norm_eps=1e-6,
        # ctx: model card states 32,768 native context; the HF config's max_position_embeddings (40,960) was not re-checked offline.
    unverified_fields=("ctx",),
    reported_params=600_000_000,   # model card: "0.6B total, 0.44B non-embedding" (rounded)
    notes="head_dim 128 with 16 query heads gives a 2048-wide q projection (> d_model).")

GEMMA3_1B = ModelShape(
    name="gemma3_1b", hf_id="google/gemma-3-1b-pt", family="gemma3",
    n_layers=26, d_model=1152, n_heads=4, n_kv_heads=1, head_dim=256, d_ff=6912,
    vocab=262144, ctx=32768, norm="rms", pos="rope", mlp="geglu", bias=False,
    tied_embeddings=True, qk_norm=True, sandwich_norm=True, sliding_window=512,
    global_every=6, rope_theta=1_000_000.0, norm_eps=1e-6,
        # vocab: 262,144 matches the reported 302M embedding parameters (262144*1152), but the exact config value (vs. a padded multimodal size) was not re-checked offline. sliding_window=512 and the 5-local:1-global pattern are from the Gemma 3 report; the per-layer placement (layer l global iff (l+1) % 6 == 0) is an assumption.
    unverified_fields=("vocab", "sliding_window", "global_every", "ctx"),
    reported_params=1_000_000_000,  # tech report: ~302M embedding + ~698M non-embedding
    notes="GeGLU with gelu(tanh); query_pre_attn_scalar folded into the score requant; "
          "embedding scaled by sqrt(d) (folded into the embedding multiplier).")

LLAMA3_8B = ModelShape(
    name="llama3_8b", hf_id="meta-llama/Meta-Llama-3-8B", family="llama3",
    n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128, d_ff=14336,
    vocab=128256, ctx=8192, norm="rms", pos="rope", mlp="swiglu", bias=False,
    tied_embeddings=False, qk_norm=False, sandwich_norm=False, sliding_window=None,
    global_every=None, rope_theta=500_000.0, norm_eps=1e-5,
    reported_params=8_030_000_000,
    notes="8B-class stress shape (workstation/cloud tier T4, not an edge target). "
          "Llama 3.1 extends ctx to 131072; we certify for 8192.")

MODELS: Dict[str, ModelShape] = {m.name: m for m in (GPT2_124M, QWEN3_0_6B, GEMMA3_1B, LLAMA3_8B)}
