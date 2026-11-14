"""Public instance of R_EDGE and the byte encodings bound by it (shared wire format).

The prover imports this module to build instances; the verifier imports it to recompute digests.  Nothing here depends on prover code.

Public instance (spec/inference_relation.md Section 3):

    I = (manifest_id, model_commitment, relation_id, proof_config_id,
         session_id, requester_nonce, input_commitment,
         output_tokens_or_commitment, context_anchor,
         output_length, previous_receipt_id, policy_epoch)

instance_digest = SHA-256("EDGE-CERT/instance/v1\\x00" || canonical_json(I)).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

from ..crypto.commit import commit as hcommit, verify_opening, Opening
from ..runtime.manifest import canonical_json

INSTANCE_FIELDS = (
    "manifest_id", "model_commitment", "relation_id", "proof_config_id",
    "session_id", "requester_nonce", "input_commitment",
    "output_tokens_or_commitment", "context_anchor",
    "output_length", "previous_receipt_id", "policy_epoch",
)
INSTANCE_DST = b"EDGE-CERT/instance/v1\x00"
GENESIS_RECEIPT_ID = "00" * 32
GENESIS_ANCHOR = hashlib.sha256(b"EDGE-CERT/context-anchor/genesis/v1").hexdigest()

INPUT_DST = b"EDGE-CERT/input-commitment/v1"
STATE_DST = b"EDGE-CERT/state-commitment/v1"
OUTPUT_DST = b"EDGE-CERT/output-commitment/v1"

class InstanceError(ValueError):
    pass

def make_instance(**kw: Any) -> Dict[str, Any]:
    missing = [f for f in INSTANCE_FIELDS if f not in kw]
    extra = [k for k in kw if k not in INSTANCE_FIELDS]
    if missing or extra:
        raise InstanceError(f"instance fields missing={missing} extra={extra}")
    return {f: kw[f] for f in INSTANCE_FIELDS}

def instance_digest(inst: Dict[str, Any]) -> bytes:
    if tuple(sorted(inst)) != tuple(sorted(INSTANCE_FIELDS)):
        raise InstanceError("instance has wrong field set")
    return hashlib.sha256(INSTANCE_DST + canonical_json(inst)).digest()

# ------------------------------------------------------------ encodings
def encode_input_record(tokens: Sequence[int], max_new_tokens: int, tokenizer_id: str) -> bytes:
    """What input_commitment commits to: prompt tokens + generation budget + tokenizer."""
    return canonical_json({"tokens": list(tokens), "max_new_tokens": int(max_new_tokens),
                           "tokenizer_id": tokenizer_id})

def encode_state(window: Sequence[int], position: int) -> bytes:
    """Logical context state at a chunk boundary (window tokens + next position)."""
    return canonical_json({"window": list(window), "position": int(position)})

def encode_output(tokens: Sequence[int]) -> bytes:
    return canonical_json({"tokens": list(tokens)})

def commit_input(tokens, max_new_tokens, tokenizer_id, randomness: Optional[bytes] = None):
    return hcommit(encode_input_record(tokens, max_new_tokens, tokenizer_id), randomness, INPUT_DST)

def check_input_opening(commitment_hex: str, tokens, max_new_tokens, tokenizer_id, randomness_hex: str) -> bool:
    try:
        return verify_opening(bytes.fromhex(commitment_hex),
                              Opening(encode_input_record(tokens, max_new_tokens, tokenizer_id),
                                      bytes.fromhex(randomness_hex)), INPUT_DST)
    except (ValueError, TypeError):
        return False

def commit_state(window, position, randomness: Optional[bytes] = None):
    return hcommit(encode_state(window, position), randomness, STATE_DST)

def check_state_opening(commitment_hex: str, window, position, randomness_hex: str) -> bool:
    try:
        return verify_opening(bytes.fromhex(commitment_hex),
                              Opening(encode_state(window, position), bytes.fromhex(randomness_hex)), STATE_DST)
    except (ValueError, TypeError):
        return False

def commit_output(tokens, randomness: Optional[bytes] = None):
    return hcommit(encode_output(tokens), randomness, OUTPUT_DST)

def check_output_opening(commitment_hex: str, tokens, randomness_hex: str) -> bool:
    try:
        return verify_opening(bytes.fromhex(commitment_hex),
                              Opening(encode_output(tokens), bytes.fromhex(randomness_hex)), OUTPUT_DST)
    except (ValueError, TypeError):
        return False

def output_field_tokens(tokens: Sequence[int]) -> Dict[str, Any]:
    return {"mode": "tokens", "tokens": list(tokens)}

def output_field_commitment(commitment_hex: str) -> Dict[str, Any]:
    return {"mode": "commitment", "commitment": commitment_hex}
