"""Verifier-issued certification receipts.

A receipt is issued ONLY by an independent verifier after every check of a proof passed.  It is signed with the verifier's Ed25519 key and identified by

    receipt_id = SHA-256("EDGE-CERT/receipt-id/v1\\x00" || canonical(body)).

The receipt carries the end-of-chunk state commitment, which becomes the context_anchor of the next chunk, and the previous receipt id, so receipts of a multi-chunk session form a hash-linked chain.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping

from ..crypto import ed25519
from ..runtime.manifest import canonical_json

RECEIPT_VERSION = "edgecert-receipt/v1"
SIG_DST = b"EDGE-CERT/receipt/v1\x00"
ID_DST = b"EDGE-CERT/receipt-id/v1\x00"

RECEIPT_FIELDS = ("receipt_version", "session_id", "instance_digest", "manifest_id", "model_commitment",
                  "relation_id", "proof_config_id", "certificate_kind", "verdict", "end_state_commitment",
                  "previous_receipt_id", "output_length", "policy_epoch", "verifier_key_id")

def receipt_id(body: Dict[str, Any]) -> str:
    return hashlib.sha256(ID_DST + canonical_json(body)).hexdigest()

def issue_receipt(body: Dict[str, Any], key: ed25519.SigningKey) -> Dict[str, Any]:
    if tuple(sorted(body)) != tuple(sorted(RECEIPT_FIELDS)):
        raise ValueError("receipt body has wrong field set")
    if body["verifier_key_id"] != key.key_id:
        raise ValueError("receipt key id mismatch")
    sig = key.sign(SIG_DST + canonical_json(body))
    return {"body": dict(body), "receipt_id": receipt_id(body), "signature": sig.hex()}

def check_receipt(receipt: Mapping[str, Any], trusted_verifier_keys: Mapping[str, bytes]) -> tuple[bool, str]:
    try:
        body = receipt["body"]
        if tuple(sorted(body)) != tuple(sorted(RECEIPT_FIELDS)):
            return False, "receipt.malformed"
        if receipt_id(body) != receipt["receipt_id"]:
            return False, "receipt.id_mismatch"
        kid = body["verifier_key_id"]
        if kid not in trusted_verifier_keys:
            return False, "receipt.untrusted_verifier_key"
        if not ed25519.verify(trusted_verifier_keys[kid], SIG_DST + canonical_json(body),
                              bytes.fromhex(receipt["signature"])):
            return False, "receipt.bad_signature"
        if body["verdict"] != "VERIFIED":
            return False, "receipt.not_verified"
        return True, "ok"
    except (KeyError, ValueError, TypeError):
        return False, "receipt.malformed"
