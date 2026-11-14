"""Signed model manifests, canonical JSON, policy epochs and rollback protection.

A manifest is the publisher's authorization of ONE numeric model (C1, spec/manifest_schema.json).  Envelope:

    {"body": {...}, "signature": {"alg": "Ed25519", "key_id": ..., "value": hex}}

* Canonical encoding: UTF-8 JSON, keys sorted, separators (',', ':'), no floats, integers in the IEEE-754 safe range |v| <= 2^53 - 1, digests as lower-case hex strings.  For this restricted value domain the encoding coincides with RFC 8785 (JCS) for ASCII keys. * Signature: Ed25519 over  "EDGE-CERT/manifest/v1\\x00" || canonical(body). * manifest_id = SHA-256("EDGE-CERT/manifest-id/v1\\x00" || canonical(body)). * Rollback protection: the verifier keeps, per (publisher key, model family), the highest policy epoch it has accepted; manifests (and public instances) with a lower epoch are rejected ("monotonic-epoch").  Manifests may revoke earlier manifest ids.

This module is shared data/encoding code: the independent verifier imports it, it never imports prover code.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field as dfield
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto import ed25519

SIG_DST = b"EDGE-CERT/manifest/v1\x00"
ID_DST = b"EDGE-CERT/manifest-id/v1\x00"
MANIFEST_VERSION = "edgecert-manifest/v1"
MAX_SAFE_INT = (1 << 53) - 1
SCHEMA_PATH = Path(__file__).resolve().parents[3] / "spec" / "manifest_schema.json"

class ManifestError(Exception):
    """Raised for malformed, unsigned, untrusted, revoked or rolled-back manifests."""

# ------------------------------------------------------------ canonical JSON
def _check_canonical_domain(obj: Any, path: str = "$") -> None:
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, int):
        if abs(obj) > MAX_SAFE_INT:
            raise ManifestError(f"{path}: integer outside the canonical safe range; encode as hex string")
        return
    if isinstance(obj, float):
        raise ManifestError(f"{path}: floats are not allowed in canonical EDGE-CERT JSON")
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_canonical_domain(v, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ManifestError(f"{path}: non-string key")
            _check_canonical_domain(v, f"{path}.{k}")
        return
    raise ManifestError(f"{path}: unsupported type {type(obj).__name__}")

def canonical_json(obj: Any) -> bytes:
    _check_canonical_domain(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def digest_obj(tag: str, obj: Any) -> str:
    """Domain-separated digest of a canonical JSON object (graph, LUT, tokenizer, ...)."""
    return sha256_hex(tag.encode() + b"\x00" + canonical_json(obj))

# ------------------------------------------------------ JSON-schema (subset)
def _type_ok(v: Any, t: str) -> bool:
    return {
        "object": isinstance(v, dict),
        "array": isinstance(v, list),
        "string": isinstance(v, str),
        "integer": isinstance(v, int) and not isinstance(v, bool),
        "number": isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": isinstance(v, bool),
        "null": v is None,
    }[t]

def validate_schema(instance: Any, schema: Dict[str, Any], root: Optional[Dict[str, Any]] = None,
                    path: str = "$") -> List[str]:
    """Validate against the JSON-Schema subset used by spec/manifest_schema.json.

    Supported keywords: $ref (#/$defs/...), type, enum, const, required,
    properties, additionalProperties (bool), items, minItems, maxItems,
    minimum, maximum, pattern, minLength, anyOf.  Returns a list of errors.
    """
    root = root if root is not None else schema
    errs: List[str] = []
    if "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/$defs/"), ref
        return validate_schema(instance, root["$defs"][ref[len("#/$defs/"):]], root, path)
    if "anyOf" in schema:
        if not any(not validate_schema(instance, s, root, path) for s in schema["anyOf"]):
            errs.append(f"{path}: matches no anyOf branch")
        return errs
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_type_ok(instance, x) for x in ts):
            return [f"{path}: expected type {t}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} not in enum")
    if "const" in schema and instance != schema["const"]:
        errs.append(f"{path}: expected const {schema['const']!r}")
    if isinstance(instance, str):
        if "pattern" in schema and not re.fullmatch(schema["pattern"], instance):
            errs.append(f"{path}: does not match pattern {schema['pattern']}")
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errs.append(f"{path}: shorter than minLength")
    if isinstance(instance, int) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append(f"{path}: below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append(f"{path}: above maximum")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errs.append(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errs.append(f"{path}: more than maxItems")
        if "items" in schema:
            for i, v in enumerate(instance):
                errs += validate_schema(v, schema["items"], root, f"{path}[{i}]")
    if isinstance(instance, dict):
        for k in schema.get("required", []):
            if k not in instance:
                errs.append(f"{path}: missing required '{k}'")
        props = schema.get("properties", {})
        for k, v in instance.items():
            if k in props:
                errs += validate_schema(v, props[k], root, f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected property '{k}'")
    return errs

_SCHEMA_CACHE: Optional[Dict[str, Any]] = None

def load_schema() -> Optional[Dict[str, Any]]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None and SCHEMA_PATH.exists():
        _SCHEMA_CACHE = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE

# ------------------------------------------------------------ build / sign
def manifest_id(body: Dict[str, Any]) -> str:
    return sha256_hex(ID_DST + canonical_json(body))

def sign_manifest(body: Dict[str, Any], key: ed25519.SigningKey) -> Dict[str, Any]:
    body = copy.deepcopy(body)
    pub = body.get("publisher", {})
    if pub.get("public_key") != key.public.hex() or pub.get("key_id") != key.key_id:
        raise ManifestError("publisher block does not match the signing key")
    sig = key.sign(SIG_DST + canonical_json(body))
    return {"body": body, "signature": {"alg": "Ed25519", "key_id": key.key_id, "value": sig.hex()}}

def verify_signature(envelope: Dict[str, Any], trusted_public_key: bytes) -> bool:
    try:
        sig = envelope["signature"]
        if sig.get("alg") != "Ed25519":
            return False
        return ed25519.verify(trusted_public_key, SIG_DST + canonical_json(envelope["body"]),
                              bytes.fromhex(sig["value"]))
    except (KeyError, ValueError, ManifestError):
        return False

# ------------------------------------------------------------ trust policy
@dataclass
class AcceptedManifest:
    manifest_id: str
    body: Dict[str, Any]
    epoch: int

@dataclass
class TrustPolicy:
    """Verifier-side manifest authorization state (outside prover control).

    trusted_publishers: key_id -> raw Ed25519 public key (the authorization root).
    """
    trusted_publishers: Dict[str, bytes]
    epoch_floor: Dict[Tuple[str, str], int] = dfield(default_factory=dict)
    revoked: Set[str] = dfield(default_factory=set)
    accepted: Dict[str, AcceptedManifest] = dfield(default_factory=dict)
    require_schema: bool = True

    def check(self, envelope: Dict[str, Any], now: Optional[int] = None) -> Tuple[bool, str, Optional[str]]:
        """Pure check (no state change).  Returns (ok, reason, manifest_id)."""
        if not isinstance(envelope, dict) or "body" not in envelope or "signature" not in envelope:
            return False, "manifest.envelope_malformed", None
        body = envelope["body"]
        try:
            mid = manifest_id(body)
        except ManifestError as e:
            return False, f"manifest.not_canonical: {e}", None
        if self.require_schema:
            schema = load_schema()
            if schema is None:
                return False, "manifest.schema_missing", mid
            errs = validate_schema(envelope, schema)
            if errs:
                return False, "manifest.schema: " + "; ".join(errs[:3]), mid
        kid = envelope["signature"].get("key_id")
        if kid not in self.trusted_publishers:
            return False, "manifest.untrusted_publisher", mid
        pk = self.trusted_publishers[kid]
        if body["publisher"]["public_key"] != pk.hex() or body["publisher"]["key_id"] != kid:
            return False, "manifest.publisher_key_mismatch", mid
        if not verify_signature(envelope, pk):
            return False, "manifest.bad_signature", mid
        if mid in self.revoked:
            return False, "manifest.revoked", mid
        pol = body["policy"]
        floor = self.epoch_floor.get((kid, body["model"]["family"]), 0)
        if pol["policy_epoch"] < floor:
            return False, "manifest.rollback_stale_epoch", mid
        if now is not None and pol.get("not_after") is not None and now > pol["not_after"]:
            return False, "manifest.expired", mid
        return True, "ok", mid

    def accept(self, envelope: Dict[str, Any], now: Optional[int] = None) -> str:
        ok, reason, mid = self.check(envelope, now)
        if not ok:
            raise ManifestError(reason)
        body = envelope["body"]
        kid = envelope["signature"]["key_id"]
        key = (kid, body["model"]["family"])
        epoch = body["policy"]["policy_epoch"]
        if body["policy"]["rollback_protection"] == "monotonic-epoch":
            self.epoch_floor[key] = max(self.epoch_floor.get(key, 0), epoch)
        for r in body["policy"].get("revokes", []):
            self.revoked.add(r)
            self.accepted.pop(r, None)
        self.accepted[mid] = AcceptedManifest(mid, copy.deepcopy(body), epoch)
        return mid

    def current(self, mid: str) -> Tuple[bool, str]:
        """Is an already-accepted manifest still acceptable now (not revoked, not superseded)?"""
        am = self.accepted.get(mid)
        if am is None:
            return False, "manifest.unknown"
        if mid in self.revoked:
            return False, "manifest.revoked"
        kid = am.body["publisher"]["key_id"]
        floor = self.epoch_floor.get((kid, am.body["model"]["family"]), 0)
        if am.epoch < floor:
            return False, "manifest.rollback_stale_epoch"
        return True, "ok"

def authorized_model_commitments(body: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Map every model commitment this manifest authorizes -> adapter record (None for base)."""
    out: Dict[str, Optional[Dict[str, Any]]] = {body["model"]["model_commitment"]: None}
    for a in body.get("adapters", []):
        out[a["combined_model_commitment"]] = a
    return out
