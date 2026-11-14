"""SHA-256 Merkle-tree commitments for integer tensors, with openings.

Leaf and node hashing are domain separated (0x00 leaf, 0x01 node, 0x02 empty pad) so that a leaf can never be reinterpreted as an inner node.  Tensor values are signed integers embedded into the Goldilocks field (edgecert.field.lift_signed) and encoded as u64 little endian, i.e. exactly the field element that the proof relation talks about.

Two modes:
  * unsalted (public tensors, e.g. public weights in the transparent
    reference mode): binding only;
  * salted: every leaf carries a 32-byte salt; with fresh salts the root is
    hiding in the random-oracle model and an opening reveals only the opened
    leaves (plus their salts).

The tensor commitment binds the shape:  tensor_root = H(0x03 || shape || merkle_root). The model commitment is a Merkle tree over (name, tensor_root) leaves in canonical (sorted) name order, so each tensor can be opened individually.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field as dfield
from typing import Dict, List, Optional, Sequence, Tuple

from ..field import P

LEAF = b"\x00"
NODE = b"\x01"
EMPTY = hashlib.sha256(b"\x02EDGE-CERT/merkle/empty/v1").digest()
TENSOR = b"\x03"
MODEL_DST = b"EDGE-CERT/model-commitment/v1"

def _h(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()

def encode_value(v: int) -> bytes:
    return (v % P).to_bytes(8, "little")

def leaf_hash(index: int, payload: bytes, salt: bytes = b"") -> bytes:
    return _h(LEAF, index.to_bytes(8, "little"), len(payload).to_bytes(4, "little"), payload, salt)

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def build_levels(leaves: Sequence[bytes]) -> List[List[bytes]]:
    n = _next_pow2(max(1, len(leaves)))
    level = list(leaves) + [EMPTY] * (n - len(leaves))
    levels = [level]
    while len(level) > 1:
        level = [_h(NODE, level[2 * i], level[2 * i + 1]) for i in range(len(level) // 2)]
        levels.append(level)
    return levels

def auth_path(levels: List[List[bytes]], index: int) -> List[bytes]:
    path = []
    for level in levels[:-1]:
        path.append(level[index ^ 1])
        index >>= 1
    return path

def verify_path(root: bytes, leaf: bytes, index: int, path: Sequence[bytes], n_leaves_padded: int) -> bool:
    if n_leaves_padded != _next_pow2(n_leaves_padded) or len(path) != n_leaves_padded.bit_length() - 1:
        return False
    if not 0 <= index < n_leaves_padded:
        return False
    cur = leaf
    for sib in path:
        cur = _h(NODE, sib, cur) if index & 1 else _h(NODE, cur, sib)
        index >>= 1
    return hmac.compare_digest(cur, root)

def flatten(values) -> Tuple[List[int], Tuple[int, ...]]:
    """Row-major flatten of a nested list of ints; returns (flat, shape)."""
    shape: List[int] = []
    v = values
    while isinstance(v, (list, tuple)):
        shape.append(len(v))
        v = v[0] if len(v) else None
    flat: List[int] = []

    def rec(x, depth):
        if depth == len(shape):
            if not isinstance(x, int) or isinstance(x, bool):
                raise TypeError("tensor entries must be int")
            flat.append(x)
            return
        if len(x) != shape[depth]:
            raise ValueError("ragged tensor")
        for y in x:
            rec(y, depth + 1)

    rec(values, 0)
    return flat, tuple(shape)

def shape_bytes(shape: Sequence[int]) -> bytes:
    return len(shape).to_bytes(4, "little") + b"".join(int(s).to_bytes(8, "little") for s in shape)

@dataclass
class TensorCommitment:
    root: bytes                 # H(0x03 || shape || merkle_root)
    merkle_root: bytes
    shape: Tuple[int, ...]
    levels: List[List[bytes]] = dfield(repr=False, default_factory=list)
    salts: Optional[List[bytes]] = dfield(repr=False, default=None)

    def open(self, flat_index: int, value: int) -> dict:
        return {
            "index": flat_index,
            "value": value,
            "salt": self.salts[flat_index].hex() if self.salts else "",
            "path": [p.hex() for p in auth_path(self.levels, flat_index)],
            "n_padded": len(self.levels[0]),
        }

def commit_tensor(values, salted: bool = False, salts: Optional[List[bytes]] = None) -> TensorCommitment:
    flat, shape = flatten(values)
    if salted and salts is None:
        salts = [secrets.token_bytes(32) for _ in flat]
    leaves = [leaf_hash(i, encode_value(v), salts[i] if salted else b"") for i, v in enumerate(flat)]
    levels = build_levels(leaves)
    mroot = levels[-1][0]
    root = _h(TENSOR, shape_bytes(shape), mroot)
    return TensorCommitment(root, mroot, shape, levels, salts if salted else None)

def tensor_root(values) -> bytes:
    return commit_tensor(values).root

def verify_tensor_opening(tc_root: bytes, shape: Sequence[int], merkle_root: bytes, opening: dict) -> bool:
    """Check that `opening` opens position opening['index'] of a committed tensor."""
    if not hmac.compare_digest(_h(TENSOR, shape_bytes(shape), merkle_root), tc_root):
        return False
    salt = bytes.fromhex(opening.get("salt", ""))
    leaf = leaf_hash(opening["index"], encode_value(opening["value"]), salt)
    return verify_path(merkle_root, leaf, opening["index"], [bytes.fromhex(p) for p in opening["path"]],
                       opening["n_padded"])

# ------------------------------------------------------------ model commitment
def model_leaf(name: str, troot: bytes) -> bytes:
    nb = name.encode("utf-8")
    return leaf_hash(0, len(nb).to_bytes(4, "little") + nb + troot)

@dataclass
class ModelCommitment:
    commitment: bytes
    names: List[str]
    levels: List[List[bytes]] = dfield(repr=False, default_factory=list)
    tensor_roots: Dict[str, bytes] = dfield(default_factory=dict)

    def open(self, name: str) -> dict:
        i = self.names.index(name)
        return {"name": name, "position": i, "tensor_root": self.tensor_roots[name].hex(),
                "path": [p.hex() for p in auth_path(self.levels, i)], "n_padded": len(self.levels[0]),
                "count": len(self.names)}

def commit_model(tensors: Dict[str, object]) -> ModelCommitment:
    names = sorted(tensors)
    roots = {n: tensor_root(tensors[n]) for n in names}
    levels = build_levels([model_leaf(n, roots[n]) for n in names])
    c = _h(MODEL_DST, len(names).to_bytes(4, "little"), levels[-1][0])
    return ModelCommitment(c, names, levels, roots)

def verify_model_opening(model_commitment: bytes, opening: dict, values) -> bool:
    """Recompute the tensor root from the revealed values and check its path."""
    troot = tensor_root(values)
    if troot.hex() != opening["tensor_root"]:
        return False
    # recompute the Merkle root implied by the path, then the model commitment
    leaf = model_leaf(opening["name"], troot)
    idx = opening["position"]
    cur = leaf
    path = [bytes.fromhex(p) for p in opening["path"]]
    if opening["n_padded"] != _next_pow2(opening["count"]) or len(path) != opening["n_padded"].bit_length() - 1:
        return False
    for sib in path:
        cur = _h(NODE, sib, cur) if idx & 1 else _h(NODE, cur, sib)
        idx >>= 1
    c = _h(MODEL_DST, opening["count"].to_bytes(4, "little"), cur)
    return hmac.compare_digest(c, model_commitment)
