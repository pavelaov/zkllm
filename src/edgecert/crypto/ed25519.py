"""Pure-Python Ed25519 (RFC 8032, Section 5.1), PROTOTYPE ONLY.

Used by EDGE-CERT for publisher manifest signatures, verifier receipts and audit checkpoints.  Correctness is checked against the RFC 8032 Section 7.1 test vectors (tests/test_security_crypto.py) and cross-checked against OpenSSL (docs/notes/security.md).

Security caveats (do not overclaim):
  * Big-integer arithmetic in CPython is NOT constant time; scalar
    multiplication uses a data-dependent double-and-add ladder.  This module
    offers no side-channel resistance and must not hold production keys.
  * Verification uses the cofactored equation [8][S]B = [8]R + [8][k]A'
    (RFC 8032 Section 5.1.7, step 3).  Non-canonical point encodings
    (y >= p, or x = 0 with the sign bit set) and S >= L are rejected.
"""
from __future__ import annotations

import hashlib
import secrets
from typing import Optional, Tuple

# Curve constants (RFC 8032 Section 5.1)
P25519 = 2 ** 255 - 19
L_ORDER = 2 ** 252 + 27742317777372353535851937790883648493
D = (-121665 * pow(121666, P25519 - 2, P25519)) % P25519
SQRT_M1 = pow(2, (P25519 - 1) // 4, P25519)

Point = Tuple[int, int, int, int]  # extended homogeneous (X, Y, Z, T), x = X/Z, y = Y/Z, xy = T/Z

def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()

def _inv(x: int) -> int:
    return pow(x, P25519 - 2, P25519)

def point_add(P1: Point, P2: Point) -> Point:
    """RFC 8032 Section 5.1.4 addition (complete for a = -1 twisted Edwards)."""
    A = (P1[1] - P1[0]) * (P2[1] - P2[0]) % P25519
    B = (P1[1] + P1[0]) * (P2[1] + P2[0]) % P25519
    C = 2 * P1[3] * P2[3] * D % P25519
    Dd = 2 * P1[2] * P2[2] % P25519
    E, F, G, H = B - A, Dd - C, Dd + C, B + A
    return (E * F % P25519, G * H % P25519, F * G % P25519, E * H % P25519)

def point_double(P1: Point) -> Point:
    return point_add(P1, P1)

IDENTITY: Point = (0, 1, 1, 0)

def point_mul(s: int, P1: Point) -> Point:
    """Double-and-add (NOT constant time; see module docstring)."""
    Q = IDENTITY
    while s > 0:
        if s & 1:
            Q = point_add(Q, P1)
        P1 = point_double(P1)
        s >>= 1
    return Q

def point_equal(P1: Point, P2: Point) -> bool:
    # x1/z1 == x2/z2 and y1/z1 == y2/z2
    if (P1[0] * P2[2] - P2[0] * P1[2]) % P25519 != 0:
        return False
    if (P1[1] * P2[2] - P2[1] * P1[2]) % P25519 != 0:
        return False
    return True

def _recover_x(y: int, sign: int) -> Optional[int]:
    """RFC 8032 Section 5.1.3 decoding step 2-4."""
    if y >= P25519:
        return None
    x2 = (y * y - 1) * _inv(D * y * y + 1) % P25519
    if x2 == 0:
        if sign:
            return None
        return 0
    x = pow(x2, (P25519 + 3) // 8, P25519)
    if (x * x - x2) % P25519 != 0:
        x = x * SQRT_M1 % P25519
    if (x * x - x2) % P25519 != 0:
        return None
    if (x & 1) != sign:
        x = P25519 - x
    return x

_GY = 4 * _inv(5) % P25519
_GX = _recover_x(_GY, 0)
BASE: Point = (_GX, _GY, 1, _GX * _GY % P25519)

def point_compress(P1: Point) -> bytes:
    zinv = _inv(P1[2])
    x = P1[0] * zinv % P25519
    y = P1[1] * zinv % P25519
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")

def point_decompress(s: bytes) -> Optional[Point]:
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % P25519)

def _secret_expand(secret: bytes) -> Tuple[int, bytes]:
    if len(secret) != 32:
        raise ValueError("Ed25519 secret key must be 32 bytes")
    h = _sha512(secret)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]

def public_key(secret: bytes) -> bytes:
    a, _ = _secret_expand(secret)
    return point_compress(point_mul(a, BASE))

def sign(secret: bytes, msg: bytes) -> bytes:
    a, prefix = _secret_expand(secret)
    A = point_compress(point_mul(a, BASE))
    r = int.from_bytes(_sha512(prefix + msg), "little") % L_ORDER
    R = point_compress(point_mul(r, BASE))
    k = int.from_bytes(_sha512(R + A + msg), "little") % L_ORDER
    S = (r + k * a) % L_ORDER
    return R + int.to_bytes(S, 32, "little")

def verify(public: bytes, msg: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    A = point_decompress(public)
    if A is None:
        return False
    Rs = signature[:32]
    R = point_decompress(Rs)
    if R is None:
        return False
    S = int.from_bytes(signature[32:], "little")
    if S >= L_ORDER:
        return False
    k = int.from_bytes(_sha512(Rs + public + msg), "little") % L_ORDER
    lhs = point_mul(8 * S, BASE)
    rhs = point_add(point_mul(8, R), point_mul(8 * k, A))
    return point_equal(lhs, rhs)

def generate_secret() -> bytes:
    """Fresh 32-byte secret from the OS CSPRNG."""
    return secrets.token_bytes(32)

class SigningKey:
    """Small convenience wrapper: key_id = first 16 hex chars of SHA-256(pk)."""

    def __init__(self, secret: Optional[bytes] = None):
        self._secret = secret if secret is not None else generate_secret()
        self.public = public_key(self._secret)
        self.key_id = hashlib.sha256(b"EDGE-CERT/key-id/v1" + self.public).hexdigest()[:16]

    def sign(self, msg: bytes) -> bytes:
        return sign(self._secret, msg)

    def __repr__(self) -> str:  # never print the secret
        return f"SigningKey(key_id={self.key_id})"

def key_id_of(public: bytes) -> str:
    return hashlib.sha256(b"EDGE-CERT/key-id/v1" + public).hexdigest()[:16]
