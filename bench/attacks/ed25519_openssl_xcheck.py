"""Cross-check edgecert.crypto.ed25519 against the local OpenSSL CLI (no downloads).

Signs 16 random messages (1..556 bytes) under random seeds with both implementations, compares signatures and public keys, and lets OpenSSL verify our signatures. Usage (repository root): python bench/attacks/ed25519_openssl_xcheck.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from edgecert.crypto import ed25519 as e  # noqa: E402

PKCS8_ED25519_PREFIX = bytes.fromhex("302e020100300506032b657004220420")

def main() -> int:
    ossl = shutil.which("openssl")
    if ossl is None:
        print("openssl not found; cross-check NOT_RUN")
        return 2
    d = tempfile.mkdtemp()
    ok = bad = 0
    try:
        for i in range(16):
            seed, msg = os.urandom(32), os.urandom(1 + i * 37)
            kp, mp, sp, pp = (os.path.join(d, n) for n in ("k.der", "m.bin", "s.bin", "p.der"))
            open(kp, "wb").write(PKCS8_ED25519_PREFIX + seed)
            open(mp, "wb").write(msg)
            subprocess.run([ossl, "pkeyutl", "-sign", "-inkey", kp, "-keyform", "DER", "-rawin", "-in", mp,
                            "-out", sp], check=True)
            subprocess.run([ossl, "pkey", "-in", kp, "-inform", "DER", "-pubout", "-outform", "DER", "-out", pp],
                           check=True)
            so, pub = open(sp, "rb").read(), open(pp, "rb").read()[-32:]
            mine = e.sign(seed, msg)
            open(sp, "wb").write(mine)
            v = subprocess.run([ossl, "pkeyutl", "-verify", "-inkey", kp, "-keyform", "DER", "-rawin", "-in", mp,
                                "-sigfile", sp], capture_output=True, text=True)
            good = so == mine and e.public_key(seed) == pub and e.verify(pub, msg, so) and v.returncode == 0
            ok += good
            bad += not good
    finally:
        shutil.rmtree(d, ignore_errors=True)
    ver = subprocess.run([ossl, "version"], capture_output=True, text=True).stdout.strip()
    print(f"{ver}: {ok} agree, {bad} disagree")
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
