"""Unit tests: canonical JSON, manifest schema, signatures, policy epochs, rollback and revocation."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edgecert.crypto.ed25519 import SigningKey  # noqa: E402
from edgecert.runtime.manifest import (ManifestError, TrustPolicy, authorized_model_commitments,  # noqa: E402
                                       canonical_json, load_schema, manifest_id, sign_manifest, validate_schema)
from edgecert.prover_adapters.toy_model import ToyConfig, build_model, manifest_body  # noqa: E402

CFG = ToyConfig(d_emb=4, hidden=(8,), vocab=8, max_position=16)

class TestCanonicalJSON(unittest.TestCase):
    def test_sorted_compact(self):
        self.assertEqual(canonical_json({"b": 1, "a": [True, None, "x"]}), b'{"a":[true,null,"x"],"b":1}')

    def test_rejects_floats_and_big_ints(self):
        with self.assertRaises(ManifestError):
            canonical_json({"x": 1.5})
        with self.assertRaises(ManifestError):
            canonical_json({"x": 1 << 60})

class TestManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = SigningKey()
        cls.m = build_model(CFG)
        cls.env1 = sign_manifest(manifest_body(cls.m, cls.key, epoch=1), cls.key)
        cls.env2 = sign_manifest(manifest_body(cls.m, cls.key, epoch=2), cls.key)

    def trust(self):
        return TrustPolicy({self.key.key_id: self.key.public})

    def test_schema_valid(self):
        schema = load_schema()
        self.assertIsNotNone(schema, "spec/manifest_schema.json missing")
        self.assertEqual(validate_schema(self.env1, schema), [])
        bad = copy.deepcopy(self.env1)
        del bad["body"]["decoding"]
        bad["body"]["extra"] = 1
        errs = validate_schema(bad, schema)
        self.assertTrue(any("decoding" in e for e in errs) and any("extra" in e for e in errs))

    def test_accept_and_signature(self):
        t = self.trust()
        mid = t.accept(self.env1)
        self.assertEqual(mid, manifest_id(self.env1["body"]))
        forged = copy.deepcopy(self.env1)
        forged["body"]["model"]["model_commitment"] = "00" * 32
        ok, why, _ = t.check(forged)
        self.assertFalse(ok)
        self.assertEqual(why, "manifest.bad_signature")

    def test_untrusted_publisher(self):
        other = SigningKey()
        env = sign_manifest(manifest_body(self.m, other, epoch=5), other)
        ok, why, _ = self.trust().check(env)
        self.assertEqual((ok, why), (False, "manifest.untrusted_publisher"))

    def test_signing_key_must_match_publisher_block(self):
        other = SigningKey()
        with self.assertRaises(ManifestError):
            sign_manifest(manifest_body(self.m, self.key), other)

    def test_rollback_protection(self):
        t = self.trust()
        mid1 = t.accept(self.env1)
        t.accept(self.env2)
        ok, why, _ = t.check(self.env1)
        self.assertEqual((ok, why), (False, "manifest.rollback_stale_epoch"))
        self.assertEqual(t.current(mid1), (False, "manifest.rollback_stale_epoch"))

    def test_revocation(self):
        t = self.trust()
        mid1 = t.accept(self.env1)
        env3 = sign_manifest(manifest_body(self.m, self.key, epoch=1, revokes=[mid1], issued_at=1790000001), self.key)
        t.accept(env3)
        self.assertEqual(t.current(mid1), (False, "manifest.unknown"))
        ok, why, _ = t.check(self.env1)
        self.assertEqual((ok, why), (False, "manifest.revoked"))

    def test_expiry(self):
        env = sign_manifest(manifest_body(self.m, self.key, epoch=3, not_after=100), self.key)
        ok, why, _ = self.trust().check(env, now=101)
        self.assertEqual((ok, why), (False, "manifest.expired"))

    def test_authorized_commitments(self):
        auth = authorized_model_commitments(self.env1["body"])
        self.assertEqual(list(auth), [self.env1["body"]["model"]["model_commitment"]])

if __name__ == "__main__":
    unittest.main()
