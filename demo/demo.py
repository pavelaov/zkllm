"""Minimal end-to-end demo of EDGE-CERT.

Runs four independent checks, one per core module of the paper: 1. signed manifest contracts  - sign a manifest, authorize it, reject a bad signature 2. TILECONTRACT               - lower GPT-2 to tile contracts and certify every op statically 3. CERTSCHED                  - plan a legal schedule under a device-memory cap 4. SUMFOLD + verifier         - transparent proof accepted, tampered proof and replay rejected

Usage (from the artifact root):
    export PYTHONPATH=$PWD/src   # Windows: set PYTHONPATH to <root>\\src
    python demo/demo.py
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgecert.compiler import graph as G  # noqa: E402
from edgecert.compiler import tile_contract as TC  # noqa: E402
from edgecert.compiler.model_shapes import MODELS  # noqa: E402
from edgecert.crypto.ed25519 import SigningKey  # noqa: E402
from edgecert.field import P  # noqa: E402
from edgecert.prover_adapters.toy_model import ToyConfig, build_model, manifest_body  # noqa: E402
from edgecert.prover_adapters.transparent import (  # noqa: E402
    HostSumcheck, SumFoldLaneEmulator, TransparentProver)
from edgecert.runtime.dag import ModelShape, build_dag  # noqa: E402
from edgecert.runtime.manifest import TrustPolicy, sign_manifest, verify_signature  # noqa: E402
from edgecert.runtime.scheduler import DeviceProfile, plan_certsched, run_policy  # noqa: E402
from edgecert.runtime.session import PrivateRequest, SessionContext  # noqa: E402
from edgecert.verifier import instance as I  # noqa: E402
from edgecert.verifier.transparent_verifier import Expectation, Verifier  # noqa: E402

TOY = ToyConfig(d_emb=8, hidden=(16,), vocab=16, max_position=32, eos_bias=0)
TINY = ModelShape("tiny", n_layers=2, d_model=16, n_heads=2, n_kv_heads=1, head_dim=8, d_ff=32,
                  vocab=64, gated_mlp=True, qk_norm=True, tied_embeddings=True, native_ctx=4096)
PROFILE = DeviceProfile("demo", int8_ops_per_s=1e14, mem_Bps=5e11, sc_elems_per_s=1e10,
                        commit_Bps=4e9, hash_Bps=2e10, link_Bps=2.5e10, ssd_read_Bps=5e9,
                        ssd_write_Bps=3e9, aes_Bps=1e10)

def step_manifest():
    key = SigningKey()
    model = build_model(TOY)
    env = sign_manifest(manifest_body(model, key, epoch=1), key)
    verifier = Verifier(TrustPolicy({key.key_id: key.public}))
    mid = verifier.authorize_manifest(env)
    assert verify_signature(env, key.public)
    bad = copy.deepcopy(env)
    val = bad["signature"]["value"]
    bad["signature"]["value"] = ("00" if val[:2] != "00" else "01") + val[2:]
    try:
        verifier.authorize_manifest(bad)
        raise AssertionError("bad signature was accepted")
    except Exception as exc:  # signature check must reject
        print(f"    bad signature rejected: {type(exc).__name__}")
    print(f"  [1/4] signed manifest OK: manifest_id {mid[:16]}..., signature rejected when forged")

def step_contracts():
    ops = G.build_graph(MODELS["gpt2_124m"])
    cons = TC.lower(ops)
    TC.assert_static(cons, P)
    kinds = sorted({tc.kind for tc in cons.values()})
    sample = next(tc for tc in cons.values() if tc.kind == "linear")
    bb_bad = sum(1 for tc in cons.values() if TC.static_check(tc, TC.BABYBEAR_P))
    print(f"  [2/4] TILECONTRACT OK: {len(cons)} contracts, {len(kinds)} kinds "
          f"({', '.join(kinds)}), goldilocks static check passed")
    print(f"        sample {sample.op_id}: gadget {sample.proof_gadget_id}, "
          f"acc width {json.dumps(sample.accumulator_width, default=str)}")
    print(f"        babybear (31-bit) counterexample: {bb_bad} contracts would violate the field")

def step_schedule():
    dag = build_dag(TINY, 16, 4)
    keep = run_policy(dag, "KEEP_ALL", PROFILE)
    cap = dag.static_bytes() + (keep.check.peak_device - dag.static_bytes()) // 3
    r = plan_certsched(dag, PROFILE, cap)
    assert r.check.legal, r.check.violations[:3]
    print(f"  [3/4] CERTSCHED OK: legal={r.check.legal}, fits={r.check.fits}, "
          f"peak {r.check.peak_device / 2**20:.1f} MiB <= cap {cap / 2**20:.1f} MiB, "
          f"overhead {r.time['overhead_s']:.4f} s")

def _setup(output_mode="tokens"):
    key = SigningKey()
    model = build_model(TOY)
    env = sign_manifest(manifest_body(model, key, epoch=1), key)
    verifier = Verifier(TrustPolicy({key.key_id: key.public}))
    mid = verifier.authorize_manifest(env)
    prompt, max_new = [3, 5, 7], 5
    ic, io = I.commit_input(prompt, max_new, model.tokenizer["id"])
    ctx = SessionContext(mid, "ab" * 8, "cd" * 32, ic.hex(), I.GENESIS_RECEIPT_ID,
                         I.GENESIS_ANCHOR, 1, output_mode)
    priv = PrivateRequest(prompt, max_new, io.randomness)
    exp = Expectation(mid, ctx.session_id, ctx.requester_nonce, ctx.input_commitment,
                      I.GENESIS_RECEIPT_ID, output_mode)
    return model, env, verifier, ctx, priv, exp

def step_proof():
    model, env, verifier, ctx, priv, exp = _setup()
    prover = TransparentProver(model, env, backend=SumFoldLaneEmulator(4))
    out = prover.infer(ctx, priv)
    pf = prover.prove(ctx, priv, out)
    r = verifier.verify(pf, exp)
    assert r.accepted, (r.failed_check, r.reason)
    r2 = verifier.verify(copy.deepcopy(pf), exp)
    assert not r2.accepted and r2.failed_check == "session.nonce_fresh", r2.failed_check
    model, env, verifier, ctx, priv, exp = _setup()
    prover = TransparentProver(model, env, backend=HostSumcheck())
    pf = prover.prove(ctx, priv, prover.infer(ctx, priv))
    bad = copy.deepcopy(pf)
    msg = bad["sumchecks"][0]["msgs"][2]
    msg[1] = f"{int(msg[1], 16) ^ 1:032x}"
    r3 = verifier.verify(bad, exp)
    assert not r3.accepted and r3.failed_check.startswith("sumcheck"), r3.failed_check
    print(f"  [4/4] SUMFOLD + verifier OK: proof accepted ({len(pf['sumchecks'])} sumchecks, "
          f"label {pf['label']}), tamper -> {r3.failed_check}, replay -> {r2.failed_check}")

def main():
    print("EDGE-CERT minimal demo")
    step_manifest()
    step_contracts()
    step_schedule()
    step_proof()
    print("DEMO PASS")

if __name__ == "__main__":
    main()
