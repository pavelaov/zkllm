"""Replay a reference trace through the field-level relation checker.

For every requant, norm, softmax, activation-table, embedding and argmax gadget in a T0 trace, re-check the spec constraints over F_p with the range-check widths of the TileContracts (non-elided lowering).  Matmul accumulators are proven by the backend sumcheck and are only range-checked here (dynamic checker), so they are not replayed.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..field import P as GOLDILOCKS_P
from ..reference import relation as rel
from .tile_contract import TileContract

def _bits(tc: TileContract, prefix: str) -> Dict[str, int]:
    return {k.name.split(".", 1)[1]: k.bits for k in tc.checks if k.name.startswith(prefix + ".")}

def replay_trace(trace, contracts: Dict[str, TileContract], im, p: int = GOLDILOCKS_P) -> Dict[str, int]:
    """Raises relation.RelationError on the first failing constraint; returns counts checked."""
    counts = {"rq_elements": 0, "norm_rows": 0, "softmax_rows": 0, "act_lookups": 0,
              "embed_rows": 0, "argmax_rows": 0}
    exp_set = rel.table_set(im.T_exp, 0, p)
    act_sets = {}
    emb_rows = {tuple(int(x) for x in row) for row in im.P["E"]}
    for op in trace:
        tc = contracts[op.op_id]
        layer = int(op.op_id[1:op.op_id.index(".")]) if op.op_id.startswith("L") else None
        for g in op.gadgets:
            if g.kind == "rq":
                b = _bits(tc, "rq")
                v = g.aux["v"]
                M = np.broadcast_to(g.aux["M"], v.shape)
                add = g.aux.get("add")
                rel.check_requant(v, M, g.params["s"], g.params["lo"], g.params["hi"], g.committed, b,
                                  add=add, p=p)
                counts["rq_elements"] += int(v.size)
            elif g.kind == "norm":
                b = _bits(tc, "nm")
                r = g.aux["r"]
                prm = g.params
                for i in range(r.shape[0]):
                    w = {k: (a[i] if a.ndim >= 1 else a) for k, a in g.committed.items()}
                    rel.check_norm_row(r[i], prm["kind"], prm["F"], prm["G"], prm["eps_int"], w, b, p=p)
                counts["norm_rows"] += int(r.shape[0])
            elif g.kind == "softmax":
                b = _bits(tc, "sm")
                t, allowed = g.aux["t"], g.aux["allowed"]
                H, n, _ = t.shape
                pos = 0
                for h in range(H):
                    for i in range(n):
                        js = np.nonzero(allowed[h, i])[0]
                        k = len(js)
                        w = {nm: g.committed[nm][pos:pos + k] for nm in ("c", "E", "P", "rho")}
                        w["m"], w["D"] = g.committed["m"][h, i], g.committed["D"][h, i]
                        rel.check_softmax_row(t[h, i, js], w, im.T_exp, im.pol.Fp, b, p=p, tset=exp_set)
                        pos += k
                        counts["softmax_rows"] += 1
            elif g.kind == "act":
                table = im.layers[layer]["T_act"]
                if layer not in act_sets:
                    act_sets[layer] = rel.table_set(table, g.params["lo"], p)
                rel.check_table(g.aux["a"], g.committed["out"], table, g.params["lo"], p, tset=act_sets[layer])
                counts["act_lookups"] += int(g.aux["a"].size)
            elif g.kind == "embed":
                for row in g.committed["row"]:
                    if tuple(int(x) for x in row) not in emb_rows:
                        raise rel.RelationError("embedding row not in the committed table")
                counts["embed_rows"] += int(g.committed["row"].shape[0])
            elif g.kind == "argmax":
                bits_d = _bits(tc, "am")["d"]
                l = g.aux["l"]
                for i in range(l.shape[0]):
                    w = {"delta": g.committed["delta"][i], "tau": g.committed["tau"][i],
                         "ltau": g.committed["ltau"][i]}
                    rel.check_argmax(l[i], w, bits_d, p=p)
                counts["argmax_rows"] += int(l.shape[0])
    return counts
