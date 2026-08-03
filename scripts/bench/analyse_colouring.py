"""Why does the coloured transport abort on some (mesh, npes) and not others?

Failing: core2 npes=8 (4x), farc npes=32 (3x).  Passing: core2 16/32/64, farc 8/16/64/128.
All die the same way — `Gloo collective permute failed: Connection closed by peer`.

Hypothesis: each colour class is a PARTIAL permutation (`shard_mesh.py:380`), so a round can
leave some ranks with nothing to send, nothing to receive, or both. XLA's gloo collective
permute may not tolerate that asymmetry, whereas a class covering every rank is safe.

This prints, per (mesh, npes), the class structure — no model run, no collectives.

    python scripts/bench/analyse_colouring.py core2:8,16 farc:32,64
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POOL = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1")
WORK = Path("/work/ab0995/a270088/fesom_jax_meshes")
MESHDIR = {"core2": Path("/home/a/a270088/port_jax/data/mesh_core2"),
           "farc": WORK / "mesh_farc", "dars": WORK / "mesh_dars", "ng5": WORK / "mesh_ng5"}


def analyse(name: str, npes: int) -> None:
    from fesom_jax import partit, shard_mesh
    from fesom_jax.mesh import load_mesh

    mesh = load_mesh(MESHDIR[name])
    part = partit.read_partition(POOL / name, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    ex = sm.exchange_coloured
    kinds = list(ex) if isinstance(ex, dict) else {"?": ex}
    for kind in kinds:
        c = ex[kind] if isinstance(ex, dict) else ex
        perms = c.perms
        print(f"\n{name} npes={npes} kind={kind}: {len(perms)} rounds")
        bad = 0
        for r, perm in enumerate(perms):
            senders = {d for d, _ in perm}
            recvers = {e for _, e in perm}
            idle_s, idle_r = npes - len(senders), npes - len(recvers)
            full = (idle_s == 0 and idle_r == 0)
            bad += 0 if full else 1
            flag = "FULL " if full else "PARTIAL"
            print(f"  round {r:>2}: {len(perm):>3} pairs  idle_send={idle_s:>3} "
                  f"idle_recv={idle_r:>3}  {flag}")
        print(f"  -> {bad}/{len(perms)} rounds are partial permutations")
        # a rank idle on BOTH sides in a round is the most asymmetric case
        both = sum(1 for perm in perms
                   for p in range(npes)
                   if p not in {d for d, _ in perm} and p not in {e for _, e in perm})
        print(f"  -> {both} (rank, round) slots where a rank neither sends nor receives")
        # Is the greedy colouring OPTIMAL? A bipartite edge-colouring needs exactly Delta
        # colours (Koenig); the greedy in shard_mesh.py:380 "may use a colour or two beyond
        # Delta", and every extra colour is one more ppermute round on EVERY exchange —
        # ~254 of them per step in the CG alone.
        import numpy as _np
        ss = _np.asarray(sm.exchange_ragged[kind].send_sizes) > 0
        delta = int(max(ss.sum(axis=1).max(), ss.sum(axis=0).max()))
        print(f"  -> rounds K={len(perms)}  vs  Delta={delta}  "
              f"({'OPTIMAL' if len(perms) == delta else f'{len(perms) - delta} round(s) of slack'})")


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        mesh, _, plist = spec.partition(":")
        for npes in (int(x) for x in plist.split(",")):
            try:
                analyse(mesh, npes)
            except Exception as e:                      # keep going across specs
                print(f"\n{mesh} npes={npes}: FAILED to analyse — {type(e).__name__}: {e}")
