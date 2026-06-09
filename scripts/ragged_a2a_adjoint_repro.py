"""Isolation repro for the JAX `lax.ragged_all_to_all` reverse-mode autodiff bug
(Phase 8b B.0c follow-up; see docs/JAX_RAGGED_A2A_BUG.md).

Tests the ADJOINT IDENTITY of a linear map f:  <f(x), y> == <x, fᵀ(y)>  (fᵀ via jax.vjp).
f's forward is exact (verified), so if the identity FAILS the registered TRANSPOSE is wrong.

Two functions, to isolate JAX vs our composition:
  (A) BARE  lax.ragged_all_to_all (operand → recv) — NO gather/scatter/where around it.
  (B) FULL  halo.run_halo_exchange_ragged (the gather → a2a → gather-back+where composition).
If (A) FAILS  → JAX's `ragged_all_to_all` transpose is wrong → file the upstream bug.
If (A) HOLDS but (B) FAILS → the bug is in OUR composition/offset maps → fix on our side.

GPU-only (ragged_all_to_all is unimplemented on XLA:CPU).
    <env-py> scripts/ragged_a2a_adjoint_repro.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

jax.config.update("jax_enable_x64", True)

CORE2_MESH = Path(__file__).resolve().parents[1] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")


def _fold0(a):
    a = np.asarray(a)
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


def main(npes=2, kind="nod"):
    if jax.devices()[0].platform == "cpu":
        print("SKIP: lax.ragged_all_to_all is unimplemented on XLA:CPU — needs GPU (NCCL)")
        return
    if len(jax.devices()) < npes:
        print(f"SKIP: need {npes} devices, have {len(jax.devices())}")
        return

    from fesom_jax import partit, shard_mesh, halo
    from fesom_jax.mesh import load_mesh

    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), part)
    rmap = sm.exchange_ragged[kind]
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    nl = sm.nl
    send_max, recv_max = int(rmap.send_max), int(rmap.recv_max)
    so = jnp.asarray(_fold0(rmap.send_offsets).astype(np.int32))
    ss = jnp.asarray(_fold0(rmap.send_sizes).astype(np.int32))
    oo = jnp.asarray(_fold0(rmap.out_offsets).astype(np.int32))
    rs = jnp.asarray(_fold0(rmap.recv_sizes).astype(np.int32))
    spec = P("p")
    rng = np.random.default_rng(0)

    def adjoint_gap(f, in_shape, out_shape, label):
        x = jnp.asarray(rng.standard_normal(in_shape))
        y = jnp.asarray(rng.standard_normal(out_shape))
        fx, vjp_fn = jax.vjp(f, x)
        xbar = vjp_fn(y)[0]
        lhs = float(jnp.sum(fx * y))            # <f(x), y>  (uses the exact FORWARD)
        rhs = float(jnp.sum(x * xbar))          # <x, fᵀ(y)> (uses the suspect TRANSPOSE)
        gap = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
        print(f"[{label}] <f(x),y>={lhs:+.6e}  <x,fT(y)>={rhs:+.6e}  rel|Δ|={gap:.3e}  "
              f"{'WRONG' if gap > 1e-9 else 'ok'}")
        return gap

    # (A) BARE primitive: operand [P*send_max, nl] -> recv [P*recv_max, nl]
    def f_bare(operand):
        body = lambda op, a, b, c, d: lax.ragged_all_to_all(
            op, jnp.zeros((recv_max, nl)), a, b, c, d, axis_name="p")
        fn = jax.shard_map(body, mesh=jmesh, in_specs=(spec,) * 5, out_specs=spec)
        return fn(operand, so, ss, oo, rs)

    gA = adjoint_gap(f_bare, (npes * send_max, nl), (npes * recv_max, nl),
                     "A bare ragged_all_to_all")

    # (B) FULL composition: field [P, Lmax, nl] -> field [P, Lmax, nl]
    Lmax = sm.Lmax[kind]
    def f_comp(field):
        return halo.run_halo_exchange_ragged(field, rmap, jmesh)
    gB = adjoint_gap(f_comp, (npes, Lmax, nl), (npes, Lmax, nl),
                     "B full composition")

    print("\n=== VERDICT ===")
    if gA > 1e-9:
        print("BARE primitive adjoint FAILS → JAX lax.ragged_all_to_all transpose is WRONG → file upstream.")
    elif gB > 1e-9:
        print("BARE ok but COMPOSITION fails → bug is in OUR gather/scatter/where or offset maps → fix our side.")
    else:
        print("Both adjoint identities hold — transpose is correct here (re-check the B.0c grad comparison).")


if __name__ == "__main__":
    for npes in (2, 4):
        print(f"\n######## npes = {npes} ########")
        main(npes)
