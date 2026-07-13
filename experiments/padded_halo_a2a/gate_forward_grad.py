"""Oracle gates for the padded-a2a halo exchange, on the REAL CORE2 partitions.

For every entity kind (nod/elem/edge) and several trailing shapes:
  FORWARD:  padded exchange == all_gather exchange on every VALID lane
  GRAD:     d(sum(w . exchange(x)))/dx identical between the two paths
            (w zeroed on invalid/pad lanes, where the two paths legitimately differ)

The all_gather path is the project's proven AD oracle (docs/JAX_RAGGED_A2A_BUG.md).
Run:  XLA_FLAGS=--xla_force_host_platform_device_count=N  python gate_forward_grad.py N
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from fesom_jax import halo, partit, shard_mesh
from fesom_jax.mesh import load_mesh

import padded_halo

MESH_DIR = pathlib.Path("~/fesom-data/core2_mesh_ic/mesh_core2").expanduser()
DIST_DIR = pathlib.Path("~/fesom-data/core2_partitions").expanduser()

npes = int(sys.argv[1])
assert len(jax.devices()) >= npes, f"need {npes} devices, have {len(jax.devices())}"

print(f"[gate] npes={npes} devices={len(jax.devices())} backend={jax.default_backend()}",
      flush=True)

mesh = load_mesh(str(MESH_DIR))
part = partit.read_partition(DIST_DIR, npes)
sm = shard_mesh.build_sharded_mesh(mesh, part)
reg = padded_halo.install(sm)
print(f"[gate] slot registry (recv_max,Lmax)->slot: {reg}", flush=True)

jmesh = halo.device_mesh(devices=jax.devices()[:npes])
rng = np.random.RandomState(42)

failures = 0
for kind in ("nod", "elem", "edge"):
    src_dev, src_lane = sm.exchange[kind]              # [P, Lmax] each
    rmap = sm.exchange_ragged[kind]
    P, Lmax = src_dev.shape
    valid = np.asarray(sm.valid_mask[kind])            # [P, Lmax] bool

    for rest in ((), (5,), (3, 2)):
        x = rng.rand(P, Lmax, *rest)

        out_all = np.asarray(halo.run_halo_exchange(
            x, src_dev, src_lane, jmesh))              # oracle (all_gather)
        out_pad = np.asarray(halo.run_halo_exchange_ragged(
            x, rmap, jmesh))                           # adapter routes to padded a2a

        vm = valid.reshape(valid.shape + (1,) * len(rest))
        fwd_diff = float(np.abs(np.where(vm, out_all - out_pad, 0.0)).max())
        fwd_ok = fwd_diff == 0.0

        # --- grad: weights on valid lanes only ---
        w = jnp.asarray(np.where(vm, rng.rand(*x.shape), 0.0))
        g_all = np.asarray(jax.grad(
            lambda z: jnp.vdot(w, halo.run_halo_exchange(z, src_dev, src_lane, jmesh)))(
            jnp.asarray(x)))
        g_pad = np.asarray(jax.grad(
            lambda z: jnp.vdot(w, halo.run_halo_exchange_ragged(z, rmap, jmesh)))(
            jnp.asarray(x)))
        # pad-lane cotangents differ by design (oracle's pad lanes read lane 0);
        # compare on valid lanes only — w is zero elsewhere, but the ORACLE also
        # deposits pad-lane cotangent onto lane 0 via its src_lane=0 convention,
        # so restrict the comparison to where w actually flowed.
        g_diff = float(np.abs(np.where(vm, g_all - g_pad, 0.0)).max())
        g_scale = max(float(np.abs(g_all).max()), 1e-300)
        g_ok = g_diff / g_scale < 1e-12

        tag = "PASS" if (fwd_ok and g_ok) else "FAIL"
        failures += tag == "FAIL"
        print(f"[gate] {kind:<5} rest={str(rest):<7} fwd_maxdiff={fwd_diff:.3e} "
              f"grad_reldiff={g_diff / g_scale:.3e}  {tag}", flush=True)

print(f"[gate] npes={npes}: {'ALL PASS' if failures == 0 else f'{failures} FAILURES'}",
      flush=True)
sys.exit(1 if failures else 0)
