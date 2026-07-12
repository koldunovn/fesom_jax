"""Verify the SHARDED (multi-device) adjoint for the TKE + NN-of-TKE path == the dense gradient.

The multi-GPU NN twin needs `jax.grad` w.r.t. `params.tke_c_k` / `params.tke_nn` of a sharded run to
equal the single-device gradient. `test_gradient_sharded` proves the all_gather-halo adjoint == dense
for PP-mixing ocean params (k_ver/a_ver) + KPP + GM/Redi, but NEVER for `tke_cfg` + the NN — the leaf
the twin actually trains. This script closes that gap on CPU with 2 fake devices (no GPU needed):

  cfg = TKE + GM + prognostic ice (the forced all-on minus zstar — matches the proven sharded forced
  setup, swapping KPP→TKE). Compares d(mean SST)/d(tke_c_k) and d(.)/d(tke_nn weights) of the sharded
  1-step run (use_ragged=False ⇒ the autodiff-CORRECT all_gather halo; the ragged path's transpose is
  the buggy one) against the single-device dense gradient.

Run:  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 python scripts/debug/verify_sharded_tke_grad.py
"""
from __future__ import annotations

import dataclasses

import numpy as np

import fesom_jax  # noqa: F401  (x64)
import jax
import jax.numpy as jnp

from fesom_jax import core2_forcing, ice, partit, shard_mesh, ssh, tke_nn
from fesom_jax import integrate_sharded as ish
from fesom_jax import step as stepmod
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.ice_evp import boundary_node_mask
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.shard_mesh import _shard_along_axis, local_sizes
from fesom_jax.tke import TkeConfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
DT = 1800.0
YEAR = 1958
NPES = 2


def main():
    ndev = len(jax.devices())
    print(f"=== sharded TKE+NN adjoint verify  devices={ndev} ({jax.devices()[0].platform}) ===",
          flush=True)
    assert ndev >= NPES, f"need {NPES} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={NPES}"

    mesh = load_mesh(MESH_DIR)
    base = core2_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(base.T[:, 0])
    state = ice.seed_ice(base, mesh, sst0)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, YEAR, sst_ic=sst0)
    sf = cf.step_forcing(*core2_forcing.dates_for_steps(YEAR, DT, 1)[0])
    fs = cf.static
    cfg = dict(tke_cfg=TkeConfig(), gm_cfg=GMConfig(), ice_cfg=IceConfig())
    print("[setup] built mesh+IC+forcing+ice; cfg=TKE+GM+ice", flush=True)

    part = partit.read_partition(DIST, NPES)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sf_p = shard_mesh.partition_step_forcing(sf, part)
    fs_p = shard_mesh.partition_forcing_static(fs, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = jnp.zeros((NPES, sm.Lmax["elem"], 2))
    _, Lmax = local_sizes(part)
    bn_p = _shard_along_axis(np.asarray(boundary_node_mask(mesh)), part.myList_nod2D,
                             Lmax["nod"], 0, False)
    # 3D owned-wet mask (NOT surface-only mean SST — that is nearly insensitive to TKE, which
    # redistributes heat VERTICALLY ⇒ d/d(tke_c_k)≈0 and the check is inconclusive). A sum of
    # T²+S² over ALL wet depths is strongly TKE-sensitive; owned-summed across P == the dense sum.
    owned3 = jnp.asarray(sm.owned_mask["nod"])[:, :, None]
    nlm = jnp.asarray(sm.fields["node_layer_mask"])
    wet3 = owned3 & nlm
    wet3_de = jnp.asarray(mesh.node_layer_mask)
    print(f"[setup] partitioned to npes={NPES}  Lmax_nod={sm.Lmax['nod']}", flush=True)

    def loss_sh(params):
        st = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT, is_first_step=True,
                                  npes=NPES, params=params, step_forcing=sf_p,
                                  forcing_static=fs_p, boundary_node_p=bn_p, use_ragged=False, **cfg)
        return jnp.sum(jnp.where(wet3, st.T * st.T + st.S * st.S, 0.0))

    def loss_de(params):
        st = stepmod.step(state, mesh, op, jnp.zeros((mesh.elem2D, 2)), params, dt=DT,
                          is_first_step=True, step_forcing=sf, forcing_static=fs, **cfg)
        return jnp.sum(jnp.where(wet3_de, st.T * st.T + st.S * st.S, 0.0))

    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)

    # --- (1) scalar TKE leaf: d/d(tke_c_k) sharded vs dense ---
    print("\n[1] d(meanSST)/d(tke_c_k)  (sharded all_gather vs dense) ...", flush=True)
    g_sh = jax.grad(loss_sh)(Params.defaults())
    g_de = jax.grad(loss_de)(Params.defaults())
    for name in ("tke_c_k", "tke_c_eps", "k_gm"):
        a, b = float(getattr(g_sh, name)), float(getattr(g_de, name))
        print(f"    d/d({name:9s})  sharded={a:+.8e}  dense={b:+.8e}  rel={rel(a,b):.2e}", flush=True)
    ck_ok = rel(float(g_sh.tke_c_k), float(g_de.tke_c_k)) < 1e-5 and float(g_de.tke_c_k) != 0.0

    # --- (2) the NN leaf: d/d(tke_nn weights) sharded vs dense ---
    print("\n[2] d(meanSST)/d(tke_nn weights)  (the actual twin leaf) ...", flush=True)
    from fesom_jax import calibrate
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(0), zero_last=False)

    def loss_sh_nn(nnw):
        return loss_sh(calibrate.build_params({"tke_nn": nnw}))

    def loss_de_nn(nnw):
        return loss_de(calibrate.build_params({"tke_nn": nnw}))

    gnn_sh = jax.grad(loss_sh_nn)(nn)
    gnn_de = jax.grad(loss_de_nn)(nn)
    fl_sh = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(gnn_sh)])
    fl_de = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(gnn_de)])
    l2 = float(np.linalg.norm(fl_sh - fl_de) / (np.linalg.norm(fl_de) + 1e-300))
    cos = float(np.dot(fl_sh, fl_de) / (np.linalg.norm(fl_sh) * np.linalg.norm(fl_de) + 1e-300))
    print(f"    |g_sh|={np.linalg.norm(fl_sh):.4e}  |g_de|={np.linalg.norm(fl_de):.4e}  "
          f"relL2(sh-de)={l2:.2e}  cos={cos:.6f}", flush=True)
    nn_ok = l2 < 1e-4 and np.linalg.norm(fl_de) > 0

    print(f"\n[gate] tke_c_k adjoint sharded==dense: {ck_ok}   tke_nn adjoint sharded==dense: {nn_ok}",
          flush=True)
    print("SHARDED_TKE_GRAD_OK" if (ck_ok and nn_ok) else "SHARDED_TKE_GRAD_FAIL", flush=True)
    return 0 if (ck_ok and nn_ok) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
