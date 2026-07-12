"""FAST CPU reproducer for the multi-step SHARDED-grad NaN (the §3 multi-GPU NN twin).

The P=4 N=48 sharded twin runs the FORWARD fine (8.3 GB/dev) but `jax.grad` of the
checkpointed sharded scan returns `|g|=nan` at it=0. Dense multi-step (E1 N=10) is finite, so
the NaN is SHARDING-specific — the suspected mechanism is an unguarded division on PADDING lanes
(`x/0` masked to 0 in the forward via `jnp.where`, but `where(mask, nan, 0)` has a NaN gradient).

This reproduces it on CPU with 2 fake devices (seconds, no GPU) so the op can be bisected:

  python scripts/debug/repro_sharded_grad_nan.py --nsteps 2 --config all3
  # bisect: --config tke|gm|ice|tkegm|all3   --nsteps 1|2|3

Reports, per n_steps/config: forward finite? grad finite (sharded vs dense)? which Params/NN
leaf is NaN. `--hardmask` multiplies the model output by owned&wet BEFORE the loss (tests whether
forcing the padding output-cotangent to 0 kills it — if not, the NaN is generated internally).

Run:  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 python scripts/debug/repro_sharded_grad_nan.py
"""
from __future__ import annotations

import argparse

import numpy as np

import fesom_jax  # noqa: F401  (x64)
import jax
import jax.numpy as jnp

from fesom_jax import calibrate, surface_forcing, ice, partit, shard_mesh, ssh, tke_nn
from fesom_jax import integrate_sharded as ish
from fesom_jax import step as stepmod
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.ice_evp import boundary_node_mask
from fesom_jax.mesh import load_mesh
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


def build_cfg(name):
    if name == "tke":
        return dict(tke_cfg=TkeConfig())
    if name == "gm":
        return dict(gm_cfg=GMConfig())
    if name == "ice":
        return dict(ice_cfg=IceConfig())
    if name == "tkegm":
        return dict(tke_cfg=TkeConfig(), gm_cfg=GMConfig())
    if name == "all3":
        return dict(tke_cfg=TkeConfig(), gm_cfg=GMConfig(), ice_cfg=IceConfig())
    raise ValueError(name)


def leaf_nan_report(g):
    bad = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(g):
        a = np.asarray(leaf)
        if a.size and not np.all(np.isfinite(a)):
            nm = jax.tree_util.keystr(path)
            bad.append(f"{nm}: nan={int(np.isnan(a).sum())}/{a.size} inf={int(np.isinf(a).sum())}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=2)
    ap.add_argument("--config", default="all3")
    ap.add_argument("--hardmask", action="store_true")
    args = ap.parse_args()
    NS = args.nsteps

    ndev = len(jax.devices())
    print(f"=== sharded-grad NaN reproducer  nsteps={NS} config={args.config} hardmask={args.hardmask} "
          f"devices={ndev} ({jax.devices()[0].platform}) ===", flush=True)
    assert ndev >= NPES, f"need {NPES} devices; XLA_FLAGS=--xla_force_host_platform_device_count={NPES}"

    mesh = load_mesh(MESH_DIR)
    base = core2_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(base.T[:, 0])
    state = ice.seed_ice(base, mesh, sst0)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=sst0)
    sf = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    fs = cf.static
    cfg = build_cfg(args.config)
    has_ice = "ice_cfg" in cfg

    part = partit.read_partition(DIST, NPES)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sf_p = shard_mesh.partition_step_forcing(sf, part)
    fs_p = shard_mesh.partition_forcing_static(fs, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = jnp.zeros((NPES, sm.Lmax["elem"], 2))
    _, Lmax = local_sizes(part)
    bn_p = _shard_along_axis(np.asarray(boundary_node_mask(mesh)), part.myList_nod2D,
                             Lmax["nod"], 0, False) if has_ice else None
    owned3 = jnp.asarray(sm.owned_mask["nod"])[:, :, None]
    nlm = jnp.asarray(sm.fields["node_layer_mask"])
    wet3 = owned3 & nlm
    wet3_de = jnp.asarray(mesh.node_layer_mask)
    print(f"[setup] partitioned npes={NPES} Lmax_nod={sm.Lmax['nod']}", flush=True)

    # ---- sharded multi-step runner (the twin's exact path) ----
    _run = ish.run_steps_sharded(sm, state_p, sop, stress_p, n_steps=NS, dt=DT, npes=NPES,
                                 step_forcing=sf_p, forcing_static=fs_p, boundary_node_p=bn_p,
                                 use_ragged=False, return_grad_fn=True, **cfg)

    def loss_sh(params):
        st = _run(params)
        T, S = st.T, st.S
        if args.hardmask:
            m = wet3.astype(T.dtype)
            T, S = T * m, S * m
        return jnp.sum(jnp.where(wet3, T * T + S * S, 0.0))

    # ---- dense multi-step reference (mirrors bodyg: step1 first, then NS-1 more) ----
    zeros_e = jnp.zeros((mesh.elem2D, 2))

    def dense_run(params):
        st = stepmod.step(state, mesh, op, zeros_e, params, dt=DT, is_first_step=True,
                          step_forcing=sf, forcing_static=fs, **cfg)
        for _ in range(NS - 1):
            st = stepmod.step(st, mesh, op, zeros_e, params, dt=DT, is_first_step=False,
                              step_forcing=sf, forcing_static=fs, **cfg)
        return st

    def loss_de(params):
        st = dense_run(params)
        return jnp.sum(jnp.where(wet3_de, st.T * st.T + st.S * st.S, 0.0))

    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(0), zero_last=True)   # the twin's start point
    f_sh = float(loss_sh(calibrate.build_params({"tke_nn": nn})))
    f_de = float(loss_de(calibrate.build_params({"tke_nn": nn})))
    print(f"[fwd] sharded={f_sh:.6e} ({'finite' if np.isfinite(f_sh) else 'NONFINITE'})  "
          f"dense={f_de:.6e} ({'finite' if np.isfinite(f_de) else 'NONFINITE'})", flush=True)

    gnn_sh = jax.grad(lambda w: loss_sh(calibrate.build_params({"tke_nn": w})))(nn)
    gnn_de = jax.grad(lambda w: loss_de(calibrate.build_params({"tke_nn": w})))(nn)
    fl_sh = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(gnn_sh)])
    fl_de = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(gnn_de)])
    sh_fin, de_fin = np.all(np.isfinite(fl_sh)), np.all(np.isfinite(fl_de))
    print(f"[grad d/d(nn)] sharded |g|={np.linalg.norm(fl_sh):.4e} finite={sh_fin}   "
          f"dense |g|={np.linalg.norm(fl_de):.4e} finite={de_fin}", flush=True)
    if not sh_fin:
        print("  sharded NaN leaves:", flush=True)
        for b in leaf_nan_report(gnn_sh):
            print("   ", b, flush=True)

    # also grad w.r.t. all Params (locates which physics leaf carries the NaN)
    gp_sh = jax.grad(loss_sh)(calibrate.build_params({"tke_nn": nn}))
    bad = leaf_nan_report(gp_sh)
    print(f"[grad d/d(Params)] sharded NaN leaves: {bad if bad else 'NONE (all finite)'}", flush=True)

    ok = sh_fin and de_fin
    if ok and np.linalg.norm(fl_de) > 0:
        l2 = float(np.linalg.norm(fl_sh - fl_de) / (np.linalg.norm(fl_de) + 1e-300))
        print(f"[match] relL2(sharded - dense) = {l2:.2e}", flush=True)
        ok = l2 < 1e-3
    print("REPRO_OK" if ok else "REPRO_NAN", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
