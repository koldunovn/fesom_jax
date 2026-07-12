#!/usr/bin/env python
"""Capture a realistic CORE2 ``ssh_rhs`` for the Phase-8 S.6 distributed-CG gate.

The distributed CG's early-stop iteration count must NOT drift across partitions
(review #1: a 1-iteration drift ≈ ``soltol·‖b‖ ≈ 1e-5`` — seven orders above the
1e-12 per-substep budget). The margin between the residual trajectory and the
stop threshold is a property of the **operator** (mesh+dt) and the **rhs**
(config), so we capture ``ssh_rhs`` from the *real* assembled CORE2 KPP+GM+ice
step (NOT pi) and record the single-device residual-vs-threshold margin here.

Runs 2 eager steps (``dt=1800`` — the Fortran KPP reference) and saves to
``data/ssh_rhs_core2/`` (on /work, gitignored):

* ``ssh_rhs_step1.npy`` — the step-1 (cold-start, ``x0=0``) rhs,
* ``d_eta_step1.npy``   — the step-1 solution (the step-2 warm start),
* ``ssh_rhs_step2.npy`` — the step-2 (warm-start) rhs.

The S.6 test (``test_ssh_sharded.py``) loads these to gate the sharded CG in
isolation against the single-device solve (per-node ~1e-12 + identical iters).

Run:  sbatch scripts/debug/capture_core2_ssh_rhs.sbatch   (or directly on a compute node)
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import core2_forcing, ice, ssh
from fesom_jax import step as stepmod
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
OUT = ROOT / "data" / "ssh_rhs_core2"
DT = 1800.0
YEAR = 1958


def residual_trajectory(op, ssh_rhs, x0, soltol=float(ssh.SOLTOL),
                        maxiter=int(ssh.MAXITER)):
    """Single-device forward PCG mirroring ``solve_ssh``'s early-stop, recording the
    RMS residual ``√(Σr²/n)`` at every iterate and the stop threshold ``rtol_fwd``.
    Returns ``(iters, rtol_fwd, [resid0, resid1, …])`` — the iteration count is the
    first ``k`` with ``resid_k < rtol_fwd`` (the C's relative-residual stop)."""
    matvec = lambda x: ssh.ssh_matvec(op, x)        # noqa: E731
    precond = lambda r: ssh.ssh_precond(op, r)      # noqa: E731
    n = op.n_nodes
    ssh_rhs = np.asarray(ssh_rhs, dtype=np.float64)
    rtol = soltol * np.sqrt(float(ssh_rhs @ ssh_rhs) / ssh_rhs.shape[0])
    b = ssh_rhs - np.asarray(matvec(jnp.asarray(x0)))    # b_eff
    delta = np.zeros_like(b)
    r = b.copy()
    z = np.asarray(precond(jnp.asarray(r)))
    p = z.copy()
    s_old = float(r @ z)
    traj = [float(np.sqrt((r @ r) / n))]
    it = 0
    while traj[-1] >= rtol and it < maxiter:
        Ap = np.asarray(matvec(jnp.asarray(p)))
        al = s_old / float(p @ Ap)
        delta = delta + al * p
        r = r - al * Ap
        z = np.asarray(precond(jnp.asarray(r)))
        sp0 = float(r @ z)
        traj.append(float(np.sqrt((r @ r) / n)))
        be = sp0 / s_old
        p = z + be * p
        s_old = sp0
        it += 1
    return it, rtol, traj


def report_margin(tag, op, ssh_rhs, x0):
    it, rtol, traj = residual_trajectory(op, ssh_rhs, x0)
    print(f"\n=== {tag}: forward PCG iteration count = {it}  (rtol_fwd = {rtol:.6e}) ===")
    for k, rr in enumerate(traj):
        mark = " <-- STOP" if (rr < rtol and (k == 0 or traj[k - 1] >= rtol)) else ""
        ratio = rr / rtol
        print(f"   iter {k:2d}: resid = {rr:.6e}   resid/rtol = {ratio:.4e}{mark}")
    # the margin that protects the iteration count from psum reassociation (~1e-12):
    # how far below rtol the STOPPING residual is, and how far ABOVE the previous one.
    if it >= 1:
        stop_ratio = traj[it] / rtol
        prev_ratio = traj[it - 1] / rtol
        print(f"   --> stop iterate resid/rtol = {stop_ratio:.4e} (below 1)")
        print(f"   --> prev iterate resid/rtol = {prev_ratio:.4e} (above 1)")
        print(f"   --> margins vs the ~1e-12 psum reassociation: "
              f"crossing factor (prev/stop) = {prev_ratio/stop_ratio:.3e}  "
              f"(≫1 ⇒ count robust)")
    return it


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh = load_mesh(MESH_DIR)
    sst0 = np.asarray(core2_initial_state(mesh, IC_DIR).T[:, 0])
    state = ice.seed_ice(core2_initial_state(mesh, IC_DIR), mesh, sst0)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, YEAR, sst_ic=sst0)
    dates = core2_forcing.dates_for_steps(YEAR, DT, 2)
    ice_cfg, gm_cfg, kpp_cfg = IceConfig(), GMConfig(), KppConfig()
    print(f"[setup] built in {time.time()-t0:.1f}s; assembled KPP+GM+ice, dt={DT:.0f}", flush=True)

    # --- step 1 (cold start, x0 = state.d_eta = 0) ---
    y, doy, sec, month = dates[0]
    sf = cf.step_forcing(y, doy, sec, month)
    ts = time.time()
    st1 = stepmod.step(state, mesh, op, None, dt=DT, is_first_step=True,
                       step_forcing=sf, forcing_static=cf.static,
                       ice_cfg=ice_cfg, gm_cfg=gm_cfg, kpp_cfg=kpp_cfg)
    print(f"[step 1] done in {time.time()-ts:.1f}s  "
          f"max|ssh_rhs|={float(jnp.max(jnp.abs(st1.ssh_rhs))):.3e}  "
          f"max|d_eta|={float(jnp.max(jnp.abs(st1.d_eta))):.3e}", flush=True)
    ssh_rhs1 = np.asarray(st1.ssh_rhs)
    d_eta1 = np.asarray(st1.d_eta)
    x0_step1 = np.asarray(state.d_eta)        # the warm start used at step 1 (== 0)

    # --- step 2 (warm start, x0 = st1.d_eta) ---
    y, doy, sec, month = dates[1]
    sf = cf.step_forcing(y, doy, sec, month)
    ts = time.time()
    st2 = stepmod.step(st1, mesh, op, None, dt=DT, is_first_step=False,
                       step_forcing=sf, forcing_static=cf.static,
                       ice_cfg=ice_cfg, gm_cfg=gm_cfg, kpp_cfg=kpp_cfg)
    print(f"[step 2] done in {time.time()-ts:.1f}s  "
          f"max|ssh_rhs|={float(jnp.max(jnp.abs(st2.ssh_rhs))):.3e}  "
          f"max|d_eta|={float(jnp.max(jnp.abs(st2.d_eta))):.3e}", flush=True)
    ssh_rhs2 = np.asarray(st2.ssh_rhs)

    np.save(OUT / "ssh_rhs_step1.npy", ssh_rhs1)
    np.save(OUT / "d_eta_step1.npy", d_eta1)
    np.save(OUT / "ssh_rhs_step2.npy", ssh_rhs2)
    print(f"\n[saved] {OUT}/ssh_rhs_step1.npy, d_eta_step1.npy, ssh_rhs_step2.npy", flush=True)

    # --- single-device iteration-count + residual-vs-threshold margin ---
    print("\n" + "=" * 72)
    print("SINGLE-DEVICE iteration count + residual-vs-threshold margin (the")
    print("reference the sharded CG must match; review #1 determinism check)")
    print("=" * 72)
    it1 = report_margin("STEP 1 (cold start, x0=0)", op, ssh_rhs1, x0_step1)
    it2 = report_margin("STEP 2 (warm start, x0=d_eta_step1)", op, ssh_rhs2, d_eta1)
    print(f"\nSUMMARY: step-1 iters={it1}, step-2 iters={it2}  "
          f"(dt={DT:.0f}, nod2D={mesh.nod2D})", flush=True)
    print("CAPTURE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
