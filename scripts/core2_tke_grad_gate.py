"""CORE2 TKE gradient gate — Phase 9b, Task JT.4 (GATE 9b AD core).

End-to-end reverse-mode AD through the assembled TKE CORE2 model (``tke_cfg=TkeConfig()``,
KPP/GM/ice OFF — the TKE-only path). The column-core + driver AD are already gated
(``test_column_grad_finite`` / ``test_mixing_tke_composition_and_grad``); this is the
**assembled multi-step** gate — the differentiability seam that makes TKE the project's
PRIMARY hybrid-ML hook. Two structural gradient paths exist per step (the §4 contract):
``tke → Kv/Av → ocean`` (the well-conditioned ML path) and the ``tke`` self-recurrence
(floor-gated by ``tke_min`` in the quiescent ocean).

Gates (the GATE 9b "Gradients" row):

* **[c_k] FD↔AD plateau on ``d(mean ML Kv)/d(tke_c_k)``** — the headline ML-seam parameter.
  ``KappaM = c_k·mxl·√tke`` ⇒ ``Kv ∝ c_k`` where ``tke>0`` (after spin-up), so the gradient
  is well-conditioned (the PP ``k_ver`` / KPP ``K_bg`` analog). Needs ``n≥2`` so the cold-start
  ``tke=0`` spins up (else ``KappaM=0`` ⇒ ``d/d(c_k)=0``).
* **[cd] FD↔AD plateau on ``d(mean surf tke)/d(tke_cd)``** — the surface-flux coefficient
  (``forc = cd·|stress|^{3/2}/dzt`` drives the surface ``tke`` at step 1; clean even at n=1).
* **[T0] masked-NaN ``d(mean SST)/d(T0)`` finite EVERYWHERE + 0 on masked lanes, TKE-ON** —
  the hard requirement (the safe-sqrt / safe-pow / clamped-divide guards hold end-to-end).
* **[tkeIC] ``d(mean SST)/d(tke-IC)`` finite through the N-step checkpointed scan** — the new
  scan-carry path (``tke`` joins the carry; ``d_tri = tke_old + dt·forc`` ⇒ the IC propagates
  linearly even at cold start, so finite + nonzero).

Usage (GPU; scripts/core2_tke_grad_gate.sbatch):  python scripts/core2_tke_grad_gate.py --n 4
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import core2_forcing, ssh
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
CFG = TkeConfig()


def build(year, n):
    mesh = load_mesh(MESH_DIR)
    state = core2_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, year, sst_ic=np.asarray(state.T[:, 0]))
    sfs = cf.stack(core2_forcing.dates_for_steps(year, DT, n))
    return mesh, state, op, cf.static, sfs


def fd_sweep(loss_jit, x0):
    rows = []
    for h in H_SWEEP:
        xp, xm = x0 * (1.0 + h), x0 * (1.0 - h)
        rows.append((h, float((loss_jit(xp) - loss_jit(xm)) / (xp - xm))))
    return rows


def _params(**kw):
    return dataclasses.replace(Params.defaults(), **kw)


def main():
    ap = argparse.ArgumentParser()
    # n=2 keeps each full-integrate backward small enough for the 40 GB A100 — the script
    # runs FOUR separate full-integrate grads (vs KPP's one), and TKE's backward is heavier
    # (the extra tke scan-carry + the mxl/thomas lax.scans). n=2 still spins tke up at step 1
    # so c_k engages KappaM at step 2; jax.clear_caches() frees executables between gates.
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--year", type=int, default=1958)
    args = ap.parse_args()
    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, st0, op, fs, sfs = build(args.year, args.n)
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0]); nwet = jnp.sum(wet0)
    iface = jnp.asarray(mesh.node_iface_mask); niface = jnp.sum(iface)
    mlay = np.asarray(mesh.node_layer_mask)
    # the surface/active mixed-layer band (top few interfaces) for the Kv/tke parameter losses
    surf = jnp.asarray(mesh.node_iface_mask) & (jnp.arange(mesh.nl)[None, :] <= 4)
    nsurf = jnp.sum(surf)
    print(f"[setup] built in {time.time()-t0:.1f}s; N={args.n} ({args.n*DT/86400:.3f} days), TKE ON",
          flush=True)

    def run(params, st=None):
        return integrate(st if st is not None else st0, mesh, op, None, n_steps=args.n, dt=DT,
                         step_forcings=sfs, forcing_static=fs, tke_cfg=CFG, params=params)

    def mean_sst(s):
        return jnp.sum(jnp.where(wet0, s.T[:, 0], 0.0)) / nwet

    passes = {}

    # ----- [c_k] d(mean ML Kv)/d(tke_c_k): the headline ML-seam parameter -----
    def ck_loss(c_k):
        fin = run(_params(tke_c_k=c_k))
        return jnp.sum(jnp.where(surf, fin.Kv, 0.0)) / nsurf

    ck0 = jnp.asarray(float(Params.defaults().tke_c_k), jnp.float64)   # 0.1
    gCk = float(jax.jit(jax.grad(ck_loss))(ck0))
    rowsCk = fd_sweep(jax.jit(ck_loss), ck0)
    platCk = min(abs(gCk - gf) / max(abs(gf), 1e-300) for _, gf in rowsCk)
    print(f"\n  [c_k] d(mean ML Kv)/d(tke_c_k) AD = {gCk:+.8e}  (c_k={float(ck0):.3f})")
    for h, gf in rowsCk:
        print(f"      h={h:.0e}  FD={gf:+.6e}  rel={abs(gCk-gf)/max(abs(gf),1e-300):.2e}")
    print(f"      plateau (min rel) = {platCk:.3e}", flush=True)
    passes["[c_k] finite_nonzero"] = bool(np.isfinite(gCk) and gCk != 0.0)
    passes["[c_k] plateau<1e-4"] = bool(np.isfinite(gCk) and gCk != 0.0 and platCk < 1e-4)
    jax.clear_caches()                                   # free the GPU executables before [cd]

    # ----- [cd] d(mean surf tke)/d(tke_cd): the surface-flux coefficient -----
    def cd_loss(cd):
        fin = run(_params(tke_cd=cd))
        return jnp.sum(jnp.where(surf, fin.tke, 0.0)) / nsurf

    cd0 = jnp.asarray(float(Params.defaults().tke_cd), jnp.float64)    # 3.75
    gCd = float(jax.jit(jax.grad(cd_loss))(cd0))
    rowsCd = fd_sweep(jax.jit(cd_loss), cd0)
    platCd = min(abs(gCd - gf) / max(abs(gf), 1e-300) for _, gf in rowsCd)
    print(f"\n  [cd] d(mean surf tke)/d(tke_cd) AD = {gCd:+.8e}  (cd={float(cd0):.3f})")
    for h, gf in rowsCd:
        print(f"      h={h:.0e}  FD={gf:+.6e}  rel={abs(gCd-gf)/max(abs(gf),1e-300):.2e}")
    print(f"      plateau (min rel) = {platCd:.3e}", flush=True)
    passes["[cd] finite_nonzero"] = bool(np.isfinite(gCd) and gCd != 0.0)
    passes["[cd] plateau<1e-4"] = bool(np.isfinite(gCd) and gCd != 0.0 and platCd < 1e-4)
    jax.clear_caches()                                   # free before [T0]

    # ----- [T0] d(mean SST)/d(T0) finite EVERYWHERE incl. masked lanes -----
    def loss_T0(T0):
        return mean_sst(run(None, dataclasses.replace(st0, T=T0)))

    t1 = time.time()
    gT = np.asarray(jax.jit(jax.grad(loss_T0))(st0.T))
    n_bad = int((~np.isfinite(gT)).sum())
    wet_max = float(np.max(np.abs(gT[mlay]))); masked_max = float(np.max(np.abs(gT[~mlay])))
    print(f"\n  [T0] N={args.n} d(mean SST)/d(T0): non-finite={n_bad}; wet max|g|={wet_max:.3e}; "
          f"masked max|g|={masked_max:.3e}  ({time.time()-t1:.1f}s)", flush=True)
    passes["[T0] masked_NaN_clean"] = bool(n_bad == 0 and wet_max > 0.0 and masked_max == 0.0)
    jax.clear_caches()                                   # free before [tkeIC]

    # ----- [tkeIC] d(mean SST)/d(tke-IC) finite through the N-step checkpointed scan -----
    def loss_tke_ic(tke0):
        return mean_sst(run(None, dataclasses.replace(st0, tke=tke0)))

    gI = np.asarray(jax.jit(jax.grad(loss_tke_ic))(st0.tke))
    n_bad_i = int((~np.isfinite(gI)).sum())
    iface_np = np.asarray(mesh.node_iface_mask)
    dry_max = float(np.max(np.abs(gI[~iface_np]))) if (~iface_np).any() else 0.0
    print(f"\n  [tkeIC] d(mean SST)/d(tke-IC): non-finite={n_bad_i}; wet max|g|="
          f"{float(np.max(np.abs(gI[iface_np]))):.3e}; dry max|g|={dry_max:.3e}", flush=True)
    passes["[tkeIC] finite"] = bool(n_bad_i == 0 and dry_max == 0.0)

    required = ["[c_k] plateau<1e-4", "[cd] finite_nonzero", "[T0] masked_NaN_clean",
                "[tkeIC] finite"]
    ok = all(passes.get(k, False) for k in required)
    print(f"\n  gate breakdown: {passes}")
    print(f"  (required: {required})")
    print("TKE_GRAD_GATE_OK" if ok else "TKE_GRAD_GATE_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
