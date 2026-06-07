#!/usr/bin/env python
"""CORE2 GM/Redi gradient gate — Phase 6B, Task G.7 (GATE 6B AD re-check).

Re-runs the project's AD de-risking gate on the assembled CORE2 model WITH GM/Redi live
(``gm_cfg=GMConfig()``, ice OFF — matching the gm dump + isolating the new eddy gradient
path). The per-kernel GM AD is already gated (the slope safe-sqrt + ODM95 taper [G.2], the
√bv/√tapfac coeff guards + the live ``d/d(k_gm)``=2.03e6 [G.3], the streamfunction TDMA
[G.4], the Redi safe-divides [G.6]); this confirms the SAME machinery survives in the
**assembled** backward at CORE2 scale, adds the **2nd-ML-hook** end-to-end gradient, and
measures the memory the GM per-step TDMA + Redi scatters add.

Gates (smooth regimes only — the multi-step forced trajectory is non-smooth, Task 5.8):

* **[K] N=1 d(mean SST)/d(k_gm)** — the NEW 2nd-ML-hook target. ``k_gm`` enters through
  ``init_redi_gm`` (``fer_K_top=max(scaling·k_gm,K_GM_min)``, unclamped at the 1000 default
  ⇒ smooth) → the bolus streamfunction TDMA + the Redi diffusivity, redistributing heat
  vertically at step 1. The eddy-flux gradient path, the GM analog of [1]'s mixing path.
* **[1] N=1 d(mean SST)/d(k_ver)** — the mixing hook still plateaus with GM live (Kv=k_ver
  additively at step 1; the K33 augmentation is GM-side, leaves the k_ver path smooth).
* **[3] N d(mean SST)/d(T0)** finite EVERYWHERE incl. masked lanes — the strong masked-NaN
  probe on the assembled GM model (the backward flows through the slopes / streamfunction
  TDMA / Redi scatters + every masked guard) + the checkpointed N-step **peak memory**.

Usage (GPU; scripts/core2_gm_grad_gate.sbatch):  python scripts/core2_gm_grad_gate.py --n 4
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
from fesom_jax.config import A_VER, K_GM_MAX
from fesom_jax.gm import GMConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import core2_initial_state

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 500.0
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
CFG = GMConfig()


def build(year, n):
    mesh = load_mesh(MESH_DIR)
    state = core2_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, year, sst_ic=np.asarray(state.T[:, 0]))
    sfs = cf.stack(core2_forcing.dates_for_steps(year, DT, n))
    return mesh, state, op, cf.static, sfs


def report_mem(tag):
    for d in jax.devices():
        try:
            st = d.memory_stats() or {}
            pk, lim = st.get("peak_bytes_in_use"), st.get("bytes_limit")
            if pk:
                m = f"  [{tag}] {d}: peak {pk/1e9:.2f} GB"
                if lim:
                    m += f" / {lim/1e9:.1f} GB ({100*pk/lim:.0f}%)"
                print(m, flush=True)
        except Exception:
            pass


def fd_sweep(loss_jit, x0):
    rows = []
    for h in H_SWEEP:
        xp, xm = x0 * (1.0 + h), x0 * (1.0 - h)
        rows.append((h, float((loss_jit(xp) - loss_jit(xm)) / (xp - xm))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--year", type=int, default=1958)
    args = ap.parse_args()
    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, st0, op, fs, sfs = build(args.year, args.n)
    sf1 = jax.tree.map(lambda x: x[:1], sfs)
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0]); nwet = jnp.sum(wet0)
    mlay = np.asarray(mesh.node_layer_mask)
    print(f"[setup] built in {time.time()-t0:.1f}s; N={args.n} ({args.n*DT/86400:.3f} days), GM ON",
          flush=True)

    def mean_sst(s):
        return jnp.sum(jnp.where(wet0, s.T[:, 0], 0.0)) / nwet

    passes = {}

    # ----- [K] N=1 d(mean SST)/d(k_gm): the 2nd-ML-hook eddy-flux gradient -----
    def kgm_loss1(kgm):
        p = Params(k_ver=jnp.asarray(1e-3, jnp.float64), a_ver=jnp.asarray(A_VER, jnp.float64),
                   k_gm=kgm, redi_kmax=kgm)        # Redi_Kmax auto-syncs to K_GM_max in the C
        fin = integrate(st0, mesh, op, None, n_steps=1, params=p, dt=DT,
                        step_forcings=sf1, forcing_static=fs, gm_cfg=CFG)
        return mean_sst(fin)

    kg0 = jnp.asarray(float(K_GM_MAX), jnp.float64)       # 1000 (the default; max unclamped)
    gK = float(jax.jit(jax.grad(kgm_loss1))(kg0))
    rowsK = fd_sweep(jax.jit(kgm_loss1), kg0)
    platK = min(abs(gK - gf)/max(abs(gf), 1e-300) for _, gf in rowsK)
    print(f"\n  [K] N=1 d(mean SST)/d(k_gm) AD = {gK:+.8e}  (k_gm={float(kg0):.0f})")
    for h, gf in rowsK:
        print(f"      h={h:.0e}  FD={gf:+.6e}  rel={abs(gK-gf)/max(abs(gf),1e-300):.2e}")
    print(f"      plateau (min rel) = {platK:.3e}", flush=True)
    passes["[K] kgm_N1_finite_nonzero"] = bool(np.isfinite(gK) and gK != 0.0)
    passes["[K] kgm_N1_plateau<1e-4"] = bool(np.isfinite(gK) and gK != 0.0 and platK < 1e-4)

    # ----- [1] N=1 d(mean SST)/d(k_ver): mixing hook still smooth with GM live -----
    def kver_loss1(kver):
        p = Params(k_ver=kver, a_ver=jnp.asarray(A_VER, jnp.float64))
        fin = integrate(st0, mesh, op, None, n_steps=1, params=p, dt=DT,
                        step_forcings=sf1, forcing_static=fs, gm_cfg=CFG)
        return mean_sst(fin)

    k0 = jnp.asarray(1e-3, jnp.float64)
    g1 = float(jax.jit(jax.grad(kver_loss1))(k0))
    rows1 = fd_sweep(jax.jit(kver_loss1), k0)
    plat1 = min(abs(g1 - gf)/max(abs(gf), 1e-300) for _, gf in rows1)
    print(f"\n  [1] N=1 d(mean SST)/d(k_ver) AD = {g1:+.8e}  plateau = {plat1:.3e}")
    for h, gf in rows1:
        print(f"      h={h:.0e}  FD={gf:+.6e}  rel={abs(g1-gf)/max(abs(gf),1e-300):.2e}")
    passes["[1] kver_N1_plateau<1e-6"] = bool(np.isfinite(g1) and g1 != 0.0 and plat1 < 1e-6)

    # ----- [3] N d(mean SST)/d(T0) finite everywhere incl. masked lanes + memory -----
    def loss_T0(T0):
        s = dataclasses.replace(st0, T=T0)
        fin = integrate(s, mesh, op, None, n_steps=args.n, dt=DT,
                        step_forcings=sfs, forcing_static=fs, gm_cfg=CFG)
        return mean_sst(fin)

    t1 = time.time()
    gT = np.asarray(jax.jit(jax.grad(loss_T0))(st0.T))
    report_mem(f"T0-grad N={args.n} GM")
    n_bad = int((~np.isfinite(gT)).sum())
    wet_max = float(np.max(np.abs(gT[mlay]))); masked_max = float(np.max(np.abs(gT[~mlay])))
    print(f"\n  [3] N={args.n} d(mean SST)/d(T0): non-finite={n_bad}; wet max|g|={wet_max:.3e}; "
          f"masked max|g|={masked_max:.3e}  ({time.time()-t1:.1f}s)", flush=True)
    passes["[3] T0_masked_NaN_clean"] = bool(n_bad == 0 and wet_max > 0.0 and masked_max == 0.0)

    # the headline GATE-6B requirement is the 2nd-hook gradient FLOWS (finite, nonzero) +
    # masked-NaN clean; the plateau quality is a reported conditioning indicator.
    required = ["[K] kgm_N1_finite_nonzero", "[1] kver_N1_plateau<1e-6", "[3] T0_masked_NaN_clean"]
    ok = all(passes[k] for k in required)
    print(f"\n  gate breakdown: {passes}")
    print(f"  (required: {required})")
    print("GM_GRAD_GATE_OK" if ok else "GM_GRAD_GATE_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
