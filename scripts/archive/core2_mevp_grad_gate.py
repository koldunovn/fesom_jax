#!/usr/bin/env python
"""CORE2 mEVP gradient gate — Phase 9c, JM.4 (the differentiability contract, assembled model).

Re-runs the assembled-model AD gate with **mEVP** ice rheology (``ice_cfg=IceConfig(whichEVP=1)``)
to confirm the trainable seams survive mEVP-ON. The mEVP kernel AD (the additive-δmin
C¹-continuity, the identity-carry masking, the 120-iteration scan masked-NaN) is already gated
eagerly in ``test_mevp.py``; this is the ASSEMBLED backward at CORE2 scale.

Gates (smooth regimes only — the forced trajectory is non-smooth, Task 5.8):
* **[1] N=1 d(mean SST)/d(k_ver)** plateau UNCHANGED mEVP-ON — k_ver is the trainable mixing
  seam on the OCEAN side; mEVP is on the ice-momentum side, so the plateau must match the
  EVP-ON value (the ocean column diffusion is smooth; the ice fluxes don't depend on k_ver).
* **[3] N d(mean SST)/d(T0)** finite EVERYWHERE incl. masked lanes — the masked-NaN probe on the
  ASSEMBLED mEVP model (backward through thermo Newton, the mEVP 120-iter scan, the FCT limiter).
* **[I] d(mean SST)/d(m_ice0)** finite + nonzero — the ice→ocean gradient path with mEVP active.

Usage (GPU; scripts/archive/core2_mevp_grad_gate.sbatch):  python scripts/archive/core2_mevp_grad_gate.py --n 4
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (x64)
import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import surface_forcing, ice, ssh
from fesom_jax.config import A_VER
from fesom_jax.ice import IceConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
CFG = IceConfig(whichEVP=1)            # mEVP-ON (ice_dt force-derived to ice_ave_steps·dt in the step)


def build(year, n):
    mesh = load_mesh(MESH_DIR)
    sst = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    state = ice.seed_ice(phc_initial_state(mesh, IC_DIR), mesh, sst)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, year, sst_ic=sst)
    sfs = cf.stack(surface_forcing.dates_for_steps(year, DT, n))
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
    print(f"[setup] built in {time.time()-t0:.1f}s; N={args.n} ({args.n*DT/86400:.3f} days), mEVP ON",
          flush=True)

    def mean_sst(s):
        return jnp.sum(jnp.where(wet0, s.T[:, 0], 0.0)) / nwet

    passes = {}

    # [1] N=1 d(mean SST)/d(k_ver) — smooth (Kv=k_ver additively at step 1); UNCHANGED mEVP-ON.
    def kver_loss1(kver):
        p = Params(k_ver=kver, a_ver=jnp.asarray(A_VER, jnp.float64))
        fin = integrate(st0, mesh, op, None, n_steps=1, params=p, dt=DT,
                        step_forcings=sf1, forcing_static=fs, ice_cfg=CFG)
        return mean_sst(fin)

    k0 = jnp.asarray(1e-3, jnp.float64)
    g1 = float(jax.jit(jax.grad(kver_loss1))(k0))
    L = jax.jit(kver_loss1)
    rows = [(h, float((L(k0*(1+h)) - L(k0*(1-h))) / (k0*(1+h) - k0*(1-h)))) for h in H_SWEEP]
    plat = min(abs(g1 - gf)/max(abs(gf), 1e-300) for _, gf in rows)
    print(f"\n  [1] N=1 d(SST)/d(k_ver) AD={g1:+.6e}  plateau={plat:.3e}", flush=True)
    for h, gf in rows:
        print(f"      h={h:.0e} FD={gf:+.6e} rel={abs(g1-gf)/max(abs(gf),1e-300):.2e}")
    passes["[1] kver_N1_plateau<1e-5"] = bool(np.isfinite(g1) and g1 != 0.0 and plat < 1e-5)

    # [3] N d(mean SST)/d(T0) finite everywhere incl. masked lanes + memory.
    def loss_T0(T0):
        s = dataclasses.replace(st0, T=T0)
        fin = integrate(s, mesh, op, None, n_steps=args.n, dt=DT,
                        step_forcings=sfs, forcing_static=fs, ice_cfg=CFG)
        return mean_sst(fin)

    t1 = time.time()
    gT = np.asarray(jax.jit(jax.grad(loss_T0))(st0.T))
    report_mem(f"T0-grad N={args.n} mEVP")
    n_bad = int((~np.isfinite(gT)).sum())
    wet_max = float(np.max(np.abs(gT[mlay]))); masked_max = float(np.max(np.abs(gT[~mlay])))
    print(f"\n  [3] N={args.n} d(SST)/d(T0): non-finite={n_bad}; wet max|g|={wet_max:.3e}; "
          f"masked max|g|={masked_max:.3e}  ({time.time()-t1:.1f}s)", flush=True)
    passes["[3] T0_masked_NaN_clean"] = bool(n_bad == 0 and wet_max > 0.0 and masked_max == 0.0)

    # [I] d(mean SST)/d(m_ice0) finite + nonzero — the ice→ocean gradient path (mEVP active).
    jax.clear_caches()

    def loss_mice(m0):
        s = dataclasses.replace(st0, m_ice=m0)
        fin = integrate(s, mesh, op, None, n_steps=args.n, dt=DT,
                        step_forcings=sfs, forcing_static=fs, ice_cfg=CFG)
        return mean_sst(fin)

    gM = np.asarray(jax.jit(jax.grad(loss_mice))(st0.m_ice))
    print(f"\n  [I] N={args.n} d(SST)/d(m_ice0): non-finite={int((~np.isfinite(gM)).sum())}; "
          f"max|g|={float(np.max(np.abs(gM))):.3e}", flush=True)
    passes["[I] mice_grad_finite_nonzero"] = bool(np.all(np.isfinite(gM)) and np.any(gM != 0.0))

    ok = all(passes.values())
    print(f"\n  gate breakdown: {passes}")
    print("MEVP_GRAD_GATE_OK" if ok else "MEVP_GRAD_GATE_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
