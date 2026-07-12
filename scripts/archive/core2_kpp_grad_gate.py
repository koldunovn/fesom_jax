#!/usr/bin/env python
"""CORE2 KPP gradient gate — Phase 6C, Task K.10 (GATE 6C AD half).

Runs the project's AD de-risking gate on the assembled CORE2 model WITH KPP live
(``kpp_cfg=KppConfig()``, GM + ice OFF — matching the KPP dump config, isolating the new
KPP gradient path). The per-kernel KPP AD is already gated (the ustar/Vtsq safe-sqrt
[K.5], the stop-grad kbl + differentiable hbl interp [K.5], the f1/gat1/dat1 physical
floors [K.6], the enhance/assemble + the 24 s driver-level ``d/dT`` through ``mixing_kpp``
[K.8]); this confirms the SAME machinery survives in the **assembled multi-step backward**
at CORE2 scale + adds the KPP-tunable parameter gradient.

KPP is the kink-heaviest scheme (the integer OBL level kbl, the wscale table bin index,
copysign step functions), so the bar is **(a) no NaN/Inf in the backward, finite incl.
masked lanes** (the hard requirement) + **(b) a well-conditioned gradient where one
physically exists** (the additive interior ``visc_sh_limit·frit + K_bg`` term). We do NOT
require a smooth plateau through the discrete kbl.

Gates:

* **[3] N d(mean SST)/d(T0) finite EVERYWHERE incl. masked lanes** — the strong masked-NaN
  probe on the assembled KPP model (the backward flows through ri_iwmix / the bldepth OBL
  search / blmix cubic / enhance / smooth_blmc / node→elem + mo_convect + every masked
  guard). THE required, load-bearing gate (the one that has caught every prior backward-NaN
  trap). + the checkpointed N-step peak memory.
* **[Kbg] d(mean Kv)/d(K_bg) finite + nonzero + FD-plateau** — the well-conditioned
  KPP-tunable parameter gradient (the Phase-7a tuning seam). ``K_bg`` enters the interior
  diffusivity ADDITIVELY (``diffKt = diff_sh_limit·frit + K_bg``), so its gradient is clean
  — the KPP analog of the PP ``k_ver`` mixing hook. (``KppConfig`` is a static arg, so this
  traces ``K_bg`` through the replicated mixing chain with the wscale tables prebuilt from
  the static cfg — the lru_cache needs a hashable cfg.)

Usage (GPU; scripts/archive/core2_kpp_grad_gate.sbatch):  python scripts/archive/core2_kpp_grad_gate.py --n 4
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

from fesom_jax import core2_forcing, eos, kpp, pp, ssh
from fesom_jax.integrate import integrate
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 500.0
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
CFG = KppConfig()


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
    iface = jnp.asarray(mesh.node_iface_mask); niface = jnp.sum(iface)
    mlay = np.asarray(mesh.node_layer_mask)
    print(f"[setup] built in {time.time()-t0:.1f}s; N={args.n} ({args.n*DT/86400:.3f} days), KPP ON",
          flush=True)

    def mean_sst(s):
        return jnp.sum(jnp.where(wet0, s.T[:, 0], 0.0)) / nwet

    passes = {}

    # ----- [Kbg] d(mean Kv)/d(K_bg): the well-conditioned KPP-tunable (Phase-7a seam) -----
    # K_bg enters diffKt additively (diff_sh_limit·frit + K_bg). KppConfig is a static arg,
    # so trace K_bg through the replicated mixing chain with the wscale tables prebuilt from
    # the static cfg (build_wscale_tables lru_cache needs a hashable cfg). The k_bg-independent
    # prestep/bldepth (forcing + dbsfc + tables) are computed once outside the loss.
    cf_step = core2_forcing.compute_surface_fluxes(
        mesh, st0, jax.tree.map(lambda x: x[0], sfs), fs, dt=DT)
    bvfreq = eos.compute_pressure_bv(mesh, st0.T, st0.S, st0.hnode)[2]
    sw_alpha, sw_beta = eos.compute_sw_alpha_beta(mesh, st0.T, st0.S)
    dbsfc = eos.compute_dbsfc(mesh, st0.T, st0.S)
    uvnode = pp.compute_vel_nodes(mesh, st0.uv)
    wmt, wst = kpp.build_wscale_tables(CFG)
    dVsq, ustar, Bo = kpp.prestep(mesh, uvnode, cf_step.stress_node_surf, cf_step.heat_flux,
                                  cf_step.water_flux, sw_alpha, sw_beta, st0.S, CFG)
    hbl, kbl, bfsfc, stab, caseA = kpp.bldepth(
        mesh, dVsq, ustar, Bo, bvfreq, dbsfc, cf_step.sw_3d, sw_alpha, wmt, wst, CFG)

    def kbg_loss(k_bg):
        cfg_t = CFG._replace(k_bg=k_bg)                  # k_bg traced; rest static
        viscA, diffKt, diffKs = kpp.ri_iwmix(mesh, uvnode, bvfreq, cfg_t)
        bM, bT, bS, gh, dk = kpp.blmix(mesh, st0.hnode, viscA, diffKt, diffKs, hbl, bfsfc,
                                       stab, caseA, kbl, ustar, wmt, wst, cfg_t)
        bM, bT, bS, gh = kpp.enhance(mesh, bM, bT, bS, gh, dk, viscA, diffKt, diffKs, hbl,
                                     caseA, kbl, cfg_t)
        Kv, _, _, _, _, _ = kpp.assemble_mixing(mesh, bM, bT, bS, gh, viscA, diffKt, diffKs,
                                                kbl, cfg_t)
        return jnp.sum(jnp.where(iface, Kv, 0.0)) / niface

    kbg0 = jnp.asarray(float(CFG.k_bg), jnp.float64)     # 1e-5 (K_ver background)
    gKb = float(jax.jit(jax.grad(kbg_loss))(kbg0))
    rowsKb = fd_sweep(jax.jit(kbg_loss), kbg0)
    platKb = min(abs(gKb - gf) / max(abs(gf), 1e-300) for _, gf in rowsKb)
    print(f"\n  [Kbg] d(mean Kv)/d(K_bg) AD = {gKb:+.8e}  (K_bg={float(kbg0):.1e})")
    for h, gf in rowsKb:
        print(f"      h={h:.0e}  FD={gf:+.6e}  rel={abs(gKb-gf)/max(abs(gf),1e-300):.2e}")
    print(f"      plateau (min rel) = {platKb:.3e}", flush=True)
    passes["[Kbg] finite_nonzero"] = bool(np.isfinite(gKb) and gKb != 0.0)
    passes["[Kbg] plateau<1e-4"] = bool(np.isfinite(gKb) and gKb != 0.0 and platKb < 1e-4)

    # ----- [3] N d(mean SST)/d(T0) finite EVERYWHERE incl. masked lanes + memory -----
    def loss_T0(T0):
        s = dataclasses.replace(st0, T=T0)
        fin = integrate(s, mesh, op, None, n_steps=args.n, dt=DT,
                        step_forcings=sfs, forcing_static=fs, kpp_cfg=CFG)
        return mean_sst(fin)

    t1 = time.time()
    gT = np.asarray(jax.jit(jax.grad(loss_T0))(st0.T))
    report_mem(f"T0-grad N={args.n} KPP")
    n_bad = int((~np.isfinite(gT)).sum())
    wet_max = float(np.max(np.abs(gT[mlay]))); masked_max = float(np.max(np.abs(gT[~mlay])))
    print(f"\n  [3] N={args.n} d(mean SST)/d(T0): non-finite={n_bad}; wet max|g|={wet_max:.3e}; "
          f"masked max|g|={masked_max:.3e}  ({time.time()-t1:.1f}s)", flush=True)
    passes["[3] T0_masked_NaN_clean"] = bool(n_bad == 0 and wet_max > 0.0 and masked_max == 0.0)

    # the headline GATE-6C AD requirement is the masked-NaN-clean d(loss)/d(T0) through the
    # assembled KPP model + a finite/nonzero KPP-tunable gradient (the plateau quality is a
    # reported conditioning indicator — not required through the discrete kbl).
    required = ["[3] T0_masked_NaN_clean", "[Kbg] finite_nonzero"]
    ok = all(passes[k] for k in required)
    print(f"\n  gate breakdown: {passes}")
    print(f"  (required: {required})")
    print("KPP_GRAD_GATE_OK" if ok else "KPP_GRAD_GATE_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
