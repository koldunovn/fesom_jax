#!/usr/bin/env python
"""CORE2 mEVP forward stability + liveness fields — Phase 9c, JM.3.

Runs the FULL assembled CORE2 model (KPP + GM/Redi + prognostic ice + FCT + thermo, **linfs** —
the mEVP reference config; mEVP's levitating ssh branch covers linfs) forward ``--steps`` jitted
timesteps at ``dt=1800`` for BOTH ice rheologies:

* **mEVP** (``ice_cfg=IceConfig(whichEVP=1)``) — the new option, the stability gate.
* **EVP**  (``ice_cfg=IceConfig()``)          — the control, for the diff-of-diffs liveness.

Stability gate (both must pass): no NaN, ``|SSH|<5 m``, ``max|vel|<3 m/s``, ``m_ice<20 m``.
Sanity direction (NOT a hard gate): the C found mEVP **damps** the ice-velocity transient vs
std-EVP — we report peak ``|uv_ice|`` for both (expect mEVP ≲ EVP).

Saves both runs' final surface fields (sst/sss/a_ice/m_ice/u_ice/v_ice) to ``--out`` for the
diff-of-diffs liveness compare (``core2_mevp_climate_compare.py``): (JAX-mEVP − JAX-EVP) must
pattern-match (C-mEVP − C-EVP).

Usage (GPU; scripts/archive/core2_mevp_stability.sbatch):  python scripts/archive/core2_mevp_stability.py --steps 480
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import surface_forcing, ice, ssh
from fesom_jax import step as stepmod
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import phc_initial_state

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"   # surface bit-identical to the 16-rank C
DT = 1800.0

SSH_ABSMAX = 5.0
VEL_ABSMAX = 3.0
MICE_MAX = 20.0


def build(year):
    mesh = load_mesh(MESH_DIR)
    sst = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    state = ice.seed_ice(phc_initial_state(mesh, IC_DIR), mesh, sst)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, year, sst_ic=sst)
    return mesh, state, op, cf


def make_diag(mesh):
    surf = jnp.asarray(np.asarray(mesh.node_layer_mask)[:, 0])
    em = jnp.asarray(mesh.elem_layer_mask)[:, :, None]
    areasvol0 = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])

    @jax.jit
    def diag(state):
        T0, S0, eta, uv = state.T[:, 0], state.S[:, 0], state.eta_n, state.uv
        finite = (jnp.isfinite(state.T).all() & jnp.isfinite(state.S).all()
                  & jnp.isfinite(uv).all() & jnp.isfinite(eta).all()
                  & jnp.isfinite(state.a_ice).all() & jnp.isfinite(state.m_ice).all()
                  & jnp.isfinite(state.u_ice).all() & jnp.isfinite(state.v_ice).all())
        vel_elem = jnp.max(jnp.where(em, jnp.abs(uv), 0.0), axis=(1, 2))
        ice_speed = jnp.sqrt(state.u_ice ** 2 + state.v_ice ** 2)
        return dict(
            finite=finite,
            sst_min=jnp.min(jnp.where(surf, T0, jnp.inf)),
            sst_max=jnp.max(jnp.where(surf, T0, -jnp.inf)),
            sss_min=jnp.min(jnp.where(surf, S0, jnp.inf)),
            sss_max=jnp.max(jnp.where(surf, S0, -jnp.inf)),
            ssh_absmax=jnp.max(jnp.where(surf, jnp.abs(eta), -jnp.inf)),
            vel_absmax=jnp.max(vel_elem),
            ice_area=jnp.sum(state.a_ice * areasvol0),
            mice_max=jnp.max(state.m_ice),
            aice_max=jnp.max(state.a_ice),
            uice_max=jnp.max(ice_speed),
        )
    return diag


def report(tag, d, dt_step=None):
    t = "" if dt_step is None else f"  [{dt_step:5.2f}s]"
    print(f"{tag:>10}  fin={int(d['finite'])}  "
          f"SST[{d['sst_min']:+6.2f},{d['sst_max']:6.2f}]  SSS[{d['sss_min']:5.2f},{d['sss_max']:5.2f}]  "
          f"|SSH|={d['ssh_absmax']:.2e} |vel|={d['vel_absmax']:.2e}  "
          f"ice[a={d['aice_max']:.3f} m={d['mice_max']:.3f} |uv|={d['uice_max']:.3e}]{t}", flush=True)


def stable(d):
    if not bool(d["finite"]):
        return False, "NaN/Inf"
    if float(d["ssh_absmax"]) >= SSH_ABSMAX:
        return False, f"|SSH|>={SSH_ABSMAX}"
    if float(d["vel_absmax"]) >= VEL_ABSMAX:
        return False, f"max|vel|>={VEL_ABSMAX}"
    if float(d["mice_max"]) >= MICE_MAX:
        return False, f"m_ice>={MICE_MAX}"
    return True, ""


def run(label, ice_cfg, mesh, state, op, cf, dates, diag, steps, every):
    cfgs = dict(kpp_cfg=KppConfig(), gm_cfg=GMConfig(), ice_cfg=ice_cfg)   # linfs (ale_cfg=None)
    fs = cf.static
    print(f"\n========== {label} ({steps} steps, {steps*DT/86400:.2f} days; KPP+GM+ice) ==========",
          flush=True)
    d = jax.device_get(diag(state)); report("init", d)
    worst_vel = 0.0; peak_uice = 0.0
    for i in range(steps):
        sf = cf.step_forcing(*dates[i])
        ts = time.time()
        state = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=(i == 0),
                                 step_forcing=sf, forcing_static=fs, **cfgs)
        d = jax.device_get(diag(state)); dts = time.time() - ts
        worst_vel = max(worst_vel, float(d["vel_absmax"]))
        peak_uice = max(peak_uice, float(d["uice_max"]))
        ok, why = stable(d)
        step = i + 1
        day_bdry = int(step * DT // 86400) > int((step - 1) * DT // 86400)
        if (not ok) or step <= 2 or step % every == 0 or step == steps or day_bdry:
            report(f"s{step}·d{int(step*DT//86400)}" if day_bdry else f"s{step}", d, dts)
        if not ok:
            print(f"\nFAIL ({label}) at step {step} (day {step*DT/86400:.3f}): {why}", flush=True)
            return False, state, worst_vel, peak_uice
    print(f"PASS ({label}): {steps} steps stable; worst max|vel|={worst_vel:.3e}; "
          f"peak |uv_ice|={peak_uice:.3e}", flush=True)
    return True, state, worst_vel, peak_uice


def surf_rms(mesh, a, b):
    surf = np.asarray(mesh.node_layer_mask)[:, 0]
    av = np.asarray(mesh.areasvol)[:, 0] * surf
    da = np.asarray(a[:, 0]) - np.asarray(b[:, 0])
    return float(np.sqrt(np.sum(av * da * da) / np.sum(av)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=480)            # ~10 days at dt=1800
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--every", type=int, default=48)
    ap.add_argument("--out", type=str, default=str(ROOT / "scripts" / "mevp_liveness_fields.npz"))
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, state0, op, cf = build(args.year)
    dates = surface_forcing.dates_for_steps(args.year, DT, args.steps)
    diag = make_diag(mesh)
    print(f"[setup] built in {time.time()-t0:.1f}s", flush=True)

    ok_m, fin_m, wv_m, pk_m = run("mEVP", IceConfig(whichEVP=1), mesh, state0, op, cf,
                                  dates, diag, args.steps, args.every)
    ok_e, fin_e, wv_e, pk_e = run("EVP", IceConfig(), mesh, state0, op, cf,
                                  dates, diag, args.steps, args.every)

    live = False
    if ok_m and ok_e:
        sst_rms = surf_rms(mesh, fin_m.T, fin_e.T)
        sss_rms = surf_rms(mesh, fin_m.S, fin_e.S)
        live = sst_rms > 1e-4                                    # mEVP genuinely differs from EVP
        print(f"\n[mEVP↔EVP after {args.steps*DT/86400:.2f} days] SST RMS={sst_rms:.4e} °C  "
              f"SSS RMS={sss_rms:.4e} psu  (live={'YES' if live else 'no'})", flush=True)
        print(f"[velocity-damping sanity] peak |uv_ice|: mEVP={pk_m:.3e}  EVP={pk_e:.3e}  "
              f"(C direction: mEVP ≲ EVP — {'consistent' if pk_m <= pk_e * 1.05 else 'NOTE: mEVP>EVP'})",
              flush=True)
        # save surface fields for the diff-of-diffs liveness compare
        np.savez(args.out,
                 sst_mevp=np.asarray(fin_m.T[:, 0]), sst_evp=np.asarray(fin_e.T[:, 0]),
                 sss_mevp=np.asarray(fin_m.S[:, 0]), sss_evp=np.asarray(fin_e.S[:, 0]),
                 aice_mevp=np.asarray(fin_m.a_ice), aice_evp=np.asarray(fin_e.a_ice),
                 mice_mevp=np.asarray(fin_m.m_ice), mice_evp=np.asarray(fin_e.m_ice),
                 uice_mevp=np.asarray(fin_m.u_ice), uice_evp=np.asarray(fin_e.u_ice),
                 vice_mevp=np.asarray(fin_m.v_ice), vice_evp=np.asarray(fin_e.v_ice),
                 steps=args.steps, dt=DT)
        print(f"[saved] liveness fields → {args.out}", flush=True)

    ok = ok_m and ok_e and live
    print(f"\n  mEVP stable={ok_m}  EVP stable={ok_e}  live={live}")
    print("MEVP_STABILITY_OK" if ok else "MEVP_STABILITY_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
