#!/usr/bin/env python
"""CORE2 GM/Redi forward stability run — Phase 6B, Task G.7 (GATE 6B).

Runs the **full production** assembled CORE2 model (PP mixing + GM/Redi + prognostic sea
ice) forward ``--steps`` jitted timesteps and monitors numerical stability + the physical
sanity sign that the eddy parameterization is doing work:

* **stability** — no NaN, |SSH|<5 m, max|vel|<3 m/s, SST capped by the ice thermo, m_ice
  bounded (the Phase-6 gate, now with GM/Redi live).
* **front smoothing** — GM's bolus advection + Redi neutral diffusion should *reduce*
  spurious convection and *smooth* mesoscale fronts. We report a front-sharpness proxy
  (the RMS surface |∇T| over wet elements) each day; with ``--gm off`` (a matched run) it
  should be SHARPER, the sign GM is flattening isopycnals rather than just being inert.

``--gm on`` (default) = GM/Redi + ice; ``--gm off`` = ice only (the Phase-6 baseline, for
the front-sharpness comparison). Forcing is generated per step (never stacked — a 1728-step
stack is ~17.5 GB resident, OOMs the A100-40).

Usage:  python scripts/core2_gm_stability_run.py --steps 1728 --every 50          # GM ON
        python scripts/core2_gm_stability_run.py --steps 1728 --every 50 --gm off  # baseline
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import core2_forcing, ice, ssh
from fesom_jax import step as stepmod
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 500.0
ALEUTIAN = 94122

SSH_ABSMAX = 5.0
VEL_ABSMAX = 3.0
SST_CAP_LO = -3.0      # with ice, SST_min should NOT go far below the freezing point (~-1.9)
MICE_MAX = 20.0        # m — ice volume should stay physical


def build(year, steps):
    mesh = load_mesh(MESH_DIR)
    sst = np.asarray(core2_initial_state(mesh, IC_DIR).T[:, 0])
    state = ice.seed_ice(core2_initial_state(mesh, IC_DIR), mesh, sst)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, year, sst_ic=sst)
    dates = core2_forcing.dates_for_steps(year, DT, steps)
    return mesh, state, op, cf, dates


def make_diag(mesh):
    surf = jnp.asarray(np.asarray(mesh.node_layer_mask)[:, 0])
    em = jnp.asarray(mesh.elem_layer_mask)[:, :, None]
    esurf = jnp.asarray(np.asarray(mesh.elem_layer_mask)[:, 0])     # (E,) wet surface elems
    areasvol0 = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])
    gsca = jnp.asarray(mesh.gradient_sca)                            # (E,6)
    enodes = jnp.asarray(mesh.elem_nodes)                           # (E,3)

    @jax.jit
    def diag(state):
        T0, S0, eta, uv = state.T[:, 0], state.S[:, 0], state.eta_n, state.uv
        finite = (jnp.isfinite(state.T).all() & jnp.isfinite(state.S).all()
                  & jnp.isfinite(uv).all() & jnp.isfinite(eta).all()
                  & jnp.isfinite(state.a_ice).all() & jnp.isfinite(state.m_ice).all())
        vel_elem = jnp.max(jnp.where(em, jnp.abs(uv), 0.0), axis=(1, 2))
        ice_area = jnp.sum(state.a_ice * areasvol0)
        ice_speed = jnp.sqrt(state.u_ice ** 2 + state.v_ice ** 2)
        # front-sharpness proxy: RMS surface |∇T| over wet elements (GM smooths ⇒ smaller).
        Te = T0[enodes]                                            # (E,3)
        gx = jnp.sum(gsca[:, 0:3] * Te, axis=1)
        gy = jnp.sum(gsca[:, 3:6] * Te, axis=1)
        gmag2 = jnp.where(esurf, gx * gx + gy * gy, 0.0)
        front_rms = jnp.sqrt(jnp.sum(gmag2) / jnp.maximum(jnp.sum(esurf), 1.0))
        return dict(
            finite=finite,
            sst_min=jnp.min(jnp.where(surf, T0, jnp.inf)),
            sst_max=jnp.max(jnp.where(surf, T0, -jnp.inf)),
            sss_min=jnp.min(jnp.where(surf, S0, jnp.inf)),
            sss_max=jnp.max(jnp.where(surf, S0, -jnp.inf)),
            ssh_absmax=jnp.max(jnp.where(surf, jnp.abs(eta), -jnp.inf)),
            vel_absmax=jnp.max(vel_elem),
            ice_area=ice_area,
            mice_max=jnp.max(state.m_ice),
            aice_max=jnp.max(state.a_ice),
            uice_max=jnp.max(ice_speed),
            front_rms=front_rms,
            ale_sst=T0[ALEUTIAN - 1],
        )
    return diag


def report(tag, d, dt_step=None):
    t = "" if dt_step is None else f"  [{dt_step:5.2f}s]"
    print(f"{tag:>10}  fin={int(d['finite'])}  "
          f"SST[{d['sst_min']:+6.2f},{d['sst_max']:6.2f}]  "
          f"SSS[{d['sss_min']:5.2f},{d['sss_max']:5.2f}]  "
          f"|SSH|={d['ssh_absmax']:.2e} |vel|={d['vel_absmax']:.2e}  "
          f"front|∇T|={d['front_rms']:.3e}  "
          f"ice[area={d['ice_area']:.3e} a={d['aice_max']:.3f} "
          f"m={d['mice_max']:.3f} |u|={d['uice_max']:.2e}]{t}", flush=True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1728)
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--gm", choices=("on", "off"), default="on")
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, state, op, cf, dates = build(args.year, args.steps)
    fs = cf.static
    ice_cfg = IceConfig()
    gm_cfg = GMConfig() if args.gm == "on" else None
    diag = make_diag(mesh)
    print(f"[setup] built in {time.time()-t0:.1f}s; running {args.steps} steps (dt={DT}) "
          f"WITH prognostic ice, GM/Redi {'ON' if gm_cfg else 'OFF (baseline)'}", flush=True)

    d = jax.device_get(diag(state))
    report("init", d)
    front0 = float(d["front_rms"])
    coldest = float(d["sst_min"]); worst_vel = 0.0; front_last = front0
    for i in range(args.steps):
        sf = cf.step_forcing(*dates[i])
        ts = time.time()
        state = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=(i == 0),
                                 step_forcing=sf, forcing_static=fs,
                                 ice_cfg=ice_cfg, gm_cfg=gm_cfg)
        d = jax.device_get(diag(state))
        dts = time.time() - ts
        coldest = min(coldest, float(d["sst_min"])); worst_vel = max(worst_vel, float(d["vel_absmax"]))
        front_last = float(d["front_rms"])
        ok, why = stable(d)
        step = i + 1
        day_bdry = int(step * DT // 86400) > int((step - 1) * DT // 86400)
        if (not ok) or step <= 2 or step % args.every == 0 or step == args.steps or day_bdry:
            report(f"s{step}·d{int(step*DT//86400)}" if day_bdry else f"s{step}", d, dts)
        if not ok:
            print(f"\nFAIL at step {step} (day {step*DT/86400:.3f}): {why}", flush=True)
            return 1

    capped = coldest >= SST_CAP_LO
    print(f"\nPASS: {args.steps} steps stable ({args.steps*DT/86400:.2f} days); "
          f"GM/Redi {'ON' if gm_cfg else 'OFF'}; worst max|vel|={worst_vel:.3e}; "
          f"coldest SST={coldest:+.2f} °C ({'capped ✓' if capped else 'supercooling ✗'}); "
          f"front|∇T| {front0:.3e}→{front_last:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
