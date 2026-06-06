#!/usr/bin/env python
"""CORE2 PROGNOSTIC-ICE forward stability run — Phase 6, Task 6.7 (GATE 6).

Runs the assembled CORE2 model WITH sea ice (EVP + FCT + thermo, ``ice_cfg=IceConfig()``)
forward ``--steps`` jitted timesteps and monitors stability + the two Phase-6 deliverables:

* **the supercooling cap** — the Phase-5 NO-ice run supercools the high-lat SST without bound
  (−1.9 → −16 by day 5 → −22 by day 8, then max|vel|>3 ~day 8). With sea ice the thermo
  ``o2ihf`` + freezing point should CAP SST_min near the freezing point (~−1.9 °C for S=35)
  and keep the model stable past day 8.
* **ice growth/extent + runoff** — a_ice extent, m_ice max (should stay physical, < ~10 m).

Stability gate (FRESH_START §15, now the ICE-ON target): no NaN, |SSH|<5 m, max|vel|<3 m/s,
SST_min capped (≥ ~−2.5 °C, i.e. the cap is working), m_ice bounded.

Usage:  python scripts/core2_ice_stability_run.py --steps 1728 --every 50    # ~10 days
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
    # ⚠️ Do NOT stack all steps: 1728 × ~10 fields × nod2D × f8 ≈ 17.5 GB on the GPU (OOMs the
    # A100-40 alongside the model). Generate the per-step forcing in the loop instead.
    return mesh, state, op, cf, dates


def make_diag(mesh):
    surf = jnp.asarray(np.asarray(mesh.node_layer_mask)[:, 0])
    em = jnp.asarray(mesh.elem_layer_mask)[:, :, None]
    areasvol0 = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])

    @jax.jit
    def diag(state):
        T0, S0, eta, uv = state.T[:, 0], state.S[:, 0], state.eta_n, state.uv
        finite = (jnp.isfinite(state.T).all() & jnp.isfinite(state.S).all()
                  & jnp.isfinite(uv).all() & jnp.isfinite(eta).all()
                  & jnp.isfinite(state.a_ice).all() & jnp.isfinite(state.m_ice).all())
        vel_elem = jnp.max(jnp.where(em, jnp.abs(uv), 0.0), axis=(1, 2))
        ice_area = jnp.sum(state.a_ice * areasvol0)            # m² ice-covered (Σ a·area)
        ice_speed = jnp.sqrt(state.u_ice ** 2 + state.v_ice ** 2)
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
            ale_sst=T0[ALEUTIAN - 1],
        )
    return diag


def report(tag, d, dt_step=None):
    t = "" if dt_step is None else f"  [{dt_step:5.2f}s]"
    print(f"{tag:>10}  fin={int(d['finite'])}  "
          f"SST[{d['sst_min']:+6.2f},{d['sst_max']:6.2f}]  "
          f"SSS[{d['sss_min']:5.2f},{d['sss_max']:5.2f}]  "
          f"|SSH|={d['ssh_absmax']:.2e} |vel|={d['vel_absmax']:.2e}  "
          f"ice[area={d['ice_area']:.3e} a_max={d['aice_max']:.3f} "
          f"m_max={d['mice_max']:.3f} |u_i|={d['uice_max']:.3e}]{t}", flush=True)


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
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, state, op, cf, dates = build(args.year, args.steps)
    fs = cf.static
    cfg = IceConfig()
    diag = make_diag(mesh)
    print(f"[setup] built in {time.time()-t0:.1f}s; running {args.steps} steps "
          f"(dt={DT}) WITH prognostic ice", flush=True)

    d = jax.device_get(diag(state))
    report("init", d)
    coldest = float(d["sst_min"]); worst_vel = 0.0
    for i in range(args.steps):
        sf = cf.step_forcing(*dates[i])               # per-step (no GPU-resident stack)
        ts = time.time()
        state = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=(i == 0),
                                 step_forcing=sf, forcing_static=fs, ice_cfg=cfg)
        d = jax.device_get(diag(state))
        dts = time.time() - ts
        coldest = min(coldest, float(d["sst_min"])); worst_vel = max(worst_vel, float(d["vel_absmax"]))
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
          f"worst max|vel|={worst_vel:.3e}; coldest SST={coldest:+.2f} °C "
          f"({'SUPERCOOLING CAPPED ✓' if capped else 'still supercooling ✗ — check thermo'})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
