#!/usr/bin/env python
"""CORE2 zstar (moving vertical coordinate) forward stability run — Phase 9a, JZ.8 (GATE 9a).

Runs the FULL assembled CORE2 model with the **z2_cdump 4-config** — KPP + GM/Redi + prognostic
ice + **zstar** (``ale_cfg=AleConfig()``) — forward ``--steps`` jitted timesteps at the zstar
reference ``dt=1800`` and monitors stability + the zstar-specific health signals (the model spins
the SSH up from rest, so the column stretches dynamically — the test is that it stays bounded):

* **no NaN / bounded fields** — ``|SSH|<5 m``, ``max|vel|<3 m/s``, ``m_ice`` bounded (the
  FRESH_START §15 gate, the ICE-ON target).
* **layer thickness stays POSITIVE** — the zstar failure mode is a column COLLAPSE (``hnode→0``
  or negative) when the SSH change exceeds the stretchable depth; ``min(hnode over wet)`` must
  stay ``> 0`` every step (a negative thickness ⇒ the moving coordinate has inverted).
* **SSH/steric conservation** — the area-weighted global-mean ``hbar`` (SSH) should stay near its
  initial value (no spurious volume drift under the real freshwater fluxes); track
  ``|⟨hbar⟩ − ⟨hbar⟩₀|`` + the SSH range ``|hbar|_max``.

Gate: no NaN, ``|SSH|<5``, ``max|vel|<3``, ``m_ice<20``, ``min hnode>0``, ``|⟨hbar⟩ drift|<0.1 m``.

Usage:  python scripts/archive/core2_zstar_stability.py --steps 480 --every 20    # ~10 days at dt=1800
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
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"   # the zstar-canonical IC (surface bit-identical to C)
DT = 1800.0
ALEUTIAN = 94122

SSH_ABSMAX = 5.0
VEL_ABSMAX = 3.0
MICE_MAX = 20.0        # m — ice volume should stay physical
HBAR_DRIFT_MAX = 0.1   # m — global-mean SSH should not drift (volume conservation)


def build(year):
    mesh = load_mesh(MESH_DIR)
    sst = np.asarray(core2_initial_state(mesh, IC_DIR).T[:, 0])
    state = ice.seed_ice(core2_initial_state(mesh, IC_DIR), mesh, sst)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, year, sst_ic=sst)
    return mesh, state, op, cf


def make_diag(mesh):
    surf = jnp.asarray(np.asarray(mesh.node_layer_mask)[:, 0])
    nlm = jnp.asarray(mesh.node_layer_mask)                    # (nod2D, nl) wet layers
    em = jnp.asarray(mesh.elem_layer_mask)[:, :, None]
    areasvol0 = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])
    oa = float(np.sum(np.asarray(mesh.areasvol)[:, 0] * np.asarray(mesh.node_layer_mask)[:, 0]))

    @jax.jit
    def diag(state):
        T0, S0, eta, uv = state.T[:, 0], state.S[:, 0], state.eta_n, state.uv
        finite = (jnp.isfinite(state.T).all() & jnp.isfinite(state.S).all()
                  & jnp.isfinite(uv).all() & jnp.isfinite(eta).all()
                  & jnp.isfinite(state.a_ice).all() & jnp.isfinite(state.m_ice).all()
                  & jnp.isfinite(state.hnode).all() & jnp.isfinite(state.hbar).all())
        vel_elem = jnp.max(jnp.where(em, jnp.abs(uv), 0.0), axis=(1, 2))
        ice_area = jnp.sum(state.a_ice * areasvol0)
        ice_speed = jnp.sqrt(state.u_ice ** 2 + state.v_ice ** 2)
        # zstar: min layer thickness over WET layers (the collapse failure mode), SSH drift.
        hnode_wet_min = jnp.min(jnp.where(nlm, state.hnode, jnp.inf))
        hbar_mean = jnp.sum(state.hbar * areasvol0 * surf) / oa     # area-weighted ⟨SSH⟩
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
            hnode_min=hnode_wet_min,
            hbar_mean=hbar_mean,
            hbar_absmax=jnp.max(jnp.abs(state.hbar)),
            ale_sst=T0[ALEUTIAN - 1],
        )
    return diag


def report(tag, d, hbar0, dt_step=None):
    t = "" if dt_step is None else f"  [{dt_step:5.2f}s]"
    drift = float(d["hbar_mean"]) - hbar0
    print(f"{tag:>10}  fin={int(d['finite'])}  "
          f"SST[{d['sst_min']:+6.2f},{d['sst_max']:6.2f}]  "
          f"SSS[{d['sss_min']:5.2f},{d['sss_max']:5.2f}]  "
          f"|SSH|={d['ssh_absmax']:.2e} |vel|={d['vel_absmax']:.2e}  "
          f"hnode_min={d['hnode_min']:.3e} |hbar|={d['hbar_absmax']:.3e} drift={drift:+.2e}  "
          f"ice[a={d['aice_max']:.3f} m={d['mice_max']:.3f}]{t}", flush=True)


def stable(d, hbar0):
    if not bool(d["finite"]):
        return False, "NaN/Inf"
    if float(d["ssh_absmax"]) >= SSH_ABSMAX:
        return False, f"|SSH|>={SSH_ABSMAX}"
    if float(d["vel_absmax"]) >= VEL_ABSMAX:
        return False, f"max|vel|>={VEL_ABSMAX}"
    if float(d["mice_max"]) >= MICE_MAX:
        return False, f"m_ice>={MICE_MAX}"
    if float(d["hnode_min"]) <= 0.0:
        return False, f"hnode_min<=0 (column collapse: {float(d['hnode_min']):.3e})"
    if abs(float(d["hbar_mean"]) - hbar0) >= HBAR_DRIFT_MAX:
        return False, f"|⟨hbar⟩ drift|>={HBAR_DRIFT_MAX} ({float(d['hbar_mean'])-hbar0:+.3e})"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=480)        # ~10 days at dt=1800
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, state, op, cf = build(args.year)
    fs = cf.static
    dates = core2_forcing.dates_for_steps(args.year, DT, args.steps)
    cfgs = dict(kpp_cfg=KppConfig(), gm_cfg=GMConfig(), ice_cfg=IceConfig(), ale_cfg=AleConfig())
    diag = make_diag(mesh)
    print(f"[setup] built in {time.time()-t0:.1f}s; running {args.steps} steps (dt={DT}) "
          f"— KPP + GM/Redi + ice + ZSTAR", flush=True)

    d = jax.device_get(diag(state))
    hbar0 = float(d["hbar_mean"])
    report("init", d, hbar0)
    worst_vel = 0.0
    min_hnode = float(d["hnode_min"])
    max_drift = 0.0
    for i in range(args.steps):
        sf = cf.step_forcing(*dates[i])                      # per-step (no GPU-resident stack)
        ts = time.time()
        state = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=(i == 0),
                                 step_forcing=sf, forcing_static=fs, **cfgs)
        d = jax.device_get(diag(state))
        dts = time.time() - ts
        worst_vel = max(worst_vel, float(d["vel_absmax"]))
        min_hnode = min(min_hnode, float(d["hnode_min"]))
        max_drift = max(max_drift, abs(float(d["hbar_mean"]) - hbar0))
        ok, why = stable(d, hbar0)
        step = i + 1
        day_bdry = int(step * DT // 86400) > int((step - 1) * DT // 86400)
        if (not ok) or step <= 2 or step % args.every == 0 or step == args.steps or day_bdry:
            report(f"s{step}·d{int(step*DT//86400)}" if day_bdry else f"s{step}", d, hbar0, dts)
        if not ok:
            print(f"\nFAIL at step {step} (day {step*DT/86400:.3f}): {why}", flush=True)
            return 1

    print(f"\nPASS: {args.steps} steps stable ({args.steps*DT/86400:.2f} days, dt={DT}); "
          f"worst max|vel|={worst_vel:.3e}; min hnode={min_hnode:.3e} m (>0 ⇒ no column collapse); "
          f"max |⟨hbar⟩ drift|={max_drift:.2e} m (<{HBAR_DRIFT_MAX} ⇒ volume conserved)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
