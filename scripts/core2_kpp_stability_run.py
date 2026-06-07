#!/usr/bin/env python
"""CORE2 KPP forward stability + climate run — Phase 6C, Task K.9 (GATE 6C end-to-end).

Runs the **full production** assembled CORE2 model (KPP vertical mixing + GM/Redi +
prognostic sea ice — the real CORE2 default config) forward ``--steps`` jitted timesteps
and monitors numerical stability + the discriminating climate sign that KPP is the
*real default* scheme, not the opt-in PP:

* **stability** — no NaN, |SSH|<5 m, max|vel|<3 m/s, m_ice bounded (the Phase-6 gate,
  now with KPP+GM/Redi live).
* **distinct from PP** — KPP and PP are genuinely different mixing schemes; a matched
  PP+GM+ice run from the same IC should give a SST/SSS climate that differs from the KPP
  run by the genuine scheme difference (the C measured C-PP vs Fortran-KPP ≈ 0.085/0.093 °C
  — ~18× the C-KPP-vs-Fortran-KPP residual of 0.005–0.013 °C). We report the KPP↔PP
  surface RMS; it should be ≫ FP noise. ("JAX-KPP ≈ C-KPP" is established at the *step*
  level by the K.8 bit-faithful gate — a multi-day field diverges by FP chaos regardless,
  so the end-to-end C-match is the step gate, not a multi-day field diff.)

``--mixing both`` (default) runs KPP then a matched PP baseline and compares; ``kpp``/``pp``
run one. Forcing is generated per step (never stacked — a long stack OOMs the GPU).

Usage:  python scripts/core2_kpp_stability_run.py --steps 1728 --every 50          # both
        python scripts/core2_kpp_stability_run.py --steps 1728 --mixing kpp
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
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 500.0
ALEUTIAN = 94122

SSH_ABSMAX = 5.0
VEL_ABSMAX = 3.0
SST_CAP_LO = -3.0
MICE_MAX = 20.0


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
    areasvol0 = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])

    @jax.jit
    def diag(state):
        T0, S0, eta, uv = state.T[:, 0], state.S[:, 0], state.eta_n, state.uv
        finite = (jnp.isfinite(state.T).all() & jnp.isfinite(state.S).all()
                  & jnp.isfinite(uv).all() & jnp.isfinite(eta).all()
                  & jnp.isfinite(state.a_ice).all() & jnp.isfinite(state.m_ice).all())
        vel_elem = jnp.max(jnp.where(em, jnp.abs(uv), 0.0), axis=(1, 2))
        ice_area = jnp.sum(state.a_ice * areasvol0)
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
    print(f"{tag:>12}  fin={int(d['finite'])}  "
          f"SST[{d['sst_min']:+6.2f},{d['sst_max']:6.2f}]  "
          f"SSS[{d['sss_min']:5.2f},{d['sss_max']:5.2f}]  "
          f"|SSH|={d['ssh_absmax']:.2e} |vel|={d['vel_absmax']:.2e}  "
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


def run_one(label, mixing, mesh, state, op, cf, dates, diag, steps, every):
    """Run one forward trajectory; return (ok, final_state, summary)."""
    fs = cf.static
    ice_cfg = IceConfig()
    gm_cfg = GMConfig()
    kpp_cfg = KppConfig() if mixing == "kpp" else None
    print(f"\n========== {label}: mixing={mixing.upper()} + GM/Redi + prognostic ice "
          f"({steps} steps, {steps*DT/86400:.2f} days) ==========", flush=True)
    d = jax.device_get(diag(state))
    report("init", d)
    coldest = float(d["sst_min"]); worst_vel = 0.0
    for i in range(steps):
        sf = cf.step_forcing(*dates[i])
        ts = time.time()
        state = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=(i == 0),
                                 step_forcing=sf, forcing_static=fs,
                                 ice_cfg=ice_cfg, gm_cfg=gm_cfg, kpp_cfg=kpp_cfg)
        d = jax.device_get(diag(state))
        dts = time.time() - ts
        coldest = min(coldest, float(d["sst_min"])); worst_vel = max(worst_vel, float(d["vel_absmax"]))
        ok, why = stable(d)
        step = i + 1
        day_bdry = int(step * DT // 86400) > int((step - 1) * DT // 86400)
        if (not ok) or step <= 2 or step % every == 0 or step == steps or day_bdry:
            report(f"s{step}·d{int(step*DT//86400)}" if day_bdry else f"s{step}", d, dts)
        if not ok:
            print(f"\nFAIL ({label}) at step {step} (day {step*DT/86400:.3f}): {why}", flush=True)
            return False, state, dict(coldest=coldest, worst_vel=worst_vel)
    print(f"PASS ({label}): {steps} steps stable; worst max|vel|={worst_vel:.3e}; "
          f"coldest SST={coldest:+.2f} °C", flush=True)
    return True, state, dict(coldest=coldest, worst_vel=worst_vel)


def surf_rms(mesh, a, b):
    """RMS over wet surface nodes of (a-b)[:,0]."""
    surf = np.asarray(mesh.node_layer_mask)[:, 0]
    d = (np.asarray(a)[:, 0] - np.asarray(b)[:, 0])[surf]
    return float(np.sqrt(np.mean(d * d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1728)
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--mixing", choices=("kpp", "pp", "both"), default="both")
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, state0, op, cf, dates = build(args.year, args.steps)
    diag = make_diag(mesh)
    print(f"[setup] built in {time.time()-t0:.1f}s", flush=True)

    results = {}
    finals = {}
    for mixing in (("kpp", "pp") if args.mixing == "both" else (args.mixing,)):
        ok, final, summ = run_one(mixing.upper(), mixing, mesh, state0, op, cf, dates,
                                  diag, args.steps, args.every)
        results[mixing] = ok
        finals[mixing] = final

    all_stable = all(results.values())
    discriminates = True
    if args.mixing == "both" and results.get("kpp") and results.get("pp"):
        sst_rms = surf_rms(mesh, finals["kpp"].T, finals["pp"].T)
        sss_rms = surf_rms(mesh, finals["kpp"].S, finals["pp"].S)
        # the genuine KPP↔PP scheme difference should dwarf FP noise (C class ~0.085 °C);
        # require a clearly-resolved separation after the run.
        discriminates = sst_rms > 1e-3
        print(f"\n[KPP↔PP climate difference after {args.steps*DT/86400:.2f} days] "
              f"surface SST RMS={sst_rms:.4e} °C   SSS RMS={sss_rms:.4e} psu", flush=True)
        print(f"  (the genuine scheme difference; C-class C-PP-vs-KPP ≈ 0.085 °C — "
              f"{'RESOLVED ✓' if discriminates else 'NOT resolved ✗'})", flush=True)

    ok = all_stable and discriminates
    print(f"\n  stability: {results}   discriminates: {discriminates}")
    print("KPP_STABILITY_GATE_OK" if ok else "KPP_STABILITY_GATE_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
