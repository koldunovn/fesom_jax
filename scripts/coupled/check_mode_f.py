#!/usr/bin/env python
"""Mode F check of the coupled ocean (fesom_jax.coupled, FESOM2's coupled flux path): one month on one GPU from the
coupled configuration's initial state, driven by fluxes an "offline atmosphere" computes every coupling interval
from JRA55-do with the CORE2 bulk formulae of fesom_jax and the ocean's current surface:

* open water (L&Y09 ncar bulk, forcing.bulk_surface_fluxes / obudget at the current SST and surface current):
  heat_swo = (1 - 0.1) * rsds, heat_oce = -qns (non-solar, + down), evap_oce = the bulk evaporation [m s-1 fresh
  water, + down], taux/tauy_oce = the bulk stress (rotated to geographic, as OASIS would deliver it);
* sea ice (fesom_jax's standalone ice-surface budget ice_thermo._budget: the 5-iteration Newton skin temperature
  with conduction through the ice and snow per ice area): tsk_ico = the skin [K], heat_ico = the net atmospheric
  flux at that skin [W m-2, + down], masked to a_ice > 0 (as the fork's ZMASK), subl_oce = the sublimation
  [m s-1 fresh water, + down]; taux/tauy_ico = forcing.atm_ice_stress with the current ice velocity;
* prec_oce / snow_oce = prra / prsn / 1000 [m s-1].

Each interval's atmosphere is the mean of its four per-step JRA55-do values. Reported every day: SST, sea-ice
area/volume NH/SH, finite, the ocean -> atmosphere ranges; at the end the same against the standalone file-driven
month (configs/core2_cpl.yaml, job ocn_t1), the restart round trip with the mode F fields, the steady ms/step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mode_b import ice_area, leaves_equal, raw_fields   # noqa: E402

from fesom_jax import forcing as F                              # noqa: E402
from fesom_jax import ice_thermo, jra55, surface_forcing, zarr_output   # noqa: E402
from fesom_jax import precision as PR                           # noqa: E402
from fesom_jax.coupled import flush_monthly, init_ocean, ocean_interval, ocean_to_atm, write_ocean_restart  # noqa
from fesom_jax.coupled.ocean import g2r, r2g                    # noqa: E402


def offline_atmosphere(state, st, atm):
    """The mode F fields (OASIS level, geographic stresses, float32) from the interval-mean JRA55-do fields ``atm``
    (contract names of mode B, float64) and the ocean's current surface."""
    f64 = lambda x: jnp.asarray(x, jnp.float64)                 # noqa: E731
    cfg = st.cfg.ice
    u, v = g2r(f64(atm["u10"]), f64(atm["v10"]), st.trig, st.M)
    tair = f64(atm["t2m"]) - 273.15
    q, sw, lw = f64(atm["q2m"]), f64(atm["swdn"]), f64(atm["lwdn"])
    rain, snow = f64(atm["rain"]) / 1000.0, f64(atm["snow"]) / 1000.0
    sst = f64(state.T[:, 0])
    uw, vw = f64(state.uvnode[:, 0, 0]), f64(state.uvnode[:, 0, 1])
    bulk = F.bulk_surface_fluxes(st.mesh, u, v, q, sw, lw, tair, rain, snow, sst, uw, vw)
    qsr = (1.0 - F.BULK_ALBW) * sw
    qns_up = bulk.heat_flux + qsr                               # heat_flux = qns - qsr (+ up)
    evap_m = bulk.water_flux + rain + snow                      # water_flux = evap - prec, evap [m s-1 at rhowat]
    evap_fw = evap_m / (F.BULK_INV_RHOWAT * 1000.0)             # -> fresh water [m s-1], + down (qa - qsat)
    tox, toy = r2g(bulk.stress_node_surf[:, 0], bulk.stress_node_surf[:, 1], st.trig, st.M)
    tix_m, tiy_m = F.atm_ice_stress(u, v, f64(state.u_ice), f64(state.v_ice))
    tix, tiy = r2g(tix_m, tiy_m, st.trig, st.M)
    a = f64(state.a_ice)
    hice = jnp.where(a > 0.0, f64(state.m_ice) / jnp.maximum(a, 1e-12), 0.0)
    hsn = jnp.where(a > 0.0, f64(state.m_snow) / jnp.maximum(a, 1e-12), 0.0)
    t0 = jnp.minimum(f64(state.cpl_ice_temp) - 273.15, 0.0)
    ug = jnp.sqrt(u * u + v * v)
    ts, fh, subli = ice_thermo._budget(cfg, hice, hsn, t0, tair, q, sw, lw, ug, f64(state.S[:, 0]),
                                       cfg.ch_atm_ice, cfg.ce_atm_ice)
    q_ice = -fh * cfg.cl                                        # hfrad + hfsen + hflat (+ down)
    on = a > 0.0
    out = {"taux_oce": tox, "tauy_oce": toy, "taux_ico": tix, "tauy_ico": tiy,
           "prec_oce": rain, "snow_oce": snow, "evap_oce": evap_fw,
           "subl_oce": jnp.where(on, subli * cfg.rhowat / 1000.0, 0.0),
           "heat_oce": -qns_up, "heat_swo": qsr, "heat_ico": jnp.where(on, q_ice, 0.0),
           "tsk_ico": ts + 273.15}
    return {k: x.astype(jnp.float32) for k, x in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/core2_cpl.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=31)
    ap.add_argument("--ref-restart", default="/work/ba1583/a270088/awicm/ocn/runs/t1_standalone_1989_01/restart")
    ap.add_argument("--time-intervals", type=int, default=48)
    a = ap.parse_args()
    t0 = time.perf_counter()
    state, st = init_ocean(a.config, mode="flux", monthly_out=f"{a.out}/monthly_modeF")
    sp = st.precision == "single"
    year = st.dates(1)[0][0]
    jra = jra55.JRA55Reader(st.mesh, year)
    surface_forcing.set_midstep(jra, st.dt, True)
    atm_fn = jax.jit(lambda s, x: offline_atmosphere(s, st, x))
    print(f"[checkF] setup {time.perf_counter() - t0:.1f}s, start {st.calendar()}", flush=True)
    ipd = int(round(86400 / (st.dt * st.nsteps)))
    report = {"days": []}
    for i in range(a.days * ipd):
        raw = raw_fields(jra, st.dates(st.nsteps), sp)
        atm = {k: v.mean(axis=0, dtype=np.float64) for k, v in raw.items()}
        fx = atm_fn(state, atm)
        state = ocean_interval(state, st, fx)
        if (i + 1) % ipd == 0:
            o2a = ocean_to_atm(state, st)
            rng = {k: [float(jnp.min(x)), float(jnp.max(x)), bool(jnp.all(jnp.isfinite(x)))] for k, x in o2a.items()}
            fr = {k: [float(jnp.min(x)), float(jnp.max(x)), float(jnp.mean(x))] for k, x in fx.items()}
            sst = np.asarray(state.T[:, 0], np.float64)
            day = {"day": (i + 1) // ipd, "calendar": st.calendar(), "sst_mean": float(sst.mean()),
                   "sst_min": float(sst.min()), "ice": ice_area(st, state),
                   "finite": bool(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state))),
                   "o2a": rng, "forcing": fr}
            report["days"].append(day)
            print(f"[checkF] day {day['day']} {day['calendar']} SST {day['sst_mean']:.4f} (min {day['sst_min']:.3f})"
                  f" ice NH {day['ice']['area_nh_1e12m2']:.4f} SH {day['ice']['area_sh_1e12m2']:.4f} "
                  f"vol NH {day['ice']['vol_nh_1e12m3']:.4f} SH {day['ice']['vol_sh_1e12m3']:.4f} "
                  f"finite {day['finite']}", flush=True)
            if not day["finite"]:
                break
    flush_monthly(st)
    print("[checkF] o2a at the end: " + json.dumps(report["days"][-1]["o2a"]), flush=True)
    print("[checkF] forcing at the end (min, max, mean): " + json.dumps(report["days"][-1]["forcing"]), flush=True)
    # against the standalone file-driven month
    if os.path.isdir(a.ref_restart):
        Tr = zarr_output.reconstruct_global(a.ref_restart, "T")[:, 0].astype(np.float64)
        ar = zarr_output.reconstruct_global(a.ref_restart, "a_ice").astype(np.float64)
        mr = zarr_output.reconstruct_global(a.ref_restart, "m_ice").astype(np.float64)
        area = np.asarray(st.mesh.areasvol[:, 0], np.float64)
        lat = np.asarray(st.mesh.geo_coord_nod2D[:, 1])
        nh = lat > 0
        sst = np.asarray(state.T[:, 0], np.float64)
        cmp = {"ref_restart": a.ref_restart, "sst_mean": float(sst.mean()), "sst_ref_mean": float(Tr.mean()),
               "sst_area_mean": float((sst * area).sum() / area.sum()),
               "sst_ref_area_mean": float((Tr * area).sum() / area.sum()),
               "sst_rms_diff": float(np.sqrt((((sst - Tr) ** 2) * area).sum() / area.sum())),
               "sst_maxabs_diff": float(np.abs(sst - Tr).max()),
               "ice_area_nh_ref": float((ar * area)[nh].sum() / 1e12), "ice_area_sh_ref": float((ar * area)[~nh].sum() / 1e12),
               "ice_vol_nh_ref": float((mr * area)[nh].sum() / 1e12), "ice_vol_sh_ref": float((mr * area)[~nh].sum() / 1e12),
               "ice": report["days"][-1]["ice"]}
        report["vs_standalone"] = cmp
        print("[checkF] vs standalone month: " + json.dumps(cmp), flush=True)
    rpath = write_ocean_restart(state, st, f"{a.out}/restart_modeF")
    back, meta = zarr_output.read_restart(rpath, st.mesh, st.part)
    back = PR.cast_state(back) if sp else back
    rt = leaves_equal(back, state)
    report["restart_roundtrip_differing_leaves"] = rt
    print(f"[checkF] restart round trip {rpath} step {meta['step']}: {'OK' if not rt else rt}", flush=True)
    fdev = fx
    for _ in range(2):
        state = ocean_interval(state, st, fdev)
    jax.block_until_ready(state)
    tt = time.perf_counter()
    for _ in range(a.time_intervals):
        state = ocean_interval(state, st, fdev)
    jax.block_until_ready(state)
    ms = (time.perf_counter() - tt) / (a.time_intervals * st.nsteps) * 1e3
    report["ms_per_step_ocean_interval"] = ms
    print(f"[checkF] ocean_interval steady {ms:.2f} ms/step over {a.time_intervals} intervals", flush=True)
    with open(f"{a.out}/check_mode_f.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"[checkF] DONE total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
