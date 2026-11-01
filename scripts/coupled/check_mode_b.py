#!/usr/bin/env python
"""Mode B check of the coupled ocean (fesom_jax.coupled): the atmosphere-array forcing provider against the
file-driven run, on one GPU, from the coupled configuration's initial state (configs/core2_cpl.yaml).

1. bitwise: ``--days-bitwise`` days of coupling intervals (4 steps each). Reference = the file-driven chunk call
   of run_from_config (SurfaceForcing.stack -> partition_step_forcing -> run_steps_sharded_forced) on the same
   chunks; test = ocean_interval with the JRA55-do fields AS THE ATMOSPHERE WOULD SEND THEM (the reader's raw
   interpolated values, before any unit conversion or rotation: u10/v10 geographic, t2m in K, rain/snow in
   kg m-2 s-1), one row a step. Both call the same compiled program; every State leaf is compared bit for bit
   after every interval.
2. interval means: from there to the end of the month, the reference continues file-driven per step and the test
   gets each interval's mean of the four per-step rows, held constant over the interval (the coupled contract).
   SST and sea-ice area/volume of the two are compared at every day; the test writes monthly means.
3. ocean_to_atm every day (finite, ranges), a restart round trip, and the steady ms/step of ocean_interval
   (pre-built device forcing, synced once per block).
"""
from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import jra55, shard_mesh, surface_forcing, zarr_output
from fesom_jax import precision as PR
from fesom_jax.coupled import init_ocean, ocean_interval, ocean_to_atm, write_ocean_restart, flush_monthly
from fesom_jax.integrate_sharded import run_steps_sharded_forced


def raw_fields(jra, dates, point_slope):
    """The atmosphere's view of JRA55-do at each step: the time-interpolated values of JRA55Reader.step before
    the rotation and the unit conversions (its lines, verbatim), as mode B's contract fields."""
    rows = {k: [] for k in ("u10", "v10", "q2m", "swdn", "lwdn", "t2m", "rain", "snow")}
    for (year, day, sec, _m) in dates:
        cal0 = jra.fields[0].calendar
        rdate = (float(jra55._julday(int(year), 1, 1, cal0)) + float(day - 1) + sec / 86400.0
                 + float(jra._midstep(year, day, sec)))
        vals = []
        for f in jra.fields:
            if jra._needs_refresh(f, rdate):
                jra._getcoeffld(f, rdate)
            if point_slope:
                el = np.float32(rdate - f.t0)
                vals.append(np.asarray(f.d1, np.float32) + el * np.asarray(f.coef_a, np.float32))
            else:
                vals.append(rdate * f.coef_a + f.coef_b)
        for k, i in (("u10", jra55.I_XWIND), ("v10", jra55.I_YWIND), ("q2m", jra55.I_HUMI),
                     ("swdn", jra55.I_QSR), ("lwdn", jra55.I_QLW), ("t2m", jra55.I_TAIR),
                     ("rain", jra55.I_PREC), ("snow", jra55.I_SNOW)):
            rows[k].append(vals[i])
    return {k: np.stack(v) for k, v in rows.items()}


def leaves_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    bad = []
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(a)]
    for n, x, y in zip(names, la, lb):
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x.view(np.uint8), y.view(np.uint8)):
            d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.shape == y.shape else -1
            bad.append((n, d))
    return bad


def ice_area(st, state):
    a = np.asarray(state.a_ice, np.float64)
    m = np.asarray(state.m_ice, np.float64)
    area = np.asarray(st.mesh.areasvol[:, 0], np.float64)
    lat = np.asarray(st.mesh.geo_coord_nod2D[:, 1])
    nh = lat > 0
    return {"area_nh_1e12m2": float((a * area)[nh].sum() / 1e12), "area_sh_1e12m2": float((a * area)[~nh].sum() / 1e12),
            "vol_nh_1e12m3": float((m * area)[nh].sum() / 1e12), "vol_sh_1e12m3": float((m * area)[~nh].sum() / 1e12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/core2_cpl.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--days-bitwise", type=int, default=1)
    ap.add_argument("--days-total", type=int, default=31)
    ap.add_argument("--time-intervals", type=int, default=48)
    a = ap.parse_args()
    t0 = time.perf_counter()
    state, st = init_ocean(a.config, sss_restoring=True, monthly_out=f"{a.out}/monthly_modeB")
    sp = st.precision == "single"
    year = st.dates(1)[0][0]
    ref_forcing = surface_forcing.build_surface_forcing(st.mesh, year, dt=st.dt)
    ref_forcing.point_slope = sp
    jra = jra55.JRA55Reader(st.mesh, year)
    surface_forcing.set_midstep(jra, st.dt, True)
    print(f"[check] setup {time.perf_counter() - t0:.1f}s, start {st.calendar()}", flush=True)

    ipd = int(round(86400 / (st.dt * st.nsteps)))
    state_ref = state
    report = {"bitwise": [], "means": []}
    n_bit = a.days_bitwise * ipd
    for i in range(a.days_total * ipd):
        dates = st.dates(st.nsteps)
        seq_p = shard_mesh.partition_step_forcing(ref_forcing.stack(dates, xp=np), st.part)
        out = run_steps_sharded_forced(
            st.sm, state_ref, st.sop, st.stress_p, seq_p, st.fs_p, st.nsteps, dt=st.dt, npes=1,
            bootstrap_ab2=st.bootstrap, state_is_folded=True, return_folded=True, boundary_node_p=st.bn_p,
            reuse_executable=True, sample_fn=st.sample_fn, ssh_gather=st.gather, precision=st.precision, **st.phys)
        state_ref = out[0] if st.sample_fn is not None else out
        raw = raw_fields(jra, dates, sp)
        if i < n_bit:
            forcing = raw                                        # one row a step
        else:
            forcing = {k: v.mean(axis=0, dtype=np.float64).astype(v.dtype) for k, v in raw.items()}
        state = ocean_interval(state, st, forcing)
        if i < n_bit:
            bad = leaves_equal(state, state_ref)
            report["bitwise"].append({"interval": i, "n_differing_leaves": len(bad), "worst": bad[:5]})
            print(f"[check] interval {i} {st.calendar()} bitwise: {'OK' if not bad else bad[:5]}", flush=True)
        if (i + 1) % ipd == 0:
            sst, sst_r = np.asarray(state.T[:, 0], np.float64), np.asarray(state_ref.T[:, 0], np.float64)
            o2a = ocean_to_atm(state, st)
            o2a_rng = {k: [float(jnp.min(v)), float(jnp.max(v)), bool(jnp.all(jnp.isfinite(v)))]
                       for k, v in o2a.items()}
            day = {"day": (i + 1) // ipd, "calendar": st.calendar(),
                   "sst_mean": float(sst.mean()), "sst_ref_mean": float(sst_r.mean()),
                   "sst_maxabs_diff": float(np.abs(sst - sst_r).max()),
                   "sst_rms_diff": float(np.sqrt(((sst - sst_r) ** 2).mean())),
                   "ice": ice_area(st, state), "ice_ref": ice_area(st, state_ref),
                   "finite": bool(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state))),
                   "o2a": o2a_rng}
            report["means"].append(day)
            print(f"[check] day {day['day']} {day['calendar']} SST {day['sst_mean']:.5f} ref {day['sst_ref_mean']:.5f}"
                  f" max|d| {day['sst_maxabs_diff']:.3e} rms {day['sst_rms_diff']:.3e} ice NH "
                  f"{day['ice']['area_nh_1e12m2']:.4f}/{day['ice_ref']['area_nh_1e12m2']:.4f} SH "
                  f"{day['ice']['area_sh_1e12m2']:.4f}/{day['ice_ref']['area_sh_1e12m2']:.4f} finite {day['finite']}",
                  flush=True)
    flush_monthly(st)
    print("[check] o2a ranges at the end: " + json.dumps(report["means"][-1]["o2a"]), flush=True)

    # restart round trip
    rpath = write_ocean_restart(state, st, f"{a.out}/restart_modeB")
    back, meta = zarr_output.read_restart(rpath, st.mesh, st.part)
    back = PR.cast_state(back) if sp else back
    rt = leaves_equal(back, state)
    report["restart_roundtrip_differing_leaves"] = rt
    print(f"[check] restart round trip {rpath} step {meta['step']}: {'OK' if not rt else rt}", flush=True)

    # steady ms/step of ocean_interval (device forcing, one sync per block)
    fdev = {k: jnp.asarray(v) for k, v in forcing.items()}
    for _ in range(2):
        state = ocean_interval(state, st, fdev)
    jax.block_until_ready(state)
    tt = time.perf_counter()
    for _ in range(a.time_intervals):
        state = ocean_interval(state, st, fdev)
    jax.block_until_ready(state)
    ms = (time.perf_counter() - tt) / (a.time_intervals * st.nsteps) * 1e3
    report["ms_per_step_ocean_interval"] = ms
    print(f"[check] ocean_interval steady {ms:.2f} ms/step over {a.time_intervals} intervals", flush=True)
    with open(f"{a.out}/check_mode_b.json", "w") as fh:
        json.dump(report, fh, indent=1)
    nbad = sum(r["n_differing_leaves"] for r in report["bitwise"])
    print(f"[check] DONE bitwise intervals {len(report['bitwise'])}, differing {nbad}; "
          f"total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
