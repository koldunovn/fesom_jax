#!/usr/bin/env python
"""A/B the host forcing cost of the perf/kokkos-m7-levers jra55 fixes on CORE2.

Leg OLD replicates the pre-fix per-step logic IN THE SCRIPT (the never-releasing
clamp `need` test + per-step rotation trig), driving the same reader internals;
leg NEW is the fixed `JRA55Reader.step`. Both walk the same model dates, and the
outputs are asserted BIT-identical (u/v to the last bit — same expressions, same
libm) before any timing is believed.

Two windows, both on the CORE2 mesh (the benchmark mesh):
  * dt=1800, 25 steps  — the standard GPU benchmark protocol window (all of it sat
    inside the old clamp window for prra/prsn; the first 3 steps for all 8 fields).
  * dt=180, 60 steps   — a production-timestep-shaped window (FORCA20/dars run
    dt=120–180; their meshes are 5–25x CORE2, so scale the Δ accordingly).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from fesom_jax import jra55
from fesom_jax.mesh import load_mesh

MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
JRA_DIR = None          # resolved by paths (Levante default)
YEAR = 1958


def step_old(r: jra55.JRA55Reader, year: int, day: int, sec: float):
    """The PRE-FIX step: old `need` logic (clamp never releases) + per-call trig."""
    cal0 = r.fields[0].calendar
    rdate = float(jra55._julday(int(year), 1, 1, cal0)) + float(day - 1) + sec / 86400.0
    vals = []
    for f in r.fields:
        need = (f.t_indx <= 0)
        if not need:
            lo = f.nc_time[f.t_indx - 1]
            hi = f.nc_time[f.t_indx_p1 - 1]
            need = (rdate < lo or rdate > hi)          # the old, never-releasing test
        if need:
            r._getcoeffld(f, rdate)
        vals.append(rdate * f.coef_a + f.coef_b)
    u, v = jra55._vector_g2r(vals[jra55.I_XWIND], vals[jra55.I_YWIND],
                             r.glon, r.glat, r.rlon, r.rlat, r.M)   # no trig cache
    return u, v, vals


def run_leg(dt: float, nsteps: int, old: bool):
    m = load_mesh(MESH)
    r = jra55.JRA55Reader(m, YEAR, JRA_DIR)
    outs, times = [], []
    for i in range(1, nsteps + 1):                     # sec = i*dt, day rolls over 86400
        sec = i * dt
        day = 1 + int(sec // 86400.0)
        sec = sec - (day - 1) * 86400.0
        t0 = time.perf_counter()
        if old:
            u, v, vals = step_old(r, YEAR, day, sec)
            outs.append((u, v, vals[2]))
        else:
            f = r.step(YEAR, day, sec)
            outs.append((f.u_wind, f.v_wind, f.shum))
        times.append(time.perf_counter() - t0)
    r.close()
    return np.asarray(times), outs


def ab(dt: float, nsteps: int, label: str):
    t_old, o_old = run_leg(dt, nsteps, old=True)
    t_new, o_new = run_leg(dt, nsteps, old=False)
    # value gate first: bit-identical or the timing is meaningless
    for i, ((uo, vo, so), (un, vn, sn)) in enumerate(zip(o_old, o_new)):
        assert np.array_equal(uo, un) and np.array_equal(vo, vn) and np.array_equal(so, sn), \
            f"step {i + 1}: OLD and NEW forcing differ — fix is NOT value-identical"
    print(f"[{label}] dt={dt:.0f}s x {nsteps} steps  (values BIT-identical)")
    print(f"  OLD  mean {1e3 * t_old.mean():8.2f} ms/step   total {t_old.sum():6.2f} s")
    print(f"  NEW  mean {1e3 * t_new.mean():8.2f} ms/step   total {t_new.sum():6.2f} s")
    print(f"  Δ    mean {1e3 * (t_old.mean() - t_new.mean()):8.2f} ms/step   "
          f"({100.0 * (1.0 - t_new.sum() / t_old.sum()):.1f} % of the host forcing)")
    per = ", ".join(f"{1e3 * x:.1f}" for x in t_old[:8])
    print(f"  OLD per-step head [ms]: {per}")
    per = ", ".join(f"{1e3 * x:.1f}" for x in t_new[:8])
    print(f"  NEW per-step head [ms]: {per}")


if __name__ == "__main__":
    ab(1800.0, 25, "bench window")
    ab(180.0, 60, "production-dt window")
