#!/usr/bin/env python
"""Smoke check of the coupled CORE3 ocean with cavities (mode F): init_ocean(configs/core3_cpl.yaml) from the
converted jaxcpl 1855 restart, a few coupling intervals with a fixed, plausible atmosphere (no atmosphere model),
finite at every interval, no sea ice under the shelves, the shelf-base layer evolving.

    python scripts/core3/check_cpl_core3.py [--intervals 3] [--nsteps 2] [--partition serial|dist_2] [--config ...]

Prints ``CPL_CORE3_OK`` when every interval is finite and the shelves stay ice-free.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "core3_cpl.yaml"))
    ap.add_argument("--intervals", type=int, default=3)
    ap.add_argument("--nsteps", type=int, default=2)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--year0", type=int, default=1855)
    a = ap.parse_args()
    import jax.numpy as jnp
    from fesom_jax.coupled import init_ocean, ocean_interval

    t0 = time.perf_counter()
    state, st = init_ocean(a.config, mode="flux", year0=a.year0, nsteps=a.nsteps, partition=a.partition)
    print(f"[cpl] init {time.perf_counter() - t0:.1f}s, calendar {st.calendar()}, npes {st.npes}", flush=True)
    n = int(st.nod2D)
    mesh = st.mesh
    uln = np.asarray(mesh.ulevels_nod2D)
    cav = uln > 1
    if st.npes == 1:
        full = lambda x: np.asarray(x, np.float64)[:n]           # serial: the folded state is global order   # noqa
    else:                                                        # folded [P*Lmax]: each node's owner lane
        from fesom_jax.zarr_output import _folded_gid_owned
        gid, owned = _folded_gid_owned(st.part, st.sm, "nod")
        lane = np.zeros(n, np.int64); lane[gid[owned]] = np.flatnonzero(owned)
        full = lambda x: np.asarray(x, np.float64)[lane]                                                   # noqa
    z = np.zeros(n)
    forcing = dict(taux_oce=z + 0.05, tauy_oce=z, taux_ico=z + 0.05, tauy_ico=z, prec_oce=z + 2e-8,
                   snow_oce=z + 5e-9, evap_oce=z - 2e-8, subl_oce=z, heat_oce=z - 60.0, heat_ico=z - 20.0,
                   heat_swo=z + 120.0, tsk_ico=z + 265.0)
    T_top0 = full(state.T)[np.arange(n), uln - 1]
    ok = True
    for i in range(a.intervals):
        t1 = time.perf_counter()
        state = ocean_interval(state, st, forcing)
        state.T.block_until_ready()
        dtw = time.perf_counter() - t1
        T, aice = full(state.T), full(state.a_ice)
        fin = all(np.all(np.isfinite(np.asarray(getattr(state, f), np.float64))) for f in
                  ("T", "S", "uv", "eta_n", "a_ice", "m_ice", "tke"))
        T_top = T[np.arange(n), uln - 1]
        print(f"[cpl] interval {i + 1}: {dtw:.1f}s ({1000 * dtw / a.nsteps:.0f} ms/step incl. compile on the first) "
              f"finite={fin} SST[{T[~cav, 0].min():.2f},{T[~cav, 0].max():.2f}] "
              f"T_shelfbase[{T_top[cav].min():.3f},{T_top[cav].max():.3f}] dT_shelfbase max "
              f"{np.abs(T_top[cav] - T_top0[cav]).max():.2e} max|uv| {np.abs(full(state.uv)).max():.3f} "
              f"a_ice under shelves max {aice[cav].max():.1e}, {st.calendar()}", flush=True)
        ok &= fin and aice[cav].max() == 0.0
    print("CPL_CORE3_OK" if ok else "CPL_CORE3_FAIL")


if __name__ == "__main__":
    main()
