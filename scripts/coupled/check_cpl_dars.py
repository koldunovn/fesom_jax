#!/usr/bin/env python
"""Smoke check of the coupled DARS ocean (mode F, configs/dars_cpl.yaml): init_ocean from the PHC cold start with
FESOM's initial sea ice, a few coupling intervals with a fixed, plausible atmosphere (no atmosphere model), finite at
every interval; prints the compile and per-step times and the peak device memory.

    python scripts/coupled/check_cpl_dars.py [--intervals 3] [--nsteps 30] [--partition dist_4] [--config ...]

Prints ``CPL_DARS_OK`` when every interval is finite.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "dars_cpl.yaml"))
    ap.add_argument("--intervals", type=int, default=3)
    ap.add_argument("--nsteps", type=int, default=30)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--year0", type=int, default=1990)
    a = ap.parse_args()
    import jax
    from fesom_jax.coupled import init_ocean, ocean_interval

    t0 = time.perf_counter()
    state, st = init_ocean(a.config, mode="flux", year0=a.year0, nsteps=a.nsteps, partition=a.partition,
                           restart_in=None, start=(a.year0, 1, 1))
    print(f"[cpl] init {time.perf_counter() - t0:.1f}s, calendar {st.calendar()}, npes {st.npes}", flush=True)
    n = int(st.nod2D)
    if st.npes == 1:
        full = lambda x: np.asarray(x, np.float64)[:n]                                                    # noqa
    else:
        from fesom_jax.zarr_output import _folded_gid_owned
        gid, owned = _folded_gid_owned(st.part, st.sm, "nod")
        lane = np.zeros(n, np.int64); lane[gid[owned]] = np.flatnonzero(owned)
        full = lambda x: np.asarray(x, np.float64)[lane]                                                   # noqa
    z = np.zeros(n)
    forcing = dict(taux_oce=z + 0.05, tauy_oce=z, taux_ico=z + 0.05, tauy_ico=z, prec_oce=z + 2e-8,
                   snow_oce=z + 5e-9, evap_oce=z - 2e-8, subl_oce=z, heat_oce=z - 60.0, heat_ico=z - 20.0,
                   heat_swo=z + 120.0, tsk_ico=z + 265.0)
    ok = True
    for i in range(a.intervals):
        t1 = time.perf_counter()
        state = ocean_interval(state, st, forcing)
        state.T.block_until_ready()
        dtw = time.perf_counter() - t1
        T, aice = full(state.T), full(state.a_ice)
        fin = all(np.all(np.isfinite(np.asarray(getattr(state, f), np.float64))) for f in
                  ("T", "S", "uv", "eta_n", "a_ice", "m_ice", "tke"))
        print(f"[cpl] interval {i + 1}: {dtw:.1f}s ({1000 * dtw / a.nsteps:.0f} ms/step incl. compile on the first) "
              f"finite={fin} SST[{T[:, 0].min():.2f},{T[:, 0].max():.2f}] max|uv| {np.abs(full(state.uv)).max():.3f} "
              f"ice area {np.sum(aice * np.asarray(st.mesh.area)[0] if np.ndim(st.mesh.area) > 1 else aice) :.4g} "
              f"{st.calendar()}", flush=True)
        ok &= fin
    peak = {str(d): round((d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30, 2) for d in jax.local_devices()}
    print(f"[cpl] peak device memory GB {peak}", flush=True)
    print("CPL_DARS_OK" if ok else "CPL_DARS_FAIL")


if __name__ == "__main__":
    main()
