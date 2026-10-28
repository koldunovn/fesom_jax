#!/usr/bin/env python
"""Host cost of the on-device forcing tables per chunk (what run.py's forcing worker does for every chunk): the JRA
bracket tables (``SurfaceForcing.stack_tables``: reads, the spatial interpolation, the coefficients) and their
partition onto the devices (``shard_mesh.partition_forcing_tables``), for N chunks of K steps of 1990 at dt 2700
in single precision (point-slope tables), CORE2 dist_4; a cProfile of the table builds by function.

    forcing_host_time.py [--chunks 20] [--K 96] [--profile]       (FESOM_FORCING_PROCS=n for read-ahead workers)
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--K", type=int, default=96)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    from fesom_jax import partit, shard_mesh, surface_forcing
    from fesom_jax.mesh import load_mesh
    from fesom_jax.run import _chunk_dates
    mesh = load_mesh(ROOT / "data" / "mesh_core2")
    part = partit.read_partition("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2", 4)
    t0 = time.perf_counter()
    f = surface_forcing.build_surface_forcing(mesh, 1990, dt=2700.0)
    f.point_slope = True
    print(f"setup {time.perf_counter() - t0:.1f} s", flush=True)
    nb = int(a.K * 2700.0) // 10800 + 2
    prof = cProfile.Profile() if a.profile else None
    tt, tp = [], []
    for c in range(a.chunks):
        dates = _chunk_dates(1990, 2700.0, c * a.K, a.K)
        t1 = time.perf_counter()
        if prof:
            prof.enable()
        tables, sched = f.stack_tables(dates, nb=nb)
        if prof:
            prof.disable()
        t2 = time.perf_counter()
        shard_mesh.partition_forcing_tables(tables, part)
        t3 = time.perf_counter()
        tt.append(t2 - t1); tp.append(t3 - t2)
        print(f"chunk {c}: tables {1e3 * (t2 - t1):.0f} ms, partition {1e3 * (t3 - t2):.0f} ms", flush=True)
    s = sorted(tt[2:]); p = sorted(tp[2:])
    print(f"median (after 2): tables {1e3 * s[len(s) // 2]:.0f} ms, partition {1e3 * p[len(p) // 2]:.0f} ms; "
          f"table bytes {sum(x.nbytes for x in tables if x is not None) / 1e6:.0f} MB", flush=True)
    if prof:
        out = io.StringIO()
        pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(18)
        print(out.getvalue())


if __name__ == "__main__":
    main()
