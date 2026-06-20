#!/usr/bin/env python
"""Convert a portable restart (gid-keyed, sharded — not viewable) into an ushow/uterm-readable
zarr (lon/lat + node-indexed fields). Task B2 — "look at the NG5 state".

    python scripts/restart_to_ushow.py <restart_store> <mesh_dir> <out_zarr> [--with-3d]

Surface fields (sst, sss, aice, mice, msnow, speed, ssh) are always written; ``--with-3d`` adds
the full-depth ``temp``/``salt`` (each ≈ 4 GB at NG5 — use a compute node, ``-p compute --mem=240G``).
Streams ONE global field at a time (``reconstruct_global``), so host peak ≈ one NG5 3-D leaf.

Then:  ushow <out_zarr>     (X11)    /    uterm <out_zarr>    (terminal)
"""
from __future__ import annotations

import argparse

import numpy as np

from fesom_jax.mesh import load_mesh
from fesom_jax.ushow_output import node_lonlat, write_ushow_zarr
from fesom_jax.zarr_output import reconstruct_global


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("store", help="restart store (a write_restart output)")
    ap.add_argument("mesh_dir", help="mesh dir (for node lon/lat)")
    ap.add_argument("out_zarr", help="output ushow zarr path")
    ap.add_argument("--with-3d", action="store_true", help="also write full-depth temp/salt (~4 GB each)")
    args = ap.parse_args()

    mesh = load_mesh(args.mesh_dir)
    lon, lat = node_lonlat(mesh)
    print(f"[ushow] mesh {len(lon)} nodes; reconstructing fields from {args.store}", flush=True)

    f2d, f3d, units = {}, {}, {}

    def g(name):
        print(f"[ushow]   reconstruct {name}", flush=True)
        return np.asarray(reconstruct_global(args.store, name))

    T = g("T")                                  # [N, nl]
    f2d["sst"] = T[:, 0]; units["sst"] = "degC"
    if args.with_3d:
        f3d["temp"] = T; units["temp"] = "degC"
    else:
        del T
    S = g("S")
    f2d["sss"] = S[:, 0]; units["sss"] = "psu"
    if args.with_3d:
        f3d["salt"] = S; units["salt"] = "psu"
    else:
        del S
    f2d["aice"] = g("a_ice"); units["aice"] = "frac"
    f2d["mice"] = g("m_ice"); units["mice"] = "m"
    f2d["msnow"] = g("m_snow"); units["msnow"] = "m"
    f2d["ssh"] = g("eta_n"); units["ssh"] = "m"
    uvn = g("uvnode")                           # [N, nl, 2]
    f2d["speed"] = np.hypot(uvn[:, 0, 0], uvn[:, 0, 1]); units["speed"] = "m/s"
    del uvn

    meta = {k: int(v) for k, v in (("nod2D", mesh.nod2D),) }
    write_ushow_zarr(args.out_zarr, lon, lat, fields2d=f2d, fields3d=f3d, units=units, attrs=meta)
    print(f"[ushow] wrote {args.out_zarr}  2D={list(f2d)}  3D={list(f3d)}", flush=True)
    print("USHOW_WRITE_OK", flush=True)


if __name__ == "__main__":
    main()
