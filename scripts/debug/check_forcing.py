#!/usr/bin/env python
"""A5: verify the RUNTIME forcing pipeline initializes + produces finite fluxes for a mesh.

    python scripts/debug/check_forcing.py MESH_DIR [--year 1958] [--ic-dir data/ic_core2]

The port interpolates forcing AT RUNTIME (like FESOM — `JRA55Reader` builds the bilinear weights once
at setup, then interpolates each step). This tool checks that *setup* runs for a given mesh — the only
forcing step that needs the mesh present — and that a 1-step forcing is finite (no pole / dateline /
coast out-of-bounds at scale). Nothing is pre-staged. Emits ``NG5_FORCING_OK`` when finite.

⚠️ MESH_DIR must be in the JAX-exported layout (`load_mesh`); the raw FESOM meshes on Levante
(`/pool/.../MESHES_FESOM2.1/{farc,dars,ng5}`) need the mesh export first (see docs/MESH_EXPORT_LAYOUT.md,
Task B0). Verified on CORE2 here; farc/dars/NG5 once exported."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fesom_jax import surface_forcing
from fesom_jax.mesh import load_mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mesh_dir")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--ic-dir", default="data/ic_core2")
    args = ap.parse_args()

    mesh = load_mesh(args.mesh_dir)
    print(f"[forcing] mesh={args.mesh_dir}  nod2D={mesh.nod2D}  elem2D={mesh.elem2D}  nl={mesh.nl}")

    # cold SST for the static a_ice mask (if a SIZE-MATCHING IC cache is present); else ice-free.
    # (A mesh-specific IC is needed to seed the a_ice mask; a CORE2 IC can't be used on farc/dars/NG5.)
    sst0 = None
    ic = Path(args.ic_dir) / "T_ic.npy"
    if ic.exists() and np.load(ic, mmap_mode="r").shape[0] == mesh.nod2D:
        from fesom_jax.phc_ic import core2_initial_state
        sst0 = np.asarray(core2_initial_state(mesh, args.ic_dir).T[:, 0])
    elif ic.exists():
        print(f"[forcing] IC {ic} ({np.load(ic, mmap_mode='r').shape[0]} nodes) != mesh "
              f"({mesh.nod2D}) — ice-free smoke (a_ice≡0)")

    # SETUP — builds the bilinear interpolation weights ONCE (the only mesh-dependent step)
    cf = surface_forcing.build_surface_forcing(mesh, args.year, sst_ic=sst0)
    print(f"[forcing] JRA55Reader + SSS/runoff/chl readers built for {mesh.nod2D} nodes")

    # 1-step RUNTIME interpolation — read disk + bilinear-interp to nodes + time-interp
    sf = cf.step_forcing(*surface_forcing.dates_for_steps(args.year, 1800.0, 1)[0])
    ok = True
    for name in sf._fields:
        a = np.asarray(getattr(sf, name))
        finite = bool(np.all(np.isfinite(a)))
        ok = ok and finite
        print(f"   {name:12s} finite={finite}  range=[{np.nanmin(a):+.3e}, {np.nanmax(a):+.3e}]")
    for name in cf.static._fields:
        a = np.asarray(getattr(cf.static, name))
        if not np.all(np.isfinite(a)):
            ok = False
            print(f"   static.{name}: NON-FINITE")

    print("NG5_FORCING_OK" if ok else "FORCING_CHECK_FAILED (non-finite forcing)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
