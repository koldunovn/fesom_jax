#!/usr/bin/env python
"""Sanity checks of a cavity mesh prepared by ``scripts/prepare_mesh.py --cavity``.

    python scripts/core3/check_cavity_mesh.py MESH_JAX [--mesh-diag fesom.mesh.diag.nc]

Checks the level invariants FESOM2's ``find_levels_cavity`` asserts (ulevels(e) >= max ulevels_nod2D of its
nodes, >= 2 wet levels everywhere ... ), that ``load_mesh`` accepts the bundle, and, if a FESOM
``fesom.mesh.diag.nc`` of the same mesh is given, compares ``ulevels``/``nlevels``/``elem_area`` and its
``nod_area`` (FESOM writes ``areasvol`` or ``area`` depending on the writer) against the bundle.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_dir")
    ap.add_argument("--mesh-diag")
    a = ap.parse_args()
    d = Path(a.mesh_dir)
    L = lambda n: np.load(d / f"{n}.npy")
    ul, nl_e, uln, nln = L("ulevels"), L("nlevels"), L("ulevels_nod2D"), L("nlevels_nod2D")
    en = L("elem_nodes"); area, asv = L("area"), L("areasvol"); ulmax = L("ulevels_nod2D_max")
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {name}")

    print(f"[check_cavity_mesh] {d}: cavity nodes {(uln > 1).sum()} of {uln.size}, "
          f"cavity elems {(ul > 1).sum()} of {ul.size}, max ulevels {ul.max()}")
    check("ulevels(e) >= max(ulevels_nod2D(nodes))", (ul >= uln[en].max(axis=1)).all())
    check("ulevels_nod2D == MIN over cells of ulevels",
          (uln == np.minimum.reduceat(ul[L('nod_in_elem2D')], L('nod_in_elem2D_offsets')[:-1])).all())
    check("ulevels_nod2D_max >= ulevels_nod2D", (ulmax >= uln).all())
    check("nlevels - ulevels >= 2 (elem)", ((nl_e - ul) >= 2).all())
    check("nlevels_nod2D - ulevels_nod2D >= 2 (node)", ((nln - uln) >= 2).all())
    k = np.arange(area.shape[1])[None, :]
    wet = (k >= uln[:, None] - 1) & (k < nln[:, None] - 1)
    check("areasvol > 0 on every wet node level", (asv[wet] > 0).all())
    check("areasvol == 0 above ulevels_nod2D / below nlevels_nod2D-1", (asv[~wet] == 0).all())
    far = ulmax == 1                                  # no cavity cell touches the node
    check("area == areasvol where no cavity cell touches the node", np.array_equal(area[far][wet[far]], asv[far][wet[far]]))
    check("areasvol == area of the level below where a cavity cell touches the upper face",
          bool(np.all(asv[~far][wet[~far]] > 0)))
    cd = L("cavity_depth")
    check("cavity_depth <= 0", (cd <= 0).all())
    meta = dict(l.split() for l in open(d / "meta.txt"))
    print(f"  ocean_area {float(meta['ocean_area']):.6e}  ocean_areawithcav {float(meta['ocean_areawithcav']):.6e}")

    from fesom_jax.mesh import load_mesh
    m = load_mesh(str(d))
    check("load_mesh accepts the bundle", m.nod2D == uln.size)

    if a.mesh_diag:
        import netCDF4 as nc
        g = nc.Dataset(a.mesh_diag)
        check("mesh.diag ulevels == bundle", np.array_equal(g["ulevels"][:], ul))
        check("mesh.diag ulevels_nod2D == bundle", np.array_equal(g["ulevels_nod2D"][:], uln))
        check("mesh.diag nlevels == bundle", np.array_equal(g["nlevels"][:], nl_e))
        check("mesh.diag nlevels_nod2D == bundle", np.array_equal(g["nlevels_nod2D"][:], nln))
        ea = np.asarray(g["elem_area"][:], np.float64)
        print(f"  elem_area rel max {np.max(np.abs(ea - L('elem_area')) / L('elem_area')):.3e}")
        na = np.asarray(g["nod_area"][:], np.float64).T          # (nod2D, nz)
        for nm, arr in (("areasvol", asv), ("area", area)):
            sc = np.where(arr > 0, arr, 1.0)
            rel = np.abs(na - arr) / sc
            print(f"  nod_area vs {nm:8s}: rel max {rel.max():.3e}, n(rel>1e-6) {(rel > 1e-6).sum()}")
    print("CAVITY_MESH_OK" if ok else "CAVITY_MESH_BAD")


if __name__ == "__main__":
    main()
