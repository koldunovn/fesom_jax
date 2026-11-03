#!/usr/bin/env python
"""Check a converted CORE3 restart against the Fortran restart it came from (wet lanes must be EXACT).

    python scripts/core3/check_restart.py JAX_RESTART MESH_JAX FORTRAN_OCE_DIR FORTRAN_ICE_DIR [ORACLE_DUMPS]

Reads every converted field back with ``zarr_output.reconstruct_global`` and compares it on the ``ulevels``-aware wet
lanes with the Fortran variable (float64 on file, so the comparison is bitwise); also reports what is in the dry
and cavity lanes. Prints ``RESTART_CHECK_OK`` when every field is bitwise on its wet lanes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fesom_jax import fortran_restart as FR, zarr_output  # noqa: E402


def main():
    rst, mdir, oce_dir, ice_dir = sys.argv[1:5]
    L = lambda n: np.load(Path(mdir) / f"{n}.npy")
    uln, nln, ule, nle = L("ulevels_nod2D"), L("nlevels_nod2D"), L("ulevels"), L("nlevels")
    N, E = uln.size, ule.size
    k = np.arange(48)[None, :]
    lay_n = (k >= uln[:, None] - 1) & (k < nln[:, None] - 1)
    ifc_n = (k >= uln[:, None] - 1) & (k <= nln[:, None] - 1)
    lay_e = (k >= ule[:, None] - 1) & (k < nle[:, None] - 1)
    cav_n = k < uln[:, None] - 1
    oce, ice = FR.FortranRestart(oce_dir), FR.FortranRestart(ice_dir)
    checks = [("T", oce, "temp", N, lay_n), ("S", oce, "salt", N, lay_n), ("T_old", oce, "temp_M1", N, lay_n),
              ("S_old", oce, "salt_M1", N, lay_n), ("hnode", oce, "hnode", N, lay_n), ("w", oce, "w", N, ifc_n),
              ("w_e", oce, "w_expl", N, ifc_n), ("tke", oce, "tke", N, ifc_n), ("eta_n", oce, "ssh", N, None),
              ("hbar", oce, "hbar", N, None), ("d_eta", oce, "d_eta", N, None), ("ssh_rhs_old", oce, "ssh_rhs_old", N, None),
              ("a_ice", ice, "area", N, None), ("m_ice", ice, "hice", N, None), ("m_snow", ice, "hsnow", N, None),
              ("u_ice", ice, "uice", N, None), ("v_ice", ice, "vice", N, None), ("t_skin", ice, "t_skin", N, None),
              ("sigma11", ice, "sigma11", E, None), ("sigma12", ice, "sigma12", E, None), ("sigma22", ice, "sigma22", E, None)]
    ok = True
    for name, grp, var, n, mask in checks:
        j = np.asarray(zarr_output.reconstruct_global(rst, name), np.float64)
        f = grp.read(var, n)
        if mask is None:
            d = np.abs(j - f).max()
            extra = ""
        else:
            nk = f.shape[1]
            m = mask[:, :nk]
            d = np.abs(j[:, :nk] - f)[m].max()
            extra = f" | dry lanes: max|jax| {np.abs(np.where(~mask, j, 0)).max():.3g}"
            if n == N:
                extra += f", cavity lanes max|jax| {np.abs(np.where(cav_n, j, 0)).max():.3g} max|fortran| " \
                         f"{np.abs(np.where(cav_n[:, :nk], f, 0)).max():.3g}"
        print(f"{name:12s} <- {var:12s} wet max|diff| {d:.3e}{extra}")
        ok &= d == 0.0
    for comp, (vu, i) in (("uv", ("u", 0)), ("uv", ("v", 1)), ("uv_rhsAB", ("urhs_AB", 0)), ("uv_rhsAB", ("vrhs_AB", 1))):
        j = np.asarray(zarr_output.reconstruct_global(rst, comp), np.float64)[..., i]
        f = oce.read(vu, E); nk = f.shape[1]
        d = np.abs(j[:, :nk] - f)[lay_e[:, :nk]].max()
        print(f"{comp + '[' + str(i) + ']':12s} <- {vu:12s} wet max|diff| {d:.3e}")
        ok &= d == 0.0
    if len(sys.argv) > 5:          # the oracle's dumps: the density the first step reads (pre_fluxes, step 1)
        from fesom_jax import io_dump
        o = io_dump.load_full_field(sys.argv[5], 1, 20, "density", N)
        j = np.asarray(zarr_output.reconstruct_global(rst, "density"), np.float64)[:, :o.shape[1]]
        m = (lay_n | cav_n)[:, :o.shape[1]]
        print(f"density (cavity run) vs oracle pre_fluxes step 1: max|diff| {np.abs(j - o)[m].max():.3e} kg/m3 "
              f"(of {np.abs(o[m]).max():.3g})")
    print("RESTART_CHECK_OK" if ok else "RESTART_CHECK_DIFF")


if __name__ == "__main__":
    main()
