"""Bitwise comparison of two FESOM2 run directories: every variable of every restart
(oce/ice/bio) and every output file both runs wrote. Usage: compare_runs.py RUN_A RUN_B"""
import sys
from pathlib import Path

import netCDF4
import numpy as np


def nc_files(d):
    out = {}
    for p in sorted(Path(d).rglob("*.nc")):
        if "exe_seg" in p.parts or p.name == "fesom.mesh.diag.nc":
            continue
        out[str(p.relative_to(d))] = p
    return out


def main(a, b):
    fa, fb = nc_files(a), nc_files(b)
    common = sorted(set(fa) & set(fb))
    only = sorted(set(fa) ^ set(fb))
    nvar = ndiff = 0
    for k in common:
        da, db = netCDF4.Dataset(fa[k]), netCDF4.Dataset(fb[k])
        for v in da.variables:
            if v not in db.variables:
                print("missing in B:", k, v)
                ndiff += 1
                continue
            xa = np.asarray(da[v][:])
            xb = np.asarray(db[v][:])
            nvar += 1
            if xa.shape != xb.shape or xa.tobytes() != xb.tobytes():
                ndiff += 1
                d = np.nanmax(np.abs(xa.astype(float) - xb.astype(float))) if xa.shape == xb.shape else "shape"
                print("DIFF", k, v, d)
    print(f"{len(common)} files, {nvar} variables compared, {ndiff} differ; only in one run: {only}")
    return 1 if ndiff else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
