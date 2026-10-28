#!/usr/bin/env python
"""Albedo: the Albani monthly dust on the packaged pi mesh, by nearest CORE2 node.

FESOM2's test/input/recom (DustClimMonthlyAlbani_pimesh.nc) is not on Albedo; the CORE2 file
(/albedo/pool/FESOM2/recom/CORE2/DustClimMonthlyAlbani.nc, DustClim(12, 126858) in the CORE2 node order)
is. For a pi smoke run the 5-degree dust field only needs to be plausible, so take, for every pi node,
the dust of the nearest CORE2 node (same units, no interpolation artefacts).

    python scripts/albedo/make_pi_dust.py CORE2_RAW_MESH_DIR OUT.nc
"""
import sys
import numpy as np
import netCDF4
from scipy.spatial import cKDTree
from fesom_jax.mesh import load_mesh

core2_raw, out = sys.argv[1], sys.argv[2]
src = "/albedo/pool/FESOM2/recom/CORE2/DustClimMonthlyAlbani.nc"
nod = np.loadtxt(f"{core2_raw}/nod2d.out", skiprows=1)        # id lon lat flag
lon, lat = np.deg2rad(nod[:, 1]), np.deg2rad(nod[:, 2])
def xyz(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
dust = netCDF4.Dataset(src).variables["DustClim"][:]           # (12, 126858)
assert dust.shape[1] == len(nod), (dust.shape, len(nod))
pi = load_mesh()
g = np.asarray(pi.geo_coord_nod2D)                              # (nod2D, 2) geographic radians
_, idx = cKDTree(xyz(lon, lat)).query(xyz(g[:, 0], g[:, 1]))
pid = np.asarray(dust[:, idx])
with netCDF4.Dataset(out, "w") as d:
    d.createDimension("months", 12); d.createDimension("nod2d", pi.nod2D)
    v = d.createVariable("DustClim", "f8", ("months", "nod2d")); v[:] = pid
    d.source = f"nearest CORE2 node of {src}; CORE2 coordinates from {core2_raw}/nod2d.out"
print("wrote", out, pid.shape, "mean", pid.mean(), "CORE2 mean", float(dust.mean()))
