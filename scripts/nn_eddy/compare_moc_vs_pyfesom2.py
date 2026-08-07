"""Task-4 gate: our overturning vs pyfesom2's, same input file, same basin."""
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, "/home/a/a270088/nn_eddy/port_jax/scripts/nn_eddy")
import mesh_diag as md

REF = Path("/scratch/a/a270088/tmp/claude-24253/-home-a-a270088-nn-eddy/"
           "646b694c-1b65-4c7d-912d-bdf8dfac6ac1/scratchpad/pyfesom2_moc_ref.npz")
WFILE = "/work/ab0995/a270088/fesom2_core2/w.fesom.1958.nc"
MESH = "/home/a/a270088/port_jax/data/mesh_core2"

ref = np.load(REF, allow_pickle=True)
ref_moc, ref_lats, ref_mask = ref["moc"], ref["lats"], ref["mask"]

m = md.load_mesh_arrays(MESH)
w = md.load_w(WFILE).T if md.load_w(WFILE).shape[0] != m["lon_deg"].size else md.load_w(WFILE)

our_mask = md.basin_mask(m["lon_deg"], m["lat_deg"], "Atlantic_MOC")
print(f"mask nodes: ours={our_mask.sum()}  pyfesom2={np.asarray(ref_mask).sum()}  "
      f"disagree={int((our_mask != np.asarray(ref_mask, bool)).sum())}")

lats, moc = md.moc_from_w(w, m["elem_nodes"], m["elem_area"], m["elem_nlevels"],
                          md.elem_centroid_lat(m["elem_nodes"], m["lat_deg"]),
                          mask=our_mask, nlats=91)
print(f"lats identical: {np.allclose(lats, ref_lats)}")
print(f"shapes ours={moc.shape} ref={ref_moc.shape}")

a, b = np.asarray(moc, float), np.asarray(ref_moc, float)
both = np.isfinite(a) & np.isfinite(b)
d = np.abs(a - b)[both]
print(f"\nfinite cells compared: {both.sum()} of {a.size}")
print(f"max|Δ| = {d.max():.3e} Sv    mean|Δ| = {d.mean():.3e} Sv")
print(f"our  range: {np.nanmin(a):+.2f} .. {np.nanmax(a):+.2f} Sv")
print(f"ref  range: {np.nanmin(b):+.2f} .. {np.nanmax(b):+.2f} Sv")
print(f"AMOC strength (20-60N, 500-3000 m): ours={md.amoc_strength(lats, a, m['zbar']):.3f} Sv  "
      f"ref={md.amoc_strength(ref_lats, b, m['zbar']):.3f} Sv")
tol = 1e-6
print(f"\nPARITY {'PASS' if d.max() < tol else 'FAIL'} (tolerance {tol:g} Sv)")
