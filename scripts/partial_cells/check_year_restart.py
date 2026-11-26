"""Stability check of the partial-cells plan (Task 7): the end-of-year restart of the JAX zstar partial-cell run.

Checks (all on the global restart arrays):
  - the restart carries use_partial_cell = 1;
  - every wet layer thickness is positive (hnode, helem), and T, S, eta_n are finite on wet points;
  - the bottom layer of each node and element still equals FESOM2's bottom_node/elem_thickness exactly
    (the zstar update never touches it, oce_ale.F90).

    <env-py> scripts/partial_cells/check_year_restart.py <restart dir> [mesh dir] [thresh]
"""
import sys

import numpy as np
import zarr

from fesom_jax import partial_cells
from fesom_jax.mesh import load_mesh

RST = sys.argv[1]
mesh = load_mesh(sys.argv[2] if len(sys.argv) > 2 else "data/mesh_core2")
mesh = partial_cells.apply(mesh, thresh=float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
g = zarr.open_group(RST, mode="r")
zbar = np.asarray(mesh.zbar)
nln, nle = np.asarray(mesh.nlevels_nod2D), np.asarray(mesh.nlevels)
ok = int(g.attrs.get("use_partial_cell", 0)) == 1
print(f"restart step {g.attrs['step']}, date {g.attrs['calendar_date']}, use_partial_cell "
      f"{g.attrs.get('use_partial_cell')}")


def wet(n_levels, ncol):
    return np.arange(ncol)[None, :] < (n_levels - 1)[:, None]


hn, he = np.asarray(g["hnode"]), np.asarray(g["helem"])
wn, we = wet(nln, hn.shape[1]), wet(nle, he.shape[1])
print(f"min hnode (wet) {hn[wn].min():.4f} m, min helem (wet) {he[we].min():.4f} m")
ok &= hn[wn].min() > 0 and he[we].min() > 0
for k in ("T", "S"):
    a = np.asarray(g[k])
    fin = np.isfinite(a[wn]).all()
    print(f"{k}: finite on wet points {fin}, range {a[wn].min():.3f} .. {a[wn].max():.3f}")
    ok &= fin
eta = np.asarray(g["eta_n"])
print(f"eta_n: finite {np.isfinite(eta).all()}, range {eta.min():.3f} .. {eta.max():.3f} m")
ok &= np.isfinite(eta).all()
r = np.arange(mesh.nod2D)
dn = hn[r, nln - 2] - (zbar[nln - 2] - np.asarray(mesh.zbar_n_bot))
e = np.arange(mesh.elem2D)
de = he[e, nle - 2] - (zbar[nle - 2] - np.asarray(mesh.zbar_e_bot))
print(f"bottom layer vs bottom thickness: max |d| nodes {np.abs(dn).max():.3e} m, elements {np.abs(de).max():.3e} m")
ok &= np.abs(dn).max() == 0 and np.abs(de).max() == 0
print("YEAR_RESTART_OK" if ok else "YEAR_RESTART_FAIL")
