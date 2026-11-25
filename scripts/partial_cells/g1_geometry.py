"""G1 (partial-cells plan, Task 1b): fesom_jax.partial_cells.apply on CORE2 against FESOM2's own partial-cell geometry,
the zbar_e_bottom / zbar_n_bottom of a Fortran use_partial_cell run's fesom.mesh.diag.nc (written as -zbar_*_bot).

Pass: bit-identical everywhere except (a) the elements/nodes whose level counts differ between our mesh export and the
Fortran run's /pool levels (regenerated 2026-07-03), plus the nodes that take their minimum from such an element, and
(b) at most 1 ulp where the element depth sum is the selected value (summation order of the three node depths). The
script also recomputes the element bottoms with the Fortran element node order (face_nodes), which must then agree
bit for bit on every element outside (a).

    <env-py> scripts/partial_cells/g1_geometry.py <fortran run dir> [mesh dir]
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np

from fesom_jax import partial_cells as pc
from fesom_jax.mesh import load_mesh

run = Path(sys.argv[1])
mesh_dir = sys.argv[2] if len(sys.argv) > 2 else "data/mesh_core2"
mesh = load_mesh(mesh_dir)
out = pc.apply(mesh)
d = netCDF4.Dataset(run / "fesom.mesh.diag.nc")
f_e = -np.asarray(d["zbar_e_bottom"][:], np.float64)
f_n = -np.asarray(d["zbar_n_bottom"][:], np.float64)
f_nlev = np.asarray(d["nlevels"][:])
f_nlevn = np.asarray(d["nlevels_nod2D"][:])
fn = np.asarray(d["face_nodes"][:]).T                      # (elem, 3)
if fn.min() == 1:
    fn = fn - 1
j_e, j_n = np.asarray(out.zbar_e_bot), np.asarray(out.zbar_n_bot)
nlev, nlevn = np.asarray(mesh.nlevels), np.asarray(mesh.nlevels_nod2D)

lev_e = f_nlev != nlev
lev_n = f_nlevn != nlevn
en = np.asarray(mesh.elem_nodes)
touch = np.zeros(mesh.nod2D, bool)
touch[en[lev_e].ravel()] = True
skip_n = lev_n | touch
print(f"levels differ (our export vs the Fortran /pool levels): {lev_e.sum()} elements, {lev_n.sum()} nodes; "
      f"nodes excluded from the node check: {skip_n.sum()}")
same_set = np.array_equal(np.sort(fn, axis=1), np.sort(en, axis=1))
print(f"element node sets equal: {same_set}; same node order: {np.array_equal(fn, en)} "
      f"({(fn != en).any(axis=1).sum()} elements ordered differently)")

ok = True
de = (j_e != f_e) & ~lev_e
ulp = np.abs(j_e - f_e) <= np.spacing(np.abs(f_e))
print(f"element bottoms: {de.sum()} of {(~lev_e).sum()} differ outside (a); within 1 ulp: {(de & ulp).sum()}; "
      f"max |diff| {np.abs(j_e - f_e)[~lev_e].max():.3e} m")
ok &= bool(np.all(ulp[de]))
# where do they differ? (b) allows 1 ulp only where the element depth dd itself is the selected value
depth = pc.clamp_depth(mesh.depth, mesh.zbar)
dd = (depth[en[:, 0]] + depth[en[:, 1]] + depth[en[:, 2]]) / 3.0
dd_sel = j_e == dd
print(f"  of the differing elements, dd is the selected value at {(de & dd_sel).sum()}")
ok &= bool(np.all(dd_sel[de]))
# diagnosis: the Fortran node order, and the division by 3 compiled as a multiply by 1/3 (Intel -no-prec-div)
dd_f = (depth[fn[:, 0]] + depth[fn[:, 1]] + depth[fn[:, 2]]) / 3.0
j_e_f = pc.elem_bottom_depth(dd_f, nlev, np.asarray(mesh.zbar), np.asarray(mesh.Z))
print(f"  recomputed with the Fortran node order: {((j_e_f != f_e) & ~lev_e).sum()} differ outside (a)")
dd_r = (depth[fn[:, 0]] + depth[fn[:, 1]] + depth[fn[:, 2]]) * (1.0 / 3.0)
j_e_r = pc.elem_bottom_depth(dd_r, nlev, np.asarray(mesh.zbar), np.asarray(mesh.Z))
print(f"  recomputed as sum * (1/3): {((j_e_r != f_e) & ~lev_e).sum()} differ outside (a)")
dn = (j_n != f_n) & ~skip_n
ulp_n = np.abs(j_n - f_n) <= np.spacing(np.abs(f_n))
print(f"node bottoms: {dn.sum()} of {(~skip_n).sum()} differ outside (a); within 1 ulp: {(dn & ulp_n).sum()}; "
      f"max |diff| {np.abs(j_n - f_n)[~skip_n].max():.3e} m")
ok &= bool(np.all(ulp_n[dn]))
from fesom_jax.partial_cells import node_bottom_depth
j_n_r = node_bottom_depth(j_e_r, mesh.nod_in_elem2D, mesh.nod_in_elem2D_offsets)
print(f"  node bottoms from the sum * (1/3) element bottoms: {((j_n_r != f_n) & ~skip_n).sum()} differ outside (a)")
full = np.asarray(mesh.zbar)[nlev - 1]
print(f"partial cells change {np.mean(f_e != full) * 100:.1f} % of the Fortran element bottoms "
      f"(mean |shift| {np.abs(f_e - full).mean():.2f} m)")
if lev_e.any():
    print(f"(a) elements: Fortran {f_e[lev_e]} vs ours {j_e[lev_e]}")
print("G1_GEOMETRY_OK" if ok else "G1_GEOMETRY_FAIL")
