"""Reference AMOC from pyfesom2 itself (runs in the pyfesom2 env, python 3.9).

Computes the Atlantic MOC from the Fortran CORE2 hindcast's annual-mean w and writes it
to an npz so the fesom-jax-side implementation can be compared numerically.
"""
import numpy as np
import pyfesom2 as pf
from netCDF4 import Dataset

MESH = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/"  # NB: this copy stores GEOGRAPHIC coords
WFILE = "/work/ab0995/a270088/fesom2_core2/w.fesom.1958.nc"
OUT = "/scratch/a/a270088/tmp/claude-24253/-home-a-a270088-nn-eddy/646b694c-1b65-4c7d-912d-bdf8dfac6ac1/scratchpad/pyfesom2_moc_ref.npz"

mesh = pf.load_mesh(MESH, abg=[0, 0, 0], usepickle=False, usejoblib=False)
ds = Dataset(WFILE)
w = np.ma.filled(np.ma.masked_greater(np.ma.asarray(ds.variables["w"][:]), 1e30), 0.0).astype(np.float64).mean(axis=0).T  # fills -> 0, mean in float64
print("w:", w.shape, "mesh nod2d:", mesh.n2d, "nlev:", mesh.nlev)

mask = pf.get_mask(mesh, "Atlantic_MOC")
print("Atlantic_MOC mask nodes:", int(np.asarray(mask).sum()))
lats, moc = pf.xmoc_data(mesh, w, nlats=91, mask=mask, return_masked=False)
moc = np.asarray(moc)
print("moc:", moc.shape, "max", np.nanmax(moc), "min", np.nanmin(moc))
np.savez(OUT, lats=np.asarray(lats), moc=moc, mask=np.asarray(mask),
         zlev=np.asarray(mesh.zlev), elem_area=None)
print("wrote", OUT)
