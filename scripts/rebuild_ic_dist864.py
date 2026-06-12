"""Build the dist_864 partition-faithful CORE2 PHC IC cache (data/ic_core2_dist864).

The `c_zstar_2yr` climate oracle was run on **864 ranks** (C zstar plan Z9, job 25495449),
NOT the 16 ranks of `z2_cdump`. The C `extrap_nod3D` GS land fill is order-dependent ⇒ the
dist_16 IC (`data/ic_core2_dist16`, which I matched bit-for-bit to z2_cdump) and the dist_864 IC
DIFFER at the partition-dependent Baltic/Kara GS-fill nodes — so a fair JAX-zstar↔c_zstar_2yr
climate comparison needs the **dist_864** IC. The 864-rank node lists (C local order) come from
the dist_864 partition files (NOT a per-rank dump — the climate run doesn't dump). No dist_864
postload dump exists to bit-gate against, so we trust the `_extrap_nod3D_mpi` algorithm (validated
bit-exact for dist_16) and report how the dist_864 IC differs from dist_16 (expect: the Baltic).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fesom_jax.mesh import load_mesh
from fesom_jax import partit, phc_ic

CORE2_MESH = Path("data/mesh_core2")
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")   # parent; read_partition appends dist_864
IC_DIR = Path("data/ic_core2_dist864")
NRANK = 864

mesh = load_mesh(CORE2_MESH)
N = int(np.asarray(mesh.nlevels_nod2D).shape[0])

# 864-rank owned-node lists in C local order, from the dist_864 partition files.
part = partit.read_partition(CORE2_DIST, NRANK)
rank_nodes = [np.asarray(part.myList_nod2D[r][: int(part.myDim_nod2D[r])], dtype=np.int64)
              for r in range(NRANK)]
total = sum(len(g) for g in rank_nodes)
assert total == N and np.unique(np.concatenate(rank_nodes)).size == N, \
    f"bad partition: total={total} N={N} unique={np.unique(np.concatenate(rank_nodes)).size}"
print(f"partition: {NRANK} ranks, {total} nodes (owned, C local order)", flush=True)

t0 = time.time()
res = phc_ic.build_and_cache_ic(mesh, out_dir=IC_DIR, rank_nodes=rank_nodes)
print(f"build_and_cache_ic(dist_864) -> {IC_DIR}: {time.time()-t0:.0f}s", flush=True)

# How does the dist_864 IC differ from the dist_16 cache? (expect: the Baltic/Kara GS-fill nodes.)
d16 = Path("data/ic_core2_dist16")
g = np.asarray(mesh.geo_coord_nod2D) * 180.0 / np.pi
baltic = (g[:, 0] > 8) & (g[:, 0] < 31) & (g[:, 1] > 53) & (g[:, 1] < 66)
if (d16 / "S_ic.npy").exists():
    for name, new in (("T", res.T), ("S", res.S)):
        old = np.load(d16 / f"{name}_ic.npy")
        ds = np.abs(new[:, 0] - old[:, 0])                 # surface
        nd = int(np.sum(ds > 0))
        nb = int(np.sum((ds > 0) & baltic))
        print(f"{name} surface vs dist_16: {nd} nodes differ (max={ds.max():.3e}); "
              f"{nb} of them in the Baltic box; Baltic max|Δ|={ds[baltic].max():.3e}", flush=True)
print("IC_DIST864_BUILD_OK", flush=True)
