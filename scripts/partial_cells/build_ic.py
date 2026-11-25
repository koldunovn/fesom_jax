"""Build a PHC IC cache for a partial-cell mesh (fesom_jax.partial_cells): each node's bottom layer interpolated at its
own partial-cell mid-depth (FESOM2 do_ic3d reads Z_3d_n), the potential-temperature conversion at the nominal |Z|
(FESOM2 insitu2pot). With --dist N the land fill follows the N-rank partition's node order, as FESOM2's MPI
extrap_nod3D does (match the Fortran run's rank count). --full-cells builds the full-cell IC with the same partition
(the matching reference cache).

    <env-py> scripts/partial_cells/build_ic.py --mesh-dir data/mesh_core2 --out-dir <dir> [--thresh 0] [--dist 512]
"""
import argparse
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mesh-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--thresh", type=float, default=0.0)
ap.add_argument("--dist", type=int, default=0, help="rank count of the partition order of the land fill (0 = serial)")
ap.add_argument("--dist-root", default="/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
ap.add_argument("--full-cells", action="store_true")
args = ap.parse_args()

from fesom_jax import partial_cells, partit, phc_ic
from fesom_jax.mesh import load_mesh

mesh = load_mesh(args.mesh_dir)
if not args.full_cells:
    mesh = partial_cells.apply(mesh, thresh=args.thresh)
rank_nodes = None
if args.dist:
    part = partit.read_partition(Path(args.dist_root), args.dist)
    rank_nodes = [np.asarray(part.myList_nod2D[r][: int(part.myDim_nod2D[r])], dtype=np.int64)
                  for r in range(args.dist)]
    assert sum(len(g) for g in rank_nodes) == mesh.nod2D
t0 = time.time()
res = phc_ic.build_and_cache_ic(mesh, out_dir=args.out_dir, rank_nodes=rank_nodes)
print(f"IC ({'full cells' if args.full_cells else f'partial cells, thresh {args.thresh}'}, "
      f"dist {args.dist or 'serial'}) -> {args.out_dir}: {time.time() - t0:.0f}s; "
      f"T [{res.T.min():.2f}, {res.T.max():.2f}]", flush=True)
