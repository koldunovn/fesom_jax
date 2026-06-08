"""Build + cache the PHC3.0 winter IC (T_ic.npy / S_ic.npy) for a mesh, to a dir on /work.
Mesh-agnostic (interpolates the global PHC nc onto the mesh). Reusable for farc/dars/NG5.

    <env-py> scripts/cache_phc_ic.py --mesh-dir <dir> --out-dir <dir>
"""
import argparse
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

ap = argparse.ArgumentParser()
ap.add_argument("--mesh-dir", required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

from fesom_jax.mesh import load_mesh
from fesom_jax import phc_ic

t = time.perf_counter()
mesh = load_mesh(args.mesh_dir)
print(f"loaded mesh nod2D={mesh.nod2D} nl={mesh.nl} ({time.perf_counter()-t:.1f}s)", flush=True)
Path(args.out_dir).mkdir(parents=True, exist_ok=True)
t = time.perf_counter()
res = phc_ic.build_and_cache_ic(mesh, out_dir=args.out_dir)
print(f"PHC IC built+cached to {args.out_dir} ({time.perf_counter()-t:.1f}s); "
      f"T range [{float(res.T.min()):.2f},{float(res.T.max()):.2f}]", flush=True)
