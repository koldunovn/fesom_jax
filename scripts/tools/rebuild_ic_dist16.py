"""Build the dist_16 partition-faithful CORE2 PHC IC cache (data/ic_core2_dist16).

The C extrap_nod3D GS land fill is order-dependent ⇒ the C IC depends on the MPI
partition. The z2_cdump oracle ran on 16 ranks, so to match it bit-for-bit the JAX IC
must replicate the dist_16 fill order (per-rank local sweeps + halo-frozen exchanges).
Rank node lists (C local order = myList_nod2D) come from the per-rank dump gid columns.

TWO caches coexist (the C itself gives a different IC per partition):
  data/ic_core2         — SERIAL (1-rank) order: the legacy CORE2 oracles
                          (core2_cdump/kpp/gm/ice-coupling dumps were 1-rank runs);
                          built by scripts/cache_phc_ic.py.
  data/ic_core2_dist16  — dist_16 order: the z2_cdump-gated zstar tests; THIS script.

Validates the rebuilt surface T/S bit-for-bit against the C 16-rank postload dump
(mevp/cdump_16r, same partition as z2_cdump — verified) and reports what differs vs
the serial-order cache.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fesom_jax.mesh import load_mesh
from fesom_jax import phc_ic

CORE2_MESH = Path("data/mesh_core2")
IC_DIR = Path("data/ic_core2_dist16")     # NOT data/ic_core2 — that stays the SERIAL build
Z2_DUMP = Path("/work/ab0995/a270088/port/zstar/z2_cdump/dump")
MEVP_DUMP = Path("/work/ab0995/a270088/port/mevp/cdump_16r/dump")
NRANK = 16

mesh = load_mesh(CORE2_MESH)
N = int(np.asarray(mesh.nlevels_nod2D).shape[0])

# Rank-owned gids in C local order, from the z2_cdump per-rank dumps (gid col 0).
rank_nodes = []
for r in range(NRANK):
    g = np.loadtxt(Z2_DUMP / f"ale_dump_s1_hnode_rank{r}.txt", comments="#",
                   usecols=0).astype(np.int64) - 1
    rank_nodes.append(g)
total = sum(len(g) for g in rank_nodes)
assert total == N and np.unique(np.concatenate(rank_nodes)).size == N, "bad partition"
print(f"partition: {NRANK} ranks, {total} nodes", flush=True)

t0 = time.time()
res = phc_ic.build_and_cache_ic(mesh, out_dir=IC_DIR, rank_nodes=rank_nodes)
print(f"build_and_cache_ic(dist_16) -> {IC_DIR}: {time.time()-t0:.0f}s", flush=True)

# --- Gate 1: surface vs the C 16r postload dump (bit-identity expected) ---
cT = np.full(N, np.nan)
cS = np.full(N, np.nan)
for r in range(NRANK):
    d = np.loadtxt(MEVP_DUMP / f"phc_dump_postload_rank{r}.txt", comments="#")
    g = d[:, 0].astype(int) - 1
    cT[g] = d[:, 1]
    cS[g] = d[:, 2]
dT = np.abs(res.T[:, 0] - cT)
dS = np.abs(res.S[:, 0] - cS)
print(f"surface vs C 16r postload: T ndiff={np.sum(dT > 0)} max={dT.max():.3e} | "
      f"S ndiff={np.sum(dS > 0)} max={dS.max():.3e}")

# --- Report: how the dist_16 IC differs from the serial-order cache (if present) ---
serial_dir = Path("data/ic_core2")
if (serial_dir / "T_ic.npy").exists():
    for name, new in (("T", res.T), ("S", res.S)):
        old = np.load(serial_dir / f"{name}_ic.npy")
        diff = new != old
        per_k = diff.sum(axis=0)
        ks = np.nonzero(per_k)[0]
        print(f"{name}: {diff.sum()} (node,k) entries differ from the serial cache; "
              f"levels {ks.min() if ks.size else '-'}..{ks.max() if ks.size else '-'}; "
              f"surface differs at {per_k[0]} nodes")
