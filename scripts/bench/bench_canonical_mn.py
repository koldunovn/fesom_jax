"""MULTI-PROCESS (2-node, 8-GPU) validation + throughput of the canonical output writers.

The real test the single-node bench couldn't do: each process sees only its LOCAL shards, so this
exercises the distributed write path (rank-0 store create + barrier + per-process disjoint chunk writes)
and the true multi-node collective cost. Validates all_gather + redistribute are canonical + agree, and
times folded vs all_gather vs redistribute. Run via bench_canonical_mn.sbatch (2 nodes).
"""
import os, sys, time, shutil
sys.path.insert(0, "/home/a/a270088/port_jax")
import jax
if os.environ.get("JDIST"):
    jax.distributed.initialize(local_device_ids=list(range(int(os.environ.get("GPUS_PER_NODE", "4")))))
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from jax.experimental import multihost_utils

from fesom_jax import halo, partit, shard_mesh
from fesom_jax.mesh import load_mesh
from fesom_jax.ushow_output import _folded_gid_owned_nod, write_global_zarr, write_ushow_sharded

LEAD = jax.process_index() == 0
def log(*a):
    if LEAD:
        print(*a, flush=True)

NPES = len(jax.devices())                 # global device count = nnodes * 4
log(f"processes={jax.process_count()} global_devices={NPES} local={len(jax.local_devices())}")
mesh = load_mesh("/home/a/a270088/port_jax/data/mesh_core2")
part = partit.read_partition("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2", NPES)
sm   = shard_mesh.build_sharded_mesh(mesh, part)
gid, owned = _folded_gid_owned_nod(part, sm)
P, Lmax, nod2D = int(sm.P), int(sm.Lmax["nod"]), int(sm.nod2D)
jmesh = halo.device_mesh("p", devices=jax.devices())
sharding = NamedSharding(jmesh, PartitionSpec("p"))
def shard(arr):
    return jax.make_array_from_callback(arr.shape, sharding, lambda idx, a=arr: a[idx])
OUT = "/work/ab0995/a270088/port_jax/runs/_bench_canon_mn"
C = 20000

# --- multi-process correctness: all_gather + redistribute canonical (value==gid) + byte-identical ---
gsafe = np.clip(gid, 0, None)
id2d = shard(np.where(owned, gsafe, -1.0).astype(np.float64))
id3d = shard((np.where(owned, gsafe, -1.0)[:, None]
              + np.where(owned, 1.0, 0.0)[:, None] * np.arange(3)[None, :] / 1000.0).astype(np.float64))
idf = {"id2d": id2d, "id3d": id3d}
log("\n=== MULTI-PROCESS correctness (value==gid, methods agree) ===")
import zarr
ref = None
for m in ["all_gather", "redistribute"]:
    write_global_zarr(f"{OUT}/c_{m}.zarr", idf, sm, part, mesh, method=m, chunk_horiz=C, chunk_vert=2)
    multihost_utils.sync_global_devices(f"corr_{m}")
    if LEAD:
        r = zarr.open_group(f"{OUT}/c_{m}.zarr", "r")
        a2 = np.asarray(r["id2d"]); a3 = np.asarray(r["id3d"])
        ok2 = np.array_equal(a2, np.arange(nod2D, dtype=np.float32))
        ok3 = all(np.array_equal(a3[k], (np.arange(nod2D) + k/1000.0).astype(np.float32)) for k in range(3))
        log(f"  {m:12s} id2d==gid:{ok2} id3d==gid:{ok3} chunks={r['id2d'].chunks}")
        assert ok2 and ok3, m
        if ref is None:
            ref = {k: np.asarray(r[k]) for k in ("lon", "lat", "id2d", "id3d")}
        else:
            for k in ref:
                assert np.array_equal(ref[k], np.asarray(r[k])), (m, k)
log("  multi-process methods agree (byte-identical canonical store)")

# --- throughput on a realistic CORE2 monthly payload ---
rng = np.random.default_rng(0)
def rfold(nl=0):
    shp = (P*Lmax, nl) if nl else (P*Lmax,)
    return shard(rng.standard_normal(shp).astype(np.float64))
payload = {"ssh": rfold(), "a_ice": rfold(), "m_ice": rfold(),
           "temp": rfold(48), "salt": rfold(48), "u": rfold(48), "v": rfold(48)}
mb = sum((48 if "te" in k or "sa" in k or k in ("u", "v") else 1) * nod2D * 8 for k in payload) / 1e6
log(f"\n=== MULTI-NODE throughput: CORE2 monthly payload (~{mb:.0f} MB global f64, {jax.process_count()} nodes) ===")

def timeit(fn, reps=5, warm=2):
    for _ in range(warm):
        fn(); multihost_utils.sync_global_devices("warm")
    ts = []
    for _ in range(reps):
        multihost_utils.sync_global_devices("s"); t = time.perf_counter()
        fn(); multihost_utils.sync_global_devices("e"); ts.append(time.perf_counter() - t)
    return np.array(ts)

def run(m):
    if m == "folded":
        return write_ushow_sharded(f"{OUT}/p_{m}.zarr", payload, sm, part, mesh)
    return write_global_zarr(f"{OUT}/p_{m}.zarr", payload, sm, part, mesh, method=m,
                             chunk_horiz=C, chunk_vert=10)

for m in ["folded", "all_gather", "redistribute"]:
    ts = timeit(lambda m=m: run(m))
    log(f"  {m:12s} {ts.mean()*1e3:7.0f} ± {ts.std()*1e3:5.0f} ms")

multihost_utils.sync_global_devices("done")
if LEAD:
    shutil.rmtree(OUT, ignore_errors=True)
log("\nBENCH_CANON_MN_OK")
