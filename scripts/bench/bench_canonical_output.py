"""GPU correctness + throughput comparison of the output writers (single-node, 4 GPUs, 1 process).

Validates the GPU-only `redistribute` (ragged_all_to_all) path against host_gather/all_gather
(byte-identical canonical store + value==gid), then times each writer on a realistic CORE2 monthly-
output payload. Run via bench_canonical_output.sbatch (1 node, 4 GPUs).
"""
import os, sys, time, shutil
import numpy as np

sys.path.insert(0, "/home/a/a270088/port_jax")
import jax
from jax.sharding import NamedSharding, PartitionSpec

from fesom_jax import halo, partit, shard_mesh
from fesom_jax.mesh import load_mesh
from fesom_jax.ushow_output import _folded_gid_owned_nod, write_global_zarr, write_ushow_sharded

MESH = "/home/a/a270088/port_jax/data/mesh_core2"
POOL = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2"
OUT  = "/work/ab0995/a270088/port_jax/runs/_bench_canon"
NPES = 4

print(f"jax {jax.__version__}  devices={jax.devices()}  platform={jax.devices()[0].platform}", flush=True)
os.makedirs(OUT, exist_ok=True)

mesh = load_mesh(MESH)
part = partit.read_partition(POOL, NPES)
sm   = shard_mesh.build_sharded_mesh(mesh, part)
gid, owned = _folded_gid_owned_nod(part, sm)
P, Lmax, nod2D = int(sm.P), int(sm.Lmax["nod"]), int(sm.nod2D)
PL = P * Lmax
jmesh = halo.device_mesh("p", devices=jax.devices()[:P])
sharding = NamedSharding(jmesh, PartitionSpec("p"))
print(f"CORE2 nod2D={nod2D} P={P} Lmax={Lmax} PL={PL}", flush=True)


def shard(arr):
    return jax.make_array_from_callback(arr.shape, sharding, lambda idx, a=arr: a[idx])


# --- correctness field: owned lane value = its global node id (2-D + 3-D) ---
gsafe = np.clip(gid, 0, None)
id2d = shard(np.where(owned, gsafe, -1.0).astype(np.float64))
id3d = shard((np.where(owned, gsafe, -1.0)[:, None]
              + np.where(owned, 1.0, 0.0)[:, None] * np.arange(3)[None, :] / 1000.0).astype(np.float64))
idflds = {"id2d": id2d, "id3d": id3d}

CHUNK = 20000
print("\n=== correctness: every method canonical (value==gid) + byte-identical ===", flush=True)
import zarr
ref = None
methods = ["host_gather", "all_gather", "redistribute"]
for m in methods:
    d = f"{OUT}/correct_{m}.zarr"
    write_global_zarr(d, idflds, sm, part, mesh, method=m, chunk_horiz=CHUNK, chunk_vert=2)
    r = zarr.open_group(d, "r")
    a2 = np.asarray(r["id2d"]); a3 = np.asarray(r["id3d"])
    ok2 = np.array_equal(a2, np.arange(nod2D, dtype=np.float32))
    ok3 = all(np.array_equal(a3[k], (np.arange(nod2D) + k/1000.0).astype(np.float32)) for k in range(3))
    print(f"  {m:12s} id2d==gid:{ok2}  id3d==gid:{ok3}  chunks={r['id2d'].chunks}", flush=True)
    assert ok2 and ok3, m
    if ref is None:
        ref = {k: np.asarray(r[k]) for k in ("lon","lat","id2d","id3d")}
    else:
        for k in ref:
            assert np.array_equal(ref[k], np.asarray(r[k])), f"{m} {k} differs from host_gather"
print("  ALL METHODS AGREE (byte-identical canonical store)", flush=True)


# --- throughput: realistic CORE2 monthly payload (ssh/a_ice/m_ice 2-D + temp/salt/u/v 3-D x48) ---
rng = np.random.default_rng(0)
def rfold(nlev=0):
    shp = (PL, nlev) if nlev else (PL,)
    return shard(rng.standard_normal(shp).astype(np.float64))
payload = {"ssh": rfold(), "a_ice": rfold(), "m_ice": rfold(),
           "temp": rfold(48), "salt": rfold(48), "u": rfold(48), "v": rfold(48)}
mb = sum(np.asarray(v).nbytes for v in payload.values()) / 1e6
print(f"\n=== throughput: CORE2 monthly payload ({mb:.0f} MB folded f64, 7 fields) ===", flush=True)

def timeit(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return np.array(ts)

def run_method(m):
    d = f"{OUT}/perf_{m}.zarr"
    if m == "folded":
        return write_ushow_sharded(d, payload, sm, part, mesh)
    return write_global_zarr(d, payload, sm, part, mesh, method=m, chunk_horiz=CHUNK, chunk_vert=10)

for m in ["folded", "host_gather", "all_gather", "redistribute"]:
    ts = timeit(lambda m=m: run_method(m))
    out_mb = mb / 2  # written f32
    print(f"  {m:12s}  {ts.mean()*1e3:7.0f} ± {ts.std()*1e3:5.0f} ms   "
          f"({out_mb/ts.mean():6.0f} MB/s written)", flush=True)

# --- profile redistribute internals: collective+scatter (jax) vs the zarr write ---
from fesom_jax.canonical_redist import _cached_maps, redistribute_fields
mapz = _cached_maps(sm, part, CHUNK)
jm = halo.device_mesh("p", devices=jax.devices()[:P])
def _collective_only():
    c = redistribute_fields(payload, mapz, jm)
    for v in c.values():
        jax.block_until_ready(v)
ts = timeit(_collective_only, reps=3, warm=2)
print(f"\nredistribute COLLECTIVE+SCATTER only (no zarr): {ts.mean()*1e3:.0f} ± {ts.std()*1e3:.0f} ms",
      flush=True)

shutil.rmtree(OUT, ignore_errors=True)
print("\nBENCH_CANON_OK", flush=True)
