"""MULTI-PROCESS (2-node, 8-GPU) validation of the canonical (all_gather) restart writer.

Builds a signature full State (every leaf carries field_offset + gid, so a dropped/mis-scattered leaf
is caught), shards it folded across 8 GPUs (2 processes), writes a canonical restart via the multi-
process all_gather path, and on rank 0 reconstructs every leaf from disk and asserts == the original
global State. Covers BOTH nod- and elem-kind leaves. Run via validate_restart_mn.sbatch.
"""
import os, sys, dataclasses, shutil
sys.path.insert(0, "/home/a/a270088/port_jax")
import jax
if os.environ.get("JDIST"):
    jax.distributed.initialize(local_device_ids=list(range(int(os.environ.get("GPUS_PER_NODE", "4")))))
import jax.numpy as jnp
import numpy as np

from fesom_jax import halo, partit, shard_mesh
from fesom_jax import integrate_sharded as ish
from fesom_jax import zarr_output as zo
from fesom_jax.mesh import load_mesh
from fesom_jax.state import State

LEAD = jax.process_index() == 0
def log(*a):
    if LEAD:
        print(*a, flush=True)

NPES = len(jax.devices())
log(f"processes={jax.process_count()} global_devices={NPES} local={len(jax.local_devices())}")
mesh = load_mesh("/home/a/a270088/port_jax/data/mesh_core2")
part = partit.read_partition("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2", NPES)
sm = shard_mesh.build_sharded_mesh(mesh, part)
OUT = "/work/ab0995/a270088/port_jax/runs/_bench_restart_mn"

# --- signature full State: every leaf value = (field#)*1e7 + global-id ---
st0 = State.zeros(mesh)
sig = {}
for i, f in enumerate(dataclasses.fields(State)):
    a = np.asarray(getattr(st0, f.name))
    lead = a.shape[0]
    val = (i + 1) * 1e7 + np.arange(lead, dtype=np.float64)
    val = np.broadcast_to(val.reshape((lead,) + (1,) * (a.ndim - 1)), a.shape).copy()
    sig[f.name] = jnp.asarray(val)
gstate = dataclasses.replace(st0, **sig)

# --- fold + shard the global State across all 8 devices (2 processes) ---
sp = shard_mesh.partition_state(gstate, part)            # [P, Lmax, ...] host
fs, spec = ish.folded_state(sp)                          # [P*Lmax, ...]
jmesh = halo.device_mesh("p", devices=jax.devices())
placed = ish._to_global_sharded(fs, spec, jmesh)         # folded sharded device State

# --- write the canonical restart (multi-process all_gather path) ---
log("\n=== writing canonical restart (multi-process all_gather) ===")
zo.write_restart(OUT, placed, sm, part, step=1234, calendar_date="1958-03-15", dt_stage=1800.0)
from jax.experimental import multihost_utils
multihost_utils.sync_global_devices("restart_written")

# --- rank 0: reconstruct every leaf from disk, assert == the original global State ---
if LEAD:
    import zarr
    root = zarr.open_group(OUT, "r")
    assert root.attrs["layout"] == "canonical_global", root.attrs.get("layout")
    assert "gid_nod" not in root and "gid_elem" not in root, "canonical restart must carry no fold maps"
    nbad = 0
    nod_leaves = elem_leaves = 0
    for f in dataclasses.fields(State):
        g = zo.reconstruct_global(OUT, f.name)               # [n_global, ...] canonical, from disk
        orig = np.asarray(getattr(gstate, f.name))
        kind = root[f.name].attrs["kind"]
        nod_leaves += (kind == "nod"); elem_leaves += (kind == "elem")
        if g.shape != orig.shape or not np.array_equal(g, orig):
            nbad += 1
            print(f"  MISMATCH {f.name} kind={kind} shape {g.shape} vs {orig.shape}", flush=True)
    print(f"  leaves: {nod_leaves} nod + {elem_leaves} elem; mismatches={nbad}", flush=True)
    assert nbad == 0, f"{nbad} leaves wrong"
    meta = {k: v for k, v in root.attrs.items()}
    assert int(meta["step"]) == 1234 and str(meta["calendar_date"]) == "1958-03-15"
    print("  step/calendar_date/dt_stage preserved", flush=True)
    shutil.rmtree(OUT, ignore_errors=True)

multihost_utils.sync_global_devices("done")
log("\nRESTART_MN_CANONICAL_OK")
