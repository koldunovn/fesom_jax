"""Phase 8b B.3 STEP 1 — multi-node jax.distributed bring-up sanity. Confirms:
  * jax.distributed.initialize() works across N SLURM nodes,
  * jax.devices() returns the GLOBAL device set (all nodes' GPUs),
  * a global sharded array places addressable shards per process,
  * cross-process collectives (psum, all_gather) inside shard_map give the right answer.
Run via srun across nodes (1 task/node, 4 GPUs/task)."""
import jax
jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()                       # SLURM auto-detect

import numpy as np
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

pid, npc = jax.process_index(), jax.process_count()
gdev, ldev = jax.devices(), jax.local_devices()


def p0(*a):
    if pid == 0:
        print(*a, flush=True)


p0(f"[sanity] processes={npc}  global_devices={len(gdev)}  local_devices/proc={len(ldev)}")
p0("[sanity] global devices:", [f"{d.platform}:{d.id}" for d in gdev])

Pn = len(gdev)
mesh = Mesh(np.asarray(gdev), ("p",))
n = Pn * 8
g_host = np.arange(n, dtype=np.float64)
gs = jax.device_put(g_host, NamedSharding(mesh, P("p")))   # multi-proc: each proc places its shards
p0(f"[sanity] addressable_shards(proc0)={len(gs.addressable_shards)} (expect local_devices)")

# plain cross-process reduction
tot = float(gs.sum())
p0(f"[sanity] gs.sum()={tot}  expected={float(g_host.sum())}  OK={abs(tot - g_host.sum()) < 1e-6}")

# the model's collectives, inside shard_map across processes
def body(x):
    s = lax.psum(jnp.sum(x), "p")                          # cross-process scalar reduce
    ag = lax.all_gather(x, "p", axis=0, tiled=True)        # cross-process gather (the halo primitive)
    return jnp.stack([s, jnp.sum(ag)])

out = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P("p"), out_specs=P()))(gs)
p0(f"[sanity] shard_map psum={float(out[0])}  allgather_sum={float(out[1])}  "
   f"expected={float(g_host.sum())}")
ok = abs(float(out[0]) - g_host.sum()) < 1e-6 and abs(float(out[1]) - g_host.sum()) < 1e-6
p0(f"[sanity] MULTINODE SANITY {'OK' if ok else 'FAILED'}")
