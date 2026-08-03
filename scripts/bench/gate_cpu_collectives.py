"""Minimal cross-process CPU-collectives gate: psum under shard_map + a global sum.

Launch one process per device (``XLA_FLAGS=--xla_force_host_platform_device_count=1``);
``FESOM_CPU_COLL`` picks gloo (default) or mpi. The mpi path needs ``MPITRAMPOLINE_LIB``
pointing at an MPIwrapper ``libmpiwrapper.so`` built against the launcher's MPI.
"""
import os
import jax, jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_cpu_collectives_implementation", os.environ.get("FESOM_CPU_COLL", "gloo"))
jax.distributed.initialize()
p, n = jax.process_index(), jax.process_count()
m = jax.make_mesh((n,), ("d",))
x = jnp.arange(n * 4, dtype=jnp.float64).reshape(n, 4)
gx = jax.device_put(x, NamedSharding(m, P("d")))
f = jax.jit(jax.shard_map(lambda a: jax.lax.psum(a, "d"), mesh=m,
                          in_specs=P("d"), out_specs=P()))
r = f(gx)
# also a global all-to-all-ish check: sum of a sharded array
s = jax.jit(jnp.sum)(gx)
if p == 0:
    print("procs", n, "devices", jax.devices(), flush=True)
    print("psum:", r, "expected:", x.sum(0), flush=True)
    print("gsum:", s, "expected:", x.sum(), flush=True)
    ok = bool(jnp.allclose(r, x.sum(0))) and bool(jnp.allclose(s, x.sum()))
    print("GATE", "PASS" if ok else "FAIL", flush=True)
