"""Standalone JAX environment smoke test (works on CPU or GPU).

Run on the login node (CPU) or via SLURM on a GPU node (scripts/debug/gpu_check.sbatch):
prints versions + devices, confirms float64 is active, runs a float64 jitted
matmul on the default device, and reports per-device memory (best effort).
"""

import fesom_jax  # enables x64 (via fesom_jax.config) and proves the editable install
from fesom_jax import config  # noqa: F401

import jax
import jax.numpy as jnp

print("fesom_jax:", fesom_jax.__version__)
print("jax      :", jax.__version__)
try:
    import jaxlib

    print("jaxlib   :", jaxlib.__version__)
except Exception as exc:  # pragma: no cover
    print("jaxlib   : (n/a)", exc)

print("backend  :", jax.default_backend())
print("devices  :", jax.devices())

# float64 must be active (config.py flips it on import).
x = jnp.ones(1)
print("x64 dtype:", x.dtype)
assert x.dtype == jnp.float64, "x64 NOT enabled"


@jax.jit
def matsum(a):
    return (a @ a).sum()


a = jnp.ones((2048, 2048))
y = matsum(a)
y.block_until_ready()
print("matmul   : sum=", float(y), " dtype=", (a @ a).dtype, " on=", list(y.devices()))

for d in jax.devices():
    try:
        stats = d.memory_stats() or {}
        lim = stats.get("bytes_limit")
        if lim:
            print(f"  {d}: memory_limit ~ {lim / 1e9:.1f} GB")
    except Exception:
        pass

print("SMOKE OK")
