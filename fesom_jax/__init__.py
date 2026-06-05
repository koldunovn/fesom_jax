"""FESOM2 → JAX: a differentiable unstructured-grid ocean model.

Importing this package enables float64 ("x64") in JAX as a side effect (via
:mod:`fesom_jax.config`). That switch must be flipped before any JAX array is
created, so always ``import fesom_jax`` (or ``from fesom_jax import config``)
at the top of an entry point, before constructing arrays.

See ``docs/plans/20260605-fesom-jax-port.md`` for the roadmap and
``/home/a/a270088/port2/FRESH_START.md`` for the physics/algorithm reference.
"""

from . import config  # noqa: F401  — side effect: jax.config x64 = True

__version__ = "0.0.0"
__all__ = ["config"]
