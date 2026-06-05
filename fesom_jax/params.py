"""Differentiable physics parameters — the ML-hook seam (Phase 3, Task 3.2).

The Phase-2 kernels read their tunables from module-level constants in
:mod:`fesom_jax.config` (``K_VER``, ``A_VER``, …). To take a gradient
``d(loss)/d(param)`` those tunables must instead be **traced leaves** flowing
through ``step``. :class:`Params` is that container: a small registered pytree of
the differentiable scalars, threaded ``step → pp.mixing_pp`` (and onward as the
physics grows).

This is also the first concrete **ML-hook seam**. Phase 7 swaps the PP mixing for
a trainable NN; that NN's weights live here too (alongside / replacing the scalar
backgrounds), so the same ``grad(loss)(params)`` call trains them. Keep it a
pytree (every field a leaf) so :func:`jax.grad` returns the same structure.

Phase 3 carries just the PP vertical-mixing backgrounds:

* ``k_ver`` — background tracer vertical diffusivity [m²/s]  (the plan's gradient
  target; enters ``Kv = mix·factor³ + k_ver`` additively → ``dKv/dk_ver = 1``,
  clean past the PP ``max(N²,0)`` kink);
* ``a_ver`` — background momentum vertical viscosity [m²/s]  (the sibling; enters
  ``Av`` and so the *within-step* CG path via ``impl_vert_visc``).

:meth:`Params.defaults` reproduces the config constants exactly (so passing
``Params.defaults()`` is numerically identical to the un-parameterised Phase-2
path — verified against the 274-test suite).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import tree_util

from .config import A_VER, K_VER


@dataclasses.dataclass(frozen=True)
class Params:
    """Differentiable physics parameters (a JAX pytree; both fields are leaves)."""

    k_ver: jax.Array  # background tracer vertical diffusivity [m²/s]
    a_ver: jax.Array  # background momentum vertical viscosity [m²/s]

    @staticmethod
    def defaults() -> "Params":
        """The config-constant baseline as float64 scalar arrays (the value used
        when ``step``/``integrate`` are called with ``params=None``)."""
        return Params(
            k_ver=jnp.asarray(K_VER, jnp.float64),
            a_ver=jnp.asarray(A_VER, jnp.float64),
        )


tree_util.register_dataclass(
    Params, data_fields=["k_ver", "a_ver"], meta_fields=[]
)
