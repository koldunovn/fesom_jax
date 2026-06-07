"""Distributed reductions for the FESOM2 → JAX port (Phase 8, Task S.5).

The JAX rendering of the C's ``integrate_nod_2D`` + ``MPI_Allreduce(SUM)``: sum over
a device's **owned, unpadded** entities (mask halo + pad → 0 to avoid double-count),
then ``jax.lax.psum`` over the device axis (mirrors
``Kokkos parallel_reduce(0, myDim)`` + ``MPI_Allreduce``).

All the per-step reductions are **node**-based (``_area_mean`` for the
virtual-salt / relax-salt / water-flux balances; the CG dot-products over nodes;
``ocean_area``). Nodes are uniquely partitioned (``owned_mask`` IS the unique-owner
mask), so an owned-node sum + ``psum`` is exact — the element/edge redundant-ownership
caveat (S.1) does NOT apply here (there are no per-step element/edge reductions).

``axis_name=None`` ⇒ the single-device path: a plain masked sum, and with an
all-True ``owned_mask`` it is the *exact* ``jnp.sum`` the dense kernels use (so the
``npes==1`` reductions stay byte-identical to ``v1.0``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def global_sum(vals, owned_mask, axis_name: str | None = None):
    """``Σ vals`` over owned lanes, then ``psum`` over the device axis.

    ``vals`` is ``[Lmax, *rest]`` (the entity axis leads); ``owned_mask`` is
    ``[Lmax]`` (True on ``[0:myDim)``). The entity axis is summed (mask → 0 on
    halo/pad); trailing axes are preserved (so a ``[Lmax]`` field → scalar, a
    ``[Lmax, k]`` field → ``[k]``). ``axis_name=None`` skips the ``psum``
    (single-device).
    """
    vals = jnp.asarray(vals)
    m = jnp.asarray(owned_mask)
    mb = m.reshape(m.shape + (1,) * (vals.ndim - m.ndim))      # broadcast over trailing
    local = jnp.sum(jnp.where(mb, vals, jnp.zeros((), vals.dtype)), axis=0)
    return local if axis_name is None else jax.lax.psum(local, axis_name)


def global_dot(a, b, owned_mask, axis_name: str | None = None):
    """``Σ (a·b)`` over owned lanes + ``psum`` — the distributed CG dot-product
    (``pp·App``, ``rr·zz``, ``rr·rr``). ``a``/``b`` are ``[Lmax, *rest]``."""
    return global_sum(jnp.asarray(a) * jnp.asarray(b), owned_mask, axis_name)
