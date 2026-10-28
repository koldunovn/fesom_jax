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

Single precision (``fesom_jax.precision``): every sum is accumulated in float64 from widened values and reduced in
float64 (FESOM2 #993/#994, ``MPI_WP_FULL``); :func:`global_sum` rounds the result once to WP (``integrate_nod_2D``),
the CG dot products keep float64 (``keep_f64``). All of it is the identity in double precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import precision as PR


def global_sum(vals, owned_mask, axis_name: str | None = None, *, keep_f64: bool = False):
    """``Σ vals`` over owned lanes, then ``psum`` over the device axis.

    ``vals`` is ``[Lmax, *rest]`` (the entity axis leads); ``owned_mask`` is
    ``[Lmax]`` (True on ``[0:myDim)``). The entity axis is summed (mask → 0 on
    halo/pad); trailing axes are preserved (so a ``[Lmax]`` field → scalar, a
    ``[Lmax, k]`` field → ``[k]``). ``axis_name=None`` skips the ``psum``
    (single-device).
    """
    vals = PR.full(jnp.asarray(vals))                         # single precision: a WP_full sum (#993)
    m = jnp.asarray(owned_mask)
    mb = m.reshape(m.shape + (1,) * (vals.ndim - m.ndim))      # broadcast over trailing
    local = jnp.sum(jnp.where(mb, vals, jnp.zeros((), vals.dtype)), axis=0)
    out = local if axis_name is None else jax.lax.psum(local, axis_name)
    # rounded once to WP (integrate_nod_2D), unless the caller keeps the WP_full value (CG scalars)
    return out if keep_f64 else PR.wp(out)


def global_dot(a, b, owned_mask, axis_name: str | None = None):
    """``Σ (a·b)`` over owned lanes + ``psum`` — the distributed CG dot-product
    (``pp·App``, ``rr·zz``, ``rr·rr``). ``a``/``b`` are ``[Lmax, *rest]``."""
    return global_sum(PR.full(jnp.asarray(a)) * PR.full(jnp.asarray(b)), owned_mask, axis_name,
                      keep_f64=True)


def global_dot_pair(a, b, c, d, owned_mask, axis_name: str | None = None):
    """``(Σ a·b, Σ c·d)`` with ONE collective instead of two — the CG's per-iteration
    ``(r·z, r·r)`` pair, which lives at the same point of the loop body and costs a
    separate ``psum`` latency each (~127 iterations × ~2 D exchanges already dominate
    the step's collective count at scale). The two OWNED-LANE partial dots are computed
    exactly as two :func:`global_dot` locals — same expressions, same reduction order —
    and only the cross-device reduction is fused into a single length-2 ``psum``
    (element-wise, same per-element device summation as two scalar ``psum``s)."""
    la = global_sum(PR.full(jnp.asarray(a)) * PR.full(jnp.asarray(b)), owned_mask, None, keep_f64=True)
    lb = global_sum(PR.full(jnp.asarray(c)) * PR.full(jnp.asarray(d)), owned_mask, None, keep_f64=True)
    if axis_name is None:
        return la, lb
    s = jax.lax.psum(jnp.stack([la, lb]), axis_name)
    return s[0], s[1]
