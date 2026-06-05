"""Mesh primitive ops: gather, scatter, mask, vectorized TDMA (Task 1.2).

These are the AD-safe building blocks every FESOM kernel is expressed with — the
JAX rendering of the C port's index loops (Locked Decision #4). They are
deliberately **mesh-agnostic** (take raw index arrays, not a ``Mesh``) so they
unit-test in isolation and compose freely.

Fidelity / AD notes
-------------------
* **gather** (``field[idx]``) is exact; its reverse-mode transpose is a
  scatter-add, so gathers cost nothing for AD.
* **scatter-add** uses :func:`jax.ops.segment_sum`. On GPU the segment order is
  not C's edge order, so a scatter reassociates the FP sum → climate-close
  (~1e-12), never bit-identical. Its transpose is exactly a gather (verified in
  ``test_ops.py``). Negative segment ids (the ``-1`` boundary sentinel in
  ``edge_tri``) are **masked to contribute nothing** — both forward and in the
  gradient — rather than relying on ``segment_sum``'s drop behaviour.
* **TDMA** (Thomas) is two :func:`jax.lax.scan` sweeps so it differentiates
  cleanly and vectorizes over the entity axis; no Python loop over a traced
  length.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


# --------------------------------------------------------------------------
# Gather
# --------------------------------------------------------------------------
def gather(field, idx):
    """``field[idx]`` along the leading (entity) axis, preserving trailing axes.

    ``field`` is ``[N, *rest]``, ``idx`` any int shape ``S`` → ``[*S, *rest]``.
    """
    return jnp.asarray(field)[jnp.asarray(idx)]


def gather_nodes_to_elem(field, elem_nodes):
    """Node field ``[nod2D, *rest]`` → per-element corner values
    ``[elem2D, 3, *rest]`` (``field`` at each element's 3 nodes)."""
    return gather(field, elem_nodes)


def gather_to_edges(field, edges):
    """Node field ``[nod2D, *rest]`` → per-edge endpoint values
    ``[edge2D, 2, *rest]`` (``field`` at each edge's 2 nodes)."""
    return gather(field, edges)


# --------------------------------------------------------------------------
# Scatter-add
# --------------------------------------------------------------------------
def scatter_add(vals, seg, num_segments: int):
    """Differentiable masked segment-sum.

    Sums ``vals`` into ``num_segments`` output rows according to ``seg``. The
    leading dims of ``vals`` must match ``seg``'s shape (both are flattened);
    trailing dims of ``vals`` are summed elementwise. Entries with ``seg < 0``
    (the boundary sentinel) contribute nothing.

    ``vals``: ``[*S, *rest]``  ``seg``: ``[*S]``  →  out ``[num_segments, *rest]``.
    """
    vals = jnp.asarray(vals)
    seg = jnp.asarray(seg)
    lead = seg.ndim
    if vals.shape[:lead] != seg.shape:
        raise ValueError(
            f"scatter_add: vals leading dims {vals.shape[:lead]} must match "
            f"seg shape {seg.shape}"
        )
    seg_flat = seg.reshape(-1)
    vals_flat = vals.reshape((seg_flat.shape[0],) + vals.shape[lead:])

    valid = seg_flat >= 0
    safe_seg = jnp.where(valid, seg_flat, 0)
    vmask = valid.reshape((-1,) + (1,) * (vals_flat.ndim - 1))
    safe_vals = jnp.where(vmask, vals_flat, 0)
    return jax.ops.segment_sum(safe_vals, safe_seg, num_segments=num_segments)


def scatter_add_edges_to_nodes(vals, edges, num_nodes: int):
    """Edge→node scatter: add each edge's two endpoint contributions
    ``vals[e,0]``→node ``edges[e,0]`` and ``vals[e,1]``→node ``edges[e,1]``.

    ``vals``: ``[edge2D, 2, *rest]``  ``edges``: ``[edge2D, 2]``.
    """
    return scatter_add(vals, edges, num_nodes)


def scatter_add_edges_to_elems(vals, edge_tri, num_elems: int):
    """Edge→element scatter (e.g. biharmonic viscosity); ``edge_tri`` carries
    ``-1`` for a boundary edge's missing neighbour, masked out automatically.

    ``vals``: ``[edge2D, 2, *rest]``  ``edge_tri``: ``[edge2D, 2]``.
    """
    return scatter_add(vals, edge_tri, num_elems)


# --------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------
def mask_below_bottom(field, level_mask):
    """Zero ``field`` wherever ``level_mask`` is False (below bottom / above top).

    ``field``: ``[n_entity, nl, *rest]``  ``level_mask``: ``[n_entity, nl]`` bool.
    The mask broadcasts over any trailing axes (e.g. the velocity component).
    """
    field = jnp.asarray(field)
    m = jnp.asarray(level_mask)
    if field.ndim > m.ndim:
        m = m.reshape(m.shape + (1,) * (field.ndim - m.ndim))
    return jnp.where(m, field, jnp.zeros((), field.dtype))


# --------------------------------------------------------------------------
# Vectorized tridiagonal solve (Thomas / TDMA)
# --------------------------------------------------------------------------
def tdma(a, b, c, d):
    """Solve a tridiagonal system per column, vectorized over the entity axis.

    For each entity, solve ``M x = d`` where ``M`` has sub-diagonal ``a``,
    diagonal ``b`` and super-diagonal ``c`` along the **last** axis (length K):

        b[0] x[0] + c[0] x[1]                       = d[0]
        a[k] x[k-1] + b[k] x[k] + c[k] x[k+1]       = d[k]      (0<k<K-1)
        a[K-1] x[K-2] + b[K-1] x[K-1]               = d[K-1]

    ``a[...,0]`` (no sub-diagonal on the first row) and ``c[...,K-1]`` (no
    super-diagonal on the last) are ignored — the scans initialise from 0 so
    their values do not matter. Inputs broadcast over leading dims; returns
    ``x`` with the same shape as ``d``.

    Implemented as a forward elimination scan + a reverse back-substitution
    scan, so it is differentiable (grad-checked against finite differences in
    ``test_ops.py``) and AD-stable.
    """
    a, b, c, d = (jnp.asarray(v) for v in (a, b, c, d))
    batch = jnp.broadcast_shapes(a.shape[:-1], b.shape[:-1], c.shape[:-1], d.shape[:-1])
    K = b.shape[-1]
    bshape = batch + (K,)
    # move the tridiagonal axis to the front for lax.scan; broadcast batch dims
    aT = jnp.moveaxis(jnp.broadcast_to(a, bshape), -1, 0)
    bT = jnp.moveaxis(jnp.broadcast_to(b, bshape), -1, 0)
    cT = jnp.moveaxis(jnp.broadcast_to(c, bshape), -1, 0)
    dT = jnp.moveaxis(jnp.broadcast_to(d, bshape), -1, 0)

    zero = jnp.zeros(batch, dtype=b.dtype)

    def forward(carry, xs):
        cp_prev, dp_prev = carry
        ai, bi, ci, di = xs
        m = bi - ai * cp_prev          # at row 0, cp_prev=0 ⇒ m=b[0] (a[0] irrelevant)
        cp = ci / m
        dp = (di - ai * dp_prev) / m
        return (cp, dp), (cp, dp)

    _, (cp, dp) = lax.scan(forward, (zero, zero), (aT, bT, cT, dT))

    def backward(x_next, xs):
        cp_i, dp_i = xs
        x_i = dp_i - cp_i * x_next      # at row K-1, x_next=0 ⇒ x=dp[K-1]
        return x_i, x_i

    _, xT = lax.scan(backward, zero, (cp, dp), reverse=True)
    return jnp.moveaxis(xT, 0, -1)
