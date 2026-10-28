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

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import column_kernel


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
def ordered_sums_enabled() -> bool:
    """``FESOM_ORDERED_SUMS=1`` (``gpu.ordered_sums``, read at trace time): :func:`scatter_add` calls that
    pass an inverse map (``inv=``) assemble their sums as ORDERED gathers (:func:`ordered_sum`) instead of a
    ``segment_sum``, which a GPU lowers to atomic scatter-adds (slow for f64 3-D fields, and
    non-deterministic). Off by default: the ``inv`` maps are then not even built (``mesh.inv_* = None``)."""
    return os.environ.get("FESOM_ORDERED_SUMS", "0").strip() not in ("", "0")


def ordered_sums_barrier() -> tuple[bool, bool]:
    """``(on the values, on the sums)``: where :func:`ordered_sum` puts an ``optimization_barrier``.
    ``FESOM_ORDERED_SUMS=1`` (``gpu.ordered_sums``, the recommended form): on the sums, so XLA cannot merge
    the gather chain into its consumers. Without it (``=2``, A/B only) the chains on 4 A100 became one
    8.6 ms multi-output fusion of ~90 gathers with the PP/TKE column work (job 28292256: 43.2 ms/step
    against 39.6 with the barrier, job 28293421). ``=3``: on the values too (no producer fused into the
    chain; 39.9 ms/step, job 28293422). The barriers do not change values: all three give the same
    numbers (CPU, one forced CORE2 step; GPU, the same noise-run states)."""
    v = os.environ.get("FESOM_ORDERED_SUMS", "0").strip()
    return v == "3", v in ("1", "3")


def inverse_map(seg, num_segments: int, width: int | None = None):
    """Host numpy inverse of an index array, for :func:`ordered_sum`.

    ``seg`` (``[*S]`` int, flattened to ``n`` entries; ``seg < 0`` or ``>= num_segments`` = no target)
    → ``inv`` ``[num_segments, D]`` int32: row ``t`` lists the flattened positions ``i`` with
    ``seg[i] == t`` in ASCENDING order (the order a sequential scatter applies its updates), padded with
    the sentinel position ``n`` (read as 0 by :func:`ordered_sum`). ``D`` = the largest row count (at
    least 1), or ``width`` (>= it) to pad several shards to one width. Integer bookkeeping only: exact."""
    import numpy as np
    s = np.asarray(seg).reshape(-1).astype(np.int64)
    n = s.size
    pos = np.arange(n, dtype=np.int64)
    ok = (s >= 0) & (s < num_segments)
    s, pos = s[ok], pos[ok]
    order = np.argsort(s, kind="stable")            # stable: ascending position within a row
    s, pos = s[order], pos[order]
    cnt = np.bincount(s, minlength=num_segments)
    D = max(int(cnt.max()) if cnt.size else 0, 1)
    if width is not None:
        if width < D:
            raise ValueError(f"inverse_map: width {width} < the largest row count {D}")
        D = int(width)
    start = np.cumsum(cnt) - cnt
    inv = np.full((num_segments, D), n, dtype=np.int32)
    inv[s, np.arange(s.size) - start[s]] = pos
    return inv


def ordered_vjp_enabled() -> bool:
    """``FESOM_ORDERED_VJP=1`` (read at trace time): :func:`ordered_sum` with its explicit transpose when the caller
    gives the forward index (:func:`scatter_add` does)."""
    return os.environ.get("FESOM_ORDERED_VJP", "0").strip() not in ("", "0")


def ordered_sum(vals_flat, inv, seg_flat=None):
    """``out[t] = 0 + vals_flat[inv[t, 0]] + vals_flat[inv[t, 1]] + ...`` as an explicit left-to-right
    chain over the ``D`` slots (no ``jnp.sum``, which XLA may reassociate); the sentinel position
    ``vals_flat.shape[0]`` reads 0 (``mode="fill"``), so a short row adds +0.0 after its last entry. The
    additions are a sequential scatter's in its update order, so on the CPU this is the ``segment_sum``
    to round-off (XLA may fuse the values' producer into the chain and contract into FMAs); on a GPU it
    has no atomics and is deterministic. ``vals_flat``: ``[n, *rest]``, ``inv``: ``[N, D]`` →
    ``[N, *rest]``. Default AD (the gathers' transpose, a scatter-add of the cotangent — atomic on a GPU); with
    ``seg_flat`` (the index ``inv`` inverts, ``[n]``) and :func:`ordered_vjp_enabled` the transpose is the gather
    ``vals_bar[i] = out_bar[seg_flat[i]]`` (0 where ``seg_flat`` has no target) instead: no atomics. The forward
    is the same either way."""
    if seg_flat is not None and ordered_vjp_enabled():
        return _ordered_sum_vjp(vals_flat, jnp.asarray(inv), jnp.asarray(seg_flat))
    return _ordered_chain(vals_flat, inv)


def _ordered_chain(vals_flat, inv):
    inv = jnp.asarray(inv)
    bar_in, bar_out = ordered_sums_barrier()
    if bar_in:
        vals_flat = lax.optimization_barrier(vals_flat)
    out = None
    for j in range(inv.shape[1]):
        g = jnp.take(vals_flat, inv[:, j], axis=0, mode="fill", fill_value=0)
        out = (jnp.zeros_like(g) + g) if out is None else out + g
    return lax.optimization_barrier(out) if bar_out else out


@jax.custom_vjp
def _ordered_sum_vjp(vals_flat, inv, seg_flat):
    return _ordered_chain(vals_flat, inv)


def _ordered_sum_vjp_fwd(vals_flat, inv, seg_flat):
    return _ordered_chain(vals_flat, inv), (seg_flat, inv.shape[0])


def _ordered_sum_vjp_bwd(res, out_bar):
    seg, n_out = res
    ok = (seg >= 0) & (seg < n_out)
    g = jnp.take(out_bar, jnp.where(ok, seg, 0), axis=0)
    g = jnp.where(ok.reshape((-1,) + (1,) * (g.ndim - 1)), g, jnp.zeros_like(g))
    return g, None, None


_ordered_sum_vjp.defvjp(_ordered_sum_vjp_fwd, _ordered_sum_vjp_bwd)


def scatter_add(vals, seg, num_segments: int, inv=None):
    """Differentiable masked segment-sum.

    Sums ``vals`` into ``num_segments`` output rows according to ``seg``. The
    leading dims of ``vals`` must match ``seg``'s shape (both are flattened);
    trailing dims of ``vals`` are summed elementwise. Entries with ``seg < 0``
    (the boundary sentinel) contribute nothing.

    ``inv`` (optional): ``seg``'s :func:`inverse_map` (e.g. ``mesh.inv_edges`` for ``mesh.edges``).
    With it and :func:`ordered_sums_enabled`, the sum is the :func:`ordered_sum` (ordered gathers, no
    atomics on a GPU); otherwise the ``segment_sum`` (the default, byte-identical).

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
    if inv is not None and ordered_sums_enabled():
        if tuple(inv.shape[:1]) != (num_segments,):
            raise ValueError(f"scatter_add: inv rows {inv.shape[0]} != num_segments {num_segments}")
        return ordered_sum(vals_flat, inv, seg_flat)

    valid = seg_flat >= 0
    safe_seg = jnp.where(valid, seg_flat, 0)
    vmask = valid.reshape((-1,) + (1,) * (vals_flat.ndim - 1))
    safe_vals = jnp.where(vmask, vals_flat, 0)
    return jax.ops.segment_sum(safe_vals, safe_seg, num_segments=num_segments)


def scatter_add_edges_to_nodes(vals, edges, num_nodes: int, inv=None):
    """Edge→node scatter: add each edge's two endpoint contributions
    ``vals[e,0]``→node ``edges[e,0]`` and ``vals[e,1]``→node ``edges[e,1]``.

    ``vals``: ``[edge2D, 2, *rest]``  ``edges``: ``[edge2D, 2]``.
    """
    return scatter_add(vals, edges, num_nodes, inv=inv)


def scatter_add_edges_to_elems(vals, edge_tri, num_elems: int, inv=None):
    """Edge→element scatter (e.g. biharmonic viscosity); ``edge_tri`` carries
    ``-1`` for a boundary edge's missing neighbour, masked out automatically.

    ``vals``: ``[edge2D, 2, *rest]``  ``edge_tri``: ``[edge2D, 2]``.
    """
    return scatter_add(vals, edge_tri, num_elems, inv=inv)


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
def column_unroll() -> bool:
    """``FESOM_COLUMN_UNROLL=1`` (read at trace time): the level scans (:func:`tdma` and the TKE
    min-scans / Thomas solve) run as :func:`unrolled_scan` on the ``[..., K]`` arrays as they are,
    instead of ``moveaxis`` + ``lax.scan``. The default scan needs the level axis in front, and
    XLA's layout assignment is free to make that transpose a bitcast by laying the whole step out
    level-major — which it does as soon as nothing else pins node-major (the compact halo
    transport, CORE2 4 A100 job 28286769: 293 transposes, one-launch 3-D kernels 5-10x slower)."""
    return os.environ.get("FESOM_COLUMN_UNROLL", "0").strip() not in ("", "0")


def unrolled_scan(body, init, xs, reverse: bool = False):
    """``lax.scan(body, init, xs, reverse)`` semantics over the LAST axis of every leaf of ``xs``
    (``[..., K]``), unrolled in Python: column ``k`` is ``v[..., k]`` and the per-column outputs are
    stacked back on the last axis. The body runs the same operations in the same order as under
    ``lax.scan``, so the values are the scan's; there is no transpose and no while loop."""
    K = jax.tree.leaves(xs)[0].shape[-1]
    carry, ys = init, [None] * K
    for k in (range(K - 1, -1, -1) if reverse else range(K)):
        carry, ys[k] = body(carry, jax.tree.map(lambda v: v[..., k], xs))
    return carry, jax.tree.map(lambda *cols: jnp.stack(cols, axis=-1), *ys)


def _tdma_forward(carry, xs):
    """:func:`tdma`'s forward-elimination step (one level)."""
    cp_prev, dp_prev = carry
    ai, bi, ci, di = xs
    m = bi - ai * cp_prev          # at row 0, cp_prev=0 ⇒ m=b[0] (a[0] irrelevant)
    cp = ci / m
    dp = (di - ai * dp_prev) / m
    return (cp, dp), (cp, dp)


def _tdma_backward(x_next, xs):
    """:func:`tdma`'s back-substitution step (one level)."""
    cp_i, dp_i = xs
    x_i = dp_i - cp_i * x_next      # at row K-1, x_next=0 ⇒ x=dp[K-1]
    return x_i, x_i


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
    zero = jnp.zeros(batch, dtype=b.dtype)
    forward, backward = _tdma_forward, _tdma_backward

    if len(bshape) == 2 and column_kernel.enabled():
        abcd = tuple(jnp.broadcast_to(v, bshape) for v in (a, b, c, d))
        return column_kernel.thomas(forward, backward, *abcd)
    if column_unroll():
        abcd = tuple(jnp.broadcast_to(v, bshape) for v in (a, b, c, d))
        _, (cp, dp) = unrolled_scan(forward, (zero, zero), abcd)
        return unrolled_scan(backward, zero, (cp, dp), reverse=True)[1]

    # move the tridiagonal axis to the front for lax.scan; broadcast batch dims
    aT = jnp.moveaxis(jnp.broadcast_to(a, bshape), -1, 0)
    bT = jnp.moveaxis(jnp.broadcast_to(b, bshape), -1, 0)
    cT = jnp.moveaxis(jnp.broadcast_to(c, bshape), -1, 0)
    dT = jnp.moveaxis(jnp.broadcast_to(d, bshape), -1, 0)

    _, (cp, dp) = lax.scan(forward, (zero, zero), (aT, bT, cT, dT))
    _, xT = lax.scan(backward, zero, (cp, dp), reverse=True)
    return jnp.moveaxis(xT, 0, -1)
