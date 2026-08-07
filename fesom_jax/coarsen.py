"""Coarse-graining for the NN-GM teacher pipeline — part I: the scale filter.

The teacher is an eddy-resolving NG5 run; the student is CORE2. To learn what the student
must parameterize, we need the teacher's state and its eddy fluxes *as the student would
see them*: smoothed to the student's resolution. This module provides the smoothing half
of that (the NG5→CORE2 aggregation is part II, `scripts/nn_eddy/coarsen_stream.py`).

**The filter.** Iterated mesh-Laplacian diffusion — the standard diffusion-based spatial
filter (Grooms et al. 2021, GCM-Filters) written for FESOM's triangular finite elements:

    phi <- phi + (kappa dt) * Laplacian(phi),  applied n_steps times

which is a Gaussian-like low-pass whose filter scale after ``n`` steps is
``L = sqrt(2 n kappa_dt)``. The Laplacian is the standard finite-element one with a
lumped mass matrix, built from the same ``gradient_sca`` shape-function gradients the
model's own operators use:

    (Lap phi)_i = -(1/A_i) * sum_e A_e * (grad N_i . grad phi|_e)

so it is consistent with FESOM's own discretization rather than a second, independent
notion of a Laplacian on the same mesh.

**Properties, all tested.** A constant field is preserved exactly (the stiffness matrix
annihilates constants, in floating point too, since the row sums cancel term by term).
The area-weighted mean is conserved to round-off, because the operator is symmetric and
the weights are the lumped areas — this is what makes the filtered fields usable as
conservative training targets. Dry nodes and sub-bottom levels never contribute and are
never contaminated: the mask enters as a zero weight on element contributions, which is
also the correct no-flux (insulating) condition at coasts and at the bottom.

**Cost.** ``n_steps`` scales as ``(L/dx)^2``, so filtering a 5 km mesh directly to a
200 km scale would need thousands of iterations over 7.4 M nodes. That is the wrong way
round: aggregate first (conservative, exact, one pass) and use this filter for the modest
extra smoothing on the coarse mesh, where ``L/dx`` is ~2 and a dozen steps suffice. See
:func:`steps_for_scale`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import ops
from .mesh import Mesh

# Explicit diffusion on triangles is stable for kappa*dt <= C * dx^2; C = 1/8 is the
# conservative choice used here (1/4 is the 1-D limit, triangles are tighter).
STABILITY_C = 0.125


def laplacian(mesh: Mesh, field, *, node_mask=None):
    """Finite-element Laplacian of a nodal field, with a lumped mass matrix.

    ``field``: ``(nod2D,)`` or ``(nod2D, nl)``. Returns the same shape.
    ``node_mask``: optional ``(nod2D,)`` / ``(nod2D, nl)`` of 1.0 (wet) and 0.0 (dry).
    Masked nodes contribute nothing and receive nothing, which is the no-flux condition
    at coasts and below the bottom.
    """
    x2 = field.ndim == 2
    phi = field if x2 else field[:, None]                      # (N, L)
    nl = phi.shape[1]
    if node_mask is None:
        m = jnp.ones_like(phi)
    else:
        m = node_mask if node_mask.ndim == 2 else jnp.broadcast_to(node_mask[:, None], phi.shape)
        m = m.astype(phi.dtype)
    phi = phi * m

    g = mesh.gradient_sca                                      # (E,6) 0:3 = dN/dx, 3:6 = dN/dy
    gx, gy = g[:, 0:3], g[:, 3:6]
    en = mesh.elem_nodes                                       # (E,3)
    phi_e = ops.gather_nodes_to_elem(phi, en)                  # (E,3,L)
    mask_e = ops.gather_nodes_to_elem(m, en)                   # (E,3,L)

    # An element contributes only where all three of its nodes are wet — otherwise the
    # gradient would be taken across a coastline.
    wet_e = jnp.prod(mask_e, axis=1)                           # (E,L)
    grad_x = jnp.sum(gx[:, :, None] * phi_e, axis=1) * wet_e   # (E,L)
    grad_y = jnp.sum(gy[:, :, None] * phi_e, axis=1) * wet_e

    # (E,3,L): -A_e * (dN_i/dx * grad_x + dN_i/dy * grad_y) for each of the 3 nodes
    area_e = mesh.elem_area[:, None, None]                     # (E,1,1)
    contrib = -area_e * (gx[:, :, None] * grad_x[:, None, :]
                         + gy[:, :, None] * grad_y[:, None, :])
    acc = ops.scatter_add(contrib, en, mesh.nod2D)              # (N,L)

    area_n = mesh.area[:, :nl] if mesh.area.ndim == 2 else mesh.area[:, None]
    inv_area = jnp.where(area_n > 0.0, 1.0 / jnp.where(area_n > 0.0, area_n, 1.0), 0.0)
    out = acc * inv_area * m
    return out if x2 else out[:, 0]


def filter_field(mesh: Mesh, field, *, n_steps: int, kappa_dt, node_mask=None,
                 check_stability: bool = True):
    """Apply ``n_steps`` of Laplacian diffusion. ``kappa_dt`` is scalar or ``(nod2D,)``.

    The resulting filter scale is ``sqrt(2 * n_steps * kappa_dt)`` (see :func:`scale_of`).
    ``kappa_dt`` may vary in space — that is how a "2x the *local* target resolution"
    filter is built on a variable-resolution mesh (:func:`plan_filter`).

    ``check_stability`` raises if ``kappa_dt`` exceeds the explicit-diffusion limit
    anywhere. Without it the failure is silent and spectacular: explicit diffusion past
    the limit amplifies grid-scale noise instead of removing it, and the field reaches
    1e47 in a few dozen steps while still looking like a filter call that "ran".
    """
    kdt = jnp.asarray(kappa_dt)
    if check_stability:
        limit = STABILITY_C * node_resolution(mesh) ** 2         # (nod2D,)
        worst = float(jnp.max(jnp.broadcast_to(jnp.atleast_1d(kdt), limit.shape) / limit))
        if worst > 1.0 + 1e-9:
            raise ValueError(
                f"kappa_dt exceeds the explicit-diffusion stability limit by {worst:.2f}x "
                f"at the finest node. On a variable-resolution mesh the limit is set by the "
                f"SMALLEST cell, not the median. Use plan_filter() for a scale that follows "
                f"the local resolution, or pass check_stability=False if you know better.")
    if kdt.ndim == 1 and field.ndim == 2:
        kdt = kdt[:, None]

    def body(phi, _):
        return phi + kdt * laplacian(mesh, phi, node_mask=node_mask), None

    out, _ = jax.lax.scan(body, field, xs=None, length=int(n_steps))
    return out


# ---------------------------------------------------------------------------
# choosing (n_steps, kappa_dt) for a target scale

def scale_of(n_steps: int, kappa_dt) -> float:
    """Filter scale [m] produced by ``n_steps`` steps of ``kappa_dt``."""
    return float(jnp.sqrt(2.0 * n_steps * jnp.asarray(kappa_dt).max()))


def steps_for_scale(target_scale, dx, *, safety: float = 1.0) -> int:
    """Fewest steps that reach ``target_scale`` without violating explicit stability.

    ``target_scale`` and ``dx`` are in metres and may be arrays (per node); the returned
    step count is the one the *tightest* point needs, since all points march together.
    Cost grows as ``(target_scale / dx)^2`` — a factor-20 scale separation costs
    thousands of iterations, which is why the pipeline aggregates first and filters after.
    """
    L = jnp.asarray(target_scale, dtype=jnp.float64)
    d = jnp.asarray(dx, dtype=jnp.float64)
    need = L ** 2 / (2.0 * STABILITY_C * safety * d ** 2)
    return int(jnp.ceil(jnp.max(need)))


def kappa_dt_for_scale(target_scale, n_steps: int):
    """Per-step diffusivity reaching ``target_scale`` in ``n_steps`` — scalar or per node."""
    L = jnp.asarray(target_scale, dtype=jnp.float64)
    return L ** 2 / (2.0 * n_steps)


def plan_filter(mesh: Mesh, factor: float, *, safety: float = 0.5):
    """Filter parameters for a scale of ``factor`` x the LOCAL node spacing.

    Returns ``(n_steps, kappa_dt)`` with ``kappa_dt`` per node. This is the intended way
    to use the filter on a real mesh, and it has a pleasant property: with the target
    scale proportional to the local spacing, ``kappa_dt(x) = factor^2 dx(x)^2 / (2n)``
    and the stability requirement ``kappa_dt <= C dx^2`` reduces to ``n >= factor^2/(2C)``
    — **independent of the mesh**. So a variable-resolution mesh costs no more iterations
    than a uniform one, whereas a single global scale would be paced by the finest cell
    (on the pi mesh: 35 km vs a 230 km median, a 44x cost penalty and, if you take the
    median by mistake, an instant blow-up).

    The plan's coarse-graining target is ``factor = 2`` (twice the local resolution of
    the student mesh).

    ``safety`` keeps the step off the stability boundary. It is not decoration: at
    exactly the limit the grid-scale mode has amplification factor -1, so it oscillates
    forever instead of decaying — the one mode a filter most needs to remove. The default
    0.5 costs 2x the iterations and damps it properly.
    """
    dx = node_resolution(mesh)
    n = int(jnp.ceil(factor ** 2 / (2.0 * STABILITY_C * safety)))
    return n, kappa_dt_for_scale(factor * dx, n)


def node_resolution(mesh: Mesh):
    """Representative grid spacing at each node [m], from its lumped area.

    ``sqrt(A_i)`` is the natural length scale of a node's control volume; for the
    equilateral case the triangle edge is ~1.5x this, but the filter only needs a
    consistent local length, and using the same measure everywhere keeps the stability
    bound honest.
    """
    a = mesh.area[:, 0] if mesh.area.ndim == 2 else mesh.area
    return jnp.sqrt(jnp.maximum(a, 0.0))
