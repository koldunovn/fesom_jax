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

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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


# ===========================================================================
# Fine mesh -> coarse mesh aggregation (NG5 -> CORE2)
# ===========================================================================

class RemapMap(NamedTuple):
    """Host-precomputed fine->coarse assignment. Built once, reused for every snapshot.

    ``fine_cell``: (n_fine,) index of the coarse node each fine node belongs to.
    ``fine_area``: (n_fine,) the fine node's horizontal control-volume area [m^2].
    ``n_coarse``: number of coarse nodes.
    ``coarse_depth``/``fine_depth``: interface depths [m, negative down] of each mesh,
    for the vertical remap.
    """
    fine_cell: jnp.ndarray
    fine_area: jnp.ndarray
    n_coarse: int
    fine_depth: jnp.ndarray
    coarse_depth: jnp.ndarray


def build_remap(fine_mesh_dir, coarse_mesh_dir) -> RemapMap:
    """Assign every fine node to the nearest coarse node — a Voronoi partition of the
    fine mesh by the coarse nodes.

    Nearest neighbour is computed on the UNIT SPHERE in 3-D Cartesian coordinates, not in
    longitude/latitude: a lon/lat metric distorts badly toward the poles and breaks
    outright across the dateline, and both regions matter here (the Arctic is where the
    teacher's eddies are smallest and the Southern Ocean spans the seam).

    Pure host-side numpy/scipy, run once and cached — never inside a training step.
    Takes the exported ``.npy`` mesh directories rather than loaded meshes, because the
    NG5 bundle is 16 GB and only two of its arrays are needed here.
    """
    from pathlib import Path

    from scipy.spatial import cKDTree

    fine_dir, coarse_dir = Path(fine_mesh_dir), Path(coarse_mesh_dir)

    def _lonlat(d):
        geo = np.load(d / "geo_coord_nod2D.npy")          # radians
        return geo[:, 0], geo[:, 1]

    def _xyz(lon, lat):
        cl = np.cos(lat)
        return np.column_stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)])

    flon, flat = _lonlat(fine_dir)
    clon, clat = _lonlat(coarse_dir)
    tree = cKDTree(_xyz(clon, clat))
    _, fine_cell = tree.query(_xyz(flon, flat), k=1, workers=-1)

    # ⚠️ mmap. On NG5 `area.npy` is (7402886, 70) f8 = 4.1 GB on disk and only its surface
    # column (59 MB) is wanted; reading the whole array to slice one column is the difference
    # between a 4 GB allocation and nothing. `np.array(...)` forces the copy so the mmap is not
    # kept alive inside the returned RemapMap.
    farea = np.load(fine_dir / "area.npy", mmap_mode="r")
    farea = np.array(farea[:, 0] if farea.ndim == 2 else farea, dtype=np.float64)
    return RemapMap(
        fine_cell=jnp.asarray(fine_cell.astype(np.int32)),
        fine_area=jnp.asarray(np.asarray(farea, dtype=np.float64)),
        n_coarse=int(clon.size),
        fine_depth=jnp.asarray(np.load(fine_dir / "zbar.npy")),
        coarse_depth=jnp.asarray(np.load(coarse_dir / "zbar.npy")),
    )


def aggregate_horizontal(field, rmap: RemapMap, *, fine_mask=None):
    """Area-weighted mean of a fine-mesh field over each coarse node's cell.

    ``field``: (n_fine,) or (n_fine, nl). ``fine_mask``: same shape, 1 wet / 0 dry.
    Conservative: the area-weighted integral over each coarse cell is preserved exactly,
    which is what lets the aggregated fields be used as conservative training targets.
    Coarse cells with no wet fine node come back as 0 with weight 0 (see
    :func:`aggregate_horizontal_with_weight` if you need to tell "empty" from "zero").
    """
    val, wgt = aggregate_horizontal_with_weight(field, rmap, fine_mask=fine_mask)
    return val


def aggregate_horizontal_with_weight(field, rmap: RemapMap, *, fine_mask=None):
    """As :func:`aggregate_horizontal`, but also returns the summed weight per coarse
    cell — zero where a coarse cell had no wet fine node underneath it."""
    x2 = field.ndim == 2
    f = field if x2 else field[:, None]
    m = (jnp.ones_like(f) if fine_mask is None else
         (fine_mask if fine_mask.ndim == 2 else jnp.broadcast_to(fine_mask[:, None], f.shape))
         .astype(f.dtype))
    w = rmap.fine_area[:, None] * m
    num = ops.scatter_add(w * f, rmap.fine_cell, rmap.n_coarse)
    den = ops.scatter_add(w, rmap.fine_cell, rmap.n_coarse)
    out = jnp.where(den > 0.0, num / jnp.where(den > 0.0, den, 1.0), 0.0)
    return (out, den) if x2 else (out[:, 0], den[:, 0])


def remap_vertical(field, fine_depth, coarse_depth, *, fine_mask=None):
    """Thickness-weighted conservative remap of a column from one set of layer
    interfaces to another (NG5's 70 levels -> CORE2's 48).

    ``field``: (..., n_fine_lev) values in each fine LAYER (between consecutive
    interfaces). Returns (..., n_coarse_lev) layer values.

    Conservative by construction: each coarse layer takes the thickness-weighted mean of
    the fine layers overlapping it, so the column integral is preserved. Below the
    deepest wet fine layer the weight is zero, so a coarse layer that reaches past the
    fine column's bottom is filled from what exists rather than from zeros — the
    sub-bottom-zero trap that has bitten this project before.
    """
    fz = jnp.asarray(fine_depth, dtype=jnp.float64)
    cz = jnp.asarray(coarse_depth, dtype=jnp.float64)
    f_top, f_bot = fz[:-1], fz[1:]                       # (nf,) negative, decreasing
    c_top, c_bot = cz[:-1], cz[1:]                       # (nc,)

    # overlap[c, f] = thickness shared by coarse layer c and fine layer f
    overlap = (jnp.minimum(c_top[:, None], f_top[None, :])
               - jnp.maximum(c_bot[:, None], f_bot[None, :]))
    overlap = jnp.maximum(overlap, 0.0)                  # (nc, nf)

    val = jnp.asarray(field, dtype=jnp.float64)
    m = jnp.ones_like(val) if fine_mask is None else jnp.asarray(fine_mask, val.dtype)
    w = jnp.einsum("cf,...f->...cf", overlap, m)
    num = jnp.einsum("...cf,...f->...c", w, val)
    den = jnp.sum(w, axis=-1)
    return jnp.where(den > 0.0, num / jnp.where(den > 0.0, den, 1.0), 0.0)


def elem_to_node(field_elem, mesh: Mesh, *, elem_mask=None):
    """Area-weighted element->node collocation.

    FESOM stores velocities on ELEMENTS while tracers live on nodes, so the teacher's
    u,v arrive elemental (and 1.93x larger than the nodal fields on CORE2). Everything
    downstream — the aggregation map, the eddy-flux products — is node-based, so this is
    where the two meet. ``field_elem``: (elem2D,) or (elem2D, nl).

    ``elem_mask`` (same shape, 1 wet / 0 dry) excludes dry elements from BOTH the numerator
    and the denominator, exactly as ``fine_mask`` does for nodes in
    :func:`aggregate_horizontal_with_weight`. A node whose every adjacent element is dry
    comes back 0.

    ⚠️ **Pass it whenever the field can contain below-bottom entries, which on real FESOM
    output is always.** Without a mask this function averages every adjacent element
    unconditionally, so a WET node next to a DRY element inherits a third of whatever that
    element holds — and FESOM writes ``_FillValue = 9.96921e+36`` there. Measured on NG5
    month 01 (2026-08-08): 473,770 of 3,778,615 wet (node, level) cells came out at ~7.6e+36,
    starting at level 3 and reaching 12.5% of the column. The surface was clean, which is
    exactly why it survived a look at a surface map. Zeroing the fill before the call is
    **not** sufficient on its own: the dry element then contributes a spurious zero and
    dilutes the node's velocity toward the bottom, which looks physical.
    """
    x2 = field_elem.ndim == 2
    fe = field_elem if x2 else field_elem[:, None]
    w = jnp.broadcast_to(mesh.elem_area[:, None], fe.shape)
    if elem_mask is not None:
        m = elem_mask if elem_mask.ndim == 2 else elem_mask[:, None]
        w = w * jnp.broadcast_to(m, fe.shape).astype(w.dtype)
    vals = jnp.broadcast_to((w * fe)[:, None, :], (fe.shape[0], 3, fe.shape[1]))
    wts = jnp.broadcast_to(w[:, None, :], vals.shape)
    num = ops.scatter_add(vals, mesh.elem_nodes, mesh.nod2D)
    den = ops.scatter_add(wts, mesh.elem_nodes, mesh.nod2D)
    out = jnp.where(den > 0.0, num / jnp.where(den > 0.0, den, 1.0), 0.0)
    return out if x2 else out[:, 0]
