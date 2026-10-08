"""Cost functionals for adjoint studies, and running accumulators over a window.

Weighted means
    :func:`box_mean_T_vw` weights each wet cell by its LIVE volume ``hnode[:, k]·areasvol[:, k]``
    (zstar layer thicknesses change every step). An unweighted mean over cells of different
    volume is not conserved by transport between them, so its gradient contains velocity
    sensitivities that belong to the cost's weighting, not to the ocean. :func:`sst_area_mean`
    and :func:`ssh_area_mean` weight the surface nodes by area.

Mixed-layer depth
    :func:`mld_field` is the density-threshold MLD of :func:`fesom_jax.obs_compare.
    mld_density_threshold` on the live mid-layer depths from :func:`fesom_jax.ale.live_geometry`.
    Where the column has no crossing (mixed to the bottom) the threshold routine returns 0 and
    ``valid=False``; ``invalid="column"`` (default) puts the depth of the column's deepest wet
    MID-layer there (:func:`deepest_wet_mid_depth`), which is the deepest value a valid
    crossing can take and the limit it reaches as the crossing deepens to the last wet level,
    so the field is continuous at the valid→invalid switch; ``invalid="zero"`` keeps the
    routine's 0 (the convention of earlier MLD calibrations).

Running accumulators
    A time-mean cost over a window must add a value after EVERY step, which a wrapper around an
    opaque chunk function cannot do (it only sees chunk boundaries). :func:`with_running_costs`
    wraps the STEP function instead: the new step carries ``(State, acc)`` and adds each
    accumulator's value on the post-step state. The carry change is visible everywhere the
    State was — the chunk function's input and output, ``init_state`` (``(state,
    init_accumulators(...))``), the loss (a function of the carry) and the cotangent traces
    (whose leaf names gain a ``0.``/``1.`` prefix). :func:`terminal` turns a State cost into a
    loss on the carry that ignores the accumulators (a terminal seed, as a screen wants);
    :func:`running_mean` reads one accumulator back as a time mean.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp

from . import ale, obs_compare

__all__ = ["box_volume_weights", "box_mean_T_vw", "sst_area_mean", "ssh_area_mean",
           "mld_field", "deepest_wet_mid_depth", "column_bottom_depth", "with_running_costs", "init_accumulators",
           "terminal", "running_mean"]


# ---------------------------------------------------------------------------
# weighted means


def box_volume_weights(state, mesh, node_mask, level_mask):
    """``[nod2D, nl]`` live cell volumes ``hnode·areasvol`` [m³] on the wet layers
    (``node_layer_mask``) of the box ``node_mask[:, None] & level_mask[None, :]``; 0 elsewhere.
    ``node_mask`` is ``[nod2D]`` bool (``None`` = every node), ``level_mask`` ``[nl]`` bool
    (``None`` = every level)."""
    m = jnp.asarray(mesh.node_layer_mask)
    if node_mask is not None:
        m = m & jnp.asarray(node_mask, bool)[:, None]
    if level_mask is not None:
        m = m & jnp.asarray(level_mask, bool)[None, :]
    return jnp.where(m, state.hnode * mesh.areasvol, 0.0)


def box_mean_T_vw(state, mesh, node_mask, level_mask):
    """Volume-weighted mean temperature of the box (see :func:`box_volume_weights`)."""
    w = box_volume_weights(state, mesh, node_mask, level_mask)
    return jnp.sum(w * state.T) / jnp.sum(w)


def _surface_weights(mesh, node_mask):
    m = jnp.asarray(mesh.node_layer_mask[:, 0])
    if node_mask is not None:
        m = m & jnp.asarray(node_mask, bool)
    return jnp.where(m, mesh.areasvol[:, 0], 0.0)


def sst_area_mean(state, mesh, node_mask=None):
    """Area-weighted mean of the top-layer temperature over the wet surface nodes in
    ``node_mask`` (``None`` = all)."""
    w = _surface_weights(mesh, node_mask)
    return jnp.sum(w * state.T[:, 0]) / jnp.sum(w)


def ssh_area_mean(state, mesh, node_mask=None):
    """Area-weighted mean of ``eta_n`` over the wet surface nodes in ``node_mask``."""
    w = _surface_weights(mesh, node_mask)
    return jnp.sum(w * state.eta_n) / jnp.sum(w)


# ---------------------------------------------------------------------------
# mixed-layer depth


def column_bottom_depth(state, mesh):
    """``[nod2D]`` depth [m, positive] of each column's bottom interface in the live geometry
    (:func:`fesom_jax.ale.live_geometry`), in the same datum as the MLD values (z = 0). Under
    zstar the bottom does not move, so this is the static bottom depth; the water-column
    thickness ``Σ hnode`` differs from it by the surface displacement.

    This is NOT the value :func:`mld_field` puts on columns mixed to the bottom: it lies half
    the bottom layer below the deepest valid crossing (:func:`deepest_wet_mid_depth`), so an
    MLD that switched to it would jump by ``hnode[kb-1]/2`` when a column became fully mixed."""
    zbar_3d_n, _ = ale.live_geometry(mesh, state.hnode)
    kb = jnp.asarray(mesh.nlevels_nod2D) - 1                     # bottom interface, 0-based
    return -jnp.take_along_axis(zbar_3d_n, kb[:, None], axis=1)[:, 0]


def _deepest_wet_mid(Z_3d_n, mesh):
    kd = jnp.clip(jnp.asarray(mesh.nlevels_nod2D) - 2, 0, Z_3d_n.shape[1] - 1)
    return -jnp.take_along_axis(Z_3d_n, kd[:, None], axis=1)[:, 0]


def deepest_wet_mid_depth(state, mesh):
    """``[nod2D]`` depth [m, positive] of each column's deepest wet MID-layer, ``-Z_3d_n[kd]``
    with ``kd = nlevels_nod2D - 2`` (the last True of ``node_layer_mask``), on the live geometry
    of :func:`fesom_jax.ale.live_geometry` — the same array and datum (z = 0) the density
    threshold interpolates on. No valid MLD is deeper than this (see :func:`mld_field`). Meaningless on dry surface nodes (``mld_field`` puts 0 there)."""
    _, Z_3d_n = ale.live_geometry(mesh, state.hnode)
    return _deepest_wet_mid(Z_3d_n, mesh)


def mld_field(state, mesh, *, invalid: str = "column", ref_depth: float = 10.0,
              dsigma: float = 0.03):
    """``[nod2D]`` density-threshold mixed-layer depth [m, positive] on the live geometry.

    Where a valid crossing can lie: :func:`fesom_jax.obs_compare.mld_density_threshold` takes
    ``kc`` = the shallowest WET level (``node_layer_mask``) with mid-depth ``Z_3d_n <=
    -ref_depth`` and σ-excess ``>= 0`` (clipped to ``kc >= 1``), and interpolates between the
    mid-layer depths of ``kc - 1`` and ``kc`` with a weight clipped to [0, 1]. So every valid
    MLD lies in ``[-Z_3d_n[kc-1], -Z_3d_n[kc]]``, hence ``<= -Z_3d_n[kd]`` with ``kd =
    nlevels_nod2D - 2`` the deepest wet layer (:func:`deepest_wet_mid_depth`); it equals that
    depth when ``kc = kd`` and the excess at ``kd`` is exactly 0, and tends to it as that
    excess decreases to 0⁺ (weight ``-e[kd-1] / (e[kd] - e[kd-1]) -> 1``).

    ``invalid`` sets the value where the threshold is never crossed (the column is mixed to the
    bottom): ``"column"`` (default) = :func:`deepest_wet_mid_depth`, the limit above, so the
    field is continuous when the last crossing, at the deepest wet level, disappears (the
    usual path: a stably stratified column that deepens its mixed layer to the bottom). It is
    NOT continuous when the last crossing to disappear is shallower (an excess that is
    non-monotone in depth), nor where the crossing hops between levels — those jumps belong to
    the threshold criterion itself. ``"zero"`` = 0, the value of
    :func:`fesom_jax.obs_compare.mld_density_threshold` itself (bit-identical to it; the
    convention of earlier MLD calibrations). Dry surface nodes are 0 in both modes."""
    if invalid not in ("column", "zero"):
        raise ValueError(f"mld_field: invalid must be 'column' or 'zero', got {invalid!r}")
    _, Z_3d_n = ale.live_geometry(mesh, state.hnode)
    mld, valid = obs_compare.mld_density_threshold(
        state.T, state.S, Z_3d_n, jnp.asarray(mesh.node_layer_mask),
        ref_depth=ref_depth, dsigma=dsigma)
    if invalid == "zero":
        return mld
    wet = jnp.asarray(mesh.node_layer_mask[:, 0])
    return jnp.where(valid, mld, jnp.where(wet, _deepest_wet_mid(Z_3d_n, mesh), 0.0))


# ---------------------------------------------------------------------------
# running accumulators


def init_accumulators(accumulators: Mapping[str, Callable[[Any], Any]], state) -> dict:
    """Zero running sums with the shape and dtype of each accumulator's value on ``state``
    (traced abstractly; nothing is computed). Pair it with the state as the carry:
    ``(state, init_accumulators(accs, state))``."""
    return {k: jnp.zeros(s.shape, s.dtype)
            for k, s in ((k, jax.eval_shape(fn, state)) for k, fn in accumulators.items())}


def with_running_costs(step_fn: Callable[[Any, Any], Any],
                       accumulators: Mapping[str, Callable[[Any], Any]]):
    """Wrap ``step_fn(state, x) -> state`` into a step on the carry ``(state, acc)`` that adds
    ``accumulators[k](new_state)`` to ``acc[k]`` after every step.

    Accumulator values may be scalars (a box mean) or fields (``{"mld": lambda s:
    mld_field(s, mesh)}`` gives a ``[nod2D]`` running sum). Pass the wrapped step to
    :func:`fesom_jax.adjoint.scan_steps` inside the chunk function; the sums then see every
    step, and a chunked reverse sweep differentiates them like any other part of the carry.
    Divide by the number of steps for a time mean (:func:`running_mean`)."""
    accs = dict(accumulators)

    def wrapped(carry, x):
        state, acc = carry
        new = step_fn(state, x)
        return new, {k: acc[k] + fn(new) for k, fn in accs.items()}

    return wrapped


def terminal(cost: Callable[[Any], Any]):
    """A loss on the carry ``(state, acc)`` that evaluates ``cost`` on the final state only and
    ignores the accumulators (a terminal seed with no running injection). A bare State is
    accepted too, so one cost serves both carries."""
    def loss(carry):
        state = carry[0] if isinstance(carry, tuple) else carry
        return cost(state)
    return loss


def running_mean(name: str, n_steps: int):
    """A loss on the carry: the accumulator ``name`` divided by ``n_steps`` (its time mean over
    the window when the sums started at zero)."""
    def loss(carry):
        return carry[1][name] / n_steps
    return loss
