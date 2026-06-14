"""Differentiable model→observation operator: 3-D T/S regrid + AD-safe MLD + temporal
aggregation (Paper-experiments Task A2 — the keystone of the obs-application half of all
three pillars).

The operator maps a FESOM nodal field ``[nod2D, nl]`` onto a regular lat/lon×depth obs grid
(WOA / EN4 / de Boyer Montégut style) so a misfit-to-obs loss can be differentiated end to
end. Two design rules make it a *correct* differentiable operator, not a post-hoc diagnostic:

1. **Only the HORIZONTAL node→cell map is host-precomputed** (:func:`build_h_map`) — nodes are
   horizontally static. The **vertical interpolation is recomputed inside** :func:`to_obs`
   **from the LIVE zstar geometry** (the per-node mid-depths ``Z_3d_n`` from
   :func:`fesom_jax.ale.live_geometry`), so ``d(misfit)/d(layer thickness)`` flows through the
   moving coordinate (verified by a nonzero FD probe in the tests). Pre-baking vertical
   weights would silently cut that gradient path (review MAJOR-4).
2. **AD masked-NaN rule** (the project-wide contract): every masked / empty lane computes a
   **finite** value. Empty obs cells (0/0 ``segment_mean``) use the
   :mod:`fesom_jax.ops` sentinel-mask precedent; the vertical interp guards its denominators;
   the MLD crossing is a **linear interpolation, never an argmax**, with guarded weights. A
   forward ``where`` alone does NOT stop a backward ``0·inf`` — the masking is built into the
   arithmetic.

Temporal aggregation is **first-class** (:func:`aggregate_windows`): the loss targets a
**climatological mean** (average over start-dates / seasons / years), not one weather
realization (review MAJOR-3). The SAME aggregation is used in the loss and in evaluation.

All depths are carried **negative-down** (matching ``mesh.Z`` ≤ 0); obs depths supplied as
positive metres are negated on ingest. Float64 throughout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import eos, ops
from .mesh import Mesh


# ==========================================================================
# Grid + host-precomputed horizontal map
# ==========================================================================
class ObsGrid(NamedTuple):
    """A regular observational grid: 1-D centre coordinates. ``lat``/``lon`` in **degrees**,
    ``depth`` in **positive metres** (negated to negative-down internally). Regular spacing is
    assumed (edges derived from centres)."""
    lat: np.ndarray        # [n_lat] degrees, ascending
    lon: np.ndarray        # [n_lon] degrees
    depth: np.ndarray      # [n_depth] positive metres, ascending


class Hmap(NamedTuple):
    """Host-precomputed horizontal node→obs-cell map + the static weights/masks the operator
    needs. Carries **no vertical weights** — those are recomputed live in :func:`to_obs`.

    Integer counts are Python ints (static under ``jit`` when ``Hmap`` is closed over by the
    loss); array fields are device constants."""
    node_cell: jax.Array   # [nod2D] int32: flattened obs-cell id per node (-1 = outside grid)
    node_area: jax.Array   # [nod2D] f8: surface CV area, the cell-mean weight
    node_mask: jax.Array   # [nod2D, nl] bool: valid tracer layers (static ragged-vertical)
    obs_z: jax.Array       # [n_depth] f8: obs depths, negative-down
    nominal_z: jax.Array   # [nl] f8: nominal mid-depths (negative-down, strictly decreasing) —
                           #   the AD-safe fill for below-bottom levels so the interp axis stays monotone
    cell_area: jax.Array   # [n_cells] f8: cos(lat) area weight per obs cell (misfit weighting)
    n_cells: int           # n_lat*n_lon
    n_lat: int
    n_lon: int


def _edges_from_centres(c: np.ndarray) -> np.ndarray:
    """Cell edges for regularly(-ish) spaced centres ``c`` (length n → n+1 edges)."""
    d = np.diff(c)
    return np.concatenate([[c[0] - d[0] / 2.0], c[:-1] + d / 2.0, [c[-1] + d[-1] / 2.0]])


def build_h_map(mesh: Mesh, obs_grid: ObsGrid) -> Hmap:
    """Host-side (numpy) precompute of the **horizontal** node→obs-cell index map only.

    Each node's geographic (lon, lat) (``mesh.geo_coord_nod2D``, radians) is binned into the
    obs lat/lon grid; ``node_cell = lat_idx*n_lon + lon_idx`` (flattened), with ``-1`` for a
    node outside the grid's latitude band (excluded from the ``segment_mean`` via the
    :mod:`fesom_jax.ops` negative-sentinel mask). Longitudes are wrapped into the grid's
    longitude range so a [-180,180) mesh and a [0,360) obs grid (or vice-versa) align.

    ⚠️ Deliberately precomputes **no vertical weights** — :func:`to_obs` builds those from the
    live geometry each call (the differentiability requirement)."""
    geo = np.asarray(mesh.geo_coord_nod2D)
    lon = np.degrees(geo[:, 0])
    lat = np.degrees(geo[:, 1])

    lat_c = np.asarray(obs_grid.lat, dtype=np.float64)
    lon_c = np.asarray(obs_grid.lon, dtype=np.float64)
    dep = np.asarray(obs_grid.depth, dtype=np.float64)
    n_lat, n_lon = lat_c.size, lon_c.size

    lat_e = _edges_from_centres(lat_c)
    lon_e = _edges_from_centres(lon_c)
    # wrap node longitude into [lon_e[0], lon_e[0]+360)
    lon = ((lon - lon_e[0]) % 360.0) + lon_e[0]

    ilat = np.clip(np.digitize(lat, lat_e) - 1, 0, n_lat - 1)
    ilon = np.clip(np.digitize(lon, lon_e) - 1, 0, n_lon - 1)
    inside = (lat >= lat_e[0]) & (lat <= lat_e[-1])
    node_cell = np.where(inside, ilat * n_lon + ilon, -1).astype(np.int32)

    # nominal mid-depths (negative-down), padded to nl and made STRICTLY decreasing so the
    # below-bottom fill never ties the interp axis (the deepest pad sits 1 m below the last layer).
    Z = np.asarray(mesh.Z, dtype=np.float64)                       # [nl-1] ≤ 0
    nominal_z = np.concatenate([Z, [Z[-1] - 1.0]])                 # [nl]

    cell_lat = np.repeat(lat_c, n_lon)                             # [n_cells]
    cell_area = np.cos(np.radians(cell_lat))

    return Hmap(
        node_cell=jnp.asarray(node_cell),
        node_area=jnp.asarray(np.asarray(mesh.area[:, 0], dtype=np.float64)),
        node_mask=jnp.asarray(np.asarray(mesh.node_layer_mask)),
        obs_z=jnp.asarray(-np.abs(dep)),                           # positive-down → negative-down
        nominal_z=jnp.asarray(nominal_z),
        cell_area=jnp.asarray(cell_area),
        n_cells=int(n_lat * n_lon),
        n_lat=int(n_lat),
        n_lon=int(n_lon),
    )


# ==========================================================================
# AD-safe vectorized vertical interpolation (the live-geometry path)
# ==========================================================================
def interp_columns(z, f, targets):
    """Piecewise-linear interp of ``f`` (defined at depths ``z``) onto ``targets``, vectorized
    over columns and AD-safe through **both** ``z`` (the live geometry) and ``f``.

    ``z`` : ``[C, nl]`` strictly **decreasing** depths (negative-down). ``f`` : ``[C, nl]``.
    ``targets`` : ``[n_t]`` depths (negative-down). Returns ``[C, n_t]``.

    The bracket index is found by counting levels at-or-above the target (a non-differentiable
    step — correctly so; the bracket is locally constant); the interpolation **weights** are
    differentiable w.r.t. the gathered ``z``/``f``. The denominator is guarded (strictly
    decreasing ⇒ nonzero) and the weight clipped to [0,1] so targets above the shallowest /
    below the deepest level get constant extrapolation (the deep side is masked out by the
    caller). No ``nan``/``inf`` reaches the backward pass."""
    nl = z.shape[1]
    ge = z[:, :, None] >= targets[None, None, :]               # [C, nl, n_t]
    iu = jnp.clip(jnp.sum(ge, axis=1) - 1, 0, nl - 2)          # [C, n_t] upper (shallower) idx
    il = iu + 1
    z_hi = jnp.take_along_axis(z, iu, axis=1)
    z_lo = jnp.take_along_axis(z, il, axis=1)
    f_hi = jnp.take_along_axis(f, iu, axis=1)
    f_lo = jnp.take_along_axis(f, il, axis=1)
    denom = z_hi - z_lo
    denom = jnp.where(denom != 0.0, denom, 1.0)
    w = jnp.clip((targets[None, :] - z_lo) / denom, 0.0, 1.0)  # weight on the shallow value
    return w * f_hi + (1.0 - w) * f_lo


def _cell_means(node_field, Hmap: Hmap):
    """Masked, area-weighted node→cell ``segment_mean`` per level: returns
    ``(cell_field[n_cells, nl], cell_valid[n_cells, nl])``. Empty (cell, level) entries
    (no wet node) → 0 (finite) with ``cell_valid=False`` (the 0/0 sentinel-mask case)."""
    w = Hmap.node_area[:, None] * Hmap.node_mask                # [nod2D, nl] area×validity
    num = ops.scatter_add(w * node_field, Hmap.node_cell, Hmap.n_cells)   # [n_cells, nl]
    den = ops.scatter_add(w, Hmap.node_cell, Hmap.n_cells)
    valid = den > 0.0
    cell = jnp.where(valid, num / jnp.where(valid, den, 1.0), 0.0)
    return cell, valid


def to_obs(field3d, Z_3d_n, Hmap: Hmap):
    """Map a nodal layer field ``[nod2D, nl]`` onto the obs grid → ``(cell_obs[n_cells,
    n_depth], obs_valid[n_cells, n_depth])``.

    Horizontal masked area-weighted node→cell mean (per level) of **both** the field and the
    **live** mid-depths ``Z_3d_n``, then a per-cell vertical interp from the (live) cell depth
    axis onto ``Hmap.obs_z``. The live depths enter the cell depth axis ⇒ the vertical
    interpolation weights are differentiable w.r.t. the layer thickness (the zstar gradient
    path). Below-bottom levels are filled with ``Hmap.nominal_z`` so the interp axis stays
    monotone; obs depths below a cell's true bottom are flagged invalid (and zeroed).

    ``Z_3d_n`` is ``ale.live_geometry(mesh, state.hnode)[1]`` (``[nod2D, nl]``, negative-down).

    Obs depths **above** a cell's shallowest layer centre (e.g. a 0 m surface obs level vs a
    ~2.5 m top mid-depth) get **constant extrapolation** from the top layer (the standard
    model↔surface-obs convention — the operator does not invent data above the top layer);
    obs depths below the cell bottom are flagged invalid. Linear interp is exact strictly
    within ``[cell_bottom, cell_top]``.
    """
    cell_field, valid_ck = _cell_means(field3d, Hmap)
    cell_depth_raw, _ = _cell_means(Z_3d_n, Hmap)
    # fill invalid (below-bottom) levels with the nominal monotone axis
    cell_depth = jnp.where(valid_ck, cell_depth_raw, Hmap.nominal_z[None, :])

    cell_obs = interp_columns(cell_depth, cell_field, Hmap.obs_z)      # [n_cells, n_depth]

    # per-cell bottom (deepest valid mid-depth, most negative) for the output mask
    cell_bottom = jnp.min(jnp.where(valid_ck, cell_depth, jnp.inf), axis=1)   # [n_cells]
    has_col = jnp.any(valid_ck, axis=1)[:, None]
    obs_valid = has_col & (Hmap.obs_z[None, :] >= cell_bottom[:, None])
    cell_obs = jnp.where(obs_valid, cell_obs, 0.0)
    return cell_obs, obs_valid


def to_obs_surface(field2d, Hmap: Hmap):
    """Horizontal-only map of a 2-D surface field ``[nod2D]`` (e.g. SST) → ``(cell[n_cells],
    valid[n_cells])``. The surface-metric special case of :func:`to_obs` (no vertical interp)."""
    w = Hmap.node_area * Hmap.node_mask[:, 0]
    num = ops.scatter_add(w * field2d, Hmap.node_cell, Hmap.n_cells)
    den = ops.scatter_add(w, Hmap.node_cell, Hmap.n_cells)
    valid = den > 0.0
    return jnp.where(valid, num / jnp.where(valid, den, 1.0), 0.0), valid


# ==========================================================================
# AD-safe mixed-layer depth (density threshold, de Boyer Montégut)
# ==========================================================================
def potential_density(T, S, node_mask):
    """Potential density σ (referenced to the surface), ``[nod2D, nl]``, masked. Uses the
    JM-EOS ``rhopot`` component (:func:`fesom_jax.eos.jm_components`). Dry lanes are fed
    **safe** ``T``/``S`` (the EOS ``sqrt(S)`` has an infinite grad at ``S=0`` — a forward mask
    does not stop the backward ``0·inf``), then masked to 0."""
    S_safe = jnp.where(node_mask, S, 35.0)
    T_safe = jnp.where(node_mask, T, 0.0)
    _, _, _, rhopot = eos.jm_components(T_safe, S_safe)
    return jnp.where(node_mask, rhopot, 0.0)


def mld_density_threshold(T, S, Z_3d_n, node_mask, *, ref_depth: float = 10.0,
                          dsigma: float = 0.03):
    """AD-safe mixed-layer depth ``[nod2D]`` by the **density-threshold** criterion (de Boyer
    Montégut: the shallowest depth below ``ref_depth`` where σ exceeds its ``ref_depth`` value
    by ``dsigma`` kg/m³). Returns ``(mld[nod2D]`` positive metres``, valid[nod2D])``.

    The crossing depth is a **linear interpolation** between the bracketing levels (NOT an
    argmax of density — that would be a hard, non-differentiable pick): the bracket index is
    located by a step search, but the returned depth interpolates the σ-excess crossing of 0,
    so ``d(MLD)/d(T,S)`` and ``d(MLD)/d(geometry)`` flow. Fully-mixed columns (no crossing) →
    ``valid=False``. Live ``Z_3d_n`` (negative-down) makes it geometry-differentiable."""
    nl = T.shape[1]
    sigma = potential_density(T, S, node_mask)                 # [N, nl]
    z = Z_3d_n                                                  # [N, nl] negative-down

    # σ at the reference depth (live interp), then the excess over (σ_ref + dsigma)
    sigma_ref = interp_columns(z, sigma, jnp.asarray([-abs(ref_depth)]))[:, 0]   # [N]
    excess = sigma - sigma_ref[:, None] - dsigma               # [N, nl]; MLD where excess→0⁺

    # search for the shallowest crossing strictly BELOW ref_depth among wet levels
    below_ref = node_mask & (z <= -abs(ref_depth))
    crossed = below_ref & (excess >= 0.0)                      # [N, nl]
    any_cross = jnp.any(crossed, axis=1)                       # [N]
    kc = jnp.clip(jnp.argmax(crossed, axis=1), 1, nl - 1)      # first True level (shallowest)

    # linear crossing depth between kc-1 (excess<0) and kc (excess≥0) on the RAW excess
    e_hi = jnp.take_along_axis(excess, (kc - 1)[:, None], axis=1)[:, 0]
    e_lo = jnp.take_along_axis(excess, kc[:, None], axis=1)[:, 0]
    z_hi = jnp.take_along_axis(z, (kc - 1)[:, None], axis=1)[:, 0]
    z_lo = jnp.take_along_axis(z, kc[:, None], axis=1)[:, 0]
    denom = e_lo - e_hi
    denom = jnp.where(denom != 0.0, denom, 1.0)
    frac = jnp.clip((0.0 - e_hi) / denom, 0.0, 1.0)
    mld_z = z_hi + frac * (z_lo - z_hi)                        # crossing depth (negative)

    valid = node_mask[:, 0] & any_cross
    return jnp.where(valid, -mld_z, 0.0), valid


# ==========================================================================
# Temporal aggregation + misfit
# ==========================================================================
def aggregate_windows(model_stats, spec: dict | None = None):
    """Aggregate a stack of per-window model statistics ``[n_windows, *rest]`` to the matched
    target statistic — **first-class temporal aggregation** so the loss targets a
    climatological mean, not one realization.

    ``spec`` :
      * ``None`` → simple mean over windows → ``[*rest]``.
      * ``{'weights': w[n_windows]}`` → normalized weighted mean → ``[*rest]``.
      * ``{'groups': g[n_windows], 'n_groups': G}`` → per-group (e.g. per-season / per-month)
        mean over windows → ``[G, *rest]``.
    Differentiable (pure averaging)."""
    stats = jnp.asarray(model_stats)
    if spec is None:
        return jnp.mean(stats, axis=0)
    if "weights" in spec:
        w = jnp.asarray(spec["weights"], stats.dtype)
        w = w / jnp.sum(w)
        return jnp.tensordot(w, stats, axes=([0], [0]))
    if "groups" in spec:
        g = jnp.asarray(spec["groups"])
        G = int(spec["n_groups"])
        sums = ops.scatter_add(stats, g, G)                    # [G, *rest]
        cnts = ops.scatter_add(jnp.ones((stats.shape[0],), stats.dtype), g, G)
        shape = (G,) + (1,) * (stats.ndim - 1)
        cnts = jnp.where(cnts > 0, cnts, 1.0).reshape(shape)
        return sums / cnts
    return jnp.mean(stats, axis=0)


def misfit(model, obs, weight, *, spec: dict | None = None):
    """Masked, area-weighted mean-squared model−obs misfit, through :func:`aggregate_windows`.

    If ``model`` has one more leading axis than ``obs`` it is first aggregated over windows
    (``spec``) to the climatology, then compared. ``weight`` is the combined non-negative
    weight (obs-cell area × model/obs validity × a **consistent ice mask** for surface metrics
    — drop seasonally ice-covered cells); it broadcasts against the difference. Returns the
    weighted-mean squared error (a scalar). Guarded so an all-zero weight → 0, not nan."""
    m = jnp.asarray(model)
    o = jnp.asarray(obs)
    if m.ndim == o.ndim + 1:
        m = aggregate_windows(m, spec)
    diff = m - o
    w = jnp.asarray(weight)
    num = jnp.sum(w * diff * diff)
    den = jnp.sum(w)
    return num / jnp.where(den > 0.0, den, 1.0)
