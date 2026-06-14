"""2-D sea-ice concentration → obs operator (Paper-experiments Task A2b — a §0 forward
diagnostic).

Maps the FESOM nodal ice concentration ``a_ice`` ``[nod2D]`` onto a polar-stereographic
satellite grid (NSIDC / OSI-SAF) for a concentration misfit. Unlike the 3-D T/S operator
(:mod:`fesom_jax.obs_compare`) this is a **forward diagnostic only** — it need NOT be
differentiable (no calibration backprops through it), so there is no live-geometry path: ice
concentration is a surface 2-D field, a plain area-weighted node→polar-cell mean.

Two grid specifics it must get right (the plan's A2b requirements):

* **Polar-stereographic projection** — spherical NSIDC convention (true scale at a standard
  parallel, e.g. 70°): each node's geographic (lon, lat) projects to grid (x, y), then bins
  into a regular polar-stereo cell. Only the correct hemisphere is projected (north: lat>0).
* **Pole hole + ice mask** — passive-microwave instruments have a circular gap at the pole
  (poleward of ~87°); those cells carry no obs and are masked in BOTH model and obs for a
  consistent comparison.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import ops
from .config import R_EARTH
from .mesh import Mesh


class PolarGrid(NamedTuple):
    """A regular polar-stereographic grid + its (spherical) projection parameters.

    ``x``/``y`` are 1-D cell-centre coordinates in **projected metres**. ``hemisphere`` is
    ``'north'`` or ``'south'``; ``lat_ts`` the standard parallel (true-scale latitude, signed,
    e.g. ``70`` / ``-70``); ``lon_0`` the central meridian (degrees); ``pole_hole_lat`` the
    latitude poleward of which there is no obs (``None`` ⇒ no hole)."""
    x: np.ndarray
    y: np.ndarray
    hemisphere: str = "north"
    lat_ts: float = 70.0
    lon_0: float = -45.0
    pole_hole_lat: float | None = 87.2
    R: float = R_EARTH


class IceMap(NamedTuple):
    """Host-precomputed node→polar-cell map + masks for the ice diagnostic."""
    node_cell: jax.Array       # [nod2D] int32: flattened polar-cell id (-1 = wrong hemi / outside)
    node_area: jax.Array       # [nod2D] f8: surface CV area (cell-mean weight)
    cell_pole_hole: jax.Array  # [n_cells] bool: inside the pole hole (no obs)
    n_cells: int
    nx: int
    ny: int


def _rho(lat_deg, grid: PolarGrid):
    """Polar-stereographic radius ρ(lat) (spherical NSIDC), true scale at ``grid.lat_ts``.
    ``lat_deg`` is the (signed) geographic latitude; works for both hemispheres via the
    absolute-value reduction to the north formula."""
    R = grid.R
    phi = np.radians(np.abs(np.asarray(lat_deg)))
    t = np.tan(np.pi / 4.0 - phi / 2.0)
    lat_ts = abs(grid.lat_ts)
    if lat_ts >= 90.0:
        return 2.0 * R * t
    m_c = np.cos(np.radians(lat_ts))
    t_c = np.tan(np.pi / 4.0 - np.radians(lat_ts) / 2.0)
    return R * m_c * t / t_c


def _project(lon_deg, lat_deg, grid: PolarGrid):
    """(lon, lat) degrees → polar-stereo (x, y) metres (spherical, NSIDC convention).
    North: ``x=ρ·sin(λ-λ0)``, ``y=-ρ·cos(λ-λ0)``. South mirrors ``y``."""
    rho = _rho(lat_deg, grid)
    dlam = np.radians(np.asarray(lon_deg) - grid.lon_0)
    x = rho * np.sin(dlam)
    if grid.hemisphere == "north":
        y = -rho * np.cos(dlam)
    else:
        y = rho * np.cos(dlam)
    return x, y


def _edges_from_centres(c: np.ndarray) -> np.ndarray:
    d = np.diff(c)
    return np.concatenate([[c[0] - d[0] / 2.0], c[:-1] + d / 2.0, [c[-1] + d[-1] / 2.0]])


def build_ice_map(mesh: Mesh, grid: PolarGrid) -> IceMap:
    """Host-side node→polar-stereo-cell index map + the pole-hole mask.

    Nodes in the correct hemisphere are projected to (x, y), binned into the regular polar
    grid; nodes in the wrong hemisphere or outside the grid extent get ``node_cell = -1``
    (excluded by the :mod:`fesom_jax.ops` negative sentinel). Cells whose centre lies poleward
    of ``grid.pole_hole_lat`` (radius < ρ(pole_hole_lat)) are flagged ``cell_pole_hole``."""
    geo = np.asarray(mesh.geo_coord_nod2D)
    lon = np.degrees(geo[:, 0])
    lat = np.degrees(geo[:, 1])

    xc = np.asarray(grid.x, dtype=np.float64)
    yc = np.asarray(grid.y, dtype=np.float64)
    nx, ny = xc.size, yc.size
    x_e, y_e = _edges_from_centres(xc), _edges_from_centres(yc)

    x, y = _project(lon, lat, grid)
    in_hemi = (lat > 0.0) if grid.hemisphere == "north" else (lat < 0.0)
    inside = (in_hemi & (x >= x_e[0]) & (x <= x_e[-1])
              & (y >= y_e[0]) & (y <= y_e[-1]))
    ix = np.clip(np.digitize(x, x_e) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, y_e) - 1, 0, ny - 1)
    node_cell = np.where(inside, iy * nx + ix, -1).astype(np.int32)

    # pole hole: cell centre radius < ρ(pole_hole_lat)
    n_cells = nx * ny
    if grid.pole_hole_lat is not None:
        rho_hole = float(_rho(grid.pole_hole_lat, grid))
        cx = np.tile(xc, ny)
        cy = np.repeat(yc, nx)
        cell_pole_hole = (np.sqrt(cx * cx + cy * cy) < rho_hole)
    else:
        cell_pole_hole = np.zeros(n_cells, dtype=bool)

    return IceMap(
        node_cell=jnp.asarray(node_cell),
        node_area=jnp.asarray(np.asarray(mesh.area[:, 0], dtype=np.float64)),
        cell_pole_hole=jnp.asarray(cell_pole_hole),
        n_cells=int(n_cells), nx=int(nx), ny=int(ny),
    )


def to_cells(node_field, ice_map: IceMap):
    """Area-weighted node→polar-cell mean of a 2-D surface field ``[nod2D]`` →
    ``(cell[n_cells], cell_valid[n_cells])``. Empty cells (no node) → 0, ``valid=False``."""
    w = ice_map.node_area
    num = ops.scatter_add(w * node_field, ice_map.node_cell, ice_map.n_cells)
    den = ops.scatter_add(w, ice_map.node_cell, ice_map.n_cells)
    valid = den > 0.0
    return jnp.where(valid, num / jnp.where(valid, den, 1.0), 0.0), valid


def ice_conc_misfit(a_ice, obs_conc, ice_map: IceMap, *, obs_valid=None):
    """Area-weighted mean-squared concentration misfit (model − obs) over valid polar cells.

    The model concentration is binned to cells (:func:`to_cells`); the comparison mask drops
    (a) empty cells, (b) **pole-hole** cells, and (c) any cell with no obs (``obs_valid``).
    ``obs_conc`` is the satellite concentration on the SAME flattened polar grid. The cell area
    weight uses ``cos(lat)``-equivalent equal-area polar cells ⇒ a plain count here (polar-stereo
    cells are ~equal area); returns the scalar weighted MSE. Forward-only (no AD contract)."""
    cell_model, cell_valid = to_cells(a_ice, ice_map)
    mask = cell_valid & (~ice_map.cell_pole_hole)
    if obs_valid is not None:
        mask = mask & jnp.asarray(obs_valid)
    w = mask.astype(cell_model.dtype)
    diff = cell_model - jnp.asarray(obs_conc)
    num = jnp.sum(w * diff * diff)
    den = jnp.sum(w)
    return num / jnp.where(den > 0.0, den, 1.0)
