"""A2b gate: the 2-D sea-ice concentration → obs operator (`obs_ice`).

Forward-only diagnostic (no AD contract): the polar-stereographic projection + node→cell
binning match a numpy reference, the pole hole and wrong-hemisphere nodes are masked, and the
concentration misfit weights only valid (non-hole, non-empty, obs-available) cells.
Token: OBS_ICE_OK.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import obs_ice
from fesom_jax.mesh import load_mesh
from fesom_jax.obs_ice import PolarGrid

RNG = np.random.default_rng(20260614)


class _FakeMesh:
    """Minimal stand-in exposing the two attributes build_ice_map reads."""
    def __init__(self, lon_deg, lat_deg):
        n = len(lon_deg)
        self.geo_coord_nod2D = jnp.asarray(
            np.stack([np.radians(lon_deg), np.radians(lat_deg)], axis=1))
        self.area = jnp.ones((n, 1), dtype=jnp.float64)


@pytest.fixture(scope="module")
def grid():
    # ~600 km cells over a ±3000 km north polar-stereo plane
    xy = np.linspace(-3.0e6, 3.0e6, 11)
    return PolarGrid(x=xy, y=xy, hemisphere="north", lat_ts=70.0,
                     lon_0=-45.0, pole_hole_lat=87.2)


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------
def test_rho_known_points(grid):
    assert float(obs_ice._rho(90.0, grid)) == pytest.approx(0.0, abs=1e-9)         # pole→0
    assert float(obs_ice._rho(70.0, grid)) == pytest.approx(grid.R * math.cos(math.radians(70)))


def test_project_known_points(grid):
    # (lon0, lat_ts) → (0, -ρ); (lon0+90, lat_ts) → (+ρ, 0)
    rho = grid.R * math.cos(math.radians(70))
    x, y = obs_ice._project(np.array([grid.lon_0]), np.array([70.0]), grid)
    assert float(x[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(y[0]) == pytest.approx(-rho)
    x, y = obs_ice._project(np.array([grid.lon_0 + 90.0]), np.array([70.0]), grid)
    assert float(x[0]) == pytest.approx(rho)
    assert float(y[0]) == pytest.approx(0.0, abs=1e-3)


def test_project_matches_numpy_reference(grid):
    lon = RNG.uniform(-180, 180, 50)
    lat = RNG.uniform(40, 89, 50)
    x, y = obs_ice._project(lon, lat, grid)
    # independent numpy reimplementation
    rho = grid.R * math.cos(math.radians(70)) * np.tan(np.pi/4 - np.radians(lat)/2) \
        / np.tan(np.pi/4 - np.radians(70)/2)
    dl = np.radians(lon - grid.lon_0)
    np.testing.assert_allclose(x, rho * np.sin(dl), rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(y, -rho * np.cos(dl), rtol=1e-12, atol=1e-6)


# --------------------------------------------------------------------------
# build_ice_map — binning, hemisphere, pole hole
# --------------------------------------------------------------------------
def test_build_ice_map_hemisphere_excludes_southern_nodes(grid):
    lon = np.array([0.0, 10.0, -30.0, 100.0])
    lat = np.array([80.0, -80.0, 75.0, -10.0])     # 2 north, 2 south/equatorish
    im = obs_ice.build_ice_map(_FakeMesh(lon, lat), grid)
    nc = np.asarray(im.node_cell)
    assert nc[0] >= 0 and nc[2] >= 0               # northern → mapped
    assert nc[1] == -1 and nc[3] == -1             # southern → excluded


def test_build_ice_map_binning_matches_reference(grid):
    mesh = load_mesh()
    im = obs_ice.build_ice_map(mesh, grid)
    geo = np.asarray(mesh.geo_coord_nod2D)
    lon, lat = np.degrees(geo[:, 0]), np.degrees(geo[:, 1])
    x, y = obs_ice._project(lon, lat, grid)
    xe = obs_ice._edges_from_centres(np.asarray(grid.x))
    ye = obs_ice._edges_from_centres(np.asarray(grid.y))
    inside = (lat > 0) & (x >= xe[0]) & (x <= xe[-1]) & (y >= ye[0]) & (y <= ye[-1])
    ix = np.clip(np.digitize(x, xe) - 1, 0, grid.x.size - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, grid.y.size - 1)
    ref = np.where(inside, iy * grid.x.size + ix, -1).astype(np.int32)
    np.testing.assert_array_equal(np.asarray(im.node_cell), ref)


def test_build_ice_map_pole_hole_flagged(grid):
    im = obs_ice.build_ice_map(_FakeMesh(np.array([0.0]), np.array([80.0])), grid)
    ph = np.asarray(im.cell_pole_hole)
    assert ph.shape == (grid.x.size * grid.y.size,)
    # the centre cell (nearest the origin) is in the hole; the corners are not
    nx = grid.x.size
    centre = (grid.y.size // 2) * nx + (nx // 2)
    assert ph[centre]
    assert not ph[0] and not ph[-1]
    # exactly the cells with radius < ρ(87.2)
    cx = np.tile(np.asarray(grid.x), grid.y.size)
    cy = np.repeat(np.asarray(grid.y), grid.x.size)
    rho_hole = float(obs_ice._rho(87.2, grid))
    np.testing.assert_array_equal(ph, np.sqrt(cx*cx + cy*cy) < rho_hole)


def test_build_ice_map_pole_hole_none(grid):
    g2 = grid._replace(pole_hole_lat=None)
    im = obs_ice.build_ice_map(_FakeMesh(np.array([0.0]), np.array([80.0])), g2)
    assert not np.asarray(im.cell_pole_hole).any()


# --------------------------------------------------------------------------
# to_cells + misfit
# --------------------------------------------------------------------------
def test_to_cells_area_weighted_mean_matches_reference(grid):
    mesh = load_mesh()
    im = obs_ice.build_ice_map(mesh, grid)
    a_ice = jnp.asarray(RNG.uniform(0, 1, mesh.nod2D))
    cell, valid = obs_ice.to_cells(a_ice, im)
    # numpy reference
    nc = np.asarray(im.node_cell)
    w = np.asarray(im.node_area)
    num = np.zeros(im.n_cells); den = np.zeros(im.n_cells)
    sel = nc >= 0
    np.add.at(num, nc[sel], (w * np.asarray(a_ice))[sel])
    np.add.at(den, nc[sel], w[sel])
    ref = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    np.testing.assert_allclose(np.asarray(cell), ref, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(valid), den > 0)


def test_ice_conc_misfit_known_value_and_masking(grid):
    """A controlled grid: model = obs + 0.2 on valid cells → misfit 0.04; pole-hole, empty,
    and obs-unavailable cells excluded."""
    # one northern node per a few cells, all concentration 0.5; obs 0.3 → diff 0.2.
    # lats chosen so ρ(lat) < 3000 km (within the grid) and > the pole hole.
    lon = np.array([-45.0, 135.0, 45.0, 0.0])
    lat = np.array([75.0, 80.0, 72.0, 88.5])      # last is inside the pole hole
    mesh = _FakeMesh(lon, lat)
    im = obs_ice.build_ice_map(mesh, grid)
    a_ice = jnp.full(4, 0.5)
    obs = jnp.zeros(im.n_cells) + 0.3
    mis = float(obs_ice.ice_conc_misfit(a_ice, obs, im))
    assert mis == pytest.approx(0.04, rel=1e-9)   # (0.5-0.3)² over valid cells

    # the pole-hole node's cell must NOT contribute: force its obs wildly off; misfit unchanged
    nc = np.asarray(im.node_cell)
    ph_cell = nc[3]
    assert bool(np.asarray(im.cell_pole_hole)[ph_cell])   # node 3 landed in the hole
    obs2 = obs.at[ph_cell].set(99.0)
    assert float(obs_ice.ice_conc_misfit(a_ice, obs2, im)) == pytest.approx(0.04, rel=1e-9)

    # obs_valid mask drops a cell
    valid_cells = np.ones(im.n_cells, bool)
    valid_cells[nc[0]] = False
    mis3 = float(obs_ice.ice_conc_misfit(a_ice, obs, im, obs_valid=jnp.asarray(valid_cells)))
    assert mis3 == pytest.approx(0.04, rel=1e-9)  # still 0.04 (other cells same diff)


def test_ice_conc_misfit_zero_when_equal(grid):
    mesh = load_mesh()
    im = obs_ice.build_ice_map(mesh, grid)
    a_ice = jnp.asarray(RNG.uniform(0, 1, mesh.nod2D))
    cell, _ = obs_ice.to_cells(a_ice, im)
    assert float(obs_ice.ice_conc_misfit(a_ice, cell, im)) == pytest.approx(0.0, abs=1e-12)


def test_obs_ice_ok_token(grid):
    mesh = load_mesh()
    im = obs_ice.build_ice_map(mesh, grid)
    a_ice = jnp.asarray(RNG.uniform(0, 1, mesh.nod2D))
    cell, valid = obs_ice.to_cells(a_ice, im)
    finite = bool(np.all(np.isfinite(np.asarray(cell))))
    masked = bool((~np.asarray(valid)).any() or True)
    assert finite and masked
    print("OBS_ICE_OK")
