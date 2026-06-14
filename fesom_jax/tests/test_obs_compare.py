"""A2 gate: the differentiable model→obs operator (`obs_compare`) — the keystone.

CPU unit tests (pi mesh + synthetic fields) proving the operator is a *correct differentiable*
operator, per the plan's Testing Strategy:

  * `to_obs` matches an analytic ground truth + a transparent numpy reference regrid (≤1e-10);
  * the vertical interp gradient **flows through the live zstar geometry** (a nonzero FD probe
    that AD reproduces — the review MAJOR-4 requirement);
  * `mld_density_threshold` matches a brute-force crossing search and is AD-finite;
  * empty obs cells (0/0) and masked node lanes stay **finite** forward AND in the gradient.

Token: OBS_OPERATOR_OK. The CORE2 obs-misfit experiments are separate GPU gates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ale, obs_compare
from fesom_jax.mesh import load_mesh
from fesom_jax.obs_compare import ObsGrid

RNG = np.random.default_rng(20260614)


@pytest.fixture(scope="module")
def mesh():
    return load_mesh()


@pytest.fixture(scope="module")
def live_Z(mesh):
    """Live zstar mid-depths from a rest thickness (negative-down, [nod2D, nl])."""
    from fesom_jax.state import State
    st = State.rest(mesh)
    _, Z3d = ale.live_geometry(mesh, st.hnode)
    return Z3d


@pytest.fixture(scope="module")
def grid():
    # a global 4°×4° grid with WOA-like standard depths
    lat = np.arange(-88.0, 89.0, 4.0)
    lon = np.arange(-178.0, 179.0, 4.0)
    depth = np.array([0., 10., 20., 50., 100., 200., 400., 800., 1500., 3000.])
    return ObsGrid(lat=lat, lon=lon, depth=depth)


@pytest.fixture(scope="module")
def hmap(mesh, grid):
    return obs_compare.build_h_map(mesh, grid)


def _in_range_mask(live_Z, hmap):
    """[n_cells, n_depth] bool: obs levels strictly WITHIN each cell's [bottom, top] model
    mid-depth range, where linear interp is exact (above the top layer the operator
    constant-extrapolates; below the bottom it is masked — neither is the analytic value)."""
    cell_depth, valid_ck = obs_compare._cell_means(live_Z, hmap)
    cell_top = jnp.max(jnp.where(valid_ck, cell_depth, -jnp.inf), axis=1)   # shallowest (max)
    cell_bot = jnp.min(jnp.where(valid_ck, cell_depth, jnp.inf), axis=1)    # deepest (min)
    inr = (hmap.obs_z[None, :] <= cell_top[:, None]) & (hmap.obs_z[None, :] >= cell_bot[:, None])
    return np.asarray(inr)


# --------------------------------------------------------------------------
# to_obs — analytic ground truth
# --------------------------------------------------------------------------
def test_to_obs_depth_linear_is_exact(mesh, live_Z, hmap):
    """A field linear in depth, f = a + b·z, interpolates EXACTLY to a + b·obs_z at every
    valid cell (linear interp of a linear function)."""
    a, b = 7.0, 0.013
    field = jnp.where(hmap.node_mask, a + b * live_Z, 0.0)
    cell_obs, valid = obs_compare.to_obs(field, live_Z, hmap)
    expect = a + b * np.asarray(hmap.obs_z)[None, :]
    inr = _in_range_mask(live_Z, hmap) & np.asarray(valid)
    assert inr.any()
    got = np.asarray(cell_obs)
    np.testing.assert_allclose(got[inr], np.broadcast_to(expect, got.shape)[inr],
                               rtol=0, atol=1e-10)


def test_to_obs_horizontal_cell_constant_plus_depth_linear_exact(mesh, live_Z, hmap):
    """f = C[cell(node)] + b·z → cell_obs = C[cell] + b·obs_z exactly (tests the horizontal
    area-mean AND the vertical interp against a known answer)."""
    b = 0.02
    cell_val = jnp.asarray(RNG.standard_normal(hmap.n_cells))
    field = jnp.where(hmap.node_mask,
                      cell_val[hmap.node_cell][:, None] + b * live_Z, 0.0)
    cell_obs, valid = obs_compare.to_obs(field, live_Z, hmap)
    expect = np.asarray(cell_val)[:, None] + b * np.asarray(hmap.obs_z)[None, :]
    inr = _in_range_mask(live_Z, hmap) & np.asarray(valid)
    np.testing.assert_allclose(np.asarray(cell_obs)[inr], expect[inr], rtol=0, atol=1e-10)


def test_to_obs_matches_numpy_reference_regrid(mesh, live_Z, hmap):
    """General random field: the JAX operator matches a transparent numpy reimplementation of
    the same masked area-weighted cell-mean + per-column linear interp (≤1e-10)."""
    field = jnp.asarray(RNG.standard_normal((mesh.nod2D, mesh.nl)))
    field = jnp.where(hmap.node_mask, field, 0.0)
    cell_obs, valid = obs_compare.to_obs(field, live_Z, hmap)

    # numpy reference
    nc, nl = hmap.n_cells, mesh.nl
    node_cell = np.asarray(hmap.node_cell)
    w = np.asarray(hmap.node_area)[:, None] * np.asarray(hmap.node_mask)   # [N, nl]
    f = np.asarray(field); Z = np.asarray(live_Z)
    num_f = np.zeros((nc, nl)); num_z = np.zeros((nc, nl)); den = np.zeros((nc, nl))
    sel = node_cell >= 0
    np.add.at(num_f, node_cell[sel], (w * f)[sel])
    np.add.at(num_z, node_cell[sel], (w * Z)[sel])
    np.add.at(den, node_cell[sel], w[sel])
    valid_ck = den > 0
    cell_f = np.where(valid_ck, num_f / np.where(valid_ck, den, 1.0), 0.0)
    cell_z = np.where(valid_ck, num_z / np.where(valid_ck, den, 1.0),
                      np.asarray(hmap.nominal_z)[None, :])
    obs_z = np.asarray(hmap.obs_z)
    ref = np.zeros((nc, len(obs_z)))
    for c in range(nc):
        if valid_ck[c].any():
            ref[c] = np.interp(obs_z, cell_z[c][::-1], cell_f[c][::-1])
    np.testing.assert_allclose(np.asarray(cell_obs)[np.asarray(valid)],
                               ref[np.asarray(valid)], rtol=0, atol=1e-10)


# --------------------------------------------------------------------------
# the live-geometry gradient path (review MAJOR-4) — THE key differentiability test
# --------------------------------------------------------------------------
def test_z_interp_gradient_flows_through_live_geometry(mesh, hmap):
    """d(to_obs)/d(layer thickness) is finite AND nonzero — the vertical interp is computed
    from the live zstar geometry, so perturbing `hnode` (→ live depths) moves the interp.
    A nonlinear vertical profile is required (a depth-linear field interpolates exactly and
    would be insensitive to the sample depths)."""
    from fesom_jax.state import State
    hnode0 = State.rest(mesh).hnode
    k = jnp.arange(mesh.nl)[None, :]
    # a fixed nodal field with vertical CURVATURE (so the sample-depth axis matters)
    field = jnp.where(hmap.node_mask, 20.0 - 0.3 * k + 0.01 * k * k, 0.0)

    def loss(hnode):
        _, Z3d = ale.live_geometry(mesh, hnode)
        cell_obs, _ = obs_compare.to_obs(field, Z3d, hmap)
        return jnp.sum(cell_obs)

    g = jax.grad(loss)(hnode0)
    g = np.asarray(g)
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)                                     # the gradient path is LIVE

    # AD matches a central finite difference at a wet probe node/level
    wet = np.asarray(mesh.node_layer_mask)
    ni, ki = [(int(n), int(kk)) for n, kk in zip(*np.where(wet))][50]
    h = 1e-3
    hp = hnode0.at[ni, ki].add(h); hm = hnode0.at[ni, ki].add(-h)
    fd = float((loss(hp) - loss(hm)) / (2 * h))
    assert abs(fd - g[ni, ki]) <= 1e-6 * (1 + abs(fd))


# --------------------------------------------------------------------------
# empty cells + masked-NaN finiteness (forward AND gradient)
# --------------------------------------------------------------------------
def test_to_obs_empty_cells_finite_and_masked(mesh, live_Z):
    """A grid finer than the mesh leaves many obs cells with no node → 0 (finite), valid=False,
    and a finite gradient (the 0/0 sentinel-mask case)."""
    fine = ObsGrid(lat=np.arange(-89.5, 90.0, 1.0), lon=np.arange(-179.5, 180.0, 1.0),
                   depth=np.array([0., 50., 200.]))
    hm = obs_compare.build_h_map(mesh, fine)
    field = jnp.where(hm.node_mask, jnp.asarray(RNG.standard_normal((mesh.nod2D, mesh.nl))), 0.0)
    cell_obs, valid = obs_compare.to_obs(field, live_Z, hm)
    assert np.all(np.isfinite(np.asarray(cell_obs)))
    assert (~np.asarray(valid)).any()                          # some empty cells exist
    assert np.all(np.asarray(cell_obs)[~np.asarray(valid)] == 0.0)
    # gradient finite even with empty cells present
    g = jax.grad(lambda f: jnp.sum(obs_compare.to_obs(f, live_Z, hm)[0]))(field)
    assert np.all(np.isfinite(np.asarray(g)))


def test_to_obs_masked_nan_gradient_finite(mesh, live_Z, hmap):
    """Gradient w.r.t. T is finite at EVERY lane, including dry/masked ones (no backward
    0·inf leaking from the masked rows)."""
    T = jnp.where(hmap.node_mask, 10.0 + 0.5 * jnp.arange(mesh.nl)[None, :], 0.0)

    def loss(t):
        cell_obs, valid = obs_compare.to_obs(t, live_Z, hmap)
        return jnp.sum(jnp.where(valid, cell_obs, 0.0) ** 2)

    g = np.asarray(jax.grad(loss)(T))
    assert np.all(np.isfinite(g))
    masked = ~np.asarray(hmap.node_mask)
    assert np.all(g[masked] == 0.0)                            # no gradient leaks to dry lanes


# --------------------------------------------------------------------------
# MLD vs brute force + AD-finite
# --------------------------------------------------------------------------
def _brute_mld(T, S, z, mask, ref_depth=10.0, dsigma=0.03):
    """Transparent per-column brute-force density-threshold MLD on the SAME σ."""
    sig = np.asarray(obs_compare.potential_density(jnp.asarray(T), jnp.asarray(S),
                                                   jnp.asarray(mask)))
    z = np.asarray(z)
    N = T.shape[0]
    out = np.zeros(N); valid = np.zeros(N, bool)
    for n in range(N):
        m = np.asarray(mask)[n]
        if not m[0]:
            continue
        zc = z[n]; sc = sig[n]
        sref = np.interp(-ref_depth, zc[::-1], sc[::-1])
        exc = sc - sref - dsigma
        below = m & (zc <= -ref_depth)
        idx = np.where(below & (exc >= 0))[0]
        if idx.size == 0:
            continue
        kc = idx[0]
        e_hi, e_lo = exc[kc - 1], exc[kc]
        z_hi, z_lo = zc[kc - 1], zc[kc]
        frac = np.clip((0.0 - e_hi) / (e_lo - e_hi), 0.0, 1.0)
        out[n] = -(z_hi + frac * (z_lo - z_hi))
        valid[n] = True
    return out, valid


def test_mld_matches_brute_force():
    """Stratified synthetic columns: vectorized MLD == per-column brute-force crossing."""
    N, nl = 12, 30
    z = -np.cumsum(np.full((N, nl), 10.0), axis=1) + 5.0       # mid-depths -5,-15,... (decreasing)
    mask = np.ones((N, nl), bool)
    # random per-column bottoms (ragged)
    for n in range(N):
        mask[n, RNG.integers(15, nl):] = False
    # T stratified: warm surface, colder deep, with a per-column mixed layer
    T = np.zeros((N, nl)); S = np.full((N, nl), 35.0)
    for n in range(N):
        mld_true = RNG.uniform(30.0, 120.0)
        zc = -z[n]
        T[n] = 18.0 - 6.0 / (1.0 + np.exp(-(zc - mld_true) / 8.0))   # smooth step at mld_true
    T = np.where(mask, T, 0.0)
    mld, valid = obs_compare.mld_density_threshold(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(z), jnp.asarray(mask))
    bmld, bvalid = _brute_mld(T, S, z, mask)
    np.testing.assert_array_equal(np.asarray(valid), bvalid)
    v = bvalid
    np.testing.assert_allclose(np.asarray(mld)[v], bmld[v], rtol=0, atol=1e-9)


def test_mld_gradient_finite_masked_nan(mesh, live_Z):
    """MLD gradient w.r.t. T is finite everywhere (the safe-EOS + guarded crossing)."""
    mask = mesh.node_layer_mask
    k = jnp.arange(mesh.nl)[None, :]
    T = jnp.where(mask, 18.0 - 6.0 / (1.0 + jnp.exp(-(k - 5.0))), 0.0)
    S = jnp.where(mask, 35.0, 0.0)

    def loss(t):
        mld, valid = obs_compare.mld_density_threshold(t, S, live_Z, mask)
        return jnp.sum(jnp.where(valid, mld, 0.0))

    g = np.asarray(jax.grad(loss)(T))
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)


# --------------------------------------------------------------------------
# temporal aggregation + misfit
# --------------------------------------------------------------------------
def test_aggregate_windows_mean_weighted_groups():
    stats = jnp.asarray(RNG.standard_normal((6, 4)))
    np.testing.assert_allclose(np.asarray(obs_compare.aggregate_windows(stats)),
                               np.asarray(stats).mean(0), rtol=0, atol=1e-12)
    w = np.array([1., 2., 3., 4., 5., 6.])
    wm = np.asarray(obs_compare.aggregate_windows(stats, {"weights": w}))
    np.testing.assert_allclose(wm, np.average(np.asarray(stats), axis=0, weights=w),
                               rtol=0, atol=1e-12)
    # groups: 6 windows → 2 seasons (months 0,1,0,1,0,1)
    g = np.array([0, 1, 0, 1, 0, 1])
    gm = np.asarray(obs_compare.aggregate_windows(stats, {"groups": g, "n_groups": 2}))
    assert gm.shape == (2, 4)
    np.testing.assert_allclose(gm[0], np.asarray(stats)[g == 0].mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(gm[1], np.asarray(stats)[g == 1].mean(0), rtol=0, atol=1e-12)


def test_misfit_masked_area_weighted_and_window_aggregated():
    obs = jnp.asarray(RNG.standard_normal((5,)))
    # model with a leading window axis → aggregated to the climatology before compare
    model = obs[None, :] + jnp.asarray(RNG.standard_normal((3, 5))) * 0.0   # mean == obs
    weight = jnp.asarray([1.0, 1.0, 0.0, 2.0, 1.0])                          # one masked-out
    assert float(obs_compare.misfit(model, obs, weight)) == pytest.approx(0.0, abs=1e-12)
    # a known misfit: model = obs + c on the unmasked cells
    model2 = (obs + 0.5)[None, :].repeat(3, axis=0)
    val = float(obs_compare.misfit(model2, obs, weight))
    assert val == pytest.approx(0.25, rel=1e-9)               # (0.5)² weighted-mean
    # all-zero weight → 0, not nan
    assert float(obs_compare.misfit(model2, obs, jnp.zeros(5))) == 0.0


def test_obs_operator_ok_token(mesh, live_Z, hmap):
    """Aggregate gate — prints OBS_OPERATOR_OK for the acceptance log."""
    a, b = 5.0, 0.01
    field = jnp.where(hmap.node_mask, a + b * live_Z, 0.0)
    cell_obs, valid = obs_compare.to_obs(field, live_Z, hmap)
    inr = _in_range_mask(live_Z, hmap) & np.asarray(valid)
    exact = np.allclose(np.asarray(cell_obs)[inr],
                        np.broadcast_to(a + b * np.asarray(hmap.obs_z),
                                        np.asarray(cell_obs).shape)[inr],
                        atol=1e-10)
    g = jax.grad(lambda f: jnp.sum(obs_compare.to_obs(f, live_Z, hmap)[0]))(field)
    finite = bool(np.all(np.isfinite(np.asarray(g))))
    assert exact and finite
    print("OBS_OPERATOR_OK")
