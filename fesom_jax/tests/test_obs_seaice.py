"""``fesom_jax.obs_seaice`` on synthetic grids and nodes (no mesh, no observation files):
nearest-node assignment and its cap, the OSI-SAF bit-mask decisions, percent -> fraction, the
half-of-available-days rule, CCI flag 0 only, effective thickness, sigma floors, the observed-area
weight and the hemisphere split."""

from __future__ import annotations

import numpy as np
import pytest

from fesom_jax import obs_seaice as O


def _gc_km(lon1, lat1, lon2, lat2):
    """Haversine great-circle distance [km] on the model sphere (independent of the KD-tree)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * O.R_EARTH / 1000.0


# --- assignment --------------------------------------------------------------------------------
def test_nearest_assignment_matches_brute_force_and_respects_cap_and_eligibility():
    rng = np.random.default_rng(3)
    # nodes: a jittered Arctic cloud, with ineligible (dry / other hemisphere) nodes interleaved,
    # so an index slip between the eligible subset and the global numbering cannot pass
    n = 400
    node_lon = rng.uniform(-180, 180, n)
    node_lat = rng.uniform(70, 89, n)
    node_lat[::7] = -75.0                       # southern nodes: never eligible for this grid
    ok = node_lat > 0
    ok[3::11] = False                           # dry nodes
    cell_lon = rng.uniform(-180, 180, 300)
    cell_lat = rng.uniform(70, 89.5, 300)
    cap = 40.0
    node, dist = O.assign_cells(cell_lon, cell_lat, node_lon, node_lat, ok, cap)

    eligible = np.flatnonzero(ok)
    d_all = _gc_km(cell_lon[:, None], cell_lat[:, None], node_lon[None, eligible],
                   node_lat[None, eligible])
    best = eligible[np.argmin(d_all, axis=1)]
    dmin = d_all.min(axis=1)
    np.testing.assert_allclose(dist, dmin, rtol=1e-9, atol=1e-6)
    within = dmin <= cap
    assert within.any() and (~within).any(), "the synthetic case must exercise both sides of the cap"
    np.testing.assert_array_equal(node[within], best[within])
    np.testing.assert_array_equal(node[~within], -1)
    assert ok[node[within]].all()


def test_assignment_cap_boundary_and_exact_positions():
    # nodes on the 80N parallel every 1 deg of longitude; one dry node in between
    node_lon = np.array([0.0, 1.0, 2.0, 3.0])
    node_lat = np.full(4, 80.0)
    ok = np.array([True, False, True, True])
    # a cell exactly on node 2, one next to the dry node 1 (goes to 0 or 2, never 1), one far away
    cell_lon = np.array([2.0, 0.9, 40.0])
    cell_lat = np.array([80.0, 80.0, 80.0])
    node, dist = O.assign_cells(cell_lon, cell_lat, node_lon, node_lat, ok, 30.0)
    assert node[0] == 2 and dist[0] == pytest.approx(0.0, abs=1e-6)
    assert node[1] == 0                          # 0.9 deg from node 0, 1.1 deg from node 2
    assert node[2] == -1
    # the cap is inclusive: a cap equal to the distance keeps the cell
    d = _gc_km(0.9, 80.0, 0.0, 80.0)
    node2, _ = O.assign_cells(cell_lon[1:2], cell_lat[1:2], node_lon, node_lat, ok, d + 1e-7)
    node3, _ = O.assign_cells(cell_lon[1:2], cell_lat[1:2], node_lon, node_lat, ok, d - 1e-6)
    assert node2[0] == 0 and node3[0] == -1


def test_no_eligible_node_raises():
    with pytest.raises(ValueError, match="no eligible node"):
        O.assign_cells([0.0], [80.0], [0.0], [80.0], [False], 30.0)


# --- OSI-SAF flags and units ----------------------------------------------------------------------
B = O.OSI_BITS


def test_flag_decoding_drops_land_lake_fill_spillover_hight2m_polehole_and_keeps_valid_zeros():
    flags = np.array([0, B["land"], B["lake"], B["open_water_filtered"], B["land_spill_over"],
                      B["high_t2m"], B["spatial_interp"], B["temporal_interp"], B["max_ice_climo"],
                      B["high_t2m"] | B["temporal_interp"],          # 80: high_t2m -> dropped
                      B["lake"] | B["open_water_filtered"],          # 6: lake wins -> dropped
                      B["land_spill_over"] | B["high_t2m"],          # 24: dropped
                      B["open_water_filtered"] | B["max_ice_climo"],  # kept
                      0, O.OSI_FLAG_FILL], dtype=np.int16)
    conc = np.array([5000, O.OSI_FILL, 7500, 0, 4940, 0, 10000, 9000, 0, 1200, 0, 300, 0,
                     O.OSI_FILL, 0], dtype=np.int32)
    want = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    np.testing.assert_array_equal(O.osisaf_valid(conc, flags), want)
    assert O.OSI_DROP_MASK == 1 + 2 + 8 + 16 + 32
    # high_t2m is dropped whatever its value, alone or combined with a kept bit
    t2m = np.array([B["high_t2m"], B["high_t2m"] | B["temporal_interp"],
                    B["high_t2m"] | B["open_water_filtered"], B["high_t2m"] | B["max_ice_climo"]])
    assert not O.osisaf_valid(np.array([0, 5000, 0, 10000]), t2m).any()
    # the kept zero-concentration flags yield zeros, not missing values
    frac = O.osisaf_fraction(conc)
    for i in (3, 8, 12):
        assert frac[i] == 0.0
    c = O.osisaf_flag_counts(conc, flags)
    assert c["land"] == 1 and c["lake"] == 2 and c["high_t2m"] == 3 and c["max_ice_climo"] == 2
    assert c["nominal"] == 2 and c["conc_fill"] == 2 and c["flag_fill"] == 1
    assert c["kept"] == int(want.sum()) and c["kept"] + c["dropped"] == flags.size


def test_percent_to_fraction():
    raw = np.array([10000, 5000, 1, 0, O.OSI_FILL])
    f = O.osisaf_fraction(raw)
    np.testing.assert_allclose(f[:4], [1.0, 0.5, 1e-4, 0.0], rtol=0, atol=1e-15)
    assert np.isnan(f[4])


# --- monthly means ----------------------------------------------------------------------------
def _month(n_files, valid_days, value=0.4, sigma=0.05):
    acc = O.MonthlyCellMean(1)
    for d in range(n_files):
        acc.add([value + 0.01 * d], [d < valid_days], [sigma])
    return acc.finalize(n_files)


@pytest.mark.parametrize("n_files", [31, 30, 15, 16])     # 15/16: every-other-day months
def test_half_of_available_days_rule(n_files):
    half = (n_files + 1) // 2
    mean, sig, nd = _month(n_files, half)
    assert np.isfinite(mean[0]) and nd[0] == half
    np.testing.assert_allclose(mean[0], 0.4 + 0.01 * np.arange(half).mean())
    mean, sig, nd = _month(n_files, half - 1)
    assert np.isnan(mean[0]) and np.isnan(sig[0]) and nd[0] == 0


def test_month_without_files_has_no_mean():
    mean, sig, nd = O.MonthlyCellMean(3).finalize(0)
    assert np.isnan(mean).all() and (nd == 0).all()


def test_monthly_sigma_is_rms_of_daily_sigma_over_days_with_sigma():
    acc = O.MonthlyCellMean(2)
    acc.add([0.2, 0.0], [True, True], [0.03, np.nan])
    acc.add([0.4, 0.0], [True, True], [0.04, np.nan])
    acc.add([0.9, 0.5], [False, False], [0.5, 0.5])         # invalid day: ignored
    acc.add([0.6, 0.0], [True, True], [np.nan, np.nan])      # valid value, no sigma
    mean, sig, nd = acc.finalize()
    np.testing.assert_allclose(mean, [0.4, 0.0])
    np.testing.assert_allclose(sig[0], np.sqrt((0.03 ** 2 + 0.04 ** 2) / 2))
    assert sig[1] == 0.0                                      # no sigma at all: the floor decides
    np.testing.assert_array_equal(nd, [3, 3])


# --- CCI -----------------------------------------------------------------------------------------
def test_cci_uses_flag_zero_only_and_effective_thickness():
    flag = np.array([0, 1, 2, 3, 4, 5, 0, 0])
    sit = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -0.3, 1.5])
    unc = np.array([0.5] * 7 + [0.4])
    sic = np.array([80.0, 80, 80, 80, 80, 80, 90, np.nan])    # flag 0 with no SIC: unusable
    H, sH, valid = O.cci_effective_thickness(sit, unc, sic, flag)
    np.testing.assert_array_equal(valid, [1, 0, 0, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(H[valid], [1.6, -0.27])        # negative SIT kept
    np.testing.assert_allclose(sH[valid], [0.4, 0.45])
    assert np.isnan(H[~valid]).all() and np.isnan(sH[~valid]).all()
    assert 2 not in [f for f, v in zip(flag, valid) if v]     # flag 2 is not a zero observation


# --- super-observations, floors, weights, hemispheres ---------------------------------------------
def test_super_obs_means_floors_and_counts():
    node = np.array([0, 0, 2, -1, 2, 2])
    val = np.array([0.2, 0.6, 1.0, 0.7, 0.5, np.nan])
    sig = np.array([0.03, 0.05, 0.02, 0.1, 0.04, 0.1])
    ok = np.array([True, True, True, True, True, False])
    obs, s, n = O.super_obs(node, val, sig, ok, 4, O.FLOOR_C)
    np.testing.assert_array_equal(n, [2, 0, 2, 0])
    np.testing.assert_allclose(obs[[0, 2]], [0.4, 0.75])
    np.testing.assert_allclose(s[0], np.sqrt((0.03 ** 2 + 0.05 ** 2) / 2 + 0.1 ** 2))
    np.testing.assert_allclose(s[2], np.sqrt((0.02 ** 2 + 0.04 ** 2) / 2 + 0.1 ** 2))
    assert np.isnan(obs[[1, 3]]).all() and np.isnan(s[[1, 3]]).all()
    _, sH, _ = O.super_obs([0], [1.0], [0.0], [True], 1, O.FLOOR_H)
    np.testing.assert_allclose(sH, [0.3])
    assert O.FLOOR_C == 0.1 and O.FLOOR_H == 0.3


def test_observed_area_weight():
    # n cells x 625 km^2 / (100 km)^2
    np.testing.assert_allclose(O.obs_weight([0, 1, 16, 3]), [0.0, 0.0625, 1.0, 0.1875])
    assert O.CELL_AREA_M2 == 625e6 and O.A_CORR_M2 == 1e10


def test_hemisphere_split_and_merge_refuses_overlap():
    lat = np.array([10.0, 0.0, -0.1, 89.0])
    np.testing.assert_array_equal(O.northern_nodes(lat), [True, False, False, True])
    obs = O.empty_obs(4)
    O.merge_into(obs["C_obs"], 0, np.array([0.5, np.nan, np.nan, 0.2]),
                 np.array([True, False, False, True]))
    O.merge_into(obs["C_obs"], 0, np.array([np.nan, 0.1, 0.0, np.nan]),
                 np.array([False, True, True, False]))
    np.testing.assert_allclose(obs["C_obs"][0], [0.5, 0.1, 0.0, 0.2])
    assert np.isnan(obs["C_obs"][1:]).all()
    with pytest.raises(ValueError, match="common"):
        O.merge_into(obs["C_obs"], 0, np.full(4, 0.3), np.array([True, False, False, False]))
    O.merge_into(obs["nC"], 0, np.array([2, 0, 0, 0]), np.array([True, False, False, False]))
    with pytest.raises(ValueError, match="common"):
        O.merge_into(obs["nC"], 0, np.array([1, 0, 0, 0]), np.array([True, False, False, False]))


def test_check_obs_shapes_and_consistency():
    obs = O.empty_obs(5)
    O.check_obs(obs, 5)
    with pytest.raises(ValueError, match=r"\[12, nod2D\]"):
        O.check_obs(obs, 6)
    bad = dict(obs, sH=np.full((12, 4), np.nan))
    with pytest.raises(ValueError, match="sH"):
        O.check_obs(bad)
    obs["C_obs"][3, 1] = 0.5                                  # value without a count
    with pytest.raises(ValueError, match="nC"):
        O.check_obs(obs)
    obs["nC"][3, 1] = 2
    obs["sC"][3, 1] = 0.1
    O.check_obs(obs)


def test_mesh_surface_nodes(tmp_path):
    geo = np.radians(np.array([[10.0, 70.0], [-20.0, -65.0], [190.0 - 360.0, 85.0]]))
    np.save(tmp_path / "geo_coord_nod2D.npy", geo)
    np.save(tmp_path / "ulevels_nod2D.npy", np.array([1, 1, 2], dtype=np.int32))
    np.save(tmp_path / "nlevels_nod2D.npy", np.array([10, 1, 10], dtype=np.int32))
    lon, lat, wet = O.mesh_surface_nodes(tmp_path)
    np.testing.assert_allclose(lon, [10.0, -20.0, -170.0])
    np.testing.assert_allclose(lat, [70.0, -65.0, 85.0])
    np.testing.assert_array_equal(wet, [True, False, False])


def test_count_above_gives_the_cell_level_extent_per_node():
    node = np.array([0, 0, 0, 2, -1, 2])
    val = np.array([0.15, 0.16, 0.9, 0.5, 0.9, np.nan])
    ok = np.array([True, True, True, True, True, False])
    np.testing.assert_array_equal(O.count_above(node, val, ok, 3), [2, 0, 1])   # > 0.15, strict
    obs = O.empty_obs(3)
    obs["nE"][0, 0] = 1                                       # more icy cells than cells
    with pytest.raises(ValueError, match="nE"):
        O.check_obs(obs)
