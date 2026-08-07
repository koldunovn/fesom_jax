"""NN-GM Task 4 — tests for the overturning diagnostic.

Synthetic meshes with an analytically known answer, plus the edge cases that break real
implementations: partial-depth triangles, basin masking, latitude binning. CPU-fast, no
model and no real mesh needed. Runs standalone (``pytest scripts/tests/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nn_eddy"))
import mesh_diag as md  # noqa: E402


def _strip(nlat_bands=5, nl=4, area=2.0):
    """A trivial 'mesh': one triangle per latitude band, all nodes at that latitude.

    Triangle j sits at latitude lats[j]; every triangle is full depth. Node k of
    triangle j is node 3j+k, so per-triangle w values are independent.
    """
    lat_centres = np.linspace(-80.0, 80.0, nlat_bands)
    elem_nodes = np.arange(3 * nlat_bands).reshape(nlat_bands, 3)
    elem_area = np.full(nlat_bands, area)
    elem_nlevels = np.full(nlat_bands, nl)
    node_lat = np.repeat(lat_centres, 3)
    node_lon = np.zeros(3 * nlat_bands)
    return dict(elem_nodes=elem_nodes, elem_area=elem_area, elem_nlevels=elem_nlevels,
                elem_lat=lat_centres, node_lat=node_lat, node_lon=node_lon,
                nnod=3 * nlat_bands, nl=nl)


def test_zero_w_gives_zero_moc():
    g = _strip()
    w = np.zeros((g["nnod"], g["nl"]))
    lats, moc = md.moc_from_w(w, g["elem_nodes"], g["elem_area"], g["elem_nlevels"], g["elem_lat"])
    assert moc.shape == (91, g["nl"])
    assert np.allclose(moc, 0.0)


def test_single_band_known_value():
    """One triangle with uniform w: the streamfunction south of it must equal
    -(w * area) in Sv, and north of it must be zero (cumulative sum from the north)."""
    g = _strip(nlat_bands=1)
    g["elem_lat"] = np.array([0.0])
    w = np.zeros((g["nnod"], g["nl"]))
    w[:, :] = 3.0e-5                       # m/s, uniform over the triangle's 3 nodes
    lats, moc = md.moc_from_w(w, g["elem_nodes"], g["elem_area"], g["elem_nlevels"], g["elem_lat"])
    expect = -(3.0e-5 * g["elem_area"][0] * 1e-6)      # Sv
    j = int(np.argmin(np.abs(lats - 0.0)))
    assert np.allclose(moc[j], expect, rtol=1e-12), (moc[j], expect)
    assert np.allclose(moc[j + 1:], 0.0)               # nothing north of the source
    assert np.allclose(moc[:j], expect, rtol=1e-12)    # accumulated to the south


def test_cumulative_sum_direction():
    """Two bands: the southern value must be the sum of both, the northern only its own."""
    g = _strip(nlat_bands=2)
    g["elem_lat"] = np.array([-40.0, 40.0])
    w = np.zeros((g["nnod"], g["nl"]))
    w[0:3, :] = 1.0e-5        # southern triangle
    w[3:6, :] = 2.0e-5        # northern triangle
    lats, moc = md.moc_from_w(w, g["elem_nodes"], g["elem_area"], g["elem_nlevels"], g["elem_lat"])
    a = g["elem_area"][0] * 1e-6
    js = int(np.argmin(np.abs(lats - (-40.0))))
    jn = int(np.argmin(np.abs(lats - 40.0)))
    assert np.allclose(moc[jn], -(2.0e-5 * a), rtol=1e-12)
    assert np.allclose(moc[js], -((1.0e-5 + 2.0e-5) * a), rtol=1e-12)


def test_partial_depth_triangle_blanks_below_bottom():
    """Levels at or below a triangle's bottom must not contribute at all."""
    g = _strip(nlat_bands=1, nl=4)
    g["elem_lat"] = np.array([0.0])
    g["elem_nlevels"] = np.array([2])          # only levels 0,1 are valid
    w = np.ones((g["nnod"], g["nl"])) * 1.0e-5
    lats, moc = md.moc_from_w(w, g["elem_nodes"], g["elem_area"], g["elem_nlevels"], g["elem_lat"])
    j = int(np.argmin(np.abs(lats - 0.0)))
    expect = -(1.0e-5 * g["elem_area"][0] * 1e-6)
    assert np.allclose(moc[j, :2], expect, rtol=1e-12)
    assert np.allclose(moc[j, 2:], 0.0)        # blanked levels contribute nothing

    # and a deeper-water value at a blanked level must not change the answer
    w2 = w.copy()
    w2[:, 2:] = 1.0e3
    _, moc2 = md.moc_from_w(w2, g["elem_nodes"], g["elem_area"], g["elem_nlevels"], g["elem_lat"])
    assert np.allclose(moc, moc2)


def test_mask_excludes_nodes():
    g = _strip(nlat_bands=2)
    g["elem_lat"] = np.array([-40.0, 40.0])
    w = np.ones((g["nnod"], g["nl"])) * 1.0e-5
    mask = np.ones(g["nnod"], dtype=bool)
    mask[3:6] = False                          # drop the northern triangle's nodes
    lats, moc = md.moc_from_w(w, g["elem_nodes"], g["elem_area"], g["elem_nlevels"],
                              g["elem_lat"], mask=mask)
    jn = int(np.argmin(np.abs(lats - 40.0)))
    js = int(np.argmin(np.abs(lats - (-40.0))))
    assert np.allclose(moc[jn], 0.0)
    assert np.allclose(moc[js], -(1.0e-5 * g["elem_area"][0] * 1e-6), rtol=1e-12)


def test_area_weighting_is_linear():
    g = _strip(nlat_bands=1, area=2.0)
    g["elem_lat"] = np.array([0.0])
    w = np.ones((g["nnod"], g["nl"])) * 1.0e-5
    _, moc_a = md.moc_from_w(w, g["elem_nodes"], g["elem_area"], g["elem_nlevels"], g["elem_lat"])
    _, moc_b = md.moc_from_w(w, g["elem_nodes"], g["elem_area"] * 3.0, g["elem_nlevels"], g["elem_lat"])
    assert np.allclose(moc_b, 3.0 * moc_a, rtol=1e-12)


def test_triangle_mean_over_three_nodes():
    """w is averaged over the triangle's three nodes, not summed."""
    g = _strip(nlat_bands=1)
    g["elem_lat"] = np.array([0.0])
    w = np.zeros((g["nnod"], g["nl"]))
    w[0, :] = 3.0e-5           # one node only -> mean is a third of it
    lats, moc = md.moc_from_w(w, g["elem_nodes"], g["elem_area"], g["elem_nlevels"], g["elem_lat"])
    j = int(np.argmin(np.abs(lats - 0.0)))
    assert np.allclose(moc[j], -(1.0e-5 * g["elem_area"][0] * 1e-6), rtol=1e-12)


def test_centroid_latitude():
    elem_nodes = np.array([[0, 1, 2], [1, 2, 3]])
    lat = np.array([0.0, 3.0, 6.0, 9.0])
    assert np.allclose(md.elem_centroid_lat(elem_nodes, lat), [3.0, 6.0])


def test_amoc_strength_picks_box_maximum():
    lats = np.linspace(-90, 90, 91)
    zbar = -np.array([0.0, 400.0, 1000.0, 2000.0, 4000.0])
    moc = np.zeros((91, 5))
    j = int(np.argmin(np.abs(lats - 40.0)))
    moc[j, 2] = 17.5                       # inside the box (40N, 1000 m)
    moc[j, 4] = 99.0                       # outside (4000 m deep)
    moc[5, 2] = 55.0                       # outside (far south)
    assert md.amoc_strength(lats, moc, zbar) == pytest.approx(17.5)


@pytest.mark.skipif(not md.GEOJSON_DIR.exists(), reason="pyfesom2 geojson polygons not available")
def test_atlantic_basin_mask_geography():
    """Points that are unambiguously in/out of the Atlantic MOC basin."""
    lon = np.array([-30.0, -160.0, 80.0, -40.0])     # N Atlantic, Pacific, Indian, S Atlantic
    lat = np.array([40.0, 20.0, -20.0, -30.0])
    m = md.basin_mask(lon, lat, "Atlantic_MOC")
    assert m[0] and m[3], "north and south Atlantic points must be inside"
    assert not m[1] and not m[2], "Pacific and Indian points must be outside"


def test_global_mask_is_everything():
    lon = np.linspace(-180, 180, 7)
    lat = np.linspace(-80, 80, 7)
    assert md.basin_mask(lon, lat, "Global").all()


def test_bin_edge_element_is_round_off_robust():
    """A triangle whose centroid sits exactly on a latitude-bin edge must land in the same
    bin whether or not the stored coordinates carry last-bit round-off.

    This is not hypothetical: on the CORE2 mesh two triangles have centroid latitudes of
    exactly 86.0 and 74.0 degrees, and the .npy mesh export differs from the mesh ASCII by
    ~1e-13 deg, which moved ~7e-3 Sv of Atlantic overturning by one bin.
    """
    g = _strip(nlat_bands=1)
    w = np.ones((g["nnod"], g["nl"])) * 1.0e-5
    out = []
    for lat in (74.0, 74.0 - 1e-13, 74.0 + 1e-13):
        _, moc = md.moc_from_w(w, g["elem_nodes"], g["elem_area"], g["elem_nlevels"],
                               np.array([lat]))
        out.append(moc)
    assert np.allclose(out[0], out[1]) and np.allclose(out[0], out[2]), \
        "bin assignment must not depend on last-bit round-off in the centroid latitude"
