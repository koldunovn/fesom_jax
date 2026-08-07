"""Tests for the section volume transport (NN-GM Task 4, second half).

The quantity is a line integral of the normal velocity, so it can be checked against arithmetic
rather than against another implementation — which is the point of choosing this discretization
over a port of pyfesom2's broken-line machinery.

What is checked, and why each one is a way to get a plausible wrong number:

* **A uniform flow through a known section gives length x depth x speed.** The single test that
  makes the whole thing trustworthy: everything else is a variation on it.
* **The sign convention.** "Left of A→B" is a choice, and a sign error in a transport is the
  easiest mistake to publish: the ACC would simply come out flowing west.
* **Convergence in the number of sub-segments.** A first-order line integral is only correct in
  the limit, and "I picked 2000" is not a justification unless the answer has stopped moving.
* **Flow parallel to the section contributes nothing**, so the normal really is the normal.
* **The bottom mask.** Water below the topography must not be counted; without the `nlevels-1`
  subtraction the deepest layer adds a level of ocean that is not there.
* **The great circle and the dateline.** A section crossing 180 degrees must not be interpolated
  in longitude, which would send it the long way round the planet.

The synthetic mesh is a lat-lon strip cut into triangles: small, exactly integrable, and with a
known bottom.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load():
    path = _ROOT / "scripts" / "nn_eddy" / "mesh_diag.py"
    spec = importlib.util.spec_from_file_location("mesh_diag", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def md():
    return _load()


NZ1 = 3
ZBAR = np.array([0.0, -100.0, -400.0, -1000.0])        # thicknesses 100, 300, 600 m


def _strip(lon_c=-68.0, lat_lo=-64.0, lat_hi=-54.0, n_lat=41, half_width=1.5):
    """A meridional strip of triangles centred on ``lon_c``, spanning the section and wider than
    it, so a section drawn down the middle is always inside an element.

    Two columns of nodes at lon_c -/+ half_width, ``n_lat`` rows, split into triangles.
    """
    lats = np.linspace(lat_lo, lat_hi, n_lat)
    lon = np.concatenate([np.full(n_lat, lon_c - half_width),
                          np.full(n_lat, lon_c + half_width)])
    lat = np.concatenate([lats, lats])
    W = np.arange(n_lat)                 # west column node ids
    E = np.arange(n_lat) + n_lat         # east column
    tris = []
    for i in range(n_lat - 1):
        tris.append([W[i], E[i], W[i + 1]])
        tris.append([E[i], E[i + 1], W[i + 1]])
    return dict(lon_deg=lon, lat_deg=lat, elem_nodes=np.asarray(tris, dtype=np.int32),
                zbar=ZBAR, elem_nlevels=np.full(len(tris), NZ1, dtype=np.int32))


def _call(md, mesh, u, v, **kw):
    nelem = mesh["elem_nodes"].shape[0]
    ue = np.full((nelem, NZ1), 0.0) + np.asarray(u, dtype=float)
    ve = np.full((nelem, NZ1), 0.0) + np.asarray(v, dtype=float)
    args = dict(lon0=-68.0, lat0=-54.5, lon1=-68.0, lat1=-63.5, n_sub=2000)
    args.update(kw)
    return md.section_transport(ue, ve, elem_nodes=mesh["elem_nodes"],
                                lon_deg=mesh["lon_deg"], lat_deg=mesh["lat_deg"],
                                elem_nlevels=mesh["elem_nlevels"], zbar=mesh["zbar"], **args)


def _meridional_length_m(lat0, lat1):
    return abs(lat1 - lat0) * np.pi / 180.0 * md_EARTH_R


md_EARTH_R = 6371000.0


# --- the one that makes the rest trustworthy ------------------------------------------------

def test_uniform_eastward_flow_gives_length_times_depth_times_speed(md):
    """A 1 m/s eastward flow through a meridional section from 54.5S to 63.5S over 1000 m of
    water must give L x H x 1 m^3/s. L = 9 degrees of latitude = 1.0006e6 m, H = 1000 m, so
    the answer is 1.0006e9 m^3/s = 1000.6 Sv. Nothing about the mesh enters this."""
    mesh = _strip()
    got = _call(md, mesh, u=1.0, v=0.0)
    length = _meridional_length_m(-54.5, -63.5)
    expect = length * 1000.0 / 1.0e6
    assert got == pytest.approx(expect, rel=2e-3), (got, expect)


def test_the_sign_convention_is_left_of_the_direction_of_travel(md):
    """A section entered north→south (as Drake is: South America to Antarctica) must give
    EASTWARD flow a positive transport. If this ever flips, the ACC flows west in every figure
    this project produces."""
    mesh = _strip()
    south = _call(md, mesh, u=1.0, v=0.0, lat0=-54.5, lat1=-63.5)
    north = _call(md, mesh, u=1.0, v=0.0, lat0=-63.5, lat1=-54.5)
    assert south > 0.0, f"north->south with eastward flow gave {south} Sv; must be positive"
    assert north == pytest.approx(-south, rel=1e-12), "reversing A and B must flip the sign"
    assert _call(md, mesh, u=-1.0, v=0.0) == pytest.approx(-south, rel=1e-12)


def test_flow_along_the_section_contributes_nothing(md):
    """A purely northward flow through a meridional section is parallel to it: exactly zero
    transport, up to the discretization's own noise. If the normal were the tangent this would
    return the full 1000 Sv."""
    mesh = _strip()
    got = _call(md, mesh, u=0.0, v=1.0)
    ref = _call(md, mesh, u=1.0, v=0.0)
    assert abs(got) < 1e-6 * abs(ref), f"parallel flow gave {got} Sv against {ref} Sv normal"


def test_it_converges_in_the_number_of_sub_segments(md):
    """First order in ds, so the number must be shown to have stopped moving rather than
    asserted to be large enough."""
    mesh = _strip()
    vals = [_call(md, mesh, u=1.0, v=0.0, n_sub=n) for n in (100, 500, 2000, 8000)]
    fine = vals[-1]
    for n, x in zip((100, 500, 2000), vals[:-1]):
        assert abs(x - fine) / abs(fine) < 5e-3, f"n_sub={n} gives {x}, n_sub=8000 gives {fine}"
    assert abs(vals[2] - fine) / abs(fine) < abs(vals[0] - fine) / abs(fine) + 1e-12, \
        "refining must not make it worse"


def test_the_bottom_mask_removes_the_water_that_is_not_there(md):
    """Halving every element's level count must remove exactly the layers it removes: with
    thicknesses 100/300/600 m, keeping two of three layers leaves 400 of 1000 m."""
    mesh = _strip()
    full = _call(md, mesh, u=1.0, v=0.0)
    mesh2 = dict(mesh, elem_nlevels=np.full(mesh["elem_nodes"].shape[0], 2, dtype=np.int32))
    two = _call(md, mesh2, u=1.0, v=0.0)
    assert two == pytest.approx(full * 400.0 / 1000.0, rel=2e-3), (two, full)
    mesh0 = dict(mesh, elem_nlevels=np.zeros(mesh["elem_nodes"].shape[0], dtype=np.int32))
    assert _call(md, mesh0, u=1.0, v=0.0) == 0.0, "a dry section must give exactly zero"


def test_the_per_level_profile_sums_to_the_total(md):
    mesh = _strip()
    total, prof = _call(md, mesh, u=1.0, v=0.0, return_profile=True)
    assert prof.shape == (NZ1,)
    assert prof.sum() == pytest.approx(total, rel=1e-12)
    # thickness-weighted: layers of 100/300/600 m under a uniform flow
    np.testing.assert_allclose(prof / prof.sum(), [0.1, 0.3, 0.6], rtol=2e-3)


def test_the_section_is_a_great_circle_not_a_lon_lat_line(md):
    """A section from 170E to 170W is 20 degrees wide across the dateline, not 340 degrees the
    other way. Checked on the geometry rather than through a transport, because a wrong path
    still returns a number."""
    pts = md.great_circle_points(170.0, 0.0, -170.0, 0.0, 21)
    lon, lat = md._xyz_to_lonlat(pts)
    assert np.allclose(lat, 0.0, atol=1e-9)
    # every point must be in [170,180] or [-180,-170]; a lon/lat interpolation would march
    # westward through 0 degrees
    on_short_way = (np.abs(lon) >= 169.999)
    assert on_short_way.all(), f"left the short way round: {lon[~on_short_way]}"


def test_element_centroids_have_no_dateline_seam(md):
    """A triangle straddling 180 degrees must have its centroid near 180, not near 0. Averaging
    longitudes arithmetically puts it in the middle of the Pacific on the wrong side."""
    lon = np.array([179.0, -179.0, 180.0])
    lat = np.array([0.0, 0.0, 1.0])
    clon, clat = md.element_centroids_lonlat(np.array([[0, 1, 2]]), lon, lat)
    assert abs(abs(float(clon[0])) - 180.0) < 1.0, f"centroid longitude {clon[0]}"
    assert 0.0 < float(clat[0]) < 1.0


def test_the_named_sections_are_oriented_north_to_south(md):
    """All three fixed sections must be entered so that eastward flow is positive, i.e. running
    from the equator-side end to the pole-side end. A section entered the other way would report
    the ACC as negative and nothing else would notice."""
    for name, s in md.SECTIONS.items():
        assert s["lat0"] > s["lat1"], f"{name} is entered south->north; eastward flow would " \
                                      f"come out negative"


def test_a_layer_count_mismatch_is_refused(md):
    """zbar and the velocity array must agree on the number of layers. Silently broadcasting
    would give a transport computed with the wrong thicknesses."""
    mesh = _strip()
    nelem = mesh["elem_nodes"].shape[0]
    with pytest.raises(ValueError, match="layers"):
        md.section_transport(np.zeros((nelem, NZ1 + 1)), np.zeros((nelem, NZ1 + 1)),
                            lon0=-68.0, lat0=-54.5, lon1=-68.0, lat1=-63.5,
                            elem_nodes=mesh["elem_nodes"], lon_deg=mesh["lon_deg"],
                            lat_deg=mesh["lat_deg"], elem_nlevels=mesh["elem_nlevels"],
                            zbar=mesh["zbar"])
