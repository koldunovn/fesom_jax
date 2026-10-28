"""REcoM's cold-start state (:mod:`fesom_jax.recom.ic`) against the Fortran reference (lane I, parent plan
Task 13).

The gate: ``recom()`` runs before the first ocean step, so the ``recom_in`` tracers of the step-1 dump of
``pi_30d_jan`` are the IC after ``recom_init``; ``tra_in.<ID>.valuesold`` at step 1 is FESOM's AB2 history,
copied before ``recom_init`` (``oce_setup_step.F90:264-268``); ``recom_in.Alk_surf``, ``Benthos`` and
``GloHplus`` are the other cold-start fields. All 31 tracers on all ``nl-1`` levels (the levels below the bottom
included) are compared BITWISE with the Fortran geographic coordinates (``mesh.geo_coord_nod2D`` of the dump).
With the JAX pi mesh's own coordinates (C-port geometry, up to ~1e-14 rad off Fortran's) the interpolated fields
move by up to ~1e-12 relative: that gate is ``rel 2e-12``, the reason being the input coordinates, not the
algorithm. Skips when the dumps or the pi IC files are absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from fesom_jax import mesh as M
from fesom_jax.recom import ic as I
from fesom_jax.recom import params as P
from fesom_jax.recom import tracers as T

REF = Path(os.environ.get("FESOM_RECOM_REF", "/work/ab0995/a270088/recom_ref"))
DUMPS = REF / "pi_30d_jan" / "dumps"
PI_IC_DIR = Path(os.environ.get("FESOM_RECOM_PI_IC_DIR", "/work/ab0995/a270088/fesom2_sp_zstar/test/input/recom"))
REPO = Path(__file__).resolve().parents[2]
NML = REPO / "scripts" / "recom" / "fortran" / "namelists" / "namelist.recom"
MESH_GEO_RTOL = 2e-12      # JAX pi mesh coordinates vs Fortran's (see the module docstring)

have_ref = pytest.mark.skipif(not (DUMPS / "step_000001").is_dir() or not PI_IC_DIR.is_dir(),
                              reason=f"REcoM reference dumps ({DUMPS}) or pi IC files ({PI_IC_DIR}) absent")


@pytest.fixture(scope="module")
def pi_mesh():
    return M.load_mesh()


@pytest.fixture(scope="module")
def namelist():
    return P.load_namelist(NML)


@pytest.fixture(scope="module")
def ref():
    from fesom_jax.io_dump import load_recom_dump
    names = (["mesh.geo_coord_nod2D", "mesh.nlevels_nod2D", "recom_in.Alk_surf", "recom_in.GloHplus",
              "recom_in.Benthos"] + [f"recom_in.{c}" for c in ("CO23D", "pH3D", "pCO23D", "HCO33D", "CO33D",
                                                                "OmegaC3D", "kspc3D", "rhoSW3D")]
             + [f"recom_in.tr{t}" for t in T.BGC_IDS] + [f"tra_in.tr{t}.valuesold" for t in T.BGC_IDS])
    f, _ = load_recom_dump(DUMPS, 1, names)
    return f


@pytest.fixture(scope="module")
def ic_fortran_geo(pi_mesh, namelist, ref):
    sw, par = namelist
    return I.bgc_initial_state(pi_mesh, I.PI_IC_FILES, sw, par, ic_dir=PI_IC_DIR,
                               geo_coord=ref["mesh.geo_coord_nod2D"])


# ------------------------------------------------------------------------------------------------ the gate
@have_ref
@pytest.mark.parametrize("tid", T.BGC_IDS)
def test_gate_tracer_bitwise(ic_fortran_geo, ref, tid):
    s = T.slot(tid)
    got, want = ic_fortran_geo.bgc[:, :, s], ref[f"recom_in.tr{tid}"]
    assert got.shape == want.shape
    bad = got != want
    assert not bad.any(), (f"{tid} {T.NAMES[s]}: {int(bad.sum())} cells differ, max abs "
                           f"{np.abs(got - want).max():.3e}")


@have_ref
@pytest.mark.parametrize("tid", T.BGC_IDS)
def test_gate_old_bitwise(ic_fortran_geo, ref, tid):
    s = T.slot(tid)
    got, want = ic_fortran_geo.bgc_old[:, :, s], ref[f"tra_in.tr{tid}.valuesold"]
    assert np.array_equal(got, want), f"{tid} {T.NAMES[s]} valuesold: max abs {np.abs(got - want).max():.3e}"


@have_ref
def test_gate_alk_benthos_hplus_cache(ic_fortran_geo, ref):
    assert np.array_equal(ic_fortran_geo.alk_surf, ref["recom_in.Alk_surf"])
    assert np.array_equal(ic_fortran_geo.alk_surf, ic_fortran_geo.bgc[:, 0, T.ialk])
    assert np.array_equal(ic_fortran_geo.benthos, ref["recom_in.Benthos"].T)
    assert np.array_equal(ic_fortran_geo.hplus, ref["recom_in.GloHplus"])
    for c in ("CO23D", "pH3D", "pCO23D", "HCO33D", "CO33D", "OmegaC3D", "kspc3D", "rhoSW3D"):
        assert not ref[f"recom_in.{c}"].any(), f"the carbonate cache {c} is not zero at step 1"


@have_ref
def test_gate_recom_init_rules(ic_fortran_geo, ref, pi_mesh):
    """The pieces the bitwise gate checks implicitly, stated: Fe x1e9 + floor, constants on every level."""
    ic = ic_fortran_geo
    nlev = np.asarray(pi_mesh.nlevels_nod2D)
    below = np.arange(ic.bgc.shape[1])[None, :] >= nlev[:, None] - 1
    fe, fe_old = ic.bgc[:, :, T.ife], ic.bgc_old[:, :, T.ife]
    assert np.all(fe[below] == T.TINY) and np.all(fe_old[below] == 0.0)
    capped = fe == I.VENT_CAP
    assert capped.any()                                                 # the vent box is on the pi mesh
    assert np.array_equal(fe[~capped & ~below], np.maximum(T.TINY, fe_old * I.FE_FACTOR)[~capped & ~below])
    _, par = P.load_namelist(NML)
    for tid in T.BGC_IDS:
        if tid > 1003 and tid not in (1018, 1019, 1022, 1037):
            assert np.all(ic.bgc[:, :, T.slot(tid)] == T.init_value(tid, par))
            assert not ic.bgc_old[:, :, T.slot(tid)].any()
    assert not ic.bgc[:, :, T.idicremin].any()


@have_ref
def test_gate_mesh_coordinates(pi_mesh, namelist, ref):
    sw, par = namelist
    ic = I.bgc_initial_state(pi_mesh, I.PI_IC_FILES, sw, par, ic_dir=PI_IC_DIR)
    for tid in T.BGC_IDS:
        s = T.slot(tid)
        for got, want in ((ic.bgc[:, :, s], ref[f"recom_in.tr{tid}"]),
                          (ic.bgc_old[:, :, s], ref[f"tra_in.tr{tid}.valuesold"])):
            np.testing.assert_allclose(got, want, rtol=MESH_GEO_RTOL, atol=0, err_msg=f"{tid} {T.NAMES[s]}")


# ------------------------------------------------------------------------------------------------ units
def test_binarysearch_vec_matches_scalar():
    from fesom_jax.phc_ic import _binarysearch_d
    arr = np.array([0.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0])
    vals = np.array([-5.0, 0.0, 1e-10, 5.0, 10.0, 10.0 + 5e-10, 49.9, 250.0, 499.0, 500.0, 600.0])
    want = np.array([_binarysearch_d(arr, v) for v in vals])
    assert np.array_equal(I._binarysearch_vec(arr, vals), want)
    assert np.array_equal(I._binarysearch_vec(arr, vals.reshape(1, -1)), want.reshape(1, -1))


def test_ic_layer_depths(pi_mesh):
    z3 = I.ic_layer_depths(pi_mesh)
    zbar, Z = np.asarray(pi_mesh.zbar), np.asarray(pi_mesh.Z)
    nlev = np.asarray(pi_mesh.nlevels_nod2D)
    n = int(np.argmax(nlev))
    kb = nlev[n] - 2
    assert z3[n, 0] == zbar[0] + (zbar[1] - zbar[0]) / 2
    assert np.array_equal(z3[n, 1:kb], Z[1:kb])
    assert z3[n, kb] == zbar[kb] + (zbar[kb + 1] - zbar[kb]) / 2
    assert not z3[n, kb + 1:].any()


class _Toy:
    """Two triangles (0,1,2), (1,3,2) with every element 3 levels deep (2 layers)."""
    elem_nodes = np.array([[0, 1, 2], [1, 3, 2]])
    nod_in_elem2D_offsets = np.array([0, 1, 3, 5, 6])
    nod_in_elem2D = np.array([0, 0, 1, 0, 1, 1])
    nlevels = np.array([3, 3])


def test_extrap_gauss_seidel_in_node_order():
    D = I.DUMMY
    nlev = np.array([3, 3, 3, 3])
    arr = np.array([[1.0, 1.0], [D, D], [D, 5.0], [3.0, 3.0]])
    I._extrap_nod3D(arr, nlev, I._neighbour_table(_Toy))
    # layer 0: node 1 first: neighbours 0 (elem 0) + 3 (elem 1), node 2 dummy -> (1 + 3)/2 = 2; node 2 then
    # sees 0, 1 (now 2) in elem 0 and 1 (2), 3 in elem 1 -> (1 + 2 + 2 + 3)/4 = 2 (in-place Gauss-Seidel)
    assert arr[1, 0] == 2.0 and arr[2, 0] == 2.0
    # layer 1: node 1 from 0, 2 (5) in elem 0 and 3, 2 in elem 1 -> (1 + 5 + 3 + 5)/4
    assert arr[1, 1] == (1.0 + 5.0 + 3.0 + 5.0) / 4


def test_extrap_skipped_when_the_surface_is_complete():
    D = I.DUMMY
    nlev = np.array([3, 3, 3, 3])
    arr = np.array([[1.0, 7.0], [2.0, D], [4.0, 5.0], [3.0, 3.0]])
    I._extrap_nod3D(arr, nlev, I._neighbour_table(_Toy))
    assert arr[1, 1] == 2.0                    # vertical fill only (the do-while is never entered)


def test_extrap_unreachable_surface_raises():
    D = I.DUMMY
    arr = np.full((4, 2), D)
    with pytest.raises(RuntimeError, match="no path"):
        I._extrap_nod3D(arr, np.array([3, 3, 3, 3]), I._neighbour_table(_Toy))


def _write_nc(path, dims=("depth", "lat", "lon"), depth=(5.0, 50.0), fill=None, dtype="f8"):
    with netCDF4.Dataset(path, "w") as nc:
        sizes = {"lon": 4, "lat": 3, "depth": len(depth)}
        for d in dims:
            nc.createDimension(d, sizes[d])
        nc.createVariable("lon", "f8", ("lon",))[:] = [45.0, 135.0, 225.0, 315.0]
        nc.createVariable("lat", "f8", ("lat",))[:] = [-60.0, 0.0, 60.0]
        nc.createVariable("depth", "f8", ("depth",))[:] = depth
        v = nc.createVariable("x", dtype, dims, fill_value=fill)
        v[:] = np.arange(np.prod([sizes[d] for d in dims]), dtype=float).reshape([sizes[d] for d in dims])


def test_read_ic_file_layout_and_fill(tmp_path):
    p = tmp_path / "ok.nc"
    _write_nc(p, depth=(-5.0, -50.0), fill=7.0)
    lon, lat, depth, data = I._read_ic_file(p, "x")
    assert np.array_equal(lon, [-45.0, 45.0, 135.0, 225.0, 315.0, 405.0])
    assert np.array_equal(depth, [5.0, 50.0])                          # sign flipped (nc_depth(2) < 0)
    assert data.shape == (6, 3, 2)
    assert np.array_equal(data[0], data[4]) and np.array_equal(data[5], data[1])
    assert data[1 + 3, 1, 0] == I.DUMMY                                # raw value 7 == _FillValue
    bad = tmp_path / "bad.nc"
    _write_nc(bad, dims=("lat", "lon", "depth"))
    with pytest.raises(ValueError, match="dimensions"):
        I._read_ic_file(bad, "x")
    f4 = tmp_path / "f4.nc"
    _write_nc(f4, fill=7.0, dtype="f4")
    with pytest.raises(NotImplementedError, match="fill value"):
        I._read_ic_file(f4, "x")


def test_resolve_ic_files_errors(tmp_path):
    with pytest.raises(KeyError, match="not a BGC tracer"):
        I.resolve_ic_files({1: ("t.nc", "temp")}, tmp_path)
    with pytest.raises(ValueError, match="ic_dir"):
        I.resolve_ic_files({1001: ("din.nc", "n_an")})
    r = I.resolve_ic_files({1001: ("din.nc", "n_an"), 1019: (tmp_path / "fe.nc", "Fe")}, tmp_path / "d")
    assert r == {1001: (tmp_path / "d" / "din.nc", "n_an"), 1019: (tmp_path / "fe.nc", "Fe")}


def test_unsupported_switch_raises(pi_mesh, namelist):
    import dataclasses
    sw, par = namelist
    with pytest.raises(NotImplementedError):
        I.bgc_initial_state(pi_mesh, {}, dataclasses.replace(sw, ciso=True), par, ic_dir=".")


def test_without_files_and_into_state(pi_mesh, namelist):
    """No IC files: every file tracer stays 0 (Fe -> tiny); with_bgc_ic pads to the State's nl levels."""
    from fesom_jax.state import State
    sw, par = namelist
    ic = I.bgc_initial_state(pi_mesh, {}, sw, par)
    assert np.all(ic.bgc[:, :, T.ife] == T.TINY) and not ic.bgc[:, :, T.idin].any()
    assert not ic.alk_surf.any()
    st = I.with_bgc_ic(State.rest(pi_mesh, xp=np), ic, xp=np)
    assert st.bgc.shape == (pi_mesh.nod2D, pi_mesh.nl, T.NTR) == st.bgc_old.shape
    assert np.array_equal(st.bgc[:, :-1], ic.bgc) and not st.bgc[:, -1].any()
    assert np.array_equal(st.bgc_old[:, :-1], ic.bgc_old)
