"""Fortran FESOM2 restart -> fesom-jax State (:mod:`fesom_jax.fortran_restart`).

Synthetic: a 4-node / 2-element mesh with uneven depths and a per-variable restart directory in the
FESOM2 layout (``time``, ``nz_1``/``nz``, ``node``/``elem``; two records, the last one counts):
wet lanes come from the file, dry lanes from the reference, ``helem`` is the element mean of
``hnode``, ``hbar_old``/``hnode_new`` follow ``hbar``/``hnode``, the stresses are zero, a Kelvin
``ice_temp`` becomes °C, an absent ``temp_M1`` falls back to ``temp``, and either axis order reads.
The source table covers every State field. CORE2 (data): the Fortran Feb-1 1958 restart of
``fesom_jax_dev/fortran/core2_rst/jan`` converts, and its wet lanes match the files bitwise.
"""

from __future__ import annotations

import dataclasses
import datetime
import types
from pathlib import Path

import numpy as np
import pytest

from fesom_jax import fortran_restart as FR
from fesom_jax.state import State

ROOT = Path(__file__).resolve().parents[2]
CORE2_FORTRAN = Path("/work/ab0995/a270088/fesom_jax_dev/fortran/core2_rst/jan")


def _mesh():
    return types.SimpleNamespace(
        nod2D=4, elem2D=2, nl=4,
        elem_nodes=np.array([[0, 1, 2], [1, 2, 3]]),
        ulevels_nod2D=np.array([1, 1, 1, 1]), nlevels_nod2D=np.array([4, 3, 4, 2]),
        ulevels=np.array([1, 1]), nlevels=np.array([3, 2]))


def _write(path: Path, name: str, data: np.ndarray, dims: tuple):
    import netCDF4
    with netCDF4.Dataset(path / f"{name}.nc", "w") as ds:
        for d, s in zip(dims, data.shape):
            if d not in ds.dimensions:
                ds.createDimension(d, s)
        ds.createVariable("time", "f8", ("time",))[:] = np.arange(data.shape[0])
        ds.createVariable(name, "f8", dims)[:] = data


def _restart(tmp_path, *, with_m1=True, node_first=False):
    """Two records; record 1 (the last) = 1000*id + 10*level + entity, record 0 = -1."""
    oce, ice = tmp_path / "fesom.2000.oce.restart", tmp_path / "fesom.2000.ice.restart"
    oce.mkdir()
    ice.mkdir()
    n, e = 4, 2

    def field(i, nk, ne):
        rec = 1000.0 * i + 10.0 * np.arange(nk)[:, None] + np.arange(ne)[None, :]
        return np.stack([np.full_like(rec, -1.0), rec])                  # (time, nk, ne)
    vars3 = {"hnode": (1, 3, "nz_1", n), "temp": (2, 3, "nz_1", n), "salt": (3, 3, "nz_1", n),
             "w": (4, 4, "nz", n), "w_expl": (5, 4, "nz", n), "w_impl": (6, 4, "nz", n),
             "tke": (7, 4, "nz", n), "u": (8, 3, "nz_1", e), "v": (9, 3, "nz_1", e),
             "urhs_AB": (10, 3, "nz_1", e), "vrhs_AB": (11, 3, "nz_1", e)}
    if with_m1:
        vars3.update({"temp_M1": (12, 3, "nz_1", n), "salt_M1": (13, 3, "nz_1", n)})
    for name, (i, nk, zd, ne) in vars3.items():
        a = field(i, nk, ne)
        ed = "node" if ne == n else "elem"
        if node_first and name == "temp":
            _write(oce, name, np.swapaxes(a, 1, 2), ("time", ed, zd))
        else:
            _write(oce, name, a, ("time", zd, ed))
    for i, name in enumerate(("ssh", "hbar", "ssh_rhs_old"), start=20):
        _write(oce, name, np.stack([np.full(n, -1.0), 1000.0 * i + np.arange(n)]), ("time", "node"))
    for i, name in enumerate(("area", "hice", "hsnow", "uice", "vice"), start=30):
        _write(ice, name, np.stack([np.full(n, -1.0), 0.1 * i + 0.01 * np.arange(n)]),
               ("time", "node"))
    _write(ice, "ice_temp", np.stack([np.full(n, -1.0), 260.0 + np.arange(n)]), ("time", "node"))
    return FR.FortranRestart(oce), FR.FortranRestart(ice)


def _ref(mesh):
    z = State.zeros(mesh, xp=np)
    return dataclasses.replace(z, **{f.name: np.full_like(getattr(z, f.name), -999.0)
                                     for f in dataclasses.fields(z)})


@pytest.mark.parametrize("node_first", [False, True])
def test_wet_lanes_from_the_restart_dry_lanes_from_the_reference(tmp_path, node_first):
    mesh = _mesh()
    oce, ice = _restart(tmp_path, node_first=node_first)
    st, rep = FR.state_from_fortran(mesh, oce, ice, _ref(mesh))
    M = FR._masks(mesh)
    lay, el = M["nod_layer"], M["elem_layer"]
    temp = 2000.0 + 10.0 * np.arange(3)[None, :] + np.arange(4)[:, None]          # (node, level)
    T = np.asarray(st.T)
    np.testing.assert_array_equal(T[:, :3][lay[:, :3]], temp[lay[:, :3]])
    assert np.all(T[~lay] == -999.0)
    u = 8000.0 + 10.0 * np.arange(3)[None, :] + np.arange(2)[:, None]
    uv = np.asarray(st.uv)
    np.testing.assert_array_equal(uv[:, :3, 0][el[:, :3]], u[el[:, :3]])
    np.testing.assert_array_equal(uv[:, :3, 1][el[:, :3]], (u + 1000.0)[el[:, :3]])
    assert np.all(uv[~el] == -999.0)
    w = np.asarray(st.w)
    iface = M["nod_iface"]
    np.testing.assert_array_equal(w[iface], (4000.0 + 10.0 * np.arange(4)[None, :]
                                             + np.arange(4)[:, None])[iface])
    hn = np.asarray(st.hnode)
    helem = np.asarray(st.helem)
    want = hn[mesh.elem_nodes].mean(axis=1)
    np.testing.assert_array_equal(helem[el], want[el])
    assert np.all(helem[~el] == -999.0)
    np.testing.assert_array_equal(np.asarray(st.hbar_old), np.asarray(st.hbar))
    np.testing.assert_array_equal(np.asarray(st.hnode_new), hn)
    np.testing.assert_array_equal(np.asarray(st.eta_n), 20000.0 + np.arange(4))
    for s in ("sigma11", "sigma12", "sigma22"):
        assert np.all(np.asarray(getattr(st, s)) == 0.0)
    np.testing.assert_allclose(np.asarray(st.t_skin), 260.0 + np.arange(4) - 273.15)
    np.testing.assert_allclose(np.asarray(st.a_ice), 3.0 + 0.01 * np.arange(4))
    assert "T_old<-temp_M1" in rep["restart"] and not rep["missing"]
    assert np.all(np.asarray(st.Kv) == -999.0) and "Kv" in rep["reference"]


def test_without_the_ab_slot_the_old_tracer_is_the_current_one(tmp_path):
    mesh = _mesh()
    oce, ice = _restart(tmp_path, with_m1=False)
    st, rep = FR.state_from_fortran(mesh, oce, ice, _ref(mesh))
    np.testing.assert_array_equal(np.asarray(st.T_old), np.asarray(st.T))
    assert any(r.startswith("T_old=T") for r in rep["derived"])


def test_the_source_table_covers_every_state_field_and_every_field_a_step_reads():
    names = {f.name for f in dataclasses.fields(State)}
    assert set(FR.SOURCES) == names
    assert set(FR.STEP_READS) <= names
    # the fields a step reads that do NOT come from the restart: each is a FESOM restart value
    # (CG warm start 0, the initial skin temperature, zero stresses) or reads only into itself
    not_restarted = {k for k in FR.STEP_READS if FR.SOURCES[k] not in ("restart", "derived")}
    assert not_restarted == {"d_eta", "t_skin", "sigma11", "sigma12", "sigma22", "del_ttf",
                             "uvnode_rhs"}


def test_restart_step_counts_whole_steps():
    assert FR.restart_step(datetime.datetime(1958, 2, 1), 1958, 1800.0) == 1488
    assert FR.restart_step(datetime.datetime(2013, 1, 1), 2008, 900.0) == 1827 * 96
    with pytest.raises(ValueError):
        FR.restart_step(datetime.datetime(1958, 2, 1, 0, 10), 1958, 1800.0)


@pytest.mark.skipif(not (CORE2_FORTRAN / "fesom.1958.oce.restart").is_dir()
                    or not (ROOT / "data" / "mesh_core2").is_dir(),
                    reason="the CORE2 Fortran Feb-1 1958 restart / CORE2 mesh are not here")
def test_core2_fortran_restart_converts_and_matches_the_files_bitwise(tmp_path):
    import sys
    import zarr
    sys.path.insert(0, str(ROOT / "scripts" / "fortran_restart"))
    import convert
    out = tmp_path / "core2_19580201"
    rc = convert.main(["--oce", str(CORE2_FORTRAN / "fesom.1958.oce.restart"),
                       "--ice", str(CORE2_FORTRAN / "fesom.1958.ice.restart"),
                       "--mesh-dir", str(ROOT / "data" / "mesh_core2"),
                       "--ic-dir", str(ROOT / "data" / "ic_core2"), "--date", "1958-02-01",
                       "--start-year", "1958", "--dt", "1800", "--out", str(out)])
    assert rc == 0
    g = zarr.open(str(out), mode="r")
    assert int(g.attrs["step"]) == 1488 and g.attrs["calendar_date"] == "1958-doy032"
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(str(ROOT / "data" / "mesh_core2"))
    M = FR._masks(mesh)
    oce = FR.FortranRestart(CORE2_FORTRAN / "fesom.1958.oce.restart")
    ice = FR.FortranRestart(CORE2_FORTRAN / "fesom.1958.ice.restart")
    n, e = int(mesh.nod2D), int(mesh.elem2D)
    lay = M["nod_layer"][:, :47]
    for name, var in (("T", "temp"), ("S", "salt"), ("hnode", "hnode"), ("T_old", "temp_M1")):
        np.testing.assert_array_equal(np.asarray(g[name][...])[:, :47][lay], oce.read(var, n)[lay])
    el = M["elem_layer"][:, :47]
    np.testing.assert_array_equal(np.asarray(g["uv"][...])[:, :47, 0][el], oce.read("u", e)[el])
    np.testing.assert_array_equal(np.asarray(g["eta_n"][...]), oce.read("ssh", n))
    for name, var in (("a_ice", "area"), ("m_ice", "hice"), ("u_ice", "uice")):
        np.testing.assert_array_equal(np.asarray(g[name][...]), ice.read(var, n))
    assert np.all(np.asarray(g["sigma11"][...]) == 0.0)
