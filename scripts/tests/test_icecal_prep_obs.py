"""``scripts/icecal/prep_obs.py`` end to end on synthetic files: two OSI-SAF daily files and one
CCI monthly file on a 2 x 3 grid, a five-node mesh (one dry node on top of a cell, one southern
node), checked against hand-computed super-observations; an existing output is refused and left
unchanged. ``scripts`` group."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "icecal"))
import prep_obs as P  # noqa: E402

from fesom_jax import obs_seaice as O  # noqa: E402

LAT = np.array([[80.0, 80.0, 80.0], [80.1, 80.1, 80.1]])
LON = np.array([[0.0, 10.0, 30.0], [0.0, 10.0, 30.0]])
F = O.OSI_FILL


def _grid(ds):
    ds.createDimension("time", None)
    ds.createDimension("yc", 2)
    ds.createDimension("xc", 3)
    ds.createVariable("xc", "f8", ("xc",))[:] = [-25.0, 0.0, 25.0]
    ds.createVariable("yc", "f8", ("yc",))[:] = [25.0, 0.0]
    ds.createVariable("lat", "f4", ("yc", "xc"))[:] = LAT
    ds.createVariable("lon", "f4", ("yc", "xc"))[:] = LON


def _osi_day(path, day, conc, unc, flag):
    with netCDF4.Dataset(path, "w") as ds:
        ds.set_auto_maskandscale(False)
        _grid(ds)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "seconds since 1978-01-01 00:00:00"
        t.calendar = "standard"
        t[0] = netCDF4.date2num(datetime.datetime(2013, 1, day, 12), t.units, t.calendar)
        for name, val in (("ice_conc", conc), ("total_standard_uncertainty", unc)):
            v = ds.createVariable(name, "i4", ("time", "yc", "xc"), fill_value=F)
            v.set_auto_maskandscale(False)       # write RAW integers, as the product stores them
            v.units = "%"
            v.scale_factor = 0.01
            v[0] = np.asarray(val, dtype=np.int32).reshape(2, 3)
        fl = ds.createVariable("status_flag", "i2", ("time", "yc", "xc"), fill_value=-32768)
        fl.flag_masks = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int16)
        fl.flag_meanings = ("land lake open_water_filtered land_spill_over high_t2m "
                            "spatial_interp temporal_interp max_ice_climo")
        fl[0] = np.asarray(flag, dtype=np.int16).reshape(2, 3)


def _cci(path, sit, unc, sic, flag):
    with netCDF4.Dataset(path, "w") as ds:
        ds.set_auto_maskandscale(False)
        _grid(ds)
        ds["lat"][:] = LAT
        for name, val, units in (("sea_ice_thickness", sit, "m"),
                                 ("sea_ice_thickness_uncertainty", unc, "m"),
                                 ("sea_ice_concentration", sic, "percent")):
            v = ds.createVariable(name, "f4", ("time", "yc", "xc"))
            v.units = units
            v[0] = np.asarray(val, dtype=np.float32).reshape(2, 3)
        fl = ds.createVariable("status_flag", "i1", ("time", "yc", "xc"))
        fl.flag_values = np.arange(6, dtype=np.int8)
        fl[0] = np.asarray(flag, dtype=np.int8).reshape(2, 3)


@pytest.fixture()
def inputs(tmp_path):
    osi = tmp_path / "osi"
    (osi / "2013" / "01").mkdir(parents=True)
    # cells, row major: c00 (0E 80N) c01 (10E) c02 (30E, beyond the cap) / c10 c11 c12 at 80.1N
    _osi_day(osi / "2013/01/ice_conc_nh_ease2-250_cdr-v3p0_201301011200.nc", 1,
             conc=[5000, 0, 8000, 7000, F, 0], unc=[300, F, 200, 400, F, F],
             flag=[0, 4, 0, 0, 1, 128])
    _osi_day(osi / "2013/01/ice_conc_nh_ease2-250_cdr-v3p0_201301021200.nc", 2,
             conc=[7000, 0, 8000, 9000, F, 0], unc=[500, F, 200, F, F, F],
             flag=[0, 128, 0, 32, 1, 128])
    cci = tmp_path / "cci"
    (cci / "2013").mkdir(parents=True)
    nan = np.nan
    _cci(cci / "2013/ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-201301-fv2.0.nc",
         sit=[2.0, nan, 3.0, 1.0, nan, nan], unc=[0.5, nan, 0.5, 0.3, nan, nan],
         sic=[90.0, 40.0, 100.0, 50.0, nan, nan], flag=[0, 2, 0, 0, 4, 1])
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    # nodes: 0 (0E 80N), 1 (10E 80N), 2 (20E 80N), 3 DRY exactly on c10, 4 southern
    geo = np.radians([[0.0, 80.0], [10.0, 80.0], [20.0, 80.0], [0.0, 80.1], [0.0, -80.0]])
    np.save(mesh / "geo_coord_nod2D.npy", geo)
    np.save(mesh / "ulevels_nod2D.npy", np.array([1, 1, 1, 1, 1], dtype=np.int32))
    np.save(mesh / "nlevels_nod2D.npy", np.array([20, 20, 20, 1, 20], dtype=np.int32))
    return {"osi": osi, "cci": cci, "mesh": mesh, "out": tmp_path / "out"}


def _run(inp):
    return P.main(["--years", "2013", "--out", str(inp["out"]), "--mesh-dir", str(inp["mesh"]),
                   "--osi-root", str(inp["osi"]), "--cci-root", str(inp["cci"]),
                   "--min-files", "2"])                     # two synthetic daily files


def test_cli_writes_the_expected_super_observations(inputs):
    assert _run(inputs) == 0
    out = inputs["out"]
    with np.load(out / "obs_2013.npz") as z:
        obs = {k: z[k] for k in z.files}
    nod = 5
    for k in O.OBS_KEYS:
        assert obs[k].shape == (12, nod)
    # node 0 <- c00 (mean 0.6 of 0.5/0.7, sigma^2 mean(0.03^2, 0.05^2)) and c10 (valid on 1 of 2
    # files: day 2 is spatial_interp -> 0.7, sigma 0.04); the dry node 3 on top of c10 gets nothing
    np.testing.assert_allclose(obs["C_obs"][0, 0], 0.65, rtol=1e-12)
    s2 = ((0.03 ** 2 + 0.05 ** 2) / 2 + 0.04 ** 2) / 2
    np.testing.assert_allclose(obs["sC"][0, 0], np.sqrt(s2 + 0.1 ** 2), rtol=1e-12)
    # node 1 <- c01: open_water_filtered then max_ice_climo, a VALID zero without uncertainty;
    # c11 is land
    assert obs["C_obs"][0, 1] == 0.0
    np.testing.assert_allclose(obs["sC"][0, 1], 0.1, rtol=1e-12)
    np.testing.assert_array_equal(obs["nC"][0], [2, 1, 0, 0, 0])
    np.testing.assert_array_equal(obs["nE"][0], [2, 0, 0, 0, 0])   # cells above 15 %
    assert np.isnan(obs["C_obs"][0, 2:]).all()
    # thickness: node 0 <- c00 (2.0 * 0.9) and c10 (1.0 * 0.5); c01 flag 2 is not a zero
    np.testing.assert_allclose(obs["H_obs"][0, 0], (1.8 + 0.5) / 2, rtol=1e-6)
    np.testing.assert_allclose(obs["sH"][0, 0],
                               np.sqrt((0.45 ** 2 + 0.15 ** 2) / 2 + 0.3 ** 2), rtol=1e-6)
    np.testing.assert_array_equal(obs["nH"][0], [2, 0, 0, 0, 0])
    # every other month empty
    assert (obs["nC"][1:] == 0).all() and (obs["nH"][1:] == 0).all()
    np.testing.assert_array_equal(obs["nh"], [True, True, True, True, False])

    prov = json.loads((out / "obs_2013.json").read_text())
    jan = prov["months"][0]["C_nh"]
    assert jan["n_files"] == 2 and jan["nodes_valid"] == 2
    assert jan["cells_valid_beyond_cap"] == 2               # c02 and c12
    assert jan["flag_cell_days"]["land"] == 2 and jan["flag_cell_days"]["spatial_interp"] == 1
    assert jan["kept_cell_days_without_sigma"] == 4     # c01 and c12 on both days
    np.testing.assert_allclose(jan["sum_weight"], 3 * 0.0625)
    assert prov["months"][0]["H_nh"]["nodes_valid"] == 1
    assert prov["npz_sha256"] == P._file_sha(out / "obs_2013.npz")
    assert (out / "assign.npz").exists()


def test_cli_refuses_an_existing_output(inputs):
    assert _run(inputs) == 0
    out = inputs["out"]
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert _run(inputs) == 3
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    # a half-written year (npz without json) is refused as well
    (out / "obs_2013.json").rename(out / "moved.json")
    assert _run(inputs) == 3


def test_years_after_2019_are_refused(inputs):
    with pytest.raises(SystemExit):
        P.main(["--years", "2019-2020", "--out", str(inputs["out"]),
                "--mesh-dir", str(inputs["mesh"])])


def test_parse_years():
    assert P.parse_years("2013") == [2013]
    assert P.parse_years("1980-1982,2013") == [1980, 1981, 1982, 2013]


def test_a_month_below_the_minimum_file_count_is_a_gap(inputs):
    assert P.main(["--years", "2013", "--out", str(inputs["out"]), "--mesh-dir", str(inputs["mesh"]),
                   "--osi-root", str(inputs["osi"]), "--cci-root", str(inputs["cci"]),
                   "--min-files", "3"]) == 0
    with np.load(inputs["out"] / "obs_2013.npz") as z:
        assert (z["nC"] == 0).all() and np.isnan(z["C_obs"]).all()
        assert z["nH"][0].sum() == 2                         # the monthly CCI file is unaffected
    jan = json.loads((inputs["out"] / "obs_2013.json").read_text())["months"][0]["C_nh"]
    assert jan["n_files"] == 2 and jan["skipped_fewer_than_min_files"] is True
    assert P.MIN_FILES == 8


def test_an_existing_assignment_is_reused_read_only_and_checked(inputs, tmp_path):
    assert _run(inputs) == 0
    cache = inputs["out"] / "assign.npz"
    before = cache.read_bytes()
    out2 = tmp_path / "out2"
    args = ["--years", "2013", "--out", str(out2), "--mesh-dir", str(inputs["mesh"]),
            "--osi-root", str(inputs["osi"]), "--cci-root", str(inputs["cci"]),
            "--assign", str(cache), "--min-files", "2"]
    assert P.main(args) == 0
    assert not (out2 / "assign.npz").exists() and cache.read_bytes() == before
    with np.load(inputs["out"] / "obs_2013.npz") as a, np.load(out2 / "obs_2013.npz") as b:
        for k in O.OBS_KEYS + ("nE",):
            np.testing.assert_array_equal(a[k], b[k])
    # a different mesh does not match the cache
    np.save(inputs["mesh"] / "nlevels_nod2D.npy", np.array([20, 20, 20, 20, 20], dtype=np.int32))
    args[3] = str(tmp_path / "out3")
    with pytest.raises(ValueError, match="mesh_sha"):
        P.main(args)
