"""``scripts/icecal/score.py`` on synthetic Fortran-like files, a synthetic JAX fields file and
synthetic observation files (six nodes, four years): the record -> month mapping, the fill value
read as ice-free, J against an independent loop, area and extent against reference numbers frozen
from the paper_jax helpers (``hemisphere_ice_area``/``hemisphere_ice_extent``/``grid_ice_area``,
evaluated once on the same synthetic data, not imported), the IIEE against a cell-level loop,
trends, observation gaps (a missing month, a missing year) and a partial model year. ``scripts``
group."""

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
import score as SC  # noqa: E402

from fesom_jax import obs_seaice as O  # noqa: E402

LAT = np.array([70.0, 75.0, 80.0, 85.0, -65.0, -70.0])
LON = np.array([0.0, 10.0, 20.0, 30.0, 0.0, 10.0])
AREA = np.array([1.0e9, 2.0e9, 1.5e9, 0.5e9, 3.0e9, 2.5e9])
NOD = LAT.size
FILL = np.float32(9.96921e36)
MODEL_MONTHS = {2001: 12, 2002: 12, 2003: 3}          # 2003: a partial year (3 records)
VALUES = np.array([0.0, 0.1, 0.3, 0.8, 1.0])


# --- the synthetic world -----------------------------------------------------------------------
def model_a(y, m):
    """Model concentration of year y, month m (0-based); exact zeros are ice-free nodes."""
    return ((np.arange(NOD) * 3 + m * 5 + y) % 10) / 10.0


def model_h(y, m):
    a = model_a(y, m)
    return np.where(a > 0, 2.0 * a + 0.1 * np.arange(NOD), 0.0)


def obs_months(y):
    """Months with observations: 2002 lacks December (a gap), 2003 has only Jan-Feb."""
    return {2001: range(12), 2002: range(11), 2003: range(2)}[y]


def cells(y, m):
    """Per node, the monthly-mean concentrations of its cells (0-2 cells) and CCI thicknesses."""
    rng = np.random.default_rng(1000 * y + m)
    conc, sit = [], []
    for n in range(NOD):
        k = (n + m + y) % 3
        conc.append(rng.choice(VALUES, size=k))
        sit.append(rng.uniform(0.2, 3.0, size=k) if (LAT[n] > 0 and (m + 1) in O.CCI_MONTHS)
                   else np.zeros(0))
    return conc, sit


def build_obs(y):
    obs = O.empty_obs(NOD)
    for m in obs_months(y):
        conc, sit = cells(y, m)
        for n in range(NOD):
            if conc[n].size:
                obs["C_obs"][m, n] = conc[n].mean()
                obs["sC"][m, n] = 0.1 + 0.01 * n
                obs["nC"][m, n] = conc[n].size
                obs["nE"][m, n] = int(np.sum(conc[n] > 0.15))
            if sit[n].size:
                obs["H_obs"][m, n] = sit[n].mean()
                obs["sH"][m, n] = 0.3 + 0.05 * n
                obs["nH"][m, n] = sit[n].size
    obs["nh"] = LAT > 0
    return obs


def model_read(y, m, f):
    """What the reader must return: float32-rounded values, 0 at the ice-free (fill) nodes."""
    return np.float32(f(y, m)).astype(np.float64)


def write_fortran(run, y, var, f):
    nrec = MODEL_MONTHS[y]
    with netCDF4.Dataset(run / f"{var}.fesom.{y}.nc", "w") as ds:
        ds.createDimension("nod2", NOD)
        ds.createDimension("time", None)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"seconds since {y}-01-01 0:0:0"
        v = ds.createVariable(var, "f4", ("time", "nod2"), fill_value=FILL)
        v.set_auto_maskandscale(False)
        for i in range(nrec):
            nxt = datetime.datetime(y + (i == 11), i % 12 + 2 if i < 11 else 1, 1)
            t[i] = (nxt - datetime.datetime(y, 1, 1)).total_seconds() - 1800.0
            a = model_a(y, i)
            v[i] = np.where(a > 0, np.float32(f(y, i)), FILL)


@pytest.fixture()
def world(tmp_path):
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    np.save(mesh / "geo_coord_nod2D.npy", np.radians(np.stack([LON, LAT], axis=1)))
    np.save(mesh / "ulevels_nod2D.npy", np.ones(NOD, dtype=np.int32))
    np.save(mesh / "nlevels_nod2D.npy", np.full(NOD, 10, dtype=np.int32))
    np.save(mesh / "area.npy", np.stack([AREA, 0.9 * AREA], axis=1))
    run = tmp_path / "fortran"
    run.mkdir()
    for y in MODEL_MONTHS:
        write_fortran(run, y, "a_ice", model_a)
        write_fortran(run, y, "m_ice", model_h)
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    for y in (2001, 2002, 2003):
        with open(obs_dir / f"obs_{y}.npz", "xb") as fh:
            np.savez(fh, **build_obs(y))
    return {"mesh": mesh, "run": run, "obs": obs_dir, "tmp": tmp_path}


# --- reference values ----------------------------------------------------------------------------
def j_reference(y):
    """Independent loop: J per term over the months the model holds."""
    obs = build_obs(y)
    J = {"J_C_NH": 0.0, "J_C_SH": 0.0, "J_H_NH": 0.0}
    for m in range(MODEL_MONTHS[y]):
        C = model_read(y, m, model_a)
        H = model_read(y, m, model_h)
        for n in range(NOD):
            if obs["nC"][m, n]:
                w = obs["nC"][m, n] * 625e6 / 1e10
                J["J_C_NH" if LAT[n] > 0 else "J_C_SH"] += (
                    0.5 * w * (C[n] - obs["C_obs"][m, n]) ** 2 / obs["sC"][m, n] ** 2)
            if obs["nH"][m, n]:
                w = obs["nH"][m, n] * 625e6 / 1e10
                J["J_H_NH"] += 0.5 * w * (H[n] - obs["H_obs"][m, n]) ** 2 / obs["sH"][m, n] ** 2
    return J


def iiee_reference(y, m, north):
    conc, _ = cells(y, m)
    C = model_read(y, m, model_a)
    over = under = 0
    for n in range(NOD):
        if (LAT[n] > 0) != north:
            continue
        for c in conc[n]:
            over += int(C[n] > 0.15 and not c > 0.15)
            under += int(c > 0.15 and not C[n] > 0.15)
    return over * 625e6 / 1e12, under * 625e6 / 1e12


# Frozen once (2026-10-10, scratch script freeze_score_refs.py) with paper_jax/scripts/common.py's hemisphere_ice_area,
# hemisphere_ice_extent (model, the fill decoded to NaN as xarray does) and grid_ice_area (the
# cells, percent, 625 km^2) on this synthetic data; [m^2] -> 10^6 km^2.
FROZEN = {
    # (year, month 1-12): (area_nh, area_sh, extent_nh, extent_sh,
    #                      obs_area_nh, obs_extent_nh, obs_area_sh, obs_extent_sh)
    (2001, 1): (0.0019499999955296516, 0.0024000000953674316, 0.0035, 0.0055,
                0.000625, 0.000625, 0.000875, 0.001875),
    (2001, 8): (0.0029499999806284903, 0.0026500000394880773, 0.005, 0.003,
                0.0010625, 0.00125, 0.000625, 0.000625),
    (2002, 3): (0.002450000021606684, 0.002949999988079071, 0.0045, 0.0055,
                0.0005, 0.000625, 0.001, 0.00125),
    (2003, 2): (0.0019500000178813933, 0.0007500000298023224, 0.003, 0.0025,
                0.0001875, 0.000625, 0.0013125, 0.001875),
}


def _run(world, out_name="out", run=None, extra=()):
    return SC.main(["--run", str(run or world["run"]), "--obs-dir", str(world["obs"]),
                    "--years", "2000-2003", "--out", str(world["tmp"] / out_name),
                    "--mesh-dir", str(world["mesh"]), *extra])


def _load(world, out_name="out"):
    return json.loads((world["tmp"] / out_name / "score.json").read_text())


# --- tests ---------------------------------------------------------------------------------------
def test_fortran_reader_maps_records_to_months_and_reads_fill_as_no_ice(world):
    C, present = SC.read_fortran_var(world["run"] / "a_ice.fesom.2003.nc", "a_ice", 2003, NOD)
    np.testing.assert_array_equal(present, [True] * 3 + [False] * 9)
    for m in range(3):
        np.testing.assert_array_equal(C[m], model_read(2003, m, model_a))
    assert np.isnan(C[3:]).all()
    a = model_a(2003, 1)
    assert (a == 0).any() and (C[1][a == 0] == 0.0).all()     # fill -> 0, not NaN, not dropped
    H, _ = SC.read_fortran_var(world["run"] / "m_ice.fesom.2003.nc", "m_ice", 2003, NOD)
    assert (H[1][a == 0] == 0.0).all()
    with netCDF4.Dataset(world["run"] / "a_ice.fesom.2003.nc") as ds:     # really the fill
        ds.set_auto_maskandscale(False)
        assert (ds["a_ice"][1][a == 0] == FILL).all()
    with pytest.raises(ValueError, match="stamped"):           # file of another year
        SC.read_fortran_var(world["run"] / "a_ice.fesom.2003.nc", "a_ice", 2004, NOD)


def test_score_j_per_year_matches_the_independent_loop(world):
    assert _run(world, extra=("--no-figures",)) == 0
    r = _load(world)
    assert r["scored_years"] == [2001, 2002, 2003]
    assert r["missing_model_years"] == [2000]
    for y in (2001, 2002, 2003):
        ref = j_reference(y)
        for t, v in ref.items():
            assert v > 0 or t == "J_H_NH"
            np.testing.assert_allclose(r["J"][str(y)][t], v, rtol=1e-12, err_msg=f"{y} {t}")
    np.testing.assert_allclose(r["total"]["J_data"],
                               sum(sum(j_reference(y).values()) for y in (2001, 2002, 2003)),
                               rtol=1e-12)
    for k in ("N_C_NH", "Nact_C_NH", "Nact_C_SH", "Nact_H_NH"):
        np.testing.assert_allclose(r["total"][k], sum(r["J"][str(y)][k] for y in (2001, 2002, 2003)),
                                   rtol=1e-14)
    assert 0 < r["total"]["Nact_C_NH"] <= r["total"]["N_C_NH"]
    d = r["decades"]["2000s"]
    assert d["n_years"] == 3
    np.testing.assert_allclose(d["NH"], d["J_C_NH"] + d["J_H_NH"], rtol=1e-14)


def test_area_extent_match_the_frozen_reference_numbers(world):
    assert _run(world, extra=("--no-figures",)) == 0
    rows = {(r["year"], r["month"]): r for r in _load(world)["monthly"]}
    assert len(rows) == 12 + 12 + 3
    for (y, mo), ref in FROZEN.items():
        r = rows[(y, mo)]
        got = (r["nh"]["area_model"], r["sh"]["area_model"], r["nh"]["extent_model"],
               r["sh"]["extent_model"], r["nh"]["area_obs"], r["nh"]["extent_obs"],
               r["sh"]["area_obs"], r["sh"]["extent_obs"])
        np.testing.assert_allclose(got, ref, rtol=1e-12, err_msg=f"{y}-{mo}")


def test_gaps_iiee_and_trends(world):
    assert _run(world) == 0
    r = _load(world)
    rows = {(x["year"], x["month"]): x for x in r["monthly"]}
    # an observation gap (2002-12, 2003-03) keeps the model numbers and has no obs numbers
    for key in ((2002, 12), (2003, 3)):
        assert rows[key]["nh"]["area_obs"] is None and rows[key]["nh"]["iiee"] is None
        assert rows[key]["nh"]["area_model"] > 0
    assert r["J"]["2002"]["J_C_NH_month"][11] == 0.0
    for (y, mo) in ((2001, 1), (2001, 7), (2002, 3), (2003, 2)):
        for hem, north in (("nh", True), ("sh", False)):
            over, under = iiee_reference(y, mo - 1, north)
            np.testing.assert_allclose(rows[(y, mo)][hem]["iiee_over"], over, atol=1e-15)
            np.testing.assert_allclose(rows[(y, mo)][hem]["iiee_under"], under, atol=1e-15)
            np.testing.assert_allclose(rows[(y, mo)][hem]["iiee"], over + under, atol=1e-15)
    s = r["nh_march_extent"]
    assert s["years"] == [2001, 2002, 2003]
    np.testing.assert_allclose(s["trend_model_per_decade"],
                               np.polyfit([2001, 2002, 2003], s["model"], 1)[0] * 10, rtol=1e-10)
    assert s["obs"][2] is None                                # 2003-03 not observed
    np.testing.assert_allclose(s["trend_obs_per_decade"], (s["obs"][1] - s["obs"][0]) * 10,
                               rtol=1e-10)
    np.testing.assert_allclose(
        s["rmse"], np.sqrt(np.mean((np.array(s["model"][:2]) - np.array(s["obs"][:2])) ** 2)),
        rtol=1e-12)
    assert r["nh_september_extent"]["years"] == [2001, 2002]   # 2003 ends in March
    c = r["sh_seasonal_cycle"]
    assert len(c["extent_model_clim"]) == 12 and c["rmse_extent"] is not None
    w = {x["winter"]: x for x in r["cci_winters"]}
    assert set(w) == {"2000/2001", "2001/2002", "2002/2003"}
    assert w["2002/2003"]["n_months"] == 2 + 2             # Oct-Nov 2002 (Dec not observed), Jan-Feb 2003
    assert set(r["figures"]) == {"extent_area_monthly.png", "nh_mar_sep_extent.png",
                                 "J_per_year.png"}


def test_jax_fields_give_the_same_score_and_existing_output_is_refused(world):
    jax_dir = world["tmp"] / "jax"
    jax_dir.mkdir()
    for y, nm in MODEL_MONTHS.items():
        C = np.zeros((12, NOD))
        H = np.zeros((12, NOD))
        n = np.zeros(12)
        for m in range(nm):
            C[m] = model_read(y, m, model_a)
            H[m] = model_read(y, m, model_h)
            n[m] = 48
        np.savez(jax_dir / f"fields_{y}.npz", C_mean=C, H_mean=H, n_mon=n, year=y)
    assert _run(world, "out_f", extra=("--no-figures",)) == 0
    assert _run(world, "out_j", run=jax_dir, extra=("--no-figures",)) == 0
    a, b = _load(world, "out_f"), _load(world, "out_j")
    assert a["J"] == b["J"] and a["monthly"] == b["monthly"]
    assert b["provenance"]["kind"] == "jax"
    before = (world["tmp"] / "out_f" / "score.json").read_bytes()
    assert _run(world, "out_f", extra=("--no-figures",)) == 3
    assert (world["tmp"] / "out_f" / "score.json").read_bytes() == before


def test_jax_reader_refuses_missing_keys(world):
    d = world["tmp"] / "bad"
    d.mkdir()
    np.savez(d / "fields.npz", C_mean=np.zeros((12, NOD)), year=2001)
    with pytest.raises(KeyError, match="H_mean"):
        SC.read_jax_fields(d, NOD)


def test_trend_and_rmse_helpers():
    np.testing.assert_allclose(SC.trend_per_decade([1980, 1990, 2000], [10.0, 9.0, 8.0]), -1.0)
    assert SC.trend_per_decade([1980, 1990], [1.0, np.nan]) is None
    np.testing.assert_allclose(SC.rmse([1.0, 2.0, np.nan], [2.0, 4.0, 1.0]), np.sqrt(2.5))
    assert SC.parse_years("1980-1982,1990") == [1980, 1981, 1982, 1990]
