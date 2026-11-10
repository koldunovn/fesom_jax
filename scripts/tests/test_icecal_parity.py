"""``scripts/icecal/parity.py`` on synthetic Fortran-like files and a synthetic JAX ``fields.npz``
(six nodes, one year): JAX equal to the Fortran values gives exactly zero differences in every
month (the record -> month mapping and the fill value read as ice-free), a known perturbation gives
the differences of an independent per-node loop, only months present in both are compared, and an
existing output directory is refused. ``scripts`` group."""

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
import parity as PA  # noqa: E402

LAT = np.array([70.0, 75.0, 80.0, 85.0, -65.0, -70.0])
LON = np.array([0.0, 10.0, 20.0, 30.0, 0.0, 10.0])
AREA = np.array([1.0e9, 2.0e9, 1.5e9, 0.5e9, 3.0e9, 2.5e9])
NOD = LAT.size
FILL = np.float32(9.96921e36)
YEAR = 2013
JAX_MISSING = 6                    # July has no JAX steps (n_mon 0): not compared


def fortran_a(m):
    """Fortran concentration of month m (0-based); exact zeros are ice-free (fill) nodes. Every
    month differs, so a shifted record -> month mapping cannot give zero differences."""
    return ((np.arange(NOD) * 3 + m * 7) % 10) / 10.0


def fortran_h(m):
    a = fortran_a(m)
    return np.where(a > 0, 2.0 * a + 0.1 * np.arange(NOD) + 0.05 * m, 0.0)


def as_read(x):
    """What the reader returns: float32-rounded values."""
    return np.float32(x).astype(np.float64)


def write_fortran(run, var, f):
    with netCDF4.Dataset(run / f"{var}.fesom.{YEAR}.nc", "w") as ds:
        ds.createDimension("nod2", NOD)
        ds.createDimension("time", None)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"seconds since {YEAR}-01-01 0:0:0"
        v = ds.createVariable(var, "f4", ("time", "nod2"), fill_value=FILL)
        v.set_auto_maskandscale(False)
        for i in range(12):
            nxt = datetime.datetime(YEAR + (i == 11), i % 12 + 2 if i < 11 else 1, 1)
            t[i] = (nxt - datetime.datetime(YEAR, 1, 1)).total_seconds() - 1800.0
            v[i] = np.where(fortran_a(i) > 0, np.float32(f(i)), FILL)


def perturbation(m):
    """JAX minus Fortran: concentration and thickness offsets (C stays within [0, 1])."""
    dC = np.where(np.arange(NOD) % 2 == 0, 0.05, -0.02) * (fortran_a(m) > 0.1)
    dH = 0.1 * np.sin(np.arange(NOD) + m)
    return dC, dH


def write_jax(d, perturbed):
    C = np.zeros((12, NOD))
    H = np.zeros((12, NOD))
    for m in range(12):
        C[m] = as_read(fortran_a(m))
        H[m] = as_read(fortran_h(m))
        if perturbed:
            dC, dH = perturbation(m)
            C[m] += dC
            H[m] += dH
    n_mon = np.full(12, 48)
    n_mon[JAX_MISSING] = 0
    C[JAX_MISSING] = H[JAX_MISSING] = 0.0
    d.mkdir()
    with open(d / "fields.npz", "xb") as fh:
        np.savez(fh, C_mean=C, H_mean=H, n_mon=n_mon, year=YEAR)
    return C, H


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
    write_fortran(run, "a_ice", fortran_a)
    write_fortran(run, "m_ice", fortran_h)
    return {"mesh": mesh, "run": run, "tmp": tmp_path}


def run_parity(world, jax_dir, name):
    out = world["tmp"] / name
    rc = PA.main(["--jax", str(jax_dir), "--fortran", str(world["run"]), "--year", str(YEAR),
                  "--out", str(out), "--mesh-dir", str(world["mesh"])])
    assert rc == 0
    with open(out / "parity.json") as f:
        return json.load(f)


def test_identical_fields_give_zero_differences_in_every_month(world):
    jd = world["tmp"] / "jax_same"
    write_jax(jd, perturbed=False)
    res = run_parity(world, jd, "out_same")
    assert res["months"] == [m + 1 for m in range(12) if m != JAX_MISSING]
    for hem in ("NH", "SH"):
        assert len(res["monthly"][hem]) == 11
        for r in res["monthly"][hem]:
            for q in ("area_diff", "extent_diff", "volume_diff", "rms_C", "rms_H",
                      "mean_C", "mean_H"):
                assert r[q] == 0.0, (hem, r["month"], q, r[q])
            assert r["area_fortran"] > 0.0          # the fill value did not leak in as ice


def test_differences_match_an_independent_loop(world):
    jd = world["tmp"] / "jax_pert"
    C, H = write_jax(jd, perturbed=True)
    res = run_parity(world, jd, "out_pert")
    for hem, sel in (("NH", LAT > 0), ("SH", LAT < 0)):
        rows = {r["month"]: r for r in res["monthly"][hem]}
        for m in range(12):
            if m == JAX_MISSING:
                assert m + 1 not in rows
                continue
            Cf, Hf = as_read(fortran_a(m)), as_read(fortran_h(m))
            area_j = area_f = ext_j = ext_f = vol_j = vol_f = 0.0
            dom_area = sC = sC2 = sH = sH2 = 0.0
            for n in range(NOD):
                if not sel[n]:
                    continue
                area_j += min(max(C[m, n], 0.0), 1.0) * AREA[n]
                area_f += min(max(Cf[n], 0.0), 1.0) * AREA[n]
                ext_j += AREA[n] if C[m, n] > 0.15 else 0.0
                ext_f += AREA[n] if Cf[n] > 0.15 else 0.0
                vol_j += H[m, n] * AREA[n]
                vol_f += Hf[n] * AREA[n]
                if C[m, n] > 0 or Cf[n] > 0:
                    dom_area += AREA[n]
                    sC += AREA[n] * (C[m, n] - Cf[n])
                    sC2 += AREA[n] * (C[m, n] - Cf[n]) ** 2
                    sH += AREA[n] * (H[m, n] - Hf[n])
                    sH2 += AREA[n] * (H[m, n] - Hf[n]) ** 2
            r = rows[m + 1]
            np.testing.assert_allclose(r["area_jax"], area_j / 1e12, rtol=1e-13)
            np.testing.assert_allclose(r["area_diff"], (area_j - area_f) / 1e12, rtol=1e-12,
                                       atol=1e-18)
            np.testing.assert_allclose(r["extent_diff"], (ext_j - ext_f) / 1e12, atol=1e-18)
            np.testing.assert_allclose(r["volume_diff"], (vol_j - vol_f) / 1e12, rtol=1e-12,
                                       atol=1e-18)
            np.testing.assert_allclose(r["ice_domain_area"], dom_area / 1e12, rtol=1e-13)
            np.testing.assert_allclose(r["mean_C"], sC / dom_area, rtol=1e-12, atol=1e-16)
            np.testing.assert_allclose(r["rms_C"], np.sqrt(sC2 / dom_area), rtol=1e-12)
            np.testing.assert_allclose(r["mean_H"], sH / dom_area, rtol=1e-12, atol=1e-16)
            np.testing.assert_allclose(r["rms_H"], np.sqrt(sH2 / dom_area), rtol=1e-12)
        # the perturbation is non-trivial in this hemisphere
        assert max(abs(r["rms_H"]) for r in rows.values()) > 0.01
        yr = res["year"][hem]["volume_diff"]
        v = np.array([r["volume_diff"] for r in rows.values()])
        np.testing.assert_allclose(yr["rms"], np.sqrt(np.mean(v ** 2)), rtol=1e-12)
        np.testing.assert_allclose(yr["max_abs"], np.max(np.abs(v)), rtol=1e-13)


def test_existing_output_directory_is_refused(world):
    jd = world["tmp"] / "jax_same"
    write_jax(jd, perturbed=False)
    out = world["tmp"] / "taken"
    out.mkdir()
    rc = PA.main(["--jax", str(jd), "--fortran", str(world["run"]), "--year", str(YEAR),
                  "--out", str(out), "--mesh-dir", str(world["mesh"])])
    assert rc == 3
    assert list(out.iterdir()) == []


def test_missing_year_is_reported(world):
    jd = world["tmp"] / "jax_same"
    write_jax(jd, perturbed=False)
    rc = PA.main(["--jax", str(jd), "--fortran", str(world["run"]), "--year", str(YEAR + 1),
                  "--out", str(world["tmp"] / "o2"), "--mesh-dir", str(world["mesh"])])
    assert rc == 2
