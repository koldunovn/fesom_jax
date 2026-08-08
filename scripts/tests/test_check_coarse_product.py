"""Tests for the coarse-product checker (`scripts/nn_eddy/check_coarse_product.py`).

A checker that only ever runs on good data is indistinguishable from one that returns 0
unconditionally. Each test here builds a file with exactly one defect and asserts the checker
finds THAT defect — including the one it was written for, FESOM's `9.96921e+36` fill leaking
into cells marked as having data, which passed a "finite" test and a surface map on 2026-08-08.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import netCDF4
import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]

N_NODE, N_LEV, N_REC = 40, 5, 2
FIELDS = ("temp", "salt", "u", "v", "uT_eddy", "vT_eddy", "uS_eddy", "vS_eddy")
GOOD = {"temp": 8.0, "salt": 34.5, "u": 0.1, "v": -0.05,
        "uT_eddy": 0.02, "vT_eddy": -0.01, "uS_eddy": 0.03, "vS_eddy": -0.02}


@pytest.fixture(scope="module")
def mod():
    path = _ROOT / "scripts" / "nn_eddy" / "check_coarse_product.py"
    spec = importlib.util.spec_from_file_location("check_coarse_product", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(path, *, mutate=None, dry_levels=1):
    """A well-formed product, optionally mutated by ``mutate(vars_dict, area)``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    area = np.ones((N_REC, N_NODE, N_LEV))
    area[:, :, N_LEV - dry_levels:] = 0.0          # the deepest level(s) are no-data
    rng = np.random.default_rng(0)
    data = {f: np.where(area > 0, GOOD[f] + 0.01 * rng.normal(size=area.shape), 0.0)
            for f in FIELDS}
    if mutate is not None:
        mutate(data, area)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", N_REC)
        ds.createDimension("nod2", N_NODE)
        ds.createDimension("nz1", N_LEV)
        ds.source_month = "1958_01"
        ds.description = "test file; cell_area == 0 marks NO DATA"
        ds.createVariable("time", "f8", ("time",))[:] = [431880.0, 863880.0]
        for f in FIELDS:
            ds.createVariable(f, "f4", ("time", "nod2", "nz1"))[:] = data[f].astype(np.float32)
        ds.createVariable("cell_area", "f4", ("time", "nod2", "nz1"))[:] = area.astype(np.float32)
    return path


def _run(mod, path, capsys):
    # main() parses argv, so drive it that way — it is the entry point users invoke
    argv = sys.argv
    sys.argv = ["check_coarse_product.py", str(path), "--teacher", str(path.parent)]
    try:
        rc = mod.main()
    finally:
        sys.argv = argv
    return rc, capsys.readouterr().out


def test_a_clean_file_passes(mod, tmp_path, capsys):
    rc, out = _run(mod, _write(tmp_path / "ok.nc"), capsys)
    assert rc == 0, out
    assert "coarse product OK" in out


def test_the_fill_value_leak_is_caught(mod, tmp_path, capsys):
    """The defect this checker exists for. 9.96921e+36 is FINITE, so a finiteness test passes;
    only a physical bound catches it. Put it below the surface, exactly as the real bug did, so
    the test also proves the checker does not merely look at level 0."""
    def mutate(data, area):
        data["u"][0, :10, 2] = 7.586e36
        data["uT_eddy"][0, :10, 2] = -7.579e36

    rc, out = _run(mod, _write(tmp_path / "fill.nc", mutate=mutate), capsys)
    assert rc == 1
    assert "u: 10 values outside" in out and "uT_eddy: 10 values outside" in out
    assert "9.96921e+36" in out, "the message should name the likely cause"


def test_a_nonzero_value_in_a_no_data_cell_is_caught(mod, tmp_path, capsys):
    """`cell_area == 0` means no measurement. A plausible-looking number there would enter a
    loss as though it were data."""
    def mutate(data, area):
        data["temp"][0, 5, N_LEV - 1] = 4.0        # inside bounds, but there is no data here

    rc, out = _run(mod, _write(tmp_path / "dry.nc", mutate=mutate), capsys)
    assert rc == 1
    assert "where cell_area == 0" in out


def test_a_dry_fraction_that_varies_between_records_is_caught(mod, tmp_path, capsys):
    """Bathymetry does not change within a month, so a per-record difference means a mask is
    being built from something that does."""
    def mutate(data, area):
        area[1, :, 1] = 0.0
        for f in FIELDS:
            data[f][1, :, 1] = 0.0

    rc, out = _run(mod, _write(tmp_path / "frac.nc", mutate=mutate), capsys)
    assert rc == 1
    assert "dry fraction varies" in out


def test_identically_zero_eddy_fluxes_are_caught(mod, tmp_path, capsys):
    """A pipeline whose covariance cancelled everywhere would produce a conservative,
    well-formed, entirely useless file — which no bound or conservation check would flag."""
    def mutate(data, area):
        for f in ("uT_eddy", "vT_eddy", "uS_eddy", "vS_eddy"):
            data[f][:] = 0.0

    rc, out = _run(mod, _write(tmp_path / "zero.nc", mutate=mutate), capsys)
    assert rc == 1
    assert "identically zero" in out


def test_a_non_finite_value_is_caught(mod, tmp_path, capsys):
    def mutate(data, area):
        data["salt"][0, 0, 0] = np.nan

    rc, out = _run(mod, _write(tmp_path / "nan.nc", mutate=mutate), capsys)
    assert rc == 1
    assert "non-finite" in out


def test_a_missing_field_is_caught(mod, tmp_path, capsys):
    path = _write(tmp_path / "short.nc")
    # rewrite without u — a pipeline change that dropped a variable must not pass silently
    with netCDF4.Dataset(path) as src:
        keep = {k: np.asarray(v[:]) for k, v in src.variables.items() if k != "u"}
    path.unlink()
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", N_REC)
        ds.createDimension("nod2", N_NODE)
        ds.createDimension("nz1", N_LEV)
        ds.source_month = "1958_01"
        for k, v in keep.items():
            dims = ("time",) if v.ndim == 1 else ("time", "nod2", "nz1")
            ds.createVariable(k, "f8" if v.ndim == 1 else "f4", dims)[:] = v
    rc, out = _run(mod, path, capsys)
    assert rc == 1
    assert "u: missing from the file" in out
