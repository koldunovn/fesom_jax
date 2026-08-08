"""Tests for the shared teacher-data layer (`scripts/nn_eddy/nn_data.py`, child plan M2).

These functions decide WHICH CELLS COUNT in every loss this project will compute, offline and
online. Each test below corresponds to a way of getting that silently wrong:

* a mark index computed instead of rounded — right for part of 1958 and wrong for the rest,
  because the teacher's dt ramps from 120 to 240 s at month 03;
* a level slice taken on faith — the model carries 48 columns and the product 47, and the trim
  is only exact because the extra columns are globally dry;
* a hole treated as a zero — 51,178 student-wet cells have no teacher behind them, and a zero
  there is a fabricated observation of freezing fresh water that a fit would happily match;
* an eta join on record order rather than mark index — one missing month shifts a whole stream;
* a de-meaning that also shifts the holes, turning no-data into a plausible anomaly.

The real-product path is covered by the signal_eval regression re-run (the M2 gate: holes
exactly 51,178 and every spread/floor to 3 significant figures); this file is the unit half and
touches no /work data.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import netCDF4
import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_module():
    """Import the script by path: it lives under scripts/, which is not a package.

    ⚠️ The module must be registered in ``sys.modules`` BEFORE ``exec_module``. ``nn_data``
    defines a dataclass, and under ``from __future__ import annotations`` the dataclass
    machinery resolves its field annotations through ``sys.modules[cls.__module__]`` — which is
    ``None`` for a module that was executed but never registered, and the failure surfaces as an
    unrelated-looking ``AttributeError`` inside ``dataclasses``."""
    path = _ROOT / "scripts" / "nn_eddy" / "nn_data.py"
    spec = importlib.util.spec_from_file_location("nn_data", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nn_data"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def nd():
    return _load_module()


# ---------------------------------------------------------------------------
# marks
# ---------------------------------------------------------------------------

def test_mark_index_absorbs_both_teacher_timesteps(nd):
    """The offset from the round mark is one FINE dt, and the teacher ramps 120 → 240 s at
    month 03 of 1958. Both must land on the same mark, which is why this rounds rather than
    subtracting a known offset."""
    for k in (1, 2, 6, 73):
        exact = k * nd.MARK_SECONDS
        assert nd.mark_index(exact) == k
        assert nd.mark_index(exact - 120.0) == k          # early 1958 dt
        assert nd.mark_index(exact - 240.0) == k          # post-ramp dt
    # and it must not silently absorb a genuinely different mark
    assert nd.mark_index(nd.MARK_SECONDS * 1.5 + 1.0) == 2
    assert nd.mark_index(431880.0) == 1                   # the real recorded value


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

def test_level_trim_accepts_dry_padding(nd):
    wet = np.zeros((5, 48), bool)
    wet[:, :47] = True
    wet[:, 40:] = False                                    # ragged is fine
    wet[0, :47] = True
    sl = nd.model_to_teacher_levels(wet, 47)
    assert sl == slice(0, 47)


def test_level_trim_refuses_to_drop_a_wet_column(nd):
    """The assert is the whole point: a product with fewer levels than the model has WET ones
    means the two are not the same ocean, and trimming would quietly discard data."""
    wet = np.zeros((5, 48), bool)
    wet[:, :48] = True                                     # column 47 is wet
    with pytest.raises(AssertionError, match="not padding"):
        nd.model_to_teacher_levels(wet, 47)


# ---------------------------------------------------------------------------
# holes
# ---------------------------------------------------------------------------

def test_combined_mask_excludes_teacher_holes_and_counts_them(nd, capsys):
    wet_model = np.array([[True, True, True], [True, True, False]])
    wet_teacher = np.array([[True, False, True], [True, True, True]])
    mask, excl = nd.combined_mask(wet_model, wet_teacher)
    np.testing.assert_array_equal(mask, np.array([[True, False, True], [True, True, False]]))
    assert excl == 1                                       # model-wet, teacher-hole
    assert "excluded" in capsys.readouterr().out           # the count is REPORTED, not hidden


def test_combined_mask_rejects_a_shape_mismatch(nd):
    """A shape mismatch here means the level trim was skipped; broadcasting would produce a
    plausible mask of the wrong size."""
    with pytest.raises(AssertionError):
        nd.combined_mask(np.ones((5, 48), bool), np.ones((5, 47), bool))


# ---------------------------------------------------------------------------
# the eta join and its de-meaning
# ---------------------------------------------------------------------------

def _write_3d(path, times, n=3, nz=2):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("nod2", n)
        ds.createDimension("nz1", nz)
        ds.createVariable("time", "f8", ("time",))[:] = times
        for name in ("temp", "salt"):
            v = ds.createVariable(name, "f4", ("time", "nod2", "nz1"))
            v[:] = np.arange(len(times) * n * nz).reshape(len(times), n, nz)
        v = ds.createVariable("cell_area", "f4", ("time", "nod2", "nz1"))
        ca = np.ones((len(times), n, nz))
        ca[:, 1, :] = 0.0                                   # node 1 is a hole
        v[:] = ca


def _write_eta(path, times, n=3):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("nod2", n)
        ds.createVariable("time", "f8", ("time",))[:] = times
        ds.createVariable("eta", "f4", ("time", "nod2"))[:] = \
            np.arange(len(times) * n).reshape(len(times), n) * 0.1
        ca = np.ones((len(times), n))
        ca[:, 1] = 0.0
        ds.createVariable("cell_area", "f4", ("time", "nod2"))[:] = ca
        ds.createVariable("eta_mean", "f8", ("time",))[:] = np.zeros(len(times))


def test_eta_joins_on_the_mark_index_not_record_order(nd, tmp_path):
    """The 3-D and eta streams are written by different scripts. Here the eta side has only the
    SECOND month, so record 0 of the eta file is mark 3 — joining on record order would pair it
    with mark 1 and mislabel every SSH target by ten days."""
    M = nd.MARK_SECONDS
    _write_3d(tmp_path / "coarse_ng5_1958_01.nc", [M - 120, 2 * M - 120])
    _write_3d(tmp_path / "coarse_ng5_1958_02.nc", [3 * M - 240, 4 * M - 240])
    _write_eta(tmp_path / "coarse_eta_ng5_1958_02.nc", [3 * M - 240, 4 * M - 240])

    t = nd.assemble_teacher(tmp_path, 1958, with_eta=True, verbose=False)
    assert t.klist == [1, 2, 3, 4]
    assert sorted(t.eta) == [3, 4], "eta was joined on record order, not the mark index"
    assert t.has_eta
    # marks with no eta yet survive with T/S — a partial eta stream must not drop marks
    assert 1 in t.marks and 1 not in t.eta


def test_assemble_teacher_intersects_a_varying_hole_mask(nd, tmp_path):
    """A cell that is a hole in ANY record is a hole in all of them: the driver masks once, so a
    per-record mask would let a hole through on the records where it happened to be filled."""
    M = nd.MARK_SECONDS
    p = tmp_path / "coarse_ng5_1958_01.nc"
    _write_3d(p, [M - 120, 2 * M - 120])
    with netCDF4.Dataset(p, "r+") as ds:
        ca = np.asarray(ds.variables["cell_area"][:])
        ca[1, 2, :] = 0.0                                   # node 2 dries out in record 1 only
        ds.variables["cell_area"][:] = ca
    t = nd.assemble_teacher(tmp_path, 1958, verbose=False)
    assert not t.wet[2].any(), "the intersection let a per-record hole through"
    assert t.n_intersected > 0


def test_demean_removes_the_area_weighted_mean_and_leaves_holes_alone(nd):
    field = np.array([1.0, 99.0, 3.0])
    weight = np.array([1.0, 0.0, 3.0])                      # cell 1 is a hole
    out = nd.demean(field, weight)
    sel = weight > 0
    mean = (weight[sel] * field[sel]).sum() / weight[sel].sum()      # (1 + 9)/4 = 2.5
    np.testing.assert_allclose(mean, 2.5)
    np.testing.assert_allclose(out[sel], field[sel] - 2.5)
    assert out[1] == 99.0, "de-meaning touched a hole, inventing an anomaly where there is no data"
    # the de-meaned field has zero area-weighted mean, which is the property the loss needs
    np.testing.assert_allclose((weight[sel] * out[sel]).sum(), 0.0, atol=1e-12)


def test_demean_of_an_all_hole_field_is_a_noop(nd):
    field = np.array([1.0, 2.0])
    np.testing.assert_array_equal(nd.demean(field, np.zeros(2)), field)


# ---------------------------------------------------------------------------
# signal_eval must use these, not its own copies
# ---------------------------------------------------------------------------

def test_signal_eval_imports_the_shared_implementation(nd):
    """The point of the module: one implementation. If signal_eval ever grows its own copy
    again, these are no longer the same function and the recorded gate numbers stop meaning
    anything about the training path."""
    path = _ROOT / "scripts" / "nn_eddy" / "signal_eval.py"
    spec = importlib.util.spec_from_file_location("signal_eval_mod", path)
    se = importlib.util.module_from_spec(spec)
    sys.modules["signal_eval_mod"] = se
    sys.path.insert(0, str(_ROOT / "scripts" / "nn_eddy"))
    spec.loader.exec_module(se)
    import nn_data as shared          # the module signal_eval itself imported
    for name in ("mark_index", "model_to_teacher_levels", "assemble_teacher", "month_files"):
        assert getattr(se, name) is getattr(shared, name), \
            f"signal_eval.{name} is not nn_data.{name} — the implementations have diverged"
