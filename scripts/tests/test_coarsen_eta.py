"""Tests for the eta side-product (`scripts/nn_eddy/coarsen_eta.py`, NN-GM child plan M2).

The aggregation arithmetic is already covered (`fesom_jax/tests/test_coarsen_remap.py`), and the
3-D script's own tests cover the fill-value and layer-convention traps. What is left is what only
THIS script can get wrong, and each item is a real failure mode:

* **the hole rule must match the 3-D product's**, because ``nn_data`` will join the two on the
  mark index and mask both with one hole mask. A cell that is a hole in one and not the other
  contributes a difference against zero — a fabricated ~1 m SSH error, in the loss, silently;
* **the fill value.** SSH is 2-D, so it needs its own reader; ``np.asarray`` on a netCDF4
  MaskedArray hands 9.96921e+36 back as a number, and one such value poisons a whole coarse cell;
* **the time axis is READ, not computed.** Records sit one fine-dt short of the round 5-day mark
  and that offset changes with the teacher's dt ramp, so a computed time would be wrong for part
  of the year and right for the rest — the worst kind of wrong;
* **refusing to overwrite.** The whole reason eta is a side-file is that the verified 3-D products
  must not be rewritten; a script that clobbers its own output has not learned that lesson either.

Everything runs on a mesh of eight nodes; nothing reads the real 16 GB export.
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
_SCRIPTS = _ROOT / "scripts" / "nn_eddy"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))       # coarsen_eta imports coarsen_stream by name

from fesom_jax import coarsen                                                 # noqa: E402

sys.path.insert(0, str(_ROOT / "scripts" / "tests"))
from test_coarsen_stream import (N_COARSE, N_FINE, _fake_coarse_mesh,         # noqa: E402
                                 _fake_fine_mesh)

FILL = 9.96921e36


def _load_module():
    path = _SCRIPTS / "coarsen_eta.py"
    spec = importlib.util.spec_from_file_location("coarsen_eta", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ce():
    return _load_module()


@pytest.fixture
def meshes(tmp_path):
    return _fake_fine_mesh(tmp_path / "fine"), _fake_coarse_mesh(tmp_path / "coarse")


def _fake_teacher(dirpath: pathlib.Path, month: str, eta, times):
    """A miniature ``ssh_snap.fesom.<month>.nc`` shaped exactly like the real one."""
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / f"ssh_snap.fesom.{month}.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("nod2", eta.shape[1])
        tv = ds.createVariable("time", "f8", ("time",))
        tv[:] = np.asarray(times)
        v = ds.createVariable("ssh_snap", "f8", ("time", "nod2"), fill_value=FILL)
        v[:] = eta
    return dirpath


# ---------------------------------------------------------------------------

def test_aggregation_is_the_area_weighted_cell_mean(ce, meshes, tmp_path):
    """The coarse value is the area-weighted mean of its cell's fine values, and the weight is
    the summed wet area — the same identity the 3-D path uses, checked by hand."""
    fine_dir, coarse_dir = meshes
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    fine = ce.FineMesh(fine_dir)
    cell = np.asarray(rmap.fine_cell)

    eta_fine = np.where(cell == 0, np.arange(N_FINE, dtype=float), 7.0)
    teacher = _fake_teacher(tmp_path / "t", "1958_01", eta_fine[None, :], [431880.0])
    times, eta, weight = ce.process_month("1958_01", rmap, fine, teacher)

    assert len(times) == 1 and times[0] == 431880.0        # read, not computed
    assert eta.shape == (1, N_COARSE)
    for c in range(N_COARSE):
        sel = cell == c
        np.testing.assert_allclose(eta[0, c], eta_fine[sel].mean())
        np.testing.assert_allclose(weight[0, c], float(sel.sum()))   # unit areas in the fixture


def test_the_netcdf_fill_value_never_enters_the_mean(ce, meshes, tmp_path):
    """A dry fine node written as FESOM's 9.96921e+36 must be excluded by the surface mask, not
    averaged in. Without the mask the affected coarse cell comes out at ~1e36; the assertion is
    that it comes out at the mean of the WET nodes only."""
    fine_dir = _fake_fine_mesh(tmp_path / "fine2", nlev=[4, 4, 4, 1, 4, 4, 4, 4])
    coarse_dir = _fake_coarse_mesh(tmp_path / "coarse2")
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    fine = ce.FineMesh(fine_dir)
    cell = np.asarray(rmap.fine_cell)

    eta_fine = np.full(N_FINE, 2.0)
    dry = np.where(np.asarray(fine.layer_mask(0)) == 0.0)[0]
    assert dry.size == 1, "the fixture must have exactly one surface-dry node for this test"
    eta_fine[dry] = FILL
    teacher = _fake_teacher(tmp_path / "t2", "1958_02", eta_fine[None, :], [0.0])
    _, eta, weight = ce.process_month("1958_02", rmap, fine, teacher)

    assert np.isfinite(eta).all()
    np.testing.assert_allclose(eta[0], 2.0)                # the fill contributed nothing
    c_dry = int(cell[dry[0]])
    assert weight[0, c_dry] == float((cell == c_dry).sum()) - 1.0   # one node fewer


def test_a_cell_with_no_wet_nodes_is_a_hole_not_a_zero(ce, meshes, tmp_path):
    """``cell_area == 0`` marks NO DATA. The check must see it, report it, and the hole pattern
    must be constant across records (the surface mask is static)."""
    # kill the whole second cluster at the surface => coarse node 1 has nothing under it
    fine_dir = _fake_fine_mesh(tmp_path / "fine3", nlev=[4, 4, 4, 4, 1, 1, 1, 1])
    coarse_dir = _fake_coarse_mesh(tmp_path / "coarse3")
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    fine = ce.FineMesh(fine_dir)

    eta_fine = np.full((2, N_FINE), 1.5)
    teacher = _fake_teacher(tmp_path / "t3", "1958_03", eta_fine, [0.0, 432000.0])
    _, eta, weight = ce.process_month("1958_03", rmap, fine, teacher)

    holes = weight[0] == 0.0
    assert holes.sum() == 1, f"expected exactly one hole, got {holes.sum()}"
    assert (weight[0] == 0.0).tolist() == (weight[1] == 0.0).tolist()
    assert ce.check(eta, weight, None) == 0            # a hole is not a failure, it is data


def test_check_fails_on_an_unphysical_field(ce, meshes, tmp_path):
    """|eta| beyond the bar is a bug in the pipeline, not an ocean, and must fail the check
    rather than be written."""
    eta = np.array([[0.5, 50.0]])
    weight = np.array([[1.0, 1.0]])
    assert ce.check(eta, weight, None) > 0


def test_check_fails_when_the_hole_pattern_differs_from_the_3d_product(ce, tmp_path):
    """The join-safety assertion. A 3-D product whose surface ``cell_area`` disagrees about even
    one cell must fail: downstream both files are masked with ONE hole mask."""
    prod = tmp_path / "coarse_ng5_1958_01.nc"
    with netCDF4.Dataset(prod, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("nod2", 3)
        ds.createDimension("nz1", 2)
        ds.createVariable("time", "f8", ("time",))[:] = [431880.0]
        v = ds.createVariable("cell_area", "f4", ("time", "nod2", "nz1"))
        v[:] = np.array([[[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]]])      # cell 1 is a hole
    eta = np.array([[0.1, 0.2, 0.3]])
    good = np.array([[1.0, 0.0, 2.0]])
    bad = np.array([[1.0, 3.0, 2.0]])                                 # cell 1 is NOT a hole
    assert ce.check(eta, good, prod) == 0
    assert ce.check(eta, bad, prod) > 0


def test_write_output_refuses_to_overwrite(ce, tmp_path):
    """The reason eta is a side-file at all: nothing in this pipeline may clobber a product that
    something else has already verified."""
    path = tmp_path / "coarse_eta_ng5_1958_01.nc"
    eta = np.array([[0.1, 0.2]])
    weight = np.array([[1.0, 2.0]])
    ce.write_output(path, "1958_01", [431880.0], eta, weight)
    assert path.exists()
    with netCDF4.Dataset(path) as ds:
        np.testing.assert_allclose(np.asarray(ds.variables["eta"][:]), eta, rtol=1e-6)
        # eta_mean is the area-weighted mean: (1*0.1 + 2*0.2)/3
        np.testing.assert_allclose(float(ds.variables["eta_mean"][0]), 0.5 / 3.0, rtol=1e-9)
    with pytest.raises(SystemExit, match="REFUSING to overwrite"):
        ce.write_output(path, "1958_01", [431880.0], eta, weight)
    assert not path.with_suffix(".tmp.nc").exists(), "a refused write left a temp file behind"
