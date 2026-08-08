"""Tests for the NG5→CORE2 streaming pipeline (`scripts/nn_eddy/coarsen_stream.py`).

The pipeline's arithmetic pieces are covered in `fesom_jax/tests/test_coarsen_remap.py`. What
is left here is what only the *script* can get wrong, and the list is short but every item on
it has a real failure mode:

* the eddy-flux identity — that ``agg(u·τ) − agg(u)·agg(τ)`` really is zero when there is no
  subgrid covariance and nonzero when there is. A sign error or a mismatched filter would still
  produce plausible-looking fields;
* the layer-validity convention. ``nlevels`` is an INTERFACE count, so the last valid layer is
  ``nlevels-2``. Off by one here silently adds a layer of sub-bottom values to every column;
* the map cache — reusing one built for a different mesh pair would be catastrophic and silent,
  so it must refuse rather than broadcast;
* end to end on a miniature fake teacher month, so the netCDF naming (``<field>_snap``), the
  dimension order (time, level, space) and the time axis are exercised as written rather than
  as remembered.

Everything runs on meshes of a few nodes; nothing reads the real 16 GB export.
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

import jax.numpy as jnp                                                       # noqa: E402

from fesom_jax import coarsen                                                 # noqa: E402
from fesom_jax.coarsen import RemapMap                                        # noqa: E402


def _load_module():
    """Import the script by path: it lives under scripts/, which is not a package."""
    path = _ROOT / "scripts" / "nn_eddy" / "coarsen_stream.py"
    spec = importlib.util.spec_from_file_location("coarsen_stream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cs():
    return _load_module()


# ---------------------------------------------------------------------------
# a miniature mesh pair: 2 coarse nodes, 8 fine nodes, 4 fine elements
# ---------------------------------------------------------------------------

N_FINE, N_COARSE, N_ELEM, NL = 8, 2, 4, 4      # NL interfaces => 3 layers


def _fake_fine_mesh(dirpath: pathlib.Path, *, nlev=None, enlev=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    # two clusters of four, one near 0E and one near 90E, so the coarse assignment is obvious
    lon = np.array([0.0, 0.5, -0.5, 0.2, 90.0, 90.5, 89.5, 90.2])
    lat = np.zeros(N_FINE)
    np.save(dirpath / "geo_coord_nod2D.npy",
            np.column_stack([np.radians(lon), np.radians(lat)]))
    np.save(dirpath / "area.npy", np.full((N_FINE, NL), 1.0))     # 2-D, as a real export is
    np.save(dirpath / "zbar.npy", np.array([0.0, -10.0, -30.0, -60.0]))
    # every element is a triangle inside one cluster, so collocation never mixes clusters
    np.save(dirpath / "elem_nodes.npy",
            np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32))
    np.save(dirpath / "elem_area.npy", np.ones(N_ELEM))
    np.save(dirpath / "ulevels_nod2D.npy", np.ones(N_FINE, dtype=np.int32))
    # nlevels is an INTERFACE count: 4 => layers 0,1,2 valid
    np.save(dirpath / "nlevels_nod2D.npy",
            (np.full(N_FINE, NL, dtype=np.int32) if nlev is None
             else np.asarray(nlev, dtype=np.int32)))
    # ELEMENTAL level counts, for u/v. ⚠️ These were missing from this fixture, which is
    # precisely why the suite could not catch the fill-value leak found on real NG5 output
    # (2026-08-08): with no elemental bathymetry there is no dry element, so there is nothing
    # for a wet node to inherit a fill value from.
    np.save(dirpath / "ulevels.npy", np.ones(N_ELEM, dtype=np.int32))
    np.save(dirpath / "nlevels.npy",
            (np.full(N_ELEM, NL, dtype=np.int32) if enlev is None
             else np.asarray(enlev, dtype=np.int32)))
    return dirpath


def _fake_coarse_mesh(dirpath: pathlib.Path):
    dirpath.mkdir(parents=True, exist_ok=True)
    np.save(dirpath / "geo_coord_nod2D.npy",
            np.column_stack([np.radians([0.0, 90.0]), np.radians([0.0, 0.0])]))
    np.save(dirpath / "area.npy", np.ones(N_COARSE))
    np.save(dirpath / "zbar.npy", np.array([0.0, -30.0, -60.0]))   # 2 coarse layers
    np.save(dirpath / "ulevels_nod2D.npy", np.ones(N_COARSE, dtype=np.int32))
    np.save(dirpath / "nlevels_nod2D.npy", np.full(N_COARSE, 3, dtype=np.int32))
    return dirpath


@pytest.fixture
def meshes(tmp_path):
    return _fake_fine_mesh(tmp_path / "fine"), _fake_coarse_mesh(tmp_path / "coarse")


# ---------------------------------------------------------------------------
# the eddy-flux identity
# ---------------------------------------------------------------------------

def test_no_subgrid_covariance_gives_exactly_zero_eddy_flux(cs, meshes):
    """If u and T are each uniform inside a coarse cell there is nothing for the closure to
    represent, and the diagnosed flux must be exactly 0 — not small. Anything else is a bias
    that Task 8 would fit and Task 9 would then be scored against."""
    fine_dir, coarse_dir = meshes
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    fine = cs.FineMesh(fine_dir)
    cell = np.asarray(rmap.fine_cell)

    T = jnp.asarray(np.where(cell == 0, 5.0, 12.0))
    S = jnp.asarray(np.where(cell == 0, 34.0, 35.0))
    # element values uniform per cluster => nodal u,v uniform per cluster too
    ue = jnp.asarray(np.array([2.0, 2.0, -1.0, -1.0]))
    ve = jnp.asarray(np.array([0.5, 0.5, 3.0, 3.0]))
    mask = jnp.ones(N_FINE)
    _, flux, w = cs.level_products(T, S, ue, ve, mask, rmap, fine)
    assert (np.asarray(w) > 0).all()
    for f, name in zip(flux, cs.FLUX_NAMES):
        assert np.abs(np.asarray(f)).max() < 1e-12, f"{name} is not zero: {np.asarray(f)}"


def test_a_covarying_pair_gives_the_flux_computed_by_hand(cs, meshes):
    """The magnitude and sign convention the offline target inherits, on a case small enough to
    do on paper.

    Cell 0 is fine nodes 0..3, all with area 1. Element u is (+1, −1, 0, 0) and elements 0 and 1
    are (0,1,2) and (0,2,3), so the collocated nodal u is (0, +1, 0, −1): node 1 sees only the
    +1 element, node 3 only the −1 one, and nodes 0 and 2 see both and average to zero. Give T
    the SAME pattern (0, +1, 0, −1). Then agg(u) = agg(T) = 0 while
    agg(u·T) = (0 + 1 + 0 + 1)/4 = 0.5, so u'T' = 0.5 — entirely subgrid, which is the point.

    ⚠️ The obvious choice of T = (+1, −1, +1, −1) gives exactly ZERO against this u, because the
    two patterns are orthogonal. That is a correct answer to a useless question, which is why
    this test also asserts the case is non-degenerate.
    """
    fine_dir, coarse_dir = meshes
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    fine = cs.FineMesh(fine_dir)
    cell = np.asarray(rmap.fine_cell)
    first = np.where(cell == 0)[0]
    np.testing.assert_array_equal(first, [0, 1, 2, 3])

    ue = jnp.asarray(np.array([1.0, -1.0, 0.0, 0.0]))
    u_node = np.asarray(coarsen.elem_to_node(ue, fine))
    np.testing.assert_allclose(u_node[first], [0.0, 1.0, 0.0, -1.0], atol=1e-15)

    T = jnp.asarray(np.where(np.arange(N_FINE) < 4, u_node, 0.0))   # same pattern as u
    _, flux, _ = cs.level_products(T, jnp.zeros(N_FINE), ue, jnp.zeros(N_ELEM),
                                   jnp.ones(N_FINE), rmap, fine)
    assert float(flux[0][0]) == pytest.approx(0.5, rel=1e-12), \
        f"u'T' in cell 0 is {float(flux[0][0])}, hand value 0.5"
    assert abs(float(flux[0][0])) > 1e-6, "the test case has no subgrid covariance to detect"

    # the generic formula, as an independent check on the hand arithmetic above
    a, sel = np.asarray(rmap.fine_area), cell == 0
    t = np.asarray(T)
    expect = ((a * u_node * t)[sel].sum() / a[sel].sum()
              - (a * u_node)[sel].sum() / a[sel].sum() * (a * t)[sel].sum() / a[sel].sum())
    assert expect == pytest.approx(0.5, rel=1e-12)


def test_the_dry_part_of_a_cell_does_not_enter_the_flux(cs, meshes):
    """A masked fine node must leave the covariance as well as the mean. If it stayed in only
    one of the two terms the residual would be nonsense — and it would look like a strong
    eddy flux exactly along coastlines and the bottom."""
    fine_dir, coarse_dir = meshes
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    fine = cs.FineMesh(fine_dir)
    cell = np.asarray(rmap.fine_cell)
    T = jnp.asarray(np.where(cell == 0, 5.0, 12.0))
    poisoned = np.where(cell == 0, 5.0, 12.0)
    poisoned[0] = -999.0
    mask = np.ones(N_FINE)
    mask[0] = 0.0
    ue = jnp.asarray(np.array([2.0, 2.0, -1.0, -1.0]))
    _, flux, _ = cs.level_products(jnp.asarray(poisoned), T, ue, jnp.zeros(N_ELEM),
                                   jnp.asarray(mask), rmap, fine)
    assert np.isfinite(np.asarray(flux[0])).all()
    assert np.abs(np.asarray(flux[0])).max() < 1e-9, \
        "a masked node still reached the covariance"


# ---------------------------------------------------------------------------
# the layer-validity convention
# ---------------------------------------------------------------------------

def test_a_dry_element_next_to_a_wet_node_cannot_leak_its_fill_value(cs, tmp_path):
    """Regression for the real failure, found on NG5 month 01 on 2026-08-08.

    ``elem_to_node`` averages every adjacent element, so a WET node beside a DRY element
    inherits a third of what that element holds — and FESOM writes ``_FillValue = 9.96921e+36``
    below the bottom. On the real month that put 473,770 of 3,778,615 wet (node, level) cells at
    ~7.6e+36, starting at level 3. The surface was clean, which is why a surface map missed it.

    Element 0 of the west cluster is given only ONE valid layer while its nodes keep three, so at
    layer 1 node 1 (which touches ONLY element 0) is wet beside a dry element, while node 3
    (which touches only the still-wet element 1) is clean. The snapshot slice carries the fill in
    element 0. The answer must be finite and physical.

    ⚠️ Getting this case to reproduce the bug took three tries, and the two failures are worth
    recording because both are ways a regression test can silently stop testing:
      1. a UNIFORM T makes the contamination cancel exactly in agg(u*T) − agg(u)*agg(T);
      2. so does a contaminated u that is uniform INSIDE the coarse cell — which is what happens
         if every element under the cell is dry.
    The covariance is zero whenever either field is constant on the cell, so the case needs both
    fields varying and a MIX of wet and dry elements under one cell. The real data has exactly
    that, which is why the real residual was 7.6e+36 rather than a clean cancellation.
    """
    fine_dir = _fake_fine_mesh(tmp_path / "fine", enlev=[2, NL, NL, NL])
    coarse_dir = _fake_coarse_mesh(tmp_path / "coarse")
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    fine = cs.FineMesh(fine_dir)

    k = 1
    em = fine.elem_layer_mask(k)
    np.testing.assert_array_equal(em, [0.0, 1.0, 1.0, 1.0])
    nm = fine.layer_mask(k)
    assert (nm > 0).all(), "the nodes must still be wet here, or the case is not the one"

    FILL = 9.96921e36
    ue = jnp.asarray(np.where(em > 0, 0.5, FILL))
    # ⚠️ T must VARY inside the cell. With a uniform T the contamination cancels exactly in
    # agg(u*T) - agg(u)*agg(T) and the unmasked flux comes out 0.0 — a correct answer that hides
    # the bug. The real data varies, which is why the real residual was 7.6e+36.
    T = jnp.asarray(np.where(np.arange(N_FINE) < 4, np.arange(N_FINE) + 1.0, 4.0))

    state, flux, w = cs.level_products(T, jnp.zeros(N_FINE), ue, jnp.zeros(N_ELEM),
                                       jnp.asarray(nm), rmap, fine, elem_mask=jnp.asarray(em))
    # the STATE is contaminated directly, with no cancellation to hide behind
    u_coarse = np.asarray(state[2])
    assert np.isfinite(u_coarse).all() and np.abs(u_coarse).max() < 1e3, \
        f"coarse u reached {np.abs(u_coarse).max():.3e} — fill leaked into the state"
    for f, name in zip(flux, cs.FLUX_NAMES):
        a = np.asarray(f)
        assert np.isfinite(a).all(), f"{name} not finite"
        assert np.abs(a).max() < 1e3, f"{name} reached {np.abs(a).max():.3e} — fill leaked"

    # and WITHOUT the mask both must blow up, or this test proves nothing
    bad_state, bad_flux, _ = cs.level_products(T, jnp.zeros(N_FINE), ue, jnp.zeros(N_ELEM),
                                               jnp.asarray(nm), rmap, fine, elem_mask=None)
    assert np.abs(np.asarray(bad_state[2])).max() > 1e30, \
        "the unmasked path did NOT leak into u, so this case no longer reproduces the bug"
    assert np.abs(np.asarray(bad_flux[0])).max() > 1e30, \
        "the unmasked path did NOT leak into the flux — is T uniform inside the cell again?"


def test_the_reader_turns_the_netcdf_fill_into_zero(cs, tmp_path):
    """``np.asarray(ds.variables[v][rec, k, :])`` strips netCDF4's mask and returns the fill as
    an ordinary number. ``read_slice`` must not do that."""
    teacher = tmp_path / "teacher"
    _fake_teacher_month(teacher, n_rec=1)
    path = teacher / "temp_snap.fesom.1958_01.nc"
    with netCDF4.Dataset(path, "a") as ds:
        v = ds.variables["temp_snap"]
        arr = np.asarray(v[:])
        arr[0, 1, 0] = 9.96921e36
        v[:] = arr
    with netCDF4.Dataset(path) as ds:
        naive = np.asarray(ds.variables["temp_snap"][0, 1, :])
        got = cs.read_slice(ds, "temp_snap", 0, 1)
    assert np.abs(naive).max() > 1e30, "the fixture did not actually store a fill value"
    assert np.abs(got).max() < 1e3, f"read_slice let the fill through: {np.abs(got).max():.3e}"
    assert got[0] == 0.0


def test_layer_mask_treats_nlevels_as_an_interface_count(cs, tmp_path):
    """``nlevels = 4`` means interfaces 0..3, i.e. layers 0,1,2 — layer 3 does not exist. Off
    by one adds a whole layer of sub-bottom values to every column, and because those values
    are usually 0 it reads as cold, fresh water rather than as an error."""
    d = _fake_fine_mesh(tmp_path / "fine", nlev=[4, 4, 3, 2, 4, 4, 4, 4])
    fine = cs.FineMesh(d)
    np.testing.assert_array_equal(fine.layer_mask(0), np.ones(N_FINE))
    np.testing.assert_array_equal(fine.layer_mask(1),
                                  [1, 1, 1, 0, 1, 1, 1, 1])   # node 3 has nlevels=2
    np.testing.assert_array_equal(fine.layer_mask(2),
                                  [1, 1, 0, 0, 1, 1, 1, 1])   # node 2 has nlevels=3
    np.testing.assert_array_equal(fine.layer_mask(3), np.zeros(N_FINE))


# ---------------------------------------------------------------------------
# the map cache
# ---------------------------------------------------------------------------

def test_the_map_cache_round_trips(cs, meshes, tmp_path):
    fine_dir, coarse_dir = meshes
    cache = tmp_path / "map.npz"
    built = cs.build_or_load_map(cache, fine_dir, coarse_dir)
    assert cache.exists()
    loaded = cs.build_or_load_map(cache, fine_dir, coarse_dir)
    np.testing.assert_array_equal(np.asarray(built.fine_cell), np.asarray(loaded.fine_cell))
    np.testing.assert_allclose(np.asarray(built.fine_area), np.asarray(loaded.fine_area))
    assert built.n_coarse == loaded.n_coarse


def test_a_cache_from_a_different_mesh_pair_is_refused(cs, meshes, tmp_path):
    """Silently reusing the wrong map would mis-assign every node on the planet and still
    produce a full, plausible-looking output file. It must stop, and it must not delete the
    cache itself (house rule: nothing in automation removes data)."""
    fine_dir, coarse_dir = meshes
    cache = tmp_path / "map.npz"
    cs.build_or_load_map(cache, fine_dir, coarse_dir)
    other = _fake_coarse_mesh(tmp_path / "coarse3")
    np.save(other / "geo_coord_nod2D.npy",
            np.column_stack([np.radians([0.0, 45.0, 90.0]), np.zeros(3)]))
    np.save(other / "nlevels_nod2D.npy", np.full(3, 3, dtype=np.int32))
    with pytest.raises(SystemExit, match="different mesh pair"):
        cs.build_or_load_map(cache, fine_dir, other)
    assert cache.exists(), "the cache must be left in place for inspection"


# ---------------------------------------------------------------------------
# end to end on a fake teacher month
# ---------------------------------------------------------------------------

def _fake_teacher_month(dirpath: pathlib.Path, month="1958_01", n_rec=2):
    """Write files shaped exactly like the real ones: variable ``<field>_snap``, dimensions
    (time, nz1, nod2) for nodal fields and (time, nz1, elem) for velocities, and a time axis
    one 120 s step short of each round 5-day mark."""
    dirpath.mkdir(parents=True, exist_ok=True)
    times = np.array([(i + 1) * 432000.0 - 120.0 for i in range(n_rec)])
    rng = np.random.default_rng(0)
    for field, n_space in (("temp", N_FINE), ("salt", N_FINE),
                           ("u", N_ELEM), ("v", N_ELEM)):
        with netCDF4.Dataset(dirpath / f"{field}_snap.fesom.{month}.nc", "w") as ds:
            ds.createDimension("time", n_rec)
            ds.createDimension("nz1", NL - 1)
            ds.createDimension("nod2" if n_space == N_FINE else "elem", n_space)
            tv = ds.createVariable("time", "f8", ("time",))
            tv[:] = times
            v = ds.createVariable(f"{field}_snap", "f4",
                                  ("time", "nz1", "nod2" if n_space == N_FINE else "elem"))
            base = {"temp": 5.0, "salt": 34.0, "u": 0.0, "v": 0.0}[field]
            v[:] = (base + rng.normal(0, 1, (n_rec, NL - 1, n_space))).astype(np.float32)
    return times


def test_process_record_end_to_end(cs, meshes, tmp_path):
    fine_dir, coarse_dir = meshes
    teacher = tmp_path / "teacher"
    times = _fake_teacher_month(teacher)
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    fine = cs.FineMesh(fine_dir)

    state, flux, weight, t_val = cs.process_record("1958_01", 1, rmap, fine, teacher,
                                                   verbose=False)
    n_lev_out = 2                                     # coarse zbar has 3 interfaces
    assert t_val == times[1], "the time must be READ from the file, not recomputed"
    assert t_val % 86400.0 != 0.0, "the fixture should carry the one-step offset"
    for a in state + flux:
        assert a.shape == (N_COARSE, n_lev_out)
        assert np.isfinite(a).all()
    assert weight.shape == (N_COARSE, n_lev_out)
    assert (weight > 0).all(), "every coarse cell here is fully wet"
    # temperatures must stay in the range the input had — the remap is an average, so it cannot
    # produce a value outside the fine field's own min/max
    with netCDF4.Dataset(teacher / "temp_snap.fesom.1958_01.nc") as ds:
        fine_T = np.asarray(ds.variables["temp_snap"][1, :, :])
    assert state[0].min() >= fine_T.min() - 1e-9 and state[0].max() <= fine_T.max() + 1e-9


def test_written_output_is_readable_and_marks_no_data(cs, meshes, tmp_path):
    fine_dir, coarse_dir = meshes
    teacher = tmp_path / "teacher"
    _fake_teacher_month(teacher, n_rec=1)
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    fine = cs.FineMesh(fine_dir)
    st, fl, w, tv = cs.process_record("1958_01", 0, rmap, fine, teacher, verbose=False)
    out = tmp_path / "out" / "coarse.nc"
    cs.write_output(out, "1958_01", [tv], [[a] for a in st], [[a] for a in fl], [w],
                    rmap.n_coarse, 2)
    with netCDF4.Dataset(out) as ds:
        assert set(cs.STATE_NAMES) | set(cs.FLUX_NAMES) | {"cell_area", "time"} <= set(
            ds.variables)
        assert ds.variables["temp"].shape == (1, N_COARSE, 2)
        assert float(ds.variables["time"][0]) == tv
        assert "NO DATA" in ds.description
