"""Tests for the NG5 -> CORE2 aggregation (NN-GM Task 6).

Task 5 (``test_coarsen.py``) covers the spatial filter. This file covers what happens after
it: mapping a 7.4 M-node, 70-level teacher state onto the 126 k-node, 48-level student mesh.
Four pieces — the fine->coarse assignment, the horizontal area-weighted aggregation, the
vertical thickness-weighted remap, and the element->node collocation the teacher's velocities
need.

The gate the plan asks for is **conservation through the chain to ~1e-12**, plus bottom masks
and idempotence. Conservation is the property that makes the aggregated fields usable as
targets at all: if the coarse-graining changed the heat or salt content of a column, every
target the network learns from would carry that bias, and no amount of training would reveal it
because the reference would be biased the same way.

Most tests here build a ``RemapMap`` directly from synthetic indices rather than from a mesh.
That is deliberate: the arithmetic being checked is the scatter and the weighting, and a
hand-built map makes the expected answer computable by hand instead of by re-implementing the
function under test. The parts that genuinely need geometry — nearest-neighbour assignment and
element->node collocation — use small synthetic spherical meshes and the packaged ``pi`` mesh
respectively.

⚠️ One conservation caveat is checked and documented rather than asserted away:
``elem_to_node`` is an *interpolation*, not a conservative remap, so the velocity path through
the chain does not conserve its area integral. See
:func:`test_elem_to_node_is_an_interpolation_and_does_not_conserve_the_integral`.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import coarsen
from fesom_jax.coarsen import RemapMap
from fesom_jax.mesh import load_mesh

PI_MESH_DIR = Path(__file__).resolve().parents[1] / "data" / "mesh_pi"


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(PI_MESH_DIR)


# ===========================================================================
# build_remap — the fine->coarse assignment
# ===========================================================================

def _write_mesh(dirpath: Path, lon_deg, lat_deg, *, area=None, zbar=(0.0, -10.0, -20.0)):
    """Write the three arrays ``build_remap`` reads, and nothing else.

    Only ``geo_coord_nod2D.npy``, ``area.npy`` and ``zbar.npy`` are touched by it — the point
    of taking directories rather than loaded meshes is that the 16 GB NG5 bundle never has to
    be materialised, and a test that had to write a whole mesh export would not be able to
    exercise the pole and dateline cases cheaply.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    lon_deg, lat_deg = np.asarray(lon_deg, float), np.asarray(lat_deg, float)
    np.save(dirpath / "geo_coord_nod2D.npy",
            np.column_stack([np.radians(lon_deg), np.radians(lat_deg)]))
    np.save(dirpath / "area.npy",
            np.ones(lon_deg.size) if area is None else np.asarray(area, float))
    np.save(dirpath / "zbar.npy", np.asarray(zbar, float))
    return dirpath


def _great_circle_nearest(flon, flat, clon, clat):
    """Brute-force nearest coarse node per fine node, by great-circle angle. Independent of
    the implementation: no KD-tree, no Cartesian conversion, just the haversine formula."""
    flon, flat = np.radians(flon)[:, None], np.radians(flat)[:, None]
    clon, clat = np.radians(clon)[None, :], np.radians(clat)[None, :]
    d = (np.sin((flat - clat) / 2) ** 2
         + np.cos(flat) * np.cos(clat) * np.sin((flon - clon) / 2) ** 2)
    return np.argmin(2 * np.arcsin(np.sqrt(np.clip(d, 0, 1))), axis=1)


def _lonlat_euclidean_nearest(flon, flat, clon, clat):
    """What a naive lon/lat metric would answer — used to prove the sphere cases are real
    discriminators and not tests that any implementation would pass."""
    d2 = ((np.asarray(flon)[:, None] - np.asarray(clon)[None, :]) ** 2
          + (np.asarray(flat)[:, None] - np.asarray(clat)[None, :]) ** 2)
    return np.argmin(d2, axis=1)


def test_build_remap_matches_a_brute_force_great_circle_search(tmp_path):
    rng = np.random.default_rng(0)
    clon, clat = rng.uniform(-180, 180, 40), np.degrees(np.arcsin(rng.uniform(-1, 1, 40)))
    flon, flat = rng.uniform(-180, 180, 500), np.degrees(np.arcsin(rng.uniform(-1, 1, 500)))
    rmap = coarsen.build_remap(_write_mesh(tmp_path / "fine", flon, flat),
                              _write_mesh(tmp_path / "coarse", clon, clat))
    assert rmap.n_coarse == 40
    np.testing.assert_array_equal(np.asarray(rmap.fine_cell),
                                  _great_circle_nearest(flon, flat, clon, clat))


def test_the_assignment_crosses_the_dateline(tmp_path):
    """A fine node just east of 180 must go to the coarse node just west of it.

    In lon/lat those two are 359.8 degrees apart and a Euclidean metric sends the fine node
    1100 km away to the wrong side instead. The Southern Ocean spans this seam, so a
    dateline-broken map would corrupt a band right through the ACC — the region the whole
    parameterization is aimed at.
    """
    flon, flat = [179.9], [0.0]
    clon, clat = [-179.9, 170.0], [0.0, 0.0]
    rmap = coarsen.build_remap(_write_mesh(tmp_path / "fine", flon, flat),
                               _write_mesh(tmp_path / "coarse", clon, clat))
    assert int(rmap.fine_cell[0]) == 0, "did not cross the dateline"
    assert _lonlat_euclidean_nearest(flon, flat, clon, clat)[0] == 1, \
        "this case no longer discriminates: a lon/lat metric would now pass it too"


def test_the_assignment_goes_over_the_pole(tmp_path):
    """Near the pole the shortest path can cross the meridian 180 degrees away.

    Fine node at (170E, 89N). Coarse 0 is 4 degrees south on the same meridian; coarse 1 is
    on the opposite meridian but only 1.5 degrees away over the pole. The Arctic is where the
    teacher's eddies are smallest, so this is exactly where a distorted metric would do the
    most damage.
    """
    flon, flat = [170.0], [89.0]
    clon, clat = [170.0, -10.0], [85.0, 89.5]
    rmap = coarsen.build_remap(_write_mesh(tmp_path / "fine", flon, flat),
                               _write_mesh(tmp_path / "coarse", clon, clat))
    assert int(rmap.fine_cell[0]) == 1, "did not take the route over the pole"
    assert _lonlat_euclidean_nearest(flon, flat, clon, clat)[0] == 0, \
        "this case no longer discriminates: a lon/lat metric would now pass it too"


def test_build_remap_takes_the_surface_column_of_a_2d_area(tmp_path):
    """``area`` is (nod2D, nl) in a real export and (nod2D,) in some tools. The horizontal
    weight is the SURFACE control-volume area, so a 2-D array must be read at level 0 and not
    summed or averaged over the column."""
    lon, lat = [0.0, 10.0, 20.0], [0.0, 0.0, 0.0]
    area2d = np.column_stack([[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]])
    rmap = coarsen.build_remap(_write_mesh(tmp_path / "fine", lon, lat, area=area2d),
                               _write_mesh(tmp_path / "coarse", [0.0], [0.0]))
    np.testing.assert_allclose(np.asarray(rmap.fine_area), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(rmap.fine_depth), [0.0, -10.0, -20.0])
    np.testing.assert_allclose(np.asarray(rmap.coarse_depth), [0.0, -10.0, -20.0])


# ===========================================================================
# aggregate_horizontal — area-weighted mean per coarse cell
# ===========================================================================

def _synthetic_map(n_fine=97, n_coarse=11, seed=0, empty_cell=None):
    """A RemapMap with random cell membership and random areas.

    ``empty_cell``: a coarse index deliberately left with no fine node, so the empty-cell
    path is exercised by construction rather than by luck.
    """
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, n_coarse, n_fine)
    if empty_cell is not None:
        cells[cells == empty_cell] = (empty_cell + 1) % n_coarse
    return RemapMap(fine_cell=jnp.asarray(cells.astype(np.int32)),
                    fine_area=jnp.asarray(rng.uniform(0.5, 4.0, n_fine)),
                    n_coarse=n_coarse,
                    fine_depth=jnp.asarray([0.0, -1.0]),
                    coarse_depth=jnp.asarray([0.0, -1.0]))


def test_aggregation_conserves_the_area_integral():
    """THE gate. sum over coarse cells of (value x cell area) must equal sum over fine nodes
    of (value x node area), because the coarse value is the area-weighted mean and the cell
    area is the summed node area. Anything else puts a bias in every training target."""
    rmap = _synthetic_map()
    rng = np.random.default_rng(1)
    field = jnp.asarray(rng.normal(10.0, 3.0, rmap.fine_cell.size))
    val, wgt = coarsen.aggregate_horizontal_with_weight(field, rmap)
    fine_integral = float(jnp.sum(rmap.fine_area * field))
    coarse_integral = float(jnp.sum(val * wgt))
    assert abs(coarse_integral - fine_integral) / abs(fine_integral) < 1e-14, \
        (fine_integral, coarse_integral)


def test_aggregation_conserves_per_level_for_a_3d_field():
    rmap = _synthetic_map(seed=2)
    rng = np.random.default_rng(3)
    field = jnp.asarray(rng.normal(0.0, 1.0, (rmap.fine_cell.size, 5)))
    val, wgt = coarsen.aggregate_horizontal_with_weight(field, rmap)
    assert val.shape == (rmap.n_coarse, 5) and wgt.shape == (rmap.n_coarse, 5)
    fine = np.asarray(jnp.einsum("n,nk->k", rmap.fine_area, field))
    coarse = np.asarray(jnp.sum(val * wgt, axis=0))
    np.testing.assert_allclose(coarse, fine, rtol=1e-14)


def test_aggregation_is_idempotent_on_an_already_coarse_field():
    """A field that is already constant within each coarse cell must come back exactly as it
    is. This is what makes the operator safe to apply twice, and it is the property that fails
    first if the weights are ever normalised wrongly."""
    rmap = _synthetic_map(seed=4)
    per_cell = np.random.default_rng(5).normal(0.0, 1.0, rmap.n_coarse)
    field = jnp.asarray(per_cell[np.asarray(rmap.fine_cell)])
    val, wgt = coarsen.aggregate_horizontal_with_weight(field, rmap)
    present = np.asarray(wgt) > 0
    np.testing.assert_allclose(np.asarray(val)[present], per_cell[present], rtol=1e-15)


def test_a_coarse_cell_with_no_wet_fine_node_is_zero_with_zero_weight():
    """"Empty" and "zero" are different answers and the caller has to be able to tell them
    apart — a coarse cell under land must not enter a loss as a 0 K temperature."""
    rmap = _synthetic_map(seed=6, empty_cell=3)
    field = jnp.ones(rmap.fine_cell.size)
    val, wgt = coarsen.aggregate_horizontal_with_weight(field, rmap)
    assert float(wgt[3]) == 0.0 and float(val[3]) == 0.0
    others = [i for i in range(rmap.n_coarse) if i != 3]
    assert (np.asarray(wgt)[others] > 0).all()
    np.testing.assert_allclose(np.asarray(val)[others], 1.0, rtol=1e-15)


def test_dry_fine_nodes_are_excluded_not_averaged_in_as_zeros():
    """The bottom/coast mask. A dry fine node must drop out of both numerator and denominator;
    averaging it in as a zero would cool every coarse cell that touches a coastline."""
    rmap = _synthetic_map(seed=7)
    n = rmap.fine_cell.size
    rng = np.random.default_rng(8)
    field = np.full(n, 20.0)
    wet = rng.random(n) > 0.3
    field[~wet] = -999.0                     # poison: must never reach the answer
    val, wgt = coarsen.aggregate_horizontal_with_weight(
        jnp.asarray(field), rmap, fine_mask=jnp.asarray(wet.astype(float)))
    touched = np.asarray(wgt) > 0
    np.testing.assert_allclose(np.asarray(val)[touched], 20.0, rtol=1e-14)

    # and the weight is the WET area only
    cells, areas = np.asarray(rmap.fine_cell), np.asarray(rmap.fine_area)
    expect = np.bincount(cells[wet], weights=areas[wet], minlength=rmap.n_coarse)
    np.testing.assert_allclose(np.asarray(wgt), expect, rtol=1e-14)


def test_a_3d_mask_can_differ_per_level():
    """Bathymetry is per level: a fine node is wet at the surface and dry at depth. A mask
    broadcast from the surface would carry deep values that do not exist."""
    rmap = _synthetic_map(n_fine=40, n_coarse=4, seed=9)
    n, nk = 40, 3
    field = jnp.asarray(np.tile(np.array([1.0, 2.0, 3.0]), (n, 1)))
    m = np.ones((n, nk))
    m[:, 2] = 0.0                            # nothing is wet at the deepest level
    val, wgt = coarsen.aggregate_horizontal_with_weight(field, rmap, fine_mask=jnp.asarray(m))
    np.testing.assert_allclose(np.asarray(val)[:, 0], 1.0, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(val)[:, 1], 2.0, rtol=1e-15)
    assert (np.asarray(wgt)[:, 2] == 0.0).all() and (np.asarray(val)[:, 2] == 0.0).all()


def test_the_1d_and_2d_paths_agree():
    """``aggregate_horizontal`` has separate code for (n,) and (n,nl); a divergence between
    them would show up only in whichever shape the pipeline happened not to use."""
    rmap = _synthetic_map(seed=10)
    f = jnp.asarray(np.random.default_rng(11).normal(0, 1, rmap.fine_cell.size))
    a, wa = coarsen.aggregate_horizontal_with_weight(f, rmap)
    b, wb = coarsen.aggregate_horizontal_with_weight(f[:, None], rmap)
    assert a.shape == (rmap.n_coarse,) and b.shape == (rmap.n_coarse, 1)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b)[:, 0])
    np.testing.assert_array_equal(np.asarray(wa), np.asarray(wb)[:, 0])
    np.testing.assert_array_equal(np.asarray(a),
                                  np.asarray(coarsen.aggregate_horizontal(f, rmap)))


# ===========================================================================
# remap_vertical — thickness-weighted conservative column remap
# ===========================================================================

def test_vertical_remap_hand_checked_on_a_two_layer_column():
    """Small enough to verify with arithmetic instead of with another implementation.

    Fine: [0,-10] = 1.0, [-10,-20] = 3.0. A coarse layer [0,-20] must give (10*1+10*3)/20 = 2;
    a coarse layer [0,-15] must give (10*1+5*3)/15 = 5/3, i.e. it must weight the partial
    overlap by its thickness and not count the fine layer whole.
    """
    fine_z = [0.0, -10.0, -20.0]
    val = jnp.asarray([1.0, 3.0])
    np.testing.assert_allclose(
        np.asarray(coarsen.remap_vertical(val, fine_z, [0.0, -20.0])), [2.0], rtol=1e-15)
    np.testing.assert_allclose(
        np.asarray(coarsen.remap_vertical(val, fine_z, [0.0, -15.0])), [5.0 / 3.0], rtol=1e-15)
    np.testing.assert_allclose(
        np.asarray(coarsen.remap_vertical(val, fine_z, [0.0, -15.0, -20.0])),
        [5.0 / 3.0, 3.0], rtol=1e-15)


def test_vertical_remap_conserves_the_column_integral():
    """70 uneven fine layers onto 12 uneven coarse layers spanning the same column: the
    thickness-weighted integral must be unchanged to round-off. This is the vertical half of
    the plan's conservation gate."""
    rng = np.random.default_rng(12)
    fine_z = np.concatenate([[0.0], -np.cumsum(rng.uniform(5.0, 60.0, 70))])
    coarse_z = np.concatenate([[0.0],
                               np.sort(rng.choice(fine_z[1:-1], 11, replace=False))[::-1],
                               [fine_z[-1]]])
    val = jnp.asarray(rng.normal(5.0, 2.0, 70))
    out = coarsen.remap_vertical(val, fine_z, coarse_z)
    fine_int = float(np.sum(-np.diff(fine_z) * np.asarray(val)))
    coarse_int = float(np.sum(-np.diff(coarse_z) * np.asarray(out)))
    assert abs(coarse_int - fine_int) / abs(fine_int) < 1e-12, (fine_int, coarse_int)


def test_vertical_remap_conserves_when_the_coarse_interfaces_do_not_align():
    """The realistic case: NG5's 70 interfaces and CORE2's 48 share almost none of their
    depths, so nearly every coarse layer is a partial-overlap sum."""
    fine_z = np.concatenate([[0.0], -np.cumsum(np.full(70, 10.0))])
    coarse_z = np.concatenate([[0.0], -np.cumsum(np.full(48, 700.0 / 48))])
    val = jnp.asarray(np.random.default_rng(13).normal(0.0, 1.0, 70))
    out = coarsen.remap_vertical(val, fine_z, coarse_z)
    fine_int = float(np.sum(-np.diff(fine_z) * np.asarray(val)))
    coarse_int = float(np.sum(-np.diff(coarse_z) * np.asarray(out)))
    assert abs(coarse_int - fine_int) / abs(fine_int) < 1e-12


def test_vertical_remap_preserves_a_constant_profile():
    fine_z = np.concatenate([[0.0], -np.cumsum(np.random.default_rng(14).uniform(5, 50, 30))])
    out = coarsen.remap_vertical(jnp.full(30, 34.7), fine_z, [0.0, -100.0, -400.0, fine_z[-1]])
    np.testing.assert_allclose(np.asarray(out), 34.7, rtol=1e-15)


def test_a_coarse_layer_below_the_fine_column_is_zero_not_extrapolated():
    """A coarse layer with no overlap at all has no information behind it. Returning 0 is the
    contract; the caller must mask it. The failure this guards against is the opposite one —
    silently extending the deepest value downwards, which would invent stratification."""
    out = coarsen.remap_vertical(jnp.asarray([2.0, 4.0]), [0.0, -10.0, -20.0],
                                 [0.0, -20.0, -50.0, -90.0])
    assert float(out[0]) == pytest.approx(3.0, rel=1e-15)
    assert float(out[1]) == 0.0 and float(out[2]) == 0.0


def test_a_masked_bottom_is_not_diluted_by_the_dry_layers_it_overlaps():
    """The sub-bottom-zero trap. A coarse layer straddling the fine column's bottom must be
    the mean of the WET part only. If the dry fine layers were averaged in as zeros, every
    near-bottom coarse cell would read cold and fresh — a bias that looks exactly like a real
    water-mass property and would be learned as one.
    """
    fine_z = [0.0, -10.0, -20.0, -30.0, -40.0]
    val = jnp.asarray([10.0, 10.0, 0.0, 0.0])        # bottom two layers are dry, value unset
    mask = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    out = coarsen.remap_vertical(val, fine_z, [0.0, -30.0, -40.0], fine_mask=mask)
    assert float(out[0]) == pytest.approx(10.0, rel=1e-15), \
        "the coarse layer [0,-30] straddles the bottom and must read the wet value, not 2/3 of it"
    assert float(out[1]) == 0.0, "a fully dry coarse layer must report 0, i.e. no data"


def test_vertical_remap_handles_a_stack_of_columns():
    """(n_nodes, n_lev) at once — the shape the pipeline actually passes, and the einsum's
    leading-dimension handling is the part that could silently transpose."""
    rng = np.random.default_rng(15)
    fine_z = np.concatenate([[0.0], -np.cumsum(np.full(20, 25.0))])
    coarse_z = [0.0, -100.0, -300.0, -500.0]
    stack = rng.normal(0.0, 1.0, (7, 20))
    out = coarsen.remap_vertical(jnp.asarray(stack), fine_z, coarse_z)
    assert out.shape == (7, 4 - 1)
    for i in range(7):
        one = coarsen.remap_vertical(jnp.asarray(stack[i]), fine_z, coarse_z)
        np.testing.assert_allclose(np.asarray(out[i]), np.asarray(one), rtol=1e-15)


# ===========================================================================
# elem_to_node — the teacher's velocities are elemental
# ===========================================================================

def test_elem_to_node_preserves_a_constant(mesh):
    """Partition of unity. Every node that has at least one element must get the constant
    back exactly, or the collocation carries a multiplicative bias that varies with mesh
    density — worst exactly where resolution changes, which is everywhere interesting."""
    out = np.asarray(coarsen.elem_to_node(jnp.full((mesh.elem2D,), -1.75), mesh))
    np.testing.assert_allclose(out, -1.75, rtol=1e-14)


def test_elem_to_node_is_the_area_weighted_mean_of_the_adjacent_elements(mesh):
    """Checked against an explicit loop over elements, so the scatter's index handling and
    the weighting are both verified against something that shares no code with them.

    ⚠️ The comparison needs an absolute floor, not rtol alone. The two implementations sum the
    same terms in different orders, and at a node whose adjacent elements nearly cancel the
    mean is tiny while the summands are O(1) — measured on the pi mesh, one node of 3140 lands
    at 1.78e-04 with the two answers 2.4e-17 apart, which is 1.4e-13 *relative* and pure
    cancellation. Relative error on a cancelled sum measures the cancellation, not the
    operator; atol=1e-15 against O(1) summands is ~4 ulp and is the honest bound.
    """
    rng = np.random.default_rng(16)
    fe = rng.normal(0.0, 1.0, mesh.elem2D)
    en = np.asarray(mesh.elem_nodes)
    ea = np.asarray(mesh.elem_area)
    num = np.zeros(mesh.nod2D)
    den = np.zeros(mesh.nod2D)
    for k in range(3):
        np.add.at(num, en[:, k], ea * fe)
        np.add.at(den, en[:, k], ea)
    expect = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    got = np.asarray(coarsen.elem_to_node(jnp.asarray(fe), mesh))
    np.testing.assert_allclose(got, expect, rtol=1e-13, atol=1e-15)


def test_elem_to_node_maps_levels_independently(mesh):
    rng = np.random.default_rng(17)
    fe = np.zeros((mesh.elem2D, 3))
    fe[:, 0] = rng.normal(0, 1, mesh.elem2D)
    fe[:, 2] = 5.0
    out = np.asarray(coarsen.elem_to_node(jnp.asarray(fe), mesh))
    assert out.shape == (mesh.nod2D, 3)
    np.testing.assert_allclose(out[:, 1], 0.0)
    np.testing.assert_allclose(out[:, 2], 5.0, rtol=1e-14)
    one = np.asarray(coarsen.elem_to_node(jnp.asarray(fe[:, 0]), mesh))
    np.testing.assert_allclose(out[:, 0], one, rtol=1e-15)


def test_elem_to_node_is_an_interpolation_and_does_not_conserve_the_integral(mesh):
    """Documented limitation, asserted so nobody has to rediscover it.

    ``elem_to_node`` returns the area-weighted MEAN of the elements around a node. That is the
    right thing for reading a velocity at a node, but it is NOT the conservative
    element->node lumping (which distributes ``A_e/3`` of each element to each of its nodes).
    So ``sum_n A_n u_n != sum_e A_e u_e``: the nodal integral of an interpolated field differs
    from the elemental one, measured on the pi mesh at ~1e-2 relative for white noise and
    ~1e-4 for a smooth field.

    Consequence for the pipeline: the tracer path (nodal T, S -> aggregate -> remap_vertical)
    conserves to 1e-12, but a velocity that has passed through here does not, so eddy-flux
    products built from it must not be presented as conservative. If a conservative velocity
    is ever needed, lump with A_e/3 instead of averaging.
    """
    rng = np.random.default_rng(18)
    an = np.asarray(mesh.area[:, 0] if mesh.area.ndim == 2 else mesh.area)
    ea = np.asarray(mesh.elem_area)

    # a constant IS conserved, because both sides collapse to (total area) x value
    c = np.asarray(coarsen.elem_to_node(jnp.full((mesh.elem2D,), 2.0), mesh))
    assert abs((an * c).sum() / (ea * 2.0).sum() - 1.0) < 1e-12

    smooth = np.sin(np.asarray(mesh.geo_coord_nod2D[:, 0])[np.asarray(mesh.elem_nodes)].mean(1))
    for fe, name, bound in ((rng.normal(0, 1, mesh.elem2D), "noise", 5e-2),
                            (smooth, "smooth", 5e-3)):
        nodal = np.asarray(coarsen.elem_to_node(jnp.asarray(fe), mesh))
        err = abs((an * nodal).sum() - (ea * fe).sum()) / abs((ea * fe).sum())
        assert err < bound, f"{name}: integral error {err:.2e} exceeds the documented {bound:.0e}"


# ===========================================================================
# the chain
# ===========================================================================

def test_the_tracer_chain_conserves_the_volume_integral():
    """The plan's gate, end to end on the path that must conserve: a fine (node, level) tracer
    field goes through the horizontal aggregation and then the vertical remap, and the volume
    integral (area x thickness x value) comes out unchanged to 1e-12.

    Order matters and this is the order the pipeline uses. Horizontal first is 20x cheaper —
    it reduces 7.4 M columns to 126 k before anything touches the vertical — and it is only
    valid because the two operators act on separate axes, which is what this test confirms by
    getting the same integral either way.
    """
    rng = np.random.default_rng(19)
    n_fine, n_lev = 300, 25
    rmap = _synthetic_map(n_fine=n_fine, n_coarse=17, seed=20)
    fine_z = np.concatenate([[0.0], -np.cumsum(rng.uniform(5.0, 40.0, n_lev))])
    coarse_z = np.concatenate([[0.0], np.sort(rng.choice(fine_z[1:-1], 8,
                                                        replace=False))[::-1], [fine_z[-1]]])
    field = jnp.asarray(rng.normal(4.0, 1.0, (n_fine, n_lev)))
    dz_f, dz_c = -np.diff(fine_z), -np.diff(coarse_z)

    fine_int = float(np.einsum("n,k,nk->", np.asarray(rmap.fine_area), dz_f, np.asarray(field)))

    h, wgt = coarsen.aggregate_horizontal_with_weight(field, rmap)
    v = coarsen.remap_vertical(h, fine_z, coarse_z)
    # the cell area is level-independent here (no mask), so take level 0
    area_c = np.asarray(wgt)[:, 0]
    chain_int = float(np.einsum("n,k,nk->", area_c, dz_c, np.asarray(v)))
    assert abs(chain_int - fine_int) / abs(fine_int) < 1e-12, (fine_int, chain_int)

    # ... and the other order must agree, which is what makes "horizontal first" a free choice
    v2 = coarsen.remap_vertical(field, fine_z, coarse_z)
    h2, wgt2 = coarsen.aggregate_horizontal_with_weight(v2, rmap)
    other = float(np.einsum("n,k,nk->", np.asarray(wgt2)[:, 0], dz_c, np.asarray(h2)))
    assert abs(other - fine_int) / abs(fine_int) < 1e-12
    np.testing.assert_allclose(np.asarray(v), np.asarray(h2), rtol=1e-12)


def test_elem_to_node_masks_dry_elements_out_of_both_numerator_and_denominator(mesh):
    """The bug this exists to prevent, found on real NG5 output 2026-08-08.

    Without a mask, ``elem_to_node`` averages every adjacent element unconditionally, so a WET
    node next to a DRY element inherits a third of whatever that element holds — and FESOM
    writes ``_FillValue = 9.96921e+36`` below the bottom. Measured on month 01: 473,770 of
    3,778,615 wet (node, level) cells came out at ~7.6e+36, starting at level 3. The surface was
    clean, which is exactly why a surface map did not show it.

    Zeroing the fill first is NOT enough, and the second half of this test is that point: a dry
    element contributing a spurious zero dilutes the node toward the bottom, which looks like
    physics. The mask must remove it from the weight as well.
    """
    rng = np.random.default_rng(21)
    dry = rng.random(mesh.elem2D) < 0.25
    m = jnp.asarray((~dry).astype(float))

    # a poisoned field: the fill value where dry, a real value where wet
    fe = np.full(mesh.elem2D, 3.0)
    fe[dry] = 9.96921e36
    got = np.asarray(coarsen.elem_to_node(jnp.asarray(fe), mesh, elem_mask=m))
    en, ea = np.asarray(mesh.elem_nodes), np.asarray(mesh.elem_area)
    touched = np.zeros(mesh.nod2D, dtype=bool)
    for k in range(3):
        np.add.at(touched, en[:, k], ~dry)
    np.testing.assert_allclose(got[touched], 3.0, rtol=1e-14)
    assert (got[~touched] == 0.0).all(), "a node with no wet element must be 0, not fill/3"
    assert np.abs(got).max() < 1e3, f"fill leaked: max {np.abs(got).max():.3e}"

    # ... and zeroing instead of masking is measurably WRONG, not merely different
    fz = np.where(dry, 0.0, 3.0)
    zeroed = np.asarray(coarsen.elem_to_node(jnp.asarray(fz), mesh))
    diluted = touched & (zeroed < 2.999)
    assert diluted.sum() > 0.1 * touched.sum(), \
        "the zeroing variant should dilute many wet nodes — if not, this mesh has too few " \
        "wet/dry neighbours for the test to discriminate"
    # the masked answer is exact where the zeroed one is diluted
    np.testing.assert_allclose(got[diluted], 3.0, rtol=1e-14)


def test_elem_to_node_without_a_mask_is_unchanged(mesh):
    """The mask is opt-in: every existing caller must get bit-identical results, since the
    suite's GM/momentum gates compare against Fortran dumps."""
    rng = np.random.default_rng(22)
    fe = jnp.asarray(rng.normal(0.0, 1.0, mesh.elem2D))
    a = np.asarray(coarsen.elem_to_node(fe, mesh))
    b = np.asarray(coarsen.elem_to_node(fe, mesh, elem_mask=jnp.ones(mesh.elem2D)))
    np.testing.assert_array_equal(a, b)


def test_elem_to_node_mask_works_per_level(mesh):
    """Bathymetry is per level: an element is wet at the surface and dry at depth, so the mask
    must be (elem2D, nl) and applied level by level."""
    nk = 3
    fe = np.full((mesh.elem2D, nk), 2.0)
    fe[:, 2] = 9.96921e36                       # whole deepest level is below the bottom
    m = np.ones((mesh.elem2D, nk))
    m[:, 2] = 0.0
    got = np.asarray(coarsen.elem_to_node(jnp.asarray(fe), mesh, elem_mask=jnp.asarray(m)))
    np.testing.assert_allclose(got[:, 0], 2.0, rtol=1e-14)
    np.testing.assert_allclose(got[:, 1], 2.0, rtol=1e-14)
    assert (got[:, 2] == 0.0).all(), "a fully dry level must be 0, not fill"
