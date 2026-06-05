"""Task 1.1 gate: the loaded Mesh must equal the C export and be self-consistent.

Every loaded array is byte-compared to its source ``.npy`` (it's the same
bytes), indices are checked to be already 0-based and in range, the ragged-level
masks are verified against ``nlevels``/``ulevels``, and the pytree round-trips.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from jax import tree_util

from fesom_jax import mesh as meshmod
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, Mesh, level_masks, load_mesh

pytestmark = pytest.mark.skipif(
    not DEFAULT_PI_MESH_DIR.is_dir(),
    reason=f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (run Task 0.3 export)",
)


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    return load_mesh(DEFAULT_PI_MESH_DIR)


# Expected pi counts (meta.txt) — pinned so a re-export regression is caught.
PI_COUNTS = dict(
    nod2D=3140, elem2D=5839, edge2D=8986, nl=48, edge2D_in=8531, myDim_edge2D=8986
)


def test_scalar_counts(mesh):
    for k, v in PI_COUNTS.items():
        assert getattr(mesh, k) == v, f"{k}={getattr(mesh, k)} != {v}"
    assert isinstance(mesh.nod2D, int)  # static metadata, not a traced array
    assert mesh.ocean_area == pytest.approx(339720478579587.94, rel=1e-12)


def test_arrays_equal_export_bit_for_bit(mesh):
    """Each Mesh array equals the raw .npy it was loaded from (same bytes)."""
    for field, fname in meshmod.mesh_field_files().items():
        raw = np.load(DEFAULT_PI_MESH_DIR / fname)
        got = np.asarray(getattr(mesh, field))
        # int → int32, float stays f8; values must be identical.
        expect = raw.astype(np.int32) if raw.dtype.kind in "iu" else raw.astype(np.float64)
        assert got.shape == expect.shape, f"{field}: shape {got.shape} != {expect.shape}"
        assert got.dtype == expect.dtype, f"{field}: dtype {got.dtype} != {expect.dtype}"
        assert np.array_equal(got, expect), f"{field}: values differ from export"


def test_dtypes(mesh):
    int_fields = {
        "coast_flag", "nlevels_nod2D", "nlevels_nod2D_min", "ulevels_nod2D",
        "ulevels_nod2D_max", "nod_in_elem2D_offsets", "elem_nodes", "nlevels",
        "ulevels", "edges", "edge_tri", "edge_up_dn_tri", "nod_in_elem2D",
    }
    for f in dataclasses.fields(Mesh):
        if f.metadata.get("static") or f.name.endswith("_mask"):
            continue
        arr = np.asarray(getattr(mesh, f.name))
        if f.name in int_fields:
            assert arr.dtype == np.int32, f"{f.name} should be int32, got {arr.dtype}"
        else:
            assert arr.dtype == np.float64, f"{f.name} should be f8, got {arr.dtype}"


def test_shapes_consistent(mesh):
    n, e, ed, nl = mesh.nod2D, mesh.elem2D, mesh.edge2D, mesh.nl
    assert mesh.coord_nod2D.shape == (n, 2)
    assert mesh.geo_coord_nod2D.shape == (n, 2)
    assert mesh.area.shape == (n, nl)
    assert mesh.areasvol.shape == (n, nl)
    assert mesh.zbar_3d_n.shape == (n, nl)
    assert mesh.nod_in_elem2D_offsets.shape == (n + 1,)
    assert mesh.elem_nodes.shape == (e, 3)
    assert mesh.gradient_sca.shape == (e, 6)
    assert mesh.edges.shape == (ed, 2)
    assert mesh.edge_tri.shape == (ed, 2)
    assert mesh.edge_cross_dxdy.shape == (ed, 4)
    assert mesh.edge_up_dn_tri.shape == (mesh.myDim_edge2D, 2)
    assert mesh.zbar.shape == (nl,)
    assert mesh.Z.shape == (nl - 1,)


def test_indices_already_zero_based_and_in_range(mesh):
    n, e = mesh.nod2D, mesh.elem2D
    en = np.asarray(mesh.elem_nodes)
    assert en.min() == 0 and en.max() == n - 1  # 0-based, full range used
    edges = np.asarray(mesh.edges)
    assert edges.min() >= 0 and edges.max() < n
    et = np.asarray(mesh.edge_tri)
    assert et.min() == -1 and et.max() == e - 1  # -1 sentinel for boundary
    assert np.all((et == -1) | ((et >= 0) & (et < e)))
    ud = np.asarray(mesh.edge_up_dn_tri)
    assert np.all((ud == -1) | ((ud >= 0) & (ud < e)))
    nie = np.asarray(mesh.nod_in_elem2D)
    assert nie.min() >= 0 and nie.max() < e


def test_edge_tri_classification(mesh):
    """Each edge has 1 (boundary) or 2 (interior) real neighbours; the interior
    count matches edge2D_in (the [0,edge2D_in) interior block)."""
    et = np.asarray(mesh.edge_tri)
    nnz = (et >= 0).sum(axis=1)
    assert set(np.unique(nnz)) <= {1, 2}
    assert int((nnz == 2).sum()) == mesh.edge2D_in
    assert int((nnz == 1).sum()) == mesh.edge2D - mesh.edge2D_in


def test_csr_well_formed_and_consistent(mesh):
    off = np.asarray(mesh.nod_in_elem2D_offsets)
    flat = np.asarray(mesh.nod_in_elem2D)
    en = np.asarray(mesh.elem_nodes)
    assert off[0] == 0
    assert off[-1] == flat.shape[0]
    assert np.all(np.diff(off) >= 0)
    # every cell listed for node n actually contains n
    for node in (0, 1001, 1500, mesh.nod2D - 1):
        cells = flat[off[node]:off[node + 1]]
        assert cells.size > 0
        assert all(node in en[c] for c in cells)


def test_levels_no_cavity(mesh):
    """pi has no cavities (ulevels==1) and well-ordered level counts."""
    uln = np.asarray(mesh.ulevels_nod2D)
    nln = np.asarray(mesh.nlevels_nod2D)
    nlnmin = np.asarray(mesh.nlevels_nod2D_min)
    assert np.all(uln == 1)
    assert np.all(np.asarray(mesh.ulevels) == 1)
    assert nln.min() >= 1 and nln.max() <= mesh.nl
    assert np.all(nlnmin <= nln)
    nlev = np.asarray(mesh.nlevels)
    assert nlev.min() >= 1 and nlev.max() <= mesh.nl


def test_masks_match_level_counts(mesh):
    """layer mask has (nlevels-ulevels) valid levels; iface has one more; both
    are a single contiguous block starting at ulevels-1."""
    uln = np.asarray(mesh.ulevels_nod2D)
    nln = np.asarray(mesh.nlevels_nod2D)
    layer = np.asarray(mesh.node_layer_mask)
    iface = np.asarray(mesh.node_iface_mask)
    assert layer.shape == (mesh.nod2D, mesh.nl)
    np.testing.assert_array_equal(layer.sum(axis=1), nln - uln)        # = nln-1
    np.testing.assert_array_equal(iface.sum(axis=1), nln - uln + 1)    # = nln
    # contiguity: valid levels are exactly [ulevels-1, nlevels-1) for layers
    k = np.arange(mesh.nl)[None, :]
    np.testing.assert_array_equal(
        layer, (k >= (uln - 1)[:, None]) & (k < (nln - 1)[:, None])
    )
    np.testing.assert_array_equal(
        iface, (k >= (uln - 1)[:, None]) & (k < nln[:, None])
    )
    # element masks
    el_layer = np.asarray(mesh.elem_layer_mask)
    el_iface = np.asarray(mesh.elem_iface_mask)
    ule = np.asarray(mesh.ulevels)
    nle = np.asarray(mesh.nlevels)
    np.testing.assert_array_equal(el_layer.sum(axis=1), nle - ule)
    np.testing.assert_array_equal(el_iface.sum(axis=1), nle - ule + 1)


def test_level_masks_helper_small():
    import jax.numpy as jnp
    # node with ulevels=1, nlevels=3 → layers k∈{0,1}, ifaces k∈{0,1,2}
    layer, iface = level_masks(jnp.array([1, 2]), jnp.array([3, 4]), nl=5)
    np.testing.assert_array_equal(np.asarray(layer)[0], [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(iface)[0], [1, 1, 1, 0, 0])
    # ulevels=2 (cavity) → starts at k=1
    np.testing.assert_array_equal(np.asarray(layer)[1], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(np.asarray(iface)[1], [0, 1, 1, 1, 0])


def test_geometry_value_ranges(mesh):
    """Sanity ranges from FRESH_START §20 / the export inspection."""
    assert np.all(np.asarray(mesh.elem_area) > 0)
    assert np.all(np.asarray(mesh.area) >= 0)
    cosv = np.asarray(mesh.elem_cos)
    assert np.all((cosv > 0) & (cosv <= 1.0 + 1e-12))
    zbar = np.asarray(mesh.zbar)
    assert zbar[0] == 0.0 and np.all(np.diff(zbar) < 0)  # 0 at surface, decreasing
    Z = np.asarray(mesh.Z)
    np.testing.assert_allclose(Z, 0.5 * (zbar[:-1] + zbar[1:]), rtol=0, atol=0)


def test_pytree_roundtrip(mesh):
    leaves, treedef = tree_util.tree_flatten(mesh)
    # leaves are arrays; static counts live in the treedef, not the leaves
    assert all(hasattr(l, "shape") for l in leaves)
    assert len(leaves) == len([f for f in dataclasses.fields(Mesh)
                               if not f.metadata.get("static")])
    rebuilt = tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.nod2D == mesh.nod2D and rebuilt.nl == mesh.nl
    np.testing.assert_array_equal(
        np.asarray(rebuilt.elem_nodes), np.asarray(mesh.elem_nodes)
    )
    # a tree_map over the pytree touches only array leaves
    import jax.numpy as jnp
    doubled = tree_util.tree_map(lambda x: x * 2, mesh)
    np.testing.assert_array_equal(
        np.asarray(doubled.elem_area), 2 * np.asarray(mesh.elem_area)
    )
