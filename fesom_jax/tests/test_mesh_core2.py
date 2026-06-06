"""Task 5.1 gate: the CORE2 mesh export loads and satisfies the SAME structural
invariants as pi — including the **clockwise-orientation** invariant (the pi↔CORE2
trap: CORE2's raw mesh is ~all CCW and only matches after the C ``orient_cw``).

We re-use the pi structural checks from ``test_mesh`` verbatim (no change to that
file) by importing them and applying them to a CORE2 ``Mesh`` fixture, and add
CORE2-specific pinned counts + a bit-for-bit-vs-export check. The test SKIPS until
the CORE2 mesh has been exported (``data/mesh_core2/``, Task 5.1 SLURM job).

Confirms the design claim that loading CORE2 needs **no** ``mesh.py``/``state.py``
change (full-cell, global zbar/Z; the ragged masks handle per-node depth): if a
change were needed, ``load_mesh`` or these invariants would fail here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from jax import tree_util

from fesom_jax import mesh as meshmod
from fesom_jax.mesh import Mesh, load_mesh
from fesom_jax.tests import test_mesh as tm  # reuse pi structural checks (read-only)

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"

pytestmark = pytest.mark.skipif(
    not CORE2_MESH_DIR.is_dir(),
    reason=f"CORE2 mesh export missing: {CORE2_MESH_DIR} (run Task 5.1 export job)",
)

# Hard CORE2 counts (mesh files + the export meta.txt) — pinned so a re-export
# regression is caught. orient_cw swapped 244654/244659 elements to CW (logged).
CORE2_COUNTS = dict(
    nod2D=126858, elem2D=244659, edge2D=371644, nl=48,
    edge2D_in=362333, myDim_edge2D=371644,
)
CORE2_OCEAN_AREA = 364354316635727.44


@pytest.fixture(scope="module")
def core2_mesh() -> Mesh:
    # load_mesh runs check_cw_orientation by default — a non-CW export raises here.
    return load_mesh(CORE2_MESH_DIR)


def test_scalar_counts(core2_mesh):
    for k, v in CORE2_COUNTS.items():
        assert getattr(core2_mesh, k) == v, f"{k}={getattr(core2_mesh, k)} != {v}"
    assert isinstance(core2_mesh.nod2D, int)
    assert core2_mesh.edges.shape[0] == core2_mesh.edge2D       # self-consistent
    assert core2_mesh.ocean_area == pytest.approx(CORE2_OCEAN_AREA, rel=1e-12)


def test_arrays_equal_export_bit_for_bit(core2_mesh):
    for field, fname in meshmod.mesh_field_files().items():
        raw = np.load(CORE2_MESH_DIR / fname)
        got = np.asarray(getattr(core2_mesh, field))
        expect = raw.astype(np.int32) if raw.dtype.kind in "iu" else raw.astype(np.float64)
        assert got.shape == expect.shape, f"{field}: {got.shape} != {expect.shape}"
        assert np.array_equal(got, expect), f"{field}: values differ from export"


# --- the orientation gate (the reason this task got flagged) ---------------
def test_orientation_all_cw(core2_mesh):
    """Every CORE2 triangle is CW — the C ``orient_cw`` normalized them before
    geometry. CORE2's raw mesh is ~all CCW, so this proves the swap survived the
    export → load round-trip (else: wrong stiffness sign → Aleutian blow-up)."""
    r = meshmod.check_cw_orientation(core2_mesh.coord_nod2D, core2_mesh.elem_nodes)
    assert r.shape == (core2_mesh.elem2D,)
    assert (r < 0).all(), f"{int((r >= 0).sum())} non-CW CORE2 triangles"


# --- reuse pi structural invariants verbatim on the CORE2 mesh -------------
def test_shapes_consistent(core2_mesh):
    tm.test_shapes_consistent(core2_mesh)


def test_indices_in_range(core2_mesh):
    tm.test_indices_already_zero_based_and_in_range(core2_mesh)


def test_edge_tri_classification(core2_mesh):
    tm.test_edge_tri_classification(core2_mesh)


def test_csr_well_formed(core2_mesh):
    tm.test_csr_well_formed_and_consistent(core2_mesh)


def test_levels_no_cavity(core2_mesh):
    # CORE2 has no cavities either (ulevels==1 everywhere), per fesom_mesh.c.
    tm.test_levels_no_cavity(core2_mesh)


def test_masks_match_level_counts(core2_mesh):
    tm.test_masks_match_level_counts(core2_mesh)


def test_geometry_value_ranges(core2_mesh):
    tm.test_geometry_value_ranges(core2_mesh)


def test_dtypes(core2_mesh):
    tm.test_dtypes(core2_mesh)


def test_pytree_roundtrip(core2_mesh):
    leaves, treedef = tree_util.tree_flatten(core2_mesh)
    assert all(hasattr(l, "shape") for l in leaves)
    rebuilt = tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.nod2D == core2_mesh.nod2D and rebuilt.elem2D == core2_mesh.elem2D
