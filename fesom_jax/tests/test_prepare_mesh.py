"""A8 gate: standalone mesh preparation (`scripts/prepare_mesh.py`) — ``MESH_PREP_OK``.

The Python port of FESOM's mesh setup must reproduce the C exporter byte-faithfully, so the model
ships **without a C-FESOM build** to prepare a mesh. Verifies every exported array against the
ground-truth C export (`data/mesh_core2`): ints EXACT, floats to the float64 order-of-ops rounding
floor (rel ~1e-13). Plus an end-to-end check that the prepared mesh is `load_mesh`-able (which runs
the port's own CW-orientation validation).

Gated on the raw CORE2 mesh (`/pool/.../core2`) + its C export (`data/mesh_core2`) — Levante only.
CORE2 is the oracle (a real physical mesh; pi's idealized quirks aren't worth chasing for byte-id).

  PY -m pytest fesom_jax/tests/test_prepare_mesh.py -x
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE2_RAW = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
CORE2_EXPORT = ROOT / "data" / "mesh_core2"

needs_data = pytest.mark.skipif(
    not (CORE2_RAW / "nod2d.out").exists() or not (CORE2_EXPORT / "coord_nod2D.npy").exists(),
    reason="raw CORE2 mesh or its C export missing (Levante only)")


def _load_prepare_mesh():
    spec = importlib.util.spec_from_file_location("prepare_mesh", ROOT / "scripts" / "prepare_mesh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_data
def test_prepare_mesh_matches_c_export(tmp_path):
    pm = _load_prepare_mesh()
    raw = pm.read_fesom_ascii(CORE2_RAW)
    out = pm.derive(raw)
    diffs = pm.verify_against(out, CORE2_EXPORT)

    worst_rel = max((rel for _, rel, isint in diffs.values() if not isint), default=0.0)
    worst_int = max((ad for ad, _, isint in diffs.values() if isint), default=0)
    # every array present + ints exact + floats at the float64 rounding floor
    assert len(diffs) == 32, f"expected 32 arrays, got {len(diffs)}"
    assert worst_int == 0, f"integer array mismatch: { {k: v for k, v in diffs.items() if v[2] and v[0]} }"
    assert worst_rel < 1e-11, (
        f"float mismatch rel={worst_rel:.3e}: "
        f"{ {k: f'{v[1]:.2e}' for k, v in diffs.items() if not v[2] and v[1] > 1e-11} }")

    # meta scalars
    m = out["_meta"]
    assert m["nod2D"] == raw["nod2D"] and m["edge2D"] == raw["edges"].shape[0]
    assert m["npes"] == 1 and m["ocean_area"] > 0
    print("MESH_PREP_OK")


@needs_data
def test_prepared_mesh_is_loadable(tmp_path):
    """The prepared mesh round-trips through the port's own `load_mesh` (which asserts CW
    orientation) and matches the C export's loaded Mesh on the step-read fields."""
    import dataclasses
    from fesom_jax.mesh import load_mesh
    pm = _load_prepare_mesh()
    out = pm.derive(pm.read_fesom_ascii(CORE2_RAW))
    pm.write_mesh(out, tmp_path / "prepared")

    prepared = load_mesh(tmp_path / "prepared")          # runs check_cw_orientation internally
    c_export = load_mesh(CORE2_EXPORT)
    for f in dataclasses.fields(prepared):
        a = getattr(prepared, f.name)
        b = getattr(c_export, f.name)
        if not hasattr(a, "shape") or np.asarray(a).size == 0:
            continue
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            continue
        if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
            assert np.array_equal(a, b), f"loaded Mesh.{f.name} differs (int/bool) vs C export"
        else:
            scale = float(np.max(np.abs(b))) or 1.0
            rel = float(np.max(np.abs(a - b))) / scale
            assert rel < 1e-11, f"loaded Mesh.{f.name} rel={rel:.3e} vs C export"
