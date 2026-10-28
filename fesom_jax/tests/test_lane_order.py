"""``gpu.lane_order`` (:mod:`fesom_jax.lane_order`): the lanes inside the shards reordered, the global numbering kept.

On the packaged pi mesh (dist_4): every rank keeps its sets and its owned / halo blocks, the communicators' lanes
follow the lists, the sharded connectivity names the same global nodes and elements, the edges' lane distance
falls, and the production driver (4 fake CPU devices, the ``core2_gpu.yaml`` options) writes the same restart with
the reordered lanes up to the order of its sums."""

from __future__ import annotations

import numpy as np
import pytest

from fesom_jax import partit, shard_mesh
from fesom_jax.lane_order import edge_lane_distance, reorder_lanes
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh


@pytest.fixture(scope="module")
def pi():
    mesh = load_mesh()
    part = partit.read_partition(DEFAULT_PI_MESH_DIR, 4)
    return mesh, part, reorder_lanes(part, mesh, "rcm")


def _blocks(part, kind, d):
    lst = np.asarray(getattr(part, f"myList_{kind}2D")[d])
    md = int(getattr(part, f"myDim_{kind}2D")[d])
    if kind == "elem":
        ed = int(part.eDim_elem2D[d])
        return [lst[:md], lst[md:md + ed], lst[md + ed:]]
    return [lst[:md], lst[md:]]


def test_reorder_keeps_blocks_and_communicators(pi):
    mesh, old, new = pi
    for d in range(4):
        for kind in ("nod", "elem", "edge"):
            for a, b in zip(_blocks(old, kind, d), _blocks(new, kind, d)):
                assert a.size == b.size and np.array_equal(np.sort(a), np.sort(b)), (kind, d)
        for kind, coms in (("nod", ("com_nod2D",)), ("elem", ("com_elem2D", "com_elem2D_full"))):
            lo, ln = (np.asarray(getattr(p, f"myList_{kind}2D")[d]) for p in (old, new))
            for c in coms:
                co, cn = getattr(old, c)[d], getattr(new, c)[d]
                for fld in ("rlist", "slist"):
                    np.testing.assert_array_equal(ln[np.asarray(getattr(cn, fld))], lo[np.asarray(getattr(co, fld))])
    assert np.all(edge_lane_distance(new, mesh) <= edge_lane_distance(old, mesh))


def test_sharded_connectivity_names_the_same_globals(pi):
    mesh, _, new = pi
    sm = shard_mesh.build_sharded_mesh(mesh, new)
    for d in range(4):
        ln = np.asarray(new.myList_nod2D[d])
        le = np.asarray(new.myList_edge2D[d])
        lel = np.asarray(new.myList_elem2D[d])
        loc = sm.fields["edges"][d, :le.size]
        ok = (loc >= 0).all(axis=1)
        np.testing.assert_array_equal(ln[loc[ok]], np.asarray(mesh.edges)[le][ok])
        loc = sm.fields["elem_nodes"][d, :lel.size]
        ok = (loc >= 0).all(axis=1)
        np.testing.assert_array_equal(ln[loc[ok]], np.asarray(mesh.elem_nodes)[lel][ok])
        np.testing.assert_array_equal(sm.fields["area"][d, :ln.size], np.asarray(mesh.area)[ln])


def test_driver_lane_order_round_off(tmp_path):
    """Six steps in 3-step chunks through ``run_from_config`` with ``--lane-order rcm`` and ``none``: the restarts
    (global ids) agree to the round-off of the reordered sums."""
    from fesom_jax.tests.test_precision_shard import _cfg, _load, _run
    cfg = _cfg(tmp_path)
    out = {}
    for lo in ("none", "rcm"):
        stdout = _run(cfg, "--lane-order", lo, "--steps", "6", "--chunk-steps", "3",
                      "--restart-out", str(tmp_path / lo))
        out[lo] = _load(tmp_path / lo)
    assert "lane_order=rcm" in stdout
    a, b = out["none"], out["rcm"]
    assert a.keys() == b.keys()
    for k in a:
        if not np.issubdtype(a[k].dtype, np.floating):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
            continue
        scale = max(float(np.max(np.abs(a[k]))), 1e-30)
        assert np.max(np.abs(a[k] - b[k])) <= 1e-8 * scale, (k, float(np.max(np.abs(a[k] - b[k]))), scale)
