"""S.2 gate: the sharded-mesh build + export (:mod:`fesom_jax.shard_mesh`).

Builds the per-device CORE2 mesh from ``dist_2``/``dist_4`` and asserts:
  * the **serial ``npes==1`` sharded mesh is array-equal to the dense ``Mesh``**
    (the no-op invariant — the single-device path is untouched);
  * connectivity is remapped to valid local indices (``< Lmax`` or ``-1``), with
    **no ``-1`` in owned elements** (the gather-on-sentinel safety proof) and the
    remap is invertible back to the global connectivity;
  * the broadcast exchange map carries each halo lane's **owner gid** (interior
    lanes are identity);
  * masks are consistent and padded lanes are inert;
  * ``export → load`` round-trips losslessly.

SKIPs cleanly when the dense mesh / partitions are absent.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fesom_jax import partit, shard_mesh
from fesom_jax.mesh import Mesh, load_mesh

CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing",
)

_KINDS = ("nod", "elem", "edge")
_DENSE_SKIP = {"nod_in_elem2D", "nod_in_elem2D_offsets"}   # IC-only, omitted


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    return load_mesh(CORE2_MESH)


def _build(mesh: Mesh, npes: int) -> shard_mesh.ShardedMesh:
    part = partit.read_partition(CORE2_DIST, npes)
    return shard_mesh.build_sharded_mesh(mesh, part)


def _gid_padded(part: partit.Partition, mylists, n_local, Lmax: int) -> np.ndarray:
    """[P, Lmax] global ids per local lane (-1 on pad), for exchange checks."""
    P = len(mylists)
    g = np.full((P, Lmax), -1, dtype=np.int64)
    for d in range(P):
        g[d, : mylists[d].size] = mylists[d]
    return g


# --------------------------------------------------------------------------
# The no-op invariant: serial sharded mesh == dense Mesh
# --------------------------------------------------------------------------
@avail
def test_serial_noop_array_equal(mesh):
    sm = shard_mesh.build_serial_sharded_mesh(mesh)
    assert sm.P == 1
    assert sm.Lmax == {"nod": mesh.nod2D, "elem": mesh.elem2D, "edge": mesh.edge2D}
    for f in dataclasses.fields(Mesh):
        name = f.name
        if f.metadata.get("static") or name in _DENSE_SKIP:
            continue
        dense = np.asarray(getattr(mesh, name))
        got = sm.fields[name] if name in shard_mesh.REPLICATED_FIELDS else sm.fields[name][0]
        assert got.shape == dense.shape, f"{name}: {got.shape} != {dense.shape}"
        assert np.array_equal(got, dense), f"{name} differs from dense mesh"
    # serial masks: everything owned & valid, identity exchange
    for k in _KINDS:
        assert sm.owned_mask[k].all() and sm.valid_mask[k].all()
        src_dev, src_lane = sm.exchange[k]
        assert (src_dev == 0).all()
        assert np.array_equal(src_lane[0], np.arange(sm.Lmax[k]))


# --------------------------------------------------------------------------
# Connectivity: remapped, in range, owned-safe, invertible
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_connectivity_in_range_and_owned_safe(mesh, npes):
    sm = _build(mesh, npes)
    Ln, Le = sm.Lmax["nod"], sm.Lmax["elem"]
    # node-valued connectivity: every index in [-1, Lmax_nod)
    for name in ("elem_nodes", "edges"):
        v = sm.fields[name]
        assert ((v >= -1) & (v < Ln)).all(), f"{name} out of range"
    # elem-valued connectivity: every index in [-1, Lmax_elem)
    for name in ("edge_tri", "edge_up_dn_tri"):
        v = sm.fields[name]
        assert ((v >= -1) & (v < Le)).all(), f"{name} out of range"
    # owned elements have all 3 vertices local (no -1) → no owned output depends
    # on a sentinel gather
    en = sm.fields["elem_nodes"]
    for d in range(npes):
        md = int(sm.counts["myDim_elem"][d])
        assert (en[d, :md] >= 0).all(), f"dev {d}: owned elem has a -1 vertex"


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_connectivity_remap_invertible(mesh, npes):
    """Local connectivity, mapped back through myList, equals the global one."""
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    en_g = np.asarray(mesh.elem_nodes)
    for d in range(npes):
        n_e = int(sm.counts["n_local_elem"][d])
        en_loc = sm.fields["elem_nodes"][d, :n_e]              # [n_e, 3] local node idx
        myl_n = part.myList_nod2D[d]
        myl_e = part.myList_elem2D[d]
        true_g = en_g[myl_e[:n_e]]                            # [n_e, 3] global
        valid = en_loc >= 0
        recon = np.where(valid, myl_n[np.where(valid, en_loc, 0)], -1)
        # every locally-mappable vertex must reconstruct the true global id
        assert np.array_equal(recon[valid], true_g[valid])


# --------------------------------------------------------------------------
# Exchange map: halo lanes carry the owner gid; interior is identity
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_exchange_carries_owner_gid(mesh, npes):
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    mylist = {"nod": part.myList_nod2D, "elem": part.myList_elem2D, "edge": part.myList_edge2D}
    for k in _KINDS:
        Lmax = sm.Lmax[k]
        src_dev, src_lane = sm.exchange[k]
        assert ((src_dev >= 0) & (src_dev < npes)).all()
        assert ((src_lane >= 0) & (src_lane < Lmax)).all()
        gid = _gid_padded(part, mylist[k], sm.counts[f"n_local_{k}"], Lmax)
        src_gid = gid[src_dev, src_lane]
        valid = sm.valid_mask[k]
        # on every valid lane the source carries the SAME global id (identity on
        # interior, owner's value on halo) — the broadcast moves entity g's value.
        assert np.array_equal(gid[valid], src_gid[valid])
        # interior lanes are strict identity (never overwrite owned data)
        owned = sm.owned_mask[k]
        self_dev = np.broadcast_to(np.arange(npes)[:, None], (npes, Lmax))
        lane = np.broadcast_to(np.arange(Lmax)[None, :], (npes, Lmax))
        assert (src_dev[owned] == self_dev[owned]).all()
        assert (src_lane[owned] == lane[owned]).all()
        # halo lanes are sourced from a DIFFERENT device
        halo = sm.valid_mask[k] & ~sm.owned_mask[k]
        assert (src_dev[halo] != self_dev[halo]).all()


# --------------------------------------------------------------------------
# Masks + padded-lane inertness
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_masks_and_padding(mesh, npes):
    sm = _build(mesh, npes)
    for k in _KINDS:
        om, vm = sm.owned_mask[k], sm.valid_mask[k]
        assert (om <= vm).all()                                   # owned ⊆ valid
        assert om.sum() == int(sm.counts[f"myDim_{k}"].sum())
        assert vm.sum() == int(sm.counts[f"n_local_{k}"].sum())
        # contiguous prefix: True exactly on [0:count)
        for d in range(npes):
            assert vm[d, : int(sm.counts[f"n_local_{k}"][d])].all()
            assert not vm[d, int(sm.counts[f"n_local_{k}"][d]):].any()
    # padded float lanes are finite & nonzero (safe denominators); connectivity pad = -1
    for d in range(npes):
        ne = int(sm.counts["n_local_elem"][d])
        if ne < sm.Lmax["elem"]:
            assert (sm.fields["elem_nodes"][d, ne:] == -1).all()
            assert np.isfinite(sm.fields["elem_area"][d, ne:]).all()


# --------------------------------------------------------------------------
# export → load round-trip (lossless serialization)
# --------------------------------------------------------------------------
def _tiny_sharded_mesh() -> shard_mesh.ShardedMesh:
    P = 2
    Lmax = {"nod": 3, "elem": 4, "edge": 5}
    rng = np.arange
    fields = {
        "depth": rng(P * 3, dtype=np.float64).reshape(P, 3),
        "elem_nodes": np.array([[[0, 1, 2], [1, 2, 0], [-1, -1, -1], [-1, -1, -1]],
                                [[0, 1, 2], [2, 1, 0], [0, 2, 1], [-1, -1, -1]]], np.int32),
        "zbar": np.linspace(0.0, -50.0, 6),
    }
    owned = {k: np.zeros((P, Lmax[k]), bool) for k in _KINDS}
    valid = {k: np.zeros((P, Lmax[k]), bool) for k in _KINDS}
    for k in _KINDS:
        owned[k][:, :2] = True
        valid[k][:, :3] = True
    exch = {k: (np.zeros((P, Lmax[k]), np.int32), np.zeros((P, Lmax[k]), np.int32))
            for k in _KINDS}
    counts = {f"myDim_{k}": np.full(P, 2, np.int32) for k in _KINDS}
    counts.update({f"n_local_{k}": np.full(P, 3, np.int32) for k in _KINDS})
    counts["eXDim_elem"] = np.zeros(P, np.int32)
    return shard_mesh.ShardedMesh(
        P=P, Lmax=Lmax, nl=6, fields=fields, owned_mask=owned, valid_mask=valid,
        exchange=exch, counts=counts, nod2D=4, elem2D=6, edge2D=8, edge2D_in=7,
        myDim_edge2D_global=8, ocean_area=1.234e14)


def test_export_load_roundtrip(tmp_path):
    sm = _tiny_sharded_mesh()
    shard_mesh.export_sharded_mesh(sm, tmp_path / "bundle")
    rt = shard_mesh.load_sharded_mesh(tmp_path / "bundle")
    assert (rt.P, rt.nl, rt.nod2D, rt.elem2D, rt.edge2D) == \
           (sm.P, sm.nl, sm.nod2D, sm.elem2D, sm.edge2D)
    assert rt.Lmax == sm.Lmax
    assert rt.ocean_area == pytest.approx(sm.ocean_area, rel=1e-15)
    assert set(rt.fields) == set(sm.fields)
    for name in sm.fields:
        np.testing.assert_array_equal(rt.fields[name], sm.fields[name])
    for k in _KINDS:
        np.testing.assert_array_equal(rt.owned_mask[k], sm.owned_mask[k])
        np.testing.assert_array_equal(rt.valid_mask[k], sm.valid_mask[k])
        np.testing.assert_array_equal(rt.exchange[k][0], sm.exchange[k][0])
    assert set(rt.counts) == set(sm.counts)


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        shard_mesh.load_sharded_mesh(tmp_path / "nope")
