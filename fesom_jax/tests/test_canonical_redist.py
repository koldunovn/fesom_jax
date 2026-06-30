"""Multi-process canonical writers (:mod:`fesom_jax.canonical_redist`): the no-single-node-gather
``redistribute`` (ragged_all_to_all) + the ``all_gather`` baseline, vs the single-process host-gather.

Runs on CPU fake-devices (one process, P devices): the collective + offset maps + chunk writes are fully
exercised; only the true multi-HOST disk split (each process writes only its local shards) needs a real
multi-node smoke. Asserts all three methods produce a BYTE-identical canonical store, that it is
node-indexed + correctly scattered (value==gid round-trips), partition-independent (P=2 ≡ P=4), and
user-chunked.

Standalone (4 fake devices):
  XLA_FLAGS=--xla_force_host_platform_device_count=4 PY -m pytest fesom_jax/tests/test_canonical_redist.py
"""
from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

pytest.importorskip("zarr")

from fesom_jax import halo, partit, shard_mesh
from fesom_jax.mesh import load_mesh
from fesom_jax.ushow_output import _folded_gid_owned_nod, write_global_zarr

ROOT = Path(__file__).resolve().parents[2]
PI_MESH = ROOT / "data" / "mesh_pi"
NDEV = len(jax.devices())

avail = pytest.mark.skipif(not PI_MESH.is_dir(), reason="pi mesh missing")


def _folded_sharded_field(values_global, gid, owned, P, Lmax, nlev=0):
    """Build a folded ``[P*Lmax(, nlev)]`` sharded jax.Array whose OWNED lanes carry
    ``values_global[gid]`` (+ a per-level ramp for 3-D), junk elsewhere."""
    PL = P * Lmax
    gsafe = np.clip(gid, 0, None)
    base = np.where(owned, values_global[gsafe], -1.0).astype(np.float64)
    if nlev:
        arr = base[:, None] + (np.where(owned, 1.0, 0.0)[:, None]
                               * np.arange(nlev)[None, :] / 1000.0)
    else:
        arr = base
    jmesh = halo.device_mesh("p", devices=jax.devices()[:P])
    from jax.sharding import NamedSharding, PartitionSpec
    sharding = NamedSharding(jmesh, PartitionSpec("p"))
    return jax.make_array_from_callback(arr.shape, sharding, lambda idx, a=arr: a[idx])


def _canon(mesh, npes, tmp, method, C):
    """Write a value==gid field (2-D + 3-D) at ``npes`` via ``method``; return the opened store."""
    import zarr
    part = partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    gid, owned = _folded_gid_owned_nod(part, sm)
    P, Lmax = int(sm.P), int(sm.Lmax["nod"])
    ids = np.arange(mesh.nod2D, dtype=np.float64)
    f2 = _folded_sharded_field(ids, gid, owned, P, Lmax, nlev=0)
    f3 = _folded_sharded_field(ids, gid, owned, P, Lmax, nlev=3)
    out = tmp / f"{method}_p{npes}.zarr"
    write_global_zarr(out, {"id2d": f2, "id3d": f3}, sm, part, mesh,
                      method=method, chunk_horiz=C, chunk_vert=2)
    return zarr.open_group(str(out), mode="r")


# ragged_all_to_all (the 'redistribute' transport) is GPU-only — XLA:CPU has no ThunkEmitter for it —
# so on CPU fake-devices we cover host_gather + all_gather; redistribute is validated on GPU (where
# CORE2's single-node 4-GPU / one-process setup exercises the real collective).
ON_GPU = jax.devices()[0].platform == "gpu"
CPU_METHODS = ["host_gather", "all_gather"]
ALL_METHODS = CPU_METHODS + (["redistribute"] if ON_GPU else [])


@avail
@pytest.mark.skipif(NDEV < 4, reason="needs >=4 (fake-)devices")
def test_methods_agree_canonical_and_partition_independent(tmp_path):
    mesh = load_mesh(PI_MESH)
    nod2D, nz, C = mesh.nod2D, 3, 1000
    ids = np.arange(nod2D, dtype=np.float32)

    # every available method at P=4 ⇒ canonical (value==gid) AND byte-identical to each other
    stores = {m: _canon(mesh, 4, tmp_path, m, C) for m in ALL_METHODS}
    for m, r in stores.items():
        np.testing.assert_array_equal(np.asarray(r["id2d"]), ids), m
        id3d = np.asarray(r["id3d"])
        assert id3d.shape == (nz, nod2D), (m, id3d.shape)
        for k in range(nz):
            np.testing.assert_array_equal(id3d[k], (np.arange(nod2D) + k / 1000.0).astype(np.float32))
        assert r["id2d"].chunks == (C,) and r["id3d"].chunks == (2, C)
        assert r.attrs["layout"] == "canonical_global"
    base = stores["host_gather"]
    for m in ALL_METHODS[1:]:
        for v in ("lon", "lat", "id2d", "id3d"):
            np.testing.assert_array_equal(np.asarray(base[v]), np.asarray(stores[m][v])), (m, v)


@avail
@pytest.mark.skipif(NDEV < 4, reason="needs >=4 (fake-)devices")
def test_partition_independent_bytes(tmp_path):
    """The fixed-C chunk grid ⇒ the on-disk store is byte-identical at any device count (only WHICH
    device writes a chunk changes). Uses the CPU-safe all_gather method (redistribute checked on GPU)."""
    mesh = load_mesh(PI_MESH)
    m = "redistribute" if ON_GPU else "all_gather"
    a = _canon(mesh, 4, tmp_path, m, 500)
    b = _canon(mesh, 2, tmp_path, m, 500)
    assert a["id2d"].chunks == b["id2d"].chunks == (500,)
    for v in ("lon", "lat", "id2d", "id3d"):
        np.testing.assert_array_equal(np.asarray(a[v]), np.asarray(b[v])), v
