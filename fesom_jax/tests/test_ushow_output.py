"""Unit test for the ushow/uterm zarr writer (Task B2 output).

Verifies the on-disk layout ushow keys on: a zarr-v2 store with ``lon``/``lat`` + each field
carrying an ``_ARRAY_DIMENSIONS`` attribute whose names land in ushow's NODE/DEPTH lists
(``nod2`` / ``nz``). The viewer itself loading the store is the integration check; this is the
fast structural guard.
"""
from pathlib import Path

import numpy as np
import pytest

from fesom_jax.ushow_output import write_ushow_zarr

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"


def test_ushow_zarr_layout(tmp_path):
    import zarr
    n, nz = 50, 4
    lon = np.linspace(-180, 180, n)
    lat = np.linspace(-80, 80, n)
    sst = np.cos(np.deg2rad(lat))
    temp = np.tile(sst[:, None], (1, nz))                 # [n, nz]
    out = write_ushow_zarr(tmp_path / "v.zarr", lon, lat,
                           fields2d={"sst": sst}, fields3d={"temp": temp}, units={"sst": "degC"})

    root = zarr.open_group(str(out), mode="r")
    # coords + 2-D field carry the node dim; 3-D is [nz, nod2] (node last, transposed)
    assert list(root["lon"].attrs["_ARRAY_DIMENSIONS"]) == ["nod2"]
    assert list(root["lat"].attrs["_ARRAY_DIMENSIONS"]) == ["nod2"]
    assert list(root["sst"].attrs["_ARRAY_DIMENSIONS"]) == ["nod2"]
    assert list(root["temp"].attrs["_ARRAY_DIMENSIONS"]) == ["nz", "nod2"]
    assert root["sst"].attrs["units"] == "degC"
    # values round-trip; 3-D stored transposed to [nz, n]
    assert root["lon"].shape == (n,) and root["sst"].shape == (n,)
    assert root["temp"].shape == (nz, n)
    np.testing.assert_array_equal(np.asarray(root["sst"]).astype(np.float64), sst.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(root["temp"])[:, 0],
                                  temp[0, :].astype(np.float32))


@pytest.mark.skipif(not CORE2_MESH.exists(), reason="CORE2 mesh not present")
def test_ushow_sharded_owned_only(tmp_path):
    """The DIRECT gather-free writer: each owner lane writes its real (lon, lat, value); non-owned
    (halo + pad) lanes get the sentinel coord + FILL. Scattering the owned lanes by gid must
    recover the global field exactly (the same gather-free guarantee as the restart, viewer layout)."""
    import zarr

    from fesom_jax import partit, shard_mesh
    from fesom_jax.mesh import load_mesh
    from fesom_jax.ushow_output import (FILL, _SENT_LAT, _folded_gid_owned_nod,
                                        node_lonlat, write_ushow_sharded)
    mesh = load_mesh(CORE2_MESH)
    part = partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    gid, owned = _folded_gid_owned_nod(part, sm)
    glon, glat = node_lonlat(mesh)

    # a known folded field: each owner lane carries its node's latitude (so we can verify values)
    field = np.where(owned, glat[np.clip(gid, 0, None)], -777.0)
    out = write_ushow_sharded(tmp_path / "s.zarr", {"latfield": field}, sm, part, mesh)

    root = zarr.open_group(str(out), mode="r")
    assert list(root["latfield"].attrs["_ARRAY_DIMENSIONS"]) == ["nod2"]
    lat = root["lat"][:]; lon = root["lon"][:]; lf = root["latfield"][:]
    # owned lanes: real coord + value; non-owned: sentinel coord + FILL
    assert np.allclose(lat[owned], glat[gid[owned]]) and np.allclose(lon[owned], glon[gid[owned]])
    assert np.allclose(lf[owned], glat[gid[owned]].astype(np.float32), atol=1e-4)
    assert np.all(lat[~owned] == _SENT_LAT) and np.all(lf[~owned] == np.float32(FILL))
    # reconstruct: scatter owned lanes by gid ⇒ every global node recovered exactly once
    recon = np.full(mesh.nod2D, np.nan, np.float64)
    recon[gid[owned]] = lf[owned]
    assert not np.isnan(recon).any(), "some node was not written by its owner"
    assert np.allclose(recon, glat.astype(np.float32), atol=1e-4)


@pytest.mark.skipif(not CORE2_MESH.exists(), reason="CORE2 mesh not present")
def test_global_zarr_canonical_partition_independent(tmp_path):
    """The CANONICAL writer (:func:`write_global_zarr`): owned lanes scatter into dense global
    ``[nod2]`` / ``[nz, nod2]`` order, so the store is node-indexed directly (no unfold), chunk-
    controlled, and **partition-INDEPENDENT** — a different device count gives a byte-identical store
    (the FESOM3 ``dist_2 ≡ dist_8`` property)."""
    import zarr

    from fesom_jax import partit, shard_mesh
    from fesom_jax.mesh import load_mesh
    from fesom_jax.ushow_output import node_lonlat, write_global_zarr
    mesh = load_mesh(CORE2_MESH)
    nod2D, nz = mesh.nod2D, 3

    def canon(npes, out):
        part = partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, npes)
        sm = shard_mesh.build_sharded_mesh(mesh, part)
        from fesom_jax.ushow_output import _folded_gid_owned_nod
        gid, owned = _folded_gid_owned_nod(part, sm)
        gsafe = np.clip(gid, 0, None)
        # folded fields whose owned lanes carry the GLOBAL node id (2-D) and id+level/1000 (3-D)
        f2 = np.where(owned, gsafe, -1).astype(np.float64)
        f3 = np.where(owned[:, None],
                      gsafe[:, None] + np.arange(nz)[None, :] / 1000.0, -1).astype(np.float64)
        write_global_zarr(out, {"id2d": f2, "id3d": f3}, sm, part, mesh,
                          chunk_horiz=20000, chunk_vert=2)
        return zarr.open_group(str(out), mode="r")

    ra = canon(4, tmp_path / "p4.zarr")
    # (1) canonical order: id2d[node] == node EXACTLY (node ids < 2^24 are exact in float32)
    np.testing.assert_array_equal(np.asarray(ra["id2d"]), np.arange(nod2D, dtype=np.float32))
    id3d = np.asarray(ra["id3d"])                                   # [nz, nod2]
    assert id3d.shape == (nz, nod2D)
    for k in range(nz):
        np.testing.assert_array_equal(id3d[k], (np.arange(nod2D) + k / 1000.0).astype(np.float32))
    # (2) user chunking honoured + node-indexed dims
    assert ra["id2d"].chunks == (20000,) and ra["id3d"].chunks == (2, 20000)
    assert list(ra["id3d"].attrs["_ARRAY_DIMENSIONS"]) == ["nz", "nod2"]
    glon, _ = node_lonlat(mesh)
    np.testing.assert_allclose(np.asarray(ra["lon"]), glon)
    # (3) partition-independence: a DIFFERENT npes ⇒ byte-identical store
    rb = canon(8, tmp_path / "p8.zarr")
    for v in ("lon", "lat", "id2d", "id3d"):
        np.testing.assert_array_equal(np.asarray(ra[v]), np.asarray(rb[v]))
