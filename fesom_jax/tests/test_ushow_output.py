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
