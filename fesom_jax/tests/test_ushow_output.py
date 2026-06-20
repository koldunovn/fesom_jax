"""Unit test for the ushow/uterm zarr writer (Task B2 output).

Verifies the on-disk layout ushow keys on: a zarr-v2 store with ``lon``/``lat`` + each field
carrying an ``_ARRAY_DIMENSIONS`` attribute whose names land in ushow's NODE/DEPTH lists
(``nod2`` / ``nz``). The viewer itself loading the store is the integration check; this is the
fast structural guard.
"""
import numpy as np

from fesom_jax.ushow_output import write_ushow_zarr


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
