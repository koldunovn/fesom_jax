"""Write ushow / uterm-readable zarr for the unstructured FESOM mesh (Task B2 output).

`ushow` (the ncview-style unstructured viewer) reads a **point cloud**: per-node ``lon``/``lat``
coordinate arrays + data variables on a **node dimension**, identified by the xarray
``_ARRAY_DIMENSIONS`` attribute (it matches the dim names against its NODE/LAT/LON/DEPTH/TIME
lists — node = ``nod2``/``ncells``/``values``…, depth = ``nz``/``depth``…, time = ``time``).

The portable restart (:func:`zarr_output.write_restart`) is gid-keyed + sharded for a
device-count-portable RELOAD, NOT for viewing — it has no lon/lat and its fields are
``[P*Lmax]`` shards, so ushow can't read it. This writes the **viewer** product: a plain
zarr-v2 store (zarr-python 2.x writes ``.zarray``/``.zattrs`` natively; we add
``_ARRAY_DIMENSIONS`` ourselves, what xarray would do — so no xarray dependency) with global
node-indexed fields. One global field is held at a time (NG5 3-D leaf ≈ 4 GB), so it streams.

Layout (matches ushow's unstructured test fixture):
  lon[nod2], lat[nod2]                         — degrees, the node coordinates
  <2-D field>[nod2]                            — e.g. sst, sss, aice, speed, ssh
  <3-D field>[nz, nod2]                        — e.g. temp, salt (node-last; ushow keys on dim NAME)
"""
from __future__ import annotations

import numpy as np

NODE_DIM = "nod2"      # in ushow's NODE_NAMES
DEPTH_DIM = "nz"       # in ushow's DEPTH_NAMES
TIME_DIM = "time"      # in ushow's TIME_NAMES


def write_ushow_zarr(out_dir, lon, lat, *, fields2d=None, fields3d=None,
                     units=None, attrs=None):
    """Write an ushow-readable zarr store.

    ``lon``/``lat``: ``[N]`` node coordinates **in degrees**. ``fields2d``: ``{name: [N]}`` (surface
    / 2-D). ``fields3d``: ``{name: [N, nlev]}`` (stored transposed to ``[nlev, N]`` = ``[nz, nod2]``).
    ``units``: optional ``{name: str}``. ``attrs``: optional store-level metadata. Returns the path."""
    import zarr

    out_dir = str(out_dir)
    root = zarr.open_group(out_dir, mode="w")
    units = units or {}

    def _put(name, data, dims):
        data = np.ascontiguousarray(data)
        a = root.create_dataset(name, data=data, shape=data.shape, dtype=data.dtype,
                                chunks=data.shape, overwrite=True)
        a.attrs["_ARRAY_DIMENSIONS"] = list(dims)
        if name in units:
            a.attrs["units"] = units[name]
        a.attrs["long_name"] = name

    _put("lon", np.asarray(lon, np.float64), [NODE_DIM])
    _put("lat", np.asarray(lat, np.float64), [NODE_DIM])
    for name, data in (fields2d or {}).items():
        _put(name, np.asarray(data, np.float32), [NODE_DIM])
    for name, data in (fields3d or {}).items():
        _put(name, np.asarray(data, np.float32).T, [DEPTH_DIM, NODE_DIM])   # [nlev, N]
    if attrs:
        root.attrs.update(attrs)
    return out_dir


def node_lonlat(mesh):
    """Node ``(lon, lat)`` in **degrees** from a mesh (``geo_coord_nod2D`` is radians)."""
    geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64) / np.pi * 180.0
    return geo[:, 0], geo[:, 1]
