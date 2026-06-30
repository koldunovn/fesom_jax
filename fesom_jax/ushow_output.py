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

import jax
import numpy as np

NODE_DIM = "nod2"      # in ushow's NODE_NAMES
DEPTH_DIM = "nz"       # in ushow's DEPTH_NAMES
TIME_DIM = "time"      # in ushow's TIME_NAMES


def write_ushow_zarr(out_dir, lon, lat, *, fields2d=None, fields3d=None,
                     units=None, attrs=None, chunk_horiz=None, chunk_vert=None):
    """Write an ushow-readable zarr store.

    ``lon``/``lat``: ``[N]`` node coordinates **in degrees**. ``fields2d``: ``{name: [N]}`` (surface
    / 2-D). ``fields3d``: ``{name: [N, nlev]}`` (stored transposed to ``[nlev, N]`` = ``[nz, nod2]``).
    ``units``: optional ``{name: str}``. ``attrs``: optional store-level metadata.

    ``chunk_horiz`` / ``chunk_vert``: optional **user-controlled** chunk sizes along the node
    (``nod2``) and depth (``nz``) axes (the FESOM3 ``chunk_horiz``/``chunk_vert`` knobs); ``None`` =
    one chunk per axis (the prior behaviour). Chunking is independent of any partition. Returns the
    path."""
    import zarr

    out_dir = str(out_dir)
    root = zarr.open_group(out_dir, mode="w")
    units = units or {}

    def _chunks(dims, shape):
        ch = list(shape)
        for i, d in enumerate(dims):
            if d == NODE_DIM and chunk_horiz:
                ch[i] = min(int(chunk_horiz), shape[i])
            elif d == DEPTH_DIM and chunk_vert:
                ch[i] = min(int(chunk_vert), shape[i])
        return tuple(ch)

    def _put(name, data, dims):
        data = np.ascontiguousarray(data)
        a = root.create_dataset(name, data=data, shape=data.shape, dtype=data.dtype,
                                chunks=_chunks(dims, data.shape), overwrite=True)
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


# A sentinel coord for non-owned (halo + pad) lanes: deep Antarctic interior (no ocean), so a
# masked-out point there is never the nearest-neighbour of any real ocean target (ushow regrids
# by NN within an influence radius). Paired with FILL so ushow masks the value too — FILL is set
# ABOVE ushow's INVALID_DATA_THRESHOLD (1e37) so it's excluded from BOTH the display AND the
# colour-range scan (1e20 = ushow's DEFAULT_FILL_VALUE is only masked in display, not the range).
_SENT_LON, _SENT_LAT, FILL = 0.0, -89.99, 1e38


def write_ushow_sharded(out_dir, fields, sm, part, mesh, *, attrs=None):
    """Write an ushow-readable zarr **directly from a sharded run, gather-free** (Task B2 output).

    ``fields``: ``{name: folded [P*Lmax_nod, …] array}`` — node diagnostic fields (e.g. ``sst``,
    ``speed``, or 3-D ``temp``), each a device-sharded ``jax.Array`` (production) or host numpy
    (test). Reuses the restart's per-shard write: rank 0 creates the metadata + the single
    ``lon``/``lat`` coordinate arrays (folded ``[P*Lmax_nod]``), then every process writes ONLY its
    addressable shards — **no ``all_gather``, no global on one device** (cost ≈ a restart write).

    **Why it's correct for a point-cloud viewer:** every lane carries its own ``lon``/``lat``. The
    UNIQUE-owner lane of each node writes the real coord + value; **non-owned lanes (halo + pad) are
    set to the Antarctic sentinel coord + ``FILL`` value**, so ushow's KDTree never lets them win a
    real ocean target (and masks them anyway). Result: each ocean node appears once, correctly —
    the same gather-free guarantee as :func:`write_state_zarr`, in viewer layout."""
    import zarr

    out_dir = str(out_dir)
    gid, owned = _folded_gid_owned_nod(part, sm)              # [P*Lmax_nod] each
    PL = gid.shape[0]
    Lmax = sm.Lmax["nod"]
    glon, glat = node_lonlat(mesh)                            # [nod2D] degrees
    safe = np.clip(gid, 0, None)                              # gid=-1 (pad) → index 0; overwritten below
    lon = np.where(owned, glon[safe], _SENT_LON).astype(np.float64)
    lat = np.where(owned, glat[safe], _SENT_LAT).astype(np.float64)

    is0 = (jax.process_index() == 0)
    if is0:
        root = zarr.open_group(out_dir, mode="w")
        zlon = root.create_dataset("lon", shape=(PL,), chunks=(Lmax,), dtype="f8", overwrite=True)
        zlat = root.create_dataset("lat", shape=(PL,), chunks=(Lmax,), dtype="f8", overwrite=True)
        zlon[:] = lon; zlon.attrs["_ARRAY_DIMENSIONS"] = [NODE_DIM]
        zlat[:] = lat; zlat.attrs["_ARRAY_DIMENSIONS"] = [NODE_DIM]
        for name, arr in fields.items():
            shp = tuple(int(s) for s in arr.shape)
            nlev = shp[1] if len(shp) > 1 else 0
            store_shape = (nlev, PL) if nlev else (PL,)
            store_chunks = (nlev, Lmax) if nlev else (Lmax,)
            z = root.create_dataset(name, shape=store_shape, chunks=store_chunks,
                                    dtype="f4", fill_value=FILL, overwrite=True)
            z.attrs["_ARRAY_DIMENSIONS"] = ([DEPTH_DIM, NODE_DIM] if nlev else [NODE_DIM])
        root.attrs.update(nod2D=int(sm.nod2D), P=int(sm.P), **(attrs or {}))

    _barrier("ushow_meta")
    root = zarr.open_group(out_dir, mode="a")
    for name, arr in fields.items():
        z = root[name]
        for lane_slice, data in _addressable_lane_chunks(arr):
            o = owned[lane_slice]                            # [chunk] bool for these lanes
            masked = np.where(o.reshape((-1,) + (1,) * (data.ndim - 1)), data, FILL).astype(np.float32)
            if data.ndim == 1:
                z[lane_slice] = masked                       # [P*Lmax] node-only field
            else:
                z[:, lane_slice] = masked.T                  # [nlev, P*Lmax]: transpose [chunk,nlev]→[nlev,chunk]
    _barrier("ushow_data")
    return out_dir


def write_global_zarr(out_dir, fields, sm, part, mesh, *, method="auto", chunk_horiz=None,
                      chunk_vert=None, units=None, attrs=None):
    """Write **partition-independent, canonical global-node-order** output — the clean alternative to
    the folded :func:`write_ushow_sharded` (it mirrors the FESOM3 ``mod_io_zarr`` on-disk layout).

    ``fields``: ``{name: folded [P*Lmax_nod, …] array}`` (a sharded ``jax.Array`` or host numpy) — the
    SAME input as :func:`write_ushow_sharded`. Each node's UNIQUE-owner lane is placed (by global id)
    into a dense ``[nod2D]`` / ``[nod2D, nz]`` array in **canonical global order**. The store is
    **partition-independent** (byte-identical at any device count — the FESOM3 ``dist_2 ≡ dist_8``
    property), **directly node-indexed** (``xr.open_zarr`` → ``[nod2]``, no ``ushow_to_nodes`` unfold;
    ushow/pyfesom2 read it as-is), and **user-chunkable** (``chunk_horiz``/``chunk_vert``).

    ``method`` selects HOW the canonical array is assembled (all give a byte-identical store):

    * ``"auto"`` (default) — ``"host_gather"`` when single-process, else ``"all_gather"`` (the
      throughput bench found ``"redistribute"`` ~6× slower — see its bullet).
    * ``"host_gather"`` — SINGLE-process: ``np.asarray`` the folded array to this host + scatter by id
      (fine while one global field fits host RAM, ≤ FORCA20). Raises if multi-process.
    * ``"all_gather"`` — MULTI-process default: gather the field to every device, each process writes a
      disjoint chunk range (replicates the field per device; fine ≤ FORCA20).
    * ``"redistribute"`` — MULTI-process, **no single-node gather**: one ``ragged_all_to_all`` ships
      owned lanes to their chunk-owner device, which writes its chunks (see :mod:`canonical_redist`).
      ~6× slower (the ``ragged_all_to_all`` primitive cost) ⇒ use explicitly only when ``"all_gather"``
      would OOM (huge multi-node mesh)."""
    if method == "auto":
        # 1 process → host_gather (simplest). Multi-process → all_gather: the throughput bench found
        # the ragged_all_to_all 'redistribute' ~6× slower (the primitive itself is ~7.7 s/CORE2-payload
        # vs all_gather ~1.5 s), and all_gather's per-device replication is fine ≤ FORCA20. Use
        # 'redistribute' (no replication) explicitly only when all_gather would OOM (huge multi-node mesh).
        method = "host_gather" if jax.process_count() == 1 else "all_gather"
    if method == "redistribute":
        from .canonical_redist import write_redistribute
        return write_redistribute(out_dir, fields, sm, part, mesh, chunk_horiz=chunk_horiz,
                                  chunk_vert=chunk_vert, attrs=attrs)
    if method == "all_gather":
        from .canonical_redist import write_all_gather
        return write_all_gather(out_dir, fields, sm, part, mesh, chunk_horiz=chunk_horiz,
                                chunk_vert=chunk_vert, attrs=attrs)
    if method != "host_gather":
        raise ValueError(f"write_global_zarr: unknown method {method!r} (host_gather|redistribute|"
                         "all_gather|auto)")
    if jax.process_count() > 1:
        raise NotImplementedError(
            "method='host_gather' host-gathers owned lanes onto one node (single-process only); a "
            "multi-process run must use method='redistribute' (no single-node gather) or 'all_gather'.")
    gid, owned = _folded_gid_owned_nod(part, sm)              # [P*Lmax_nod] each
    nod2D = int(sm.nod2D)
    nowned = int(np.asarray(owned).sum())
    if nowned != nod2D:
        raise ValueError(f"owned lanes ({nowned}) != nod2D ({nod2D}); each node must have exactly one "
                         "owner — mesh/partition mismatch?")
    g_idx = gid[owned]                                        # global id of each owned lane
    glon, glat = node_lonlat(mesh)
    f2d, f3d = {}, {}
    for name, arr in fields.items():
        host = arr if isinstance(arr, np.ndarray) else np.asarray(arr)   # folded [P*Lmax, …] on host
        ow = host[owned]                                                 # owned lanes (drops halo/pad)
        g = np.zeros((nod2D,) + ow.shape[1:], dtype=host.dtype)
        g[g_idx] = ow                                                    # canonical [nod2D] / [nod2D, nlev]
        (f2d if g.ndim == 1 else f3d)[name] = g
    meta = {"layout": "canonical_global", "nod2D": nod2D, "P": int(sm.P)}
    if attrs:
        meta.update(attrs)
    return write_ushow_zarr(out_dir, glon, glat, fields2d=f2d, fields3d=f3d, units=units,
                            attrs=meta, chunk_horiz=chunk_horiz, chunk_vert=chunk_vert)


def _folded_gid_owned_nod(part, sm):
    from .zarr_output import _folded_gid_owned
    return _folded_gid_owned(part, sm, "nod")


def _barrier(tag):
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(tag)


def _addressable_lane_chunks(arr):
    """Yield ``(lane_slice, host_data)`` for each addressable shard of a folded ``[P*Lmax, …]``
    leaf — a device ``jax.Array`` (its ``addressable_shards``) or a plain host numpy (one chunk)."""
    if isinstance(arr, jax.Array):
        for shard in arr.addressable_shards:
            yield shard.index[0], np.asarray(shard.data)
    else:
        a = np.asarray(arr)
        yield slice(0, a.shape[0]), a
