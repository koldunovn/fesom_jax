"""Multi-process, no-single-node-gather **canonical** (partition-independent) Zarr output.

Companion to :func:`fesom_jax.ushow_output.write_global_zarr` (the SINGLE-process host-gather path):
these writers produce the SAME canonical global-node-order, user-chunked, partition-independent store —
but across MANY processes WITHOUT ever materialising a global field on one node. It is the JAX analogue
of FESOM3 ``mod_io_decomp``'s ``MPI_Alltoallv`` writer. Two methods (selectable for the throughput
comparison; both give a BYTE-identical store):

* ``redistribute`` — the **true no-gather** path. Each device ships only its OWNED lanes to the device
  that owns that gid's CHUNK, via one forward :func:`jax.lax.ragged_all_to_all`; it receives exactly the
  entities for its chunk range, scatters them into a local canonical buffer, and writes its (disjoint)
  chunks lock-free. Per-device peak ≈ O(chunk); data moves once. Mirrors :mod:`fesom_jax.halo`'s ragged
  halo exchange (same primitive + offset-map construction).
* ``all_gather`` — the simpler baseline. Every device gets the full global field (:func:`jax.lax.all_gather`,
  replicated); each process then writes its DISJOINT chunk range lock-free. Moves P× data + a full field
  per device (fine ≤ FORCA20; OOM-risk at NG5) — kept for the throughput comparison.

**Fixed chunk grid ⇒ partition-independent bytes.** Both use a P-independent ``C = chunk_horiz`` grid with
a *block* chunk→writer assignment, so a chunk's CONTENT (and thus its on-disk bytes) is identical at any
device count; only WHICH device writes a given chunk changes. **Output ⇒ no autodiff ⇒ the
``ragged_all_to_all`` grad-transpose bug does not apply** (forward transport only).

Scope: NODE fields (the monthly/daily output use case). Element-kind / restart reuse the same maps later.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec

from . import halo


# --------------------------------------------------------------------------
# Chunk→writer assignment + ragged redistribution maps (host, built once)
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RedistMaps:
    """Per-device ragged maps to redistribute folded ``[P*Lmax]`` owned lanes → canonical chunk order.

    For device ``d``: ships ``send_sizes[d,w]`` owned lanes to writer ``w``, gathered from local lanes
    ``send_idx[d, send_offsets[d,w] : +send_sizes[d,w]]``; writer ``w`` receives ``recv_sizes[w,d]`` from
    ``d`` (== ``send_sizes[d,w]``) and scatters recv slot ``s`` into its local canonical buffer at
    ``recv_place[w,s]`` (``= gid − first_gid[w]``; pad slots → the dump index ``BUF``). Each writer's
    ``[BUF]`` buffer covers a contiguous block of whole chunks ⇒ lock-free disjoint zarr writes."""
    send_idx: np.ndarray       # [P, send_max] int32 — local lanes to gather (ordered by dest writer)
    send_sizes: np.ndarray     # [P, P] int32
    send_offsets: np.ndarray   # [P, P] int32 — input_offsets (exclusive cumsum of send_sizes, axis 1)
    out_offsets: np.ndarray    # [P, P] int32 — output_offsets (= recv_offsets.T)
    recv_sizes: np.ndarray     # [P, P] int32
    recv_place: np.ndarray     # [P, recv_max] int32 — recv slot → local buffer idx (pad → BUF)
    send_max: int
    recv_max: int
    BUF: int                   # per-device canonical buffer rows (= chunks_per_writer * C)
    first_gid: np.ndarray      # [P] int64 — first global row of each device's block
    n_real: np.ndarray         # [P] int64 — real rows each device writes (0 if it owns no chunk)
    C: int
    n_global: int


def build_redist_maps(gid: np.ndarray, owned: np.ndarray, n_global: int, P: int,
                      Lmax: int, C: int) -> RedistMaps:
    """Build :class:`RedistMaps` from the folded gid/owned maps + a chunk size ``C`` (host numpy)."""
    nchunks = (n_global + C - 1) // C
    cpw = (nchunks + P - 1) // P                      # chunks per writer (block assignment)
    BUF = cpw * C
    first_gid = np.array([w * cpw * C for w in range(P)], dtype=np.int64)
    n_real = np.clip(n_global - first_gid, 0, BUF).astype(np.int64)

    # recv_lists[w][d] = [(local_buf_idx, owner_local_lane, gid)] entities owned by d → writer w
    recv_lists = [[[] for _ in range(P)] for _ in range(P)]
    gid = np.asarray(gid); owned = np.asarray(owned)
    for d in range(P):
        base = d * Lmax
        loc = np.nonzero(owned[base:base + Lmax])[0]          # owned local lanes on device d
        for l in loc:
            g = int(gid[base + int(l)])
            if g < 0:
                continue
            w = (g // C) // cpw
            recv_lists[w][d].append((g - int(first_gid[w]), int(l), g))
    for w in range(P):
        for d in range(P):
            recv_lists[w][d].sort(key=lambda t: t[2])         # gid asc ⇒ send/recv align

    send_sizes = np.zeros((P, P), np.int32)
    recv_sizes = np.zeros((P, P), np.int32)
    for w in range(P):
        for d in range(P):
            n = len(recv_lists[w][d])
            recv_sizes[w, d] = n
            send_sizes[d, w] = n
    send_offsets = np.zeros((P, P), np.int32)
    recv_offsets = np.zeros((P, P), np.int32)
    if P:
        send_offsets[:, 1:] = np.cumsum(send_sizes, axis=1)[:, :-1]
        recv_offsets[:, 1:] = np.cumsum(recv_sizes, axis=1)[:, :-1]
    send_max = int(send_sizes.sum(axis=1).max()) if P else 0
    recv_max = int(recv_sizes.sum(axis=1).max()) if P else 0
    send_idx = np.zeros((P, max(send_max, 1)), np.int32)
    recv_place = np.full((P, max(recv_max, 1)), BUF, np.int32)   # pad slots → dump index BUF
    for d in range(P):                                   # send buffer order: by dest writer w
        pos = 0
        for w in range(P):
            for (_b, l, _g) in recv_lists[w][d]:
                send_idx[d, pos] = l
                pos += 1
    for w in range(P):                                   # recv buffer order: by source device d
        pos = 0
        for d in range(P):
            for (b, _l, _g) in recv_lists[w][d]:
                recv_place[w, pos] = b
                pos += 1
    out_offsets = recv_offsets.T.copy()
    return RedistMaps(send_idx, send_sizes, send_offsets, out_offsets, recv_sizes, recv_place,
                      send_max, recv_max, BUF, first_gid, n_real, int(C), int(n_global))


# --------------------------------------------------------------------------
# The collective transport (inside shard_map) + driver
# --------------------------------------------------------------------------
def _redist_local(field, send_idx, send_off, send_sizes, out_off, recv_sizes,
                  recv_place, recv_max, BUF, axis_name):
    """One device's redistribute: gather its owned lanes into a send buffer, ``ragged_all_to_all`` to
    the chunk-owner devices, scatter received entities into a ``[BUF, *rest]`` canonical buffer."""
    rest = field.shape[1:]
    operand = field[send_idx]                                      # [send_max, *rest]
    output = jnp.zeros((recv_max,) + rest, dtype=field.dtype)
    recv = lax.ragged_all_to_all(operand, output, send_off, send_sizes,
                                 out_off, recv_sizes, axis_name=axis_name)
    buf = jnp.zeros((BUF,) + rest, dtype=field.dtype)
    # pad recv slots carry recv_place == BUF (out-of-bounds) ⇒ mode='drop' discards them; valid slots
    # are unique gids ⇒ no duplicate-index scatter (the slow XLA path). This is the canonical buffer.
    return buf.at[recv_place].set(recv, mode="drop")              # [BUF, *rest]


# Compiled shard_map callables are cached so repeated writes (each field, each output step) REUSE the
# executable instead of recompiling — the same lesson as the per-chunk-recompile step driver. Keyed by
# (recv_max, BUF, axis_name, device ids): a stable callable ⇒ XLA caches compilation per field shape.
_FN_CACHE: dict = {}


def _redist_callable(jmesh, recv_max, BUF, axis_name):
    key = (recv_max, BUF, axis_name, tuple(int(d.id) for d in np.asarray(jmesh.devices).flat))
    fn = _FN_CACHE.get(key)
    if fn is None:
        spec = PartitionSpec(axis_name)

        def f(fld, si, so, ss, oo, rs, rp):
            return _redist_local(fld, si, so, ss, oo, rs, rp, recv_max, BUF, axis_name)

        fn = jax.shard_map(f, mesh=jmesh, in_specs=(spec,) * 7, out_specs=spec)
        _FN_CACHE[key] = fn
    return fn


def _allgather_callable(jmesh, axis_name):
    """Cached ``all_gather`` shard_map (replicated out_spec) — reused across fields + output steps."""
    key = ("ag", axis_name, tuple(int(d.id) for d in np.asarray(jmesh.devices).flat))
    fn = _FN_CACHE.get(key)
    if fn is None:
        fn = jax.shard_map(lambda x: lax.all_gather(x, axis_name, axis=0, tiled=True),
                           mesh=jmesh, in_specs=PartitionSpec(axis_name),
                           out_specs=PartitionSpec(), check_vma=False)
        _FN_CACHE[key] = fn
    return fn


def redistribute_fields(fields, maps: RedistMaps, jmesh, axis_name: str = "p"):
    """Redistribute a dict of folded ``[P*Lmax, *rest]`` sharded fields → canonical ``[P*BUF, *rest]``
    sharded (device ``w`` owns its chunk block), no global field on any one device. The offset maps are
    folded to device ONCE (shared by all fields) and the ``shard_map`` callable is cached, so XLA
    compiles per field-SHAPE once and reuses it across fields + output steps."""
    fold0 = lambda a: halo._fold(np.asarray(a))[0]
    si = fold0(maps.send_idx).astype(jnp.int32)
    so = fold0(maps.send_offsets).astype(jnp.int32)
    ss = fold0(maps.send_sizes).astype(jnp.int32)
    oo = fold0(maps.out_offsets).astype(jnp.int32)
    rs = fold0(maps.recv_sizes).astype(jnp.int32)
    rp = fold0(maps.recv_place).astype(jnp.int32)
    fn = _redist_callable(jmesh, int(maps.recv_max), int(maps.BUF), axis_name)
    return {name: fn(arr, si, so, ss, oo, rs, rp) for name, arr in fields.items()}


# --------------------------------------------------------------------------
# The two multi-process canonical writers (node fields)
# --------------------------------------------------------------------------
def _create_store(out_dir, fields, glon, glat, n_global, C, chunk_vert, attrs, method, time_days=None):
    """Rank-0: create the store, write the (host-global) lon/lat coords, define each field dataset in
    canonical ``[..., n_global]`` shape with the fixed ``C`` chunk grid. Returns nothing.

    ``time_days`` (see :func:`ushow_output.write_ushow_zarr`): if given, EVERY field dataset also
    gains a leading ``time`` axis (size 1) -- required for ushow's per-point time-series extraction
    to find/use the ``time`` coordinate at all (it keys off the DATA variable's own dims, not just a
    bare store-level ``time`` array)."""
    import zarr
    from .ushow_output import DEPTH_DIM, NODE_DIM, TIME_DIM, _put_time
    has_time = time_days is not None
    root = zarr.open_group(str(out_dir), mode="w")
    zlon = root.create_dataset("lon", shape=(n_global,), chunks=(C,), dtype="f8", overwrite=True)
    zlat = root.create_dataset("lat", shape=(n_global,), chunks=(C,), dtype="f8", overwrite=True)
    zlon[:] = np.asarray(glon, np.float64)
    zlon.attrs["_ARRAY_DIMENSIONS"] = [NODE_DIM]; zlon.attrs["long_name"] = "longitude"
    zlat[:] = np.asarray(glat, np.float64)
    zlat.attrs["_ARRAY_DIMENSIONS"] = [NODE_DIM]; zlat.attrs["long_name"] = "latitude"
    for name, arr in fields.items():
        nlev = int(arr.shape[1]) if arr.ndim > 1 else 0
        shape = (nlev, n_global) if nlev else (n_global,)
        chunks = ((chunk_vert or nlev, C) if nlev else (C,))
        dims = [DEPTH_DIM, NODE_DIM] if nlev else [NODE_DIM]
        if has_time:
            shape = (1,) + shape
            chunks = (1,) + chunks
            dims = [TIME_DIM] + dims
        z = root.create_dataset(name, shape=shape, chunks=chunks, dtype="f4", overwrite=True)
        z.attrs["_ARRAY_DIMENSIONS"] = dims
        z.attrs["long_name"] = name
    _put_time(root, time_days)
    root.attrs.update(nod2D=int(n_global), layout="canonical_global", write_method=method,
                      **(attrs or {}))


def write_redistribute(out_dir, fields, sm, part, mesh, *, chunk_horiz=None, chunk_vert=None,
                       attrs=None, devices=None, time_days=None):
    """No-single-node-gather canonical writer via ``ragged_all_to_all`` (see module docstring)."""
    import zarr

    from .ushow_output import node_lonlat
    out_dir = str(out_dir)
    P, Lmax, n_global = int(sm.P), int(sm.Lmax["nod"]), int(sm.nod2D)
    C = int(chunk_horiz) if chunk_horiz else n_global
    maps = _cached_maps(sm, part, C)
    glon, glat = node_lonlat(mesh)
    devs = jax.devices()[:P] if devices is None else list(devices)
    jmesh = halo.device_mesh("p", devices=devs)

    if jax.process_index() == 0:
        _create_store(out_dir, fields, glon, glat, n_global, C, chunk_vert, attrs, "redistribute",
                      time_days=time_days)
    _barrier("canon_redist_meta")

    has_time = time_days is not None
    t0 = (0,) if has_time else ()                                 # leading time index, see _create_store
    canon = redistribute_fields(fields, maps, jmesh)             # {name: folded [P*BUF, *rest] sharded}
    root = zarr.open_group(out_dir, mode="a")
    for name, c in canon.items():
        z = root[name]
        for shard in c.addressable_shards:
            w = int(shard.index[0].start) // maps.BUF            # which writer device this shard is
            nr = int(maps.n_real[w])
            if nr <= 0:
                continue
            g0 = int(maps.first_gid[w])
            data = np.asarray(shard.data[:nr]).astype(np.float32)
            if data.ndim == 1:
                z[t0 + (slice(g0, g0 + nr),)] = data
            else:
                z[t0 + (slice(None), slice(g0, g0 + nr))] = data.T   # [nr, nlev] → [nlev, nr]
    _barrier("canon_redist_data")
    if jax.process_index() == 0:
        zarr.consolidate_metadata(out_dir)
    return out_dir


_MAPS_CACHE: dict = {}


def _cached_maps(sm, part, C: int) -> RedistMaps:
    """Cache the (host-built) redistribution maps per (sm, part, C) so repeated output steps don't
    re-run the O(P*Lmax) host map-build. sm/part persist for a whole run ⇒ id()-keying is stable."""
    from .ushow_output import _folded_gid_owned_nod
    key = (id(sm), id(part), int(C))
    maps = _MAPS_CACHE.get(key)
    if maps is None:
        gid, owned = _folded_gid_owned_nod(part, sm)
        maps = build_redist_maps(gid, owned, int(sm.nod2D), int(sm.P), int(sm.Lmax["nod"]), int(C))
        _MAPS_CACHE[key] = maps
    return maps


def write_all_gather(out_dir, fields, sm, part, mesh, *, chunk_horiz=None, chunk_vert=None,
                     attrs=None, devices=None, time_days=None):
    """Simpler baseline: ``all_gather`` the full field to every device, then each PROCESS writes its
    disjoint chunk range lock-free. Replicates the field per device (memory-heavy) — for the comparison."""
    import zarr

    from .ushow_output import _folded_gid_owned_nod, node_lonlat
    out_dir = str(out_dir)
    P, Lmax, n_global = int(sm.P), int(sm.Lmax["nod"]), int(sm.nod2D)
    C = int(chunk_horiz) if chunk_horiz else n_global
    gid, owned = _folded_gid_owned_nod(part, sm)
    g_idx = gid[owned]
    glon, glat = node_lonlat(mesh)
    devs = jax.devices()[:P] if devices is None else list(devices)
    jmesh = halo.device_mesh("p", devices=devs)

    # process→chunk-range assignment (disjoint, block) so writes never collide / need no lock
    nchunks = (n_global + C - 1) // C
    nproc = jax.process_count()
    pid = jax.process_index()
    cpp = (nchunks + nproc - 1) // nproc
    c0, c1 = pid * cpp, min((pid + 1) * cpp, nchunks)
    r0, r1 = c0 * C, min(c1 * C, n_global)                        # this process's global row range

    if pid == 0:
        _create_store(out_dir, fields, glon, glat, n_global, C, chunk_vert, attrs, "all_gather",
                      time_days=time_days)
    _barrier("canon_ag_meta")

    gather_fn = _allgather_callable(jmesh, "p")

    def _gather(field_folded):
        # all_gather → the full [P*Lmax, *rest] REPLICATED on every device (cached callable ⇒ compiled
        # once per shape); out_spec replicated (a sharded out_spec would re-tile the gathered array).
        return np.asarray(gather_fn(field_folded))               # [P*Lmax, *rest] host

    has_time = time_days is not None
    t0 = (0,) if has_time else ()                                 # leading time index, see _create_store
    root = zarr.open_group(out_dir, mode="a")
    for name, arr in fields.items():
        folded_host = _gather(arr)                                # every process has the full folded
        g = np.zeros((n_global,) + folded_host.shape[1:], dtype=folded_host.dtype)
        g[g_idx] = folded_host[owned]                             # canonical global (host, this process)
        if r1 > r0:
            data = g[r0:r1].astype(np.float32)
            z = root[name]
            if data.ndim == 1:
                z[t0 + (slice(r0, r1),)] = data
            else:
                z[t0 + (slice(None), slice(r0, r1))] = data.T
    _barrier("canon_ag_data")
    if pid == 0:
        zarr.consolidate_metadata(out_dir)
    return out_dir


def _barrier(tag):
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(tag)
