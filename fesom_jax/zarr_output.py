"""Sharded, gather-free model output to Zarr (Phase 8b B.3).

The FESOM2 → JAX model state lives **sharded** across the device mesh as folded
``[P*Lmax_kind, …]`` arrays (device ``d`` owns rows ``[d*Lmax : (d+1)*Lmax]``). Writing
output the C/Kokkos way — gather the global field to rank 0, then write one NetCDF — does
not scale: at NG5 (7.4 M × 70) that global gather is ~tens of GB on one rank (the OOM the
Kokkos ``SCALING_NG5`` notes, and the same single-device-materialization class as the input
placement fix). Instead, **each GPU writes its own shard in parallel** to a Zarr store:

* the on-disk array is the *folded* ``[P*Lmax_kind, …]`` shape, **chunked at ``Lmax_kind``
  along axis 0** so each device's shard is exactly one chunk ⇒ different processes/devices
  write disjoint chunk files ⇒ no locking, no gather, fully parallel;
* per-kind index maps ``gid`` (global entity id of each folded lane) and ``owned`` (the lane
  is its entity's unique owner) are written once, so :func:`reconstruct_global` can scatter
  the owned lanes back to a dense global ``[nod2D, …]`` / ``[elem2D, …]`` array on read.

This is the output analogue of :func:`fesom_jax.integrate_sharded._to_global_sharded` (input):
nothing global is ever materialized on a single device.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax

# Key prognostic fields a typical run wants on disk (override via ``fields=``).
DEFAULT_FIELDS = ("T", "S", "uv", "eta_n", "w",
                  "a_ice", "m_ice", "m_snow", "u_ice", "v_ice")


def _kind_of(lead: int, P: int, Lmax: dict) -> str:
    """Map a folded leading dim ``P*Lmax_kind`` back to its entity kind."""
    for k in ("nod", "elem", "edge"):
        if lead == P * Lmax[k]:
            return k
    raise ValueError(f"folded leading dim {lead} matches no kind (P={P}, Lmax={Lmax})")


def _folded_gid_owned(part, sm, kind: str):
    """Folded ``[P*Lmax]`` global-id + owned-mask for ``kind`` (the reconstruction maps).

    ``gid[d*Lmax + lane]`` = the global entity id of device ``d``'s local lane (``-1`` on
    pad lanes); ``owned`` is ``sm.owned_mask[kind]`` folded — True only on the lane that is
    the entity's UNIQUE owner, so scattering owned lanes by ``gid`` reproduces the dense
    global field with no double-write."""
    P, Lmax = sm.P, sm.Lmax[kind]
    mylist = {"nod": part.myList_nod2D, "elem": part.myList_elem2D,
              "edge": part.myList_edge2D}[kind]
    gid = np.full((P * Lmax,), -1, dtype=np.int64)
    for d in range(P):
        ml = np.asarray(mylist[d])
        gid[d * Lmax: d * Lmax + ml.size] = ml
    owned = np.asarray(sm.owned_mask[kind]).reshape(-1)        # [P*Lmax] bool, same fold
    return gid, owned


def write_state_zarr(out_dir, state_folded, sm, part, *, fields=None, attrs=None):
    """Write the *folded sharded* ``state_folded`` to a Zarr store at ``out_dir`` — each
    process writes ONLY its addressable shards, in parallel, no gather.

    ``state_folded`` is a :class:`~fesom_jax.state.State` of folded ``[P*Lmax_kind, …]``
    sharded ``jax.Array`` leaves (e.g. the output of
    ``run_steps_sharded(..., return_executable=True)`` then ``jfn(*args)``). Rank 0 creates
    the array metadata + the per-kind index maps; a multi-host barrier; then every process
    writes its local device shards (``arr.addressable_shards`` → one ``Lmax``-chunk each).
    Returns the store path."""
    import zarr
    from jax.experimental import multihost_utils

    out_dir = Path(out_dir)
    fields = tuple(fields) if fields is not None else DEFAULT_FIELDS
    leaves = {name: getattr(state_folded, name) for name in fields}
    P = sm.P
    is0 = jax.process_index() == 0

    # 1) rank 0 creates the dataset metadata + index maps (writing .zarray once, no race).
    if is0:
        root = zarr.open_group(str(out_dir), mode="a")
        kinds = set()
        for name, arr in leaves.items():
            kind = _kind_of(arr.shape[0], P, sm.Lmax)
            kinds.add(kind)
            shape = tuple(int(s) for s in arr.shape)
            chunks = (sm.Lmax[kind],) + shape[1:]
            z = root.require_dataset(name, shape=shape, chunks=chunks,
                                     dtype=np.dtype(arr.dtype), overwrite=True)
            z.attrs["kind"] = kind
        for kind in kinds:
            gid, owned = _folded_gid_owned(part, sm, kind)
            zg = root.require_dataset(f"gid_{kind}", shape=gid.shape, chunks=gid.shape,
                                      dtype=gid.dtype, overwrite=True); zg[:] = gid
            zo = root.require_dataset(f"owned_{kind}", shape=owned.shape, chunks=owned.shape,
                                      dtype=owned.dtype, overwrite=True); zo[:] = owned
        root.attrs.update(P=int(P), nod2D=int(sm.nod2D), elem2D=int(sm.elem2D),
                          edge2D=int(sm.edge2D), nl=int(sm.nl), **(attrs or {}))

    multihost_utils.sync_global_devices("fesom_zarr_meta")

    # 2) every process writes its addressable device shards (disjoint Lmax-chunks → parallel).
    root = zarr.open_group(str(out_dir), mode="a")
    for name, arr in leaves.items():
        z = root[name]
        for shard in arr.addressable_shards:
            z[shard.index] = np.asarray(shard.data)

    multihost_utils.sync_global_devices("fesom_zarr_data")
    return out_dir


def reconstruct_global(out_dir, field: str):
    """Read a folded Zarr field back to a dense global ``[n_global, …]`` array (host numpy):
    scatter the OWNED lanes by their global id. Inverse of :func:`write_state_zarr` for one
    field — for analysis / validation, not the hot path."""
    import zarr
    root = zarr.open_group(str(Path(out_dir)), mode="r")
    z = root[field]
    kind = z.attrs["kind"]
    data = z[:]                                            # [P*Lmax, …]
    gid = root[f"gid_{kind}"][:]
    owned = root[f"owned_{kind}"][:]
    n_global = {"nod": int(root.attrs["nod2D"]), "elem": int(root.attrs["elem2D"]),
                "edge": int(root.attrs["edge2D"])}[kind]
    out = np.zeros((n_global,) + data.shape[1:], dtype=data.dtype)
    out[gid[owned]] = data[owned]
    return out
