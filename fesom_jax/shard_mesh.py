"""Sharded-mesh build + export for the FESOM2 → JAX port (Phase 8, Task S.2).

Turns the dense single-device :class:`~fesom_jax.mesh.Mesh` + a
:class:`~fesom_jax.partit.Partition` (S.1) into a **per-device** :class:`ShardedMesh`:
every entity-leading mesh field is gathered to each device's local id list, the
connectivity is remapped global→local, all kinds are padded to a common per-kind
``Lmax`` (so JAX can shard a rectangular ``[P, Lmax, …]`` array), and the masks +
halo-exchange index maps the rest of Phase 8 consumes are built.

This is **host-side numpy** (like the C ``fesom_partit`` reader) — it produces a
container of numpy arrays. Device placement under ``shard_map`` is S.7.

S.2b adds the matching state/forcing partitioners (:func:`partition_state`,
:func:`partition_forcing_static`, :func:`partition_step_forcing`): a global IC /
forcing is built on the host (existing single-device builders) then gathered to
per-device padded pytrees that pad to the **same** ``Lmax`` as the mesh (via
:func:`local_sizes`). Building the IC globally also sidesteps the C's PHC
``extrap_nod3D`` per-sweep halo exchange.

Design choices (see ``docs/PORTING_LESSONS.md``)
------------------------------------------------
* **Local lane order is ``[interior (myDim) | halo (eDim[+eXDim])]``** — the FESOM
  ``myList_*`` order. So owned lanes are ``[0:myDim)``, halo lanes ``[myDim:n_local)``,
  pad lanes ``[n_local:Lmax)``.
* **Exchange map = ``(src_dev, src_lane)`` per kind**, built for the ``all_gather``
  primitive S.3 commits to (the simplest verifiable collective; ``ragged_all_to_all``
  is a perf follow-up). A halo lane reads its owner's *interior* value; an interior
  lane reads **itself** (identity) — exactly the C broadcast (owner→halo overwrite,
  interior untouched, ``fesom_halo.c:135-201``). For elements/edges (which are
  *redundantly* owned at the boundary — see S.1) the halo source is the lowest-id
  interior owner; all interior owners compute the same value (redundant compute) so
  the choice is immaterial to ~1e-15.
* **``nod_in_elem2D`` (the CSR) is OMITTED** — it is used only by the host PHC IC
  builder (``phc_ic.py``), never by a step kernel, and S.2b builds the IC on the host
  then partitions it. Keeping it out of the per-device bundle avoids a ragged-CSR
  pad for zero step-path benefit.
* **gather-on-sentinel safety**: connectivity carries ``-1`` for a boundary/unmappable
  index. ``ops.gather`` at ``-1`` returns the LAST lane (``ops.py:33``), so kernels
  must clamp+mask. Proven safe for owned entities: an owned element's 3 vertices are
  ALL local (verified — 0 sentinels in ``elem_nodes[:myDim]``), so no **owned** output
  depends on a sentinel gather; only halo/eXDim lanes carry ``-1`` and those are
  masked / refreshed by the exchange.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .mesh import Mesh
from .partit import Partition, synth_serial

# --------------------------------------------------------------------------
# Mesh-field categorisation (leading entity axis). Mirrors the Mesh dataclass.
# --------------------------------------------------------------------------
# Plain gather-by-id fields (values are data, not ids):
NODE_FIELDS = (
    "coord_nod2D", "geo_coord_nod2D", "coast_flag", "nlevels_nod2D",
    "nlevels_nod2D_min", "ulevels_nod2D", "ulevels_nod2D_max", "depth",
    "mesh_resolution", "coriolis_node", "area", "areasvol", "zbar_3d_n",
    "node_layer_mask", "node_iface_mask",
)
ELEM_FIELDS = (
    "nlevels", "ulevels", "elem_area", "elem_cos", "metric_factor", "coriolis",
    "elem_center_x", "elem_center_y", "gradient_sca",
    "elem_layer_mask", "elem_iface_mask",
)
EDGE_FIELDS = ("edge_dxdy", "edge_cross_dxdy")

# Connectivity: (field, row_kind, value_kind) — gather rows by row_kind's myList,
# remap the int VALUES from global to local via the value_kind's global→local map.
CONN_FIELDS = (
    ("elem_nodes",      "elem", "nod"),
    ("edges",           "edge", "nod"),
    ("edge_tri",        "edge", "elem"),
    ("edge_up_dn_tri",  "edge", "elem"),
)

# Replicated (no entity axis): kept as a single global copy on every device.
REPLICATED_FIELDS = ("zbar", "Z")

# Omitted from the per-device bundle (host-IC only): nod_in_elem2D[_offsets].

_KINDS = ("nod", "elem", "edge")


# --------------------------------------------------------------------------
# ShardedMesh container
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ShardedMesh:
    """Per-device padded mesh bundle (host numpy arrays).

    ``fields`` maps each Mesh field name to its sharded array: entity-leading
    fields are ``[P, Lmax_kind, …]`` (pad lanes carry a safe value), replicated
    fields (``zbar``/``Z``) are kept global ``[nl]``/``[nl-1]``. ``owned_mask`` /
    ``valid_mask`` are ``{kind: [P, Lmax]}`` bool (interior / interior+halo).
    ``exchange`` is ``{kind: (src_dev [P,Lmax], src_lane [P,Lmax])}`` for the
    broadcast halo exchange. ``counts`` carries the per-device ``myDim/eDim/eXDim``.
    """

    P: int
    Lmax: dict          # {kind: int}
    nl: int
    fields: dict        # {name: np.ndarray}
    owned_mask: dict    # {kind: [P, Lmax] bool}
    valid_mask: dict    # {kind: [P, Lmax] bool}
    exchange: dict      # {kind: (src_dev, src_lane)}
    counts: dict        # {'myDim_nod': [P], 'eDim_nod': [P], 'eXDim_elem': [P], ...}
    # static globals carried verbatim (replicated metadata)
    nod2D: int = 0
    elem2D: int = 0
    edge2D: int = 0
    edge2D_in: int = 0
    myDim_edge2D_global: int = 0
    ocean_area: float = 0.0
    exchange_ragged: dict = None  # {kind: RaggedExchange} — Phase 8b B.0 (None ⇒ not built)
    exchange_coloured: dict = None  # {kind: ColouredExchange} — Phase 8d (None ⇒ not built)


# --------------------------------------------------------------------------
# Build helpers
# --------------------------------------------------------------------------
def _default_pad(dtype: np.dtype):
    """Safe pad value: bool→False, float→1.0 (nonzero, so 1/x stays finite on a
    masked pad lane — the masked-NaN AD rule), int→0."""
    if dtype.kind == "b":
        return False
    if dtype.kind == "f":
        return 1.0
    return 0


def local_sizes(partition: Partition):
    """Per-device local entity counts ``n_local`` (interior+halo[+eXDim]) and the
    common per-kind ``Lmax`` (max over devices) — the padded sharding extent.
    Shared by :func:`build_sharded_mesh` and :func:`partition_state` so the mesh
    and state pad to the SAME ``Lmax`` per device."""
    n_local = {
        "nod": partition.myDim_nod2D + partition.eDim_nod2D,
        "elem": partition.myDim_elem2D + partition.eDim_elem2D + partition.eXDim_elem2D,
        "edge": partition.myDim_edge2D + partition.eDim_edge2D,
    }
    Lmax = {k: int(n_local[k].max()) for k in _KINDS}
    return n_local, Lmax


def _inv_map(mylist: np.ndarray, global_count: int) -> np.ndarray:
    """global id → local index (``-1`` if not on this device)."""
    g2l = np.full(global_count, -1, dtype=np.int32)
    g2l[mylist] = np.arange(mylist.size, dtype=np.int32)
    return g2l


def _gather_pad(global_arr, mylists, Lmax: int, pad) -> np.ndarray:
    """Gather ``global_arr`` rows by each device's id list and pad to
    ``[P, Lmax, *rest]`` with ``pad``."""
    g = np.asarray(global_arr)
    P = len(mylists)
    out = np.full((P, Lmax) + g.shape[1:], pad, dtype=g.dtype)
    for d, ml in enumerate(mylists):
        out[d, : ml.size] = g[ml]
    return out


def _gather_pad_conn(global_conn, rows_mylists, g2l_list, Lmax: int) -> np.ndarray:
    """Connectivity: gather rows by ``rows_mylists`` then remap the int values
    global→local via ``g2l_list`` (one per device). Existing ``-1`` (boundary)
    and unmappable (outer-halo) values both stay ``-1``; pad lanes are ``-1``."""
    g = np.asarray(global_conn)
    P = len(rows_mylists)
    out = np.full((P, Lmax, g.shape[1]), -1, dtype=np.int32)
    for d, ml in enumerate(rows_mylists):
        rows = g[ml]                                   # [n_local, k] global value-ids
        valid = rows >= 0
        remapped = g2l_list[d][np.where(valid, rows, 0)]   # clamp -1→0 before gather
        out[d, : ml.size] = np.where(valid, remapped, -1)  # g2l gives -1 if unmappable
    return out


def _owner_map(mylists, mydim, global_count: int):
    """Lowest-id interior owner of each global entity + its interior lane index.
    Nodes have a unique owner; elements/edges pick the min-id interior owner."""
    owner = np.full(global_count, -1, dtype=np.int32)
    owner_local = np.full(global_count, -1, dtype=np.int32)
    for d, ml in enumerate(mylists):
        md = int(mydim[d])
        interior = ml[:md]
        unset = owner[interior] < 0
        sel = interior[unset]
        owner[sel] = d
        owner_local[sel] = np.arange(md, dtype=np.int32)[unset]
    return owner, owner_local


def _exchange_map(mylists, mydim, Lmax: int, owner, owner_local):
    """``(src_dev, src_lane)`` ``[P, Lmax]`` for the broadcast exchange: interior &
    pad lanes are identity (read self), halo lanes read their owner's interior."""
    P = len(mylists)
    src_dev = np.zeros((P, Lmax), dtype=np.int32)
    src_lane = np.zeros((P, Lmax), dtype=np.int32)
    for d in range(P):
        md = int(mydim[d])
        n_local = mylists[d].size
        src_dev[d, :] = d                               # default: self (identity)
        src_lane[d, :md] = np.arange(md)                # interior identity
        halo = mylists[d][md:n_local]                   # halo global ids
        src_dev[d, md:n_local] = owner[halo]
        src_lane[d, md:n_local] = owner_local[halo]
        # pad lanes [n_local:Lmax] keep src_dev=d, src_lane=0 (masked).
    return src_dev, src_lane


@dataclasses.dataclass(frozen=True)
class RaggedExchange:
    """Per-device ragged halo-exchange maps for ``lax.ragged_all_to_all`` (Phase 8b
    Task B.0) — the halo-only, point-to-point replacement for the O(P·N_local)
    ``all_gather`` exchange (:func:`_exchange_map`). Built from the **same** owner
    map, so it reproduces the broadcast exchange byte-identically on valid lanes;
    only the transport changes (each device ships just its boundary lanes to the
    neighbours that need them, like the C/Kokkos MPI).

    All arrays are device-leading ``[P, …]``. For device ``d``:
      * ships ``send_sizes[d, e]`` interior lanes to device ``e``, gathered from its
        local lanes ``send_idx[d, send_offsets[d,e] : +send_sizes[d,e]]``;
      * receives ``recv_sizes[d, e]`` lanes from device ``e`` into its local halo
        lanes ``recv_idx[d, recv_offsets[d,e] : +recv_sizes[d,e]]``.
    ``send_sizes[d,e] == recv_sizes[e,d]`` by construction; the per-``(e,d)`` block
    is ordered by increasing halo-lane index on the receiver, so send and recv
    sides align element-wise. ``send_idx``/``recv_idx`` are padded to ``send_max`` /
    ``recv_max`` (max total over devices); offsets are per-device exclusive cumsums
    (the ``input_offsets``/``output_offsets`` ``ragged_all_to_all`` wants)."""

    send_idx: np.ndarray      # [P, send_max] int32 — local interior lanes to gather
    send_sizes: np.ndarray    # [P, P] int32 — send_sizes[d,e] = #lanes d ships to e
    send_offsets: np.ndarray  # [P, P] int32 — input_offsets: exclusive cumsum of send_sizes (axis 1)
    recv_idx: np.ndarray      # [P, recv_max] int32 — local halo lanes per recv-buffer slot
    recv_sizes: np.ndarray    # [P, P] int32 — recv_sizes[d,e] = #lanes d gets from e
    recv_offsets: np.ndarray  # [P, P] int32 — exclusive cumsum of recv_sizes (axis 1)
    # --- the extras the in-shard_map ragged_all_to_all primitive consumes ---
    out_offsets: np.ndarray   # [P, P] int32 == recv_offsets.T — ragged_all_to_all `output_offsets`
    #                           (out_offsets[d,e] = where d's slice lands on receiver e = recv_offsets[e,d])
    recv_gather: np.ndarray   # [P, Lmax] int32 — inverse of recv_idx: per lane, its recv-buffer slot (0 if
    #                           not a halo lane; masked out by halo_mask)
    halo_mask: np.ndarray     # [P, Lmax] bool — True on halo lanes (md ≤ lane < n_local); the exchange
    #                           overwrites these with the gathered owner value, leaves interior+pad as-is
    send_max: int
    recv_max: int
    # --- the slot-PADDED dense-all_to_all maps (Phase 8c, merged from
    # experiments/padded_halo_a2a): the SAME point-to-point exchange rendered as ONE
    # dense ``lax.all_to_all`` over P slots of ``pad_slot`` lanes each (pair-chunks
    # zero-padded to the max chunk). Runs on every backend (``ragged_all_to_all`` is
    # unimplemented on XLA:CPU) and its transpose is another ``all_to_all`` (trusted)
    # ⇒ gradients are correct by construction, unlike the ragged primitive
    # (docs/JAX_RAGGED_A2A_BUG.md). Overhead vs ragged = the padding zeros:
    # 1.8×@P4 … 12×@P32 lanes shipped, still 50–140× under all_gather (CORE2). ---
    pad_src: np.ndarray       # [P, P*pad_slot] int32 — send buffer: slot e = the d→e chunk's
    #                           local interior lanes (send_idx order), 0 on pad positions
    pad_valid: np.ndarray     # [P, P*pad_slot] bool — True on real chunk lanes (pad → zeroed;
    #                           load-bearing for the TRANSPOSE: kills cotangents of the
    #                           duplicated pad gathers)
    pad_slotpos: np.ndarray   # [P, Lmax] int32 — per halo lane, its position in the received
    #                           slotted buffer (= owner*pad_slot + within-chunk); 0 off-halo
    pad_slot: int             # static slot width = max (d,e) pair-chunk size (≥1)


def _ragged_exchange_map(mylists, mydim, Lmax: int, owner, owner_local) -> RaggedExchange:
    """Build the :class:`RaggedExchange` for one entity kind from the owner map
    (the same ``owner``/``owner_local`` :func:`_exchange_map` uses)."""
    P = len(mylists)
    # recv_pairs[e][d] = (halo_lane_on_e, owner_local_lane_on_d) for halo lanes on e
    # owned by d, in increasing halo-lane order — the canonical per-(e,d) ordering
    # shared by the send and recv sides so the transported chunks align.
    recv_pairs = [[[] for _ in range(P)] for _ in range(P)]
    for e in range(P):
        md = int(mydim[e])
        ml = mylists[e]
        for lane in range(md, ml.size):                 # halo lanes on e
            gid = int(ml[lane])
            recv_pairs[e][int(owner[gid])].append((lane, int(owner_local[gid])))

    send_sizes = np.zeros((P, P), dtype=np.int32)
    recv_sizes = np.zeros((P, P), dtype=np.int32)
    for e in range(P):
        for d in range(P):
            n = len(recv_pairs[e][d])
            recv_sizes[e, d] = n        # e receives n lanes from d
            send_sizes[d, e] = n        # d sends n lanes to e
    send_offsets = np.zeros((P, P), dtype=np.int32)
    recv_offsets = np.zeros((P, P), dtype=np.int32)
    if P:
        send_offsets[:, 1:] = np.cumsum(send_sizes, axis=1)[:, :-1]
        recv_offsets[:, 1:] = np.cumsum(recv_sizes, axis=1)[:, :-1]
    send_max = int(send_sizes.sum(axis=1).max()) if P else 0
    recv_max = int(recv_sizes.sum(axis=1).max()) if P else 0
    send_idx = np.zeros((P, max(send_max, 1)), dtype=np.int32)
    recv_idx = np.zeros((P, max(recv_max, 1)), dtype=np.int32)
    for d in range(P):                                  # send buffer, ordered by dest e
        pos = 0
        for e in range(P):
            for _lane, ol in recv_pairs[e][d]:
                send_idx[d, pos] = ol
                pos += 1
    recv_gather = np.zeros((P, Lmax), dtype=np.int32)   # per lane → its recv-buffer slot
    halo_mask = np.zeros((P, Lmax), dtype=bool)
    for e in range(P):                                  # recv buffer, ordered by source d
        pos = 0
        for d in range(P):
            for lane, _ol in recv_pairs[e][d]:
                recv_idx[e, pos] = lane
                recv_gather[e, lane] = pos              # inverse map for the gather-back
                pos += 1
        md = int(mydim[e])
        halo_mask[e, md:mylists[e].size] = True         # halo lanes [md, n_local)
    out_offsets = recv_offsets.T.copy()                 # ragged_all_to_all `output_offsets`

    # Padded dense-a2a maps (Phase 8c): lay each d→e chunk into slot e of d's send
    # buffer; on the receiver (after the tiled all_to_all, slot d = what d sent me)
    # each halo lane reads owner_slot*pad_slot + its within-chunk position. Both
    # sides inherit the canonical per-(e,d) chunk ordering above, so this is the
    # ragged exchange bit-for-bit — only the transport differs.
    pad_slot = max(int(recv_sizes.max()), 1) if P else 1
    pad_src = np.zeros((P, P * pad_slot), dtype=np.int32)
    pad_valid = np.zeros((P, P * pad_slot), dtype=bool)
    for d in range(P):
        for e in range(P):
            n, off = int(send_sizes[d, e]), int(send_offsets[d, e])
            pad_src[d, e * pad_slot: e * pad_slot + n] = send_idx[d, off: off + n]
            pad_valid[d, e * pad_slot: e * pad_slot + n] = True
    pad_slotpos = np.zeros((P, Lmax), dtype=np.int32)
    for e in range(P):
        for d in range(P):
            n, off = int(recv_sizes[e, d]), int(recv_offsets[e, d])
            pad_slotpos[e, recv_idx[e, off: off + n]] = (
                d * pad_slot + np.arange(n, dtype=np.int32))

    return RaggedExchange(send_idx, send_sizes, send_offsets,
                          recv_idx, recv_sizes, recv_offsets,
                          out_offsets, recv_gather, halo_mask, send_max, recv_max,
                          pad_src, pad_valid, pad_slotpos, pad_slot)


@dataclasses.dataclass(frozen=True)
class ColouredExchange:
    """The **coloured-``ppermute``** exchange maps (Phase 8d) — the transport that keeps
    the padded exchange's AD-correctness without its padding tax.

    ``all_to_all`` sends one message per rank *by definition*, so the padded transport must
    give every rank a slot even though a spatial partition has only ~6–7 real neighbours at
    any ``P``; at NG5-128 that is 98.4 % zeros on the wire (``scripts/bench/probe_pad_factor.py``).
    Here the neighbour graph is instead **bipartite-edge-coloured** (left = senders, right =
    receivers ⇒ König: Δ colours suffice, not Vizing's Δ+1). Each colour class is a *partial
    permutation* — every device sends to ≤1 and receives from ≤1 peer — i.e. exactly one legal
    :func:`jax.lax.ppermute`. So one exchange = ``K = Δ`` ppermute rounds, each with its **own**
    slot width (the max chunk *in that class*, not the global max — which kills the chunk-size
    variance tax as well as the zero-slot tax).

    Wire volume lands at 1.0–1.4× the true halo at every scale (vs padded's 1.8×…41×).
    ``ppermute`` is CPU-capable and its transpose is another ``ppermute`` ⇒ the gradient is
    correct by construction, exactly as for :func:`~fesom_jax.halo.halo_exchange_padded`.

    The K rounds are **packed into one buffer** (round r occupies ``[offs[r], offs[r]+slots[r])``)
    so the transport needs only one send-index array per kind rather than K shard_map inputs;
    ``offs``/``slots``/``perms`` are static (``ppermute`` needs a Python perm), so they travel as
    metadata, never as pytree leaves."""

    send_idx: np.ndarray      # [P, total] int32 — packed send buffer: round r at [offs[r], +slots[r])
    send_valid: np.ndarray    # [P, total] bool  — True on real chunk lanes (pad → zeroed; the
    #                           `where` is load-bearing for the transpose, as in the padded path)
    colpos: np.ndarray        # [P, Lmax] int32 — per halo lane, its position in the CONCATENATED
    #                           received buffer (= offs[r] + within-chunk); 0 off-halo (masked)
    perms: tuple              # K tuples of (src, dst) pairs — one lax.ppermute round each (STATIC)
    slots: tuple              # K ints — per-round slot width (max chunk in that colour class)
    offs: tuple               # K ints — exclusive cumsum of slots
    total: int                # = sum(slots) — the packed/concatenated buffer extent


def _colour_edges(send_sizes: np.ndarray, P: int) -> list:
    """Greedy bipartite edge-colouring of the neighbour graph (left = senders, right =
    receivers). Returns the colour classes as lists of ``(d, e, n)`` edges; each class is a
    partial permutation, i.e. one legal ``lax.ppermute`` round.

    Greedy (smallest colour free at both endpoints) rather than the König/augmenting-path
    construction: it may use a colour or two beyond Δ, which costs only an extra round, and
    the neighbour graphs here are tiny (≤ P·~7 edges). Big chunks first so the wide rounds
    pack together and the tail rounds stay narrow (per-round slots = max chunk in the class)."""
    edges = [(d, e, int(send_sizes[d, e]))
             for d in range(P) for e in range(P) if send_sizes[d, e] > 0]
    edges.sort(key=lambda t: -t[2])
    used_send = [set() for _ in range(P)]
    used_recv = [set() for _ in range(P)]
    classes: dict[int, list] = {}
    for d, e, n in edges:
        c = 0
        while c in used_send[d] or c in used_recv[e]:
            c += 1
        used_send[d].add(c)
        used_recv[e].add(c)
        classes.setdefault(c, []).append((d, e, n))
    return [classes[c] for c in sorted(classes)]


def _coloured_exchange_map(rex: RaggedExchange, P: int, Lmax: int) -> ColouredExchange:
    """Build the :class:`ColouredExchange` from the already-computed :class:`RaggedExchange`
    (pure index bookkeeping — no rebuild of the expensive owner/halo-lane walk).

    Both sides inherit the ragged map's canonical per-(e,d) chunk ordering, so position *i*
    of a chunk means the same lane on the sender and the receiver: this is the ragged exchange
    bit-for-bit, only the transport differs (the same contract the padded path satisfies)."""
    classes = _colour_edges(rex.send_sizes, P)
    slots = tuple(max(n for _, _, n in cls) for cls in classes)
    offs, acc = [], 0
    for s in slots:
        offs.append(acc)
        acc += s
    offs, total = tuple(offs), int(acc)
    perms = tuple(tuple((d, e) for d, e, _ in cls) for cls in classes)

    send_idx = np.zeros((P, max(total, 1)), dtype=np.int32)
    send_valid = np.zeros((P, max(total, 1)), dtype=bool)
    colpos = np.zeros((P, Lmax), dtype=np.int32)
    for r, cls in enumerate(classes):
        base = offs[r]
        for d, e, n in cls:
            so = int(rex.send_offsets[d, e])           # d's chunk for e (ordered by dest)
            send_idx[d, base: base + n] = rex.send_idx[d, so: so + n]
            send_valid[d, base: base + n] = True
            ro = int(rex.recv_offsets[e, d])           # e's chunk from d (ordered by source)
            colpos[e, rex.recv_idx[e, ro: ro + n]] = base + np.arange(n, dtype=np.int32)
    return ColouredExchange(send_idx, send_valid, colpos, perms, slots, offs, total)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_sharded_mesh(mesh: Mesh, partition: Partition) -> ShardedMesh:
    """Build the per-device :class:`ShardedMesh` from a dense :class:`Mesh` and a
    :class:`Partition`. ``partition.npes == 1`` (``synth_serial``) yields a no-op
    sharded mesh array-equal to the dense ``Mesh`` (the single-device invariant)."""
    P = partition.npes
    mylist = {
        "nod": partition.myList_nod2D,
        "elem": partition.myList_elem2D,
        "edge": partition.myList_edge2D,
    }
    mydim = {
        "nod": partition.myDim_nod2D,
        "elem": partition.myDim_elem2D,
        "edge": partition.myDim_edge2D,
    }
    n_local, Lmax = local_sizes(partition)
    gcount = {"nod": mesh.nod2D, "elem": mesh.elem2D, "edge": mesh.edge2D}

    # global→local maps (only node + elem ids appear as connectivity values)
    g2l = {
        "nod": [_inv_map(mylist["nod"][d], mesh.nod2D) for d in range(P)],
        "elem": [_inv_map(mylist["elem"][d], mesh.elem2D) for d in range(P)],
    }

    fields: dict[str, np.ndarray] = {}
    for name in NODE_FIELDS:
        arr = np.asarray(getattr(mesh, name))
        fields[name] = _gather_pad(arr, mylist["nod"], Lmax["nod"], _default_pad(arr.dtype))
    for name in ELEM_FIELDS:
        arr = np.asarray(getattr(mesh, name))
        fields[name] = _gather_pad(arr, mylist["elem"], Lmax["elem"], _default_pad(arr.dtype))
    for name in EDGE_FIELDS:
        arr = np.asarray(getattr(mesh, name))
        fields[name] = _gather_pad(arr, mylist["edge"], Lmax["edge"], _default_pad(arr.dtype))
    for name, row_kind, val_kind in CONN_FIELDS:
        fields[name] = _gather_pad_conn(
            getattr(mesh, name), mylist[row_kind], g2l[val_kind], Lmax[row_kind])
    for name in REPLICATED_FIELDS:
        fields[name] = np.asarray(getattr(mesh, name))         # global, replicated

    owned_mask, valid_mask, exchange, exchange_ragged = {}, {}, {}, {}
    exchange_coloured = {}
    lane = {k: np.arange(Lmax[k])[None, :] for k in _KINDS}      # [1, Lmax]
    for k in _KINDS:
        md = mydim[k][:, None]                                  # [P, 1]
        nlc = n_local[k][:, None]
        owned_mask[k] = lane[k] < md
        valid_mask[k] = lane[k] < nlc
        owner, owner_local = _owner_map(mylist[k], mydim[k], gcount[k])
        exchange[k] = _exchange_map(mylist[k], mydim[k], Lmax[k], owner, owner_local)
        exchange_ragged[k] = _ragged_exchange_map(
            mylist[k], mydim[k], Lmax[k], owner, owner_local)
        exchange_coloured[k] = _coloured_exchange_map(exchange_ragged[k], P, Lmax[k])

    counts = {
        "myDim_nod": partition.myDim_nod2D, "eDim_nod": partition.eDim_nod2D,
        "myDim_elem": partition.myDim_elem2D, "eDim_elem": partition.eDim_elem2D,
        "eXDim_elem": partition.eXDim_elem2D,
        "myDim_edge": partition.myDim_edge2D, "eDim_edge": partition.eDim_edge2D,
        "n_local_nod": n_local["nod"], "n_local_elem": n_local["elem"],
        "n_local_edge": n_local["edge"],
    }

    return ShardedMesh(
        P=P, Lmax=Lmax, nl=mesh.nl, fields=fields,
        owned_mask=owned_mask, valid_mask=valid_mask, exchange=exchange,
        exchange_ragged=exchange_ragged, exchange_coloured=exchange_coloured, counts=counts,
        nod2D=mesh.nod2D, elem2D=mesh.elem2D, edge2D=mesh.edge2D,
        edge2D_in=mesh.edge2D_in, myDim_edge2D_global=mesh.myDim_edge2D,
        ocean_area=mesh.ocean_area,
    )


def build_serial_sharded_mesh(mesh: Mesh) -> ShardedMesh:
    """Convenience: the ``npes==1`` sharded mesh (uses :func:`synth_serial`)."""
    return build_sharded_mesh(mesh, synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D))


# --------------------------------------------------------------------------
# S.2b — partition State / forcing / IC (host gather → per-device padded pytrees)
# --------------------------------------------------------------------------
def _shard_along_axis(arr, mylists, Lmax: int, axis: int, pad) -> np.ndarray:
    """Gather ``arr`` along ``axis`` by each device's id list, pad that axis to
    ``Lmax``, and stack devices on a new leading axis → ``[P, …]``."""
    outs = []
    for ml in mylists:
        local = np.take(arr, ml, axis=axis)
        pw = [(0, 0)] * local.ndim
        pw[axis] = (0, Lmax - local.shape[axis])
        outs.append(np.pad(local, pw, constant_values=pad))
    return np.stack(outs, axis=0)


def _node_axis(shape, nod2D: int) -> int:
    """The axis whose length is ``nod2D`` (the one to shard); the last such axis."""
    cand = [i for i, n in enumerate(shape) if n == nod2D]
    if not cand:
        raise ValueError(f"no nod2D(={nod2D}) axis in shape {shape}")
    return cand[-1]


def partition_state(state, partition: Partition):
    """Gather a global :class:`~fesom_jax.state.State` to per-device padded form.

    Each field is gathered along its leading entity axis (node- or elem-leading,
    detected by the leading dim) by the matching ``myList`` and padded to ``Lmax``;
    the result is a ``State`` with ``[P, Lmax, …]`` leaves (host numpy). Padded
    lanes carry a finite, masked-safe value (float→1.0). ``partition.npes==1`` is a
    no-op (squeeze the ``P=1`` axis ⇒ the dense ``State``).
    """
    from .state import State

    _, Lmax = local_sizes(partition)
    out = {}
    for f in dataclasses.fields(State):
        arr = np.asarray(getattr(state, f.name))
        n0 = arr.shape[0]
        if n0 == partition.nod2D:
            ml, L = partition.myList_nod2D, Lmax["nod"]
        elif n0 == partition.elem2D:
            ml, L = partition.myList_elem2D, Lmax["elem"]
        else:
            raise ValueError(f"State.{f.name} leading dim {n0} is neither nod2D "
                             f"({partition.nod2D}) nor elem2D ({partition.elem2D})")
        out[f.name] = _shard_along_axis(arr, ml, L, 0, _default_pad(arr.dtype))
    return State(**out)


def partition_forcing_static(fs, partition: Partition):
    """Gather a :class:`~fesom_jax.surface_forcing.ForcingStatic` to per-device padded
    form. Node fields → ``[P, Lmax_nod]``; the scalar ``ocean_area`` is replicated
    (kept as-is — it becomes a ``psum`` over owned nodes in S.5)."""
    from .surface_forcing import ForcingStatic

    _, Lmax = local_sizes(partition)
    out = {}
    for name in fs._fields:
        arr = np.asarray(getattr(fs, name))
        if arr.ndim == 0:
            out[name] = arr                                   # scalar — replicated
        elif arr.shape[0] == partition.nod2D:
            out[name] = _shard_along_axis(arr, partition.myList_nod2D, Lmax["nod"],
                                          0, _default_pad(arr.dtype))
        else:
            raise ValueError(f"ForcingStatic.{name} shape {arr.shape} not node-leading")
    return ForcingStatic(**out)


def partition_step_forcing(sf, partition: Partition):
    """Gather a :class:`~fesom_jax.surface_forcing.StepForcing` to per-device padded
    form, handling both a single step (``[nod2D]`` fields → ``[P, Lmax_nod]``) and a
    scanned stack (``[n_steps, nod2D]`` → ``[P, n_steps, Lmax_nod]``): the node axis
    is detected by size and sharded, the ``n_steps`` axis is preserved."""
    from .surface_forcing import StepForcing

    _, Lmax = local_sizes(partition)
    out = {}
    for name in sf._fields:
        arr = np.asarray(getattr(sf, name))
        ax = _node_axis(arr.shape, partition.nod2D)
        out[name] = _shard_along_axis(arr, partition.myList_nod2D, Lmax["nod"],
                                      ax, _default_pad(arr.dtype))
    return StepForcing(**out)


def partition_forcing_tables(ft, partition: Partition):
    """Gather a :class:`~fesom_jax.surface_forcing.ForcingTables` to per-device padded
    form: every leaf's (last) ``nod2D`` axis is sharded (``[…, nod2D]`` →
    ``[P, …, Lmax_nod]``), the small table axes are preserved."""
    from .surface_forcing import ForcingTables

    _, Lmax = local_sizes(partition)
    out = {}
    for name in ft._fields:
        arr = np.asarray(getattr(ft, name))
        ax = _node_axis(arr.shape, partition.nod2D)
        out[name] = _shard_along_axis(arr, partition.myList_nod2D, Lmax["nod"],
                                      ax, _default_pad(arr.dtype))
    return ForcingTables(**out)


def partition_forcing_const(fc, partition: Partition):
    """Gather a :class:`~fesom_jax.surface_forcing.ForcingDeviceConst` to per-device
    padded form: the ``[nod2D]`` trig leaves shard to ``[P, Lmax_nod]``; the tiny
    ``M`` rotation matrix is replicated as-is."""
    from .surface_forcing import ForcingDeviceConst

    _, Lmax = local_sizes(partition)
    out = {}
    for name in fc._fields:
        arr = np.asarray(getattr(fc, name))
        if name == "M" or arr.ndim == 0:
            out[name] = arr                                   # [9] M / () scalars — replicated
        else:
            out[name] = _shard_along_axis(arr, partition.myList_nod2D, Lmax["nod"],
                                          0, _default_pad(arr.dtype))
    return ForcingDeviceConst(**out)


# --------------------------------------------------------------------------
# Export / reload  (flat .npy directory, loadable like load_mesh)
# --------------------------------------------------------------------------
_META_INTS = ("P", "nl", "nod2D", "elem2D", "edge2D", "edge2D_in",
              "myDim_edge2D_global", "Lmax_nod", "Lmax_elem", "Lmax_edge")


def export_sharded_mesh(sm: ShardedMesh, out_dir: str | Path) -> Path:
    """Write ``sm`` as a flat ``.npy`` bundle + ``meta.txt`` under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in sm.fields.items():
        np.save(out_dir / f"f_{name}.npy", arr)
    for k in _KINDS:
        np.save(out_dir / f"mask_owned_{k}.npy", sm.owned_mask[k])
        np.save(out_dir / f"mask_valid_{k}.npy", sm.valid_mask[k])
        np.save(out_dir / f"exch_srcdev_{k}.npy", sm.exchange[k][0])
        np.save(out_dir / f"exch_srclane_{k}.npy", sm.exchange[k][1])
    for name, arr in sm.counts.items():
        np.save(out_dir / f"count_{name}.npy", arr)
    meta = dict(P=sm.P, nl=sm.nl, nod2D=sm.nod2D, elem2D=sm.elem2D, edge2D=sm.edge2D,
                edge2D_in=sm.edge2D_in, myDim_edge2D_global=sm.myDim_edge2D_global,
                Lmax_nod=sm.Lmax["nod"], Lmax_elem=sm.Lmax["elem"], Lmax_edge=sm.Lmax["edge"])
    lines = [f"{k} {meta[k]}" for k in _META_INTS]
    lines.append(f"ocean_area {sm.ocean_area!r}")
    (out_dir / "meta.txt").write_text("\n".join(lines) + "\n")
    return out_dir


def load_sharded_mesh(in_dir: str | Path) -> ShardedMesh:
    """Reload an :func:`export_sharded_mesh` bundle into a :class:`ShardedMesh`."""
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"sharded mesh dir not found: {in_dir}")
    meta: dict[str, float] = {}
    for line in (in_dir / "meta.txt").read_text().splitlines():
        if line.strip():
            key, val = line.split()
            meta[key] = float(val)

    fields = {p.name[2:-4]: np.load(p) for p in in_dir.glob("f_*.npy")}
    owned_mask = {k: np.load(in_dir / f"mask_owned_{k}.npy") for k in _KINDS}
    valid_mask = {k: np.load(in_dir / f"mask_valid_{k}.npy") for k in _KINDS}
    exchange = {k: (np.load(in_dir / f"exch_srcdev_{k}.npy"),
                    np.load(in_dir / f"exch_srclane_{k}.npy")) for k in _KINDS}
    counts = {p.name[6:-4]: np.load(p) for p in in_dir.glob("count_*.npy")}
    Lmax = {k: int(meta[f"Lmax_{k}"]) for k in _KINDS}

    return ShardedMesh(
        P=int(meta["P"]), Lmax=Lmax, nl=int(meta["nl"]), fields=fields,
        owned_mask=owned_mask, valid_mask=valid_mask, exchange=exchange, counts=counts,
        nod2D=int(meta["nod2D"]), elem2D=int(meta["elem2D"]), edge2D=int(meta["edge2D"]),
        edge2D_in=int(meta["edge2D_in"]),
        myDim_edge2D_global=int(meta["myDim_edge2D_global"]),
        ocean_area=float(meta["ocean_area"]),
    )
