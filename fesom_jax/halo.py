"""Broadcast halo-exchange primitive for the FESOM2 → JAX port (Phase 8, Task S.3).

The JAX rendering of the C ``fesom_halo_exchange`` (``port2/.../fesom_halo.c:135-201``):
a **broadcast** exchange that overwrites each rank's halo copies with their owner's
value (owner→halo, no additive accumulate), leaving interior lanes untouched.

Under ``jax.shard_map`` over a 1-D device mesh ``('p',)``, each device holds its
rank's padded ``[Lmax, …]`` local array (the S.2 sharding). The exchange is:

    gathered = all_gather(field)          # [P, Lmax, …] — every device's shard
    out      = gathered[src_dev, src_lane]  # per lane, read its owner's value

where ``(src_dev, src_lane)`` is the S.2 :class:`~fesom_jax.shard_mesh.ShardedMesh`
exchange map for the kind: an **interior** lane reads itself (identity — interior is
never overwritten, matching the C), a **halo** lane reads its owner's interior lane,
a **pad** lane reads lane 0 (a valid owned lane; masked downstream). ``src_lane`` is
always ``≥0`` so the gather never hits a sentinel.

``all_gather`` is the simplest verifiable collective and is correct for the 2–4
device gate; ``ragged_all_to_all`` (per-neighbour ragged sizes, using the
``Partition`` ``ComStruct`` slist/rlist) is a Post-Completion perf item.

**Sharding convention** — the device axis is folded into the leading dim: a global
array ``[P*Lmax, …]`` sharded ``PartitionSpec('p')`` gives each device ``[Lmax, …]``
(no squeeze), so the step body (S.7) operates on ``[Lmax, …]`` unchanged.

**AD**: ``all_gather`` + gather are linear in ``field``; the vjp is the reverse
exchange (``all_gather`` transpose = reduce-scatter ``psum``, gather transpose =
scatter-add), so a halo cotangent flows additively back to its owner — handled by
JAX automatically and grad-checked in the tests.
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, PartitionSpec

DEFAULT_AXIS = "p"


@dataclasses.dataclass(frozen=True)
class HaloCtx:
    """The per-device sharding context threaded into :func:`fesom_jax.step.step` (S.7),
    **constructed inside** ``shard_map``. ``exch`` maps a kind (``'nod'``/``'elem'``/
    ``'edge'``) to its ``(src_dev, src_lane)`` ``[Lmax]`` exchange map;
    :meth:`exchange` is the broadcast halo refresh the step inserts at the C's
    exchange points. ``ssh_halo`` is the S.6 :class:`~fesom_jax.ssh.SSHHalo` for the CG;
    ``owned_mask`` (``{kind: [Lmax]}``) feeds the distributed reductions. ``halo_ctx=None``
    in ``step`` is the dense single-device path (every exchange is an identity no-op ⇒
    byte-identical ``v1.0``)."""

    exch: dict           # {kind: (src_dev [Lmax], src_lane [Lmax])}
    axis_name: str
    ssh_halo: object     # ssh.SSHHalo (kept opaque to avoid a circular import)
    owned_mask: dict     # {kind: [Lmax] bool}
    # --- Phase 8b B.0: the halo-only ragged_all_to_all path (None ⇒ all_gather) ---
    exch_ragged: dict = None    # {kind: {send_idx, send_sizes, send_off, out_off,
    #                                     recv_sizes, recv_gather, halo_mask}} per-device arrays;
    #                             under use_padded the SAME slot carries the padded maps
    #                             ({pad_src, pad_valid, pad_slotpos, halo_mask}), and under
    #                             use_coloured the coloured maps ({send_idx, send_valid,
    #                             colpos, halo_mask}) — one slot, three shapes
    recv_max: dict = None       # {kind: int} static recv-buffer leading dim (ragged only)
    use_ragged: bool = False    # pick ragged_all_to_all over all_gather (GPU, forward-only)
    use_padded: bool = False    # pick the slot-padded dense all_to_all (Phase 8c) —
    #                             every backend, AD-correct; mutually exclusive with use_ragged
    use_coloured: bool = False  # pick the coloured-ppermute rounds (Phase 8d) — every backend,
    #                             AD-correct, and without padded's P-growing pad factor
    col_meta: dict = None       # {kind: (perms, slots, offs)} — STATIC (lax.ppermute needs a
    #                             Python perm); never a pytree leaf
    ssh_gather: object = None   # ssh.SSHGather ⇒ the step solves the SSH system gathered onto
    #                             every device (opt-in ``ssh.gathered``); None ⇒ the sharded CG
    exch_compact: dict = None   # {kind: (send [S], read [Lmax])} ⇒ the all_gather transport
    #                             gathers only the lanes other devices read (opt-in
    #                             FESOM_HALO_COMPACT=1, :func:`compact_exchange_maps`)
    mevp_wide: object = None    # ice_wide.MevpWide ⇒ the sharded mEVP runs K subcycles per
    #                             exchange on its K-ring ice lanes (opt-in gpu.mevp_wide /
    #                             FESOM_MEVP_WIDE); None ⇒ the per-subcycle exchange
    se_wide: object = None      # se_wide.SeWide ⇒ the split-explicit substeps run K per exchange on
    #                             K+1-ring lanes (opt-in gpu.se_wide / FESOM_SE_WIDE); None ⇒ two
    #                             exchanges per substep

    def exchange(self, field, kind: str):
        """Broadcast-refresh ``field``'s halo lanes from their owners (a no-op on
        interior + pad lanes), for the given entity ``kind``. Transport: the coloured
        ``ppermute`` rounds when ``use_coloured`` (Phase 8d), the padded dense
        ``all_to_all`` when ``use_padded`` (Phase 8c), the halo-only ``ragged_all_to_all``
        when ``use_ragged`` (Phase 8b), else ``all_gather``. ``FESOM_HALO_BARRIER``
        (:func:`halo_barrier_mode`) fences the exchange off from XLA fusion."""
        mode = halo_barrier_mode()
        if mode == "both":
            field = lax.optimization_barrier(field)
        out = self._transport(field, kind)
        return out if mode == "off" else lax.optimization_barrier(out)

    def _transport(self, field, kind: str):
        if self.use_coloured and self.exch_ragged is not None:
            return halo_exchange_coloured(
                field, self.exch_ragged[kind], self.col_meta[kind], self.axis_name)
        if self.use_padded and self.exch_ragged is not None:
            return halo_exchange_padded(field, self.exch_ragged[kind], self.axis_name)
        if self.use_ragged and self.exch_ragged is not None:
            return halo_exchange_ragged(
                field, self.exch_ragged[kind], self.recv_max[kind], self.axis_name)
        if self.exch_compact is not None:
            maps = self.exch_compact[kind]
            if len(maps) == 4:
                return halo_exchange_compact_scatter(field, maps[0], maps[2], maps[3], self.axis_name)
            send, read = maps
            return halo_exchange_compact(field, send, read, self.axis_name)
        src_dev, src_lane = self.exch[kind]
        return halo_exchange(field, src_dev, src_lane, self.axis_name)


def exchange_arrays(exch):
    """``(arrays, rebuild)`` for a halo exchanger: the exchange maps it reads and a function that
    rebuilds the same exchanger from them.

    For a custom derivative rule. Under ``shard_map`` the maps are tracers of the enclosing
    trace, and a rule is traced lazily (``jax.checkpoint`` re-traces it in the reverse sweep), so
    an exchanger it closes over carries stale tracers into the program (``TypeError: No constant
    handler for type: DynamicJaxprTracer`` at lowering). Passing ``arrays`` as an explicit
    argument of the rule and calling ``rebuild(arrays)`` inside avoids that.

    ``exch`` is :meth:`HaloCtx.exchange` (the sharded step) or any other callable, e.g. the
    dense step's identity: then ``arrays`` is ``()`` and ``rebuild`` returns ``exch`` itself."""
    ctx = getattr(exch, "__self__", None)
    if not isinstance(ctx, HaloCtx):
        return (), lambda _arrays: exch
    arrays = (ctx.exch, ctx.exch_ragged)

    def rebuild(a):
        return dataclasses.replace(ctx, exch=a[0], exch_ragged=a[1]).exchange
    return arrays, rebuild


def halo_barrier_mode() -> str:
    """``FESOM_HALO_BARRIER`` (read at trace time): ``out`` puts a ``lax.optimization_barrier``
    on every exchanged field, ``both`` on the field going in as well; unset or ``0`` ⇒ ``off``.
    The barrier is the identity on values and gradients. It exists because the padded
    transport's 6x cheaper NCCL (CORE2 4 A100, job 28283046) bought nothing: single
    consumer kernels such as ``impl_vert_visc`` and ``diff_part_hor_redi`` grew by ~2 ms
    each, the pattern of XLA fusing the exchange's gather into its consumers."""
    m = os.environ.get("FESOM_HALO_BARRIER", "0").strip().lower()
    if m in ("", "0", "off"):
        return "off"
    if m not in ("out", "both"):
        raise ValueError(f"FESOM_HALO_BARRIER must be 0|out|both, got {m!r}")
    return m


def exchange_pair(exch, a, b, kind: str):
    """Refresh two same-shape fields' halos in ONE exchange: stack them on a new
    trailing axis (every transport indexes only the leading lane axis — trailing
    axes ride along), exchange once, unstack. The exchanged VALUES are identical
    to two single-field calls; the collective count halves — which is the point
    for per-iteration callers (the EVP/mEVP subcycle fires 120 × 2 node exchanges
    per ice step, and under the coloured transport each exchange is K = Δ
    ``ppermute`` rounds). ``exch=None`` (the dense single-device path) returns the
    pair untouched, keeping that path's graph byte-identical, not merely
    value-identical."""
    if exch is None:
        return a, b
    ab = exch(jnp.stack([a, b], axis=-1), kind)
    return ab[..., 0], ab[..., 1]


def halo_exchange(field, src_dev, src_lane, axis_name: str = DEFAULT_AXIS):
    """One broadcast halo exchange, **called inside** ``shard_map``.

    ``field`` is this device's ``[Lmax, *rest]`` local array; ``src_dev`` /
    ``src_lane`` are its ``[Lmax]`` exchange map. Returns the refreshed
    ``[Lmax, *rest]`` (halo lanes overwritten by their owner's value, interior +
    pad lanes unchanged). Handles ``[Lmax]``, ``[Lmax, nl]``, ``[Lmax, nl, 2]`` —
    the gather indexes the leading two axes, trailing axes ride along.
    """
    gathered = lax.all_gather(field, axis_name, axis=0, tiled=False)  # [P, Lmax, *rest]
    return gathered[src_dev, src_lane]


def halo_compact_enabled() -> bool:
    """``FESOM_HALO_COMPACT=1`` (read on the host when the sharded driver folds its maps): the
    all_gather transport ships only the lanes other devices read, :func:`halo_exchange_compact`.
    ``=2``: the same, with the receive as a scatter into the halo and pad lanes only
    (:func:`halo_exchange_compact_scatter`)."""
    return os.environ.get("FESOM_HALO_COMPACT", "0").strip() not in ("", "0")


def halo_compact_scatter() -> bool:
    return os.environ.get("FESOM_HALO_COMPACT", "0").strip() == "2"


def compact_update_maps(read):
    """``(upd [P, U], src [P, U])`` for :func:`halo_exchange_compact_scatter` from a ``read`` map of
    :func:`compact_exchange_maps` (``[P, Lmax]``, halo lanes read ``Lmax + slot``): the lanes that do not
    keep their own value — halo lanes (source: their gathered slot) and pad lanes (source: lane 0, the
    appended row ``P*S``) — padded to the longest list by repeating the first entry (the same write)."""
    read = np.asarray(read)
    P, L = read.shape
    lanes = np.arange(L)
    upd, src = [], []
    for d in range(P):
        moved = np.nonzero(read[d] != lanes)[0]
        s = np.where(read[d, moved] >= L, read[d, moved] - L, -1)
        upd.append(moved)
        src.append(s)
    U = max(1, max(u.size for u in upd))
    out_u = np.zeros((P, U), dtype=np.int32)
    out_s = np.zeros((P, U), dtype=np.int32)
    for d in range(P):
        n = upd[d].size
        if n:
            out_u[d, :n], out_s[d, :n] = upd[d], src[d]
            out_u[d, n:], out_s[d, n:] = upd[d][0], src[d][0]
    return out_u, out_s       # src -1 = "lane 0": resolved against the gathered length at trace time


def halo_exchange_compact_scatter(field, send, upd, src, axis_name: str = DEFAULT_AXIS):
    """:func:`halo_exchange_compact` with the receive written only into the lanes that change (halo lanes
    from their gathered slot, pad lanes from lane 0): ``field.at[upd].set(...)``, which XLA can do in place,
    instead of a full elementwise pass over the field. The same value on every lane."""
    sent = lax.all_gather(field[send], axis_name, axis=0, tiled=True)    # [P*S, *rest]
    ext = jnp.concatenate([sent, field[0:1]], axis=0)
    return field.at[upd].set(ext[jnp.where(src < 0, sent.shape[0], src)])


def compact_exchange_maps(src_dev, src_lane, field_len=None):
    """``(send [P, S], read [P, Lmax])`` for :func:`halo_exchange_compact`, from the all_gather
    exchange map ``(src_dev, src_lane)`` ``[P, Lmax]``. ``send[d]`` lists, sorted and padded with
    lane 0, the lanes of device ``d`` that some other device reads (``S`` = the largest such set).
    ``read`` indexes ``concatenate([field, all_gather(field[send]).reshape(P*S, …)])``: a lane
    whose source is its own device (interior identity, pad lane 0) reads ``src_lane`` from the
    field itself; a halo lane reads slot ``Lmax + src_dev*S + k`` with ``send[src_dev, k] ==
    src_lane``. So every lane gets exactly the value the all_gather exchange gives it.

    ``field_len`` (default ``Lmax``) is the lane count of the field that is SENT and read from
    when it differs from the map's (the mEVP wide halo, :mod:`fesom_jax.ice_wide`, moves a
    standard-layout field onto its longer ice lanes)."""
    src_dev = np.asarray(src_dev)
    src_lane = np.asarray(src_lane)
    P, Lmax = src_dev.shape
    if field_len is not None:
        Lmax = int(field_len)
    remote = src_dev != np.arange(P)[:, None]
    uniq = [np.unique(src_lane[remote & (src_dev == d)]) for d in range(P)]
    S = max(1, max(u.size for u in uniq))
    send = np.zeros((P, S), dtype=np.int32)
    for d, u in enumerate(uniq):
        send[d, :u.size] = u
    read = src_lane.astype(np.int32).copy()
    for e in range(P):
        for d in range(P):
            m = remote[e] & (src_dev[e] == d)
            if m.any():
                read[e, m] = Lmax + d * S + np.searchsorted(uniq[d], src_lane[e, m])
    if field_len is None:
        # halo_exchange_compact gives an own-device lane that does not read itself lane 0's value
        other = ~remote & (src_lane != np.arange(Lmax)[None, :])
        if (src_lane[other] != 0).any():
            raise ValueError("compact_exchange_maps: an own-device lane reads neither itself nor lane 0")
    return send, read


def halo_exchange_compact(field, send, read, axis_name: str = DEFAULT_AXIS):
    """One broadcast halo exchange through ``all_gather``, but of only the lanes other devices
    read (``field[send]``, ``[S, …]`` with ``S`` ≪ ``Lmax``) instead of the whole local array;
    the refreshed field is then one elementwise pass: a halo lane (``read >= Lmax``) takes its slot of
    the gathered buffer, a pad lane (an own-device lane that does not read itself) lane 0's value, as
    :func:`halo_exchange` gives it, every other lane keeps its own. Values equal :func:`halo_exchange`'s
    on every lane; every step is linear with a registered transpose. (It used to be one gather out of
    ``concatenate([field, gathered])``: a full copy of the field per exchange, ~4 ms a CORE2 step on
    4 GH200, job 28291187.)"""
    sent = lax.all_gather(field[send], axis_name, axis=0, tiled=True)    # [P*S, *rest]
    L = field.shape[0]
    halo = read >= L
    pad = (read != jnp.arange(L, dtype=read.dtype)) & ~halo
    shape = (L,) + (1,) * (field.ndim - 1)
    hv = sent[jnp.where(halo, read - L, 0)]
    return jnp.where(halo.reshape(shape), hv, jnp.where(pad.reshape(shape), field[0:1], field))


def halo_exchange_ragged(field, r: dict, recv_max: int, axis_name: str = DEFAULT_AXIS):
    """One broadcast halo exchange via **halo-only point-to-point**
    ``lax.ragged_all_to_all`` (Phase 8b B.0) — the scaling replacement for
    :func:`halo_exchange`'s ``all_gather``. Called inside ``shard_map``; ``field`` is
    this device's ``[Lmax, *rest]`` local array, ``r`` its per-device ragged maps
    (:class:`~fesom_jax.shard_mesh.RaggedExchange`, folded to this shard), ``recv_max``
    the static recv-buffer extent. Returns the refreshed ``[Lmax, *rest]`` (halo lanes
    overwritten by their owner's value; interior + pad lanes untouched).

    Each device ships only its boundary lanes to the neighbours that need them
    (``operand = field[send_idx]``), exchanges them point-to-point, and reads the
    received values back per lane (``recv[recv_gather]``), overwriting halo lanes via
    ``halo_mask``. All three steps (gather / ``ragged_all_to_all`` / gather-back +
    masked select) are linear in ``field`` with registered transposes ⇒ the reverse
    pass scatter-adds each halo cotangent back to its owner (the AD gate's oracle)."""
    rest = field.shape[1:]
    operand = field[r["send_idx"]]                                   # [send_max, *rest]
    output = jnp.zeros((recv_max,) + rest, dtype=field.dtype)        # [recv_max, *rest]
    recv = lax.ragged_all_to_all(
        operand, output,
        r["send_off"], r["send_sizes"],        # input_offsets, send_sizes
        r["out_off"], r["recv_sizes"],         # output_offsets (= recv_offsets.T row), recv_sizes
        axis_name=axis_name)                                         # [recv_max, *rest]
    gathered = recv[r["recv_gather"]]                                # [Lmax, *rest]
    mask = r["halo_mask"].reshape(r["halo_mask"].shape + (1,) * len(rest))
    return jnp.where(mask, gathered, field)


def halo_exchange_padded(field, r: dict, axis_name: str = DEFAULT_AXIS):
    """One broadcast halo exchange via the **slot-padded dense** ``lax.all_to_all``
    (Phase 8c, merged from ``experiments/padded_halo_a2a``) — the CPU-capable,
    AD-correct substitute for :func:`halo_exchange_ragged`: ``ragged_all_to_all`` is
    unimplemented on XLA:CPU and its reverse-mode transpose is broken everywhere
    (``docs/JAX_RAGGED_A2A_BUG.md``), while ``all_to_all`` exists on every backend and
    transposes to another ``all_to_all`` (one of JAX's oldest rules) ⇒ the gradient is
    correct by construction (gated bit-exact vs the all_gather oracle, CORE2
    dist_2..32, forward AND ``jax.grad``).

    Called inside ``shard_map``; ``field`` is this device's ``[Lmax, *rest]`` local
    array, ``r`` its per-device padded maps (:class:`~fesom_jax.shard_mesh.RaggedExchange`
    ``pad_*`` fields + ``halo_mask``, folded to this shard). The send buffer is P slots
    of ``pad_slot`` lanes (slot e = my chunk for device e, zero-padded — the ``where``
    on ``pad_valid`` is load-bearing for the transpose: it kills the cotangents of the
    duplicated pad gathers); after ONE tiled ``all_to_all``, received slot d holds what
    device d sent me and ``pad_slotpos`` reads each halo lane straight out of it.
    Returns the refreshed ``[Lmax, *rest]`` (halo lanes overwritten by their owner's
    value; interior + pad lanes untouched — the same contract as the other two)."""
    rest = field.shape[1:]
    buf = field[r["pad_src"]]                                        # [P*slot, *rest]
    vmask = r["pad_valid"].reshape(r["pad_valid"].shape + (1,) * len(rest))
    buf = jnp.where(vmask, buf, jnp.zeros_like(buf))
    recv = lax.all_to_all(buf, axis_name, split_axis=0, concat_axis=0, tiled=True)
    gathered = recv[r["pad_slotpos"]]                                # [Lmax, *rest]
    mask = r["halo_mask"].reshape(r["halo_mask"].shape + (1,) * len(rest))
    return jnp.where(mask, gathered, field)


def halo_exchange_coloured(field, c: dict, meta, axis_name: str = DEFAULT_AXIS):
    """One broadcast halo exchange via **coloured ``lax.ppermute`` rounds** (Phase 8d) — the
    AD-correct transport *without* the padded exchange's P-growing padding tax.

    :func:`halo_exchange_padded` must give every rank a slot (``all_to_all`` sends one message
    per rank by definition), so at large ``P`` it ships ~98 % zeros. Here the neighbour graph is
    bipartite-edge-coloured (:func:`~fesom_jax.shard_mesh._coloured_exchange_map`): each colour
    class is a partial permutation — every device sends to ≤1 and receives from ≤1 peer — i.e.
    one legal ``ppermute``. An exchange is ``K = Δ`` such rounds (Δ = max #neighbours ≈ 3–14),
    each carrying only its own class's max chunk. Wire volume: 1.0–1.4× the true halo at every
    scale, against padded's 1.8×…41× (``scripts/bench/probe_pad_factor.py``).

    Called inside ``shard_map``; ``field`` is this device's ``[Lmax, *rest]`` local array, ``c``
    its per-device coloured maps (``send_idx``/``send_valid``/``colpos``/``halo_mask``), ``meta``
    the STATIC ``(perms, slots, offs)``. The K rounds are packed into one buffer — round *r* lives
    at ``[offs[r], offs[r]+slots[r])`` — so the send side is a single gather; the received rounds
    are re-concatenated and ``colpos`` reads each halo lane straight out of the result. Returns the
    refreshed ``[Lmax, *rest]`` (halo lanes overwritten by their owner's value; interior + pad lanes
    untouched — the same contract as the other three).

    **Why the gradient is right:** every step is linear with a registered transpose — gather,
    ``where`` (the ``send_valid`` mask is load-bearing: it kills the cotangents of the pad lanes'
    duplicated gathers), static slice, ``ppermute`` (transposes to the inverse ``ppermute``),
    concatenate, gather-back, ``where``. Verified against the ``all_gather`` oracle, forward AND
    ``jax.grad``, in ``fesom_jax/tests/test_halo.py::test_coloured_*``."""
    perms, slots, offs = meta
    if not perms:                       # no halo at all (npes==1) — the exchange is the identity
        return field
    rest = field.shape[1:]
    buf = field[c["send_idx"]]                                       # [total, *rest]
    vmask = c["send_valid"].reshape(c["send_valid"].shape + (1,) * len(rest))
    buf = jnp.where(vmask, buf, jnp.zeros_like(buf))
    rounds = [lax.ppermute(buf[o: o + s], axis_name, perm=list(p))   # static o/s ⇒ static slice
              for p, s, o in zip(perms, slots, offs)]
    recv = jnp.concatenate(rounds, axis=0) if len(rounds) > 1 else rounds[0]   # [total, *rest]
    gathered = recv[c["colpos"]]                                     # [Lmax, *rest]
    mask = c["halo_mask"].reshape(c["halo_mask"].shape + (1,) * len(rest))
    return jnp.where(mask, gathered, field)


def device_mesh(axis_name: str = DEFAULT_AXIS, devices=None) -> Mesh:
    """A 1-D :class:`jax.sharding.Mesh` over all (or the given) devices."""
    devs = np.asarray(jax.devices() if devices is None else devices)
    return Mesh(devs, (axis_name,))


def _fold(arr_PL: np.ndarray):
    """``[P, Lmax, *rest]`` → ``[P*Lmax, *rest]`` (fold the device axis into the
    leading dim for ``PartitionSpec('p')`` sharding)."""
    a = jnp.asarray(arr_PL)
    P, Lmax = a.shape[0], a.shape[1]
    return a.reshape((P * Lmax,) + a.shape[2:]), P, Lmax


def _unfold(arr, P: int, Lmax: int):
    """``[P*Lmax, *rest]`` → ``[P, Lmax, *rest]``."""
    return jnp.asarray(arr).reshape((P, Lmax) + jnp.asarray(arr).shape[1:])


def run_halo_exchange(field_PL, src_dev_PL, src_lane_PL, jmesh: Mesh,
                      axis_name: str = DEFAULT_AXIS):
    """Convenience wrapper: run :func:`halo_exchange` on a stacked ``[P, Lmax, *rest]``
    field via ``shard_map`` and return the result as ``[P, Lmax, *rest]``. The
    device mesh ``jmesh`` must have ``P`` devices on ``axis_name``.

    For S.7 the primitive is called directly inside the step's ``shard_map``; this
    wrapper is for standalone use and the S.3 gate.
    """
    field, P, Lmax = _fold(field_PL)
    sdev, _, _ = _fold(src_dev_PL)
    slane, _, _ = _fold(src_lane_PL)
    spec = PartitionSpec(axis_name)
    fn = jax.shard_map(
        lambda f, sd, sl: halo_exchange(f, sd, sl, axis_name),
        mesh=jmesh, in_specs=(spec, spec, spec), out_specs=spec,
    )
    out = fn(field, sdev.astype(jnp.int32), slane.astype(jnp.int32))
    return _unfold(out, P, Lmax)


def run_halo_exchange_compact(field_PL, send_PS, read_PL, jmesh: Mesh,
                              axis_name: str = DEFAULT_AXIS):
    """Standalone :func:`halo_exchange_compact` on a stacked ``[P, Lmax, *rest]`` field, with the
    ``[P, S]`` / ``[P, Lmax]`` maps of :func:`compact_exchange_maps`. Any backend."""
    field, P, Lmax = _fold(field_PL)
    send, _, _ = _fold(np.asarray(send_PS))
    read, _, _ = _fold(np.asarray(read_PL))
    spec = PartitionSpec(axis_name)
    fn = jax.shard_map(lambda f, s, r: halo_exchange_compact(f, s, r, axis_name),
                       mesh=jmesh, in_specs=(spec,) * 3, out_specs=spec)
    return _unfold(fn(field, send.astype(jnp.int32), read.astype(jnp.int32)), P, Lmax)


def run_halo_exchange_ragged(field_PL, rmap, jmesh: Mesh,
                             axis_name: str = DEFAULT_AXIS):
    """Standalone :func:`halo_exchange_ragged` (the B.0 gate + standalone use): run the
    halo-only ``ragged_all_to_all`` exchange on a stacked ``[P, Lmax, *rest]`` field via
    ``shard_map``. ``rmap`` is the kind's :class:`~fesom_jax.shard_mesh.RaggedExchange`
    (global ``[P, …]`` arrays); ``jmesh`` must have ``P`` devices on ``axis_name``."""
    field, P, Lmax = _fold(field_PL)
    recv_max = int(rmap.recv_max)
    fold0 = lambda a: _fold(np.asarray(a))[0]
    si = fold0(rmap.send_idx).astype(jnp.int32)
    ss = fold0(rmap.send_sizes).astype(jnp.int32)
    so = fold0(rmap.send_offsets).astype(jnp.int32)
    oo = fold0(rmap.out_offsets).astype(jnp.int32)
    rs = fold0(rmap.recv_sizes).astype(jnp.int32)
    rg = fold0(rmap.recv_gather).astype(jnp.int32)
    hm = fold0(rmap.halo_mask)
    spec = PartitionSpec(axis_name)

    def f(fld, si, ss, so, oo, rs, rg, hm):
        r = {"send_idx": si, "send_sizes": ss, "send_off": so, "out_off": oo,
             "recv_sizes": rs, "recv_gather": rg, "halo_mask": hm}
        return halo_exchange_ragged(fld, r, recv_max, axis_name)

    fn = jax.shard_map(f, mesh=jmesh, in_specs=(spec,) * 8, out_specs=spec)
    out = fn(field, si, ss, so, oo, rs, rg, hm)
    return _unfold(out, P, Lmax)


def run_halo_exchange_padded(field_PL, rmap, jmesh: Mesh,
                             axis_name: str = DEFAULT_AXIS):
    """Standalone :func:`halo_exchange_padded` (the Phase 8c gate + standalone use): run
    the slot-padded dense-``all_to_all`` exchange on a stacked ``[P, Lmax, *rest]`` field
    via ``shard_map``. ``rmap`` is the kind's :class:`~fesom_jax.shard_mesh.RaggedExchange`
    (global ``[P, …]`` arrays; only its ``pad_*``/``halo_mask`` fields are consumed);
    ``jmesh`` must have ``P`` devices on ``axis_name``. Any backend."""
    field, P, Lmax = _fold(field_PL)
    fold0 = lambda a: _fold(np.asarray(a))[0]
    ps = fold0(rmap.pad_src).astype(jnp.int32)
    pv = fold0(rmap.pad_valid)
    pp = fold0(rmap.pad_slotpos).astype(jnp.int32)
    hm = fold0(rmap.halo_mask)
    spec = PartitionSpec(axis_name)

    def f(fld, ps, pv, pp, hm):
        r = {"pad_src": ps, "pad_valid": pv, "pad_slotpos": pp, "halo_mask": hm}
        return halo_exchange_padded(fld, r, axis_name)

    fn = jax.shard_map(f, mesh=jmesh, in_specs=(spec,) * 5, out_specs=spec)
    out = fn(field, ps, pv, pp, hm)
    return _unfold(out, P, Lmax)


def run_halo_exchange_coloured(field_PL, cmap, hmask, jmesh: Mesh,
                               axis_name: str = DEFAULT_AXIS):
    """Standalone :func:`halo_exchange_coloured` (the Phase 8d gate + standalone use): run the
    coloured-``ppermute`` exchange on a stacked ``[P, Lmax, *rest]`` field via ``shard_map``.
    ``cmap`` is the kind's :class:`~fesom_jax.shard_mesh.ColouredExchange`, ``hmask`` its
    ``RaggedExchange.halo_mask`` (``[P, Lmax]``); ``jmesh`` must have ``P`` devices on
    ``axis_name``. Any backend."""
    field, P, Lmax = _fold(field_PL)
    fold0 = lambda a: _fold(np.asarray(a))[0]
    si = fold0(cmap.send_idx).astype(jnp.int32)
    sv = fold0(cmap.send_valid)
    cp = fold0(cmap.colpos).astype(jnp.int32)
    hm = fold0(hmask)
    meta = (cmap.perms, cmap.slots, cmap.offs)          # static
    spec = PartitionSpec(axis_name)

    def f(fld, si, sv, cp, hm):
        c = {"send_idx": si, "send_valid": sv, "colpos": cp, "halo_mask": hm}
        return halo_exchange_coloured(fld, c, meta, axis_name)

    fn = jax.shard_map(f, mesh=jmesh, in_specs=(spec,) * 5, out_specs=spec)
    out = fn(field, si, sv, cp, hm)
    return _unfold(out, P, Lmax)
