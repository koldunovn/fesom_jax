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
    #                             ({pad_src, pad_valid, pad_slotpos, halo_mask}) instead
    recv_max: dict = None       # {kind: int} static recv-buffer leading dim (ragged only)
    use_ragged: bool = False    # pick ragged_all_to_all over all_gather (GPU, forward-only)
    use_padded: bool = False    # pick the slot-padded dense all_to_all (Phase 8c) —
    #                             every backend, AD-correct; mutually exclusive with use_ragged

    def exchange(self, field, kind: str):
        """Broadcast-refresh ``field``'s halo lanes from their owners (a no-op on
        interior + pad lanes), for the given entity ``kind``. Transport: the padded
        dense ``all_to_all`` when ``use_padded`` (Phase 8c), the halo-only
        ``ragged_all_to_all`` when ``use_ragged`` (Phase 8b), else ``all_gather``."""
        if self.use_padded and self.exch_ragged is not None:
            return halo_exchange_padded(field, self.exch_ragged[kind], self.axis_name)
        if self.use_ragged and self.exch_ragged is not None:
            return halo_exchange_ragged(
                field, self.exch_ragged[kind], self.recv_max[kind], self.axis_name)
        src_dev, src_lane = self.exch[kind]
        return halo_exchange(field, src_dev, src_lane, self.axis_name)


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
