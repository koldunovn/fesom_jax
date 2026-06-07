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

    def exchange(self, field, kind: str):
        """Broadcast-refresh ``field``'s halo lanes from their owners (a no-op on
        interior + pad lanes), for the given entity ``kind``."""
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
