"""Padded dense-all_to_all halo exchange — a CPU-capable, AD-correct stand-in for
``lax.ragged_all_to_all`` (which is UNIMPLEMENTED on XLA:CPU and has a broken
reverse-mode transpose everywhere; see docs/JAX_RAGGED_A2A_BUG.md).

Idea: the ragged exchange ships, for each device pair (d→e), a chunk of
``send_sizes[d,e]`` lanes. A dense ``lax.all_to_all`` ships fixed-size blocks
between ALL pairs. So pad every pair-chunk to the global max chunk size
(``slot = recv_sizes.max()``, a static int), lay the send buffer out as P slots
of ``slot`` lanes, exchange once, and read halo lanes back out of the slotted
recv buffer. ``all_to_all``'s transpose is another ``all_to_all`` (trusted), so
gradients are correct by construction — unlike ``ragged_all_to_all``.

Measured volumes (CORE2, per device, per 3-D field exchange): all_gather 47 MB,
padded 0.24 MB @ dist_4 / 0.34 MB @ dist_8 (true ragged: 0.14 / 0.10 MB).

This module is DELIBERATELY non-invasive (prototype rule: new folder, no repo
edits): :func:`install` rebinds ``halo_exchange_ragged`` in ``fesom_jax.halo``
AND ``fesom_jax.ssh`` (ssh.py binds the name at import time), so running the
existing model with ``use_ragged=True`` transparently uses the padded exchange.
Everything is derived from the per-device ragged arrays already flowing through
the ``use_ragged`` plumbing:

  * receiver-frame chunk offsets are reconstructed in-trace as the exclusive
    cumsum of ``recv_sizes`` — valid because ``shard_mesh._ragged_exchange_map``
    builds ``recv_offsets`` exactly that way (axis-1 exclusive cumsum, sources
    in rank order), with the same canonical per-(e,d) chunk ordering on both
    sides;
  * the static slot width per entity kind is looked up by the trace-static key
    ``(recv_max, Lmax)`` — ``recv_max`` is passed as a python int and ``Lmax``
    is ``halo_mask.shape[0]``; :func:`install` asserts the keys are unique.

To productize: move the body into ``fesom_jax/halo.py`` as a third mode
(``use_padded``) with host-built index maps instead of in-trace reconstruction.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax

import fesom_jax.halo as _halo
import fesom_jax.ssh as _ssh

_ORIG_RAGGED = _halo.halo_exchange_ragged     # kept for uninstall(); never deleted


def padded_exchange(field, r: dict, slot: int, axis_name: str):
    """One broadcast halo exchange via slot-padded dense ``lax.all_to_all``.

    ``field``: this device's ``[Lmax, *rest]``; ``r``: the per-device ragged maps
    (same dict ``halo_exchange_ragged`` receives); ``slot``: static max pair-chunk
    size. Returns ``[Lmax, *rest]`` with halo lanes refreshed, interior/pad lanes
    untouched — same contract as the all_gather and ragged exchanges."""
    rest = field.shape[1:]
    P = r["send_sizes"].shape[0]
    S = int(slot)

    # --- send side: lay my per-destination chunks into P slots of S lanes ---
    p_ar = jnp.arange(S, dtype=jnp.int32)                              # [S]
    pos = r["send_off"][:, None] + p_ar[None, :]                       # [P,S] → send_idx
    valid = p_ar[None, :] < r["send_sizes"][:, None]                   # [P,S]
    nsend = r["send_idx"].shape[0]
    src = r["send_idx"][jnp.clip(pos, 0, nsend - 1)]                   # [P,S] local lanes
    buf = field[src.reshape(P * S)]                                    # [P*S, *rest]
    vmask = valid.reshape((P * S,) + (1,) * len(rest))
    buf = jnp.where(vmask, buf, jnp.zeros_like(buf))

    # --- the one collective: slot d on the recv side = what device d sent me ---
    recv = lax.all_to_all(buf, axis_name, split_axis=0, concat_axis=0, tiled=True)

    # --- recv side: ragged position q → (source d, within-chunk p) → slot index ---
    csum = jnp.cumsum(r["recv_sizes"])
    off = jnp.concatenate([jnp.zeros((1,), csum.dtype), csum[:-1]])    # my recv_offsets row
    q = r["recv_gather"]                                               # [Lmax]
    d = jnp.clip(jnp.searchsorted(off, q, side="right") - 1, 0, P - 1)
    slotpos = jnp.clip(d * S + (q - off[d]), 0, P * S - 1)             # [Lmax]
    gathered = recv[slotpos]                                           # [Lmax, *rest]

    hmask = r["halo_mask"].reshape(r["halo_mask"].shape + (1,) * len(rest))
    return jnp.where(hmask, gathered, field)


def make_adapter(slot_by_key: dict):
    """An ``halo_exchange_ragged``-signature function routing to :func:`padded_exchange`.

    ``slot_by_key``: ``{(recv_max, Lmax): slot}`` — both key parts are static at
    trace time (``recv_max`` is a python int argument, ``Lmax`` a shape)."""

    def adapter(field, r, recv_max, axis_name=_halo.DEFAULT_AXIS):
        key = (int(recv_max), int(r["halo_mask"].shape[0]))
        slot = slot_by_key[key]          # KeyError ⇒ a kind install() didn't see
        return padded_exchange(field, r, slot, axis_name)

    return adapter


def slot_registry(sm) -> dict:
    """Host-side: per-kind static slot widths from the ShardedMesh's ragged maps."""
    reg = {}
    for k in ("nod", "elem", "edge"):
        rx = sm.exchange_ragged[k]
        key = (int(rx.recv_max), int(rx.halo_mask.shape[1]))
        if key in reg and reg[key] != int(rx.recv_sizes.max()):
            raise AssertionError(f"slot-registry key collision for kind {k}: {key}")
        reg[key] = max(int(rx.recv_sizes.max()), 1)
    return reg


def install(sm) -> dict:
    """Rebind ``halo_exchange_ragged`` in fesom_jax.halo AND fesom_jax.ssh to the
    padded exchange (ssh.py imported the name directly, so both need rebinding).
    Call BEFORE tracing. Returns the slot registry (for logging)."""
    reg = slot_registry(sm)
    adapter = make_adapter(reg)
    _halo.halo_exchange_ragged = adapter
    _ssh.halo_exchange_ragged = adapter
    return reg


def uninstall() -> None:
    """Restore the original ``ragged_all_to_all`` implementation."""
    _halo.halo_exchange_ragged = _ORIG_RAGGED
    _ssh.halo_exchange_ragged = _ORIG_RAGGED
