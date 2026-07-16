"""Per-process LOCAL forcing build (Task B2 efficiency — the NG5 host-forcing fix).

Measured at NG5 (7.4 M / 64 GPU): the per-chunk HOST forcing build was ~110 s, **84 % of it the
JRA bilinear interpolation** (`step_forcing` = 4 s/step) — and it ran **redundantly on every one of
the 16 nodes**, each interpolating all 7.4 M global nodes when it owns only ~472 k (4 of 64
partitions). That idle-GPU overhead is ~half of R0's wall-clock and makes R1+ infeasible.

This builds the forcing for **only the nodes this process's local partitions need** (their
`myList`, owned + halo) and scatters them into a `[P, n_steps, Lmax]` `StepForcing` with **only the
local partitions filled** (others left at the pad value). That is **bit-identical** to the global
build's local shards because :func:`fesom_jax.integrate_sharded._to_global_sharded` places the
sharded array via `make_array_from_callback`, whose per-shard callback reads ONLY the local
partitions' slices of the host array (the non-local rows are never touched). The win is the
``npes / len(local_parts)`` (= 16×) reduction in interpolation work.

The local reader is a fresh :class:`~fesom_jax.surface_forcing.SurfaceForcing` built on a **sub-mesh** of
the local nodes (the JRA/SSS/chl readers need only node lon/lat), so it has fully independent state
(the JRA `_Field` carries mutable per-field coefficients — a shared reader would corrupt them).
"""
from __future__ import annotations

import numpy as np

from . import jra55, sss_runoff
from .surface_forcing import SurfaceForcing, StepForcing
from .shard_mesh import _default_pad, local_sizes


class _SubMesh:
    """The minimal mesh view the forcing readers consume: node lon/lat for a node subset.

    ``JRA55Reader`` reads ``geo_coord_nod2D`` / ``coord_nod2D`` / ``nod2D``; the SSS and chl
    builders read ``geo_coord_nod2D`` only. Subsetting these to ``nodes`` builds readers that
    interpolate exactly those global nodes (in ``nodes`` order)."""

    def __init__(self, mesh, nodes):
        self.nod2D = int(len(nodes))
        self.geo_coord_nod2D = np.asarray(mesh.geo_coord_nod2D)[nodes]
        self.coord_nod2D = np.asarray(mesh.coord_nod2D)[nodes]


def local_partitions(npes: int):
    """The partition indices whose folded shard is **addressable by this process**.

    The sharded forcing maps partition ``i`` → ``jax.devices()[:npes][i]``; this process owns the
    partitions whose device is in ``jax.local_devices()``. Single-process ⇒ all ``npes``."""
    import jax
    all_devs = jax.devices()[:npes]
    local = set(jax.local_devices())
    return [i for i in range(npes) if all_devs[i] in local]


class LocalForcing:
    """A local-node :class:`SurfaceForcing` + the scatter map into ``[P, n_steps, Lmax]``.

    Build once (per process) via :func:`build_local_forcing`; call :meth:`stack_partitioned` per
    chunk. The output is consumed exactly like :func:`shard_mesh.partition_step_forcing`'s — a
    ``StepForcing`` of ``[P, n_steps, Lmax]`` leaves — but only ``local_parts`` are filled."""

    def __init__(self, cf: SurfaceForcing, part, npes: int, local_parts, local_nodes):
        self.cf = cf
        self.part = part
        self.npes = int(npes)
        self.local_parts = list(local_parts)
        self.local_nodes = np.asarray(local_nodes)
        _, Lmax = local_sizes(part)
        self.L = int(Lmax["nod"])
        # lane positions: for each local partition, where its myList nodes sit in local_nodes
        # (local_nodes is sorted-unique ⊇ every local myList ⇒ searchsorted is exact).
        self.lane = {p: np.searchsorted(self.local_nodes, np.asarray(part.myList_nod2D[p]))
                     for p in self.local_parts}

    def _scatter_last(self, arr):
        """Scatter a ``[…, n_local]`` host array into ``[P, …, Lmax]`` (only
        ``local_parts``' myList lanes filled; every other row/lane takes
        ``_default_pad`` — never read by the per-shard sharding callback)."""
        pad = _default_pad(arr.dtype)
        full = np.full((self.npes,) + arr.shape[:-1] + (self.L,), pad, arr.dtype)
        for p in self.local_parts:
            idx = self.lane[p]
            full[p, ..., :len(idx)] = arr[..., idx]
        return full

    def stack_tables_partitioned(self, dates, *, nb: int, nm: int = 2):
        """The ``forcing.on_device`` twin of :meth:`stack_partitioned` (the NG5 increment):
        per-chunk bracket coefficient TABLES built on the local sub-mesh only — the same
        ``npes / len(local_parts)`` interpolation saving, since ``bracket_schedule``'s
        ``_gather`` is per-node — scattered into ``[P, …, Lmax]``
        :class:`~fesom_jax.surface_forcing.ForcingTables` exactly like
        :func:`~fesom_jax.shard_mesh.partition_forcing_tables`'s local shards (bit-identical;
        the non-local rows stay pad). The returned
        :class:`~fesom_jax.surface_forcing.ForcingSched` is node-independent, hence identical
        to the global build's."""
        tables, sched = self.cf.stack_tables(dates, nb=nb, nm=nm)
        return (type(tables)(**{name: self._scatter_last(np.asarray(getattr(tables, name)))
                                for name in tables._fields}), sched)

    def forcing_const_partitioned(self):
        """The local twin of ``partition_forcing_const(forcing_device_const(cf))``: the g2r
        trig table comes from the LOCAL sub-mesh reader (per-node ⇒ bit-identical to the
        global table's local shards), scattered like :meth:`_scatter_last`; the ``M``
        rotation matrix and scalars are replicated as-is."""
        from .surface_forcing import forcing_device_const
        fc = forcing_device_const(self.cf)
        out = {}
        for name in fc._fields:
            arr = np.asarray(getattr(fc, name))
            out[name] = arr if (name == "M" or arr.ndim == 0) else self._scatter_last(arr)
        return type(fc)(**out)

    def stack_partitioned(self, dates, *, xp=np) -> StepForcing:
        """Interpolate ``dates`` for the local nodes and scatter into ``[P, n_steps, Lmax]``
        (only ``local_parts`` filled; pad lanes + non-local rows take ``_default_pad`` — the
        non-local rows are never read by the per-shard sharding callback)."""
        seq_local = self.cf.stack(dates, xp=xp)            # [n_steps, n_local] host leaves
        out = {}
        for name in seq_local._fields:
            arr = np.asarray(getattr(seq_local, name))     # [n_steps, n_local]
            n_steps = arr.shape[0]
            pad = _default_pad(arr.dtype)
            full = np.full((self.npes, n_steps, self.L) + arr.shape[2:], pad, arr.dtype)
            for p in self.local_parts:
                idx = self.lane[p]                         # [len_p] positions into local_nodes
                full[p, :, :len(idx)] = arr[:, idx]        # myList lanes; rest stay pad
            out[name] = full
        return StepForcing(**out)

    def reopen_year(self, year: int):
        """Roll the local forcing to ``year`` (delegates to the wrapped SurfaceForcing/JRA reader; the
        local sub-mesh stencil is year-independent and kept). Returns ``self``."""
        self.cf.reopen_year(year)
        return self


def build_local_forcing(mesh, year, part, npes, *, sst_ic=None, local_parts=None,
                        jra_dir=None,
                        sss_path=None,
                        runoff_path=None,
                        chl_path=None,
                        chl_const=None, static=None) -> LocalForcing:
    """Build a :class:`LocalForcing` for this process's local partitions.

    The four input-path kwargs default to ``None`` ⇒ resolved by the readers through
    :mod:`fesom_jax.paths` (env var, else the Levante default); pass a string to override.

    ``local_parts`` overrides the :func:`local_partitions` device-derived set (the test path —
    so a single process can exercise the multi-partition scatter without N real devices).
    ``static`` is the global :class:`~fesom_jax.surface_forcing.ForcingStatic` (reused as-is; the
    per-step path never touches it). ``sst_ic`` is accepted for signature parity (the static, not
    the per-step forcing, uses it) and ignored here."""
    if local_parts is None:
        local_parts = local_partitions(npes)
    lists = [np.asarray(part.myList_nod2D[p]) for p in local_parts]
    local_nodes = (np.unique(np.concatenate(lists)) if lists
                   else np.zeros(0, dtype=np.int64))
    sub = _SubMesh(mesh, local_nodes)
    chl = (np.full((12, int(len(local_nodes))), float(chl_const), dtype=np.float64)
           if chl_const is not None else sss_runoff.build_chl_clim(sub, chl_path))
    cf = SurfaceForcing(jra=jra55.JRA55Reader(sub, int(year), jra_dir),
                     sss=sss_runoff.build_reader(sub, sss_path, runoff_path),
                     chl_clim=chl, static=static)
    return LocalForcing(cf, part, npes, local_parts, local_nodes)
