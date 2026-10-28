"""SSH RHS + CG solve — substeps 8–9 (Task 2.7), the AD-critical solver.

Literal vectorized port of ``fesom_ssh.c`` for the Phase-2 pi config (linfs ALE,
``alpha=theta=1``, no cavity / no partial cells, single MPI rank):

* :func:`compute_ssh_rhs` — substep 8 (``fesom_compute_ssh_rhs_linfs``,
  ``fesom_ssh.c:261``): an antisymmetric edge→node scatter of the depth-integrated
  transport ``alpha·((v+vr)·dx − (u+ur)·dy)·helem``. At rest ``uv=0`` but
  ``uv_rhs=du`` (the wind-forced increment from substep 7), so the field is **not**
  trivial. With ``alpha=1`` the ``(1−alpha)·ssh_rhs_old`` blend term vanishes.

* :class:`SSHOperator` / :func:`build_ssh_operator` — the element-Galerkin
  stiffness matrix ``S`` and the **MITgcm-style symmetric** preconditioner
  (``fesom_ssh.c:36-255``). In **linfs the operator is static** (built once, reused
  every step — ``fesom_ssh.c:9-12``: ``update_stiff_mat_ale`` is gated off): the
  factor ``−g·dt·α·θ·(depth)`` uses the fixed ``zbar`` depths, never the evolving
  ``hbar``. We represent both ``S`` and ``M⁻¹`` as **static sparse COO matvecs**
  (``segment_sum``), exactly the precomputed-CSR the C reuses — not a per-step
  closure.

* :func:`solve_ssh` — substep 9 (``fesom_ssh_solve_cg``, ``fesom_ssh.c:384``):
  preconditioned CG. **Fidelity note (verified by prototype):** the C stops at a
  *loose* relative residual ``soltol=1e-5`` (only ~3 iterations on pi, ``cond(S)≈
  800``), so the dumped ``d_eta`` is the **early-stopped** iterate — it matches the
  3-iteration PCG to ~1e-18 but the *exact* solve only to ~2e-10. We therefore
  **replicate the C PCG exactly** (same static ``S``, same MITgcm preconditioner,
  same ``x0``, same ``‖r‖<soltol·‖b‖`` stop) for the forward value, and give it an
  implicit-derivative rule (:func:`jax.custom_jvp`) so the *derivative* is the clean
  ``S⁻¹`` in both directions (tangent and transpose solves converge tightly). Forward =
  dump-matching early-stop; tangent-linear and adjoint = exact-solve implicit derivative,
  exact transposes of each other. The residual trajectory
  ``[65, 1.0, 0.015]`` vs ``rtol=0.197`` makes the 3-iteration stop unambiguous, so
  ``segment_sum`` reassociation cannot change the iteration count.
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from functools import partial

from jax import lax, tree_util
from jax.ad_checkpoint import checkpoint_name
from jax.custom_derivatives import SymbolicZero

from . import ops
from .config import DT_DEFAULT, G, MAXITER, SOLTOL, SSH_ALPHA, SSH_THETA
from .halo import (halo_exchange, halo_exchange_coloured, halo_exchange_padded,
                   halo_exchange_ragged)
from .mesh import Mesh
from . import precision as PR
from .reductions import global_dot, global_dot_pair

# Tolerance for the *gradient* (transpose) solve — tight so the implicit-diff
# cotangent ``S⁻¹·x̄`` is accurate, independent of the loose forward ``soltol``.
GRAD_SOLTOL = 1.0e-13


# ==========================================================================
# Static stiffness operator + MITgcm preconditioner
# ==========================================================================
@dataclasses.dataclass(frozen=True)
class SSHOperator:
    """Static linfs SSH operator as two sparse COO matvecs sharing one pattern.

    ``rows``/``cols`` index the ``nnz`` structural entries (node + edge-adjacency,
    mirroring the C neighbour list); ``stiff_vals`` are ``S`` and ``precond_vals``
    are the MITgcm ``M⁻¹``. Built once per mesh (linfs ⇒ never rebuilt).
    """

    rows: jax.Array          # (nnz,) i4
    cols: jax.Array          # (nnz,) i4
    stiff_vals: jax.Array    # (nnz,) f8   S
    precond_vals: jax.Array  # (nnz,) f8   M^{-1} (MITgcm symmetric)
    diag: jax.Array          # (N,)  f8    diag(S)  (diagnostics / preconditioner)
    n_nodes: int = dataclasses.field(metadata={"static": True})
    # CGPOLY (M7 lever, port_kokkos Task E.3): (degree, lam_min, lam_max) enables the
    # degree-k Chebyshev polynomial preconditioner (:func:`ssh_precond_cheb`) in place
    # of the MITgcm M⁻¹. STATIC meta (Python floats via :func:`enable_cheb_precond`) —
    # identical on every process ⇒ identical executables. None (default) ⇒ byte-identical.
    cheb: tuple = dataclasses.field(default=None, metadata={"static": True})
    # ELL layout (:func:`to_ell`, opt-in FESOM_SSH_ELL / ``gpu.ssh_ell``): ``ell = D`` ⇒ the nnz arrays are
    # slot-major ``[D, n_nodes]`` (entry ``k·n_nodes + i`` = row i's k-th nonzero, zero-padded) and the
    # matvecs sum each row's slots in order instead of a ``segment_sum`` (no atomics on a GPU). 0 ⇒ COO.
    ell: int = dataclasses.field(default=0, metadata={"static": True})
    # The zstar increment as explicit per-entry weights (:func:`with_increment`, opt-in FESOM_SSH_INC_ELL /
    # ``gpu.ssh_inc_ell``): entry i's ΔA value is ``g·dt·α·θ · Σ_j depth[inc_el[i, j]] · inc_w[i, j]`` with
    # ``depth = −mean₃(hbar)`` per element (0 in a cavity; unused slots: element 0, weight 0). None ⇒ the matrix-free
    # :func:`stiff_increment_matvec`.
    inc_el: jax.Array = None     # (nnz, Dc) i4
    inc_w: jax.Array = None      # (nnz, Dc) f8


tree_util.register_dataclass(
    SSHOperator,
    data_fields=["rows", "cols", "stiff_vals", "precond_vals", "diag", "inc_el", "inc_w"],
    meta_fields=["n_nodes", "cheb", "ell"],
)


# ==========================================================================
# Distributed CG context + partitioned operator (Phase 8, Task S.6)
# ==========================================================================
@dataclasses.dataclass(frozen=True)
class SSHHalo:
    """Per-device node halo-exchange context for the distributed CG, **carried
    inside** ``shard_map``. ``src_dev``/``src_lane`` are the
    :class:`~fesom_jax.shard_mesh.ShardedMesh` node exchange map (``[Lmax_nod]``);
    ``owned_mask`` (``[Lmax_nod]`` bool, True on ``[0:myDim)``) selects the owned
    lanes for the CG dot-products; ``n_global`` is the global ``nod2D`` (so the
    residual RMS ``√(Σr²/n)`` divides by the GLOBAL count, identical to single
    device — the load-bearing iteration-count invariant). Passing ``halo=None`` to
    the solver is the dense single-device path."""

    src_dev: jax.Array       # (Lmax_nod,) i4
    src_lane: jax.Array      # (Lmax_nod,) i4
    owned_mask: jax.Array    # (Lmax_nod,) bool
    n_global: int = dataclasses.field(metadata={"static": True})
    axis_name: str = dataclasses.field(default="p", metadata={"static": True})
    # --- Phase 8b: the halo-only ragged_all_to_all node exchange for the CG (the
    # DOMINANT per-step comm — ~2 exchanges × ~127 CG iters). None ⇒ all_gather.
    # Under ``use_padded`` (Phase 8c) the SAME slot carries the padded dense-a2a maps
    # ({pad_src, pad_valid, pad_slotpos, halo_mask}) instead. ---
    ragged: dict = dataclasses.field(default=None)          # {send_idx, send_sizes, send_off,
    #                                       out_off, recv_sizes, recv_gather, halo_mask} or None
    recv_max: int = dataclasses.field(default=0, metadata={"static": True})
    use_ragged: bool = dataclasses.field(default=False, metadata={"static": True})
    use_padded: bool = dataclasses.field(default=False, metadata={"static": True})
    # --- Phase 8d: the coloured-ppermute rounds. `ragged` above carries the coloured maps
    # ({send_idx, send_valid, colpos, halo_mask}); `col_meta` is the node kind's STATIC
    # (perms, slots, offs) — lax.ppermute needs a Python perm, so it must be a meta field
    # (a pytree leaf would be traced into an array and the perm would be unusable). ---
    use_coloured: bool = dataclasses.field(default=False, metadata={"static": True})
    col_meta: tuple = dataclasses.field(default=None, metadata={"static": True})


tree_util.register_dataclass(
    SSHHalo,
    data_fields=["src_dev", "src_lane", "owned_mask", "ragged"],
    meta_fields=["n_global", "axis_name", "recv_max", "use_ragged", "use_padded",
                 "use_coloured", "col_meta"],
)


def _halo_refresh(x, halo: "SSHHalo"):
    """The CG's node halo refresh, routed by the SSHHalo's transport flags: coloured
    ppermute (Phase 8d, every backend + AD-correct + no pad factor) > padded dense a2a
    (Phase 8c, every backend + AD-correct) > ragged (Phase 8b, GPU forward-only) >
    all_gather (the oracle). Shared by matvec / precond / zstar. This is the DOMINANT
    per-step comm (~2 exchanges × ~127 CG iterations), so the transport choice is felt
    here first."""
    if halo.use_coloured:
        return halo_exchange_coloured(x, halo.ragged, halo.col_meta, halo.axis_name)
    if halo.use_padded:
        return halo_exchange_padded(x, halo.ragged, halo.axis_name)
    if halo.use_ragged:
        return halo_exchange_ragged(x, halo.ragged, halo.recv_max, halo.axis_name)
    return halo_exchange(x, halo.src_dev, halo.src_lane, halo.axis_name)


@dataclasses.dataclass(frozen=True)
class ShardedSSHOperator:
    """Per-device padded SSH operator (host numpy ``[P, …]`` arrays). Each device's
    slice is a LOCAL-indexed COO matvec: ``rows``/``cols`` reference local node lanes
    ``[0, Lmax_nod)``, ``stiff_vals``/``precond_vals`` are the kept ``S`` / ``M⁻¹``
    entries, padded to a common ``nnz_max`` with zero-value entries (``row=col=0,
    val=0`` ⇒ scatter ``0`` to lane 0 ⇒ inert). Reconstruct one device's
    :class:`SSHOperator` (``n_nodes=Lmax_nod``) inside the ``shard_map`` body."""

    rows: np.ndarray          # [P, nnz_max] i4  local row (node) lane
    cols: np.ndarray          # [P, nnz_max] i4  local col (node) lane
    stiff_vals: np.ndarray    # [P, nnz_max] f8
    precond_vals: np.ndarray  # [P, nnz_max] f8
    diag: np.ndarray          # [P, Lmax_nod] f8  (the Chebyshev diag scaling; pad lanes 1.0)
    P: int
    nnz_max: int
    Lmax_nod: int
    cheb: tuple = None        # carried through from the global op (static meta)
    ell: int = 0              # :func:`to_ell_sharded`: nnz_max = ell·Lmax_nod in the ELL layout


def ssh_ell_enabled() -> bool:
    """``FESOM_SSH_ELL=1`` (``gpu.ssh_ell``, read where the run builds its operators): the SSH matvecs run on
    the ELL layout of :func:`to_ell` / :func:`to_ell_sharded`."""
    return os.environ.get("FESOM_SSH_ELL", "0").strip() not in ("", "0")


def _ell_layout(rows, cols, sv, pv, n_rows: int, keep):
    """Slot-major ELL arrays from COO ones: each row's kept entries in their COO order (a stable sort by
    row), slot k of row i at ``k·n_rows + i``, missing slots ``(row i, col 0, value 0)``."""
    rows, cols, sv, pv = rows[keep], cols[keep], sv[keep], pv[keep]
    order = np.argsort(rows, kind="stable")
    r = rows[order]
    counts = np.bincount(r, minlength=n_rows)
    D = int(counts.max())
    slot = np.arange(r.size) - np.repeat(np.cumsum(counts) - counts, counts)
    pos = slot * n_rows + r
    out_rows = np.tile(np.arange(n_rows, dtype=np.int32), D)
    out = [np.zeros(D * n_rows, dtype=a.dtype) for a in (cols, sv, pv)]
    for o, a in zip(out, (cols, sv, pv)):
        o[pos] = a[order]
    return D, out_rows, out[0], out[1], out[2]


def ssh_inc_ell_enabled() -> bool:
    """``FESOM_SSH_INC_ELL=1`` (``gpu.ssh_inc_ell``): the gathered solve's zstar increment as explicit entries."""
    return os.environ.get("FESOM_SSH_INC_ELL", "0").strip() not in ("", "0")


def with_increment(op: SSHOperator, mesh) -> SSHOperator:
    """``op`` (global, COO) with the zstar increment's explicit weights (host). ΔA·x
    (:func:`stiff_increment_matvec`) is linear in x and in the element depth: edge e, side s (element
    ``edge_tri[e, s]``, cross columns (0, 1) with sign +1 or (2, 3) with −1) adds, for each vertex k of that element,
    ``w = sign·(∂N_k/∂x·cross[e, dy] − ∂N_k/∂y·cross[e, dx])`` times depth to entry (edges[e, 0], vertex) and
    ``−w`` to entry (edges[e, 1], vertex). Every such entry is in the operator's pattern (asserted)."""
    rows, cols = np.asarray(op.rows), np.asarray(op.cols)
    N = op.n_nodes
    key = rows.astype(np.int64) * N + cols
    order = np.argsort(key)
    en = np.asarray(mesh.elem_nodes)
    g = np.asarray(mesh.gradient_sca)
    et = np.asarray(mesh.edge_tri)
    cross = np.asarray(mesh.edge_cross_dxdy)
    edges = np.asarray(mesh.edges)
    R, C, L, W = [], [], [], []
    for side, (dx, dy, sign) in enumerate(((0, 1, 1.0), (2, 3, -1.0))):
        el = et[:, side]
        ok = el >= 0
        els = np.where(ok, el, 0)
        for k in range(3):
            w = sign * (g[els, k] * cross[:, dy] - g[els, 3 + k] * cross[:, dx])
            col = en[els, k]
            for tgt, sg in ((edges[:, 0], 1.0), (edges[:, 1], -1.0)):
                R.append(tgt[ok]); C.append(col[ok]); L.append(el[ok]); W.append(sg * w[ok])
    R, C, L, W = (np.concatenate(v) for v in (R, C, L, W))
    k2 = R.astype(np.int64) * N + C
    at = np.searchsorted(key[order], k2)
    at = np.minimum(at, key.size - 1)
    if not np.array_equal(key[order][at], k2):
        raise ValueError("with_increment: a zstar increment entry lies outside the operator's pattern")
    slot = order[at]                                         # COO entry of each contribution
    srt = np.argsort(slot, kind="stable")
    slot, L, W = slot[srt], L[srt], W[srt]
    counts = np.bincount(slot, minlength=key.size)
    Dc = int(counts.max())
    j = np.arange(slot.size) - np.repeat(np.cumsum(counts) - counts, counts)
    inc_el = np.zeros((key.size, Dc), dtype=np.int32)       # unused slots: element 0 with weight 0
    inc_w = np.zeros((key.size, Dc), dtype=np.float64)
    inc_el[slot, j] = L
    inc_w[slot, j] = W
    return dataclasses.replace(op, inc_el=jnp.asarray(inc_el), inc_w=jnp.asarray(inc_w))


def _increment_values(op: SSHOperator, mesh, hbar, factor: float):
    """The explicit ΔA entries of :func:`with_increment` for this ``hbar``.

    Single precision (D1): the entries are formed in WP_full from the WP ``hbar`` and stay WP_full (the live matrix
    ``S + ΔA`` is then float64; :func:`_spmv` sums its rows in float64). Identity in double precision."""
    hbar = PR.full(hbar)
    en = mesh.elem_nodes
    dhe = (hbar[en[:, 0]] + hbar[en[:, 1]] + hbar[en[:, 2]]) / 3.0
    depth = jnp.where(mesh.ulevels == 1, -dhe, 0.0)
    gath = depth[op.inc_el] * PR.full(op.inc_w)               # (nnz, Dc)
    acc = gath[:, 0]
    for k in range(1, gath.shape[1]):
        acc = acc + gath[:, k]
    return acc * factor


def to_ell(op: SSHOperator) -> SSHOperator:
    """``op`` in the ELL layout (host). The matvecs add each row's products in the COO order of that row; XLA
    fuses each multiply-add of that chain into an FMA, so the result is the ``segment_sum``'s to round-off
    (1 ulp), not bitwise (``tests/test_ssh_ell.py``)."""
    rows = np.asarray(op.rows)
    D, r, c, sv, pv = _ell_layout(rows, np.asarray(op.cols), np.asarray(op.stiff_vals),
                                  np.asarray(op.precond_vals), op.n_nodes, np.ones(rows.size, bool))
    extra = {}
    if op.inc_el is not None:                                 # move the increment rows like the values
        real = np.zeros(D * op.n_nodes, bool)
        order = np.argsort(rows, kind="stable")
        counts = np.bincount(rows[order], minlength=op.n_nodes)
        slot = np.arange(rows.size) - np.repeat(np.cumsum(counts) - counts, counts)
        dest = slot * op.n_nodes + rows[order]
        real[dest] = True
        src = np.zeros(D * op.n_nodes, dtype=np.int64)
        src[dest] = order
        ie, iw = np.asarray(op.inc_el), np.asarray(op.inc_w)
        extra = {"inc_el": jnp.asarray(np.where(real[:, None], ie[src], 0).astype(np.int32)),
                 "inc_w": jnp.asarray(np.where(real[:, None], iw[src], 0.0))}
    return dataclasses.replace(op, rows=jnp.asarray(r), cols=jnp.asarray(c), stiff_vals=jnp.asarray(sv),
                               precond_vals=jnp.asarray(pv), ell=D, **extra)


def to_ell_sharded(sop: "ShardedSSHOperator") -> "ShardedSSHOperator":
    """Each device's local operator in the ELL layout, padded to the largest row count over the devices.
    The partition's padding entries ``(row 0, col 0, both values 0)`` are dropped first (a real (0, 0)
    entry is a diagonal and never zero)."""
    lay = []
    for d in range(sop.P):
        rows, cols = sop.rows[d], sop.cols[d]
        sv, pv = sop.stiff_vals[d], sop.precond_vals[d]
        pad = (rows == 0) & (cols == 0) & (sv == 0.0) & (pv == 0.0)
        lay.append(_ell_layout(rows, cols, sv, pv, sop.Lmax_nod, ~pad))
    D = max(x[0] for x in lay)
    L = sop.Lmax_nod
    arr = {k: np.zeros((sop.P, D * L), dtype=dt) for k, dt in
           (("rows", np.int32), ("cols", np.int32), ("sv", np.float64), ("pv", np.float64))}
    for d, (Dd, r, c, sv, pv) in enumerate(lay):
        arr["rows"][d] = np.tile(np.arange(L, dtype=np.int32), D)
        arr["cols"][d, :Dd * L], arr["sv"][d, :Dd * L], arr["pv"][d, :Dd * L] = c, sv, pv
    return dataclasses.replace(sop, rows=arr["rows"], cols=arr["cols"], stiff_vals=arr["sv"],
                               precond_vals=arr["pv"], nnz_max=D * L, ell=D)


def _spmv(op: SSHOperator, vals, x):
    """``Σ_col vals·x[col]`` per row: a ``segment_sum`` on the COO layout, or the ELL layout's slots added in
    order (one fused gather-and-add, deterministic, no atomics).

    Single precision (FESOM2 #994, D7): products and row sums in WP_full on both layouts; the callers round the row
    once (identity in double precision)."""
    p = PR.full(vals) * PR.full(x[op.cols])
    if not op.ell:
        return jax.ops.segment_sum(p, op.rows, num_segments=op.n_nodes)
    p = p.reshape(op.ell, op.n_nodes)
    y = p[0]
    for k in range(1, op.ell):
        y = y + p[k]
    return y


def partition_ssh_operator(op: SSHOperator, partition) -> ShardedSSHOperator:
    """Partition the static global :class:`SSHOperator` by node ownership.

    For each device keep the structural entries whose **row AND column are local
    nodes** (interior+halo), remapped to local lanes; the matvec is then a local
    ``segment_sum`` into ``[Lmax_nod]``. Owned rows are EXACT because the operator
    loop-bound holds: every dropped owned-row entry (column outside the node halo)
    has **exactly zero** value (verified on CORE2 dist_2/dist_4 — the operator keeps
    the topological pattern incl. numeric zeros, and the far "wing" columns are all
    zeros). The build **asserts** no NONZERO owned-row entry is dropped, so a mesh /
    config that ever violated it would fail loudly rather than silently corrupt the
    owned matvec. ``partition.npes == 1`` (``synth_serial``) is the identity remap ⇒
    the per-device operator equals the dense one (the no-op invariant).
    """
    from .shard_mesh import local_sizes

    rows_g = np.asarray(op.rows)
    cols_g = np.asarray(op.cols)
    sv_g = np.asarray(op.stiff_vals)
    pv_g = np.asarray(op.precond_vals)
    diag_g = np.asarray(op.diag)
    N = op.n_nodes
    P = partition.npes
    nonzero = (sv_g != 0.0) | (pv_g != 0.0)

    _, Lmax = local_sizes(partition)
    Lmax_nod = Lmax["nod"]

    sel_rows, sel_cols, sel_sv, sel_pv, nnz = [], [], [], [], []
    diag_PL = np.full((P, Lmax_nod), 1.0, dtype=np.float64)
    for d in range(P):
        ml = np.asarray(partition.myList_nod2D[d])
        md = int(partition.myDim_nod2D[d])
        n_local = ml.size
        g2l = np.full(N, -1, dtype=np.int64)
        g2l[ml] = np.arange(n_local)
        rl = g2l[rows_g]
        cl = g2l[cols_g]
        owned_row = (rl >= 0) & (rl < md)
        dropped_nonzero = owned_row & (cl < 0) & nonzero
        if dropped_nonzero.any():
            raise ValueError(
                f"partition_ssh_operator: device {d} drops "
                f"{int(dropped_nonzero.sum())} NONZERO owned-row operator entries "
                f"(column outside the node halo) — the SSH stencil exceeds the node "
                f"halo; the local matvec would be incomplete on owned rows."
            )
        keep = (rl >= 0) & (cl >= 0)
        sel_rows.append(rl[keep].astype(np.int32))
        sel_cols.append(cl[keep].astype(np.int32))
        sel_sv.append(sv_g[keep].astype(np.float64))
        sel_pv.append(pv_g[keep].astype(np.float64))
        nnz.append(int(keep.sum()))
        diag_PL[d, :n_local] = diag_g[ml]

    nnz_max = int(max(nnz))
    rows_PL = np.zeros((P, nnz_max), dtype=np.int32)
    cols_PL = np.zeros((P, nnz_max), dtype=np.int32)
    sv_PL = np.zeros((P, nnz_max), dtype=np.float64)
    pv_PL = np.zeros((P, nnz_max), dtype=np.float64)
    for d in range(P):
        k = nnz[d]
        rows_PL[d, :k] = sel_rows[d]
        cols_PL[d, :k] = sel_cols[d]
        sv_PL[d, :k] = sel_sv[d]
        pv_PL[d, :k] = sel_pv[d]

    return ShardedSSHOperator(
        rows=rows_PL, cols=cols_PL, stiff_vals=sv_PL, precond_vals=pv_PL,
        diag=diag_PL, P=P, nnz_max=nnz_max, Lmax_nod=Lmax_nod, cheb=op.cheb,
    )


def local_ssh_operator(sop: ShardedSSHOperator, d: int) -> SSHOperator:
    """Reconstruct device ``d``'s local :class:`SSHOperator` (host helper / tests).
    Inside ``shard_map`` build the same object from the per-device sharded leaves."""
    return SSHOperator(
        rows=jnp.asarray(sop.rows[d]), cols=jnp.asarray(sop.cols[d]),
        stiff_vals=jnp.asarray(sop.stiff_vals[d]),
        precond_vals=jnp.asarray(sop.precond_vals[d]),
        diag=jnp.asarray(sop.diag[d]), n_nodes=sop.Lmax_nod, cheb=sop.cheb, ell=sop.ell,
    )


# ==========================================================================
# The SSH solve gathered onto every device (opt-in ``ssh.gathered``; CORE2 GPU speed plan A4)
# ==========================================================================
# The sharded CG pays ~2 collectives per iteration (the halo refresh of the iterate and the
# reduction of its dot products), ~130-250 latency-bound collectives a CORE2 step on 4 GPUs.
# On a small mesh the whole 2-D system fits every device: all-gather the right-hand side, the
# warm start and hbar ONCE, run the dense single-device solve redundantly on every device, keep
# the local lanes. On identical global inputs that is the dense solve itself (bitwise). Worth it
# only where the 2-D system is small (CORE2-class); off by default.
@dataclasses.dataclass(frozen=True)
class SSHGeom:
    """The global mesh arrays :func:`stiff_increment_matvec` reads (the zstar increment), as a
    pytree that can be replicated into ``shard_map`` without the whole 3-D mesh."""

    elem_nodes: jax.Array
    gradient_sca: jax.Array
    ulevels: jax.Array
    edge_tri: jax.Array
    edge_cross_dxdy: jax.Array
    edges: jax.Array
    nod2D: int = dataclasses.field(metadata={"static": True})
    inv_edges: jax.Array = None     # ops.inverse_map(edges) with gpu.ordered_sums, else None


tree_util.register_dataclass(
    SSHGeom,
    data_fields=["elem_nodes", "gradient_sca", "ulevels", "edge_tri", "edge_cross_dxdy", "edges",
                 "inv_edges"],
    meta_fields=["nod2D"],
)


@dataclasses.dataclass(frozen=True)
class SSHGather:
    """Per-device context of the gathered solve, built inside ``shard_map`` by
    :func:`local_ssh_gather`: the GLOBAL operator and zstar geometry and the owner
    ``(device, lane)`` of every global node (all replicated), and this device's lanes
    (``lane_glob``: the global node of each lane, ``lane_valid``: owned or halo, ``[Lmax]``)."""

    op: SSHOperator
    geom: SSHGeom
    own_dev: jax.Array       # (nod2D,) i4
    own_lane: jax.Array      # (nod2D,) i4
    lane_glob: jax.Array     # (Lmax,) i4
    lane_valid: jax.Array    # (Lmax,) bool
    axis_name: str = dataclasses.field(default="p", metadata={"static": True})


tree_util.register_dataclass(
    SSHGather,
    data_fields=["op", "geom", "own_dev", "own_lane", "lane_glob", "lane_valid"],
    meta_fields=["axis_name"],
)


@dataclasses.dataclass(frozen=True)
class SSHGatherHost:
    """Host-built inputs of the gathered solve (:func:`build_ssh_gather`)."""

    op: SSHOperator          # the global operator, with its preconditioner setting
    geom: SSHGeom
    own_dev: np.ndarray      # (nod2D,) i4
    own_lane: np.ndarray     # (nod2D,) i4
    lane_glob: np.ndarray    # (P, Lmax) i4, 0 on padding lanes
    lane_valid: np.ndarray   # (P, Lmax) bool
    Lmax: int
    npes: int


def build_ssh_gather(op: SSHOperator, mesh: Mesh, partition) -> SSHGatherHost:
    """The gathered solve's host tables for ``partition``. ``op`` is the GLOBAL operator (with
    the Chebyshev setting the run uses); the lanes follow the partition's ``myList_nod2D``
    (owned first, then halo) padded to the sharded State's ``Lmax["nod"]``."""
    from .shard_mesh import local_sizes

    _, Lmax = local_sizes(partition)
    L, P, N = Lmax["nod"], partition.npes, int(mesh.nod2D)
    own_dev = np.full(N, -1, np.int32)
    own_lane = np.full(N, -1, np.int32)
    lane_glob = np.zeros((P, L), np.int32)
    lane_valid = np.zeros((P, L), bool)
    for d in range(P):
        ml = np.asarray(partition.myList_nod2D[d])
        md = int(partition.myDim_nod2D[d])
        own_dev[ml[:md]] = d
        own_lane[ml[:md]] = np.arange(md, dtype=np.int32)
        lane_glob[d, :ml.size] = ml
        lane_valid[d, :ml.size] = True
    if (own_dev < 0).any():
        raise ValueError(f"build_ssh_gather: {int((own_dev < 0).sum())} nodes have no owner")
    geom = SSHGeom(elem_nodes=np.asarray(mesh.elem_nodes), gradient_sca=np.asarray(mesh.gradient_sca),
                   ulevels=np.asarray(mesh.ulevels), edge_tri=np.asarray(mesh.edge_tri),
                   edge_cross_dxdy=np.asarray(mesh.edge_cross_dxdy), edges=np.asarray(mesh.edges),
                   nod2D=N, inv_edges=(ops.inverse_map(np.asarray(mesh.edges), N)
                                       if ops.ordered_sums_enabled() else None))
    return SSHGatherHost(op=op, geom=geom, own_dev=own_dev, own_lane=own_lane,
                         lane_glob=lane_glob, lane_valid=lane_valid, Lmax=L, npes=P)


def gather_inputs(g: SSHGatherHost):
    """``(replicated, lanes)``: the ``shard_map`` inputs of the gathered solve — the replicated
    pytree (``PartitionSpec()``) and the folded ``[P*Lmax]`` lane tables (``PartitionSpec('p')``)."""
    rep = (g.op, g.geom, g.own_dev, g.own_lane)
    lanes = (g.lane_glob.reshape(-1), g.lane_valid.reshape(-1))
    return rep, lanes


def local_ssh_gather(rep, lanes, axis_name: str = "p") -> SSHGather:
    """Inside ``shard_map``: this device's :class:`SSHGather` from :func:`gather_inputs`' slices."""
    op, geom, own_dev, own_lane = rep
    return SSHGather(op=op, geom=geom, own_dev=own_dev, own_lane=own_lane,
                     lane_glob=lanes[0], lane_valid=lanes[1], axis_name=axis_name)


def build_ssh_operator(
    mesh: Mesh, *, dt: float, alpha: float = SSH_ALPHA, theta: float = SSH_THETA,
    precond_variant: int = 0,
) -> SSHOperator:
    """Assemble the static stiffness matrix + MITgcm preconditioner (host numpy).

    Mirrors ``fesom_ssh_stiff_alloc_and_build`` + ``fesom_ssh_preconditioner``
    (``fesom_ssh.c:36-255``). Element-Galerkin: for each edge and each adjacent
    cell ``el(i)``, ``fy[k] = depth·(∂N_k/∂x·dy_i − ∂N_k/∂y·dx_i)`` (negated for
    ``i=1``), scattered with sign ``+`` into row ``edges[ed,0]`` and ``−`` into row
    ``edges[ed,1]`` at columns = the cell's 3 nodes, times ``factor = g·dt·α·θ``.
    The mass term ``areasvol[n,0]/dt`` is added on the diagonal (non-cavity).

    ``precond_variant`` (FESOM2 #995, ``namelist.dyn`` ``precond_variant``): 0 = the formula FESOM2 has used since
    60b46bdc (the default, byte-identical); 1 = the textbook symmetric Jacobi ``M⁻¹ = 2D⁻¹ − D⁻¹AD⁻¹``
    (off-diagonal ``−a_rc/(a_rr·a_cc)``), which upstream selects automatically in a single-precision build.
    """
    if precond_variant not in (0, 1):
        raise ValueError(f"precond_variant must be 0 or 1, got {precond_variant!r}")
    N = mesh.nod2D
    E = mesh.myDim_edge2D
    edges = np.asarray(mesh.edges)[:E]
    edge_tri = np.asarray(mesh.edge_tri)[:E]
    elem_nodes = np.asarray(mesh.elem_nodes)
    gradient_sca = np.asarray(mesh.gradient_sca)
    edge_cross = np.asarray(mesh.edge_cross_dxdy)[:E]
    zbar = np.asarray(mesh.zbar)
    nlevels = np.asarray(mesh.nlevels)
    ulevels_nod = np.asarray(mesh.ulevels_nod2D)
    areasvol = np.asarray(mesh.areasvol)

    factor = float(G) * float(dt) * float(alpha) * float(theta)
    e0, e1 = edges[:, 0], edges[:, 1]

    rows_l, cols_l, vals_l = [], [], []
    for i in (0, 1):
        eli = edge_tri[:, i]
        valid = eli >= 0
        elis = np.where(valid, eli, 0)
        en = elem_nodes[elis]                      # (E,3) the cell's 3 nodes
        g = gradient_sca[elis]                     # (E,6) ∂N/∂x(0:3), ∂N/∂y(3:6)
        depth = zbar[nlevels[elis] - 1] - zbar[0]  # (E,) zbar_bot − zbar_srf (<0)
        dx_i = edge_cross[:, 2 * i + 0]
        dy_i = edge_cross[:, 2 * i + 1]
        fy = depth[:, None] * (g[:, 0:3] * dy_i[:, None] - g[:, 3:6] * dx_i[:, None])
        if i == 1:
            fy = -fy
        fy = np.where(valid[:, None], fy, 0.0) * factor   # (E,3)
        rows_l += [np.repeat(e0, 3), np.repeat(e1, 3)]
        cols_l += [en.ravel(), en.ravel()]
        vals_l += [fy.ravel(), -fy.ravel()]

    # mass term on the diagonal (non-cavity nodes; all of pi)
    top = ulevels_nod - 1
    nocav = ulevels_nod == 1
    mass = np.where(nocav, areasvol[np.arange(N), top] / float(dt), 0.0)
    rows_l.append(np.arange(N))
    cols_l.append(np.arange(N))
    vals_l.append(mass)

    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)

    # Assemble S (sums the duplicate contributions per (row,col)); keep the full
    # topological pattern (do NOT eliminate numeric zeros) so it matches the C
    # neighbour list the preconditioner iterates over.
    S = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    S.sum_duplicates()
    diag = S.diagonal()
    row_of = np.repeat(np.arange(N), np.diff(S.indptr))
    col_of = S.indices
    val_of = S.data

    # MITgcm-style symmetric preconditioner (fesom_ssh.c:239-253):
    #   pr[diag]          = 1 / diag(row)
    #   pr[row,col≠row]   = −0.5·(S[row,col]/diag(row)) / (diag(row)+diag(col))
    dr = diag[row_of]
    dc = diag[col_of]
    is_diag = row_of == col_of
    with np.errstate(divide="ignore", invalid="ignore"):
        if precond_variant == 0:
            pr = np.where(is_diag, 1.0 / dr, -0.5 * (val_of / dr) / (dr + dc))
        else:
            pr = np.where(is_diag, 1.0 / dr, -val_of / (dr * dc))

    return SSHOperator(
        rows=jnp.asarray(row_of.astype(np.int32)),
        cols=jnp.asarray(col_of.astype(np.int32)),
        stiff_vals=jnp.asarray(val_of.astype(np.float64)),
        precond_vals=jnp.asarray(pr.astype(np.float64)),
        diag=jnp.asarray(diag.astype(np.float64)),
        n_nodes=N,
    )


def ssh_matvec(op: SSHOperator, x, halo: "SSHHalo | None" = None):
    """``S @ x`` — the static stiffness matvec (``y[row] = Σ S[row,col]·x[col]``).

    Distributed (``halo`` given, inside ``shard_map``): first **broadcast-exchange**
    ``x`` (refresh the halo node copies from their owners — the C's "exchange ``pp``
    before each SpMV"), then a LOCAL ``segment_sum`` into ``[Lmax_nod]``. Owned rows
    come out COMPLETE because every nonzero column of an owned row is a local node
    (the operator loop-bound, asserted in :func:`partition_ssh_operator`); halo rows
    are incomplete but never trusted (overwritten by the next exchange). ``halo=None``
    is the dense single-device path — byte-identical to ``v1.0``.
    """
    if halo is not None:
        x = _halo_refresh(x, halo)
    # single precision (FESOM2 #994): each row of A·x summed in WP_full; the caller rounds it once (PR.wp).
    return _spmv(op, op.stiff_vals, x)


def ssh_precond(op: SSHOperator, r, halo: "SSHHalo | None" = None):
    """``M⁻¹ @ r`` — the MITgcm symmetric preconditioner matvec (same pattern).

    Distributed: exchange ``r`` first (the C's "exchange ``rr`` after the residual
    update", which is exactly before this preconditioner SpMV), then the local
    ``segment_sum``. ``halo=None`` ⇒ dense (byte-identical)."""
    if halo is not None:
        r = _halo_refresh(r, halo)
    return PR.wp(_spmv(op, op.precond_vals, r))           # rows in WP_full, rounded once (#994)


def _power_lam(A, n: int, power_iters: int) -> float:
    """Fixed-seed power iteration for the largest eigenvalue of the CSR matrix ``A``.

    The norms are numpy's own pairwise sums (``np.sum``), not BLAS ``nrm2``: a threaded BLAS sums
    in an order that depends on the host thread count, which moved ``lam_max`` (and so the Chebyshev
    coefficients baked into the program) by an ulp between nodes with 32 and 64 CPUs. scipy's CSR
    matvec is single-threaded. Same floats on every host."""
    v = np.random.default_rng(0).standard_normal(n)     # FIXED seed ⇒ same floats everywhere
    v /= np.sqrt(np.sum(v * v))
    lam = 1.0
    for _ in range(power_iters):
        w = A @ v
        lam = float(np.sqrt(np.sum(w * w)))
        if lam == 0.0:                                   # degenerate (S=0) — keep lam sane
            lam = 1.0
            break
        v = w / lam
    return lam


def enable_cheb_precond(op: SSHOperator, degree: int, *, kappa_guess: float = 30.0,
                        power_iters: int = 100, safety: float = 1.05) -> SSHOperator:
    """Enable the degree-``degree`` Chebyshev polynomial preconditioner on ``op``
    (the M7 CGPOLY lever, port_kokkos Task E.3) — call on the GLOBAL operator
    before :func:`partition_ssh_operator` (the bounds ride along as static meta).

    Host-only and DETERMINISTIC: ``lam_max`` of the diag-scaled operator
    ``D^{-1/2}·S·D^{-1/2}`` via fixed-seed power iteration (×``safety`` head-room —
    power iteration converges from below), ``lam_min = lam_max/kappa_guess``. The
    ``kappa_guess`` only tunes EFFICIENCY, never correctness: the Chebyshev
    polynomial ``p_k`` is positive on ``(0, lam_max]`` regardless of where the true
    smallest eigenvalue sits, so ``M⁻¹`` stays SPD for any SPD ``S`` whose spectrum
    is ≤ ``lam_max`` (the ×1.05 covers the power-iteration under-estimate)."""
    if degree < 1:
        raise ValueError(f"Chebyshev precond degree must be >= 1, got {degree}")
    rows = np.asarray(op.rows)
    cols = np.asarray(op.cols)
    vals = np.asarray(op.stiff_vals)
    diag = np.asarray(op.diag)
    n = op.n_nodes
    dis = 1.0 / np.sqrt(np.where(diag > 0.0, diag, 1.0))
    A = sp.coo_matrix((vals * dis[rows] * dis[cols], (rows, cols)),
                      shape=(n, n)).tocsr()
    lam_max = _power_lam(A, n, power_iters) * safety
    lam_min = lam_max / kappa_guess
    return dataclasses.replace(op, cheb=(int(degree), float(lam_min), float(lam_max)))


def ssh_precond_cheb(op: SSHOperator, r, halo: "SSHHalo | None" = None):
    """``M⁻¹ @ r`` — the degree-k Chebyshev polynomial preconditioner (CGPOLY):
    ``k`` Chebyshev semi-iterations on the diag-scaled system ``(D⁻¹S)·z = D⁻¹r``
    from ``z₀=0``, i.e. ``M⁻¹ = p_k(D⁻¹S)·D⁻¹ = D^{-1/2}·p_k(D^{-1/2}SD^{-1/2})·D^{-1/2}``
    — symmetric positive definite (``p_k > 0`` on ``(0, lam_max]``).

    Cost: ``k`` SpMV(+halo) and NO dot products — that is the point of the lever:
    the per-CG-iteration collective count stays at 2 psums while the ITERATION count
    drops (the polynomial eats condition number), so the Allreduce-latency share of
    the solve shrinks with the iteration ratio. Payoff regime is many-node
    (Allreduce-latency-bound); at small scale it is roughly halo-neutral (adds k
    SpMV-halos per apply while cutting iterations) — a flat small-scale A/B is
    EXPECTED (port_kokkos E.3 spec). The recurrence is the classical Chebyshev
    iteration (Saad, Alg. 12.1) with STATIC coefficients (``op.cheb`` meta floats),
    so no extra device scalars or collectives enter the graph. Output valid on
    OWNED lanes (halo rows incomplete — the same contract as :func:`ssh_precond`)."""
    degree, lam_min, lam_max = op.cheb
    theta = 0.5 * (lam_max + lam_min)
    delta = 0.5 * (lam_max - lam_min)
    sigma1 = theta / delta
    inv_d = 1.0 / op.diag                 # sharded: pad lanes are 1.0 ⇒ finite
    f = inv_d * r
    rho = 1.0 / sigma1
    d = f / theta
    z = d
    for _ in range(degree):
        res = f - inv_d * PR.wp(ssh_matvec(op, z, halo))   # SpMV refreshes z's halo lanes; rows f64 (D7)
        rho_new = 1.0 / (2.0 * sigma1 - rho)
        d = rho_new * rho * d + (2.0 * rho_new / delta) * res
        z = z + d
        rho = rho_new
    return z


def ssh_precond_for(op: SSHOperator, halo: "SSHHalo | None" = None):
    """The preconditioner apply ``r → M⁻¹r`` selected by ``op.cheb``: the Chebyshev
    polynomial when enabled, else the MITgcm ``M⁻¹`` (byte-identical default)."""
    if op.cheb is not None:
        return lambda r: ssh_precond_cheb(op, r, halo)
    return lambda r: ssh_precond(op, r, halo)


# ==========================================================================
# Substep 8 — SSH RHS (linfs)
# ==========================================================================
def compute_ssh_rhs(mesh: Mesh, uv, uv_rhs, helem, *, alpha: float = SSH_ALPHA,
                    ssh_rhs_old=None, water_flux=None):
    """SSH equation RHS ``[nod2D]`` (substep 8, ``fesom_compute_ssh_rhs_linfs``).

    Per edge, sum over each adjacent cell's layers
    ``c_i = ±alpha·Σ_nz ((v+vr)·dx_i − (u+ur)·dy_i)·helem`` (``+`` for ``el1`` with
    ``(dx1,dy1)``, ``−`` for ``el2`` with ``(dx2,dy2)``), then scatter
    ``ssh_rhs[n1] += c, ssh_rhs[n2] −= c``. ``uv``/``uv_rhs`` are ``[elem2D,nl,2]``,
    ``helem`` ``[elem2D,nl]``. With ``alpha=1`` the ``(1−alpha)·ssh_rhs_old`` term
    (``fesom_ssh.c:325``) is zero; it is added when ``ssh_rhs_old`` is supplied.

    **zstar (Phase 9a, JZ.3).** ``water_flux`` (``[nod2D]``) adds the real-freshwater
    tail (``fesom_ssh.c:413-421``): ``ssh_rhs[n] += −α·wf·areasvol[n,0]`` on non-cavity
    nodes (the cavity arm is unported). ``water_flux=None`` ⇒ the linfs path (no tail),
    byte-identical.
    """
    et = mesh.edge_tri
    el1, el2 = et[:, 0], et[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s = jnp.where(has1, el1, 0)
    el2s = jnp.where(has2, el2, 0)
    cross = mesh.edge_cross_dxdy
    U, V = uv[:, :, 0], uv[:, :, 1]
    Ur, Vr = uv_rhs[:, :, 0], uv_rhs[:, :, 1]
    lm = mesh.elem_layer_mask

    def cterm(els, has, dxcol, dycol, sign):
        dx = cross[:, dxcol : dxcol + 1]                  # (edge,1)
        dy = cross[:, dycol : dycol + 1]
        u = U[els]; v = V[els]; ur = Ur[els]; vr = Vr[els]   # (edge,nl)
        h = helem[els]
        m = lm[els] & has[:, None]
        term = jnp.where(m, ((v + vr) * dx - (u + ur) * dy) * h, 0.0)
        return sign * alpha * term.sum(axis=1)            # (edge,)

    c = cterm(el1s, has1, 0, 1, 1.0) + cterm(el2s, has2, 2, 3, -1.0)
    vals = jnp.stack([c, -c], axis=1)                     # n1 += c, n2 −= c
    if ops.ordered_sums_enabled() and getattr(mesh, "inv_edges", None) is not None:
        ssh_rhs = ops.ordered_sum(vals.reshape(-1), mesh.inv_edges)   # gpu.ordered_sums
    else:
        ssh_rhs = jax.ops.segment_sum(
            vals.reshape(-1), mesh.edges.reshape(-1), num_segments=mesh.nod2D
        )
    # zstar water-flux tail (fesom_ssh.c:413-421): −α·wf·areasvol[n,0] on non-cavity.
    if water_flux is not None:
        nocav = mesh.ulevels_nod2D == 1
        ssh_rhs = ssh_rhs + jnp.where(nocav, -alpha * water_flux * mesh.areasvol[:, 0], 0.0)
    if ssh_rhs_old is not None and alpha != 1.0:
        ssh_rhs = ssh_rhs + (1.0 - alpha) * ssh_rhs_old
    return ssh_rhs


# ==========================================================================
# Substep 9 — preconditioned CG solve
# ==========================================================================
def cg_batch() -> int:
    """``FESOM_CG_BATCH=k`` (``gpu.cg_batch``, read at trace time; default 1): the CG runs k iterations per
    trip of its while loop, the extra ones masked by the stopping test, so the iterate and the iteration
    count are the one-at-a-time loop's while the GPU syncs with the host k times less often."""
    k = int(os.environ.get("FESOM_CG_BATCH", "1") or 1)
    if k < 1:
        raise ValueError(f"FESOM_CG_BATCH must be >= 1, got {k}")
    return k


def _pcg(matvec, precond, b, x0, soltol, maxiter, *, rtol_abs=None,
         reduce=None, reduce_pair=None, n_global=None, return_iters=False):
    """Preconditioned CG, a literal port of ``fesom_ssh_solve_cg``
    (``fesom_ssh.c:384``). Stops at ``‖r‖/√N < soltol·‖b‖/√N`` (relative residual
    ``soltol``). ``x0`` is the initial guess; returns the (possibly early-stopped)
    iterate. Zero-rhs short-circuits to 0, matching the C.

    ``rtol_abs`` overrides the (RMS) stop threshold directly — used for the
    warm-started forward solve, where the C measures the residual against the
    **original** ``‖ssh_rhs‖`` rather than the deflated ``‖b_eff‖`` (see
    :func:`solve_ssh`).

    Distributed CG (Phase 8, S.6): ``reduce`` is a global dot ``(u,v) → Σ_owned u·v``
    then ``psum`` (so every dot/residual is a GLOBAL reduction, device-identical) and
    ``n_global`` is the global node count for the residual RMS. The ``matvec``/
    ``precond`` passed in halo-exchange ``x``/``r`` internally, so the loop body is
    structurally unchanged. Because the residual is a ``psum`` it is identical on all
    devices ⇒ the data-dependent ``while_loop`` trip count is device-identical (no
    deadlock) and — load-bearing — matches the single-device count (review #1/#4).
    ``reduce=None``/``n_global=None`` ⇒ the dense path (plain ``jnp.sum``,
    ``n=b.shape[0]``), byte-identical to ``v1.0``. ``return_iters`` (diagnostic /
    tests) also returns the iteration count; it leaves the default graph untouched.

    ``reduce_pair`` (``(r, z) → (Σ r·z, Σ r·r)``) fuses the loop body's two
    same-point reductions into ONE collective (``reductions.global_dot_pair``) —
    2 of the 3 per-iteration ``psum``s become 1. ``None`` ⇒ two ``dot`` calls,
    keeping the dense graph (and any caller not passing it) untouched."""
    n = b.shape[0] if n_global is None else n_global
    dot = (lambda u, v: jnp.sum(PR.full(u) * PR.full(v))) if reduce is None else reduce
    dot_rz_rr = ((lambda r, z: (dot(r, z), dot(r, r)))
                 if reduce_pair is None else reduce_pair)
    s0 = dot(b, b)
    rtol = (soltol * jnp.sqrt(s0 / n)) if rtol_abs is None else rtol_abs

    def run(_):
        # single precision (#994): the matvec rows come back in WP_full and are rounded once into the WP
        # vectors; dots, scalars and the stopping test stay WP_full (identity in double precision).
        r0 = PR.wp(PR.full(b) - matvec(x0))
        z0 = precond(r0)
        sold0, rr0 = dot_rz_rr(r0, z0)
        resid0 = jnp.sqrt(rr0 / n)

        def cond(c):
            _x, _r, _z, _p, _s, resid, it = c
            return (resid >= rtol) & (it < maxiter)

        def body(c):
            x, r, z, p, s_old, _resid, it = c
            Ap = PR.wp(matvec(p))
            al = s_old / dot(p, Ap)
            x = PR.wp(PR.full(x) + al * PR.full(p))
            r = PR.wp(PR.full(r) - al * PR.full(Ap))
            z = precond(r)
            sp0, sp1 = dot_rz_rr(r, z)
            resid = jnp.sqrt(sp1 / n)
            be = sp0 / s_old
            p = PR.wp(PR.full(z) + be * PR.full(p))
            return (x, r, z, p, sp0, resid, it + 1)

        k = cg_batch()
        if k > 1:
            # k iterations per trip of the while loop, each after the first masked by the stopping
            # test: the iterate is the one-at-a-time loop's (the masked steps are discarded by
            # `where`), but the host reads the loop predicate once per k iterations
            body1 = body

            def body(c):
                c = body1(c)
                for _ in range(k - 1):
                    live = cond(c)
                    c = jax.tree.map(lambda new, old: jnp.where(live, new, old), body1(c), c)
                return c

        init = (x0, r0, z0, z0, sold0, resid0, jnp.array(0))
        final = lax.while_loop(cond, body, init)
        # DIAGNOSTIC (env-gated, inert by default): report the PCG trip count + whether it
        # STOPPED ON maxiter rather than on the tolerance.  Added while chasing the ng5 x
        # P=128 cliff: the loop is data-dependent, so a configuration that stops converging
        # and runs to MAXITER (=500, vs the ~127 typical) multiplies every per-iteration cost
        # — which is invisible to every static/structural check.  `hit_cap=1` is the smoking
        # gun and ALSO a correctness warning: the returned eta is then NOT solved to soltol.
        if os.environ.get("FESOM_CG_ITERS"):
            # float32/int32 operands: a 64-bit host callback is not allowed in a single-precision (x64-off) trace
            jax.debug.print(
                "[cg-iters] it={it} hit_cap={cap} resid={r:.3e} rtol={t:.3e}",
                it=final[6].astype(jnp.int32), cap=(final[6] >= maxiter).astype(jnp.int32),
                r=final[5].astype(jnp.float32), t=jnp.asarray(rtol).astype(jnp.float32), ordered=False)
        return final if return_iters else final[0]

    if return_iters:
        zero_carry = (jnp.zeros_like(x0), jnp.zeros_like(b), jnp.zeros_like(b),
                      jnp.zeros_like(b), jnp.zeros((), s0.dtype),
                      jnp.zeros((), s0.dtype), jnp.array(0))
        out = lax.cond(s0 > 0.0, run, lambda _: zero_carry, operand=None)
        return out[0], out[6]

    # Zero rhs → d_eta = 0 (fesom_ssh.c:414), and avoids 0/0 in the recurrence.
    return lax.cond(s0 > 0.0, run, lambda _: jnp.zeros_like(x0), operand=None)


def stiff_increment_matvec(mesh: Mesh, hbar, x, *, dt: float = DT_DEFAULT,
                           alpha: float = SSH_ALPHA, theta: float = SSH_THETA,
                           halo: "SSHHalo | None" = None):
    """``ΔA·x`` — the zstar per-step SSH-stiffness increment (D2), recomputed each step
    from the start-of-step ``hbar`` rather than carried as the C's cumulative CSR.

    The C ``fesom_update_stiff_mat_ale`` (``fesom_ssh.c:238-296``) adds, per step, the
    **base edge-assembly** with the static column depth replaced by ``−dhe`` — the
    elemental ``mean₃(hbar−hbar_old)`` (``fesom_step.c:216-227``). Telescoping over steps
    at cold start (``hbar_init=0``) gives ``Σ dhe ≡ mean₃(hbar)``, so the live matrix is
    ``A_base + ΔA`` with ``ΔA = assembly(depth = −mean₃(hbar_el))`` and the SAME
    ``factor = g·dt·α·θ``. ΔA is the identical antisymmetric edge→node scatter as
    :func:`compute_ssh_rhs`, with the "velocity" replaced by the **element gradient of
    the iterate** ``x`` (``∂ₓx = Σ_k ∂N_k/∂x·x[node_k]``), so it is linear in ``x`` and
    **symmetric** (the base operator's symmetry is preserved ⇒
    ``custom_linear_solve(symmetric=True)`` still holds). At cold start ``hbar=0 ⇒ ΔA=0``
    (the step-1 no-op). Cavity elements (``ulevels>1``) contribute 0 (the C's dhe guard).

    Distributed (``halo`` given): exchange ``x`` first (like :func:`ssh_matvec`); the
    element gradients + edge scatter then run on the local mesh. ``halo=None`` is the
    dense single-device path (validated at JZ.3; the sharded zstar solve is JZ.7)."""
    if halo is not None:
        x = _halo_refresh(x, halo)
    # single precision (D1, user 2026-10-20): the increment is rebuilt from the WP hbar with WP_full
    # arithmetic, and its rows are summed in WP_full with the base rows (the caller rounds once).
    x, hbar = PR.full(x), PR.full(hbar)
    en = mesh.elem_nodes
    g = PR.full(mesh.gradient_sca)                         # (elem2D,6): ∂N/∂x(0:3),∂N/∂y(3:6)
    xe = x[en]                                             # (elem2D,3)
    gx = jnp.sum(g[:, 0:3] * xe, axis=1)                   # ∂x/∂x per element
    gy = jnp.sum(g[:, 3:6] * xe, axis=1)                   # ∂x/∂y per element
    # increment depth = −mean₃(hbar) (the telescoped Σdhe); cavity element ⇒ 0.
    dhe = (hbar[en[:, 0]] + hbar[en[:, 1]] + hbar[en[:, 2]]) / 3.0
    depth = jnp.where(mesh.ulevels == 1, -dhe, 0.0)
    factor = float(G) * float(dt) * float(alpha) * float(theta)

    et = mesh.edge_tri
    el1, el2 = et[:, 0], et[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s = jnp.where(has1, el1, 0)
    el2s = jnp.where(has2, el2, 0)
    cross = PR.full(mesh.edge_cross_dxdy)

    def cterm(els, has, dxcol, dycol, sign):
        # base flux (fesom_ssh.c:157), k-summed against x: depth·(∂ₓx·dy_i − ∂ᵧx·dx_i)
        f = depth[els] * (gx[els] * cross[:, dycol] - gy[els] * cross[:, dxcol])
        return sign * jnp.where(has, f, 0.0)

    c = cterm(el1s, has1, 0, 1, 1.0) + cterm(el2s, has2, 2, 3, -1.0)
    vals = jnp.stack([c, -c], axis=1) * factor            # n1 += c·factor, n2 −= c·factor
    if ops.ordered_sums_enabled() and getattr(mesh, "inv_edges", None) is not None:
        return ops.ordered_sum(vals.reshape(-1), mesh.inv_edges)      # gpu.ordered_sums
    return jax.ops.segment_sum(
        vals.reshape(-1), mesh.edges.reshape(-1), num_segments=mesh.nod2D)


def _ssh_solver_parts(op: SSHOperator, mesh, halo, cfg, hbar):
    """The pieces every SSH solve shares: the live operator ``A(hbar)`` as a matvec, the
    preconditioner and the (global, under ``halo``) reductions. ``cfg`` is the static
    ``(dt, alpha, theta, forward_tol, grad_tol, maxiter)``."""
    dt, alpha, theta = cfg[0], cfg[1], cfg[2]
    if hbar is None:
        matvec = lambda x: ssh_matvec(op, x, halo)    # noqa: E731
    elif op.inc_el is not None and halo is None:
        # the increment's entries once per solve, then one matvec on (S + ΔA)
        live = dataclasses.replace(op, stiff_vals=PR.full(op.stiff_vals) + _increment_values(
            op, mesh, hbar, float(G) * float(dt) * float(alpha) * float(theta)))
        matvec = lambda x: ssh_matvec(live, x)        # noqa: E731
    else:
        def matvec(x):
            return ssh_matvec(op, x, halo) + stiff_increment_matvec(
                mesh, hbar, x, dt=dt, alpha=alpha, theta=theta, halo=halo)
    # MITgcm M⁻¹ by default; the Chebyshev polynomial (CGPOLY) when op.cheb is set —
    # the stop criterion measures the UNpreconditioned residual either way, so the
    # two preconds solve to the same tolerance (only the iteration count changes).
    precond = ssh_precond_for(op, halo)
    if halo is None:
        reduce_fn, reduce_pair_fn, n_global = None, None, None
    else:
        reduce_fn = lambda u, v: global_dot(u, v, halo.owned_mask, halo.axis_name)  # noqa: E731
        # the (r·z, r·r) pair in ONE psum — same locals, half the loop's collectives
        reduce_pair_fn = lambda r, z: global_dot_pair(                             # noqa: E731
            r, z, r, r, halo.owned_mask, halo.axis_name)
        n_global = halo.n_global
    return matvec, precond, reduce_fn, reduce_pair_fn, n_global


def _ssh_forward_solve(cfg, op, mesh, halo, b_eff, hbar, rtol_fwd):
    """The early-stopped forward PCG from zero on ``b_eff`` (stop at the absolute RMS
    residual ``rtol_fwd``) — the dump-matching iterate."""
    matvec, precond, reduce_fn, reduce_pair_fn, n_global = _ssh_solver_parts(
        op, mesh, halo, cfg, hbar)
    return _pcg(matvec, precond, b_eff, jnp.zeros_like(b_eff), cfg[3], cfg[5],
                rtol_abs=rtol_fwd, reduce=reduce_fn, reduce_pair=reduce_pair_fn,
                n_global=n_global)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _ssh_implicit(cfg, op, mesh, halo, b_eff, hbar, rtol_fwd):
    """``δ ≈ A(hbar)⁻¹·b_eff`` by the early-stopped PCG, with the implicit derivative.

    Value: :func:`_ssh_forward_solve` (unchanged arithmetic ⇒ forward bitwise). Tangent:
    ``dδ = A⁻¹·(d b_eff − dA·δ)`` with ``A⁻¹`` a TIGHT PCG (relative tolerance
    ``grad_tol``, from zero) in both the tangent and its transpose, so the tangent-linear
    model and the adjoint are exact transposes. The forward's stop threshold is an ABSOLUTE
    number derived from the primal right-hand side; reusing that solve for a tangent would
    stop a small tangent after zero iterations. ``rtol_fwd`` is therefore an argument whose
    tangent is ignored (the caller also ``stop_gradient``-s it).

    Only the static Python ``cfg`` ``(dt, alpha, theta, forward_tol, grad_tol, maxiter)`` is
    a non-differentiated argument. Every array — operator, mesh, halo maps included — is an
    explicit argument: the rule is traced lazily, possibly after the caller's trace has
    finished (e.g. ``jax.jvp`` of an already-jitted step), and an array it closed over (or
    received through ``nondiff_argnums``) would then be a stale tracer of that trace. The
    operator, mesh and halo are treated as constants: their tangents are ignored.

    The value is named ``"ssh_delta"`` (:func:`jax.ad_checkpoint.checkpoint_name`) here and
    in the rule, so the output a differentiated program sees carries the name."""
    return checkpoint_name(_ssh_forward_solve(cfg, op, mesh, halo, b_eff, hbar, rtol_fwd),
                           "ssh_delta")


def _ssh_implicit_jvp(cfg, primals, tangents):
    op, mesh, halo, b_eff, hbar, rtol_fwd = primals
    _dop, _dmesh, _dhalo, db, dhbar, _drtol = tangents
    # The name sits HERE (and in the primal function), not on the caller's copy of the
    # output. Under differentiation this rule computes the primal itself, its δ IS the
    # solve's output, and under zstar the tangent needs δ once more (dA·δ below). Named only
    # at the caller, a policy saving "ssh_delta" kept the caller's copy and still re-ran the
    # forward CG to rebuild this δ in every reverse step; named in both places, it stored
    # the same [nod2D] vector twice.
    delta = checkpoint_name(_ssh_forward_solve(cfg, op, mesh, halo, b_eff, hbar, rtol_fwd),
                            "ssh_delta")
    rhs = jnp.zeros_like(b_eff) if isinstance(db, SymbolicZero) else db
    if hbar is not None and not isinstance(dhbar, SymbolicZero):
        # A(hbar) = A_base + ΔA(hbar), ΔA bilinear in (hbar, x) ⇒ dA·δ = ΔA(dhbar)·δ.
        rhs = rhs - stiff_increment_matvec(mesh, dhbar, delta, dt=cfg[0], alpha=cfg[1],
                                           theta=cfg[2], halo=halo)
    matvec, precond, reduce_fn, reduce_pair_fn, n_global = _ssh_solver_parts(
        op, mesh, halo, cfg, hbar)

    def tight(mv, b):
        x = _pcg(mv, precond, b, jnp.zeros_like(b), cfg[4], cfg[5],
                 reduce=reduce_fn, reduce_pair=reduce_pair_fn, n_global=n_global)
        if halo is None:
            return x
        # Sharded: the PCG leaves incomplete partial sums on halo lanes (the local
        # operator's halo rows). Harmless in a forward value (the caller's next exchange
        # overwrites them), but in reverse mode the caller's INPUT exchange transposes to
        # a scatter-add of every halo-lane cotangent onto its owner. Returning zero off the
        # owned lanes makes the derivative solve E·S⁻¹·Eᵀ in lane space (E = owned-lane
        # embedding): symmetric, so tangent and transpose stay exact transposes, and no
        # halo or padding value reaches an owner.
        return jnp.where(halo.owned_mask, x, jnp.zeros_like(x))

    ddelta = jax.lax.custom_linear_solve(matvec, rhs, tight, tight, symmetric=True)
    return delta, ddelta


_ssh_implicit.defjvp(_ssh_implicit_jvp, symbolic_zeros=True)


def solve_ssh(op: SSHOperator, ssh_rhs, *, x0=None, halo: "SSHHalo | None" = None,
              mesh: "Mesh | None" = None, hbar=None, dt: float = DT_DEFAULT,
              alpha: float = SSH_ALPHA, theta: float = SSH_THETA,
              forward_tol: float = SOLTOL, grad_tol: float = GRAD_SOLTOL,
              maxiter: int = MAXITER):
    """Solve ``S·d_eta = ssh_rhs`` (substep 9), returning ``d_eta`` ``[nod2D]``.

    Forward: PCG to the loose ``forward_tol`` (``=soltol`` ⇒ matches the dump's
    early-stopped iterate). Derivatives: the implicit derivative of the solve
    (:func:`_ssh_implicit`, a :func:`jax.custom_jvp`): the tangent is
    ``A⁻¹·(d b − dA·δ)`` with a *tight* solve (``grad_tol``, relative, from zero), and
    reverse mode transposes that tangent with the same tight solve. So
    ``d(d_eta)/d(ssh_rhs)`` is the accurate ``S⁻¹`` regardless of the loose forward stop,
    and the tangent-linear model and the adjoint pass a dot test to round-off at any
    tangent amplitude. The derivative is that of the exact solution of ``A δ = b_eff``,
    taken at the early-stopped iterate.

    The correction ``δ`` is named ``"ssh_delta"`` (:func:`jax.ad_checkpoint.checkpoint_name`,
    inside :func:`_ssh_implicit` and its rule), so a remat policy
    ``save_only_these_names("ssh_delta")`` can keep it (one ``[nod2D]`` per step) instead of
    re-running the forward CG in the reverse sweep.

    ``x0`` is the warm-start guess (the C reuses the previous step's ``d_eta``;
    ``None`` ⇒ 0, the step-1 case). It is ``stop_gradient``-ed (the solution is
    mathematically independent of ``x0``) and folded into the rhs so the inner
    solve is a *linear* map of its argument.

    Distributed (Phase 8, S.6): pass a :class:`SSHHalo` (with ``op`` a per-device
    :func:`partition_ssh_operator` slice) and call inside ``shard_map``. The matvec/
    precond then broadcast-exchange their input, the CG dots become global
    (``psum``) reductions, and the residual RMS divides by the global node count —
    so the early-stop iteration count is device-identical to single device (the
    load-bearing determinism). ``halo=None`` is the dense single-device path. The
    derivative solves return zero on halo and padding lanes (the forward leaves partial
    sums there, which the caller's next exchange overwrites); this keeps the lane-space
    derivative symmetric, so the transpose of the caller's input exchange adds nothing
    to the owners.

    zstar (Phase 9a, JZ.3): pass ``mesh`` + ``hbar`` (the start-of-step ``st.hbar``) to
    add the D2 stiffness increment ``ΔA(mean₃(hbar))`` (:func:`stiff_increment_matvec`)
    to the matvec — the operator becomes a differentiable function of state. ``hbar=None``
    (linfs) ⇒ the static operator, no increment traced. At cold start (``hbar=0``) the
    increment is 0, so the step-1 solve is the linfs solve exactly."""
    n = op.n_nodes
    x0 = jnp.zeros((n,), ssh_rhs.dtype) if x0 is None else x0
    x0 = lax.stop_gradient(x0)
    cfg = (float(dt), float(alpha), float(theta), forward_tol, grad_tol, int(maxiter))
    matvec, _precond, reduce_fn, _pair, n_global = _ssh_solver_parts(op, mesh, halo, cfg,
                                                                     hbar)

    # Solve for the correction δ = S⁻¹·b_eff from δ0=0 (linear), then d_eta = x0+δ.
    # For x0=0 (step 1) b_eff = ssh_rhs and this is exactly the C's solve-from-zero.
    b_eff = PR.wp(PR.full(ssh_rhs) - matvec(x0))

    # ⚠️ Warm-start fidelity (step ≥2): the C measures the CG residual against the
    # ORIGINAL ‖ssh_rhs‖, not the deflated ‖b_eff‖. Because the inner residual
    # ``b_eff − A·δ_k`` equals the full residual ``ssh_rhs − A·(x0+δ_k)``, stopping
    # the inner solve at ``soltol·‖ssh_rhs‖`` replicates the C's warm-started
    # early-stop exactly (a good warm start ⇒ b_eff already below threshold ⇒ 0
    # iters ⇒ d_eta = x0). Deriving rtol from ‖b_eff‖ instead would over-converge.
    # The threshold is a property of the forward iteration only: no derivative flows
    # through it.
    if halo is None:
        nrhs = ssh_rhs.shape[0]
        rtol_fwd = forward_tol * jnp.sqrt(jnp.sum(PR.full(ssh_rhs) * PR.full(ssh_rhs)) / nrhs)
    else:
        rtol_fwd = forward_tol * jnp.sqrt(reduce_fn(ssh_rhs, ssh_rhs) / n_global)
    rtol_fwd = lax.stop_gradient(rtol_fwd)

    delta = _ssh_implicit(cfg, op, mesh, halo, b_eff, hbar, rtol_fwd)   # named "ssh_delta"
    return x0 + delta


def solve_ssh_gathered(g: SSHGather, ssh_rhs, x0, hbar=None, *, dt: float = DT_DEFAULT,
                       alpha: float = SSH_ALPHA, theta: float = SSH_THETA,
                       forward_tol: float = SOLTOL, grad_tol: float = GRAD_SOLTOL,
                       maxiter: int = MAXITER):
    """:func:`solve_ssh` with the 2-D system gathered onto every device (inside ``shard_map``).

    ``ssh_rhs``, ``x0`` and ``hbar`` (zstar; ``None`` under linfs) are this device's
    ``[Lmax]`` lanes. ONE ``all_gather`` brings their OWNED lanes to every device in global node
    order, the dense single-device solve runs on the global operator (its implicit-derivative
    rule included), and each device keeps its valid lanes (owned and halo; padding lanes 0).
    The value is the dense solve's on the same global inputs, bit for bit. Derivatives: the
    gather transposes to a reduce-scatter onto the owned lanes, so halo and padding lanes get
    no cotangent."""
    parts = [ssh_rhs, x0] + ([] if hbar is None else [hbar])
    allv = lax.all_gather(jnp.stack(parts), g.axis_name)          # (P, k, Lmax)
    glob = allv[g.own_dev, :, g.own_lane]                         # (nod2D, k), global order
    d = solve_ssh(g.op, glob[:, 0], x0=glob[:, 1], halo=None,
                  mesh=None if hbar is None else g.geom,
                  hbar=None if hbar is None else glob[:, 2], dt=dt, alpha=alpha,
                  theta=theta, forward_tol=forward_tol, grad_tol=grad_tol, maxiter=maxiter)
    return jnp.where(g.lane_valid, d[g.lane_glob], jnp.zeros((), d.dtype))


# ==========================================================================
# Substeps 11–12 — hbar (transport divergence) + eta_n blend
# ==========================================================================
def compute_hbar(mesh: Mesh, uv, helem, hbar, *, dt: float = DT_DEFAULT,
                 water_flux=None):
    """Transport-divergence → ``ssh_rhs_old``, then the hbar update (substep 11).

    Mirror of ``fesom_compute_hbar`` (``fesom_momentum.c:779``; Fortran
    ``oce_ale.F90:2005-2102``). Returns ``(ssh_rhs_old, hbar_new)``:

    * ``ssh_rhs_old`` ``[nod2D]`` — the antisymmetric edge→node transport
      divergence of the **new** (post-:func:`~fesom_jax.momentum.update_vel`)
      velocity ``uv`` with the static thickness ``helem``, ``alpha=1`` and **no**
      AB-velocity (``u+ur → u``). This is exactly :func:`compute_ssh_rhs` with
      ``uv_rhs=0``. It is saved as the next step's AB history: linfs reads it via
      the ``(1−alpha)·ssh_rhs_old`` term of the next ``compute_ssh_rhs`` — which
      vanishes at ``alpha=1`` but is still computed (fidelity + the field is
      dumped at substep 11's siblings).
    * ``hbar_new = hbar + ssh_rhs_old·dt / areasvol[n, top]`` on non-cavity nodes
      (all of pi; ``top = ulevels_nod2D−1 = 0``); cavity nodes keep ``hbar``. The
      input ``hbar`` *is* ``hbar_old`` (the caller saves it before overwriting).

    The division by the large CV area (``~1e9–1e12 m²``) strongly suppresses the
    absolute error of the near-cancelling ``ssh_rhs_old`` (see the ssh/rhs
    lesson), so ``hbar`` matches the dump tightly in absolute terms.

    **zstar (Phase 9a, JZ.3).** ``water_flux`` subtracts the real-freshwater term
    (``fesom_momentum.c:839-846``): ``ssh_rhs_old[n] −= wf·areasvol[n,0]`` on non-cavity,
    **before** the hbar update — so the wf-modified ``ssh_rhs_old`` is BOTH what hbar
    consumes and what the next step reads via the ``(1−α)·ssh_rhs_old`` term. The inner
    transport divergence is the bare ``uv`` (no wf); the tail is added here, not in the
    inner :func:`compute_ssh_rhs`. ``water_flux=None`` ⇒ linfs (byte-identical)."""
    ssh_rhs_old = compute_ssh_rhs(mesh, uv, jnp.zeros_like(uv), helem, alpha=1.0)
    if water_flux is not None:
        nocav0 = mesh.ulevels_nod2D == 1
        ssh_rhs_old = ssh_rhs_old + jnp.where(
            nocav0, -water_flux * mesh.areasvol[:, 0], 0.0)
    top = mesh.ulevels_nod2D - 1                          # 0 for pi (non-cavity)
    area = jnp.take_along_axis(mesh.areasvol, top[:, None], axis=1)[:, 0]
    nocav = mesh.ulevels_nod2D == 1
    safe_area = jnp.where(nocav, area, 1.0)               # AD-safe (no 0/0 at cavity)
    hbar_new = jnp.where(nocav, hbar + ssh_rhs_old * dt / safe_area, hbar)
    return ssh_rhs_old, hbar_new


def eta_n_update(mesh: Mesh, eta_n_prev, hbar, hbar_old, *, alpha: float = SSH_ALPHA):
    """SSH elevation blend (substep 12): ``eta_n = α·hbar + (1−α)·hbar_old`` on
    non-cavity nodes; cavity nodes keep ``eta_n_prev``. With ``α=1`` this is just
    ``eta_n = hbar``. Mirror of the inline blend at ``fesom_step.c:257-268``
    (Fortran ``oce_ale.F90:3771-3775``)."""
    nocav = mesh.ulevels_nod2D == 1
    blend = alpha * hbar + (1.0 - alpha) * hbar_old
    return jnp.where(nocav, blend, eta_n_prev)
