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
  same ``x0``, same ``‖r‖<soltol·‖b‖`` stop) for the forward value, and wrap it in
  :func:`jax.lax.custom_linear_solve` so the *gradient* is the clean implicit-diff
  ``S⁻¹`` (the transpose solve converges tightly). Forward = dump-matching
  early-stop; backward = exact-solve implicit gradient. The residual trajectory
  ``[65, 1.0, 0.015]`` vs ``rtol=0.197`` makes the 3-iteration stop unambiguous, so
  ``segment_sum`` reassociation cannot change the iteration count.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import lax, tree_util

from .config import DT_DEFAULT, G, MAXITER, SOLTOL, SSH_ALPHA, SSH_THETA
from .mesh import Mesh

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


tree_util.register_dataclass(
    SSHOperator,
    data_fields=["rows", "cols", "stiff_vals", "precond_vals", "diag"],
    meta_fields=["n_nodes"],
)


def build_ssh_operator(
    mesh: Mesh, *, dt: float, alpha: float = SSH_ALPHA, theta: float = SSH_THETA
) -> SSHOperator:
    """Assemble the static stiffness matrix + MITgcm preconditioner (host numpy).

    Mirrors ``fesom_ssh_stiff_alloc_and_build`` + ``fesom_ssh_preconditioner``
    (``fesom_ssh.c:36-255``). Element-Galerkin: for each edge and each adjacent
    cell ``el(i)``, ``fy[k] = depth·(∂N_k/∂x·dy_i − ∂N_k/∂y·dx_i)`` (negated for
    ``i=1``), scattered with sign ``+`` into row ``edges[ed,0]`` and ``−`` into row
    ``edges[ed,1]`` at columns = the cell's 3 nodes, times ``factor = g·dt·α·θ``.
    The mass term ``areasvol[n,0]/dt`` is added on the diagonal (non-cavity).
    """
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
        pr = np.where(is_diag, 1.0 / dr, -0.5 * (val_of / dr) / (dr + dc))

    return SSHOperator(
        rows=jnp.asarray(row_of.astype(np.int32)),
        cols=jnp.asarray(col_of.astype(np.int32)),
        stiff_vals=jnp.asarray(val_of.astype(np.float64)),
        precond_vals=jnp.asarray(pr.astype(np.float64)),
        diag=jnp.asarray(diag.astype(np.float64)),
        n_nodes=N,
    )


def ssh_matvec(op: SSHOperator, x):
    """``S @ x`` — the static stiffness matvec (``y[row] = Σ S[row,col]·x[col]``)."""
    return jax.ops.segment_sum(
        op.stiff_vals * x[op.cols], op.rows, num_segments=op.n_nodes
    )


def ssh_precond(op: SSHOperator, r):
    """``M⁻¹ @ r`` — the MITgcm symmetric preconditioner matvec (same pattern)."""
    return jax.ops.segment_sum(
        op.precond_vals * r[op.cols], op.rows, num_segments=op.n_nodes
    )


# ==========================================================================
# Substep 8 — SSH RHS (linfs)
# ==========================================================================
def compute_ssh_rhs(mesh: Mesh, uv, uv_rhs, helem, *, alpha: float = SSH_ALPHA,
                    ssh_rhs_old=None):
    """SSH equation RHS ``[nod2D]`` (substep 8, ``fesom_compute_ssh_rhs_linfs``).

    Per edge, sum over each adjacent cell's layers
    ``c_i = ±alpha·Σ_nz ((v+vr)·dx_i − (u+ur)·dy_i)·helem`` (``+`` for ``el1`` with
    ``(dx1,dy1)``, ``−`` for ``el2`` with ``(dx2,dy2)``), then scatter
    ``ssh_rhs[n1] += c, ssh_rhs[n2] −= c``. ``uv``/``uv_rhs`` are ``[elem2D,nl,2]``,
    ``helem`` ``[elem2D,nl]``. With ``alpha=1`` the ``(1−alpha)·ssh_rhs_old`` term
    (``fesom_ssh.c:325``) is zero; it is added when ``ssh_rhs_old`` is supplied.
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
    ssh_rhs = jax.ops.segment_sum(
        vals.reshape(-1), mesh.edges.reshape(-1), num_segments=mesh.nod2D
    )
    if ssh_rhs_old is not None and alpha != 1.0:
        ssh_rhs = ssh_rhs + (1.0 - alpha) * ssh_rhs_old
    return ssh_rhs


# ==========================================================================
# Substep 9 — preconditioned CG solve
# ==========================================================================
def _pcg(matvec, precond, b, x0, soltol, maxiter):
    """Preconditioned CG, a literal port of ``fesom_ssh_solve_cg``
    (``fesom_ssh.c:384``). Stops at ``‖r‖/√N < soltol·‖b‖/√N`` (relative residual
    ``soltol``). ``x0`` is the initial guess; returns the (possibly early-stopped)
    iterate. Zero-rhs short-circuits to 0, matching the C."""
    n = b.shape[0]
    s0 = jnp.sum(b * b)
    rtol = soltol * jnp.sqrt(s0 / n)

    def run(_):
        r0 = b - matvec(x0)
        z0 = precond(r0)
        resid0 = jnp.sqrt(jnp.sum(r0 * r0) / n)
        sold0 = jnp.sum(r0 * z0)

        def cond(c):
            _x, _r, _z, _p, _s, resid, it = c
            return (resid >= rtol) & (it < maxiter)

        def body(c):
            x, r, z, p, s_old, _resid, it = c
            Ap = matvec(p)
            al = s_old / jnp.sum(p * Ap)
            x = x + al * p
            r = r - al * Ap
            z = precond(r)
            sp0 = jnp.sum(r * z)
            sp1 = jnp.sum(r * r)
            resid = jnp.sqrt(sp1 / n)
            be = sp0 / s_old
            p = z + be * p
            return (x, r, z, p, sp0, resid, it + 1)

        init = (x0, r0, z0, z0, sold0, resid0, jnp.array(0))
        x = lax.while_loop(cond, body, init)[0]
        return x

    # Zero rhs → d_eta = 0 (fesom_ssh.c:414), and avoids 0/0 in the recurrence.
    return lax.cond(s0 > 0.0, run, lambda _: jnp.zeros_like(x0), operand=None)


def solve_ssh(op: SSHOperator, ssh_rhs, *, x0=None, forward_tol: float = SOLTOL,
              grad_tol: float = GRAD_SOLTOL, maxiter: int = MAXITER):
    """Solve ``S·d_eta = ssh_rhs`` (substep 9), returning ``d_eta`` ``[nod2D]``.

    Forward: PCG to the loose ``forward_tol`` (``=soltol`` ⇒ matches the dump's
    early-stopped iterate). Gradient: :func:`jax.lax.custom_linear_solve` provides
    the implicit-diff cotangent ``S⁻¹·x̄`` via a *tight* transpose solve
    (``grad_tol``), so ``d(d_eta)/d(ssh_rhs)`` is the accurate ``S⁻¹`` regardless of
    the loose forward stop.

    ``x0`` is the warm-start guess (the C reuses the previous step's ``d_eta``;
    ``None`` ⇒ 0, the step-1 case). It is ``stop_gradient``-ed (the solution is
    mathematically independent of ``x0``) and folded into the rhs so the inner
    solve is a *linear* map of its argument — required by ``custom_linear_solve``.
    """
    n = op.n_nodes
    x0 = jnp.zeros((n,), ssh_rhs.dtype) if x0 is None else x0
    x0 = lax.stop_gradient(x0)

    matvec = lambda x: ssh_matvec(op, x)        # noqa: E731
    precond = lambda r: ssh_precond(op, r)      # noqa: E731

    # Solve for the correction δ = S⁻¹·b_eff from δ0=0 (linear), then d_eta = x0+δ.
    # For x0=0 (step 1) b_eff = ssh_rhs and this is exactly the C's solve-from-zero.
    b_eff = ssh_rhs - matvec(x0)

    def solve(mv, b):           # forward: early-stopped (dump-matching)
        return _pcg(mv, precond, b, jnp.zeros_like(b), forward_tol, maxiter)

    def transpose_solve(mv, b):  # reverse cotangent: tight (accurate gradient)
        return _pcg(mv, precond, b, jnp.zeros_like(b), grad_tol, maxiter)

    delta = jax.lax.custom_linear_solve(
        matvec, b_eff, solve, transpose_solve, symmetric=True
    )
    return x0 + delta


# ==========================================================================
# Substeps 11–12 — hbar (transport divergence) + eta_n blend
# ==========================================================================
def compute_hbar(mesh: Mesh, uv, helem, hbar, *, dt: float = DT_DEFAULT):
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
    lesson), so ``hbar`` matches the dump tightly in absolute terms."""
    ssh_rhs_old = compute_ssh_rhs(mesh, uv, jnp.zeros_like(uv), helem, alpha=1.0)
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
