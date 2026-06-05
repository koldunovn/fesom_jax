"""Task 2.7 gate — SSH RHS (substep 8) + CG solve (substep 9), the AD-critical solver.

* **Dump gate (step 1):** ``ssh_rhs`` (substep 8) and ``d_eta`` (substep 9) at all 5
  node probes vs the C dump. Step 1 is NOT trivial here: ``uv=0`` but
  ``uv_rhs=du`` (the wind-forced increment from substep 7) drives ``ssh_rhs``, which
  drives the CG solve. The dumped ``d_eta`` is the C's **early-stopped** (≈3-iter,
  ``soltol=1e-5``) PCG iterate, so we replicate the C PCG exactly (static stiffness
  ``S`` + MITgcm preconditioner) and match it to ~1e-18.
* **Synthetic test:** step-1 ``uv=0`` leaves the ``(u+ur)`` velocity part of
  ``compute_ssh_rhs`` dormant, so the full edge scatter (both ``el1``/``el2``
  branches, nonzero ``uv``) is checked against an independent loop-based numpy
  reference of ``fesom_ssh.c:282-320``.
* **Operator / preconditioner:** ``S`` is symmetric; the MITgcm preconditioner has
  off-diagonal terms and is **load-bearing** (a Jacobi/diagonal variant gives a
  different early-stopped ``d_eta`` that fails the dump).
* **AD gate:** ``d(d_eta)/d(ssh_rhs)`` from ``custom_linear_solve`` equals the exact
  ``S⁻¹`` (vs an independent tight solve + finite differences), is finite, and flows
  back through ``compute_ssh_rhs`` to the upstream velocity increment.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import eos, forcing, ic, momentum, pgf, pp, ssh, verify
from fesom_jax.io_dump import find_record
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh

NODE_PROBES = [1001, 1500, 2000, 2500, 3000]
DT = 100.0

# ssh_rhs is a transport divergence with heavy cancellation (the wind-forced
# convergence is a small residual of large opposing edge fluxes ~1e4). Its abs
# floor (~5e-9 at the cancellation probe 1500) is set by the upstream `du`
# fidelity (~1e-12 rel) amplified by dx·helem ~1e7 — NOT the ssh_rhs scatter
# itself. Calibrated atol with margin.
SSH_RHS_ATOL = 1e-7


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


@pytest.fixture(scope="module")
def chain(mesh):
    """Step-1 momentum chain through substep 7, then ssh_rhs (8) + operator.

    Returns ``(st, du, ssh_rhs, op)`` where ``du`` is the substep-7 velocity
    increment that lands in ``uv_rhs`` (what substep 8 reads), ``ssh_rhs`` the
    substep-8 field, and ``op`` the static linfs stiffness operator."""
    st = ic.initial_state(mesh)
    _, hp, bv = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode)
    px, py = pgf.pressure_force_linfs(mesh, hp)
    _, Av, _ = pp.mixing_pp(mesh, st.uv, bv)
    uvr, _ = momentum.compute_vel_rhs(mesh, st.uv, st.uv_rhsAB, st.eta_n, px, py,
                                      st.w_e, st.hnode, is_first_step=True, dt=DT)
    uvr = momentum.visc_filt_bidiff(mesh, st.uv, uvr, dt=DT)
    du = momentum.impl_vert_visc(mesh, st.uv, uvr, Av, forcing.surface_stress(mesh),
                                 dt=DT)
    ssh_rhs = ssh.compute_ssh_rhs(mesh, st.uv, du, st.helem)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    return st, du, ssh_rhs, op


# --------------------------------------------------------------------------
# Independent loop-based numpy reference of fesom_compute_ssh_rhs_linfs.
# --------------------------------------------------------------------------
def _ssh_rhs_ref(mesh, uv, uv_rhs, helem, alpha=1.0):
    edges = np.asarray(mesh.edges)
    etri = np.asarray(mesh.edge_tri)
    cross = np.asarray(mesh.edge_cross_dxdy)
    ule, nle = np.asarray(mesh.ulevels), np.asarray(mesh.nlevels)
    uv, ur, h = np.asarray(uv), np.asarray(uv_rhs), np.asarray(helem)
    rhs = np.zeros(mesh.nod2D)
    for ed in range(mesh.myDim_edge2D):
        n1, n2 = edges[ed]
        el1, el2 = etri[ed]
        c = 0.0
        if el1 >= 0:
            dx1, dy1 = cross[ed, 0], cross[ed, 1]
            for nz in range(ule[el1] - 1, nle[el1] - 1):
                u, v = uv[el1, nz]; uu, vv = ur[el1, nz]
                c += alpha * ((v + vv) * dx1 - (u + uu) * dy1) * h[el1, nz]
        if el2 >= 0:
            dx2, dy2 = cross[ed, 2], cross[ed, 3]
            for nz in range(ule[el2] - 1, nle[el2] - 1):
                u, v = uv[el2, nz]; uu, vv = ur[el2, nz]
                c -= alpha * ((v + vv) * dx2 - (u + uu) * dy2) * h[el2, nz]
        rhs[n1] += c
        rhs[n2] -= c
    return rhs


def _synthetic_uv(mesh):
    e = np.arange(mesh.elem2D)[:, None]
    k = np.arange(mesh.nl)[None, :]
    uv = np.stack([0.1 * np.cos(0.3 * k + 0.01 * e),
                   0.05 * np.sin(0.2 * k - 0.013 * e)], axis=-1)
    uvr = np.stack([1e-3 * np.cos(0.05 * k + 0.002 * e),
                    1e-3 * np.sin(0.07 * k) * np.ones_like(e)], axis=-1)
    m = mesh.elem_layer_mask[..., None]
    return (jnp.where(m, jnp.asarray(uv), 0.0), jnp.where(m, jnp.asarray(uvr), 0.0))


# --------------------------------------------------------------------------
# 1. substep 8 — ssh_rhs dump gate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gid", NODE_PROBES)
def test_ssh_rhs_matches_dump_step1(load_dump, mesh, chain, gid):
    _, _, ssh_rhs, _ = chain
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=8,
                      field="ssh_rhs", probe_gid=gid)
    verify.assert_close(np.asarray(ssh_rhs)[gid - 1:gid], rec, kind="scatter",
                        atol=SSH_RHS_ATOL)


def test_ssh_rhs_nonzero_and_wind_driven(mesh, chain):
    """At step 1 uv=0 but uv_rhs=du (wind-forced) → ssh_rhs is genuinely nonzero,
    so the substep-8 dump gate is meaningful (not a trivial zero match)."""
    _, _, ssh_rhs, _ = chain
    assert float(jnp.max(jnp.abs(ssh_rhs))) > 1.0


def test_ssh_rhs_synthetic_matches_reference(mesh, chain):
    """Nonzero uv exercises the dormant (u+ur) velocity part + both el1/el2
    branches of the edge scatter, vs an independent numpy loop reference."""
    st, *_ = chain
    uv, uvr = _synthetic_uv(mesh)
    out = np.asarray(ssh.compute_ssh_rhs(mesh, uv, uvr, st.helem))
    ref = _ssh_rhs_ref(mesh, uv, uvr, st.helem)
    # pure scatter reassociation (same inputs) over ~1e6-magnitude summands →
    # abs floor ~6e-8 on a 1.7e8-magnitude field (rel ~3.5e-16)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-6)
    assert np.max(np.abs(out)) > 0  # genuinely exercised


# --------------------------------------------------------------------------
# 2. substep 9 — d_eta dump gate + solver structure
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gid", NODE_PROBES)
def test_d_eta_matches_dump_step1(load_dump, mesh, chain, gid):
    _, _, ssh_rhs, op = chain
    d_eta = np.asarray(ssh.solve_ssh(op, ssh_rhs))
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=9,
                      field="d_eta", probe_gid=gid)
    verify.assert_close(d_eta[gid - 1:gid], rec, kind="scatter")


def test_forward_solve_residual_below_soltol(mesh, chain):
    """The early-stopped forward solve satisfies the C's relative-residual stop
    (‖S·d_eta − b‖ / ‖b‖ < soltol = 1e-5) — i.e. it stops at the right place."""
    _, _, ssh_rhs, op = chain
    d_eta = ssh.solve_ssh(op, ssh_rhs)
    res = ssh.ssh_matvec(op, d_eta) - ssh_rhs
    rel = float(jnp.linalg.norm(res) / jnp.linalg.norm(ssh_rhs))
    assert rel < 1e-5


def test_operator_symmetric(mesh, chain):
    """S is symmetric: ⟨y, Sx⟩ == ⟨x, Sy⟩ (required for CG + custom_linear_solve)."""
    *_, op = chain
    rng = np.random.RandomState(3)
    x = jnp.asarray(rng.randn(mesh.nod2D))
    y = jnp.asarray(rng.randn(mesh.nod2D))
    lhs = float(jnp.sum(y * ssh.ssh_matvec(op, x)))
    rhs = float(jnp.sum(x * ssh.ssh_matvec(op, y)))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs))


def test_preconditioner_is_mitgcm_and_load_bearing(load_dump, mesh, chain):
    """The MITgcm preconditioner has off-diagonal terms (not Jacobi) and is
    load-bearing: a diagonal-only variant gives a different early-stopped d_eta
    that fails the dump (wrong preconditioner ⇒ wrong Krylov path)."""
    _, _, ssh_rhs, op = chain
    rows, cols = np.asarray(op.rows), np.asarray(op.cols)
    off = rows != cols
    pr = np.asarray(op.precond_vals)
    assert off.sum() > 0 and np.max(np.abs(pr[off])) > 0  # genuinely off-diagonal

    d_mit = np.asarray(ssh.solve_ssh(op, ssh_rhs))
    op_jac = dataclasses.replace(op, precond_vals=jnp.where(jnp.asarray(off), 0.0,
                                                            op.precond_vals))
    d_jac = np.asarray(ssh.solve_ssh(op_jac, ssh_rhs))
    cv = find_record(load_dump("pi_cdump.00000"), step=1, substep=9,
                     field="d_eta", probe_gid=1001).values[0]
    assert abs(d_mit[1000] - cv) < 1e-12    # MITgcm matches the dump
    assert abs(d_jac[1000] - cv) > 1e-11    # Jacobi does NOT → preconditioner matters


# --------------------------------------------------------------------------
# 3. AD gate — implicit-diff gradient through custom_linear_solve
# --------------------------------------------------------------------------
def test_grad_equals_exact_inverse(mesh, chain):
    """d(Σ w·d_eta)/d(ssh_rhs) from custom_linear_solve == S⁻¹·w (S symmetric),
    vs an independent tight solve. Validates the implicit-diff cotangent."""
    _, _, ssh_rhs, op = chain
    w = jnp.asarray(np.random.RandomState(1).randn(mesh.nod2D))

    def loss(b):
        return jnp.sum(w * ssh.solve_ssh(op, b))

    g_ad = np.asarray(jax.grad(loss)(ssh_rhs))
    u_ref = np.asarray(ssh.solve_ssh(op, w, forward_tol=1e-14))  # tight S⁻¹ w
    assert np.all(np.isfinite(g_ad))
    scale = float(np.max(np.abs(u_ref)))
    assert np.max(np.abs(g_ad - u_ref)) <= 1e-9 * scale


def test_grad_finite_difference(mesh, chain):
    """AD vs central finite differences on a tight-forward loss (so the FD
    derivative is the meaningful S⁻¹, matching the implicit-diff AD)."""
    _, _, ssh_rhs, op = chain
    w = jnp.asarray(np.random.RandomState(2).randn(mesh.nod2D))

    def loss(b):
        return jnp.sum(w * ssh.solve_ssh(op, b, forward_tol=1e-14))

    g_ad = np.asarray(jax.grad(loss)(ssh_rhs))
    assert np.all(np.isfinite(g_ad))
    h = 1e-3 * float(np.max(np.abs(np.asarray(ssh_rhs))))
    for j in (0, 1500, mesh.nod2D - 1):
        gfd = float((loss(ssh_rhs.at[j].add(h)) - loss(ssh_rhs.at[j].add(-h))) / (2 * h))
        assert abs(g_ad[j] - gfd) <= 1e-7 * max(abs(gfd), 1.0) + 1e-12, \
            f"comp {j}: AD {g_ad[j]:.6e} vs FD {gfd:.6e}"


def test_grad_flows_to_upstream_increment(mesh, chain):
    """Gradient flows through both substeps: d(Σ d_eta)/d(du) is finite & nonzero
    — exercising compute_ssh_rhs → custom_linear_solve end-to-end."""
    st, du, _, op = chain

    def loss(du_in):
        rhs = ssh.compute_ssh_rhs(mesh, st.uv, du_in, st.helem)
        return jnp.sum(ssh.solve_ssh(op, rhs))

    g = np.asarray(jax.grad(loss)(du))
    assert np.all(np.isfinite(g))
    assert np.max(np.abs(g)) > 0
