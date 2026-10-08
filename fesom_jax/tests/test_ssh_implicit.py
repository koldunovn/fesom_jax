"""Tangent-linear exactness of the SSH solve (pi mesh, linfs).

The forward SSH solve is a PCG stopped early at an ABSOLUTE residual threshold derived from
the primal right-hand side (``soltol·‖ssh_rhs‖``, see :func:`fesom_jax.ssh.solve_ssh`). A
derivative rule that reuses that forward solve for the tangent stops a small tangent after
zero iterations, so the tangent-linear model and the adjoint disagree. The dot test of one
model step at amplitudes 1, 1e-4 and 1e-8 exposes it; the reference solve reconstructed in
``_ssh_implicit_common`` is the negative control that must fail it.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import forcing, ic, ssh
from fesom_jax.config import MAXITER
from fesom_jax import step as stepmod
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.tests import _ssh_implicit_common as C

DT = 100.0
AMPS = (1.0, 1e-4, 1e-8)
DOT_TOL = 1e-10


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    return load_mesh()


@pytest.fixture(scope="module")
def op(mesh):
    return ssh.build_ssh_operator(mesh, dt=DT)


@pytest.fixture(scope="module")
def stress(mesh):
    return forcing.surface_stress(mesh)


def _step(first=False):
    def f(st, mesh, op, stress):
        return stepmod.step(st, mesh, op, stress, Params.defaults(), dt=DT,
                            is_first_step=first)
    return f


@pytest.fixture(scope="module")
def st2(mesh, op, stress):
    """A NON-REST state after two steps from a perturbed IC: d_eta ≠ 0, so the third
    solve is warm-started and its forward stop threshold is set by a non-zero ‖ssh_rhs‖
    (from rest, ssh_rhs = 0 gives a zero threshold and every solve runs to maxiter)."""
    st = ic.initial_state(mesh)
    st = dataclasses.replace(st, T=st.T + C.node3d(mesh, 0.5, 0),
                             uv=st.uv + C.elem_uv(mesh, 0.02, 0))
    st = jax.jit(_step(first=True))(st, mesh, op, stress)
    st = jax.jit(_step())(st, mesh, op, stress)
    assert float(jnp.max(jnp.abs(st.d_eta))) > 0.0
    return st


def _directions(mesh):
    x = {"T": C.node3d(mesh, 1.0, 1), "S": C.node3d(mesh, 0.1, 0),
         "uv": C.elem_uv(mesh, 0.02, 1), "eta_n": C.node2d(mesh, 0.05, 0),
         "hbar": C.node2d(mesh, 0.05, 1)}
    y = {"T": C.node3d(mesh, 1.0, 0), "S": C.node3d(mesh, 10.0, 1),
         "uv": C.elem_uv(mesh, 50.0, 0), "eta_n": C.node2d(mesh, 20.0, 1),
         "hbar": C.node2d(mesh, 20.0, 0), "d_eta": C.node2d(mesh, 100.0, 1)}
    return x, y


def test_dot_test_step_linfs(mesh, op, stress, st2, capsys):
    """One pi step (linfs): tangent-linear and adjoint agree to ≤ 1e-10 at every
    amplitude."""
    x, y = _directions(mesh)
    rows = C.step_dot_test(_step(), st2, x, y, AMPS, consts=(mesh, op, stress))
    with capsys.disabled():
        print("\n" + C.format_rows("pi linfs, solve_ssh", rows))
    for a, _lhs, _rhs, rel, _reln in rows:
        assert rel <= DOT_TOL, f"dot test amp={a:.0e}: rel {rel:.2e} > {DOT_TOL:.0e}"


def test_dot_test_negative_control_reference_solve(mesh, op, stress, st2, monkeypatch,
                                                   capsys):
    """Negative control: with the reference solve (tangent through the early-stopped
    forward solve) the same dot test FAILS at amplitude 1e-8."""
    monkeypatch.setattr(ssh, "solve_ssh", C.solve_ssh_reference)
    x, y = _directions(mesh)
    rows = C.step_dot_test(_step(), st2, x, y, AMPS, consts=(mesh, op, stress))
    with capsys.disabled():
        print("\n" + C.format_rows("pi linfs, reference solve", rows))
    rel_small = rows[-1][3]
    assert rel_small > 1e-7, (f"the reference solve passed the dot test at 1e-8 "
                              f"(rel {rel_small:.2e}): the test cannot see the defect")


# --------------------------------------------------------------------------
# compatibility with the reference solve: forward bitwise, reverse unchanged
# --------------------------------------------------------------------------
def test_step_forward_bitwise_vs_reference(mesh, op, stress, st2, monkeypatch):
    """The value of a full pi step is bit-identical with the reference solve (CPU)."""
    new = jax.jit(_step())(st2, mesh, op, stress)
    monkeypatch.setattr(ssh, "solve_ssh", C.solve_ssh_reference)
    ref = jax.jit(_step())(st2, mesh, op, stress)
    for f in dataclasses.fields(new):
        a, b = np.asarray(getattr(new, f.name)), np.asarray(getattr(ref, f.name))
        assert np.array_equal(a, b), f"State.{f.name} differs from the reference solve"


def test_step_reverse_gradient_equals_reference(mesh, op, stress, st2, monkeypatch):
    """The one-step reverse gradient equals the reference path's to ≤ 1e-12 relative
    (both transpose the same tight solve; only the tangent solve changed)."""
    x, y = _directions(mesh)
    fin = tuple(sorted(x))

    def grads():
        def loss(xin, st, cs):
            out = _step()(dataclasses.replace(st, **xin), *cs)
            return sum(jnp.vdot(y[k], getattr(out, k)) for k in sorted(y))
        g = jax.jit(jax.grad(loss))({k: getattr(st2, k) for k in fin}, st2,
                                    (mesh, op, stress))
        return {k: np.asarray(v) for k, v in g.items()}

    g_new = grads()
    monkeypatch.setattr(ssh, "solve_ssh", C.solve_ssh_reference)
    g_ref = grads()
    for k in fin:
        den = np.linalg.norm(g_ref[k])
        assert den > 0.0, f"zero reference gradient for {k}: the test is not live"
        rel = np.linalg.norm(g_new[k] - g_ref[k]) / den
        assert rel <= 1e-12, f"d loss / d {k}: rel {rel:.2e} vs the reference path"


# --------------------------------------------------------------------------
# solve-level tests (pi operator; the zstar increment is a function of hbar only)
# --------------------------------------------------------------------------
HBAR_AMP = 10.0


@pytest.fixture(scope="module")
def solve_inputs(mesh):
    rhs = C.node2d(mesh, 1.0e3, 0)
    hbar = C.node2d(mesh, HBAR_AMP, 1)
    x0 = ssh.solve_ssh(ssh.build_ssh_operator(mesh, dt=DT), rhs * 1.01)
    return rhs, hbar, x0


def test_solve_forward_bitwise_vs_reference_zstar(mesh, op, solve_inputs):
    """Solve level, zstar operator (hbar ≠ 0), warm-started: value bit-identical."""
    rhs, hbar, x0 = solve_inputs
    new = jax.jit(lambda r, h, x: ssh.solve_ssh(op, r, x0=x, mesh=mesh, hbar=h, dt=DT))(
        rhs, hbar, x0)
    ref = jax.jit(lambda r, h, x: C.solve_ssh_reference(op, r, x0=x, mesh=mesh, hbar=h,
                                                        dt=DT))(rhs, hbar, x0)
    assert np.array_equal(np.asarray(new), np.asarray(ref))


def test_increment_bilinear_direct_form(mesh, solve_inputs):
    """``jvp`` of ``stiff_increment_matvec`` w.r.t. hbar equals the increment evaluated
    at the tangent (bilinear in (hbar, x)) — the rule's direct ``dA·δ``."""
    rhs, hbar, x0 = solve_inputs
    dh = C.node2d(mesh, 1.0, 0)
    delta = x0
    _, jv = jax.jvp(lambda h: ssh.stiff_increment_matvec(mesh, h, delta, dt=DT),
                    (hbar,), (dh,))
    direct = ssh.stiff_increment_matvec(mesh, dh, delta, dt=DT)
    rel = float(jnp.linalg.norm(jv - direct) / jnp.linalg.norm(jv))
    assert float(jnp.linalg.norm(jv)) > 0.0
    assert rel <= 1e-14, f"increment is not bilinear to round-off: rel {rel:.2e}"


def test_hbar_tangent_vs_fd(mesh, op, solve_inputs, capsys):
    """zstar operator term: JVP w.r.t. hbar vs central FD of a tightly converged solve
    (``forward_tol=1e-14``; the default early stop would mix its own error into FD),
    over an h-sweep. The plateau must agree to ≤ 1e-6."""
    rhs, hbar, x0 = solve_inputs
    w = C.node2d(mesh, 1.0, 0)
    dh = C.node2d(mesh, HBAR_AMP, 0)
    J = jax.jit(lambda h: jnp.vdot(w, ssh.solve_ssh(op, rhs, x0=x0, mesh=mesh, hbar=h,
                                                    dt=DT, forward_tol=1e-14)))
    _, ad = jax.jvp(J, (hbar,), (dh,))
    ad = float(ad)
    rows = []
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        fd = float((J(hbar + h * dh) - J(hbar - h * dh)) / (2.0 * h))
        rows.append((h, fd, abs(fd - ad) / abs(ad)))
    with capsys.disabled():
        print(f"\n  d<w,δ>/d hbar · dh: AD = {ad:+.12e}")
        for h, fd, rel in rows:
            print(f"    h={h:.0e}  FD={fd:+.12e}  rel={rel:.2e}")
    assert ad != 0.0
    assert min(r for _, _, r in rows) <= 1e-6


def test_tangent_matches_early_stopped_default_forward(mesh, op, solve_inputs):
    """The tangent of the DEFAULT (early-stopped) solve equals the tangent of a tightly
    converged solve up to the early-stop error of δ in the dA·δ term: the tangent no
    longer depends on the forward tolerance through its own stop."""
    rhs, hbar, x0 = solve_inputs
    dr, dh = C.node2d(mesh, 1.0e3, 1), C.node2d(mesh, HBAR_AMP, 0)
    t_def = jax.jvp(lambda r, h: ssh.solve_ssh(op, r, x0=x0, mesh=mesh, hbar=h, dt=DT),
                    (rhs, hbar), (dr, dh))[1]
    t_tight = jax.jvp(lambda r, h: ssh.solve_ssh(op, r, x0=x0, mesh=mesh, hbar=h, dt=DT,
                                                 forward_tol=1e-14),
                      (rhs, hbar), (dr, dh))[1]
    rel = float(jnp.linalg.norm(t_def - t_tight) / jnp.linalg.norm(t_tight))
    assert rel <= 1e-4, f"tangent depends on the forward tolerance: rel {rel:.2e}"


def test_zero_rhs(mesh, op):
    """ssh_rhs = 0: δ = 0 exactly, and the tangent is finite and equals S⁻¹·d(rhs)."""
    z = jnp.zeros((mesh.nod2D,), jnp.float64)
    dr = C.node2d(mesh, 1.0, 0)
    val, tan = jax.jvp(lambda r: ssh.solve_ssh(op, r), (z,), (dr,))
    assert np.array_equal(np.asarray(val), np.zeros(mesh.nod2D))
    assert np.all(np.isfinite(np.asarray(tan)))
    ref = ssh.solve_ssh(op, dr, forward_tol=1e-14)
    rel = float(jnp.linalg.norm(tan - ref) / jnp.linalg.norm(ref))
    assert rel <= 1e-11, f"zero-rhs tangent ≠ S⁻¹ dr: rel {rel:.2e}"
    g = jax.grad(lambda r: jnp.vdot(dr, ssh.solve_ssh(op, r)))(z)
    assert np.all(np.isfinite(np.asarray(g)))


def test_hbar_zero_equals_linfs(mesh, op, solve_inputs):
    """Cold start (hbar = 0): value bit-identical to the linfs solve, tangent w.r.t. the
    right-hand side equal to round-off."""
    rhs, _hbar, x0 = solve_inputs
    z = jnp.zeros((mesh.nod2D,), jnp.float64)
    dr = C.node2d(mesh, 1.0e3, 1)
    v_lin, t_lin = jax.jvp(lambda r: ssh.solve_ssh(op, r, x0=x0, dt=DT), (rhs,), (dr,))
    v_z, t_z = jax.jvp(lambda r: ssh.solve_ssh(op, r, x0=x0, mesh=mesh, hbar=z, dt=DT),
                       (rhs,), (dr,))
    assert np.array_equal(np.asarray(v_lin), np.asarray(v_z))
    rel = float(jnp.linalg.norm(t_lin - t_z) / jnp.linalg.norm(t_lin))
    assert rel <= 1e-14, f"hbar=0 tangent ≠ linfs tangent: rel {rel:.2e}"


def test_tangent_independent_of_warm_start(mesh, op, solve_inputs):
    """The tangent does not depend on the warm start x0: linfs to 1e-12; zstar only up
    to the early-stop error of the iterate that enters dA·δ (≈ soltol·‖dA‖)."""
    rhs, hbar, x0 = solve_inputs
    z = jnp.zeros((mesh.nod2D,), jnp.float64)
    dr, dh = C.node2d(mesh, 1.0e3, 1), C.node2d(mesh, HBAR_AMP, 0)
    lin = [jax.jvp(lambda r: ssh.solve_ssh(op, r, x0=x, dt=DT), (rhs,), (dr,))[1]
           for x in (z, x0)]
    rel = float(jnp.linalg.norm(lin[0] - lin[1]) / jnp.linalg.norm(lin[0]))
    assert rel <= 1e-12, f"linfs tangent depends on x0: rel {rel:.2e}"
    zs = [jax.jvp(lambda r, h: ssh.solve_ssh(op, r, x0=x, mesh=mesh, hbar=h, dt=DT),
                  (rhs, hbar), (dr, dh))[1] for x in (z, x0)]
    rel_z = float(jnp.linalg.norm(zs[0] - zs[1]) / jnp.linalg.norm(zs[0]))
    assert rel_z <= 1e-4, f"zstar tangent depends on x0 beyond the early stop: {rel_z:.2e}"


def test_tight_solves_stop_before_maxiter(mesh, op, stress, st2, monkeypatch, capfd):
    """Every CG of the one-step tangent and adjoint programs (forward, tight tangent,
    tight transpose) stops on its tolerance, not on MAXITER: an unconverged tight solve
    would look like a transpose defect in the dot test."""
    monkeypatch.setenv("FESOM_CG_ITERS", "1")
    x, y = _directions(mesh)
    C.step_dot_test(_step(), st2, x, y, (1.0,), consts=(mesh, op, stress))
    lines = [ln for ln in capfd.readouterr().out.splitlines() if "[cg-iters]" in ln]
    its = C.parse_cg_iters(lines)
    with capfd.disabled():
        print(f"\n  pi CG solves in the one-step TL + ADJ programs (it, rtol): {its}")
    assert len(its) >= 3, f"expected ≥ 3 CG solves (fwd, tangent, transpose), got {its}"
    assert all(it < MAXITER for it, _ in its), f"a CG hit MAXITER={MAXITER}: {its}"


def test_differentiate_an_already_jitted_solve(mesh, op, solve_inputs):
    """``jax.jvp`` / ``jax.grad`` applied OUTSIDE a jitted solve whose operator, mesh and
    hbar are jit arguments: the rule is then traced after the jit trace has finished, so
    any array it closed over would be a stale tracer ("No constant handler for type:
    DynamicJaxprTracer" at lowering). Tangent and gradient equal the inside-jit ones."""
    rhs, hbar, x0 = solve_inputs
    dr = C.node2d(mesh, 1.0e3, 1)
    w = C.node2d(mesh, 1.0, 0)
    F = jax.jit(lambda r, o, m, h, x: ssh.solve_ssh(o, r, x0=x, mesh=m, hbar=h, dt=DT))
    t_out = jax.jvp(lambda r: F(r, op, mesh, hbar, x0), (rhs,), (dr,))[1]
    t_in = jax.jit(lambda r, d, o, m, h, x: jax.jvp(
        lambda rr: ssh.solve_ssh(o, rr, x0=x, mesh=m, hbar=h, dt=DT), (r,), (d,))[1])(
        rhs, dr, op, mesh, hbar, x0)
    assert np.array_equal(np.asarray(t_out), np.asarray(t_in))
    g_out = jax.grad(lambda h: jnp.vdot(w, F(rhs, op, mesh, h, x0)))(hbar)
    g_in = jax.jit(jax.grad(lambda h, o, m: jnp.vdot(
        w, ssh.solve_ssh(o, rhs, x0=x0, mesh=m, hbar=h, dt=DT))))(hbar, op, mesh)
    assert float(jnp.linalg.norm(g_in)) > 0.0
    rel = float(jnp.linalg.norm(g_out - g_in) / jnp.linalg.norm(g_in))
    assert rel <= 1e-13, f"outside-jit gradient differs: rel {rel:.2e}"
