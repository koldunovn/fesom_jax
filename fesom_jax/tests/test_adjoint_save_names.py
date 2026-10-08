"""``save_names``: a remat policy that keeps the SSH solution (pi mesh, linfs and zstar).

The per-step ``jax.checkpoint`` of :func:`fesom_jax.adjoint.scan_steps` and
:func:`fesom_jax.integrate.integrate` re-runs the whole step in the reverse sweep, including the
early-stopped forward CG of the SSH solve (40-130 iterations on CORE2). ``save_names=
("ssh_delta",)`` keeps the solve's output instead. The gates:

* the empty tuple traces the identical program (jaxpr text);
* the reverse program loses one forward CG per checkpointed step body, and a misspelled name
  removes nothing (the negative control);
* exactly one extra ``[nod2D]`` float64 residual per step, from the SSH solve;
* the gradient moves by less than rematerialisation itself moves it.

linfs AND zstar, because under zstar the derivative rule itself needs the solution (dA·δ): with
the name only on the caller's copy of the solve output, the policy kept that copy and still
re-ran the CG to rebuild the rule's δ (see ``_ssh_implicit_jvp`` in ``fesom_jax/ssh.py``).

⚠️ **The gradient is not bitwise equal with and without the policy, and cannot be.** The two are
different compiled programs: without the policy the reverse program recomputes the early-stopped
forward CG in its own fusion, whose round-off differs from the forward program's, and the early
stop (an absolute residual threshold) turns a round-off difference into an iterate difference
of up to the solver tolerance. The same mechanism already separates rematerialised from
non-rematerialised gradients. Measured on pi, 8 steps (job 27735534): policy vs no policy
1.4e-15 (linfs) and 3.6e-8 (zstar) relative through ``scan_steps``; remat vs no remat 6.2e-10
and 2.1e-6. The gate is therefore "the policy moves the gradient by less than remat on/off
does", with the numbers printed.

A separate file from ``test_adjoint_chunked.py`` on purpose: these tests compile a dozen large
reverse programs, and the XLA CPU JIT's memory-map count grows within a test module until the
module ends (``conftest.py``); in one module with the chunked-sweep tests it aborted a run
(job 27735194).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import forcing, ic, ssh
from fesom_jax.adjoint import scan_steps
from fesom_jax.ale import AleConfig
from fesom_jax.config import A_VER
from fesom_jax.integrate import integrate
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.step import step

DT = 100.0
N = 8               # √8 ⇒ 3 segments of 2 + a tail of 2: both scan bodies are exercised
KVER = 1e-4


@pytest.fixture(scope="module")
def fixture():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    mesh = load_mesh()
    return mesh, ssh.build_ssh_operator(mesh, dt=DT), forcing.surface_stress(mesh), \
        ic.initial_state(mesh)


def _params():
    return Params(k_ver=jnp.asarray(KVER, jnp.float64), a_ver=jnp.asarray(A_VER, jnp.float64))


def _ale(mode):
    return AleConfig() if mode == "zstar" else None


def _xs():
    return jnp.asarray(1.0 + 0.05 * np.cos(np.arange(N)), jnp.float64)


def _one_step(fixture, p, ale):
    mesh, op, stress, _ = fixture

    def one(s, x):
        return step(s, mesh, op, stress * x, p, dt=DT, is_first_step=False, ale_cfg=ale)
    return one


_S1: dict = {}


def _first_step(fixture, mode):
    """One jitted first step from the pi IC: a non-rest start for the one-step programs."""
    if mode not in _S1:
        mesh, op, stress, st0 = fixture
        ale, p = _ale(mode), _params()
        _S1[mode] = jax.jit(lambda s: step(s, mesh, op, stress, p, dt=DT, is_first_step=True,
                                           ale_cfg=ale))(st0)
    return _S1[mode]


def _grad_program(f, x):
    """The jaxpr text of the forward AND reverse program of ``f`` at ``x``, traced and never
    executed (eager differentiation compiles hundreds of small executables). The forward part
    is the same with and without a policy, so a difference in the count of a primitive is a
    difference in the reverse part."""
    ct = jax.tree.map(lambda a: jnp.ones(a.shape, a.dtype), jax.eval_shape(f, x))
    return str(jax.make_jaxpr(lambda x_: jax.vjp(f, x_)[1](ct))(x))


def _residuals(f, *args):
    """``print_saved_residuals`` lines of ``f`` (public API; one line per saved value)."""
    from jax.ad_checkpoint import print_saved_residuals
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_saved_residuals(f, *args)
    return [ln for ln in buf.getvalue().splitlines() if ln.strip()]


def _flat(tree):
    return np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree.leaves(tree)
                           if jnp.issubdtype(x.dtype, jnp.floating)])


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_empty_save_names_traces_the_identical_program(fixture):
    """``save_names=()`` is exactly today's call, through both entry points and both of
    integrate's loop shapes."""
    mesh, op, stress, st0 = fixture
    p = _params()
    one = _one_step(fixture, p, None)
    xs = _xs()
    a = str(jax.make_jaxpr(lambda s, x: scan_steps(one, s, x))(st0, xs))
    b = str(jax.make_jaxpr(lambda s, x: scan_steps(one, s, x, save_names=()))(st0, xs))
    assert a == b, "scan_steps(save_names=()) traces a different program"
    for seg in (0, -1):
        a = str(jax.make_jaxpr(lambda s: integrate(s, mesh, op, stress, N, p, dt=DT,
                                                   remat_segments=seg))(st0))
        b = str(jax.make_jaxpr(lambda s: integrate(s, mesh, op, stress, N, p, dt=DT,
                                                   remat_segments=seg, save_names=()))(st0))
        assert a == b, f"integrate(save_names=()) traces a different program (segments={seg})"


@pytest.mark.parametrize("mode", ["linfs", "zstar"])
def test_save_names_removes_the_forward_cg_from_the_reverse_program(fixture, mode, capsys):
    """Count the CG ``while`` loops in the program text.

    One checkpointed step, forward + reverse: the forward CG, the tight derivative solve
    printed twice (``custom_linear_solve`` carries its solve and its transpose-solve
    sub-programs) and, without a policy, the reverse re-run of the forward CG — 4 loops in the
    text, 3 with the policy (measured on pi, linfs and zstar). Through ``scan_steps`` at 8
    steps (3 segments of 2 + a tail of 2) each of the two per-step scan bodies loses its
    forward CG. A misspelled name removes nothing."""
    p, ale, xs = _params(), _ale(mode), _xs()
    s1 = _first_step(fixture, mode)
    one = _one_step(fixture, p, ale)

    def single(names):
        pol = jax.checkpoint_policies.save_only_these_names(*names) if names else None
        f = jax.checkpoint(lambda s: one(s, 1.0), policy=pol)
        return _grad_program(f, s1).count("while[")

    def scanned(names):
        return _grad_program(lambda s: scan_steps(one, s, xs, save_names=names),
                             s1).count("while[")

    n0, n1, nx = single(()), single(("ssh_delta",)), single(("ssh_dleta",))
    m0, m1, mx = scanned(()), scanned(("ssh_delta",)), scanned(("ssh_dleta",))
    print(f"\n{mode}: while loops in the program text — one step: none {n0} / ssh_delta {n1} "
          f"/ misspelled {nx};  scan_steps({N}): none {m0} / ssh_delta {m1} / misspelled {mx}")
    assert n1 == n0 - 1, (n0, n1)
    assert m1 == m0 - 2, (m0, m1)
    assert nx == n0 and mx == m0, "a misspelled name must keep nothing"


@pytest.mark.parametrize("mode", ["linfs", "zstar"])
def test_save_names_saves_one_nod2d_vector_per_step(fixture, mode, capsys):
    """Residuals: exactly one extra float64 ``[nod2D]`` per step, from the SSH solve.

    ``saved_residuals`` labels the saved value "output of reduce_precision" (JAX inserts one
    after every remat residual) and, under ``lax.scan``, the stacked value "output of scan" —
    not with the name. So the source is checked by file on a single non-scanned checkpoint, and
    the scan by shape (one extra ``[n, nod2D]`` for integrate's per-step path, n = steps - 1).
    With the name both in the rule and on the caller's output, zstar saved the vector twice."""
    mesh, op, stress, st0 = fixture
    p, ale = _params(), _ale(mode)
    s1 = _first_step(fixture, mode)
    one = _one_step(fixture, p, ale)
    nod = mesh.nod2D

    def single(names):
        pol = jax.checkpoint_policies.save_only_these_names(*names) if names else None
        return _residuals(jax.checkpoint(lambda s: one(s, 1.0), policy=pol), s1)

    r0, r1, rx = single(()), single(("ssh_delta",)), single(("ssh_dleta",))
    extra = [ln for ln in r1 if ln not in r0]
    print(f"\n{mode}: one step, residuals none {len(r0)} / ssh_delta {len(r1)} / "
          f"misspelled {len(rx)}; extra: {extra}")
    assert rx == r0, "a misspelled name must save nothing"
    assert len(r1) == len(r0) + 1, (len(r0), len(r1))
    assert len(extra) == 1 and extra[0].startswith(f"f64[{nod}]") and "ssh.py" in extra[0], extra

    def scanned(names):
        return _residuals(lambda s: integrate(s, mesh, op, stress, N, p, dt=DT, ale_cfg=ale,
                                              save_names=names), st0)

    q0, q1 = scanned(()), scanned(("ssh_delta",))
    stacked = f"f64[{N - 1},{nod}]"
    k0 = sum(ln.startswith(stacked) for ln in q0)
    k1 = sum(ln.startswith(stacked) for ln in q1)
    print(f"{mode}: integrate({N}) residuals none {len(q0)} / ssh_delta {len(q1)}; "
          f"{stacked} none {k0} / ssh_delta {k1}")
    assert len(q1) == len(q0) + 1 and k1 == k0 + 1, (len(q0), len(q1), k0, k1)


@pytest.mark.parametrize("mode", ["linfs", "zstar"])
def test_save_names_moves_the_gradient_less_than_remat_does(fixture, mode, capsys):
    """State AND params cotangents with the policy, without it, and without rematerialisation.

    Not bitwise (see the module docstring): the reference for "how much may a change in what
    is recomputed move the gradient" is rematerialisation itself, measured here in the same
    process. Also a repeat of the no-policy program, which must be bitwise (the programs are
    deterministic on CPU; the difference is between programs)."""
    mesh, op, stress, st0 = fixture
    p, ale, xs = _params(), _ale(mode), _xs()

    def via_scan(names, remat=True):
        def f(s, pp):
            return scan_steps(_one_step(fixture, pp, ale), s, xs, rematerialize=remat,
                              save_names=names)
        # structured cotangent: the surface temperature plus the surface elevation
        shp = jax.eval_shape(f, st0, p)
        ct = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), shp)
        ct = dataclasses.replace(ct, T=ct.T.at[:, 0].set(1.0),
                                 eta_n=jnp.ones(shp.eta_n.shape, shp.eta_n.dtype))
        return _flat(jax.jit(lambda s, pp, c: jax.vjp(f, s, pp)[1](c))(st0, p, ct))

    def via_integrate(names, remat=True):
        wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])

        def loss(s, pp):
            fin = integrate(s, mesh, op, stress, N, pp, dt=DT, ale_cfg=ale, checkpoint=remat,
                            save_names=names)
            return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)
        return _flat(jax.jit(jax.grad(loss, argnums=(0, 1)))(st0, p))

    try:
        for name, run in (("scan_steps", via_scan), ("integrate", via_integrate)):
            ref, pol, nor = run(()), run(("ssh_delta",)), run((), remat=False)
            d_pol, d_remat = _rel(pol, ref), _rel(ref, nor)
            print(f"\n{mode} {name}: {ref.size} entries, |g| {np.linalg.norm(ref):.3e}; "
                  f"policy vs none {d_pol:.3e} ({int(np.count_nonzero(pol != ref))} differ); "
                  f"remat vs no remat {d_remat:.3e}")
            assert np.linalg.norm(ref) > 0.0, "the reference gradient is zero"
            assert np.all(np.isfinite(pol))
            if name == "scan_steps":
                assert np.array_equal(run(()), ref), "the no-policy program is not deterministic"
            assert d_pol <= max(d_remat, 1e-14), (
                f"{mode} {name}: the policy moved the gradient by {d_pol:.3e}, more than "
                f"rematerialisation itself ({d_remat:.3e})")
    finally:
        jax.clear_caches()
