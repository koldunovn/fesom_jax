"""Task 3.1 gate — the checkpointed ``lax.scan`` time loop (:mod:`fesom_jax.integrate`).

The de-risking question: does wrapping ``step`` in ``lax.scan`` + ``jax.checkpoint``
(a) reproduce the Phase-2 Python ``run`` loop forward, and (b) admit a finite
end-to-end reverse-mode gradient (scan + remat + the CG ``custom_linear_solve`` all
composing)? This file proves (a) and a small-N (b); the N=200 backward-pass **memory**
sanity runs on the GPU (``docs/REFERENCE_RUNS.md``), CPU being fine for correctness.

* **Forward equivalence:** ``integrate`` == ``run`` (the eager step-1 + scanned 2..N
  reproduces the loop that sets ``is_first_step`` on step 0 only) — bit-identical here.
* **Checkpoint is forward-transparent:** ``checkpoint=True`` vs ``False`` give the
  exact same forward state (remat only changes the backward pass).
* **Backward is finite:** ``jax.grad`` of a scalar loss through the scan is finite and
  nonzero (the full AD value gate is ``test_gradient.py``; this is the scan-plumbing
  smoke).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import forcing, ic, ssh
from fesom_jax import step as stepmod
from fesom_jax.integrate import integrate, integrate_jit
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params

DT = 100.0


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


@pytest.fixture(scope="module")
def op(mesh):
    return ssh.build_ssh_operator(mesh, dt=DT)


@pytest.fixture(scope="module")
def stress(mesh):
    return forcing.surface_stress(mesh)


@pytest.fixture(scope="module")
def st0(mesh):
    return ic.initial_state(mesh)


# --------------------------------------------------------------------------
# 1. forward equivalence: scan == the Phase-2 run loop
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_scan_matches_run_loop(mesh, op, stress, st0, n):
    """``integrate`` (eager step-1 + scanned 2..N) reproduces ``run`` (the Python
    loop) for every N — including N=1 (the eager-only branch). Bit-identical here
    (the scan body and the per-step jit fuse to the same XLA), so gate tight."""
    fin_int = integrate_jit(st0, mesh, op, stress, n_steps=n, dt=DT)
    fin_run = stepmod.run(st0, mesh, op, stress, n, dt=DT)
    for f in ["uv", "eta_n", "d_eta", "hbar", "w", "density", "bvfreq", "T", "S"]:
        a = np.asarray(getattr(fin_int, f))
        b = np.asarray(getattr(fin_run, f))
        d = np.max(np.abs(a - b))
        assert d < 1e-12, f"N={n} {f}: integrate−run {d:.2e}"


# --------------------------------------------------------------------------
# 2. checkpoint is forward-transparent
# --------------------------------------------------------------------------
def test_checkpoint_forward_identical(mesh, op, stress, st0):
    """``jax.checkpoint`` (rematerialization) changes only the backward pass — the
    forward state must be bit-identical with it on vs off."""
    a = integrate_jit(st0, mesh, op, stress, n_steps=10, dt=DT, checkpoint=True)
    b = integrate_jit(st0, mesh, op, stress, n_steps=10, dt=DT, checkpoint=False)
    for f in ["uv", "eta_n", "T", "S", "density", "w", "hbar"]:
        d = np.max(np.abs(np.asarray(getattr(a, f)) - np.asarray(getattr(b, f))))
        assert d == 0.0, f"checkpoint on/off {f}: {d:.2e}"


# --------------------------------------------------------------------------
# 3. backward pass is finite (the scan + checkpoint + CG AD plumbing works)
# --------------------------------------------------------------------------
def test_backward_through_scan_finite(mesh, op, stress, st0):
    """A small-N reverse-mode gradient through the checkpointed scan is finite and
    nonzero — proving ``lax.scan`` + ``jax.checkpoint`` + the CG ``custom_linear_solve``
    compose under AD. (Value-correctness is ``test_gradient.py``.)"""
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    nwet0 = jnp.sum(wet0)

    def loss(kver):
        p = Params(k_ver=kver, a_ver=jnp.asarray(1e-4, jnp.float64))
        fin = integrate(st0, mesh, op, stress, n_steps=8, params=p, dt=DT)
        return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / nwet0

    g = float(jax.grad(loss)(jnp.asarray(1e-5, jnp.float64)))
    assert np.isfinite(g) and g != 0.0


def test_backward_uncheckpointed_matches(mesh, op, stress, st0):
    """The gradient is the same with and without checkpointing (remat must not change
    the gradient value, only the memory it costs to compute it)."""
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    nwet0 = jnp.sum(wet0)

    def loss(kver, ckpt):
        p = Params(k_ver=kver, a_ver=jnp.asarray(1e-4, jnp.float64))
        fin = integrate(st0, mesh, op, stress, n_steps=8, params=p, dt=DT,
                        checkpoint=ckpt)
        return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / nwet0

    k0 = jnp.asarray(1e-5, jnp.float64)
    g_ck = float(jax.grad(lambda k: loss(k, True))(k0))
    g_no = float(jax.grad(lambda k: loss(k, False))(k0))
    assert abs(g_ck - g_no) <= 1e-12 * max(abs(g_no), 1.0)
