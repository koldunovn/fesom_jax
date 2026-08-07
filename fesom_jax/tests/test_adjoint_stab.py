"""Tests for the backward-pass stabilizers (NN-GM Task 2 follow-on).

Runs on the packaged `pi` mesh — small, real FESOM geometry, CPU-fast; no model steps.

What these check is the property that makes each stabilizer trustworthy rather than merely
plausible. A stabilizer is inserted into a backward pass whose output nobody can eyeball,
so the failure mode is silent: an operator that quietly rescaled the gradient, or that
applied the filter instead of its transpose, would look exactly like a stabilizer that
worked. Every claim in the module docstring gets a test here.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import adjoint_stab, coarsen
from fesom_jax.mesh import load_mesh

PI_MESH_DIR = Path(__file__).resolve().parents[1] / "data" / "mesh_pi"


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(PI_MESH_DIR)


# ---------------------------------------------------------------------------
# the primitive

def test_cotangent_op_is_the_identity_in_the_forward():
    """The whole design rests on this: inserting a stabilizer must not move the forward."""
    f = adjoint_stab.make_discounter(0.5)
    x = jnp.array([1.0, -2.0, 3.5])
    out = adjoint_stab.cotangent_op(f, x)
    assert np.array_equal(np.asarray(out), np.asarray(x))


def test_cotangent_op_applies_the_map_in_the_backward():
    f = adjoint_stab.make_discounter(0.25)

    def loss(x):
        return jnp.sum(adjoint_stab.cotangent_op(f, x) ** 2)

    x = jnp.array([1.0, 2.0, 3.0])
    g = jax.grad(loss)(x)
    assert np.allclose(np.asarray(g), 0.25 * 2.0 * np.asarray(x))


def test_discount_compounds_over_a_scan():
    """gamma applied once per step must weight a step k back by gamma**k — the property the
    whole 'discounted adjoint' interpretation depends on."""
    gamma, nstep = 0.5, 6
    f = adjoint_stab.make_discounter(gamma)

    def loss(x):
        def body(c, _):
            return adjoint_stab.cotangent_op(f, c), None
        out, _ = jax.lax.scan(body, x, xs=None, length=nstep)
        return jnp.sum(out)

    g = float(jax.grad(loss)(jnp.array(1.0)))
    assert np.isclose(g, gamma ** nstep)


def test_freeze_increment_keeps_the_value_but_not_the_derivative():
    """freeze_increment must be exact in the forward and see only `base` in the backward."""
    def base_fn(x):
        return 2.0 * x

    def full_fn(x):
        return 2.0 * x + 10.0 * x ** 2          # the "stage" adds a stiff term

    def f(x):
        return adjoint_stab.freeze_increment(base_fn(x), full_fn(x))

    x = jnp.array(3.0)
    assert np.isclose(float(f(x)), float(full_fn(x)))        # value: the full stage
    assert np.isclose(float(jax.grad(f)(x)), 2.0)            # slope: base only, not 2+20x


def test_clipper_caps_the_norm_and_keeps_the_direction():
    cap = 1.0
    f = adjoint_stab.make_clipper(cap)
    big = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([0.0])}
    out = f(big)
    nrm = float(adjoint_stab.cotangent_norm(out))
    assert np.isclose(nrm, cap, rtol=1e-12)
    # direction preserved
    assert np.allclose(np.asarray(out["a"]), np.asarray(big["a"]) / 5.0)
    # below the cap it is the identity
    small = {"a": jnp.array([0.3, 0.4]), "b": jnp.array([0.0])}
    assert np.allclose(np.asarray(f(small)["a"]), np.asarray(small["a"]))


def test_cotangent_norm_skips_float0():
    """jax hands back float0 cotangents for integer leaves; arithmetic on them raises, so the
    norm and every scalar map must step over them rather than blow up."""
    tree = {"x": jnp.array([3.0, 4.0]),
            "i": np.zeros((2,), dtype=jax.dtypes.float0)}
    assert np.isclose(float(adjoint_stab.cotangent_norm(tree)), 5.0)
    out = adjoint_stab.make_discounter(0.5)(tree)
    assert np.allclose(np.asarray(out["x"]), [1.5, 2.0])
    assert out["i"].dtype == jax.dtypes.float0


# ---------------------------------------------------------------------------
# the adjoint diffusion

def test_smoother_is_the_exact_transpose_of_the_filter(mesh):
    """<F x, y> == <x, F^T y>. This is the test that catches applying the filter itself to
    the cotangent: on a mesh whose cell areas span orders of magnitude, F and F^T differ by
    the area metric, and the mistake is invisible in any single field."""
    factor = 2.0
    apply_T, (n_steps, kappa_dt) = adjoint_stab.make_nodal_smoother(mesh, factor)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(mesh.nod2D,)))
    y = jnp.asarray(rng.normal(size=(mesh.nod2D,)))
    mask = jnp.asarray(mesh.node_layer_mask, jnp.float64)[:, 0]
    Fx = coarsen.filter_field(mesh, x, n_steps=n_steps, kappa_dt=kappa_dt,
                              node_mask=mask, check_stability=False)
    FTy = apply_T(y)
    lhs, rhs = float(jnp.sum(Fx * y)), float(jnp.sum(x * FTy))
    assert np.isclose(lhs, rhs, rtol=1e-9), f"<Fx,y>={lhs} != <x,F^T y>={rhs}"


def test_smoother_conserves_total_sensitivity(mesh):
    """F preserves constants, so F^T preserves the SUM of the cotangent. A stabilizer that
    could rescale the gradient total would be indistinguishable from a bug."""
    apply_T, _ = adjoint_stab.make_nodal_smoother(mesh, 2.0)
    rng = np.random.default_rng(1)
    g = jnp.asarray(rng.normal(size=(mesh.nod2D,)))
    out = apply_T(g)
    assert np.isclose(float(jnp.sum(out)), float(jnp.sum(g)), rtol=1e-9)


def test_smoother_actually_smooths(mesh):
    """Grid-scale structure must come down. Uses the roughness measure the filter targets:
    the norm of the discrete Laplacian."""
    apply_T, _ = adjoint_stab.make_nodal_smoother(mesh, 3.0)
    rng = np.random.default_rng(2)
    g = jnp.asarray(rng.normal(size=(mesh.nod2D,)))
    rough = lambda v: float(jnp.linalg.norm(coarsen.laplacian(mesh, v)))
    assert rough(apply_T(g)) < 0.5 * rough(g)


def test_smoother_handles_3d_and_leaves_non_nodal_fields_alone(mesh):
    """A State cotangent is a mixed bag: (N,), (N,nl), (N,nl,2) and element fields. The
    smoother must act on the nodal ones and pass the rest through untouched — silently
    smoothing an element field with a nodal operator would be a shape-lucky disaster."""
    apply_T, _ = adjoint_stab.make_nodal_smoother(mesh, 2.0)
    nl = int(mesh.nl)
    rng = np.random.default_rng(3)
    for shape in [(mesh.nod2D,), (mesh.nod2D, nl), (mesh.nod2D, nl, 2)]:
        out = apply_T(jnp.asarray(rng.normal(size=shape)))
        assert out.shape == shape
    elem = jnp.asarray(rng.normal(size=(mesh.elem2D, nl)))
    assert np.array_equal(np.asarray(apply_T(elem)), np.asarray(elem))
    assert float(apply_T(jnp.asarray(2.0))) == 2.0                 # scalar passes through


# ---------------------------------------------------------------------------
# assembly

def test_stabilizer_is_none_when_nothing_is_asked_for(mesh):
    """Off must mean absent, not an identity wrapper — the traced graph of an unstabilized
    run has to stay exactly what it was."""
    assert adjoint_stab.make_state_stabilizer(mesh, adjoint_stab.StabConfig()) is None
    assert not adjoint_stab.StabConfig().active
    assert adjoint_stab.StabConfig().describe() == "none"


def test_stabilizer_composes_and_leaves_the_forward_alone(mesh):
    cfg = adjoint_stab.StabConfig(gamma=0.5, filter_factor=2.0)
    stab = adjoint_stab.make_state_stabilizer(mesh, cfg)
    assert stab is not None and "gamma=0.5" in cfg.describe() and "filter=2dx" in cfg.describe()
    tree = {"nodal": jnp.ones((mesh.nod2D,)), "elem": jnp.ones((mesh.elem2D,))}
    out = stab(tree)
    for k in tree:
        assert np.array_equal(np.asarray(out[k]), np.asarray(tree[k])), "forward must be identity"

    def loss(t):
        return jnp.sum(stab(t)["nodal"]) + jnp.sum(stab(t)["elem"])

    g = jax.grad(loss)(tree)
    # element field: discount only. nodal: discount * F^T, whose sum is conserved.
    assert np.allclose(np.asarray(g["elem"]), 0.5)
    assert np.isclose(float(jnp.sum(g["nodal"])), 0.5 * mesh.nod2D, rtol=1e-9)
