"""A4 gate: the gradient-free EKI driver (`eki`).

CPU unit tests proving the ensemble-mean recovers known parameters from a noisy analytic
forward — the machinery the slow GM→stratification calibration (§2, review MAJOR-1) uses where
the adjoint cannot reach. No adjoint is taken anywhere here (forward evaluations only).
Token: EKI_OK.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import eki

RNG = np.random.default_rng(20260614)


# --------------------------------------------------------------------------
# covariance coercion
# --------------------------------------------------------------------------
def test_as_cov_scalar_vector_matrix():
    np.testing.assert_array_equal(np.asarray(eki._as_cov(0.04, 3)), 0.04 * np.eye(3))
    np.testing.assert_array_equal(np.asarray(eki._as_cov(np.array([1., 2., 3.]), 3)),
                                  np.diag([1., 2., 3.]))
    M = np.array([[2., 0.3], [0.3, 1.]])
    np.testing.assert_array_equal(np.asarray(eki._as_cov(M, 2)), M)


# --------------------------------------------------------------------------
# scalar recovery from a noisy linear forward (the headline EKI test)
# --------------------------------------------------------------------------
def test_eki_recovers_known_scalar():
    """G(θ) = 2.5·θ, truth θ*=4 ⇒ y* = 10 (+noise). The ensemble mean recovers θ* within the
    observation-noise tolerance."""
    A = 2.5
    theta_true = 4.0
    sigma = 0.05
    y_obs = jnp.asarray([A * theta_true + float(RNG.normal(0, sigma))])

    def G(theta):
        return A * theta                                   # θ:[1] → [1]

    J = 64
    theta0 = jnp.asarray(RNG.normal(0.0, 5.0, (J, 1)))     # broad initial ensemble
    mean, history, _ = eki.eki_run(theta0, G, y_obs, sigma ** 2, n_iters=30,
                                   key=jax.random.PRNGKey(0), perturb_obs=True)
    assert abs(float(mean[0]) - theta_true) < 0.05         # within ~1.2%
    assert history[-1]["misfit"] < 1e-2 * history[0]["misfit"]   # misfit collapses


def test_eki_recovers_multiple_parameters():
    """Overdetermined linear forward G(θ)=A θ (A: 4×2): recover both components of θ*."""
    A = jnp.asarray(RNG.standard_normal((4, 2)))
    theta_true = jnp.asarray([1.5, -2.0])
    y_obs = A @ theta_true + jnp.asarray(RNG.normal(0, 0.02, 4))

    def G(theta):
        return A @ theta

    J = 80
    theta0 = jnp.asarray(RNG.normal(0.0, 4.0, (J, 2)))
    mean, history, ens = eki.eki_run(theta0, G, y_obs, 0.02 ** 2, n_iters=40,
                                     key=jax.random.PRNGKey(1), perturb_obs=True)
    np.testing.assert_allclose(np.asarray(mean), np.asarray(theta_true), atol=0.05)
    assert history[-1]["misfit"] < history[0]["misfit"]


def test_eki_nonlinear_forward_recovers():
    """A mildly nonlinear forward G(θ)=[θ, θ²]: the ensemble mean still recovers θ*."""
    theta_true = 2.0

    def G(theta):
        t = theta[0]
        return jnp.array([t, t * t])

    y_obs = jnp.asarray([theta_true, theta_true ** 2]) + jnp.asarray(RNG.normal(0, 0.01, 2))
    J = 100
    theta0 = jnp.asarray(RNG.uniform(0.5, 4.0, (J, 1)))    # positive (avoid the θ→-θ² ambiguity)
    mean, history, _ = eki.eki_run(theta0, G, y_obs, 0.01 ** 2, n_iters=40,
                                   key=jax.random.PRNGKey(2), perturb_obs=True)
    assert abs(float(mean[0]) - theta_true) < 0.1


def test_eki_deterministic_no_perturbation_converges():
    """Deterministic EKI (perturb_obs=False) needs no key and still converges the mean."""
    A = 1.5
    theta_true = 3.0
    y_obs = jnp.asarray([A * theta_true])

    def G(theta):
        return A * theta

    theta0 = jnp.asarray(RNG.normal(0.0, 5.0, (50, 1)))
    mean, history, _ = eki.eki_run(theta0, G, y_obs, 0.01, n_iters=30, perturb_obs=False)
    assert abs(float(mean[0]) - theta_true) < 0.05
    # misfit is non-increasing across iterations (deterministic descent)
    misfits = [h["misfit"] for h in history]
    assert misfits[-1] < misfits[0]


def test_eki_step_single_update_shapes():
    """eki_step vmaps the forward + updates; shapes preserved."""
    def G(theta):
        return jnp.array([theta[0] + theta[1], theta[0] - theta[1]])

    theta0 = jnp.asarray(RNG.standard_normal((30, 2)))
    new = eki.eki_step(theta0, G, jnp.asarray([1.0, 0.0]), 0.1,
                       key=jax.random.PRNGKey(3), perturb_obs=True)
    assert new.shape == (30, 2)
    assert np.all(np.isfinite(np.asarray(new)))


def test_eki_update_requires_key_when_perturbing():
    with pytest.raises(ValueError, match="requires a PRNG"):
        eki.eki_update(jnp.zeros((4, 1)), jnp.zeros((4, 1)), jnp.zeros(1), 0.1,
                       perturb_obs=True)


def test_eki_ok_token():
    """Aggregate gate — prints EKI_OK."""
    def G(theta):
        return 3.0 * theta

    theta_true = 2.0
    y_obs = jnp.asarray([3.0 * theta_true])
    theta0 = jnp.asarray(RNG.normal(0.0, 4.0, (64, 1)))
    mean, history, _ = eki.eki_run(theta0, G, y_obs, 0.01, n_iters=25,
                                   key=jax.random.PRNGKey(4), perturb_obs=True)
    recovered = abs(float(mean[0]) - theta_true) < 0.05
    reduced = history[-1]["misfit"] < 1e-2 * history[0]["misfit"]
    assert recovered and reduced
    print("EKI_OK")
