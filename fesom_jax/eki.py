"""Gradient-free Ensemble Kalman Inversion (EKI) — the slow/equilibrium calibration tool
(Paper-experiments Task A4 / the §2 GM-T·S pillar).

EKI (Iglesias, Law & Stuart 2013) minimizes a data-misfit ``‖y* − G(θ)‖²_Γ⁻¹`` using **only
forward evaluations** of ``G`` — no adjoint. That is exactly what the slow targets need: the
GM→stratification calibration is a multi-decade-ish equilibrium quantity where the adjoint hits
the chaos/memory ceiling (review MAJOR-1), but a forward ensemble is immune to it and
``vmap``-parallel (the fast JAX/GPU forward makes the ensemble cheap). EKI is also the
cross-check partner for the short-window adjoint on the shared ``k_gm`` scalar (§1).

One EKI iteration over a J-member ensemble ``{θ_j}`` (perturbed-observation / stochastic form):

    g_j   = G(θ_j)                                  (the forward, vmapped over members)
    C_θg  = cov(θ, g),   C_gg = cov(g, g)           (empirical ensemble covariances)
    θ_j  += C_θg (C_gg + Γ)⁻¹ (y* [+ η_j] − g_j),   η_j ~ N(0, Γ)

The ensemble **mean** converges to the minimizer. ``Γ`` is the observation-noise covariance
(scalar σ², a per-component vector, or a full matrix). Float64 throughout.

This module is the generic driver; the real-model EKI budget (warm-started 16–32-member
few-year ensemble on ``k_gm``, upper-ocean target) lives in the plan (A4/D2b). The unit test is
a noisy analytic forward.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def _as_cov(gamma, d: int):
    """Coerce ``gamma`` (scalar σ², ``[d]`` diagonal, or ``[d, d]`` matrix) → a ``[d, d]``
    covariance matrix."""
    g = jnp.asarray(gamma, jnp.float64)
    if g.ndim == 0:
        return g * jnp.eye(d)
    if g.ndim == 1:
        return jnp.diag(g)
    return g


def eki_update(thetas, g, y_obs, gamma, *, key=None, perturb_obs: bool = True):
    """One EKI update from **precomputed** forwards ``g`` (no forward eval here).

    ``thetas`` ``[J, p]`` (ensemble of parameters), ``g`` ``[J, d]`` (their forwards), ``y_obs``
    ``[d]``, ``gamma`` the obs-noise covariance. With ``perturb_obs`` each member sees ``y_obs``
    plus an independent ``N(0, Γ)`` draw (the stochastic EKI; needs ``key``). Returns the updated
    ``thetas`` ``[J, p]``."""
    thetas = jnp.asarray(thetas, jnp.float64)
    g = jnp.asarray(g, jnp.float64)
    J, d = g.shape
    Gamma = _as_cov(gamma, d)

    tbar = jnp.mean(thetas, axis=0)
    gbar = jnp.mean(g, axis=0)
    dth = thetas - tbar                                  # [J, p]
    dg = g - gbar                                        # [J, d]
    C_tg = dth.T @ dg / J                                # [p, d]
    C_gg = dg.T @ dg / J                                 # [d, d]

    y = jnp.broadcast_to(jnp.asarray(y_obs, jnp.float64), (J, d))
    if perturb_obs:
        if key is None:
            raise ValueError("eki_update: perturb_obs=True requires a PRNG `key`")
        L = jnp.linalg.cholesky(Gamma)
        z = jax.random.normal(key, (J, d), jnp.float64)
        y = y + z @ L.T                                  # y_obs + N(0, Γ) per member
    innov = y - g                                        # [J, d]
    # solve (C_gg + Γ) X = innovᵀ  → X [d, J]; update = C_θg X
    M = C_gg + Gamma
    sol = jnp.linalg.solve(M, innov.T)                   # [d, J]
    update = (C_tg @ sol).T                              # [J, p]
    return thetas + update


def eki_step(thetas, forward_fn: Callable, y_obs, gamma, *, key=None,
             perturb_obs: bool = True):
    """One EKI step: ``vmap`` the forward over the ensemble, then :func:`eki_update`.
    ``forward_fn`` maps a single ``θ`` ``[p]`` → an observable ``[d]``."""
    g = jax.vmap(forward_fn)(jnp.asarray(thetas, jnp.float64))
    return eki_update(thetas, g, y_obs, gamma, key=key, perturb_obs=perturb_obs)


def eki_run(theta0, forward_fn: Callable, y_obs, gamma, *, n_iters: int,
            key=None, perturb_obs: bool = True, on_step: Callable | None = None):
    """Run EKI for ``n_iters`` iterations from the initial ensemble ``theta0`` ``[J, p]``.

    Each iteration evaluates the forward **once** (``vmap`` over members), logs the data misfit
    of the ensemble-mean prediction + the parameter spread, then applies the EKI update. Returns
    ``(theta_mean[p], history, ensemble[J, p])`` — the recovered parameters are the final
    ensemble mean. ``on_step(record)`` is called per iter for live logging."""
    thetas = jnp.asarray(theta0, jnp.float64)
    y = jnp.asarray(y_obs, jnp.float64)
    d = y.shape[0]
    Gamma = _as_cov(gamma, d)
    Ginv = jnp.linalg.inv(Gamma)
    history: list[dict] = []
    if key is None:
        key = jax.random.PRNGKey(0)
    for it in range(1, n_iters + 1):
        key, sub = jax.random.split(key)
        g = jax.vmap(forward_fn)(thetas)                 # [J, d]
        gbar = jnp.mean(g, axis=0)
        resid = y - gbar
        misfit = float(resid @ Ginv @ resid)
        rec = {
            "it": it,
            "theta_mean": jax.device_get(jnp.mean(thetas, axis=0)),
            "spread": float(jnp.mean(jnp.std(thetas, axis=0))),
            "misfit": misfit,
        }
        history.append(rec)
        if on_step is not None:
            on_step(rec)
        thetas = eki_update(thetas, g, y, gamma, key=sub, perturb_obs=perturb_obs)
    return jnp.mean(thetas, axis=0), history, thetas
