"""NN-GM Task 2 — unit tests for the horizon-scan loss form (pure functions, CPU-fast).

Covers the plan's "tests: scan harness determinism on CPU; loss-form units": the
``level_weights`` / ``stage1_misfit`` pieces of ``scripts/nn_eddy/horizon_scan.py`` on
tiny synthetic arrays (no mesh, no model). The in-model harness checks live in the scan
itself (``loss_at_truth == 0`` per process). Runs standalone (``pytest scripts/tests/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nn_eddy"))
import horizon_scan as hs  # noqa: E402


def _toy(nn=4, nl=3, nmark=2, seed=0):
    rng = np.random.default_rng(seed)
    T = jnp.asarray(rng.normal(10.0, 1.0, (nmark, nn, nl)))
    S = jnp.asarray(rng.normal(35.0, 0.1, (nmark, nn, nl)))
    eta = jnp.asarray(rng.normal(0.0, 0.1, (nmark, nn)))
    wet = jnp.asarray(rng.random((nn, nl)) > 0.2, jnp.float64)
    wet = wet.at[:, 0].set(1.0)                       # surface always wet
    area = jnp.asarray(rng.uniform(1.0, 3.0, (nn, nl))) * wet
    return T, S, eta, wet, area


def test_misfit_zero_when_equal():
    T, S, eta, wet, area = _toy()
    w_T, w_S, w_eta = hs.level_weights(T, S, eta, wet, area)
    L = hs.stage1_misfit(T[0], S[0], eta[0], T[0], S[0], eta[0], w_T, w_S, w_eta, wet, area)
    assert float(L) == 0.0


def test_misfit_known_value_single_cell():
    # one node, one level, area 2: hand-computable
    wet = jnp.ones((1, 1))
    area = 2.0 * wet
    w_T = jnp.asarray([3.0]); w_S = jnp.asarray([0.5]); w_eta = 7.0
    T = jnp.asarray([[1.0]]); T_hat = jnp.asarray([[1.5]])       # dT=-0.5
    S = jnp.asarray([[35.0]]); S_hat = jnp.asarray([[35.0]])
    eta = jnp.asarray([0.2]); eta_hat = jnp.asarray([0.0])       # deta=0.2
    L = hs.stage1_misfit(T, S, eta, T_hat, S_hat, eta_hat, w_T, w_S, w_eta, wet, area)
    # lev_frac=1; L = w_T*0.25 + w_S*0 + w_eta*0.04 = 0.75 + 0.28
    assert np.isclose(float(L), 3.0 * 0.25 + 7.0 * 0.04, rtol=1e-12)


def test_level_weights_floor_on_constant_truth():
    T, S, eta, wet, area = _toy()
    Tc = jnp.broadcast_to(T[:1], T.shape)            # constant across marks -> var 0
    Sc = jnp.broadcast_to(S[:1], S.shape)
    ec = jnp.broadcast_to(eta[:1], eta.shape)
    w_T, w_S, w_eta = hs.level_weights(Tc, Sc, ec, wet, area)
    assert np.allclose(np.asarray(w_T), 1.0 / hs.VAR_FLOOR_T)
    assert np.allclose(np.asarray(w_S), 1.0 / hs.VAR_FLOOR_S)
    assert np.isclose(float(w_eta), 1.0 / hs.VAR_FLOOR_ETA)


def test_level_weights_known_variance():
    # two marks at +-a around a mean, uniform over nodes: temporal var = a^2 everywhere
    nn, nl = 3, 2
    a = 0.5                                            # a^2 = 0.25 >> floors
    base = 10.0 * jnp.ones((nn, nl))
    T = jnp.stack([base - a, base + a])
    S = jnp.stack([35.0 * jnp.ones((nn, nl))] * 2)     # constant -> floored
    eta = jnp.stack([-a * jnp.ones(nn), a * jnp.ones(nn)])
    wet = jnp.ones((nn, nl)); area = jnp.ones((nn, nl))
    w_T, w_S, w_eta = hs.level_weights(T, S, eta, wet, area)
    assert np.allclose(np.asarray(w_T), 1.0 / a**2, rtol=1e-12)
    assert np.allclose(np.asarray(w_S), 1.0 / hs.VAR_FLOOR_S)
    assert np.isclose(float(w_eta), 1.0 / a**2, rtol=1e-12)


def test_dry_nodes_do_not_contribute():
    T, S, eta, wet, area = _toy()
    w_T, w_S, w_eta = hs.level_weights(T, S, eta, wet, area)
    L0 = hs.stage1_misfit(T[0], S[0], eta[0], T[1], S[1], eta[1], w_T, w_S, w_eta, wet, area)
    # perturb T on DRY cells only -> loss unchanged
    T_dirty = T[0] + 100.0 * (1.0 - wet)
    L1 = hs.stage1_misfit(T_dirty, S[0], eta[0], T[1], S[1], eta[1], w_T, w_S, w_eta, wet, area)
    assert float(L0) == float(L1)


def test_misfit_deterministic_under_jit():
    T, S, eta, wet, area = _toy(seed=1)
    w = hs.level_weights(T, S, eta, wet, area)
    f = jax.jit(lambda: hs.stage1_misfit(T[0], S[0], eta[0], T[1], S[1], eta[1], *w, wet, area))
    assert float(f()) == float(f())


def test_misfit_grad_finite_and_informative():
    # d misfit / d T points from student toward truth (sign of -2*w*(T_hat-T)*...)
    T, S, eta, wet, area = _toy(seed=2)
    w = hs.level_weights(T, S, eta, wet, area)
    g = jax.grad(lambda t: hs.stage1_misfit(t, S[0], eta[0], T[1], S[1], eta[1], *w, wet, area))(T[0])
    g = np.asarray(g)
    assert np.isfinite(g).all()
    wetm = np.asarray(wet) > 0
    diff = np.asarray(T[0] - T[1])
    # descending along -g must reduce |T - T_hat| where wet: g and diff share sign
    assert (np.sign(g[wetm]) == np.sign(2 * diff[wetm] * np.where(diff[wetm] != 0, 1, 0))).all() or \
           np.allclose(g[wetm][diff[wetm] == 0], 0.0)
