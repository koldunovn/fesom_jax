"""``fesom_jax.adjoint_check`` on small maps whose answer is known — no model, seconds.

A correct derivative passes the dot test and a planted wrong transpose fails it; the FD sweep
finds the plateau of a smooth cost and reports the noise floor of a noisy one; power iteration
recovers the top singular value of a matrix with a chosen weighted SVD, in a leaf-restricted
norm and under a mask; ``fdlin`` marks its regimes. The window maps on the model are in
``test_adjoint_check.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import adjoint_check as AC


# ---------------------------------------------------------------------------
# dot test


_A = np.array([[2.0, 1.0, 0.0, 0.5, -1.0],
               [0.0, 3.0, 1.0, 0.0, 0.2],
               [1.0, -2.0, 1.0, 4.0, 0.0],
               [0.3, 0.0, 0.0, 1.0, 2.0]])          # nonsymmetric, 4x5


def _toy_f(x):
    z = jnp.concatenate([jnp.tanh(x["a"]), x["b"] ** 3])
    return {"y": jnp.asarray(_A) @ z}


def _toy_x():
    return {"a": jnp.array([0.3, -0.2, 0.1]), "b": jnp.array([0.5, 1.2])}


def _toy_v():
    return {"a": jnp.array([1.0, 0.5, -0.3]), "b": jnp.array([-0.7, 0.2])}


def test_dot_test_passes_on_a_correct_derivative():
    rows = AC.dot_test(_toy_f, _toy_x(), _toy_v(), {"y": jnp.array([1.0, -2.0, 0.5, 0.3])})
    assert [r.amp for r in rows] == [1.0, 1e-4, 1e-8]
    for r in rows:
        assert r.rel <= 1e-13 and r.rel_norm <= 1e-13, r


def test_dot_test_catches_a_wrong_transpose():
    """A planted adjoint that applies M instead of Mᵀ (M nonsymmetric) fails at every
    amplitude."""
    M = jnp.asarray(_A[:, :4])
    rows = AC.dot_test(None, None, jnp.array([1.0, -1.0, 0.5, 2.0]),
                       jnp.array([0.3, 1.0, -0.4, 0.8]),
                       jvp=lambda t: M @ t, vjp=lambda c: M @ c)
    for r in rows:
        assert r.rel > 1e-2 and r.rel_norm > 1e-2, r
    good = AC.dot_test(None, None, jnp.array([1.0, -1.0, 0.5, 2.0]),
                       jnp.array([0.3, 1.0, -0.4, 0.8]),
                       jvp=lambda t: M @ t, vjp=lambda c: M.T @ c)
    assert all(r.rel <= 1e-14 for r in good)


def test_dot_test_with_explicit_maps_matches_the_function_form():
    x, v, w = _toy_x(), _toy_v(), {"y": jnp.array([1.0, -2.0, 0.5, 0.3])}
    rows_f = AC.dot_test(_toy_f, x, v, w, jit=False)
    rows_m = AC.dot_test(None, None, v, w,
                         jvp=lambda t: jax.jvp(_toy_f, (x,), (t,))[1],
                         vjp=lambda c: jax.vjp(_toy_f, x)[1](c)[0])
    for a, b in zip(rows_f, rows_m):
        assert a.lhs == pytest.approx(b.lhs, rel=1e-14)
        assert a.rhs == pytest.approx(b.rhs, rel=1e-14)
    with pytest.raises(ValueError, match="both"):
        AC.dot_test(None, None, v, w, jvp=lambda t: t)


# ---------------------------------------------------------------------------
# finite differences


def _J_smooth(x):
    return jnp.sum(jnp.sin(x) * jnp.exp(0.1 * x))


HS = [10.0 ** -k for k in range(1, 11)]


def test_fd_sweep_finds_the_plateau_of_a_smooth_cost():
    x = jnp.linspace(-1.0, 1.0, 7)
    d = jnp.cos(jnp.arange(7.0))
    ad = float(jnp.vdot(jax.grad(_J_smooth)(x), d))
    rows = AC.fd_sweep(_J_smooth, x, d, HS, ad, repeats=2)
    assert all(r.noise_over_h == 0.0 for r in rows)         # deterministic
    p = AC.plateau(rows, rtol=1e-9)
    assert p is not None and p["n"] >= 3, p
    assert p["rel_err"] <= 1e-9, p
    # truncation error excludes the large steps, round-off the small ones
    assert 1e-8 <= p["h_min"] and p["h_max"] <= 1e-3, p


def test_fd_sweep_reports_the_noise_floor_of_a_noisy_cost():
    rng = np.random.default_rng(0)
    x = jnp.linspace(-1.0, 1.0, 7)
    d = jnp.cos(jnp.arange(7.0))
    ad = float(jnp.vdot(jax.grad(_J_smooth)(x), d))

    def J_noisy(x_):
        return float(_J_smooth(x_)) + 1e-9 * rng.standard_normal()

    rows = AC.fd_sweep(J_noisy, x, d, HS, ad, repeats=3)
    by_h = {r.h: r for r in rows}
    for r in rows:
        assert r.noise_over_h > 0.0
    # the noise dominates at small h and is what the row reports
    assert by_h[1e-10].rel_err > 1e-2
    assert by_h[1e-10].noise_over_h > 1e-2 * abs(ad)
    p = AC.plateau(rows, rtol=1e-4)
    assert p is not None and p["h_min"] >= 1e-5 and p["rel_err"] <= 1e-4, p


def test_plateau_is_none_without_agreement():
    rows = [AC.FDRow(h, fd, 1.0, 0.0, 0.0) for h, fd in [(1e-1, 1.0), (1e-2, 2.0), (1e-3, 4.0)]]
    assert AC.plateau(rows, rtol=1e-3) is None


# ---------------------------------------------------------------------------
# power iteration in a weighted norm


def _weighted_problem(seed=1):
    """A map between {"T": 4, "S": 2} pytrees whose weighted singular values are chosen:
    ``M = W_out^{-1/2} U Σ Vᵀ W_in^{1/2}``."""
    rng = np.random.default_rng(seed)
    n = 6
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sig = np.array([5.0, 2.5, 1.0, 0.5, 0.2, 0.1])
    w_in = rng.uniform(0.5, 2.0, n)
    w_out = rng.uniform(0.5, 2.0, n)
    M = np.diag(w_out ** -0.5) @ U @ np.diag(sig) @ V.T @ np.diag(w_in ** 0.5)

    def to_vec(t):
        return jnp.concatenate([t["T"], t["S"]])

    def to_tree(y):
        return {"T": y[:4], "S": y[4:]}

    apply_M = lambda t: to_tree(jnp.asarray(M) @ to_vec(t))          # noqa: E731
    apply_MT = lambda c: to_tree(jnp.asarray(M).T @ to_vec(c))       # noqa: E731
    wi = {"T": jnp.asarray(w_in[:4]), "S": jnp.asarray(w_in[4:])}
    wo = {"T": jnp.asarray(w_out[:4]), "S": jnp.asarray(w_out[4:])}
    return M, sig, w_in, w_out, apply_M, apply_MT, wi, wo, to_vec


def test_power_iteration_recovers_the_weighted_top_singular_value():
    M, sig, w_in, w_out, apply_M, apply_MT, wi, wo, to_vec = _weighted_problem()
    v0 = {"T": jnp.ones(4), "S": jnp.ones(2)}
    res = AC.power_iteration(apply_M, apply_MT, v0, weights=wi, out_weights=wo,
                             iters=100, tol=1e-13)
    assert res.converged
    assert res.sigma == pytest.approx(sig[0], rel=1e-10)
    v = np.asarray(to_vec(res.v))
    assert np.sum(w_in * v ** 2) == pytest.approx(1.0, rel=1e-12)
    u = np.asarray(to_vec(res.u))
    assert np.allclose(u, M @ v / res.sigma, rtol=0, atol=1e-12)
    assert res.history[0]["sigma"] < res.sigma


def test_power_iteration_leaf_norm_and_mask():
    """A T-only norm and a mask on T select the top singular value of the matching block."""
    M, _, _, _, apply_M, apply_MT, _, _, to_vec = _weighted_problem(seed=2)
    v0 = {"T": jnp.array([1.0, 2.0, 3.0, 4.0]), "S": jnp.array([5.0, 6.0])}
    res = AC.power_iteration(apply_M, apply_MT, v0, weights={"T": 1.0}, iters=200, tol=1e-14)
    assert res.sigma == pytest.approx(np.linalg.svd(M[:4, :4], compute_uv=False)[0], rel=1e-9)
    assert np.all(np.asarray(res.v["S"]) == 0.0)
    m = np.array([True, False, True, True])
    res = AC.power_iteration(apply_M, apply_MT, v0, weights={"T": 1.0},
                             mask={"T": jnp.asarray(m)}, iters=200, tol=1e-14)
    assert res.sigma == pytest.approx(np.linalg.svd(M[:4, :4][:, m], compute_uv=False)[0],
                                      rel=1e-9)
    assert float(res.v["T"][1]) == 0.0
    # a leaf missing from the mask is unrestricted: T masked, S free
    res = AC.power_iteration(apply_M, apply_MT, v0, weights={"T": 1.0, "S": 1.0},
                             out_weights={"T": 1.0}, mask={"T": jnp.asarray(m)},
                             iters=200, tol=1e-14)
    cols = np.concatenate([np.flatnonzero(m), [4, 5]])
    assert res.sigma == pytest.approx(np.linalg.svd(M[:4][:, cols], compute_uv=False)[0],
                                      rel=1e-9)
    assert float(res.v["T"][1]) == 0.0 and float(jnp.max(jnp.abs(res.v["S"]))) > 0.0
    with pytest.raises(ValueError, match="norm"):
        AC.power_iteration(apply_M, apply_MT, {"T": jnp.zeros(4), "S": jnp.ones(2)},
                           weights={"T": 1.0}, iters=3)


# ---------------------------------------------------------------------------
# fdlin


def test_fdlin_marks_the_regimes():
    def J_map(x):
        return {"T": x["T"] + 0.3 * x["T"] ** 3, "S": 2.0 * x["S"]}

    x = {"T": jnp.array([0.5, -0.4, 1.0]), "S": jnp.array([1.0])}
    v = {"T": jnp.array([1.0, 0.5, -1.0]), "S": jnp.array([0.3])}
    tl = jax.jvp(J_map, (x,), (v,))[1]
    hs = [1.0, 1e-1, 1e-2, 1e-3, 1e-7, 1e-12]
    rows = AC.fdlin(J_map, x, v, hs, tl, predicted_range=0.05, noise_floor=1e-13)
    reg = {r.h: r.regime for r in rows}
    assert reg[1.0] == "beyond-predicted-range" and reg[1e-1] == "beyond-predicted-range"
    assert reg[1e-2] == "linear" and reg[1e-3] == "linear"
    assert reg[1e-12] == "noise-limited"
    by_h = {r.h: r for r in rows}
    assert by_h[1e-3].rel_err < 1e-5 < by_h[1e-1].rel_err
    assert set(by_h[1e-3].per_field) == {"T", "S"}
    # weighted output norm: an S-only norm sees only the (exactly linear) S part
    rows_s = AC.fdlin(J_map, x, v, [1e-1], tl, weights={"S": 1.0})
    assert rows_s[0].rel_err < 1e-12 and set(rows_s[0].per_field) == {"S"}
