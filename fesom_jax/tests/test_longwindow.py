"""Long-window Task B0 — the ensemble-averaged adjoint seam (:mod:`fesom_jax.longwindow`).

The CPU unit test reproduces the **Lea, Allen & Haine (2000)** benchmark on **Lorenz-63**: the
climate sensitivity ``d⟨z⟩/dρ`` (of the long-time-mean of ``z`` to the forcing parameter ``ρ``).

* **A single long adjoint DIVERGES** — ``|d⟨z⟩_window/dρ|`` grows exponentially with the window
  length (the chaotic tangent-linear blow-up): meaningless from one long backward pass.
* **The ENSEMBLE AVERAGE of SHORT adjoints CONVERGES** — averaging ``d⟨z⟩_window/dρ`` over many
  short bursts seeded along the attractor recovers the finite-difference truth (``≈1``) within
  the Lea-et-al O(25%) tolerance; the running mean stabilizes as the burst count grows.

Plus the seam's host-side machinery (seed spreading, masked streaming mean + standard error,
the convergence diagnostic). Token: LONGWINDOW_SEAM_OK. The only library-surface task of the
plan — ``params=None`` invariant untouched (this module imports nothing from ``step``).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from fesom_jax import longwindow as lw   # noqa: E402


# ==========================================================================
# Seam machinery — seed spreading / streaming average / convergence
# ==========================================================================
def test_spread_indices_range_and_spread():
    idx = lw.spread_indices(1000, 10)
    assert idx.shape == (10,)
    assert idx.min() >= 0 and idx.max() < 1000
    assert np.all(np.diff(idx) > 0)                    # strictly spread out
    # evenly spaced bin-centres are deterministic
    assert np.array_equal(idx, lw.spread_indices(1000, 10))


def test_spread_indices_edge_cases():
    assert lw.spread_indices(5, 0).size == 0
    assert np.array_equal(lw.spread_indices(3, 10), np.arange(3))   # K≥n → all
    rng = np.random.default_rng(0)
    j = lw.spread_indices(1000, 20, rng)
    assert j.min() >= 0 and j.max() < 1000 and j.size == 20


def test_seed_starts_from_list_and_stack():
    states = [jnp.array([float(i), 2.0 * i]) for i in range(100)]
    picked = lw.seed_starts(states, 5)
    assert len(picked) == 5
    # pytree with a leading time axis
    stack = jnp.arange(300.0).reshape(100, 3)
    picked2 = lw.seed_starts(stack, 4)
    assert len(picked2) == 4 and picked2[0].shape == (3,)


def test_average_grads_masks_nonfinite_and_reports_stderr():
    # scalar bursts around 2.0, with one NaN burst that must be masked out
    bursts = [2.0, 2.2, np.nan, 1.8, 2.0]
    stats, n = lw.average_grads(iter(bursts))
    assert n == 5
    assert np.isfinite(float(stats["mean"]))
    assert float(stats["mean"]) == pytest.approx(np.mean([2.0, 2.2, 1.8, 2.0]))
    assert float(stats["count"]) == 4                  # the NaN burst excluded
    assert float(stats["stderr"]) > 0.0


def test_average_grads_pytree_leafwise():
    # each burst is a dict pytree; reduced leafwise
    bursts = [{"a": np.array([1.0, 2.0]), "b": 10.0},
              {"a": np.array([3.0, 4.0]), "b": 20.0}]
    stats, n = lw.average_grads(bursts)
    assert n == 2
    assert np.allclose(np.asarray(stats["mean"]["a"]), [2.0, 3.0])
    assert float(stats["mean"]["b"]) == pytest.approx(15.0)


def test_average_grads_empty():
    stats, n = lw.average_grads(iter([]))
    assert stats is None and n == 0


def test_convergence_running_mean_stabilizes():
    rng = np.random.default_rng(1)
    g = 1.0 + 0.3 * rng.standard_normal(500)           # noisy around 1.0
    c = lw.convergence(g, tol=0.05)
    assert c["running_mean"].shape == (500,)
    assert c["final"] == pytest.approx(np.mean(g))
    assert 0 < c["n_stable"] <= 500
    # the running mean's tail is close to the final (stabilized)
    assert abs(c["running_mean"][-1] - 1.0) < 0.1
    # masked: a NaN burst doesn't poison the running mean
    g2 = g.copy(); g2[100] = np.nan
    assert np.isfinite(lw.convergence(g2)["running_mean"]).all()


# ==========================================================================
# Lorenz-63 — the Lea/Allen/Haine 2000 ensemble-adjoint benchmark
# ==========================================================================
SIGMA, BETA, DT, RHO = 10.0, 8.0 / 3.0, 0.01, 28.0


def _rhs(s, rho):
    x, y, z = s[0], s[1], s[2]
    return jnp.array([SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z])


def _rk4(s, rho):
    k1 = _rhs(s, rho); k2 = _rhs(s + 0.5 * DT * k1, rho)
    k3 = _rhs(s + 0.5 * DT * k2, rho); k4 = _rhs(s + DT * k3, rho)
    return s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _integ(s0, rho, n):
    def body(s, _):
        s2 = _rk4(s, rho)
        return s2, s2[2]
    return jax.lax.scan(body, s0, None, length=n)       # (final_state, z_series)


def _window_zmean(s0, rho, n):
    return jnp.mean(_integ(s0, rho, n)[1])


_grad_z = jax.jit(jax.grad(_window_zmean, argnums=1), static_argnums=2)


def _traj_full(s0, rho, n):
    """ONE scan emitting the full state each step (the reference trajectory) — fast."""
    def body(s, _):
        s2 = _rk4(s, rho)
        return s2, s2
    return jax.lax.scan(body, s0, None, length=n)


_traj_full = jax.jit(_traj_full, static_argnums=2)


@pytest.fixture(scope="module")
def lorenz():
    """Spin up onto the attractor, then a reference trajectory (states every 20 steps) + the
    finite-difference truth d⟨z⟩/dρ over long trajectories. (~6 s on CPU.)"""
    sf = _traj_full(jnp.array([1.0, 1.0, 1.0]), RHO, 10000)[0]    # discard the T=100 transient
    ref = np.asarray(_traj_full(sf, RHO, 100000)[1])[::20]        # 5000 states, every 20 steps
    # FD truth: central difference of the long-time-mean z at ρ±δ (a few ICs averaged)
    dlt = 1.0
    fd = []
    for ic in ref[::1500][:3]:
        sp = _traj_full(jnp.asarray(ic), RHO + dlt, 3000)[0]      # re-equilibrate at ρ±δ
        sm = _traj_full(jnp.asarray(ic), RHO - dlt, 3000)[0]
        zp = float(jnp.mean(_integ(sp, RHO + dlt, 150000)[1]))
        zm = float(jnp.mean(_integ(sm, RHO - dlt, 150000)[1]))
        fd.append((zp - zm) / (2 * dlt))
    return dict(ref=ref, fd_truth=float(np.mean(fd)))


def test_single_long_adjoint_diverges(lorenz):
    """One long-window adjoint blows up: |grad| grows ~exponentially with the window."""
    seed = jnp.asarray(lorenz["ref"][1234])
    g_short = abs(float(_grad_z(seed, RHO, 50)))        # T=0.5
    g_long = abs(float(_grad_z(seed, RHO, 2000)))       # T=20
    assert np.isfinite(g_long)
    assert g_long > 50.0                                # astronomically larger than the climate ~1
    assert g_long > 30.0 * max(g_short, 1e-3)           # grows with the window


def test_ensemble_average_recovers_climate_sensitivity(lorenz):
    """The ensemble average of SHORT adjoints recovers the FD truth within Lea's O(25%),
    while the running mean stabilizes (divergence-then-convergence)."""
    ref, fd_truth = lorenz["ref"], lorenz["fd_truth"]
    assert fd_truth == pytest.approx(1.0, abs=0.35)     # the known Lorenz climate sensitivity ~1

    N_SHORT, K = 50, 4000                               # T=0.5 burst, 4000 attractor seeds
    seeds = jnp.asarray(ref[lw.spread_indices(len(ref), K)])
    bursts = np.asarray(jax.vmap(lambda s: _grad_z(s, RHO, N_SHORT))(seeds))

    stats, n = lw.average_grads(bursts)
    ens_mean = float(stats["mean"])
    assert n == K and np.isfinite(ens_mean)
    rel_err = abs(ens_mean - fd_truth) / abs(fd_truth)
    assert rel_err < 0.25, f"ensemble adjoint {ens_mean:.3f} vs FD {fd_truth:.3f} ({rel_err:.0%})"

    # convergence: the running mean settles well before all K bursts
    c = lw.convergence(bursts, tol=0.05)
    assert c["n_stable"] < K
    assert abs(c["final"] - ens_mean) < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
