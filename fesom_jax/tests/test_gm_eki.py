"""D2b seam: GM→T/S stratification calibration via EKI (the slow-target pillar).

CPU unit tests guarding the two pieces of machinery the GPU driver
(``scripts/paper/core2_paper_calib_gm_eki.py``) wires together — with NO full model (forward-only,
analytic surrogate):

  1. :func:`fesom_jax.obs_compare.basin_mean_profiles` — the area-weighted basin-mean profile
     reduction (the low-dim stratification observable) vs an independent loop reference, plus
     the AD masked-NaN / empty-basin guarantees (empty ⇒ finite 0, never nan).
  2. The driver's EKI wiring: recover a planted ``k_gm`` through the basin reduction with the
     **log-parameter** + **sequential** ensemble evaluation (:func:`fesom_jax.eki.sequential_eval`,
     the memory-safe ``map_fn`` for a heavy forward) — exactly the driver's call shape, on a cheap
     surrogate forward.

Token: GM_EKI_SEAM_OK.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import eki, obs_compare

RNG = np.random.default_rng(20260615)


# --------------------------------------------------------------------------
# basin_mean_profiles — area-weighted basin-mean vertical profiles
# --------------------------------------------------------------------------
def _ref_basin(cell_field, cell_valid, basin_weight):
    """Independent triple-loop reference for the basin reduction."""
    nb, nc = basin_weight.shape
    nd = cell_field.shape[1]
    out = np.zeros((nb, nd))
    for b in range(nb):
        for d in range(nd):
            w = basin_weight[b] * cell_valid[:, d]
            s = w.sum()
            out[b, d] = (w * cell_field[:, d]).sum() / s if s > 0 else 0.0
    return out


def test_basin_mean_profiles_matches_reference():
    nc, nd, nb = 60, 5, 4
    cf = RNG.standard_normal((nc, nd)) * 3.0 + 10.0
    cv = (RNG.uniform(size=(nc, nd)) > 0.2).astype(np.float64)     # ~80% valid
    bw = np.zeros((nb, nc))
    assign = RNG.integers(0, nb, size=nc)                          # each cell → one basin
    area = RNG.uniform(0.3, 1.0, size=nc)
    for c in range(nc):
        bw[assign[c], c] = area[c]
    got = np.asarray(obs_compare.basin_mean_profiles(
        jnp.asarray(cf), jnp.asarray(cv), jnp.asarray(bw)))
    ref = _ref_basin(cf, cv, bw)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_basin_mean_profiles_empty_is_finite_zero():
    """An empty basin (no weight) and an all-invalid depth → finite 0, never nan (masked-NaN rule)."""
    nc, nd, nb = 20, 3, 3
    cf = RNG.standard_normal((nc, nd))
    cf[5, 1] = np.nan                                              # a NaN at an invalidated lane
    cv = np.ones((nc, nd))
    cv[:, 1] = 0.0                                                 # depth 1 entirely invalid
    bw = np.zeros((nb, nc))
    bw[0] = RNG.uniform(0.2, 1.0, size=nc)                         # basin 0 populated
    # basin 1 left all-zero (empty); basin 2 populated
    bw[2] = RNG.uniform(0.2, 1.0, size=nc)
    out = np.asarray(obs_compare.basin_mean_profiles(
        jnp.asarray(np.nan_to_num(cf, nan=0.0)), jnp.asarray(cv), jnp.asarray(bw)))
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[1], np.zeros(nd))           # empty basin → 0
    np.testing.assert_array_equal(out[:, 1], np.zeros(nb))        # all-invalid depth → 0


def test_basin_mean_profiles_gradient_flows():
    """d(profile)/d(cell_field) is finite & nonzero on valid lanes (the observable is
    differentiable — the adjoint↔EKI cross-check / any adjoint use needs this)."""
    nc, nd, nb = 30, 4, 2
    cv = np.ones((nc, nd))
    bw = np.zeros((nb, nc))
    bw[0, :15] = 1.0
    bw[1, 15:] = 1.0

    def scalar(cf):
        return jnp.sum(obs_compare.basin_mean_profiles(cf, jnp.asarray(cv), jnp.asarray(bw)))

    g = np.asarray(jax.grad(scalar)(jnp.asarray(RNG.standard_normal((nc, nd)))))
    assert np.all(np.isfinite(g))
    assert np.count_nonzero(g) == nc * nd                         # every valid cell contributes


# --------------------------------------------------------------------------
# sequential_eval — the memory-safe ensemble map (== vmap, just looped)
# --------------------------------------------------------------------------
def test_sequential_eval_matches_vmap():
    def G(theta):
        return jnp.array([theta[0] + theta[1], theta[0] * theta[1]])

    thetas = jnp.asarray(RNG.standard_normal((7, 2)))
    seq = np.asarray(eki.sequential_eval(G, thetas))
    vm = np.asarray(jax.vmap(G)(thetas))
    np.testing.assert_allclose(seq, vm, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# the driver's EKI wiring: recover k_gm through the basin reduction, log-param, sequential
# --------------------------------------------------------------------------
def _surrogate_gm(nc=48, nd=4, nb=3):
    """A cheap surrogate of the GM observable: per-cell T/S linear in k_gm (mimicking GM heat
    redistribution), reduced to basin-mean profiles. Returns ``model_obs(k)->[d]`` and the
    fixed basin weights / mask."""
    cv = np.ones((nc, nd))
    bw = np.zeros((nb, nc))
    assign = RNG.integers(0, nb, size=nc)
    for c in range(nc):
        bw[assign[c], c] = RNG.uniform(0.3, 1.0)
    baseT = jnp.asarray(RNG.uniform(2.0, 18.0, (nc, nd)))
    baseS = jnp.asarray(RNG.uniform(34.0, 35.0, (nc, nd)))
    gradT = jnp.asarray(RNG.uniform(-1.0, 1.0, (nc, nd)))         # dT/d(k/1000)
    gradS = jnp.asarray(RNG.uniform(-0.2, 0.2, (nc, nd)))
    cvj, bwj = jnp.asarray(cv), jnp.asarray(bw)

    def model_obs(k):
        u = k / 1000.0
        cT = baseT + gradT * u
        cS = baseS + gradS * u
        pT = obs_compare.basin_mean_profiles(cT, cvj, bwj)
        pS = obs_compare.basin_mean_profiles(cS, cvj, bwj)
        return jnp.concatenate([pT.reshape(-1), pS.reshape(-1)])

    return model_obs, nb, nd


def test_gm_eki_recovers_planted_kgm():
    """Plant k_gm=1500, observe its basin-mean T/S, recover from a prior centred at 1000 via EKI
    on log(k_gm) with the sequential map — the driver's exact wiring."""
    model_obs, nb, nd = _surrogate_gm()
    truth = 1500.0
    y_obs = model_obs(jnp.asarray(truth, jnp.float64))
    d = int(y_obs.shape[0])
    gamma = jnp.asarray(np.r_[np.full(nb * nd, 0.02 ** 2), np.full(nb * nd, 0.005 ** 2)])

    J = 64
    theta0 = jnp.asarray((np.log(1000.0) + 0.3 * RNG.standard_normal((J, 1))))

    mean, history, ens = eki.eki_run(
        theta0, lambda th: model_obs(jnp.exp(th[0])), y_obs, gamma,
        n_iters=12, key=jax.random.PRNGKey(0), perturb_obs=False,
        map_fn=eki.sequential_eval)
    k_rec = float(np.exp(np.asarray(mean)[0]))
    assert abs(k_rec - truth) / truth < 0.05                     # within 5% of the planted truth
    assert history[-1]["misfit"] < 1e-2 * history[0]["misfit"]   # misfit collapses


def test_gm_eki_seam_ok_token():
    """Aggregate gate — prints GM_EKI_SEAM_OK."""
    model_obs, nb, nd = _surrogate_gm()
    truth = 1300.0
    y_obs = model_obs(jnp.asarray(truth, jnp.float64))
    gamma = jnp.asarray(np.r_[np.full(nb * nd, 0.02 ** 2), np.full(nb * nd, 0.005 ** 2)])
    theta0 = jnp.asarray((np.log(900.0) + 0.3 * RNG.standard_normal((48, 1))))
    mean, history, _ = eki.eki_run(
        theta0, lambda th: model_obs(jnp.exp(th[0])), y_obs, gamma,
        n_iters=12, key=jax.random.PRNGKey(1), perturb_obs=False, map_fn=eki.sequential_eval)
    k_rec = float(np.exp(np.asarray(mean)[0]))
    assert abs(k_rec - truth) / truth < 0.05
    assert np.all(np.isfinite(np.asarray(mean)))
    print("GM_EKI_SEAM_OK")
