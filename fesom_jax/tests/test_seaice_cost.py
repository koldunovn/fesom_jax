"""``fesom_jax.seaice_cost`` on synthetic data (no mesh): monthly accumulators driven by a per-chunk
month scalar, the month-straddle check, the cost against an independent numpy loop, the numpy
scorer against the jnp one, the gradient against central FD, masking and shape errors."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import adjoint_costs as AC
from fesom_jax import seaice_cost as S

NOD = 7


class Toy(NamedTuple):
    a_ice: jax.Array
    m_ice: jax.Array


# --- accumulators ------------------------------------------------------------------------------
def test_accumulators_follow_the_chunk_month_and_compile_once():
    traces = []

    def step(st, x):
        return Toy(st.a_ice + x, st.m_ice + 2.0 * x)

    @jax.jit
    def chunk(carry, xs):
        forcing, month = xs
        traces.append(1)
        wrapped = AC.with_running_costs(step, S.monthly_accumulators(month))
        carry, _ = jax.lax.scan(lambda c, x: (wrapped(c, x), None), carry, forcing)
        return carry

    st = Toy(jnp.zeros(NOD), jnp.ones(NOD))
    acc = AC.init_accumulators(S.monthly_accumulators(0), st)
    assert acc["a_mon"].shape == (12, NOD) and acc["n_mon"].shape == (12,)
    rng = np.random.default_rng(0)
    K = 4
    xs = rng.normal(size=(3, K, NOD))
    months = [0, 1, 1]
    carry = (st, acc)
    for c in range(3):
        carry = chunk(carry, (jnp.asarray(xs[c]), jnp.asarray(months[c], jnp.int32)))
    assert len(traces) == 1, "a new month must not recompile the chunk"
    st_f, acc = carry

    # reference: post-step states by hand
    a = np.zeros(NOD)
    m = np.ones(NOD)
    ref_a = np.zeros((12, NOD))
    ref_m = np.zeros((12, NOD))
    for c in range(3):
        for k in range(K):
            a = a + xs[c, k]
            m = m + 2.0 * xs[c, k]
            ref_a[months[c]] += a
            ref_m[months[c]] += m
    np.testing.assert_array_equal(np.asarray(acc["n_mon"]), [4, 8] + [0] * 10)
    np.testing.assert_allclose(np.asarray(acc["a_mon"]), ref_a, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(acc["m_mon"]), ref_m, rtol=1e-13, atol=1e-13)
    Cbar, Hbar, have = S.monthly_means(acc)
    np.testing.assert_allclose(np.asarray(Cbar[1]), ref_a[1] / 8, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(Hbar[0]), ref_m[0] / 4, rtol=1e-13)
    np.testing.assert_array_equal(np.asarray(have), [True, True] + [False] * 10)
    assert np.all(np.asarray(Cbar[2:]) == 0.0)


def test_chunk_months_whole_year_and_leap_february():
    m = S.chunk_months("2013-01-01T00:00", 1800.0, 24, 730)
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    np.testing.assert_array_equal(np.bincount(m, minlength=12), 2 * np.array(days))
    assert m.dtype == np.int32 and m[0] == 0 and m[-1] == 11
    m12 = S.chunk_months("2012-02-01T12:00", 1800.0, 24, 58)
    assert (m12[:57] == 1).all() and m12[57] == 2           # 29 days: 57 half-days left in Feb
    # the step that ends at 00:00 on the 1st belongs to the month before
    one = S.chunk_months("2013-01-31T12:00", 1800.0, 24, 2)
    np.testing.assert_array_equal(one, [0, 1])


@pytest.mark.parametrize("start, steps, n", [("2013-01-01T06:00", 24, 62),
                                            ("2013-01-01T00:00", 25, 60),
                                            ("2013-01-01T00:00", 48 * 31 + 1, 1)])
def test_chunk_months_refuses_a_straddling_chunk(start, steps, n):
    with pytest.raises(ValueError, match="straddles a month boundary"):
        S.chunk_months(start, 1800.0, steps, n)


def test_chunk_months_refuses_a_second_year():
    with pytest.raises(ValueError, match="one calendar year"):
        S.chunk_months("2013-12-31T00:00", 1800.0, 24, 3)


# --- the cost ------------------------------------------------------------------------------------
def _synthetic(seed=1):
    rng = np.random.default_rng(seed)
    nh = np.array([True, True, True, True, False, False, False])
    nC = rng.integers(0, 4, size=(12, NOD)).astype(np.int32)
    nH = rng.integers(0, 3, size=(12, NOD)).astype(np.int32)
    nH[:, ~nh] = 0                                           # CCI is Arctic
    nH[4:9] = 0                                              # no summer thickness
    C = np.where(nC > 0, rng.uniform(0, 1, (12, NOD)), np.nan)
    sC = np.where(nC > 0, rng.uniform(0.1, 0.2, (12, NOD)), np.nan)
    H = np.where(nH > 0, rng.uniform(-0.2, 3, (12, NOD)), np.nan)
    sH = np.where(nH > 0, rng.uniform(0.3, 1.0, (12, NOD)), np.nan)
    obs = {"C_obs": C, "sC": sC, "nC": nC, "H_obs": H, "sH": sH, "nH": nH, "nh": nh}
    n_mon = np.array([48, 96, 10, 0, 1, 5, 7, 0, 3, 4, 2, 8], dtype=np.float64)
    a_mon = rng.uniform(0, 1, (12, NOD)) * n_mon[:, None]
    m_mon = rng.uniform(0, 3, (12, NOD)) * n_mon[:, None]
    return obs, {"a_mon": a_mon, "m_mon": m_mon, "n_mon": n_mon}


def _reference(obs, acc):
    """Independent loop: weight n * 625 km^2 / (100 km)^2, months with steps only."""
    J = {"J_C_NH": 0.0, "J_C_SH": 0.0, "J_H_NH": 0.0}
    for m in range(12):
        if acc["n_mon"][m] == 0:
            continue
        for n in range(NOD):
            Cb = acc["a_mon"][m, n] / acc["n_mon"][m]
            Hb = acc["m_mon"][m, n] / acc["n_mon"][m]
            if obs["nC"][m, n] > 0:
                w = obs["nC"][m, n] * 625e6 / 1e10
                j = 0.5 * w * (Cb - obs["C_obs"][m, n]) ** 2 / obs["sC"][m, n] ** 2
                J["J_C_NH" if obs["nh"][n] else "J_C_SH"] += j
            if obs["nH"][m, n] > 0 and obs["nh"][n]:
                w = obs["nH"][m, n] * 625e6 / 1e10
                J["J_H_NH"] += 0.5 * w * (Hb - obs["H_obs"][m, n]) ** 2 / obs["sH"][m, n] ** 2
    return J


def test_cost_matches_the_numpy_reference():
    obs, acc = _synthetic()
    ob = S.cost_obs(obs)
    got = S.cost_terms({k: jnp.asarray(v) for k, v in acc.items()}, ob)
    ref = _reference(obs, acc)
    for t in S.TERMS:
        assert ref[t] > 0
        np.testing.assert_allclose(float(got[t]), ref[t], rtol=1e-12)
    np.testing.assert_allclose(float(got["J_data"]), sum(ref.values()), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(got["J_C_NH_month"]).sum(), ref["J_C_NH"], rtol=1e-12)
    # weight sums over the scored observations
    have = acc["n_mon"] > 0
    nC_nh = obs["nC"][have][:, obs["nh"]].sum()
    np.testing.assert_allclose(float(got["N_C_NH"]), nC_nh / 16.0, rtol=1e-12)
    # active weight sums: observation > 0 or model mean > 0
    Cb = acc["a_mon"] / np.where(have, acc["n_mon"], 1.0)[:, None]
    Hb = acc["m_mon"] / np.where(have, acc["n_mon"], 1.0)[:, None]
    for t, n_key, o_key, mod, hem in (("C_NH", "nC", "C_obs", Cb, obs["nh"]),
                                      ("C_SH", "nC", "C_obs", Cb, ~obs["nh"]),
                                      ("H_NH", "nH", "H_obs", Hb, obs["nh"])):
        ref_act = 0.0
        for m in np.flatnonzero(have):
            for n in np.flatnonzero(hem):
                if obs[n_key][m, n] and (obs[o_key][m, n] > 0 or mod[m, n] > 0):
                    ref_act += obs[n_key][m, n] * 625e6 / 1e10
        np.testing.assert_allclose(float(got["Nact_" + t]), ref_act, rtol=1e-12, err_msg=t)
        assert float(got["Nact_" + t]) <= float(got["N_" + t])


def test_inactive_zeros_count_in_n_but_not_in_nact():
    obs, acc = _synthetic(7)
    obs = {k: v.copy() for k, v in obs.items()}
    acc = {k: v.copy() for k, v in acc.items()}
    m, n = 0, 0                                               # a far-field zero: obs 0, model 0
    obs["C_obs"][m, n], obs["nC"][m, n], obs["sC"][m, n] = 0.0, 16, 0.1
    acc["a_mon"][m, n] = 0.0
    base = S.cost_terms_np(acc["a_mon"] / acc["n_mon"][:, None].clip(1),
                           acc["m_mon"] / acc["n_mon"][:, None].clip(1), S.cost_obs(obs))
    obs["nC"][m, n] = 0
    obs["C_obs"][m, n] = obs["sC"][m, n] = np.nan
    less = S.cost_terms_np(acc["a_mon"] / acc["n_mon"][:, None].clip(1),
                           acc["m_mon"] / acc["n_mon"][:, None].clip(1), S.cost_obs(obs))
    np.testing.assert_allclose(base["N_C_NH"] - less["N_C_NH"], 1.0, rtol=1e-12)
    assert base["Nact_C_NH"] == less["Nact_C_NH"] and base["J_C_NH"] == less["J_C_NH"]


def test_numpy_scorer_equals_the_jnp_cost():
    obs, acc = _synthetic(2)
    ob = S.cost_obs(obs)
    got = S.cost_terms({k: jnp.asarray(v) for k, v in acc.items()}, ob)
    have = acc["n_mon"] > 0
    n_safe = np.where(have, acc["n_mon"], 1.0)[:, None]
    Cm = acc["a_mon"] / n_safe
    Hm = acc["m_mon"] / n_safe
    Cm[~have] = np.nan                                       # unscored rows may be NaN
    Hm[~have] = np.nan
    ref = S.cost_terms_np(Cm, Hm, ob, months=np.flatnonzero(have))
    for k, v in ref.items():
        np.testing.assert_allclose(np.asarray(got[k]), v, rtol=1e-13, atol=0, err_msg=k)
    assert isinstance(ref["J_data"], float)


def test_gradient_wrt_the_accumulators_equals_central_fd():
    obs, acc = _synthetic(3)
    ob = S.cost_obs(obs)
    acc_j = {k: jnp.asarray(v) for k, v in acc.items()}

    def J(a, m):
        return S.cost_terms({"a_mon": a, "m_mon": m, "n_mon": acc_j["n_mon"]}, ob)["J_data"]

    ga, gm = jax.grad(J, argnums=(0, 1))(acc_j["a_mon"], acc_j["m_mon"])
    rng = np.random.default_rng(4)
    da = rng.normal(size=(12, NOD))
    dm = rng.normal(size=(12, NOD))
    h = 1e-4
    fd = (J(acc_j["a_mon"] + h * da, acc_j["m_mon"] + h * dm)
          - J(acc_j["a_mon"] - h * da, acc_j["m_mon"] - h * dm)) / (2 * h)
    adj = float(jnp.sum(ga * da) + jnp.sum(gm * dm))
    assert abs(adj) > 1e-3
    np.testing.assert_allclose(adj, float(fd), rtol=1e-8)
    # entries without an observation, and months without steps, get no gradient
    gC = np.asarray(ga)
    no_obs = ~(obs["nC"] > 0)
    assert np.all(gC[no_obs] == 0.0) and np.all(gC[acc["n_mon"] == 0] == 0.0)
    assert np.any(gC[(obs["nC"] > 0) & (acc["n_mon"] > 0)[:, None]] != 0.0)


def test_masked_nodes_and_months_do_not_contribute():
    obs, acc = _synthetic(5)
    ob = S.cost_obs(obs)
    base = S.cost_terms({k: jnp.asarray(v) for k, v in acc.items()}, ob)
    acc2 = {k: v.copy() for k, v in acc.items()}
    no_obs = (obs["nC"] == 0) & (obs["nH"] == 0)
    acc2["a_mon"][no_obs] += 50.0
    acc2["m_mon"][no_obs] -= 50.0
    acc2["m_mon"][:, ~obs["nh"]] += 9.0                      # southern thickness is never scored
    empty = acc["n_mon"] == 0
    acc2["a_mon"][empty] = 0.0                               # sums of months without steps
    got = S.cost_terms({k: jnp.asarray(v) for k, v in acc2.items()}, ob)
    for t in S.TERMS:
        assert float(got[t]) == float(base[t])
    # a NaN model value where nothing is observed is harmless in the numpy scorer too
    Cm = np.where(no_obs, np.nan, 0.3)
    r = S.cost_terms_np(Cm, np.where(no_obs, np.nan, 1.0), ob)
    assert np.isfinite(r["J_data"])


def test_wrong_shapes_raise():
    obs, acc = _synthetic(6)
    ob = S.cost_obs(obs)
    with pytest.raises(ValueError, match="C mean"):
        S.cost_terms_np(np.zeros((12, NOD + 1)), np.zeros((12, NOD)), ob)
    with pytest.raises(ValueError, match="H mean"):
        S.cost_terms_np(np.zeros((12, NOD)), np.zeros((11, NOD)), ob)
    bad = dict(acc, a_mon=acc["a_mon"][:, :-1])
    with pytest.raises(ValueError, match="C mean"):
        S.cost_terms({k: jnp.asarray(v) for k, v in bad.items()}, ob)
    with pytest.raises(ValueError, match=r"\[12, nod2D\]"):
        S.cost_obs({k: (v[:6] if k != "nh" else v) for k, v in obs.items()})
    with pytest.raises(ValueError, match="nh"):
        S.cost_obs(dict(obs, nh=obs["nh"][:-1]))
