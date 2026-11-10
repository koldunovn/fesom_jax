"""Conditioned controls (fesom_jax.controls): log-normal transforms, prior, chain rule, domain check.

Unit tests, no mesh. The chain-rule test is the one that matters for calibration: the u-space
gradient JAX produces through ``ControlSet.to_params`` must equal ``sigma·p·dJ/dp`` computed from
an analytic dJ/dp — if it did not, every optimiser step in u would be mis-scaled.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax.controls import (ICE_THERMO_FIELDS, SIGMA_DEFAULT, TKE_FIELDS, U_MAX, ControlSet,
                                LogNormalControl, ice_thermo_control_set, tke_control_set)
from fesom_jax.params import Params


def _analytic_J(p):
    """A smooth nonlinear cost of the four TKE constants and its analytic gradient."""
    ck, ceps, cd, al = p
    J = ck ** 2 * ceps + jnp.sin(cd) + jnp.log(al) * ck
    dJ = (2 * ck * ceps + jnp.log(al), ck ** 2, jnp.cos(cd), ck / al)
    return J, dJ


def _J_of_params(pr: Params):
    J, _ = _analytic_J((pr.tke_c_k, pr.tke_c_eps, pr.tke_cd, pr.tke_alpha))
    return J


def test_defaults_are_the_config_constants_and_ln2():
    cs = tke_control_set()
    d = Params.defaults()
    assert cs.names == TKE_FIELDS
    assert np.array_equal(cs.p0, np.array([0.1, 0.7, 3.75, 30.0]))
    assert np.array_equal(cs.p0, np.array([float(getattr(d, f)) for f in TKE_FIELDS]))
    assert np.all(cs.sigma == math.log(2.0)) and SIGMA_DEFAULT == math.log(2.0)


def test_u_zero_gives_the_first_guess_bitwise():
    cs = tke_control_set()
    pr = cs.to_params(jnp.zeros(4, jnp.float64))
    d = Params.defaults()
    for f in dataclasses.fields(Params):
        a, b = getattr(pr, f.name), getattr(d, f.name)
        if a is None:
            assert b is None
            continue
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), f.name


def test_round_trip_u_p():
    cs = tke_control_set(sigma=[0.3, 0.7, 1.1, 0.05])
    rng = np.random.default_rng(0)
    u = rng.normal(size=4) * 3.0
    p = np.asarray(cs.to_physical(u))
    np.testing.assert_allclose(cs.to_u(p), u, rtol=0, atol=1e-12)
    p2 = np.array([0.24, 0.35, 5.0, 12.0])
    np.testing.assert_allclose(np.asarray(cs.to_physical(cs.to_u(p2))), p2, rtol=1e-14)
    # one control at a time agrees with the vector form
    for i, c in enumerate(cs.controls):
        assert float(c.to_physical(u[i])) == pytest.approx(p[i], rel=1e-15)


def test_to_params_sets_only_the_controlled_leaves():
    cs = tke_control_set()
    u = jnp.asarray([0.5, -1.0, 2.0, -0.25], jnp.float64)
    base = dataclasses.replace(Params.defaults(), k_ver=jnp.asarray(3e-5, jnp.float64))
    pr = cs.to_params(u, base=base)
    p = np.asarray(cs.to_physical(u))
    for i, f in enumerate(TKE_FIELDS):
        assert float(getattr(pr, f)) == pytest.approx(p[i], rel=1e-15)
        assert getattr(pr, f).dtype == jnp.float64 and not getattr(pr, f).weak_type
    assert float(pr.k_ver) == 3e-5
    assert float(pr.k_gm) == float(Params.defaults().k_gm)


def test_gradient_in_u_is_sigma_p_dJdp():
    cs = tke_control_set(sigma=[0.4, SIGMA_DEFAULT, 1.3, 0.2])
    u = jnp.asarray([0.3, -0.7, 0.9, 1.5], jnp.float64)
    g_u = np.asarray(jax.grad(lambda v: _J_of_params(cs.to_params(v)))(u))
    p = np.asarray(cs.to_physical(u))
    _, dJ = _analytic_J(tuple(jnp.asarray(x) for x in p))
    expect = cs.sigma * p * np.array([float(x) for x in dJ])
    np.testing.assert_allclose(g_u, expect, rtol=1e-13, atol=0)
    np.testing.assert_allclose(cs.grad_u(np.asarray(u), [float(x) for x in dJ]), expect,
                               rtol=1e-13, atol=0)
    # negative control: the same gradient with p in place of sigma·p is NOT equal
    assert not np.allclose(g_u, p * np.array([float(x) for x in dJ]), rtol=1e-6)


def test_prior_value_and_gradient():
    u = jnp.asarray([0.3, -0.7, 0.9, 1.5], jnp.float64)
    assert float(ControlSet.prior(u)) == pytest.approx(0.5 * float(jnp.sum(u * u)), rel=1e-15)
    np.testing.assert_array_equal(np.asarray(jax.grad(ControlSet.prior)(u)), np.asarray(u))
    H = np.asarray(jax.hessian(ControlSet.prior)(u))
    np.testing.assert_array_equal(H, np.eye(4))


@pytest.mark.parametrize("sigma", [SIGMA_DEFAULT, 1.0, 3.0])
def test_positive_and_finite_inside_the_domain(sigma):
    cs = tke_control_set(sigma=sigma)
    for val in (-U_MAX, U_MAX):
        u = np.full(4, val)
        cs.check(u)                                  # inside the domain: no raise
        p = np.asarray(cs.to_physical(u))
        assert np.all(np.isfinite(p)) and np.all(p > 0.0)
        g = np.asarray(jax.grad(lambda v: _J_of_params(cs.to_params(v)))(jnp.asarray(u)))
        assert np.all(np.isfinite(g))


@pytest.mark.parametrize("bad", [
    [U_MAX + 1e-9, 0.0, 0.0, 0.0],
    [0.0, -U_MAX - 1.0, 0.0, 0.0],
    [np.nan, 0.0, 0.0, 0.0],
    [0.0, 0.0, np.inf, 0.0],
    [0.0, 0.0, 0.0],                                 # wrong shape
])
def test_check_raises_outside_the_domain(bad):
    with pytest.raises(ValueError):
        tke_control_set().check(np.asarray(bad, np.float64))


def test_controlset_is_hashable_ordered_and_a_static_jit_arg():
    a = tke_control_set()
    b = tke_control_set(sigma=np.float64(math.log(2.0)), p0=[np.float32(0.1).item(), 0.7, 3.75, 30])
    c = ControlSet(tuple(reversed(a.controls)))
    assert hash(a) == hash(tke_control_set()) and a == tke_control_set()
    assert a.names == ("tke_c_k", "tke_c_eps", "tke_cd", "tke_alpha")
    assert c.names == tuple(reversed(a.names)) and c != a
    assert {a: 1}[tke_control_set()] == 1
    assert b != a                                    # float32-rounded p0 is a different spec

    calls = []

    @jax.jit
    def f(u):
        calls.append(1)                              # counts traces, not calls
        return _J_of_params(a.to_params(u))

    g = jax.jit(lambda cs, u: _J_of_params(cs.to_params(u)), static_argnums=0)
    u = jnp.zeros(4, jnp.float64)
    assert float(g(a, u)) == float(g(tke_control_set(), u + 0.0))
    f(u); f(u + 1.0)
    assert len(calls) == 1


def test_unknown_or_invalid_controls_raise():
    with pytest.raises(ValueError, match="not a scalar Params field"):
        LogNormalControl("tke_ck", 0.1, 0.5)
    with pytest.raises(ValueError, match="not a scalar Params field"):
        LogNormalControl("tke_nn", 1.0, 0.5)
    with pytest.raises(ValueError):
        LogNormalControl("tke_c_k", 0.0, 0.5)
    with pytest.raises(ValueError):
        LogNormalControl("tke_c_k", 0.1, -1.0)
    c = LogNormalControl("tke_c_k", 0.1, 0.5)
    with pytest.raises(ValueError, match="duplicate"):
        ControlSet((c, c))
    with pytest.raises(ValueError):
        ControlSet(())
    with pytest.raises(ValueError):
        c.to_u(-1.0)
    with pytest.raises(ValueError):
        tke_control_set().to_params(jnp.zeros(3, jnp.float64))


# ---------------------------------------------------------------------------------------------
# sea-ice controls: ice.<IceConfig field>, collected into Params.ice_ctrl

#: The control table of the ice calibration plan (Technical Details), value by value:
#: (name, p0, sigma in ln p, p at u = -1, p at u = +1). The albedo sigma is 0.05/p0 (table to 3
#: significant figures), the others ln 2.
ICE_TABLE = [
    ("ice.albsn", 0.81, 0.0617, 0.762, 0.862),
    ("ice.albsnm", 0.77, 0.0649, 0.722, 0.822),
    ("ice.albi", 0.70, 0.0714, 0.652, 0.752),
    ("ice.albim", 0.68, 0.0735, 0.632, 0.732),
    ("ice.con", 2.1656, 0.693, 1.08, 4.33),
    ("ice.consn", 0.31, 0.693, 0.155, 0.62),
    ("ice.h0", 0.5, 0.693, 0.25, 1.0),
    ("ice.h0_s", 0.5, 0.693, 0.25, 1.0),
    ("ice.c_melt", 0.5, 0.693, 0.25, 1.0),
    ("ice.st_oce_ice", 0.006, 0.693, 0.003, 0.012),
]


def test_ice_thermo_set_matches_the_table():
    cs = ice_thermo_control_set()
    assert cs.names == tuple(r[0] for r in ICE_TABLE)
    assert cs.names == tuple("ice." + f for f in ICE_THERMO_FIELDS)
    for c, (name, p0, sig, pm, pp) in zip(cs.controls, ICE_TABLE):
        assert c.p0 == p0, name
        assert float(f"{c.sigma:.3g}") == sig, name         # the table's 3 significant figures
        if name.startswith("ice.alb"):
            assert c.sigma == 0.05 / p0, name
        else:
            assert c.sigma == math.log(2.0), name
        lo, hi = (float(c.to_physical(jnp.asarray(x))) for x in (-1.0, 1.0))
        assert float(f"{lo:.3g}") == pm, (name, lo)
        assert float(f"{hi:.3g}") == pp, (name, hi)
        if name.startswith("ice.alb"):
            assert hi - p0 == pytest.approx(0.05, rel=0.05), name


def test_ice_round_trip_u_p():
    cs = ice_thermo_control_set()
    u = np.linspace(-2.0, 2.0, cs.n)
    p = np.asarray(cs.to_physical(u))
    np.testing.assert_allclose(cs.to_u(p), u, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(np.asarray(cs.to_physical(np.zeros(cs.n))), cs.p0)


def _ice_J(d):
    """A smooth nonlinear cost of the ten ice controls (dict) and its analytic gradient."""
    x = [d[f] for f in ICE_THERMO_FIELDS]
    J = sum((k + 1.0) * jnp.sin(3.0 * v) + v * v * x[(k + 1) % 10] for k, v in enumerate(x))
    dJ = [(k + 1.0) * 3.0 * jnp.cos(3.0 * x[k]) + 2.0 * x[k] * x[(k + 1) % 10]
          + x[(k - 1) % 10] ** 2 for k in range(10)]
    return J, dJ


def test_ice_gradient_in_u_is_sigma_p_dJdp():
    cs = ice_thermo_control_set()
    u = jnp.asarray(np.linspace(-1.5, 1.2, cs.n), jnp.float64)
    g_u = np.asarray(jax.grad(lambda v: _ice_J(cs.to_params(v).ice_ctrl)[0])(u))
    p = np.asarray(cs.to_physical(u))
    _, dJ = _ice_J({f: jnp.asarray(p[i]) for i, f in enumerate(ICE_THERMO_FIELDS)})
    dJ = np.array([float(x) for x in dJ])
    np.testing.assert_allclose(g_u, cs.sigma * p * dJ, rtol=1e-12, atol=0)
    np.testing.assert_allclose(cs.grad_u(np.asarray(u), dJ), cs.sigma * p * dJ, rtol=1e-13)


def test_to_params_builds_the_ice_ctrl_dict():
    cs = ice_thermo_control_set()
    u = jnp.asarray(np.linspace(-1.0, 1.0, cs.n), jnp.float64)
    base = dataclasses.replace(Params.defaults(), k_ver=jnp.asarray(3e-5, jnp.float64))
    pr = cs.to_params(u, base=base)
    p = np.asarray(cs.to_physical(u))
    assert isinstance(pr.ice_ctrl, dict) and sorted(pr.ice_ctrl) == sorted(ICE_THERMO_FIELDS)
    for i, f in enumerate(ICE_THERMO_FIELDS):
        v = pr.ice_ctrl[f]
        assert float(v) == pytest.approx(p[i], rel=1e-15)
        assert v.dtype == jnp.float64 and not v.weak_type and v.shape == ()
    # the scalar leaves are left alone, and nothing else is set
    for f in dataclasses.fields(Params):
        if f.name != "ice_ctrl":
            a, b = getattr(pr, f.name), getattr(base, f.name)
            assert (a is None and b is None) or float(a) == float(b), f.name
    assert pr.ice_pstar is None and pr.ice_cd is None
    assert len(jax.tree.leaves(pr)) == len(jax.tree.leaves(base)) + cs.n


def test_mixed_set_and_existing_ice_ctrl():
    """TKE and ice controls in one set: the TKE names replace their fields, the ice names go
    into the dict, merged over the base's own ice_ctrl."""
    cs = ControlSet((LogNormalControl("tke_c_k", 0.1, 0.5),
                     LogNormalControl("ice.albsn", 0.81, 0.05 / 0.81),
                     LogNormalControl("ice_pstar", 3e4, 0.3)))
    base = dataclasses.replace(Params.defaults(),
                               ice_ctrl={"h_ml": jnp.asarray(2.0, jnp.float64)})
    u = jnp.asarray([0.2, -0.4, 0.1], jnp.float64)
    pr = cs.to_params(u, base=base)
    p = np.asarray(cs.to_physical(u))
    assert float(pr.tke_c_k) == pytest.approx(p[0], rel=1e-15)
    assert float(pr.ice_pstar) == pytest.approx(p[2], rel=1e-15)
    assert sorted(pr.ice_ctrl) == ["albsn", "h_ml"]
    assert float(pr.ice_ctrl["albsn"]) == pytest.approx(p[1], rel=1e-15)
    assert float(pr.ice_ctrl["h_ml"]) == 2.0
    assert sorted(base.ice_ctrl) == ["h_ml"]          # the base is not mutated


@pytest.mark.parametrize("name", ["ice.albedo", "ice.ice_dt", "ice.whichEVP", "ice.iclasses",
                                  "ice.adjoint_theta", "ice.", "ice_ctrl", "surface_adj"])
def test_unknown_ice_controls_raise(name):
    with pytest.raises(ValueError, match="ice|Params"):
        LogNormalControl(name, 0.5, 0.5)


def test_ice_check_raises_outside_u_max():
    cs = ice_thermo_control_set()
    u = np.zeros(cs.n)
    u[3] = U_MAX + 1.0
    with pytest.raises(ValueError, match="ice.albim"):
        cs.check(u)


def test_in_domain():
    cs = ice_thermo_control_set()
    assert cs.in_domain(np.zeros(cs.n)) and cs.out_of_domain(np.zeros(cs.n)) == {}

    def u_at(name, p):
        u = np.zeros(cs.n)
        i = cs.names.index(name)
        u[i] = cs.controls[i].to_u(p)
        return u
    assert not cs.in_domain(u_at("ice.albsn", 0.96))
    assert set(cs.out_of_domain(u_at("ice.albsn", 0.96))) == {"ice.albsn"}
    assert cs.in_domain(u_at("ice.albsn", 0.949))
    assert not cs.in_domain(u_at("ice.albim", 1.02))
    assert not cs.in_domain(u_at("ice.con", 2.1656 * 10.5))
    assert not cs.in_domain(u_at("ice.st_oce_ice", 0.006 / 10.5))
    assert cs.in_domain(u_at("ice.st_oce_ice", 0.006 * 9.5))
    bad = np.zeros(cs.n)
    bad[0] = np.nan
    assert not cs.in_domain(bad)
    assert not cs.in_domain(np.zeros(cs.n - 1))
