"""The dynamic sea-ice controls of the joint ice calibration (``controls.ice_joint_control_set``):
the ice strength P*, the ice-ocean drag and the atmosphere-ice drag ``IceConfig.cd_atm_ice``,
which the wind stress on the ice (``forcing.atm_ice_stress``) now reads instead of the module
constant. What is pinned here:

* the set: the ten thermodynamic controls in their order, then the three dynamic ones at the
  ``IceConfig`` defaults with σ = ln 2 (data-free);
* the default drag is the old constant, and the stress at it is bitwise the old stress
  (data-free);
* CORE2: one ice step with the three dynamic controls traced through ``ice_ctrl`` equals the
  step with the directly replaced config, and a 1 % move of each changes the ice velocity
  (before the drag was read from the config, a ``cd_atm_ice`` move changed nothing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import forcing, ice_step
from fesom_jax.config import CD_ATM_ICE
from fesom_jax.controls import (ICE_DYN_FIELDS, ICE_PREFIX, SIGMA_DEFAULT,
                                ice_joint_control_set, ice_thermo_control_set)
from fesom_jax.ice import IceConfig
from fesom_jax.params import Params
from fesom_jax.tests import test_ice_stanton as stan
from fesom_jax.tests.test_ice_ctrl import _with, setup  # noqa: F401  (module fixture)

DT = stan.DT
DYN = ("pstar", "cd_oce_ice", "cd_atm_ice")


def test_joint_set_is_the_thermo_set_then_the_dynamics():
    th, jt = ice_thermo_control_set(), ice_joint_control_set()
    assert ICE_DYN_FIELDS == DYN
    assert jt.names == th.names + tuple(ICE_PREFIX + f for f in DYN)
    np.testing.assert_array_equal(jt.p0[:th.n], th.p0)
    np.testing.assert_array_equal(jt.sigma[:th.n], th.sigma)
    d = IceConfig()
    for c, f in zip(jt.controls[th.n:], DYN):
        assert c.p0 == float(getattr(d, f)) and c.sigma == SIGMA_DEFAULT


def test_default_atm_ice_drag_is_the_old_constant():
    assert IceConfig().cd_atm_ice == forcing.BULK_CD_ATM_ICE == CD_ATM_ICE
    rng = np.random.default_rng(0)
    ua, va, ui, vi = (jnp.asarray(rng.normal(size=64)) for _ in range(4))
    ref = forcing.atm_ice_stress(ua, va, ui, vi)
    got = forcing.atm_ice_stress(ua, va, ui, vi, cd=IceConfig().cd_atm_ice)
    for r, g in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(r), np.asarray(g))
    dbl = forcing.atm_ice_stress(ua, va, ui, vi, cd=2.0 * forcing.BULK_CD_ATM_ICE)
    for r, g in zip(ref, dbl):
        np.testing.assert_allclose(np.asarray(g), 2.0 * np.asarray(r), rtol=1e-15)


@stan.core2
def test_dynamic_controls_reach_the_ice_step(setup):  # noqa: F811
    mesh, state, op, fs, sfs = setup
    cfg = IceConfig(**stan.PROD)

    def ctrl(lnp):
        return {f: jnp.asarray(getattr(cfg, f), jnp.float64) * jnp.exp(lnp[i])
                for i, f in enumerate(DYN)}

    def ice(c, p):
        o = ice_step.ice_surface_step(c, mesh, state, sfs[0], fs, dt=DT, params=p)
        return o.a_ice, o.m_ice, o.u_ice, o.v_ice

    via_ctrl = jax.jit(lambda lnp: ice(cfg, _with(Params.defaults(), ice_ctrl=ctrl(lnp))))
    via_cfg = jax.jit(lambda lnp: ice(cfg._replace(**ctrl(lnp)), None))
    z = jnp.zeros(len(DYN), jnp.float64)
    ref = [np.asarray(x) for x in via_ctrl(z)]
    quiet, mismatch = [], []
    for i, f in enumerate(DYN):
        lnp = z.at[i].set(0.01)
        a = [np.asarray(x) for x in via_ctrl(lnp)]
        b = [np.asarray(x) for x in via_cfg(lnp)]
        d_ab = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
        moved = max(float(np.max(np.abs(x - r))) for x, r in zip(a[2:], ref[2:]))
        print(f"[dyn ice_ctrl] {f:10s} max|ctrl - cfg| {d_ab:.3e}  max|moved u_ice| {moved:.3e}")
        if d_ab > 1e-13:
            mismatch.append((f, d_ab))
        if not moved > 1e-9:
            quiet.append((f, moved))
    assert via_ctrl._cache_size() == 1
    assert not mismatch, f"ice_ctrl and the directly replaced config differ: {mismatch}"
    assert not quiet, f"a 1 % move left the ice velocity unchanged: {quiet}"
