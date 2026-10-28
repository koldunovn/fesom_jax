"""Sea-ice controls ``Params.ice_pstar`` (P*) and ``Params.ice_cd`` (ice-ocean C_d).

``None`` (the default) must leave the program untouched; a value replaces the ``IceConfig``
field inside ``ice_surface_step`` as a traced scalar. What is pinned here, each where it is
exact:

* unset ⇒ ``Params.defaults()`` has no ice leaves and traces the same ice step as no
  ``params`` (the comparison with the pre-control program is the jaxpr-hash baseline check);
* set to the configured constants ⇒ a CORE2 step is the same to rounding (CPU), for mEVP and
  standard EVP — not bitwise: a traced value is a different XLA program from a constant;
* P* reaches the mEVP pressure linearly: ``d pressure_fac / d ln P* = pressure_fac`` bitwise;
* C_d reaches each of its four ice-path sites linearly: the mEVP and EVP drag products
  (captured at the site), the ice-ocean stress (affine in C_d), the friction velocity
  (``u* ∝ √C_d``);
* the tangent of a 2-step ice-drift functional w.r.t. ln P* and ln C_d agrees with a central
  finite difference under ``adjoint_mode="exact"`` with 4 mEVP sweeps (short enough that the
  exact tangent-linear model does not blow up), one control set and the other ``None``.

Needs the CORE2 mesh and PHC IC (sea ice is a forced-path feature); skips without them.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ice as icemod
from fesom_jax import ice_coupling, ice_evp, ice_mevp, ice_step, ice_thermo
from fesom_jax import ssh, surface_forcing
from fesom_jax import step as stepmod
from fesom_jax.config import DENSITY_0
from fesom_jax.forcing import _safe_speed
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958
ARCTIC_LAT = 70.0

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing (sea ice needs the forced path)")

P0 = IceConfig().pstar
CD0 = IceConfig().cd_oce_ice


@pytest.fixture(scope="module")
def setup():
    mesh = load_mesh(MESH_DIR)
    bare = phc_initial_state(mesh, IC_DIR)
    state = icemod.seed_ice(bare, mesh, bare.T[:, 0])
    assert int(jnp.count_nonzero(state.a_ice)) > 1000, "seed_ice produced no ice"
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    dates = surface_forcing.dates_for_steps(YEAR, DT, 2)
    sfs = [cf.step_forcing(*d) for d in dates]
    return mesh, state, op, cf.static, sfs


def _canon(text: str) -> str:
    return re.sub(r"0x[0-9a-fA-F]+", "0x?", text)


def _with(base: Params, **kw) -> Params:
    return dataclasses.replace(base, **kw)


# ---------------------------------------------------------------------------
# unset ⇒ nothing traced


def test_defaults_have_no_ice_leaves():
    p = Params.defaults()
    assert p.ice_pstar is None and p.ice_cd is None
    assert len(jax.tree.leaves(p)) == 8
    q = _with(p, ice_pstar=jnp.asarray(P0), ice_cd=jnp.asarray(CD0))
    assert len(jax.tree.leaves(q)) == 10


@pytest.mark.parametrize("whichEVP", [1, 0])
def test_unset_controls_trace_the_same_ice_step(setup, whichEVP):
    """No ``params``, ``params=None`` and ``Params.defaults()`` (ice leaves unset) trace the
    same ice-step program, and setting a leaf changes it. This does NOT show that the unset
    program equals the one before the controls existed (all three calls run the current code);
    that is the cross-commit jaxpr-hash check of one CORE2 step per adjoint mode, run against
    the saved baseline after every library change."""
    mesh, state, op, fs, sfs = setup
    cfg = IceConfig(whichEVP=whichEVP)

    def jaxpr(**kw):
        return _canon(str(jax.make_jaxpr(
            lambda st: ice_step.ice_surface_step(cfg, mesh, st, sfs[0], fs, dt=DT, **kw))(state)))

    ref = jaxpr()
    assert jaxpr(params=None) == ref
    assert jaxpr(params=Params.defaults()) == ref
    # and setting one is visible (the comparison can fail)
    assert jaxpr(params=_with(Params.defaults(), ice_cd=jnp.asarray(CD0))) != ref


# ---------------------------------------------------------------------------
# set to the configured constants ⇒ the same forward to rounding


SIGMA = ("sigma11", "sigma12", "sigma22")
STEP_RTOL = 1e-9          #: every field but σ, max|Δ|/max|x| after one step (measured: mEVP
#                            3.8e-11, EVP 1.6e-10 in u_ice; job 27984771)
SIGMA_RTOL = 1e-6         #: σ (measured ≤ 2.0e-8: the subcycles amplify the rounding)


@pytest.mark.parametrize("whichEVP", [1, 0])
def test_configured_values_give_the_step_to_rounding(setup, whichEVP):
    """The controls at the configured constants give the constant-config step to ROUNDING, not
    bitwise. A traced P* or C_d is a different XLA program from a compile-time constant (the
    compiler can fold and reassociate constants it can see), and each control alone already
    moves the jitted ice step by ulps (job 27984560, CPU: P* alone u_ice 2.4e-11, σ11 1.2e-8
    relative after one step; C_d alone 7.8e-12 and 4.0e-9). Every run that compares with
    another (base, replay, FD) therefore carries the controls, so they share one program."""
    if jax.devices()[0].platform != "cpu":
        pytest.skip("the rounding bound is calibrated on CPU")
    mesh, state, op, fs, sfs = setup
    cfg = IceConfig(whichEVP=whichEVP)

    @jax.jit
    def run(st, p):
        return stepmod.step(st, mesh, op, None, p, dt=DT, is_first_step=True,
                            step_forcing=sfs[0], forcing_static=fs, ice_cfg=cfg)

    ref = run(state, Params.defaults())
    alt = run(state, _with(Params.defaults(), ice_pstar=jnp.asarray(P0, jnp.float64),
                           ice_cd=jnp.asarray(CD0, jnp.float64)))
    rel = {}
    for f in dataclasses.fields(ref):
        a, b = np.asarray(getattr(ref, f.name)), np.asarray(getattr(alt, f.name))
        if a.dtype.kind == "f":
            rel[f.name] = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
    print(f"[ctl at constants] whichEVP={whichEVP} worst rel "
          f"{sorted(rel.items(), key=lambda kv: -kv[1])[:5]}")
    bad = {k: v for k, v in rel.items()
           if not np.isfinite(v) or v > (SIGMA_RTOL if k in SIGMA else STEP_RTOL)}
    assert not bad, f"whichEVP={whichEVP}: beyond rounding: {bad}"
    assert float(jnp.max(jnp.abs(alt.u_ice))) > 0.0, "the step moved no ice"


# ---------------------------------------------------------------------------
# exact plumbing at the sites


def test_pressure_fac_is_linear_in_pstar(setup):
    """``d pressure_fac / d ln P* = pressure_fac``, bitwise (P* enters as one factor)."""
    mesh, state, *_ = setup
    cfg = IceConfig(whichEVP=1)
    bnd = ice_evp.boundary_node_mask(mesh)
    z = jnp.zeros(mesh.nod2D, jnp.float64)

    def pf(lnp):
        c = cfg._replace(pstar=P0 * jnp.exp(lnp))
        return ice_mevp.mevp_setup(c, mesh, state.a_ice, state.m_ice, state.m_snow, z,
                                   bnd).pressure_fac

    prim, tan = jax.jvp(pf, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    prim, tan = np.asarray(prim), np.asarray(tan)
    assert np.count_nonzero(prim) > 1000
    np.testing.assert_array_equal(tan, prim)
    ref = np.asarray(ice_mevp.mevp_setup(cfg, mesh, state.a_ice, state.m_ice, state.m_snow, z,
                                         bnd).pressure_fac)
    np.testing.assert_array_equal(prim, ref)


class _Probe:
    """Stands in for ``DENSITY_0`` in a module: records the operand it multiplies (the drag
    product up to that factor) and returns the true product."""

    def __init__(self):
        self.seen = []

    def __rmul__(self, x):
        self.seen.append(x)
        return x * DENSITY_0

    __mul__ = __rmul__


def _ice_inputs(mesh, state):
    lat = jnp.asarray(mesh.geo_coord_nod2D)[:, 1]
    lon = jnp.asarray(mesh.geo_coord_nod2D)[:, 0]
    u = 0.1 * jnp.cos(3.0 * lon) * jnp.cos(lat)
    v = 0.05 * jnp.sin(2.0 * lon)
    uw = 0.02 * jnp.sin(lat)
    vw = -0.03 * jnp.cos(lon)
    return u, v, uw, vw


def test_cd_site_mevp_drag(setup, monkeypatch):
    mesh, state, *_ = setup
    cfg = IceConfig(whichEVP=1)
    bnd = ice_evp.boundary_node_mask(mesh)
    z = jnp.zeros(mesh.nod2D, jnp.float64)
    st = ice_mevp.mevp_setup(cfg, mesh, state.a_ice, state.m_ice, state.m_snow, z, bnd)
    u, v, uw, vw = _ice_inputs(mesh, state)
    ne = mesh.elem2D
    s = jnp.zeros(ne, jnp.float64)

    def site(lncd):
        probe = _Probe()
        monkeypatch.setattr(ice_mevp, "DENSITY_0", probe)
        c = cfg._replace(cd_oce_ice=CD0 * jnp.exp(lncd))
        ice_mevp.mevp_iterate(c, mesh, u, v, s, s, s, st, u, v, uw, vw, z + 0.05, z, 1800.0)
        monkeypatch.setattr(ice_mevp, "DENSITY_0", DENSITY_0)
        assert len(probe.seen) == 1, "expected one DENSITY_0 product (the drag) in a sweep"
        return probe.seen[0]

    prim, tan = jax.jvp(site, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    prim, tan = np.asarray(prim), np.asarray(tan)
    assert np.count_nonzero(prim) > 1000
    np.testing.assert_array_equal(tan, prim)


def test_cd_site_evp_drag(setup, monkeypatch):
    mesh, state, *_ = setup
    cfg = IceConfig(whichEVP=0)
    bnd = ice_evp.boundary_node_mask(mesh)
    u, v, uw, vw = _ice_inputs(mesh, state)
    z = jnp.zeros(mesh.nod2D, jnp.float64)
    inv_mass = jnp.where(state.a_ice >= 0.01, 1.0 / 910.0, 0.0)

    def site(lncd):
        probe = _Probe()
        monkeypatch.setattr(ice_evp, "DENSITY_0", probe)
        c = cfg._replace(cd_oce_ice=CD0 * jnp.exp(lncd))
        ice_evp.velocity_update(c, mesh, u, v, z, z, uw, vw, z + 0.05, z, inv_mass,
                                state.a_ice, bnd)
        monkeypatch.setattr(ice_evp, "DENSITY_0", DENSITY_0)
        assert len(probe.seen) == 1, "expected one DENSITY_0 product (the drag)"
        return probe.seen[0]

    prim, tan = jax.jvp(site, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    prim, tan = np.asarray(prim), np.asarray(tan)
    assert np.count_nonzero(prim) > 1000
    np.testing.assert_array_equal(tan, prim)


def test_cd_site_ice_ocean_stress(setup):
    """The ice-ocean stress blend is affine in C_d: ``f(C_d) = f(0) + C_d·B``, so
    ``d f / d ln C_d = f(C_d0) - f(0)`` (to rounding). The primal of the JVP is compared with
    the constant-C_d value to rounding, not bitwise: the node-to-element average is evaluated
    inside a different program there (measured: 1 ulp, 2.2e-16 relative)."""
    mesh, state, *_ = setup
    cfg = IceConfig(whichEVP=1)
    u, v, uw, vw = _ice_inputs(mesh, state)
    atm = jnp.stack([0.1 + 0.0 * u, -0.05 + 0.0 * u], axis=1)
    ow = jnp.asarray(mesh.ulevels_nod2D) <= 1

    def f(c):
        return ice_coupling.ice_oce_fluxes_mom(mesh, state.a_ice, u, v, uw, vw, atm, ow, c)

    def g(lncd):
        return f(cfg._replace(cd_oce_ice=CD0 * jnp.exp(lncd)))

    prim, tan = jax.jvp(g, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    zero = f(cfg._replace(cd_oce_ice=0.0))
    for p, t, z0, ref in zip(prim, tan, zero, f(cfg)):
        p, t, z0 = np.asarray(p), np.asarray(t), np.asarray(z0)
        scale = np.max(np.abs(p))
        np.testing.assert_allclose(p, np.asarray(ref), rtol=0.0, atol=4e-16 * scale)
        assert np.count_nonzero(t) > 1000
        np.testing.assert_allclose(t, p - z0, rtol=1e-12, atol=1e-14 * scale)


def test_cd_site_friction_velocity(setup):
    """``u* = √(|Δu|²·C_d)`` ⇒ ``d u* / d ln C_d = u*/2``."""
    mesh, state, *_ = setup
    cfg = IceConfig(whichEVP=1)
    u, v, uw, vw = _ice_inputs(mesh, state)

    def g(lncd):
        return ice_thermo.compute_ustar(cfg._replace(cd_oce_ice=CD0 * jnp.exp(lncd)), u, v, uw, vw)

    prim, tan = jax.jvp(g, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    prim, tan = np.asarray(prim), np.asarray(tan)
    np.testing.assert_array_equal(prim, np.asarray(ice_thermo.compute_ustar(cfg, u, v, uw, vw)))
    assert np.count_nonzero(prim) > 1000
    np.testing.assert_allclose(tan, 0.5 * prim, rtol=1e-14, atol=0.0)


# ---------------------------------------------------------------------------
# a 2-step drift functional: tangent vs central FD under "exact" with 4 sweeps


def _drift_fn(setup, control):
    mesh, state, op, fs, sfs = setup
    cfg = IceConfig(whichEVP=1, adjoint_mode="exact")._replace(evp_rheol_steps=4)
    lat = np.degrees(np.asarray(mesh.geo_coord_nod2D)[:, 1])
    box = (lat >= ARCTIC_LAT) & np.asarray(mesh.node_layer_mask[:, 0])
    w = jnp.where(jnp.asarray(box), jnp.asarray(mesh.areasvol[:, 0]), 0.0)
    base = Params.defaults()

    def drift(s):
        wa = w * s.a_ice
        return jnp.sum(wa * _safe_speed(s.u_ice, s.v_ice)) / jnp.sum(wa)

    def J(lnp):
        if control == "pstar":
            p = _with(base, ice_pstar=jnp.asarray(cfg.pstar) * jnp.exp(lnp))
        else:
            p = _with(base, ice_cd=jnp.asarray(cfg.cd_oce_ice) * jnp.exp(lnp))
        s, tot = state, 0.0
        for k in range(2):
            s = stepmod.step(s, mesh, op, None, p, dt=DT, is_first_step=(k == 0),
                             step_forcing=sfs[k], forcing_static=fs, ice_cfg=cfg)
            tot = tot + drift(s)
        return tot / 2.0
    return J


@pytest.mark.parametrize("control", ["pstar", "cd"])
def test_drift_tangent_matches_fd(setup, control):
    J = _drift_fn(setup, control)
    Jj = jax.jit(J)
    tl = jax.jit(lambda x: jax.jvp(J, (x,), (jnp.ones_like(x),)))
    x0 = jnp.asarray(0.0, jnp.float64)
    j0, ad = (float(v) for v in tl(x0))
    assert np.isfinite(ad) and ad != 0.0, f"{control}: tangent {ad!r}"
    np.testing.assert_array_equal(j0, float(Jj(x0)))
    fds = {}
    for h in (1e-3, 1e-4):
        fds[h] = (float(Jj(x0 + h)) - float(Jj(x0 - h))) / (2.0 * h)
    print(f"[drift FD] {control}: J0 {j0:.15e} tl {ad:.15e} "
          + " ".join(f"fd(h={h:g}) {v:.15e} rel {abs(v - ad) / abs(ad):.3e}"
                     for h, v in fds.items()))
    rel = min(abs(v - ad) / abs(ad) for v in fds.values())
    assert rel < 1e-4, f"{control}: tangent {ad:.6e} vs FD {fds} (best rel {rel:.2e})"
