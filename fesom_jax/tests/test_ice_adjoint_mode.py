"""``IceConfig.adjoint_mode`` — the four ways the backward pass can treat sea ice.

The forward must be bit-for-bit identical in all four, because `stop_gradient` is the
identity: whichever mode you pick, the ice trajectory is the validated model and only the
gradient changes. That property is what makes "rheology" safe to switch on by default, so it
is tested per mode rather than assumed.

"frozen" cuts the whole ice update; "rheology" cuts only the momentum solver's subcycles and
keeps thermodynamics, advection, cut-off and the ocean-to-ice coupling differentiated. Those two
must differ, and through a full step they do — 1.85% apart.

"prescribed" cuts the ice STATE at the step's entry and exit and nothing else, so it is defined
by two structural zeros and one structural non-zero, all tested below at the level where they
are exact: no gradient reaches the entering ice state; none leaves through the updated ice
state; and the gradient with respect to the atmosphere is kept. The last is the one "frozen"
loses — its cut is the whole ``IceStepOut``, fluxes included, so under "frozen"
dJ/d(atmosphere) is identically zero. That was found the expensive way (a 30-day production
gradient with a forcing sensitivity of exactly 0.0, mhw_optimals job 27172648), and it is
pinned here so it cannot be mistaken for a feature again.

⚠️ "rheology" vs "exact" is NOT checked through a step, and the file used to try. The only thing
between those two modes is the momentum solve, whose contribution to a one-step
surface-temperature loss is below floating-point rounding, so the old assertion was comparing
XLA's rounding and flipped between passing and failing depending on the evaluation point. It is
checked instead at the solver's own boundary, where ``adjoint_subcycles=0`` gives a structural
exact zero. General lesson, and the reason this file was wrong in three separate ways: a gradient
assertion needs something independent to compare against — a finite difference, or a quantity
that is zero by construction — not just "finite", "nonzero" or "different".

Needs the CORE2 mesh and initial condition (sea ice is a forced-path feature; the packaged pi
setup has no ice), so it skips when those are absent. The fixture seeds the ice; without that
there is none, and every gradient here is ice-free.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ssh, surface_forcing
from fesom_jax import step as stepmod
from fesom_jax import ice as icemod
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.tests._identity_gates import assert_forward_identity

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing (sea ice needs the forced path)")

MODES = ("exact", "rheology", "prescribed", "frozen")


@pytest.fixture(scope="module")
def core2_setup():
    """The one CORE2 step's ingredients, loaded once for every builder in this module."""
    mesh = load_mesh(MESH_DIR)
    _bare = phc_initial_state(mesh, IC_DIR)
    state = icemod.seed_ice(_bare, mesh, _bare.T[:, 0])
    assert int(jnp.count_nonzero(state.a_ice)) > 1000, \
        "seed_ice produced no ice — every gradient in this file would be ice-free"
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    fs = cf.static
    sf0 = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    return mesh, state, op, fs, sf0, wet0


@pytest.fixture(scope="module")
def core2(core2_setup):
    """One CORE2 step WITH SEA ICE PRESENT, as a function of a surface-temperature perturbation.

    ⚠️ ``seed_ice`` is not optional and ``k_ver`` is not a usable control. Both were wrong here
    until 2026-08-07 and each independently made every gradient in this file ice-free.

    **The ice must be seeded.** ``phc_initial_state`` does NOT seed sea ice — ``seed_sea_ice`` is
    a parameter of ``cold_start_state``, a different function. ``phc_ic.py:503`` states the rule:
    "Every validated entry point does ``ice.seed_ice(phc_initial_state(...), mesh, sst)``", and
    every other ice test file does exactly that (``test_ice_step.py``, ``test_ice_ic.py``,
    ``test_ale_zstar.py``, ``test_gradient_sharded.py``). This one did not, so it ran the sea-ice
    adjoint on an ocean with ``a_ice`` identically zero — measured, 0 nodes. Seeded: 37091 nodes,
    ``a_ice`` to 0.9, ``m_ice`` to 2.0.

    **The control must be the state.** ``ice_surface_step`` is handed the input state and never
    receives ``Params`` (``step.py:197``), so NO parameter has a gradient path into the ice at
    step one. With ``k_ver``, even with ice present, all three modes agree to 1 ulp
    (−1.73583430338209e+00) — which is why ``rheology != frozen`` was failing: it was comparing
    XLA rounding, and it flipped between pass and fail with the evaluation point. A surface
    perturbation reaches the ice through ``ocean2ice`` → thermodynamics: seeded, ``frozen`` gives
    +8.80388483606558e−01 against +8.64085668842909e−01 for the other two, a real 1.85%."""
    mesh, state, op, fs, sf0, wet0 = core2_setup

    def make(mode, whichEVP=1):
        cfg = IceConfig(whichEVP=whichEVP, adjoint_mode=mode)

        def loss(eps):
            st = dataclasses.replace(state, T=state.T.at[:, 0].add(jnp.where(wet0, eps, 0.0)))
            fin = stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT, is_first_step=True,
                               step_forcing=sf0, forcing_static=fs, ice_cfg=cfg)
            return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)

        return jax.jit(loss), jax.jit(jax.grad(loss))

    return make, jnp.asarray(0.0, jnp.float64)


@pytest.fixture(scope="module")
def core2_ctl(core2_setup):
    """The same step as a scalar function of a chosen CONTROL, measured on a chosen OUTPUT.

    ``control``: ``"sst"`` (surface T, as in ``core2``), ``"longwave"`` (this step's downward
    longwave — the atmosphere), or ``"a_ice"`` (the entering ice concentration, added only where
    ice exists so the clamp to [0, 1] is not crossed). ``output``: the wet-mean surface T
    (``"sst"``) or the ice-mean updated concentration (``"a_ice"``). Every combination is a
    single step, so each gradient here is a one-step statement — enough, because the properties
    tested are structural zeros and non-zeros, not magnitudes."""
    mesh, state, op, fs, sf0, wet0 = core2_setup
    icy = state.a_ice > 0.0

    def make(mode, control, output):
        cfg = IceConfig(whichEVP=1, adjoint_mode=mode)

        def loss(eps):
            st, sf = state, sf0
            if control == "sst":
                st = dataclasses.replace(st, T=st.T.at[:, 0].add(jnp.where(wet0, eps, 0.0)))
            elif control == "longwave":
                sf = sf._replace(longwave=sf.longwave + eps)
            elif control == "a_ice":
                st = dataclasses.replace(st, a_ice=st.a_ice + jnp.where(icy, 0.01 * eps, 0.0))
            else:
                raise ValueError(control)
            fin = stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT, is_first_step=True,
                               step_forcing=sf, forcing_static=fs, ice_cfg=cfg)
            if output == "sst":
                return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)
            if output == "a_ice":
                return jnp.sum(jnp.where(icy, fin.a_ice, 0.0)) / jnp.sum(icy)
            raise ValueError(output)

        return jax.jit(jax.grad(loss))

    return make, jnp.asarray(0.0, jnp.float64)


def test_unknown_mode_fails_loudly(core2):
    """A typo must raise, not silently fall through to 'exact' and produce NaN later."""
    make, k0 = core2
    loss, _ = make("frzoen")
    with pytest.raises(NotImplementedError, match="adjoint_mode"):
        loss(k0)


@pytest.mark.parametrize("mode", MODES)
def test_forward_is_identical_across_modes(core2, mode):
    """stop_gradient is the identity: the ice trajectory must not move by a single bit,
    whichever mode is chosen. This is the whole basis for switching modes in production.

    Bit-identical on CPU. On GPU the two modes are separate ``jax.jit`` closures, so this
    compares two separately-compiled programs; ``frozen`` landed 2 ulp from ``exact`` there
    (job 26780293) and failed a bare ``==``. The GPU bound is calibrated instead — see
    ``_identity_gates``."""
    make, k0 = core2
    ref, _ = make("exact")
    loss, _ = make(mode)
    assert_forward_identity(ref, loss, k0, f"adjoint_mode={mode!r}")


@pytest.mark.parametrize("mode", MODES)
def test_gradient_is_finite_except_where_documented(core2, mode):
    """Over a single step 'exact' has not yet had time to overflow, so all three are finite
    here. The point of the test is that 'rheology' and 'frozen' are finite and nonzero — a
    silently zero gradient would mean the mode had cut more than it claims."""
    make, k0 = core2
    _, grad = make(mode)
    g = float(grad(k0))
    assert np.isfinite(g), f"adjoint_mode={mode!r} gave a non-finite gradient"
    if mode != "exact":
        assert g != 0.0, f"adjoint_mode={mode!r} zeroed the gradient entirely"


def test_rheology_differs_from_frozen(core2):
    """The half of the claim that IS observable through a full step: 'rheology' keeps ice
    thermodynamics, advection, cut-off and the ocean coupling differentiated, so it must not
    match 'frozen', which cuts all of them. Measured seeded: +8.64085668842909e-01 vs
    +8.80388483606558e-01, 1.85% apart.

    The other half — 'rheology' vs 'exact' — is NOT tested here, and deliberately. The only
    difference between those two is the momentum solve, and its contribution to a one-step
    surface-temperature loss is below rounding: measured seeded, SST gives the two modes
    BIT-IDENTICAL values, and an ocean-velocity control (which does reach ``srfoce_u/v`` and so
    the EVP) separates them by 23 ulp, which is noise, not signal. Waiting for a longer window
    is not an option either, since 'exact' returns NaN inside the first day. That comparison
    belongs at the solver's own boundary — see
    :func:`test_adjoint_subcycles_zero_cuts_the_momentum_gradient`."""
    make, k0 = core2
    _, g_rh = make("rheology")
    _, g_fr = make("frozen")
    rh, fr = float(g_rh(k0)), float(g_fr(k0))
    assert np.isfinite(rh) and np.isfinite(fr)
    assert abs(rh - fr) > 1e-3 * abs(fr), \
        f"rheology {rh:.6e} matched frozen {fr:.6e} — the extra ice differentiation did " \
        f"nothing. Is the ice actually seeded?"


@pytest.mark.parametrize("whichEVP", [1, 0])
def test_adjoint_subcycles_zero_cuts_the_momentum_gradient(whichEVP):
    """The momentum-solver cut, tested where it acts instead of through eight layers of dilution.

    ``adjoint_mode='rheology'`` passes ``adjoint_subcycles=0`` to the solver, which runs every
    sweep in the forward and then stop_gradients the WHOLE carried state
    (``ice_mevp.py``, after the ``n_grad < n_free`` split). So at the solver's own boundary the
    gradient is EXACTLY zero — a structural zero that rounding cannot fake, unlike the ulp-level
    differences the step-level version was reduced to comparing. Measured seeded, mEVP:
    −5.21993727702461e+04 with all sweeps, +0.0 with none.

    Parametrized over both solvers because the mode must not be silently mEVP-only: whichEVP=0
    carries the same knob and the same 120-subcycle structure."""
    from fesom_jax import ice_evp, ice_mevp
    mesh = load_mesh(MESH_DIR)
    _bare = phc_initial_state(mesh, IC_DIR)
    st = icemod.seed_ice(_bare, mesh, _bare.T[:, 0])
    solver = ice_mevp.mevp_dynamics if whichEVP == 1 else ice_evp.evp_dynamics
    cfg = IceConfig(whichEVP=whichEVP)
    z = jnp.zeros(mesh.nod2D, jnp.float64)

    def grad_for(**kw):
        def loss(eps):
            u, v, *_ = solver(
                cfg, mesh, a_ice=st.a_ice, m_ice=st.m_ice, m_snow=st.m_snow,
                u_ice=st.u_ice, v_ice=st.v_ice,
                sigma11=st.sigma11, sigma12=st.sigma12, sigma22=st.sigma22,
                srfoce_u=z + eps, srfoce_v=z, elevation=z,
                stress_ax=z + 0.05, stress_ay=z, boundary_node=None, exch=None, **kw)
            return jnp.sum(u) + jnp.sum(v)
        return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))

    g_all = grad_for()                              # default: every sweep differentiated
    g_none = grad_for(adjoint_subcycles=0)          # what adjoint_mode="rheology" passes
    assert np.isfinite(g_all) and g_all != 0.0, \
        f"whichEVP={whichEVP}: the momentum solve has no gradient even with all sweeps " \
        f"unrolled ({g_all!r}) — is the ice seeded and the forcing nonzero?"
    assert g_none == 0.0, \
        f"whichEVP={whichEVP}: adjoint_subcycles=0 left a gradient of {g_none!r}; the carry " \
        f"is supposed to be stop_gradient'd entirely, so this must be exactly zero"


def test_prescribed_keeps_the_atmosphere_gradient_that_frozen_zeroes(core2_ctl):
    """The defining non-zero of "prescribed", and the defining defect of "frozen".

    The surface fluxes leave ``ice_surface_step`` inside ``IceStepOut``. "frozen" stop_gradients
    that whole tuple, so the downward longwave — which reaches the ocean ONLY through those
    fluxes — has a gradient of exactly 0.0 under it: a structural zero, not a small number.
    "prescribed" cuts the ice state instead and must keep it; "rheology" keeps it too, and the
    two differ only by the ice-mediated part (longwave -> ice growth -> under-ice flux), which
    is real but small. Both are asserted: "prescribed" non-zero, and within 10 % of "rheology"
    — if it ever matched "frozen" the exit cut would have swallowed a flux."""
    make, k0 = core2_ctl
    g_pr = float(make("prescribed", "longwave", "sst")(k0))
    g_rh = float(make("rheology", "longwave", "sst")(k0))
    g_fr = float(make("frozen", "longwave", "sst")(k0))
    assert g_fr == 0.0, \
        f"frozen kept a longwave gradient of {g_fr!r} — the whole-IceStepOut cut has moved"
    assert np.isfinite(g_pr) and g_pr != 0.0, \
        f"prescribed zeroed the longwave gradient ({g_pr!r}): the exit cut swallowed a flux"
    assert np.isfinite(g_rh) and g_rh != 0.0
    assert abs(g_pr - g_rh) <= 0.10 * abs(g_rh), \
        f"prescribed {g_pr:.6e} vs rheology {g_rh:.6e}: the flux path should carry nearly " \
        f"all of the longwave sensitivity through one step"


def test_prescribed_has_no_gradient_through_the_entering_ice_state(core2_ctl):
    """The entry cut. A perturbation of the ice concentration the step receives changes the
    surface temperature it produces (through the open-water fraction of every flux), and
    "rheology" sees that: measured non-zero. "prescribed" must return exactly 0.0, because the
    nine prognostic ice fields are stop_gradient'd before anything reads them."""
    make, k0 = core2_ctl
    g_rh = float(make("rheology", "a_ice", "sst")(k0))
    g_pr = float(make("prescribed", "a_ice", "sst")(k0))
    assert np.isfinite(g_rh) and g_rh != 0.0, \
        f"rheology has no SST sensitivity to the entering a_ice ({g_rh!r}) — is the ice seeded?"
    assert g_pr == 0.0, \
        f"prescribed let {g_pr!r} through the entering ice state; the entry cut is incomplete"


def test_prescribed_has_no_gradient_into_the_updated_ice_state(core2_ctl):
    """The exit cut. A surface-temperature perturbation changes the ice the step hands to the
    next one (thermodynamic growth and melt), and "rheology" sees that: measured non-zero.
    "prescribed" must return exactly 0.0 on the updated ``a_ice`` — this is the cut that stops
    the recurrence whose gain at an ice edge is the reason the mode exists."""
    make, k0 = core2_ctl
    g_rh = float(make("rheology", "sst", "a_ice")(k0))
    g_pr = float(make("prescribed", "sst", "a_ice")(k0))
    assert np.isfinite(g_rh) and g_rh != 0.0, \
        f"rheology's updated a_ice does not depend on SST ({g_rh!r}) — is the ice seeded?"
    assert g_pr == 0.0, \
        f"prescribed let {g_pr!r} into the updated ice state; the exit cut is incomplete"


def test_rheology_forward_is_identical_on_standard_evp_too(core2):
    """The mode must not be silently mEVP-only: whichEVP=0 is a different solver with the same
    120-subcycle structure and it carries the same knob.

    This half checks the FORWARD, which is the production-relevant claim — switching the mode
    must not move the trajectory on either solver. That the knob actually reaches the standard-EVP
    backward is checked by
    :func:`test_adjoint_subcycles_zero_cuts_the_momentum_gradient`\\ ``[0]``, at the solver's own
    boundary; asserting it through a full step here would be comparing ulps (see
    :func:`test_rheology_differs_from_frozen`).

    ⚠️ This is the loosest of the identity gates on GPU and the reason is structural: the two
    programs differ by a ``stop_gradient`` inside a 120-sweep subcycle loop, which is enough for
    XLA to reassociate the accumulation, and the two forwards landed 2.9e-13 apart — ~160 ulp,
    against a ``rel=1e-14`` tolerance that had been sized from the 1-ulp mEVP case (job
    26780293). The calibrated bound in ``_identity_gates`` covers it with ~300x margin while
    still being 1e-11 of the value."""
    make, k0 = core2
    l_ex, _ = make("exact", whichEVP=0)
    l_rh, _ = make("rheology", whichEVP=0)
    assert_forward_identity(l_ex, l_rh, k0, "adjoint_mode='rheology' on the standard EVP")
