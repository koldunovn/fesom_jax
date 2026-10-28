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
gradient with a forcing sensitivity of exactly 0.0), and it is
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

    def make(mode, whichEVP=1, **ice_kw):
        # _replace: the drivers' path (it skips IceConfig's validating __new__)
        cfg = IceConfig(whichEVP=whichEVP, adjoint_mode=mode)._replace(**ice_kw)

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

    def make(mode, control, output, **ice_kw):
        cfg = IceConfig(whichEVP=1, adjoint_mode=mode)._replace(**ice_kw)

        def loss(eps):
            st, sf = state, sf0
            if control == "sst":
                st = dataclasses.replace(st, T=st.T.at[:, 0].add(jnp.where(wet0, eps, 0.0)))
            elif control == "longwave":
                sf = sf._replace(longwave=sf.longwave + eps)
            elif control == "a_ice":
                st = dataclasses.replace(st, a_ice=st.a_ice + jnp.where(icy, 0.01 * eps, 0.0))
            elif control == "u_ice":
                st = dataclasses.replace(st, u_ice=st.u_ice + jnp.where(icy, 0.01 * eps, 0.0))
            else:
                raise ValueError(control)
            fin = stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT, is_first_step=True,
                               step_forcing=sf, forcing_static=fs, ice_cfg=cfg)
            if output == "sst":
                return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)
            if output == "a_ice":
                return jnp.sum(jnp.where(icy, fin.a_ice, 0.0)) / jnp.sum(icy)
            if output == "u_ice":
                return jnp.sum(jnp.where(icy, fin.u_ice, 0.0)) / jnp.sum(icy)
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


# ---------------------------------------------------------------------------------------------
# adjoint_mode="dynamics" (stage 1 of the ice-dynamics adjoint; mEVP only)

#: the damping settings the stage-1 runs use, and the truncation knob
DYN_SETTINGS = [
    {},
    dict(adjoint_damping="local", adjoint_kappa=0.25),
    dict(adjoint_damping="local", adjoint_kappa=0.0),
    dict(adjoint_sweeps=8),
    dict(adjoint_damping="local", adjoint_kappa=0.25, adjoint_sweeps=8),
]


@pytest.mark.parametrize("ice_kw", DYN_SETTINGS, ids=lambda d: "+".join(
    f"{k.split('_')[1]}={v}" for k, v in d.items()) or "plain")
def test_dynamics_forward_is_identical_to_prescribed(core2, ice_kw):
    """Every "dynamics" setting is derivative-only: stop_gradient and scale_tangent are the
    identity in the forward, so the step equals "prescribed"'s (bitwise on CPU)."""
    make, k0 = core2
    ref, _ = make("prescribed")
    loss, _ = make("dynamics", **ice_kw)
    assert_forward_identity(ref, loss, k0, f"adjoint_mode='dynamics' {ice_kw}")


def test_dynamics_cuts_the_entering_ice_state_and_keeps_the_ice_velocity(core2_ctl):
    """The entry cut of "dynamics" is the ice STATE only. A perturbation of the entering a_ice
    has exactly zero gradient (as under "prescribed"); a perturbation of the entering u_ice
    reaches the updated u_ice (the momentum solve is differentiated, with the damping), where
    "prescribed" gives exactly zero."""
    make, k0 = core2_ctl
    kw = dict(adjoint_damping="local", adjoint_kappa=0.25)
    g_a = float(make("dynamics", "a_ice", "sst", **kw)(k0))
    assert g_a == 0.0, f"dynamics let {g_a!r} through the entering a_ice"
    g_u = float(make("dynamics", "u_ice", "u_ice", **kw)(k0))
    g_u_pr = float(make("prescribed", "u_ice", "u_ice")(k0))
    assert np.isfinite(g_u) and g_u != 0.0, \
        f"dynamics has no gradient from the entering into the updated u_ice ({g_u!r})"
    assert g_u_pr == 0.0


def test_dynamics_cuts_the_carried_ice_state_at_exit(core2_ctl):
    """The exit cut: the updated a_ice the step hands on carries no tangent, from the ocean
    (SST) or from the ice velocity."""
    make, k0 = core2_ctl
    kw = dict(adjoint_damping="local", adjoint_kappa=0.25)
    assert float(make("dynamics", "sst", "a_ice", **kw)(k0)) == 0.0
    assert float(make("dynamics", "u_ice", "a_ice", **kw)(k0)) == 0.0


def test_dynamics_two_step_gradient_is_finite(core2_setup):
    """Two CORE2 steps, the cost the stage-1 runs use (the area-weighted mean ice speed over the
    ice nodes), differentiated w.r.t. the entering ice velocity under "dynamics" with the local
    damping: finite and non-zero; damped and undamped differ."""
    mesh, state, op, fs, sf0, wet0 = core2_setup
    icy = state.a_ice > 0.0
    w = jnp.where(icy, mesh.area[:, 0] * state.a_ice, 0.0)

    def make(**ice_kw):
        cfg = IceConfig(whichEVP=1, adjoint_mode="dynamics")._replace(**ice_kw)

        def loss(eps):
            st = dataclasses.replace(state, u_ice=state.u_ice + jnp.where(icy, 0.01 * eps, 0.0))
            for k in range(2):
                st = stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT,
                                  is_first_step=(k == 0), step_forcing=sf0, forcing_static=fs,
                                  ice_cfg=cfg)
            sp = jnp.sqrt(st.u_ice ** 2 + st.v_ice ** 2 + 1e-30)
            return jnp.sum(w * sp) / jnp.sum(w)
        return jax.jit(jax.grad(loss))

    g = float(make(adjoint_damping="local", adjoint_kappa=0.25)(jnp.asarray(0.0, jnp.float64)))
    g0 = float(make(adjoint_damping="local", adjoint_kappa=0.0)(jnp.asarray(0.0, jnp.float64)))
    assert np.isfinite(g) and g != 0.0, f"2-step dynamics gradient {g!r}"
    assert np.isfinite(g0) and g0 != 0.0 and g0 != g


def test_dynamics_needs_mevp(core2):
    make, k0 = core2
    loss, _ = make("dynamics", whichEVP=0)
    with pytest.raises(ValueError, match="whichEVP"):
        loss(k0)
    loss, _ = make("prescribed", adjoint_damping="local")
    with pytest.raises(ValueError, match="does not"):
        loss(k0)


# ---------------------------------------------------------------------------------------------
# IceConfig.adjoint_cuts (diagnostic cuts inside the "dynamics" step), at the ice step's own
# boundary: one jitted ice_surface_step, a tangent from one entering field into one output.

#: cut -> (perturbed input field, IceStepOut output, extra cuts that close the other paths)
CUT_CHECKS = {
    "ssh": ("hbar", "u_ice", ()),
    "ocean_vel": ("uvnode", "u_ice", ()),
    "advect": ("u_ice", "water_flux", ()),
    "ice2ocean": ("u_ice", "stress_surf", ("advect",)),
}


@pytest.fixture(scope="module")
def ice_step_jvp(core2_setup):
    """``(cuts, field, output) -> (primal outputs, tangent of output)`` for one ice step under
    "dynamics" + local κ = 0.25 with ``adjoint_cuts=cuts``; the perturbation of ``field`` is
    a fixed random pattern on the icy nodes (uvnode: its surface level)."""
    from fesom_jax import ice_step as ice_step_mod
    mesh, state, op, fs, sf0, wet0 = core2_setup
    icy = np.asarray(state.a_ice > 0.0)
    rng = np.random.default_rng(7)
    pat = jnp.asarray(np.where(icy, rng.standard_normal(mesh.nod2D), 0.0))

    def tangent_state(field):
        z = jax.tree.map(jnp.zeros_like, state)
        if field == "uvnode":
            return dataclasses.replace(z, uvnode=z.uvnode.at[:, 0, 0].set(0.01 * pat))
        return dataclasses.replace(z, **{field: 0.01 * pat})

    cache = {}

    def run(cuts, field, output):
        key = (tuple(cuts), field, output)
        if key not in cache:
            cfg = IceConfig(whichEVP=1, adjoint_mode="dynamics")._replace(
                adjoint_damping="local", adjoint_kappa=0.25, adjoint_cuts=tuple(cuts))

            def f(st):
                return ice_step_mod.ice_surface_step(cfg, mesh, st, sf0, fs, dt=DT)

            out, tout = jax.jit(lambda st, t: jax.jvp(f, (st,), (t,)))(
                state, tangent_state(field))
            cache[key] = (out, np.asarray(getattr(tout, output)))
        return cache[key]

    fwd = {}

    def forward(cuts):
        if tuple(cuts) not in fwd:
            cfg = IceConfig(whichEVP=1, adjoint_mode="dynamics")._replace(
                adjoint_damping="local", adjoint_kappa=0.25, adjoint_cuts=tuple(cuts))
            fwd[tuple(cuts)] = jax.jit(
                lambda st: ice_step_mod.ice_surface_step(cfg, mesh, st, sf0, fs, dt=DT))(state)
        return fwd[tuple(cuts)]

    return run, forward


@pytest.mark.parametrize("cut", sorted(CUT_CHECKS))
def test_each_cut_zeroes_its_path_and_leaves_the_forward(ice_step_jvp, cut):
    """Each cut is one stop_gradient: the forward of the ice step is bitwise unchanged (CPU),
    the tangent from the input the cut path starts at into the output it ends in is non-zero
    without it and exactly zero with it (with ``extra`` closing the parallel paths)."""
    run, forward = ice_step_jvp
    field, output, extra = CUT_CHECKS[cut]
    ref = forward(())
    for a, b in zip(ref, forward((cut,))):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), cut
    _, t_open = run(extra, field, output)
    _, t_cut = run(extra + (cut,), field, output)
    assert np.all(np.isfinite(t_open)) and np.any(t_open != 0.0), \
        f"{cut}: no tangent from {field} into {output} even without the cut"
    assert np.all(t_cut == 0.0), f"{cut}: {np.max(np.abs(t_cut))!r} left from {field} to {output}"


def test_empty_cuts_trace_the_same_step(core2_setup):
    from fesom_jax import ice_step as ice_step_mod
    mesh, state, op, fs, sf0, wet0 = core2_setup

    def jaxpr(**kw):
        cfg = IceConfig(whichEVP=1, adjoint_mode="dynamics")._replace(**kw)
        return str(jax.make_jaxpr(
            lambda st: ice_step_mod.ice_surface_step(cfg, mesh, st, sf0, fs, dt=DT))(state))

    base = jaxpr()
    assert jaxpr(adjoint_cuts=()) == base
    assert jaxpr(adjoint_cuts=("ssh",)) != base


@pytest.mark.parametrize("cut", ["fct_limiter", "fct_flux"])
def test_fct_cuts_under_exact(core2_setup, cut):
    """``adjoint_cuts=("fct_limiter",)`` (the ice advection's Zalesak limiter factors enter the
    tangent as constants) and ``("fct_flux",)`` (the whole limited antidiffusive correction
    does, so the advection tangent is the low-order one) under "exact": the forward of the ice
    step is bitwise unchanged (CPU); the tangent from m_ice into the updated m_ice stays finite
    and changes, with the same tangent away from the ice; "prescribed" rejects the cut, which
    would act on a step it does not differentiate."""
    from fesom_jax import ice_step as ice_step_mod
    mesh, state, op, fs, sf0, wet0 = core2_setup
    icy = np.asarray(state.a_ice > 0.0)
    pat = jnp.asarray(np.where(icy, np.random.default_rng(3).standard_normal(mesh.nod2D), 0.0))
    tan = dataclasses.replace(jax.tree.map(jnp.zeros_like, state), m_ice=0.01 * pat)

    def run(cuts):
        cfg = IceConfig(whichEVP=1, adjoint_mode="exact")._replace(
            adjoint_damping="local", adjoint_kappa=0.25, adjoint_cuts=cuts)

        def f(st):
            return ice_step_mod.ice_surface_step(cfg, mesh, st, sf0, fs, dt=DT)
        return jax.jit(lambda st, t: jax.jvp(f, (st,), (t,)))(state, tan)

    (out0, t0), (out1, t1) = run(()), run((cut,))
    for a, b in zip(out0, out1):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    m0, m1 = np.asarray(t0.m_ice), np.asarray(t1.m_ice)
    assert np.all(np.isfinite(m0)) and np.all(np.isfinite(m1))
    changed = m0 != m1
    print(f"[{cut}] m_ice tangent changed on {changed.sum()} of {icy.sum()} icy nodes, "
          f"max |diff| {np.max(np.abs(m0 - m1)):.3e} of max {np.max(np.abs(m0)):.3e}")
    assert changed.any() and not changed.all()
    with pytest.raises(ValueError, match="adjoint_cuts"):
        ice_step_mod.validate_adjoint_settings(
            IceConfig(whichEVP=1, adjoint_mode="prescribed")._replace(
                adjoint_cuts=(cut,)))


THIN_MASS = 50.0     # kg/m² (the farc day-35.75 polynya node carried 15-35)


def _thin_thick_ice(state, mesh):
    """Ice of two loads on the seeded cover: 3 cm (load 27 kg/m² per ice area, above the 9 kg/m²
    floor of inv_thickness and below THIN_MASS) on a random half of the ice nodes, 1 m on the
    rest; snow 10 % of the ice."""
    a = state.a_ice
    rng = np.random.default_rng(5)
    h = jnp.asarray(np.where(rng.random(mesh.nod2D) < 0.5, 0.03, 1.0))
    return a, a * h, 0.1 * a * h, h


def test_thin_mass_cuts_the_load_derivative_on_thin_ice_only(core2_setup):
    """``IceConfig.adjoint_thin_mass``: in ``ice_mevp.mevp_setup`` the tangent of the inverse
    load (``inv_thickness``) and of the regularized mass is exactly zero on the ice nodes whose
    load (ρᵢ m + ρₛ m_snow)/a is below the threshold, and bitwise the uncut one everywhere else;
    the values are bitwise unchanged."""
    from fesom_jax import ice_evp, ice_mevp
    mesh, state, op, fs, sf0, wet0 = core2_setup
    a, m, ms, h = _thin_thick_ice(state, mesh)
    bn = ice_evp.boundary_node_mask(mesh)
    tan = jnp.asarray(np.random.default_rng(6).standard_normal(mesh.nod2D)) * m

    def run(thin):
        cfg = IceConfig(whichEVP=1, adjoint_mode="exact")._replace(adjoint_thin_mass=thin)

        def f(mm):
            st = ice_mevp.mevp_setup(cfg, mesh, a, mm, 0.1 * mm, state.eta_n, bn)
            return st.inv_thickness, st.mass
        return jax.jvp(f, (m,), (tan,))

    (p0, t0), (p1, t1) = run(0.0), run(THIN_MASS)
    c = IceConfig()
    load = np.asarray((c.rhoice * m + c.rhosno * ms) / jnp.where(a > 0, a, 1.0))
    icy = np.asarray(a >= 0.01)
    thin = icy & (load < THIN_MASS)
    assert thin.sum() > 100 and (icy & ~thin).sum() > 100, (thin.sum(), icy.sum())
    for name, x0, x1, d0, d1 in zip(("inv_thickness", "mass"), p0, p1, t0, t1):
        assert np.asarray(x0).tobytes() == np.asarray(x1).tobytes(), name
        d0, d1 = np.asarray(d0), np.asarray(d1)
        assert np.count_nonzero(d0[thin]) > 0.9 * thin.sum(), f"{name}: no tangent to cut"
        assert np.all(d1[thin] == 0.0), f"{name}: {np.max(np.abs(d1[thin]))!r} left on thin ice"
        assert d1[~thin].tobytes() == d0[~thin].tobytes(), f"{name}: changed off thin ice"


def test_thin_mass_zero_is_the_same_step_and_positive_keeps_the_forward(core2_setup):
    """``adjoint_thin_mass=0`` traces the unchanged ice step; a positive value traces a
    different one whose forward is bitwise the same (CPU)."""
    from fesom_jax import ice_step as ice_step_mod
    mesh, state, op, fs, sf0, wet0 = core2_setup
    a, m, ms, _ = _thin_thick_ice(state, mesh)
    st = dataclasses.replace(state, a_ice=a, m_ice=m, m_snow=ms)

    def cfg(**kw):
        return IceConfig(whichEVP=1, adjoint_mode="exact")._replace(
            adjoint_damping="frozen_sweep", adjoint_theta=0.9, **kw)

    def step(c):
        return lambda s: ice_step_mod.ice_surface_step(c, mesh, s, sf0, fs, dt=DT)

    base = str(jax.make_jaxpr(step(cfg()))(st))
    assert str(jax.make_jaxpr(step(cfg(adjoint_thin_mass=0.0)))(st)) == base
    assert str(jax.make_jaxpr(step(cfg(adjoint_thin_mass=THIN_MASS)))(st)) != base
    out0 = jax.jit(step(cfg()))(st)
    out1 = jax.jit(step(cfg(adjoint_thin_mass=THIN_MASS)))(st)
    for x0, x1 in zip(out0, out1):
        assert np.asarray(x0).tobytes() == np.asarray(x1).tobytes()


@pytest.mark.parametrize("kw", [dict(adjoint_thin_mass=-1.0),
                                dict(adjoint_thin_mass=float("nan")),
                                dict(adjoint_thin_mass="x"),
                                dict(adjoint_thin_mass=50.0, adjoint_mode="prescribed"),
                                dict(adjoint_thin_mass=50.0, whichEVP=0)],
                         ids=["negative", "nan", "text", "prescribed", "std_evp"])
def test_thin_mass_validation(kw):
    from fesom_jax import ice_step as ice_step_mod
    base = dict(whichEVP=1, adjoint_mode="exact")
    base.update(kw)
    with pytest.raises(ValueError, match="adjoint_thin_mass"):
        ice_step_mod.validate_adjoint_settings(IceConfig()._replace(**base))


def test_strength_conc_cuts_the_concentration_factor_of_the_strength(core2_setup):
    """``adjoint_cuts=("strength_conc",)``: in ``ice_mevp.mevp_setup`` the tangent of the
    strength (``pressure_fac``) from a_ice is exactly zero and the one from m_ice bitwise the
    uncut one; the values are bitwise unchanged. The forward of the ice step under "exact" +
    frozen_sweep is bitwise unchanged (CPU); standard EVP rejects the cut."""
    from fesom_jax import ice_evp, ice_mevp
    from fesom_jax import ice_step as ice_step_mod
    mesh, state, op, fs, sf0, wet0 = core2_setup
    a, m, ms, _ = _thin_thick_ice(state, mesh)
    bn = ice_evp.boundary_node_mask(mesh)
    rng = np.random.default_rng(8)
    tan = {"a": jnp.asarray(rng.standard_normal(mesh.nod2D)) * a,
           "m": jnp.asarray(rng.standard_normal(mesh.nod2D)) * m}

    def run(cuts, field):
        cfg = IceConfig(whichEVP=1, adjoint_mode="exact")._replace(adjoint_cuts=cuts)

        def f(x):
            aa, mm = (x, m) if field == "a" else (a, x)
            return ice_mevp.mevp_setup(cfg, mesh, aa, mm, ms, state.eta_n, bn).pressure_fac
        return jax.jvp(f, ((a if field == "a" else m),), (tan[field],))

    (p0, da0), (p1, da1) = run((), "a"), run(("strength_conc",), "a")
    (_, dm0), (_, dm1) = run((), "m"), run(("strength_conc",), "m")
    assert np.asarray(p0).tobytes() == np.asarray(p1).tobytes()
    assert np.count_nonzero(np.asarray(da0)) > 100, "no concentration tangent to cut"
    assert np.all(np.asarray(da1) == 0.0), np.max(np.abs(np.asarray(da1)))
    assert np.count_nonzero(np.asarray(dm0)) > 100
    assert np.asarray(dm0).tobytes() == np.asarray(dm1).tobytes()

    st = dataclasses.replace(state, a_ice=a, m_ice=m, m_snow=ms)

    def step(cuts):
        c = IceConfig(whichEVP=1, adjoint_mode="exact")._replace(
            adjoint_damping="frozen_sweep", adjoint_theta=0.9, adjoint_cuts=cuts)
        return jax.jit(lambda s: ice_step_mod.ice_surface_step(c, mesh, s, sf0, fs, dt=DT))(st)

    for x0, x1 in zip(step(()), step(("strength_conc",))):
        assert np.asarray(x0).tobytes() == np.asarray(x1).tobytes()
    with pytest.raises(ValueError, match="strength_conc"):
        ice_step_mod.validate_adjoint_settings(
            IceConfig(whichEVP=0, adjoint_mode="exact")._replace(adjoint_cuts=("strength_conc",)))


def test_sweep_guard_off_large_and_tripped(core2_setup):
    """``IceConfig.adjoint_sweep_guard`` (frozen_sweep): 0 traces the unchanged ice step; a large
    G leaves the forward and the VJP of a contracting step bitwise unchanged (CPU); a G below
    the first transposed sweep's growth trips, and the step's mEVP solve enters the tangent as a
    constant: no cotangent from the new u_ice into the entering stress (its only path) and the
    same as "dark" elsewhere. Damping other than frozen_sweep, G < 0 and a non-number raise."""
    from fesom_jax import ice_step as ice_step_mod
    mesh, state, op, fs, sf0, wet0 = core2_setup
    icy = np.asarray(state.a_ice > 0.0)
    w = jnp.asarray(np.where(icy, np.random.default_rng(9).standard_normal(mesh.nod2D), 0.0))

    def cfg(**kw):
        return IceConfig(whichEVP=1, adjoint_mode="exact")._replace(
            adjoint_damping="frozen_sweep", adjoint_theta=0.9, **kw)

    def step(c):
        return lambda s: ice_step_mod.ice_surface_step(c, mesh, s, sf0, fs, dt=DT)

    base = str(jax.make_jaxpr(step(cfg()))(state))
    assert str(jax.make_jaxpr(step(cfg(adjoint_sweep_guard=0.0)))(state)) == base

    def vjp(c):
        out, f_vjp = jax.vjp(step(c), state)
        return out, f_vjp(jax.tree.map(jnp.zeros_like, out)._replace(u_ice=w))[0]

    out0, g0 = jax.jit(lambda: vjp(cfg()))()
    out1, g1 = jax.jit(lambda: vjp(cfg(adjoint_sweep_guard=1e6)))()
    for x0, x1 in zip(out0, out1):
        assert np.asarray(x0).tobytes() == np.asarray(x1).tobytes()
    for f in ("sigma11", "u_ice", "a_ice", "m_ice"):
        assert np.asarray(getattr(g0, f)).tobytes() == np.asarray(getattr(g1, f)).tobytes(), f
    assert np.count_nonzero(np.asarray(g0.sigma11)) > 100, "no stress cotangent to drop"
    _, g2 = jax.jit(lambda: vjp(cfg(adjoint_sweep_guard=1e-12)))()
    assert np.all(np.asarray(g2.sigma11) == 0.0) and np.all(np.asarray(g2.u_ice) == 0.0)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(g2))

    for kw in (dict(adjoint_sweep_guard=-1.0), dict(adjoint_sweep_guard="x"),
               dict(adjoint_sweep_guard=float("inf"))):
        with pytest.raises(ValueError, match="adjoint_sweep_guard"):
            ice_step_mod.validate_adjoint_settings(cfg(**kw))
    with pytest.raises(ValueError, match="adjoint_sweep_guard"):
        ice_step_mod.validate_adjoint_settings(
            IceConfig(whichEVP=1, adjoint_mode="exact")._replace(adjoint_damping="local",
                                                                adjoint_sweep_guard=10.0))
