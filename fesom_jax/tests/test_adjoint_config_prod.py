"""The three ``AdjointConfig`` flags that only exist in the PRODUCTION configuration.

``test_adjoint_config.py`` runs the pi mesh (linfs, PP mixing, no GM) and
``test_adjoint_config_tke.py`` runs CORE2 with the TKE closure. Between them they leave three
flags with no effect-test anywhere, and the reason is structural rather than accidental — each
of the three sits inside a Python ``if`` that neither fixture enters:

===================== ============================== ==============================
flag                  guard                          what it freezes
===================== ============================== ==============================
``freeze_pgf``        ``if ale_cfg is not None:``    the zstar baroclinic PGF
``freeze_gm_slope``   ``if gm_cfg is not None:``     N² inside the eddy block
``freeze_k33``        ``if gm_cfg is not None:``     the GM/Redi vertical augmentation
===================== ============================== ==============================

A flag wired to a branch the tests never enter cannot be distinguished from a flag wired
nowhere. Both return the baseline gradient bit-for-bit, and the report that comes out of a
horizon campaign is then "this term is not the amplifier" for a term that was never cut —
exactly the failure that ``docs/ADJOINT_HORIZON.md`` had to be corrected for after it claimed
every flag was covered.

Production runs **zstar + TKE + mEVP + GM**, so that is what the fixture here builds. It is the
most expensive fixture in the adjoint set (one assembled CORE2 step with all four subsystems, and
a separate ``jax.jit`` closure per configuration), which is why it lives in its own file and
tests exactly the three flags that need it.

⚠️ Two traps this fixture must avoid, both of which have already caught this project once:
the TKE must be **seeded** (``phc_initial_state`` carries ``tke = 0`` and ``KappaH`` is
proportional to it, so a cold column makes every derivative through the closure exactly zero),
and the ice must be **seeded** (otherwise ``a_ice`` is identically zero and the sea-ice adjoint
is being tested on an ice-free ocean). Both are asserted, not assumed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ssh, surface_forcing
from fesom_jax import ice as icemod
from fesom_jax import step as stepmod
from fesom_jax.adjoint_stab import FREEZE_FLAGS, AdjointConfig
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.tests._identity_gates import assert_forward_identity
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958
TKE_SEED = 1.0e-3

core2_only = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing")

# The three flags this file exists for, and the guard each one needs.
PROD_ONLY_FLAGS = ["freeze_pgf", "freeze_gm_slope", "freeze_k33"]


@pytest.fixture(scope="module")
def prod():
    """The production configuration — zstar + TKE + mEVP + GM — as a function of a uniform
    surface-temperature perturbation.

    The control is an SST perturbation for the reason spelled out in
    ``test_adjoint_config_tke.py``: a mixing parameter is not read on the TKE path at all, and
    ``freeze_mixing`` would stop_gradient it anyway, so any test built on one compares zero with
    zero. A surface perturbation reaches T through stratification→Kv, through advection, through
    the pressure gradient and through the eddy block, so no single flag can sever the whole path.
    """
    mesh = load_mesh(MESH_DIR)
    bare = phc_initial_state(mesh, IC_DIR)

    imask = np.asarray(mesh.node_iface_mask)
    state = dataclasses.replace(
        bare, tke=jnp.asarray(np.where(imask, TKE_SEED, 0.0), jnp.float64))
    state = icemod.seed_ice(state, mesh, state.T[:, 0])

    # assert the two seedings worked, rather than trusting them: a silently unseeded fixture
    # measures a switched-off subsystem and PASSES.
    assert float(jnp.max(state.tke)) > 0.0, "TKE seeding did nothing"
    n_icy = int(np.count_nonzero(np.asarray(state.a_ice) > 0.0))
    assert n_icy > 0, "seed_ice produced no ice — the sea-ice path is not exercised"

    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    fs = cf.static
    sf0 = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])

    def make(cfg):
        def loss(eps):
            st = dataclasses.replace(state,
                                     T=state.T.at[:, 0].add(jnp.where(wet0, eps, 0.0)))
            fin = stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT,
                               is_first_step=True, step_forcing=sf0, forcing_static=fs,
                               ale_cfg=AleConfig(), gm_cfg=GMConfig(), tke_cfg=TkeConfig(),
                               ice_cfg=IceConfig(whichEVP=1, adjoint_mode="rheology"),
                               adjoint_cfg=cfg)
            return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)

        return jax.jit(loss), jax.jit(jax.grad(loss))

    def make_param(cfg, which):
        """The same step, differentiated with respect to one GM ``Params`` leaf.

        ``which`` is ``"k_gm"`` or ``"redi_kmax"``. Both exist because **they are separate
        differentiable leaves in this port** and they feed different terms: ``k_gm`` sets
        ``fer_K`` (the bolus coefficient) and ``redi_kmax`` sets ``Ki`` (the isoneutral
        diffusivity), from which ``K33`` is built. The C auto-syncs ``Redi_Kmax = K_GM_max``;
        ``Params`` deliberately does not (see ``calibrate.build_params``), and ``gm_nn`` emits a
        separate multiplier for each — so which leaf you differentiate is a real distinction, not
        a detail."""
        def loss(val):
            p = dataclasses.replace(Params.defaults(), **{which: val})
            fin = stepmod.step(state, mesh, op, None, p, dt=DT, is_first_step=True,
                               step_forcing=sf0, forcing_static=fs,
                               ale_cfg=AleConfig(), gm_cfg=GMConfig(), tke_cfg=TkeConfig(),
                               ice_cfg=IceConfig(whichEVP=1, adjoint_mode="rheology"),
                               adjoint_cfg=cfg)
            return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)

        return jax.jit(loss), jax.jit(jax.grad(loss))

    return (make, jnp.asarray(0.0, jnp.float64), n_icy, make_param,
            Params.defaults().k_gm, Params.defaults().redi_kmax)


@core2_only
def test_the_fixture_actually_runs_the_production_configuration(prod):
    """If the assembled step with all four subsystems does not produce a finite, nonzero
    gradient, nothing else in this file means anything."""
    make, k0, n_icy, *_ = prod
    loss, grad = make(AdjointConfig())
    l0, g0 = float(loss(k0)), float(grad(k0))
    assert np.isfinite(l0) and np.isfinite(g0)
    assert g0 != 0.0, "the SST control does not reach the loss in this configuration"
    print(f"\nproduction baseline: loss {l0!r} grad {g0!r} ({n_icy} icy nodes)")


@core2_only
@pytest.mark.parametrize("flag", PROD_ONLY_FLAGS)
def test_the_production_only_flag_changes_the_gradient(prod, flag):
    """The point of the file. Each flag must MOVE the gradient here, which is the only evidence
    that it is wired to a live call site rather than to a branch nothing enters."""
    make, k0, *_ = prod
    _, g_off = make(AdjointConfig())
    _, g_on = make(AdjointConfig(**{flag: True}))
    a, b = float(g_off(k0)), float(g_on(k0))
    assert np.isfinite(a) and np.isfinite(b)
    assert a != b, (f"{flag}=True returned the baseline gradient ({a!r}) bit-for-bit in the "
                   f"production configuration — it is not reaching the graph")
    print(f"\n{flag}: {a!r} -> {b!r}  (relative {abs(b - a) / abs(a):.3e})")


@core2_only
@pytest.mark.parametrize("flag", PROD_ONLY_FLAGS)
def test_the_production_only_flag_leaves_the_forward_alone(prod, flag):
    """``stop_gradient`` is the identity in the forward. These three are the flags that touch
    the PGF and the eddy block — the terms that most directly move the trajectory — so this is
    where a wiring mistake would change the physics of a production run rather than only its
    gradient."""
    make, k0, *_ = prod
    l_off, _ = make(AdjointConfig())
    l_on, _ = make(AdjointConfig(**{flag: True}))
    assert_forward_identity(l_off, l_on, k0, f"{flag}=True in the production configuration")


@core2_only
def test_gm_n2_cut_everywhere_is_freeze_gm_slope(prod):
    """The masked GM-input cut is wired to the same N² as ``freeze_gm_slope``: with a raw
    threshold above every N² the mask covers the whole field and the gradient is the
    ``freeze_gm_slope`` one; the production values (R* 100, raw* -1e-6) leave the forward
    alone and give a finite gradient."""
    make, k0, *_ = prod
    _, g_off = make(AdjointConfig())
    _, g_frz = make(AdjointConfig(freeze_gm_slope=True))
    _, g_all = make(AdjointConfig(gm_n2_raw_cut=1.0))
    a, b, c = float(g_off(k0)), float(g_frz(k0)), float(g_all(k0))
    print(f"\nbaseline {a!r} freeze_gm_slope {b!r} gm_n2_raw_cut=1 {c!r}")
    assert c != a
    np.testing.assert_allclose(c, b, rtol=1e-12, atol=0.0)
    l_off, _ = make(AdjointConfig())
    l_cut, g_cut = make(AdjointConfig(gm_n2_ratio=100.0, gm_n2_raw_cut=-1e-6))
    assert_forward_identity(l_off, l_cut, k0, "gm_n2_ratio/gm_n2_raw_cut in the production "
                                              "configuration")
    assert np.isfinite(float(g_cut(k0)))


@core2_only
def test_gm_named_n2_cuts_together_are_freeze_gm_slope(prod):
    """The three named N² cuts of the eddy block (slope, wave speed, streamfunction solve)
    cover every N² use there: together they give the ``freeze_gm_slope`` gradient. Each one
    alone and the steep-slope cut move the gradient and leave the forward alone."""
    make, k0, *_ = prod
    l_off, g_off = make(AdjointConfig())
    _, g_frz = make(AdjointConfig(freeze_gm_slope=True))
    three = frozenset({"gm.slope_n2", "gm.cm_n2", "gm.bvp_n2"})
    _, g_all = make(AdjointConfig(diag_cuts=three))
    a, b, c = float(g_off(k0)), float(g_frz(k0)), float(g_all(k0))
    print(f"\nbaseline {a!r} freeze_gm_slope {b!r} three named cuts {c!r}")
    np.testing.assert_allclose(c, b, rtol=1e-12, atol=0.0)
    for cfg in [AdjointConfig(diag_cuts=frozenset({n})) for n in sorted(three)] + [
            AdjointConfig(gm_slope_cut=2e-3)]:
        l_on, g_on = make(cfg)
        v = float(g_on(k0))
        print(f"{cfg.describe()}: {v!r} (relative {abs(v - a) / abs(a):.3e})")
        assert np.isfinite(v) and v != a, cfg.describe()
        assert_forward_identity(l_off, l_on, k0, f"{cfg.describe()} in the production "
                                                 "configuration")


@core2_only
def test_freeze_k33_cuts_redi_kmax_and_is_exactly_free_for_k_gm(prod):
    """WHICH trained parameter ``freeze_k33`` is not free with respect to — and it is not the one
    the name GM suggests.

    ``AdjointConfig`` says K33 "cuts part of the gradient path of the eddy coefficients themselves
    — unlike the others, this one is not free with respect to the parameter being trained". The
    coefficients are two separate ``Params`` leaves in this port, and they feed different terms:

        k_gm      -> fer_K, the bolus coefficient
        redi_kmax -> Ki, the isoneutral diffusivity, from which K33 is built (gm.py: ki_top,
                     then k33_augmentation(mesh, slope_tapered, Ki, ...))

    So ``d(K33)/d(k_gm)`` is **structurally zero** and ``freeze_k33`` is *exactly* free for
    ``k_gm``; the leaf it bites is ``redi_kmax``. Both directions are asserted here, because the
    exact zero is the stronger statement and rounding cannot fake it.

    ⚠️ Two earlier versions of this test were falsified by their own runs, which is why the whole
    story is written out:
      1. "freeze_k33 cuts more than freeze_gm_slope", measured on an SST control — against SST it
         moved the gradient 1.390e-03 against freeze_gm_slope's 1.679e-03. Nothing implied that
         ordering, and an SST control cannot see a trained parameter's own path.
      2. "freeze_k33 moves d(loss)/d(k_gm)" — measured **0.000e+00**, for the structural reason
         above. That is the correct answer to a question about the wrong leaf.
    The C auto-syncs ``Redi_Kmax = K_GM_max``, which is probably what made "the GM coefficient"
    feel like one parameter; ``Params`` deliberately keeps them separate and ``gm_nn`` emits a
    multiplier for each, so Task 9 must treat them as two.
    """
    _, _, _, make_param, kgm0, redi0 = prod

    _, g_gm_off = make_param(AdjointConfig(), "k_gm")
    _, g_gm_k33 = make_param(AdjointConfig(freeze_k33=True), "k_gm")
    base_gm = float(g_gm_off(kgm0))
    assert np.isfinite(base_gm) and base_gm != 0.0, \
        f"d(loss)/d(k_gm) is {base_gm!r} — the bolus path is not reaching the loss"
    d_gm = abs(float(g_gm_k33(kgm0)) - base_gm)
    print(f"\nd(loss)/d(k_gm)      = {base_gm!r}; freeze_k33 moves it {d_gm:.3e} (expect 0.0)")
    assert d_gm == 0.0, (
        f"freeze_k33 moved d(loss)/d(k_gm) by {d_gm:.3e}. K33 is built from Ki, which comes from "
        f"redi_kmax, so this must be EXACTLY zero — a nonzero value means k_gm has acquired a "
        f"path into K33 and the two leaves are no longer independent")

    _, g_rd_off = make_param(AdjointConfig(), "redi_kmax")
    _, g_rd_k33 = make_param(AdjointConfig(freeze_k33=True), "redi_kmax")
    base_rd = float(g_rd_off(redi0))
    assert np.isfinite(base_rd) and base_rd != 0.0, \
        f"d(loss)/d(redi_kmax) is {base_rd!r} — the isoneutral path is not reaching the loss"
    r_rd = abs(float(g_rd_k33(redi0)) - base_rd) / abs(base_rd)
    print(f"d(loss)/d(redi_kmax) = {base_rd!r}; freeze_k33 moves it {r_rd:.3e} relative")
    assert r_rd > 1e-6, (
        f"freeze_k33 changed d(loss)/d(redi_kmax) by only {r_rd:.3e} — this IS the leaf whose "
        f"gradient path it cuts, so it must move materially")


@core2_only
def test_freeze_pgf_reaches_the_gradient_at_all(prod):
    """``freeze_pgf`` removes the whole baroclinic pressure-gradient path from the backward, and
    ``AdjointConfig`` calls the resulting gradient unusable for training. **Over ONE step that
    still shows up as a tiny number, and that is correct, not a bug.**

    Measured on this fixture: +7.919790246248722e-01 → +7.91978909160614e-01, i.e. 1.46e-07
    relative. The reason it is small has nothing to do with wiring: an SST perturbation reaches
    momentum through the PGF, and momentum can only get back into T through advection *within
    the same step*, so at one step the PGF contributes to dT/dSST at second order. Over a
    30-day window that path is the tracer→momentum→tracer feedback and it dominates — which is
    what "unusable for training" refers to.

    So this test asserts the flag REACHES the graph (1.46e-07 is 145× the 1e-9 floor and
    infinitely more than the exact 0.0 it returned before ``ale_cfg`` was passed), and does not
    pretend a one-step fixture can measure how blunt the cut is. Doing that needs a multi-step
    window, which belongs in the horizon campaign, not in a unit test.
    """
    make, k0, *_ = prod
    _, g_off = make(AdjointConfig())
    _, g_pgf = make(AdjointConfig(freeze_pgf=True))
    base, cut = float(g_off(k0)), float(g_pgf(k0))
    rel = abs(cut - base) / abs(base)
    print(f"\nfreeze_pgf: {base!r} -> {cut!r} (relative {rel:.3e})")
    assert rel > 1e-9, (
        f"freeze_pgf changed the gradient by only {rel:.3e} relative — below the 1e-9 floor, "
        f"which for a one-step fixture means the zstar branch is probably not the one being "
        f"taken, i.e. ale_cfg is not reaching step()")


@core2_only
def test_every_production_only_flag_is_listed_here():
    """Keeps this file honest as the flag set grows: any flag guarded by ale_cfg or gm_cfg must
    appear above, or it silently returns to having no effect-test at all."""
    import inspect

    from fesom_jax import step as _step
    src = inspect.getsource(_step.step)
    guarded = {f for f in FREEZE_FLAGS if f in src}
    covered = set(PROD_ONLY_FLAGS)
    # the pi and TKE files cover these; listed explicitly so a new flag lands in neither set
    # and fails this test rather than disappearing
    elsewhere = {"freeze_mixing", "freeze_stratification", "freeze_convection",
                 "freeze_vert_visc", "freeze_tke_state", "freeze_advection",
                 # the FCT limiter factors: test_adjoint_config.py (pi) and test_tracers.py
                 "freeze_fct_limiter",
                 # the TKE-closure flag: test_adjoint_config_tke.py (TKE_PATH_FLAGS) and
                 # test_tke_candidates{,_core2}.py
                 "freeze_mixing_state",
                 # ⚠️ covered by a THIRD file, which this set did not know about:
                 # test_gm_nn_seam.py::test_freeze_nn_features_is_forward_identical_and_keeps_
                 # the_weight_gradient. It cannot go in PROD_ONLY_FLAGS — it is read only inside
                 # `if nn is not None` (gm.py:501), and the production fixture here attaches no
                 # network, so the effect test would fail against a knob with no code path, which
                 # is exactly what happened to freeze_pgf on the pi fixture on 2026-08-07.
                 #
                 # It was added by the seam work and classified in test_adjoint_config.py's
                 # INACTIVE_ON_PI on 2026-08-10 (8f89684) — and THIS registry was missed, so the
                 # core2 group stayed red. One registry describing another's contents drifts; a
                 # flag now needs an entry in three places and only two of them fail loudly.
                 "freeze_nn_features"}
    assert guarded <= covered | elsewhere, (
        f"AdjointConfig flags read by step() but classified nowhere: "
        f"{sorted(guarded - covered - elsewhere)}")
    assert covered <= guarded, f"listed here but not read by step(): {sorted(covered - guarded)}"
