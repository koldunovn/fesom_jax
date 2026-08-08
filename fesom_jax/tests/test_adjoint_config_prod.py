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
from fesom_jax.adjoint_stab import AdjointConfig
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

    def make_kgm(cfg):
        """The same step, differentiated with respect to ``params.k_gm`` — the GM diffusivity,
        i.e. the parameter Arm A scales. Needed because ``AdjointConfig``'s claim about
        ``freeze_k33`` is about the TRAINED parameter's gradient, which an SST control cannot
        see."""
        def loss(kgm):
            p = dataclasses.replace(Params.defaults(), k_gm=kgm)
            fin = stepmod.step(state, mesh, op, None, p, dt=DT, is_first_step=True,
                               step_forcing=sf0, forcing_static=fs,
                               ale_cfg=AleConfig(), gm_cfg=GMConfig(), tke_cfg=TkeConfig(),
                               ice_cfg=IceConfig(whichEVP=1, adjoint_mode="rheology"),
                               adjoint_cfg=cfg)
            return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)

        return jax.jit(loss), jax.jit(jax.grad(loss))

    return make, jnp.asarray(0.0, jnp.float64), n_icy, make_kgm, Params.defaults().k_gm


@core2_only
def test_the_fixture_actually_runs_the_production_configuration(prod):
    """If the assembled step with all four subsystems does not produce a finite, nonzero
    gradient, nothing else in this file means anything."""
    make, k0, n_icy, _, _ = prod
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
def test_freeze_k33_is_not_free_with_respect_to_the_trained_parameter(prod):
    """``AdjointConfig``'s distinctive claim about ``freeze_k33``, tested against the gradient the
    claim is actually about.

    The docstring says K33 "cuts part of the gradient path of the eddy coefficients themselves —
    unlike the others, this one is not free with respect to the parameter being trained". That is
    a statement about **d(loss)/d(k_gm)**, not about d(loss)/d(SST).

    ⚠️ I first wrote this as "freeze_k33 cuts more than freeze_gm_slope" measured on the SST
    control, and the run falsified it: against SST, freeze_gm_slope moved the gradient 1.679e-03
    and freeze_k33 only 1.390e-03. Nothing ever implied that ordering — the two cut different
    paths, and an SST control cannot see the trained parameter's own path at all. The measurement
    was right and my reasoning was wrong.

    So: differentiate with respect to ``k_gm`` and assert the qualitative claim. ``freeze_k33``
    must move that gradient materially; ``freeze_mixing`` — documented as free with respect to it,
    because the eddy augmentation is added AFTER the closure runs — must move it far less.
    """
    _, _, _, make_kgm, kgm0 = prod
    _, g_off = make_kgm(AdjointConfig())
    _, g_k33 = make_kgm(AdjointConfig(freeze_k33=True))
    _, g_mix = make_kgm(AdjointConfig(freeze_mixing=True))
    base = float(g_off(kgm0))
    assert np.isfinite(base) and base != 0.0, \
        f"d(loss)/d(k_gm) is {base!r} — GM is not reaching the loss, so nothing here is testable"
    r_k33 = abs(float(g_k33(kgm0)) - base) / abs(base)
    r_mix = abs(float(g_mix(kgm0)) - base) / abs(base)
    print(f"\nd(loss)/d(k_gm) = {base!r}; freeze_k33 moves it {r_k33:.3e} relative, "
          f"freeze_mixing {r_mix:.3e}")
    assert r_k33 > 1e-6, (
        f"freeze_k33 changed d(loss)/d(k_gm) by only {r_k33:.3e} — it is supposed to cut part of "
        f"the trained parameter's own gradient path")
    assert r_k33 > r_mix, (
        f"freeze_k33 ({r_k33:.3e}) did not cut the trained parameter's gradient more than "
        f"freeze_mixing ({r_mix:.3e}), which AdjointConfig documents as free with respect to it")


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
    guarded = {f for f in AdjointConfig._fields if f in src}
    covered = set(PROD_ONLY_FLAGS)
    # the pi and TKE files cover these; listed explicitly so a new flag lands in neither set
    # and fails this test rather than disappearing
    elsewhere = {"freeze_mixing", "freeze_stratification", "freeze_convection",
                 "freeze_vert_visc", "freeze_tke_state", "freeze_advection"}
    assert guarded <= covered | elsewhere, (
        f"AdjointConfig flags read by step() but classified nowhere: "
        f"{sorted(guarded - covered - elsewhere)}")
    assert covered <= guarded, f"listed here but not read by step(): {sorted(covered - guarded)}"
