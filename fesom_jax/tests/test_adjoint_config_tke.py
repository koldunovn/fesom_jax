"""``AdjointConfig`` on the TKE path — the configuration the flags actually exist for.

``test_adjoint_config.py`` exercises the wiring on the packaged `pi` mesh, but that mesh runs
PP mixing: TKE is a forced-path feature (its surface flux needs the wind stress at nodes) and
raises on the pi path. So the pi tests cannot touch the term that motivated all of this — the
prognostic turbulence closure's response to the ocean state, which on the realistic
configuration is what makes the reverse pass grow by ~1.3x per timestep.

This file closes that gap in two stages, cheapest first:

* **Closure level, pi mesh, no forcing needed.** ``mixing_tke`` can be called directly from a
  rest state with a seeded turbulent kinetic energy and a synthetic node stress (the
  ``test_sensitivity.py`` pattern). That is enough to show the closure *is* sensitive to
  stratification and shear — i.e. that there is a real feedback loop to cut — and to test the
  convection flag against the term it targets. Runs anywhere, in seconds.

* **Step level, CORE2, one step.** The real assembled TKE step with JRA55 forcing, checking
  that each flag leaves the forward bit-identical and that the ones on the TKE path move the
  gradient. Skipped when the CORE2 mesh or its initial condition is not on disk.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ale, ssh, surface_forcing
from fesom_jax import step as stepmod
from fesom_jax.adjoint_stab import AdjointConfig
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.state import State
from fesom_jax.tke import TkeConfig, mixing_tke

RNG = np.random.default_rng(0)


# ===========================================================================
# 1. closure level — is there a feedback loop to cut at all?
# ===========================================================================

@pytest.fixture(scope="module")
def closure():
    """A callable ``(bvfreq, uv, tke, freeze_convection) -> (Kv, Av)`` on the pi mesh.

    Rest state plus a seeded tke>0 so the closure engages (with tke==0 the mixing length and
    hence every derivative collapses to zero and the test would pass vacuously)."""
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    mesh = load_mesh()
    st = State.rest(mesh)
    imask = np.asarray(mesh.node_iface_mask)
    tke_seed = jnp.asarray(np.where(imask, 1.0e-3, 0.0))
    stress_node = jnp.asarray(RNG.standard_normal((mesh.nod2D, 2)) * 0.05)
    _, Z3d = ale.live_geometry(mesh, st.hnode)
    # a stratification field with both signs, so the N^2<0 convective switch is live on part
    # of the domain — a strictly stable field would leave the convection flag nothing to do
    bv = jnp.asarray(np.where(imask, RNG.standard_normal(imask.shape) * 1.0e-4, 0.0))

    def run(bvfreq, freeze_convection=False):
        Kv, Av, _, _ = mixing_tke(mesh, st.uv, bvfreq, tke_seed, stress_node, st.hnode,
                                  TkeConfig(), Params.defaults(), dt=1800.0, Z3d=Z3d,
                                  freeze_convection=freeze_convection)
        return Kv, Av

    return dict(mesh=mesh, run=run, bv=bv)


def test_closure_is_sensitive_to_stratification(closure):
    """The premise of everything else: the TKE closure's coefficients depend on the ocean
    state strongly enough to matter. If this were zero there would be no feedback loop, no
    amplification, and no reason for any of these flags to exist."""
    run, bv = closure["run"], closure["bv"]
    g = np.asarray(jax.grad(lambda b: jnp.sum(run(b)[0]))(bv))
    assert np.all(np.isfinite(g)), "the closure's dKv/dN^2 has non-finite entries"
    assert np.count_nonzero(g) > 0, "dKv/dN^2 is identically zero — the closure is not engaged"


def test_convection_flag_keeps_the_value_and_drops_the_switch(closure):
    """The convective bump must survive in the forward and vanish from the backward.

    This is a ``freeze_increment``, not a plain ``stop_gradient``: the closure's own
    state-dependence stays differentiated and only the ``N^2 < 0`` switch is cut, so the
    gradient must change without becoming zero."""
    run, bv = closure["run"], closure["bv"]
    Kv_exact, Av_exact = run(bv, freeze_convection=False)
    Kv_frozen, Av_frozen = run(bv, freeze_convection=True)
    assert np.array_equal(np.asarray(Kv_exact), np.asarray(Kv_frozen)), "forward Kv moved"
    assert np.array_equal(np.asarray(Av_exact), np.asarray(Av_frozen)), "forward Av moved"

    g_exact = np.asarray(jax.grad(lambda b: jnp.sum(run(b, False)[0]))(bv))
    g_frozen = np.asarray(jax.grad(lambda b: jnp.sum(run(b, True)[0]))(bv))
    assert np.all(np.isfinite(g_frozen))
    assert not np.array_equal(g_exact, g_frozen), \
        "freeze_convection did not change the gradient — is the N^2<0 branch reached?"
    assert np.count_nonzero(g_frozen) > 0, \
        "freeze_convection zeroed the whole gradient; it must cut only the switch"


# ===========================================================================
# 2. step level — the real assembled TKE step on CORE2
# ===========================================================================

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958

core2_only = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing")

# Live on the TKE path WITHIN A SINGLE STEP. `freeze_tke_state` is deliberately absent: it
# stop_gradients `tke_new`, the prognostic TKE handed to the NEXT step, which a one-step fixture
# produces and never consumes. Measured 2026-08-07 — one step: 1.11e-16 (one ulp, i.e. nothing);
# two steps: 2.18e-04. It gets its own two-step test below; no control variable can rescue it
# here. The remaining flags need the eddy parameterization (off here) or are covered on pi.
TKE_PATH_FLAGS = ["freeze_mixing", "freeze_stratification", "freeze_convection",
                  "freeze_vert_visc"]

TKE_SEED = 1.0e-3          # the value the closure-level fixture above uses, for the same reason


def _build_core2(n_steps=1):
    """CORE2 TKE step(s) as a function of a uniform SURFACE-TEMPERATURE perturbation.

    ⚠️ Two things here are load-bearing and both were got wrong originally (measured 2026-08-07,
    identical on CPU and GPU).

    **The TKE must be seeded.** ``phc_initial_state`` carries ``tke = 0`` (``state.py:81`` — the
    C cold-start ``calloc`` mirror), ``KappaH`` is proportional to the turbulent kinetic energy,
    so a cold column emits ``Kv/Av = 0`` and EVERY derivative through the closure is exactly
    zero. With the cold state all five flags returned the baseline gradient to the last bit —
    the tests were measuring a closure that was switched off. This is the same trap the
    ``closure`` fixture above documents ("with tke==0 ... every derivative collapses to zero and
    the test would pass vacuously"); the step-level fixture walked into it.

    **The control cannot be a mixing parameter.** ``k_ver`` is not read at all on the TKE path
    (``step.py:320`` — only the PP ``else:`` branch reads it), so d/d(k_ver) is exactly 0 here.
    And ``freeze_mixing`` stop_gradients Kv/Av (``step.py:331``), so ``tke_c_k`` dies too: seeded,
    it is −2.349392554415e−02 with no flags and exactly 0.0 under ``freeze_mixing``. A surface
    perturbation survives, reaching T through stratification→Kv and through advection/PGF.
    Seeded, with this control: baseline +8.081567181611e−01, freeze_mixing/freeze_stratification
    +8.070188136225e−01, freeze_vert_visc Δ8.79e−07, freeze_convection Δ3.76e−09."""
    mesh = load_mesh(MESH_DIR)
    state = phc_initial_state(mesh, IC_DIR)
    imask = np.asarray(mesh.node_iface_mask)
    state = dataclasses.replace(
        state, tke=jnp.asarray(np.where(imask, TKE_SEED, 0.0), jnp.float64))
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    fs = cf.static
    sfs = [cf.step_forcing(*d) for d in surface_forcing.dates_for_steps(YEAR, DT, n_steps)]
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])

    def make(cfg):
        def loss(eps):
            st = dataclasses.replace(state, T=state.T.at[:, 0].add(jnp.where(wet0, eps, 0.0)))
            for i, sf in enumerate(sfs):
                st = stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT,
                                  is_first_step=(i == 0), step_forcing=sf, forcing_static=fs,
                                  tke_cfg=TkeConfig(), adjoint_cfg=cfg)
            return jnp.sum(jnp.where(wet0, st.T[:, 0], 0.0)) / jnp.sum(wet0)
        return jax.jit(loss), jax.jit(jax.grad(loss))

    return make, jnp.asarray(0.0, jnp.float64)   # control point: unperturbed initial state


@pytest.fixture(scope="module")
def core2_tke():
    return _build_core2(n_steps=1)


@pytest.fixture(scope="module")
def core2_tke_2step():
    return _build_core2(n_steps=2)


@core2_only
def test_core2_tke_default_is_a_dead_branch(core2_tke):
    make, k0 = core2_tke
    l_none, g_none = make(None)
    l_off, g_off = make(AdjointConfig())
    assert float(l_none(k0)) == float(l_off(k0))
    assert float(g_none(k0)) == float(g_off(k0))


@core2_only
@pytest.mark.parametrize("flag", AdjointConfig._fields)
def test_core2_tke_forward_bit_identical(core2_tke, flag):
    """The production configuration's forward must not move under any flag.

    ⚠️ EXACT on CPU, tolerance on GPU, and that is not a fudge. Each ``make()`` builds a fresh
    ``jax.jit`` closure, so this compares SEPARATELY COMPILED instances of the same program, and
    on GPU those can differ by an ulp from reduction-order alone. Measured 2026-08-07: setting
    ``freeze_gm_slope`` — read only inside ``if gm_cfg is not None:`` (``step.py:259``), a Python
    branch this fixture never enters, so the flag cannot reach the traced graph at all — still
    moved the GPU forward by exactly one ulp, while a different, arbitrary subset of flags
    "moved" on each run. An inert flag cannot change a computation: the differences come from
    compilation, not from the flags. Byte-identity is a CPU property
    (``docs/PORTING_LESSONS.md``, 2026-06); asserting it on GPU tests the compiler, not the port.
    """
    make, k0 = core2_tke
    l_off, _ = make(AdjointConfig())
    l_on, _ = make(AdjointConfig(**{flag: True}))
    a, b = float(l_off(k0)), float(l_on(k0))
    if jax.default_backend() == "cpu":
        assert a == b, f"{flag}=True moved the CORE2 TKE forward: {a!r} vs {b!r}"
    else:
        assert a == pytest.approx(b, rel=1e-14, abs=0.0), \
            f"{flag}=True moved the CORE2 TKE forward beyond recompile noise: {a!r} vs {b!r}"


@core2_only
@pytest.mark.parametrize("flag", TKE_PATH_FLAGS)
def test_core2_tke_flag_changes_the_gradient(core2_tke, flag):
    """Each flag that is live on the TKE path must actually reach it. A flag that silently
    does nothing here would report 'this term is not the amplifier' for a term never cut."""
    make, k0 = core2_tke
    _, g_off = make(AdjointConfig())
    _, g_on = make(AdjointConfig(**{flag: True}))
    a, b = float(g_off(k0)), float(g_on(k0))
    assert np.isfinite(a) and np.isfinite(b)
    assert a != b, f"{flag}=True did not change the CORE2 TKE gradient ({a:.6e})"


@core2_only
def test_core2_tke_state_flag_needs_two_steps(core2_tke, core2_tke_2step):
    """``freeze_tke_state`` cuts the gradient of ``tke_new`` — the prognostic TKE handed to the
    NEXT step. Over one step that output is produced and never consumed, so the flag provably
    cannot do anything; over two it must. Both halves are asserted, because the one-step half is
    what stops someone "fixing" the two-step fixture back into a one-step one.

    Measured 2026-08-07: 1 step 1.11e-16 (one ulp of the +8.08e-01 baseline, i.e. nothing);
    2 steps +7.400086138349e-01 -> +7.397908554600e-01, a difference of 2.18e-04."""
    make1, k1 = core2_tke
    _, g1_off = make1(AdjointConfig())
    _, g1_on = make1(AdjointConfig(freeze_tke_state=True))
    a1, b1 = float(g1_off(k1)), float(g1_on(k1))
    assert abs(a1 - b1) <= 4.0 * np.spacing(abs(a1)), \
        f"freeze_tke_state moved the ONE-step gradient by {abs(a1 - b1):.3e} — more than " \
        f"rounding. The carry it freezes should be unread after a single step; re-derive."

    make2, k2 = core2_tke_2step
    _, g2_off = make2(AdjointConfig())
    _, g2_on = make2(AdjointConfig(freeze_tke_state=True))
    a2, b2 = float(g2_off(k2)), float(g2_on(k2))
    assert np.isfinite(a2) and np.isfinite(b2)
    assert abs(a2 - b2) > 1e-6 * abs(a2), \
        f"freeze_tke_state did not change the TWO-step gradient ({a2:.6e} vs {b2:.6e}) — the " \
        f"inter-step TKE carry is not reaching the loss, so the flag is untested"


@core2_only
def test_core2_tke_frozen_mixing_is_finite_and_signed(core2_tke):
    """The headline configuration. Over a single step the closure's feedback has had almost
    no time to act, so cutting it must not flip the sign — a sign change here would mean the
    cut removed signal rather than stiffness."""
    make, k0 = core2_tke
    _, g_off = make(AdjointConfig())
    _, g_on = make(AdjointConfig(freeze_mixing=True))
    a, b = float(g_off(k0)), float(g_on(k0))
    assert np.isfinite(b) and b != 0.0
    assert np.sign(a) == np.sign(b), f"exact {a:+.3e} vs mixing-frozen {b:+.3e}: sign flipped"
