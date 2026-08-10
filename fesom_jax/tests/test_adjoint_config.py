"""Tests for the per-term backward-only simplifications (``AdjointConfig``).

Runs the assembled model on the packaged `pi` mesh — small, real FESOM geometry, CPU-fast.

Three properties are checked, and the first is the one everything else rests on.

1. **All flags off changes nothing.** ``adjoint_cfg=None`` and ``AdjointConfig()`` must give a
   bit-identical forward *and* a bit-identical gradient to the model as it was before this
   feature existed. If that ever breaks, every previously validated result is in question,
   which is why it is tested per-flag and not just for the default object.

2. **Any flag on leaves the FORWARD bit-identical.** ``stop_gradient`` is the identity in the
   forward, so switching a term into "frozen" must not move the trajectory by a single bit.
   This is the whole justification for using these in production: the model you run is still
   the validated model, only the gradient changes.

3. **A flag that is on actually does something to the GRADIENT.** A flag wired to a call site
   that is not on the active code path would silently do nothing, and a knob that silently
   does nothing is worse than no knob — it produces "that term is not the problem" for a term
   that was never tested. Each flag is therefore asserted to change the gradient.

The pi configuration runs PP mixing rather than the prognostic TKE closure, so the flags that
only exist on the TKE path are exercised for wiring, not for effect, and are marked as such.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ssh
from fesom_jax.adjoint_stab import AdjointConfig
from fesom_jax.config import A_VER
from fesom_jax import forcing, ic
from fesom_jax.integrate import integrate
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.tests._identity_gates import (assert_dead_branch_gradient,
                                             assert_forward_identity)

DT = 100.0
N = 8            # enough steps for the mixing/convection feedback to reach the loss


@pytest.fixture(scope="module")
def setup():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    mesh = load_mesh()
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st0 = ic.initial_state(mesh)
    return mesh, op, stress, st0


K_VER_LIFTED = jnp.asarray(1.0e-4, jnp.float64)   # signal-lifted, as in test_gradient.py


def _loss_fn(setup, cfg):
    """mean SST after N steps, as a function of a uniform SURFACE-TEMPERATURE perturbation.

    ⚠️ The control is the initial STATE, not ``k_ver``, and that is load-bearing. ``k_ver``
    reaches the loss only through Kv/Av (``step.py:320``, the PP branch), and ``freeze_mixing``
    puts ``stop_gradient`` on exactly those (``step.py:331``) — so d(loss)/d(k_ver) is EXACTLY
    zero under that flag, at any number of steps, and any test comparing gradients under it is
    comparing zero with zero. Measured 2026-08-07 on this fixture: baseline −1.084736e−03,
    freeze_mixing +0.0. That is what broke ``test_flags_compose`` (it got ab == a == 0) and
    ``test_frozen_mixing_gradient_stays_sane`` (it asserts nonzero). Swapping in a *different*
    mixing parameter does not help — they all die the same way.

    A surface-temperature perturbation reaches T through stratification→Kv AND through
    advection and the pressure gradient, so no single flag can sever the whole path. Measured
    with this control: baseline +8.207637e−01, freeze_mixing +8.138278e−01 (same sign, nonzero),
    and mixing / advection / both give three distinct values."""
    mesh, op, stress, st0 = setup
    p = Params(k_ver=K_VER_LIFTED, a_ver=jnp.asarray(A_VER, jnp.float64))

    def loss(eps):
        st = dataclasses.replace(st0, T=st0.T.at[:, 0].add(eps))
        fin = integrate(st, mesh, op, stress, n_steps=N, params=p, dt=DT, adjoint_cfg=cfg)
        return jnp.mean(fin.T[:, 0])

    return jax.jit(loss), jax.jit(jax.grad(loss))


K0 = jnp.asarray(0.0, jnp.float64)        # control point: the unperturbed initial state

# Every flag, and whether the pi/PP configuration actually exercises it. The TKE-only and
# GM-only flags are wired through the same seam but their code path is inactive here, so
# they are checked for "does not break anything" rather than "changes the gradient".
ALL_FLAGS = [f for f in AdjointConfig._fields]
ACTIVE_ON_PI = ["freeze_mixing", "freeze_convection", "freeze_vert_visc",
                "freeze_advection"]
# ⚠️ freeze_pgf is INACTIVE here, not active. Its stop_gradient sits inside
# `if ale_cfg is not None:` (step.py:270-274) — the zstar branch — and this fixture calls
# `integrate` with no ale_cfg, so it runs the linfs pressure-gradient path where the flag is
# never read. Listing it as active made test_active_flag_changes_the_gradient[freeze_pgf] fail
# against a knob that cannot fire; measured 2026-08-07, it returns bit-identical to baseline.
# Production DOES run zstar, so freeze_pgf needs an effect-test under ale_cfg — see
# test_adjoint_config_tke.py, which is the CORE2 tier.
# ⚠️ freeze_nn_features is INACTIVE here for the same STRUCTURAL reason as freeze_pgf: it is
# read only inside `if nn is not None:` (gm.py:501-505), the branch taken when a network is
# attached to `params`, and this fixture attaches none. A flag that cannot fire is not a flag
# that does nothing — listing it as active would fail against a knob with no code path, which is
# what happened to freeze_pgf on 2026-08-07. Its effect test belongs with the seam
# (tests/test_gm_nn_seam.py), where a network exists.
INACTIVE_ON_PI = ["freeze_stratification", "freeze_tke_state", "freeze_gm_slope", "freeze_k33",
                  "freeze_pgf", "freeze_nn_features"]


def test_every_flag_is_covered():
    """If someone adds a flag, this test fails until they classify it — otherwise a new knob
    can be added and silently never tested."""
    assert sorted(ACTIVE_ON_PI + INACTIVE_ON_PI) == sorted(ALL_FLAGS)


def test_default_config_is_a_dead_branch(setup):
    """adjoint_cfg=None and AdjointConfig() must agree, forward and gradient.

    ⚠️ Exact on CPU, calibrated on GPU. The two configs are separate ``jax.jit`` closures, so
    this compares two separately-compiled programs; on GPU XLA reassociates the 8-step reverse
    pass differently between them and the gradients landed 4.7e-11 apart (job 26780293). That
    is 2e5 ulp, not one, so a "one ulp" tolerance was the wrong shape of fix. The GPU branch
    instead measures what a flag that really does something is worth in this fixture and
    requires the dead-branch difference to be a millionth of it. See
    ``_identity_gates.py``."""
    loss_none, grad_none = _loss_fn(setup, None)
    loss_off, grad_off = _loss_fn(setup, AdjointConfig())
    what = "adjoint_cfg=None vs AdjointConfig()"
    assert_forward_identity(loss_off, loss_none, K0, what)

    # freeze_mixing is the reference effect: measured on this fixture it moves the gradient
    # from +8.207637e-01 to +8.138278e-01, i.e. by ~7e-3. Built lazily — CPU never needs it.
    def signal():
        _, grad_frozen = _loss_fn(setup, AdjointConfig(freeze_mixing=True))
        return float(grad_frozen(K0))

    assert_dead_branch_gradient(float(grad_none(K0)), float(grad_off(K0)), what,
                                signal_fn=signal)


@pytest.mark.parametrize("flag", ALL_FLAGS)
def test_flag_leaves_the_forward_bit_identical(setup, flag):
    """stop_gradient is the identity in the forward. Any flag that moves the trajectory by a
    single bit is wired wrong, and would silently change the physics of a production run.

    Bit-identical on CPU; on GPU, below a tenth of the response to a 1e-10 K perturbation,
    because the comparison is between two separately-compiled programs (``_identity_gates``)."""
    loss_off, _ = _loss_fn(setup, AdjointConfig())
    loss_on, _ = _loss_fn(setup, AdjointConfig(**{flag: True}))
    assert_forward_identity(loss_off, loss_on, K0, f"{flag}=True")


@pytest.mark.parametrize("flag", ACTIVE_ON_PI)
def test_active_flag_changes_the_gradient(setup, flag):
    """A knob wired to an inactive call site would report 'this term is not the problem' for
    a term it never touched. Each flag that IS on the pi code path must move the gradient."""
    _, grad_off = _loss_fn(setup, AdjointConfig())
    _, grad_on = _loss_fn(setup, AdjointConfig(**{flag: True}))
    g0, g1 = float(grad_off(K0)), float(grad_on(K0))
    assert np.isfinite(g0) and np.isfinite(g1)
    assert g0 != g1, f"{flag}=True did not change the gradient ({g0:.6e}) — is it wired?"


@pytest.mark.parametrize("flag", INACTIVE_ON_PI)
def test_inactive_flag_is_harmless(setup, flag):
    """The GM/TKE-only flags have no code path on pi (PP mixing, no GM). They must still
    produce a finite gradient rather than raising or emitting NaN."""
    _, grad_on = _loss_fn(setup, AdjointConfig(**{flag: True}))
    assert np.isfinite(float(grad_on(K0)))


def test_flags_compose(setup):
    """Turning on two flags must apply both, not the last one written."""
    _, g_mix = _loss_fn(setup, AdjointConfig(freeze_mixing=True))
    _, g_adv = _loss_fn(setup, AdjointConfig(freeze_advection=True))
    _, g_both = _loss_fn(setup, AdjointConfig(freeze_mixing=True, freeze_advection=True))
    a, b, ab = float(g_mix(K0)), float(g_adv(K0)), float(g_both(K0))
    assert ab != a and ab != b, "the combined config matched a single flag — one was dropped"


def test_describe_and_active():
    assert AdjointConfig().describe() == "none"
    assert not AdjointConfig().active
    c = AdjointConfig(freeze_mixing=True, freeze_advection=True)
    assert c.active and c.describe() == "mixing+advection"


def test_frozen_mixing_gradient_stays_sane(setup):
    """The headline use: freezing the mixing coefficients must give a finite, nonzero
    gradient with the same sign as the exact one. Freezing a feedback loop is allowed to
    change the magnitude — that is the point — but a sign flip on a well-conditioned short
    window would mean the cut removed something essential rather than something stiff."""
    _, g_off = _loss_fn(setup, AdjointConfig())
    _, g_on = _loss_fn(setup, AdjointConfig(freeze_mixing=True))
    a, b = float(g_off(K0)), float(g_on(K0))
    assert np.isfinite(b) and b != 0.0
    assert np.sign(a) == np.sign(b), f"exact {a:+.3e} vs mixing-frozen {b:+.3e}: sign flipped"
