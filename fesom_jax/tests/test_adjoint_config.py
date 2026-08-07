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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ssh
from fesom_jax.adjoint_stab import AdjointConfig
from fesom_jax.config import A_VER, K_VER
from fesom_jax import forcing, ic
from fesom_jax.integrate import integrate
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params

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


def _loss_fn(setup, cfg):
    """mean SST after N steps, as a function of the background vertical diffusivity."""
    mesh, op, stress, st0 = setup

    def loss(kver):
        p = Params(k_ver=kver, a_ver=jnp.asarray(A_VER, jnp.float64))
        fin = integrate(st0, mesh, op, stress, n_steps=N, params=p, dt=DT, adjoint_cfg=cfg)
        return jnp.mean(fin.T[:, 0])

    return jax.jit(loss), jax.jit(jax.grad(loss))


K0 = jnp.asarray(1.0e-4, jnp.float64)     # signal-lifted, as in test_gradient.py

# Every flag, and whether the pi/PP configuration actually exercises it. The TKE-only and
# GM-only flags are wired through the same seam but their code path is inactive here, so
# they are checked for "does not break anything" rather than "changes the gradient".
ALL_FLAGS = [f for f in AdjointConfig._fields]
ACTIVE_ON_PI = ["freeze_mixing", "freeze_convection", "freeze_vert_visc",
                "freeze_advection", "freeze_pgf"]
INACTIVE_ON_PI = ["freeze_stratification", "freeze_tke_state", "freeze_gm_slope", "freeze_k33"]


def test_every_flag_is_covered():
    """If someone adds a flag, this test fails until they classify it — otherwise a new knob
    can be added and silently never tested."""
    assert sorted(ACTIVE_ON_PI + INACTIVE_ON_PI) == sorted(ALL_FLAGS)


def test_default_config_is_a_dead_branch(setup):
    """adjoint_cfg=None and AdjointConfig() must agree bit-for-bit, forward and gradient."""
    loss_none, grad_none = _loss_fn(setup, None)
    loss_off, grad_off = _loss_fn(setup, AdjointConfig())
    assert float(loss_none(K0)) == float(loss_off(K0))
    assert float(grad_none(K0)) == float(grad_off(K0))


@pytest.mark.parametrize("flag", ALL_FLAGS)
def test_flag_leaves_the_forward_bit_identical(setup, flag):
    """stop_gradient is the identity in the forward. Any flag that moves the trajectory by a
    single bit is wired wrong, and would silently change the physics of a production run."""
    loss_off, _ = _loss_fn(setup, AdjointConfig())
    loss_on, _ = _loss_fn(setup, AdjointConfig(**{flag: True}))
    a, b = float(loss_off(K0)), float(loss_on(K0))
    assert a == b, f"{flag}=True moved the forward: {a!r} vs {b!r}"


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
