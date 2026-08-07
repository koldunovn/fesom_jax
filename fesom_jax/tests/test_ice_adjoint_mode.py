"""``IceConfig.adjoint_mode`` — the three ways the backward pass can treat sea ice.

The forward must be bit-for-bit identical in all three, because `stop_gradient` is the
identity: whichever mode you pick, the ice trajectory is the validated model and only the
gradient changes. That property is what makes "rheology" safe to switch on by default, so it
is tested per mode rather than assumed.

The gradients must differ, and in a specific order. "frozen" cuts the whole ice update;
"rheology" cuts only the momentum solver's subcycles and keeps thermodynamics, advection,
cut-off and the ocean-to-ice coupling differentiated. So "rheology" must sit somewhere between
"frozen" and "exact" — if it matched "frozen" exactly, the extra differentiation would be doing
nothing and would not be worth its cost.

Needs the CORE2 mesh and initial condition (sea ice is a forced-path feature; the packaged pi
setup has no ice), so it skips when those are absent.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ssh, surface_forcing
from fesom_jax import step as stepmod
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing (sea ice needs the forced path)")

MODES = ("exact", "rheology", "frozen")


@pytest.fixture(scope="module")
def core2():
    mesh = load_mesh(MESH_DIR)
    state = phc_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    fs = cf.static
    sf0 = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])

    def make(mode):
        cfg = IceConfig(whichEVP=1, adjoint_mode=mode)

        def loss(kver):
            p = Params(k_ver=kver, a_ver=jnp.asarray(Params.defaults().a_ver, jnp.float64))
            fin = stepmod.step(state, mesh, op, None, p, dt=DT, is_first_step=True,
                               step_forcing=sf0, forcing_static=fs, ice_cfg=cfg)
            return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)

        return jax.jit(loss), jax.jit(jax.grad(loss))

    return make, jnp.asarray(1.0e-4, jnp.float64)


def test_unknown_mode_fails_loudly(core2):
    """A typo must raise, not silently fall through to 'exact' and produce NaN later."""
    make, k0 = core2
    loss, _ = make("frzoen")
    with pytest.raises(NotImplementedError, match="adjoint_mode"):
        loss(k0)


@pytest.mark.parametrize("mode", MODES)
def test_forward_is_identical_across_modes(core2, mode):
    """stop_gradient is the identity: the ice trajectory must not move by a single bit,
    whichever mode is chosen. This is the whole basis for switching modes in production."""
    make, k0 = core2
    ref, _ = make("exact")
    loss, _ = make(mode)
    a, b = float(ref(k0)), float(loss(k0))
    assert a == b, f"adjoint_mode={mode!r} moved the forward: {a!r} vs {b!r}"


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


def test_rheology_differs_from_both_frozen_and_exact(core2):
    """'rheology' must be its own thing. Matching 'frozen' would mean the extra differentiated
    ice components contribute nothing and the mode is pointless; matching 'exact' would mean
    the momentum-solver cut never took effect."""
    make, k0 = core2
    _, g_ex = make("exact")
    _, g_rh = make("rheology")
    _, g_fr = make("frozen")
    ex, rh, fr = float(g_ex(k0)), float(g_rh(k0)), float(g_fr(k0))
    assert rh != fr, "rheology matched frozen — the extra ice differentiation did nothing"
    assert rh != ex, "rheology matched exact — the momentum-solver cut did not take effect"


def test_rheology_works_for_standard_evp_too(core2):
    """The mode must not be silently mEVP-only: whichEVP=0 uses a different solver with the
    same 120-subcycle structure, and it carries the same knob."""
    make, k0 = core2
    mesh = load_mesh(MESH_DIR)
    state = phc_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    sf0 = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])

    def g_for(mode):
        cfg = IceConfig(whichEVP=0, adjoint_mode=mode)      # standard EVP

        def loss(kver):
            p = Params(k_ver=kver, a_ver=jnp.asarray(Params.defaults().a_ver, jnp.float64))
            fin = stepmod.step(state, mesh, op, None, p, dt=DT, is_first_step=True,
                               step_forcing=sf0, forcing_static=cf.static, ice_cfg=cfg)
            return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)

        return float(jax.jit(loss)(jnp.asarray(1.0e-4, jnp.float64))), \
            float(jax.jit(jax.grad(loss))(jnp.asarray(1.0e-4, jnp.float64)))

    l_ex, g_ex = g_for("exact")
    l_rh, g_rh = g_for("rheology")
    assert l_ex == l_rh, "standard-EVP forward moved under adjoint_mode='rheology'"
    assert np.isfinite(g_rh) and g_rh != g_ex, \
        "adjoint_mode='rheology' had no effect on the standard-EVP path"
