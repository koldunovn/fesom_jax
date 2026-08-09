"""Tests for the product-derived features and geometry (`scripts/nn_eddy/features_from_product.py`,
child plan M3).

The thing this module can get silently wrong is not a crash — it is producing features that are
*plausible* but not the ones the deployed seam produces. A fit trained on those learns a mapping
the model never asks for, and the symptom appears much later as "Stage 0 initialisation does not
help", with nothing to point at. So the tests below are all about identity with the seam, and
about the two hole failure modes:

* a hole left at its stored 0.0 °C / 0 psu puts a fabricated density cliff into the ELEMENT
  gradient stencil, which corrupts good columns that merely share an element with it;
* a hole that is filled and then quietly fitted on is a fabricated observation.

Runs on the packaged `pi` mesh — real FESOM geometry, small, CPU-fast, no /work data.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp  # noqa: E402
from fesom_jax import gm  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.state import State  # noqa: E402

PI_MESH_DIR = _ROOT / "fesom_jax" / "data" / "mesh_pi"


def _load_module():
    path = _ROOT / "scripts" / "nn_eddy" / "features_from_product.py"
    spec = importlib.util.spec_from_file_location("features_from_product", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["features_from_product"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fp():
    return _load_module()


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(PI_MESH_DIR)


@pytest.fixture(scope="module")
def state(mesh):
    """A stratified, laterally varying rest state — flat T/S would make every slope zero and
    every test below vacuous."""
    st = State.rest(mesh, T0=10.0, S0=35.0)
    nl = mesh.nl
    zl = np.abs(np.asarray(mesh.Z))                   # (nl-1,) layer mid-depths
    z = np.concatenate([zl, zl[-1:]])[None, :nl]      # (1,nl); the padded column is dry
    lon = np.asarray(mesh.coord_nod2D)[:, 0][:, None]
    T = 4.0 + 16.0 * np.exp(-z / 500.0) + 1.5 * np.cos(3.0 * lon)
    S = 34.5 + 1.0 * (1.0 - np.exp(-z / 800.0)) + 0.3 * np.sin(2.0 * lon)
    wet = np.asarray(mesh.node_layer_mask, bool)
    T = np.where(wet, T, 0.0)
    S = np.where(wet, S, 0.0)
    import dataclasses
    return dataclasses.replace(st, T=jnp.asarray(T), S=jnp.asarray(S))


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

def test_expand_levels_is_the_inverse_of_the_trim(fp, mesh):
    nl = mesh.nl
    a = np.arange(mesh.nod2D * (nl - 1), dtype=np.float64).reshape(mesh.nod2D, nl - 1)
    out = fp.expand_levels(a, nl)
    assert out.shape == (mesh.nod2D, nl)
    assert np.array_equal(out[:, :nl - 1], a)
    assert np.all(out[:, nl - 1:] == 0.0)


def test_expand_levels_refuses_more_levels_than_the_model_has(fp, mesh):
    a = np.zeros((mesh.nod2D, mesh.nl + 1))
    with pytest.raises(AssertionError):
        fp.expand_levels(a, mesh.nl)


# ---------------------------------------------------------------------------
# the identity that matters: same features as the seam
# ---------------------------------------------------------------------------

def test_derive_uses_the_seams_own_feature_call(fp, mesh, state):
    """No-hole case: `derive` on a product built FROM a state must reproduce
    `derive_from_state` at static geometry EXACTLY, because both end in the same
    `gm.gm_nn_features` call on the same inputs. Any drift here is a second implementation
    creeping in, which is the one failure this module exists to prevent."""
    nl = mesh.nl
    wet = np.asarray(mesh.node_layer_mask, bool)
    T47, S47, have47 = np.asarray(state.T)[:, :nl - 1], np.asarray(state.S)[:, :nl - 1], \
        wet[:, :nl - 1]
    d_prod = fp.derive(mesh, T47, S47, have47, verbose=False)
    d_stat = fp.derive_from_state(mesh, state, static_geometry=True)
    lay = np.asarray(d_stat.lay, bool)
    assert d_prod.n_filled == 0, "this fixture has no coastline mismatch"
    for name in ("feats", "fer_K_lay", "bvfreq", "sigma_xy", "slope_tapered"):
        a, b = np.asarray(getattr(d_prod, name)), np.asarray(getattr(d_stat, name))
        assert a.shape == b.shape, name
        assert np.array_equal(a, b), f"{name} differs — the two paths have diverged"
    assert np.asarray(d_prod.feats).shape[-1] == gm._gm_nn.n_features(False)
    assert np.isfinite(np.asarray(d_prod.feats)[lay]).all()


def test_features_are_the_seams_features_bit_for_bit(fp, mesh, state):
    """Belt and braces on the same claim, phrased against `gm_nn_multipliers`: a zero-last
    network on the product-derived features must give multipliers of exactly 1, and a
    non-trivial one must give the same numbers the seam would."""
    import jax
    from fesom_jax import gm_nn
    d = fp.derive_from_state(mesh, state, static_geometry=True)
    nn0 = gm_nn.init_gm_nn(jax.random.PRNGKey(0), zero_last=True)
    m = gm_nn.multiplier(nn0, d.feats)
    assert np.allclose(np.asarray(m), 1.0, rtol=0, atol=0), "zero_last must be exactly 1"
    nn1 = gm_nn.init_gm_nn(jax.random.PRNGKey(1), zero_last=False, w_scale=0.5)
    m_here = np.asarray(gm_nn.multiplier(nn1, d.feats))
    m_gm, m_redi = gm.gm_nn_multipliers(mesh, nn1, d.slope_tapered, d.sigma_xy, d.bvfreq,
                                        fp.GM_CFG, Z3d=None)
    lay = np.asarray(d.lay, bool)
    assert np.array_equal(m_here[..., 1][lay], np.asarray(m_redi)[:, :mesh.nl - 1][lay])
    assert np.array_equal(m_here[..., 0][lay], np.asarray(m_gm)[:, :mesh.nl - 1][lay])


# ---------------------------------------------------------------------------
# holes
# ---------------------------------------------------------------------------

def test_fill_holes_removes_the_fabricated_cliff(fp, mesh, state):
    """A hole left at 0.0 must visibly corrupt ∇_h σ at its NEIGHBOURS; filling must stop that.

    This is the test that justifies filling at all. If an unfilled hole made no difference to
    the neighbours, the fill would be pointless complexity."""
    nl = mesh.nl
    wet = np.asarray(mesh.node_layer_mask, bool)
    T47, S47 = np.asarray(state.T)[:, :nl - 1], np.asarray(state.S)[:, :nl - 1]
    have = wet[:, :nl - 1].copy()
    victim = int(np.argmax(have[:, 0]))          # a wet surface node
    have[victim, :] = False                      # the teacher has no column here

    d_full = fp.derive(mesh, T47, S47, wet[:, :nl - 1], verbose=False)
    d_fill = fp.derive(mesh, T47, S47, have, verbose=False)

    # unfilled control: the same hole, left at the stored 0.0
    T_raw, S_raw = T47.copy(), S47.copy()
    T_raw[victim, :] = 0.0
    S_raw[victim, :] = 0.0
    d_raw = fp.derive_from_state(
        mesh,
        __import__("dataclasses").replace(
            state, T=jnp.asarray(fp.expand_levels(T_raw, nl)),
            S=jnp.asarray(fp.expand_levels(S_raw, nl))),
        static_geometry=True)

    ring = np.asarray(fp.one_ring(mesh, jnp.asarray(
        fp.expand_levels(~have, nl) & wet)), bool)[:, :nl - 1]
    ring_only = ring.copy()
    ring_only[victim] = False
    sel = ring_only & np.asarray(d_full.lay, bool)
    assert sel.sum() > 0, "the fixture must have neighbours to corrupt"

    sig = lambda d: np.linalg.norm(np.asarray(d.sigma_xy)[:, :nl - 1, :], axis=-1)  # noqa: E731
    err_raw = np.abs(sig(d_raw) - sig(d_full))[sel].max()
    err_fill = np.abs(sig(d_fill) - sig(d_full))[sel].max()
    assert err_raw > 0.0, "an unfilled 0.0 hole must perturb its neighbours"
    assert err_fill < err_raw, (
        f"filling must reduce the neighbour damage: {err_fill:.3e} vs {err_raw:.3e}")


def test_usable_excludes_the_hole_and_its_one_ring(fp, mesh, state):
    """Filled values are a numerical convenience, never data. Nothing inside the one-ring may
    reach a fit."""
    nl = mesh.nl
    wet = np.asarray(mesh.node_layer_mask, bool)
    have = wet[:, :nl - 1].copy()
    victim = int(np.argmax(have[:, 0]))
    have[victim, :] = False
    d = fp.derive(mesh, np.asarray(state.T)[:, :nl - 1], np.asarray(state.S)[:, :nl - 1],
                  have, verbose=False)
    usable = np.asarray(d.usable, bool)
    lay = np.asarray(d.lay, bool)
    assert not usable[victim].any(), "the hole itself must never be usable"
    ring = np.asarray(fp.one_ring(mesh, jnp.asarray(fp.expand_levels(~have, nl) & wet)),
                      bool)[:, :nl - 1]
    assert not usable[ring & lay].any(), "the one-ring must never be usable"
    assert usable[lay & ~ring].all(), "everything else must stay usable"
    assert d.n_filled == int((wet[:, :nl - 1] & ~have).sum())


def test_no_holes_means_nothing_is_excluded(fp, mesh, state):
    nl = mesh.nl
    wet = np.asarray(mesh.node_layer_mask, bool)
    d = fp.derive(mesh, np.asarray(state.T)[:, :nl - 1], np.asarray(state.S)[:, :nl - 1],
                  wet[:, :nl - 1], verbose=False)
    assert d.n_filled == 0 and d.n_ring_excluded == 0
    assert np.array_equal(np.asarray(d.usable), np.asarray(d.lay))


def test_one_ring_is_the_element_stencil(fp, mesh):
    """`one_ring` must mark exactly the nodes that share an element with a flagged node —
    the same stencil `compute_sigma_xy` gathers over, which is why it is the right exclusion."""
    flag = np.zeros((mesh.nod2D, mesh.nl), bool)
    node = 3
    flag[node, :] = True
    ring = np.asarray(fp.one_ring(mesh, jnp.asarray(flag)), bool)
    e3 = np.asarray(mesh.elem_nodes)
    expect = set()
    for tri in e3[(e3 == node).any(axis=1)]:
        expect.update(int(x) for x in tri if x >= 0)
    assert set(np.where(ring[:, 0])[0].tolist()) == expect


# ---------------------------------------------------------------------------
# the base kappa
# ---------------------------------------------------------------------------

def test_base_kappa_is_init_redi_gm_not_a_reimplementation(fp, mesh, state):
    """`fer_K`/`Ki` must come from the model's own builder with the student's own params, or
    the fitted ratio κ̂/κ_base is against a coefficient the model never uses."""
    d = fp.derive_from_state(mesh, state, static_geometry=True)
    from fesom_jax.calibrate import build_params
    hn = State.rest(mesh).hnode
    _ns, _st, tap = gm.compute_neutral_slope(mesh, d.sigma_xy, d.bvfreq, fp.GM_CFG)
    fer_K, Ki, _c, _s = gm.init_redi_gm(mesh, d.bvfreq, hn, tap, build_params({}), fp.GM_CFG,
                                        zbar3=None)
    assert np.array_equal(np.asarray(d.fer_K), np.asarray(fer_K))
    assert np.array_equal(np.asarray(d.Ki), np.asarray(Ki))
    lay = np.asarray(d.lay, bool)
    assert (np.asarray(d.fer_K_lay)[lay] > 0.0).all(), "GM diffusivity must be positive"
    assert (np.asarray(d.Ki)[:, :mesh.nl - 1][lay] > 0.0).all()
