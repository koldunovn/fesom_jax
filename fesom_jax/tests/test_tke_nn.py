"""A5 gate: the structure-preserving NN-on-TKE closure (`tke_nn`) + its `tke.py` wiring.

The four invariants the design promises, asserted on CPU:
  * **zero last layer ⇒ multiplier ≡ 1 exactly** ⇒ the model is **bit-identical** to default TKE
    — both at the column-solver level AND through the assembled `mixing_tke` driver (the
    `params=None` regression guard + deployment fallback);
  * **bounded ⇒ positive-definite**: m ∈ (1/m_max, m_max), always > 0, for any weights/inputs;
  * **FD↔AD through the weights** is clean (the NN trains through the same grad(loss)(params));
  * **masked-NaN**: features are finite at every (incl. dry) column, so a zero last layer can
    never propagate a 0·NaN.
Token: TKE_NN_OK. The CORE2 NN twin/obs training are separate GPU gates (E1/E2).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import tke, tke_nn
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.state import State
from fesom_jax.tke import TkeConfig

RNG = np.random.default_rng(20260614)


# --------------------------------------------------------------------------
# multiplier invariants (pure)
# --------------------------------------------------------------------------
def test_zero_last_layer_multiplier_is_exactly_one():
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(0))            # zero_last=True default
    feats = jnp.asarray(RNG.standard_normal((40, tke_nn.N_FEATURES)))
    m = tke_nn.multiplier(nn, feats)
    assert m.shape == (40, tke_nn.N_OUT)
    assert bool(jnp.all(m == 1.0))                           # EXACTLY 1, not ≈1


def test_multiplier_finite_and_one_even_for_garbage_features():
    """A degenerate column (huge / extreme features) still gives a FINITE multiplier — and with
    a zero last layer, exactly 1 (the masked-NaN + fallback guarantee)."""
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(0))
    garbage = jnp.asarray([[1e12, -1e12, 0.0, 1e8, -1e8, 1e10]])
    m = tke_nn.multiplier(nn, garbage)
    assert bool(jnp.all(jnp.isfinite(m))) and bool(jnp.all(m == 1.0))


def test_multiplier_bounded_positive_definite():
    """For ANY weights/inputs m ∈ (1/m_max, m_max) and > 0 (⇒ positive-definite diffusivities)."""
    m_max = 3.0
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(2), m_max=m_max, w_scale=5.0, zero_last=False)
    feats = jnp.asarray(RNG.standard_normal((200, tke_nn.N_FEATURES)) * 50.0)   # extreme
    m = tke_nn.multiplier(nn, feats)
    assert bool(jnp.all(m > 0.0))
    assert float(m.min()) >= 1.0 / m_max - 1e-12
    assert float(m.max()) <= m_max + 1e-12


def test_multiplier_fd_ad_through_weights():
    """Reverse-mode grad of a multiplier loss w.r.t. the NN weights matches a finite difference."""
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(3), zero_last=False)
    feats = jnp.asarray(RNG.standard_normal((8, tke_nn.N_FEATURES)))

    def loss(nn_):
        return jnp.sum(tke_nn.multiplier(nn_, feats) ** 2)

    g = jax.grad(loss)(nn)
    assert all(bool(jnp.all(jnp.isfinite(w))) for w in g.Ws)
    # FD on a hidden-layer weight entry
    W0 = nn.Ws[0]
    h = 1e-6
    for (i, j) in [(0, 0), (2, 1)]:
        Wp = W0.at[i, j].add(h); Wm = W0.at[i, j].add(-h)
        nnp = dataclasses.replace(nn, Ws=(Wp,) + nn.Ws[1:])
        nnm = dataclasses.replace(nn, Ws=(Wm,) + nn.Ws[1:])
        fd = float((loss(nnp) - loss(nnm)) / (2 * h))
        assert abs(fd - float(g.Ws[0][i, j])) <= 1e-6 * (1 + abs(fd))


def test_column_features_finite_at_dry_columns():
    """A dry column (no interior) → count-guarded → finite features (not 0/0)."""
    N, nl = 5, 10
    interior = jnp.asarray(np.zeros((N, nl), bool))          # ALL dry
    feats = tke_nn.column_features(
        forc_tke_surf=jnp.zeros(N), coriolis=jnp.asarray(RNG.standard_normal(N)),
        depth=jnp.zeros(N), n2_col=jnp.zeros((N, nl)), shear2_col=jnp.zeros((N, nl)),
        tke_col=jnp.zeros((N, nl)), interior=interior)
    assert bool(jnp.all(jnp.isfinite(feats)))


# --------------------------------------------------------------------------
# column-solver bit-identity (c_k scalar vs c_k = scalar·m_zero)
# --------------------------------------------------------------------------
def test_integrate_tke_column_bit_identical_under_unit_multiplier():
    """Passing c_k/c_eps/cd as [N,1] arrays equal to the scalar (m=1) is bit-for-bit the scalar
    path — the algebraic core of the model-level invariant."""
    from fesom_jax import cvmix_tke
    N, nl = 12, 16
    nlev = np.full(N, nl - 2, np.int32)
    tke_old = jnp.asarray(np.abs(RNG.standard_normal((N, nl))) * 1e-4)
    Ssqr = jnp.asarray(np.abs(RNG.standard_normal((N, nl))) * 1e-6)
    Nsqr = jnp.asarray(np.abs(RNG.standard_normal((N, nl))) * 1e-5)
    dzw = jnp.asarray(np.full((N, nl), 10.0))
    dzt = jnp.asarray(np.full((N, nl), 10.0))
    forc = jnp.asarray(np.abs(RNG.standard_normal(N)) * 1e-4)
    kw = dict(alpha_tke=30.0, mxl_min=1e-8, tke_min=1e-6, kappaM_max=100.0, dt=500.0)
    ck, ce, cd = 0.1, 0.7, 3.75
    ref = cvmix_tke.integrate_tke_column(tke_old, Ssqr, Nsqr, dzw, dzt, forc, nlev,
                                         c_k=ck, c_eps=ce, cd=cd, **kw)
    ones = jnp.ones((N, 1))
    got = cvmix_tke.integrate_tke_column(tke_old, Ssqr, Nsqr, dzw, dzt, forc, nlev,
                                         c_k=ck * ones, c_eps=ce * ones, cd=cd * ones, **kw)
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------
# model-level wiring: mixing_tke NN-off vs zero-NN  ⇒  bit-identical
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tke_inputs():
    mesh = load_mesh()
    st = State.rest(mesh)
    n, e, nl = mesh.nod2D, mesh.elem2D, mesh.nl
    uv = jnp.asarray(RNG.standard_normal((e, nl, 2)) * 0.05)
    bvfreq = jnp.where(mesh.node_iface_mask,
                       jnp.asarray(np.abs(RNG.standard_normal((n, nl))) * 1e-5), 0.0)
    tke_state = jnp.where(mesh.node_iface_mask,
                          jnp.asarray(np.abs(RNG.standard_normal((n, nl))) * 1e-4), 0.0)
    stress = jnp.asarray(RNG.standard_normal((n, 2)) * 0.1)
    return mesh, uv, bvfreq, tke_state, stress, st.hnode


def _run_mixing(tke_inputs, params):
    mesh, uv, bvfreq, tke_state, stress, hnode = tke_inputs
    return tke.mixing_tke(mesh, uv, bvfreq, tke_state, stress, hnode,
                          TkeConfig(), params, dt=500.0)


def test_mixing_tke_zero_nn_bit_identical_to_no_nn(tke_inputs):
    base = Params.defaults()                                  # tke_nn=None
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(7))            # zero last layer
    withnn = dataclasses.replace(Params.defaults(), tke_nn=nn)
    out0 = _run_mixing(tke_inputs, base)
    out1 = _run_mixing(tke_inputs, withnn)
    names = ("Kv", "Av", "uvnode", "tke_new")
    for nm, a, b in zip(names, out0, out1):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                      err_msg=f"{nm} differs between NN-off and zero-NN")


def test_mixing_tke_nonzero_nn_changes_output_and_grad_finite(tke_inputs):
    """A trained (nonzero) NN actually modifies the mixing, and the gradient w.r.t. the NN
    weights flows finite through the assembled driver (masked-NaN clean)."""
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(8), zero_last=False, w_scale=1.0)
    base = _run_mixing(tke_inputs, Params.defaults())
    withnn = _run_mixing(tke_inputs, dataclasses.replace(Params.defaults(), tke_nn=nn))
    assert not np.allclose(np.asarray(base[0]), np.asarray(withnn[0]))   # Kv changed

    mesh = tke_inputs[0]
    wet = mesh.node_iface_mask

    def loss(nn_):
        p = dataclasses.replace(Params.defaults(), tke_nn=nn_)
        Kv, Av, _, tke_new = _run_mixing(tke_inputs, p)
        return jnp.sum(jnp.where(wet, Kv, 0.0) ** 2) + jnp.sum(tke_new ** 2)

    g = jax.grad(loss)(nn)
    assert all(bool(jnp.all(jnp.isfinite(w))) for w in g.Ws)
    assert all(bool(jnp.all(jnp.isfinite(b))) for b in g.bs)
    assert any(float(jnp.sum(jnp.abs(w))) > 0 for w in g.Ws)             # gradient is live


def test_tke_nn_ok_token(tke_inputs):
    """Aggregate gate — prints TKE_NN_OK."""
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(9))
    out0 = _run_mixing(tke_inputs, Params.defaults())
    out1 = _run_mixing(tke_inputs, dataclasses.replace(Params.defaults(), tke_nn=nn))
    bit_identical = all(np.array_equal(np.asarray(a), np.asarray(b))
                        for a, b in zip(out0, out1))
    feats = jnp.asarray(RNG.standard_normal((10, tke_nn.N_FEATURES)) * 30.0)
    m = tke_nn.multiplier(tke_nn.init_tke_nn(jax.random.PRNGKey(1), zero_last=False), feats)
    bounded = bool(jnp.all((m > 1.0 / 3.0 - 1e-12) & (m < 3.0 + 1e-12)))
    assert bit_identical and bounded
    print("TKE_NN_OK")
