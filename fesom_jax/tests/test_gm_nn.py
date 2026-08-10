"""Tests for ``gm_nn`` — the bounded GM/Redi multiplier network (NN-GM Task 7, Arm A).

Three properties carry the whole arm, and they are the first three tests here.

1. **``zero_last`` ⇒ multiplier is exactly 1.0** ⇒ ``kappa = base * m`` is bit-for-bit
   ``base``. This is what makes "the NN is wired into GM" a change that no regression test can
   detect until the network is trained, and it is also the deployment fallback. "Approximately
   1" would not do: the suite's GM gates compare against Fortran dumps bit-for-bit.

2. **The multiplier is bounded and positive for every finite input.** ``kappa`` therefore keeps
   the sign of the base diffusivity and the GM term stays an APE sink whatever the weights do.
   That is the safety property that lets this arm run unattended, so it is tested against
   inputs no ocean would produce, not against plausible ones.

3. **The gradient at the bound is small but never exactly zero.** ``tanh`` saturates; a clip
   would make d(loss)/d(weights) structurally zero for any column pinned at a bound, and a
   structurally zero gradient is indistinguishable from "this term is not the problem" — the
   failure mode that cost this project a day on 2026-08-07.

Everything is a pure MLP on synthetic features: no mesh, no model step, milliseconds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util

from fesom_jax import gm_nn

KEY = jax.random.PRNGKey(0)
SEVEN = dict(slope=1.0e-3, n2=1.0e-5, grad_b=1.0e-8, coriolis=1.0e-4,
             depth=2.0e3, depth_frac=0.5, strain=1.0e-10)


def _feats(n=32, *, two_scale=False, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(0.0, 1.0, (n, gm_nn.n_features(two_scale))))
    return x


# --- 1. the identity property ---------------------------------------------------------------

def test_zero_last_gives_a_multiplier_of_exactly_one():
    """Exact, not approximate: the GM gates elsewhere in the suite are bit-for-bit against
    Fortran dumps, so anything but 1.0 to the last bit would break them."""
    nn = gm_nn.init_gm_nn(KEY, zero_last=True)
    m = gm_nn.multiplier(nn, _feats(64))
    assert m.shape == (64, gm_nn.N_OUT)
    assert (np.asarray(m) == 1.0).all(), "zero_last must give exactly 1.0, not 1.0 +/- eps"


def test_a_kappa_scaled_by_the_untrained_network_is_bit_identical():
    """The property as the caller sees it: base * m must be the same float as base."""
    nn = gm_nn.init_gm_nn(KEY, zero_last=True)
    base = jnp.asarray(np.random.default_rng(1).uniform(100.0, 3000.0, (48, 1)))
    m = gm_nn.multiplier(nn, _feats(48))[:, :1]
    np.testing.assert_array_equal(np.asarray(base * m), np.asarray(base))


def test_zero_last_false_actually_produces_a_nontrivial_network():
    """Guard on the guard: if the random branch were also zeroed, test 1 would pass for the
    wrong reason and the network would be permanently inert."""
    nn = gm_nn.init_gm_nn(KEY, zero_last=False)
    m = np.asarray(gm_nn.multiplier(nn, _feats(64, seed=2)))
    assert m.std() > 1e-6, "an initialised network must vary across columns"
    assert not np.allclose(m[:, 0], m[:, 1]), "K_GM and K_Redi must be separate outputs"


# --- 2. the bound ---------------------------------------------------------------------------

@pytest.mark.parametrize("m_max", [1.5, 3.0, 10.0])
def test_the_multiplier_stays_inside_its_bounds_under_absurd_inputs(m_max):
    """Inputs no ocean produces, on purpose. An ML closure is a stability incident waiting for
    the right initial condition, and the bound is the thing that makes that impossible rather
    than unlikely."""
    nn = gm_nn.init_gm_nn(KEY, m_max=m_max, zero_last=False, w_scale=50.0)
    wild = jnp.asarray(np.array([[-1e30], [-1e6], [-1.0], [0.0], [1.0], [1e6], [1e30]])
                       * np.ones((1, gm_nn.n_features())))
    m = np.asarray(gm_nn.multiplier(nn, wild))
    assert np.isfinite(m).all(), "a finite input must give a finite multiplier"
    assert (m > 0.0).all(), "the multiplier must be positive so kappa keeps its sign"
    assert (m <= m_max * (1 + 1e-12)).all() and (m >= 1.0 / m_max * (1 - 1e-12)).all(), \
        f"outside [1/{m_max}, {m_max}]: {m.min()} .. {m.max()}"


def test_the_bounds_are_actually_reachable():
    """A bound that cannot be approached would mean the network can never ask for the full
    range it is documented to have."""
    nn = gm_nn.init_gm_nn(KEY, m_max=3.0, zero_last=False)
    # drive the last layer hard rather than the input, so this does not depend on feature scales
    big = gm_nn.GmNN(Ws=(*nn.Ws[:-1], nn.Ws[-1] * 0.0 + 1e3), bs=nn.bs,
                     log_m_max=nn.log_m_max)
    m = np.asarray(gm_nn.multiplier(big, _feats(16, seed=3)))
    assert (np.isclose(m, 3.0) | np.isclose(m, 1.0 / 3.0)).all(), \
        f"saturated network should sit at a bound, got {np.unique(np.round(m, 6))}"


def test_the_multiplier_is_multiplicatively_symmetric():
    """m = exp(log_m_max * tanh(raw)): flipping the sign of the raw output must invert the
    multiplier, so amplifying and damping are treated alike. An asymmetric parameterization
    would bias the closure toward one of them before any training happened."""
    nn = gm_nn.init_gm_nn(KEY, zero_last=False)
    flipped = gm_nn.GmNN(Ws=(*nn.Ws[:-1], -nn.Ws[-1]), bs=(*nn.bs[:-1], -nn.bs[-1]),
                         log_m_max=nn.log_m_max)
    f = _feats(24, seed=4)
    a = np.asarray(gm_nn.multiplier(nn, f))
    b = np.asarray(gm_nn.multiplier(flipped, f))
    np.testing.assert_allclose(a * b, 1.0, rtol=1e-13)


# --- 3. the gradient at the bound -----------------------------------------------------------

def test_the_gradient_at_the_bound_is_small_but_not_zero():
    """The anti-clip property. tanh saturates, so a pinned column still passes a (tiny)
    gradient; jnp.clip would pass exactly 0.0 and the optimiser would be told the weights of
    every saturated column are irrelevant."""
    nn = gm_nn.init_gm_nn(KEY, m_max=3.0, zero_last=False)
    f = _feats(8, seed=5)

    def loss(net):
        return jnp.sum(gm_nn.multiplier(net, f))

    for scale, label in ((1.0, "unsaturated"), (1e2, "saturated")):
        net = gm_nn.GmNN(Ws=(*nn.Ws[:-1], nn.Ws[-1] * 0.0 + scale), bs=nn.bs,
                         log_m_max=nn.log_m_max)
        g = jax.grad(loss)(net)
        last = np.asarray(g.Ws[-1])
        assert np.isfinite(last).all(), f"{label}: non-finite gradient"
        assert np.abs(last).max() > 0.0, \
            f"{label}: gradient is exactly zero — the bound is clipping, not saturating"
    # and saturation really does shrink it, which is the honest half of the claim
    def gmax(scale):
        net = gm_nn.GmNN(Ws=(*nn.Ws[:-1], nn.Ws[-1] * 0.0 + scale), bs=nn.bs,
                         log_m_max=nn.log_m_max)
        return np.abs(np.asarray(jax.grad(loss)(net).Ws[-1])).max()

    assert gmax(1e2) < gmax(1.0)


# --- features -------------------------------------------------------------------------------

def test_features_are_finite_for_zero_and_negative_inputs():
    """arcsinh, not log. These fields carry masked lanes on an unstructured mesh, and one NaN
    poisons the entire reverse pass — so zero and negative inputs must be ordinary, not
    special."""
    for v in (0.0, -1.0, -1e12, 1e12):
        f = gm_nn.column_features(*[jnp.asarray([v])] * 7)
        assert np.isfinite(np.asarray(f)).all(), f"feature map not finite at {v}"


def test_features_have_the_documented_width_and_ordering():
    one = gm_nn.column_features(**{k: jnp.asarray([v]) for k, v in SEVEN.items()})
    assert one.shape == (1, gm_nn.N_FEATURES_LOCAL) == (1, gm_nn.n_features(False))
    smoothed = tuple(jnp.asarray([v]) for v in SEVEN.values())
    two = gm_nn.column_features(**{k: jnp.asarray([v]) for k, v in SEVEN.items()},
                                smoothed=smoothed)
    assert two.shape == (1, gm_nn.n_features(True)) == (1, 2 * gm_nn.N_FEATURES_LOCAL)
    # the local half must be untouched by the presence of the smoothed half, and with identical
    # inputs the two halves must agree — that is what pins the ordering between them
    np.testing.assert_array_equal(np.asarray(two)[:, :gm_nn.N_FEATURES_LOCAL], np.asarray(one))
    np.testing.assert_array_equal(np.asarray(two)[:, gm_nn.N_FEATURES_LOCAL:], np.asarray(one))


def test_a_wrong_length_smoothed_tuple_is_refused():
    """Silently accepting six fields would feed the network a permuted input for every column
    and show up as "the two-scale variant trains worse", not as an error."""
    kw = {k: jnp.asarray([v]) for k, v in SEVEN.items()}
    with pytest.raises(ValueError, match="same 7 fields"):
        gm_nn.column_features(**kw, smoothed=tuple(jnp.asarray([1.0]) for _ in range(6)))


def test_depth_fraction_is_passed_through_unnormalised():
    """z/H is already O(1) by construction, so it must NOT get an arcsinh — that would compress
    the only feature that is already on the right scale."""
    f = np.asarray(gm_nn.column_features(**{**{k: jnp.asarray([0.0]) for k in SEVEN},
                                            "depth_frac": jnp.asarray([0.75])}))
    assert f[0, 5] == 0.75


# --- pytree / training plumbing -------------------------------------------------------------

def test_the_bound_is_static_and_the_weights_are_leaves():
    """``log_m_max`` must be a meta field: if it were a data leaf, the optimiser would train the
    safety bound away, which defeats the entire point of having one."""
    nn = gm_nn.init_gm_nn(KEY)
    leaves, treedef = tree_util.tree_flatten(nn)
    assert len(leaves) == len(nn.Ws) + len(nn.bs)
    assert all(isinstance(x, jax.Array) for x in leaves)
    assert str(nn.log_m_max) in str(treedef) or nn.log_m_max == pytest.approx(np.log(3.0)), \
        "log_m_max should travel in the treedef, not as a leaf"
    rebuilt = tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.log_m_max == nn.log_m_max


def test_gradients_flow_to_every_weight_and_bias():
    """A layer that never receives a gradient is a layer that is not in the graph. Checked per
    array rather than in aggregate, because a single detached tensor is invisible in a norm."""
    nn = gm_nn.init_gm_nn(KEY, zero_last=False)
    f = _feats(16, seed=6)
    g = jax.grad(lambda net: jnp.sum(gm_nn.multiplier(net, f) ** 2))(nn)
    for i, W in enumerate(g.Ws):
        assert np.abs(np.asarray(W)).max() > 0.0, f"no gradient reached Ws[{i}]"
    for i, b in enumerate(g.bs[:-1]):
        assert np.abs(np.asarray(b)).max() > 0.0, f"no gradient reached bs[{i}]"


def test_zero_last_still_passes_a_gradient_to_the_last_layer():
    """The initialisation must be trainable, not merely inert: at m == 1 the derivative of
    exp(log_m_max*tanh(raw)) is log_m_max != 0, so the first optimiser step CAN move it. A
    zeroed layer that also had zero gradient would never leave the identity."""
    nn = gm_nn.init_gm_nn(KEY, zero_last=True)
    f = _feats(16, seed=7)
    g = jax.grad(lambda net: jnp.sum(gm_nn.multiplier(net, f)))(nn)
    assert np.abs(np.asarray(g.Ws[-1])).max() > 0.0
    assert np.abs(np.asarray(g.bs[-1])).max() > 0.0


def test_it_jits_and_keeps_float64():
    nn = gm_nn.init_gm_nn(KEY, zero_last=False)
    f = _feats(8, seed=8)
    m = jax.jit(gm_nn.multiplier)(nn, f)
    assert m.dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(m), np.asarray(gm_nn.multiplier(nn, f)))


def test_the_hidden_layers_are_tanh_and_the_output_is_linear():
    """The squashing lives in ``multiplier`` alone. If ``mlp_raw`` also squashed, the bound
    would be applied twice and the reachable range would silently shrink."""
    nn = gm_nn.init_gm_nn(KEY, zero_last=False)
    f = _feats(8, seed=9)
    raw = np.asarray(gm_nn.mlp_raw(nn, f))
    doubled = gm_nn.GmNN(Ws=(*nn.Ws[:-1], nn.Ws[-1] * 2.0), bs=(*nn.bs[:-1], nn.bs[-1] * 2.0),
                         log_m_max=nn.log_m_max)
    np.testing.assert_allclose(np.asarray(gm_nn.mlp_raw(doubled, f)), 2.0 * raw, rtol=1e-13)

    # ... and it must be able to leave (-1,1), which a tanh output never can. The default
    # initialisation is small (|raw| ~ 0.13 here), so this needs a deliberately large last
    # layer: scaling it by 100 must scale the output by 100, not saturate at 1.
    loud = gm_nn.GmNN(Ws=(*nn.Ws[:-1], nn.Ws[-1] * 100.0), bs=nn.bs, log_m_max=nn.log_m_max)
    assert np.abs(np.asarray(gm_nn.mlp_raw(loud, f))).max() > 1.0, \
        "mlp_raw looks squashed — the bound must live only in multiplier()"


def test_the_parameter_count_stays_tiny():
    """~1-2k weights is the budget, because Task 13 hand-writes this matmul in Fortran. A net
    that quietly grew would move that from an afternoon to a project."""
    nn = gm_nn.init_gm_nn(KEY, hidden=(16, 16))
    n = sum(int(np.asarray(W).size) for W in nn.Ws) + sum(int(np.asarray(b).size) for b in nn.bs)
    assert n < 2000, f"{n} parameters — over the Fortran round-trip budget"
    two = gm_nn.init_gm_nn(KEY, hidden=(16, 16), two_scale=True)
    n2 = sum(int(np.asarray(W).size) for W in two.Ws) + sum(int(np.asarray(b).size)
                                                            for b in two.bs)
    assert n < n2 < 2000, f"two-scale variant has {n2} parameters"
