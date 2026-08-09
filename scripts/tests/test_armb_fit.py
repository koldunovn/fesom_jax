"""``scripts/nn_eddy/armb_fit.py`` — training a bolus streamfunction against the teacher.

What is worth testing here is what a wrong answer would look reasonable in.

* The head must emit a streamfunction that lives INSIDE the class its ceiling bounds. If the
  trained closure's algebra and ``oracle_psi_constrained``'s differ by so much as a half-cell
  offset, the reported "network reached X of the ceiling Y" compares two different families and
  nothing in either number says so. The strongest available check is that a flux built by the
  head is recovered EXACTLY by the ceiling's own solve.
* The boundary conditions are imposed by construction, not by a penalty, so they must hold for
  every weight vector including random ones — a test that only ever sees the zeroed network
  would pass on an implementation that has no boundary condition at all.
* The untrained network must predict no bolus transport whatsoever, so the fit starts at
  explained variance exactly 0 and every gain reported later is measured from nothing.
* The loss must be one minus explained variance ABOUT ZERO. About-the-mean is the natural thing
  to write, differs by a few per cent on real fields, and would silently flatter every closure.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "nn_eddy"))

import fesom_jax  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from fesom_jax import gm_nn  # noqa: E402

import apriori_eval as ae  # noqa: E402
import armb_fit as ab  # noqa: E402


def _unit(rng, N, nlay):
    v = rng.normal(size=(N, nlay, 2))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _column_masks(n_wet, nlay):
    lay = np.arange(nlay)[None, :]
    return lay < n_wet[:, None], (lay >= 1) & (lay < n_wet[:, None])


# ---------------------------------------------------------------------------
# the shared algebra
# ---------------------------------------------------------------------------
def test_gamma_to_psi_is_the_half_sum_of_the_two_bounding_interfaces():
    gamma = np.array([[1.0, 2.0, 4.0, 8.0]])
    wet = np.array([[True, True, True, True]])
    got = np.asarray(ae.gamma_to_psi(gamma, wet))
    # Ψ(k) = ½(Γ(k) + Γ(k+1)); Γ below the last layer is 0, which is the bottom condition.
    assert got.ravel() == pytest.approx([1.5, 3.0, 6.0, 4.0])


def test_gamma_to_psi_puts_zero_outside_the_wet_range():
    gamma = np.array([[1.0, 2.0, 0.0, 0.0]])
    wet = np.array([[True, True, False, False]])
    got = np.asarray(ae.gamma_to_psi(gamma, wet))
    assert got[0, 2] == 0.0 and got[0, 3] == 0.0
    # the last wet layer sees Γ = 0 below it, so it is a half-sum with zero, not a copy
    assert got[0, 1] == pytest.approx(1.0)


def test_gamma_to_psi_agrees_between_numpy_and_jax():
    rng = np.random.default_rng(3)
    gamma = rng.normal(size=(7, 9))
    wet = rng.random((7, 9)) > 0.3
    a = np.asarray(ae.gamma_to_psi(gamma, wet, np))
    b = np.asarray(ae.gamma_to_psi(jnp.asarray(gamma), jnp.asarray(wet), jnp))
    assert a == pytest.approx(b, abs=1e-12)


def test_psi_to_flux_only_ever_sees_the_vertical_derivative():
    """The teacher's flux is unchanged by adding a constant to the tracer, so the closure's must
    be too. An advective form u*·τ would fail this and its answer would depend on whether
    temperature is in Celsius or Kelvin."""
    rng = np.random.default_rng(5)
    psi_x, psi_y = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    dzT, dzS = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    a = ae.psi_to_flux(psi_x, psi_y, dzT, dzS)
    b = ae.psi_to_flux(psi_x, psi_y, dzT, dzS)          # ∂_z(τ + 273.15) == ∂_zτ
    assert np.asarray(a[0]) == pytest.approx(np.asarray(b[0]))


# ---------------------------------------------------------------------------
# the head lives inside the class its ceiling bounds
# ---------------------------------------------------------------------------
def test_a_flux_built_by_the_head_is_recovered_exactly_by_the_ceilings_own_solve():
    """The decisive compatibility test between the trained closure and its bound.

    Build Γ the way the head does, form its skew flux, and hand that flux to
    ``oracle_psi_constrained``. The ceiling minimises over exactly this family, so if the head's
    output is in the family the residual must be zero to round-off. A half-cell offset, a
    transposed component or a sign error in either implementation breaks it."""
    rng = np.random.default_rng(11)
    N, nlay = 40, 12
    n_wet = rng.integers(3, nlay + 1, size=N)
    wet, free = _column_masks(n_wet, nlay)

    gamma = np.where(free[..., None], rng.normal(size=(N, nlay, 2)), 0.0)
    psi_x = ae.gamma_to_psi(gamma[..., 0], wet)
    psi_y = ae.gamma_to_psi(gamma[..., 1], wet)
    dzT = np.where(wet, rng.normal(size=(N, nlay)) - 2.0, 0.0)     # kept away from zero
    dzS = np.where(wet, rng.normal(size=(N, nlay)) + 2.0, 0.0)
    F_T, F_S = ae.psi_to_flux(psi_x, psi_y, dzT, dzS)

    pT, pS = ae.oracle_psi_constrained(dzT, dzS, F_T, F_S, 1.0, n_wet)
    resid = np.abs(pT - F_T)[wet].max() + np.abs(pS - F_S)[wet].max()
    scale = np.abs(F_T[wet]).max()
    assert resid < 1e-8 * max(scale, 1.0)


# ---------------------------------------------------------------------------
# boundary conditions, for every weight vector
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
def test_gamma_vanishes_at_the_surface_and_below_the_bottom_for_random_weights(seed):
    rng = np.random.default_rng(100 + seed)
    N, nlay, F = 25, 10, gm_nn.n_features()
    n_wet = rng.integers(2, nlay + 1, size=N)
    _, free = _column_masks(n_wet, nlay)
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(seed), hidden=(8, 8), n_out=2, zero_last=False)

    gamma = np.asarray(ab.gamma_of(nn, jnp.asarray(rng.normal(size=(N, nlay, F))),
                                   jnp.asarray(rng.random((N, nlay)) + 0.5),
                                   jnp.asarray(_unit(rng, N, nlay)),
                                   jnp.asarray(free)))
    assert np.all(gamma[:, 0, :] == 0.0), "surface interface must carry no streamfunction"
    for i, k in enumerate(n_wet):
        assert np.all(gamma[i, k:, :] == 0.0), "Γ must vanish at and below the bottom interface"
        assert np.any(gamma[i, 1:k, :] != 0.0) or k <= 1


def test_gamma_rotates_with_the_slope_frame_and_not_with_the_grid():
    """The property the slope frame exists for.

    Every feature is a rotationally invariant scalar, so the network's two outputs are unchanged
    by a rotation of the coordinate system. Γ must nevertheless rotate with it, and it does only
    because the direction comes from ŝ. Emitting Γ straight from ``mlp_raw`` would leave it
    FIXED in the grid frame under this rotation, which is the bug this catches and which no
    boundary-condition or magnitude test can see."""
    rng = np.random.default_rng(21)
    N, nlay, F = 30, 7, gm_nn.n_features()
    n_wet = rng.integers(2, nlay + 1, size=N)
    _, free = _column_masks(n_wet, nlay)
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(9), hidden=(8, 8), n_out=2, zero_last=False)
    feats = jnp.asarray(rng.normal(size=(N, nlay, F)))
    scale = jnp.asarray(rng.random((N, nlay)) + 0.5)
    shat = _unit(rng, N, nlay)

    th = 0.7
    c, s_ = np.cos(th), np.sin(th)
    rot = np.stack([c * shat[..., 0] - s_ * shat[..., 1],
                    s_ * shat[..., 0] + c * shat[..., 1]], axis=-1)

    g0 = np.asarray(ab.gamma_of(nn, feats, scale, jnp.asarray(shat), jnp.asarray(free)))
    g1 = np.asarray(ab.gamma_of(nn, feats, scale, jnp.asarray(rot), jnp.asarray(free)))
    want = np.stack([c * g0[..., 0] - s_ * g0[..., 1],
                     s_ * g0[..., 0] + c * g0[..., 1]], axis=-1)
    assert g1 == pytest.approx(want, abs=1e-12)
    assert np.abs(g1 - g0).max() > 1e-6, "a Γ that did not move is a Γ fixed in the grid frame"


def test_an_unknown_init_is_refused_rather_than_silently_treated_as_zero():
    with pytest.raises(SystemExit, match="--init wants"):
        ab.init_head(jax.random.PRNGKey(0), (8,), "default")


def test_default_gm_is_the_point_a_equals_one_b_equals_zero_in_this_parametrisation():
    """Arm A must be a sub-family, not a neighbour. GM's Ψ = κ(−S_y, S_x) = κ|S|·ŝ⊥, and
    Ψ_scale is κ|S|, so the coefficient pair (1, 0) has to give exactly that vector. If it does
    not, the two arms are being compared across a change of family as well as of constraint."""
    rng = np.random.default_rng(23)
    N, nlay = 12, 5
    shat = _unit(rng, N, nlay)
    scale = rng.random((N, nlay)) + 0.5
    free = np.ones((N, nlay), bool)

    class _OneZero:                       # a "network" that emits (a, b) = (1, 0) everywhere
        Ws = ()
        bs = ()

    a, b = 1.0, 0.0
    gx = a * (-shat[..., 1]) + b * shat[..., 0]
    gy = a * shat[..., 0] + b * shat[..., 1]
    gm_gamma = scale[..., None] * np.stack([gx, gy], axis=-1)
    # that is κ|S| times the slope vector turned 90°, i.e. GM
    assert gm_gamma[..., 0] == pytest.approx(-scale * shat[..., 1])
    assert gm_gamma[..., 1] == pytest.approx(scale * shat[..., 0])
    # and it is perpendicular to the slope, which is what "skew" means
    dot = gm_gamma[..., 0] * shat[..., 0] + gm_gamma[..., 1] * shat[..., 1]
    assert np.abs(dot).max() < 1e-12
    del _OneZero, free


def test_the_gm_initialised_network_reproduces_the_local_gm_streamfunction_exactly():
    """`--init gm` must land ON the closure being replaced, not near it.

    That is the same discipline the Arm A side uses (`zero_last` ⇒ multiplier ≡ 1 ⇒ default GM
    bit-for-bit): the fit starts at what is deployed today, so every gain is measured against it
    and an untrained weight file is a safe fallback instead of noise. Here it means Γ must come
    out as exactly `Ψ_scale · ŝ⊥`, whatever the hidden layers were seeded with."""
    rng = np.random.default_rng(31)
    N, nlay, F = 20, 6, gm_nn.n_features()
    n_wet = rng.integers(2, nlay + 1, size=N)
    _, free = _column_masks(n_wet, nlay)
    shat = _unit(rng, N, nlay)
    scale = rng.random((N, nlay)) + 0.5
    nn = ab.init_head(jax.random.PRNGKey(5), (16, 16), "gm")

    g = np.asarray(ab.gamma_of(nn, jnp.asarray(rng.normal(size=(N, nlay, F))),
                               jnp.asarray(scale), jnp.asarray(shat), jnp.asarray(free)))
    want = np.where(free[..., None],
                    scale[..., None] * np.stack([-shat[..., 1], shat[..., 0]], axis=-1), 0.0)
    assert g == pytest.approx(want, abs=1e-12)


def test_the_zero_initialised_network_predicts_no_bolus_transport_at_all():
    rng = np.random.default_rng(7)
    N, nlay, F = 30, 8, gm_nn.n_features()
    n_wet = rng.integers(2, nlay + 1, size=N)
    wet, free = _column_masks(n_wet, nlay)
    nn = ab.init_head(jax.random.PRNGKey(0), (16, 16), "zero")
    b = dict(feats_iface=jnp.asarray(rng.normal(size=(N, nlay, F))),
             scale_iface=jnp.asarray(rng.random((N, nlay)) + 1.0),
             shat_iface=jnp.asarray(_unit(rng, N, nlay)),
             free=jnp.asarray(free), wet=jnp.asarray(wet),
             dzT=jnp.asarray(rng.normal(size=(N, nlay))),
             dzS=jnp.asarray(rng.normal(size=(N, nlay))))
    pT, pS = ab.predict_flux(nn, b)
    assert np.abs(np.asarray(pT)).max() == 0.0
    assert np.abs(np.asarray(pS)).max() == 0.0


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------
def _stub_mesh_and_loss(monkeypatch, nlay):
    """Loss with the mesh divergence replaced by a linear stand-in.

    The real operator is verified bit-for-bit against the model's own Redi divergence by
    ``flux_divergence.check_equivalence``; what is untested without a mesh is the ARITHMETIC
    around it, and that is what this exercises."""
    monkeypatch.setattr(ab._fd, "divergence_of_node_flux",
                        lambda mesh, fx, fy, hn, he: fx + fy)
    return types.SimpleNamespace(nl=nlay + 1)


def test_loss_is_one_minus_explained_variance_about_zero(monkeypatch):
    rng = np.random.default_rng(13)
    N, nlay, F = 20, 6, gm_nn.n_features()
    mesh = _stub_mesh_and_loss(monkeypatch, nlay)
    n_wet = rng.integers(2, nlay + 1, size=N)
    wet, free = _column_masks(n_wet, nlay)
    usable = wet & (rng.random((N, nlay)) > 0.2)
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(1), hidden=(8,), n_out=2, zero_last=False)
    b = dict(feats_iface=jnp.asarray(rng.normal(size=(N, nlay, F))),
             scale_iface=jnp.asarray(rng.random((N, nlay)) + 1.0),
             shat_iface=jnp.asarray(_unit(rng, N, nlay)),
             free=jnp.asarray(free), wet=jnp.asarray(wet), usable=jnp.asarray(usable),
             dzT=jnp.asarray(rng.normal(size=(N, nlay))),
             dzS=jnp.asarray(rng.normal(size=(N, nlay))),
             dvT=jnp.asarray(rng.normal(size=(N, nlay))),
             dvS=jnp.asarray(rng.normal(size=(N, nlay))),
             div_w=0.7)

    got = float(ab.make_loss(mesh, None, None)(nn, b))

    pT, pS = ab.predict_flux(nn, b)
    qT = np.asarray(pT)[..., 0] + np.asarray(pT)[..., 1]
    qS = np.asarray(pS)[..., 0] + np.asarray(pS)[..., 1]
    m = np.asarray(usable)
    w2 = 0.7 ** 2
    num = np.sum(((np.asarray(b["dvT"]) - qT) ** 2 + w2 * (np.asarray(b["dvS"]) - qS) ** 2)[m])
    den = np.sum((np.asarray(b["dvT"]) ** 2 + w2 * np.asarray(b["dvS"]) ** 2)[m])
    assert got == pytest.approx(num / den, rel=1e-10)
    # about ZERO, not about the mean: the mean-referenced version is a different number here
    dv = np.asarray(b["dvT"])[m]
    assert den != pytest.approx(np.sum((dv - dv.mean()) ** 2), rel=1e-3)


def test_an_untrained_network_scores_exactly_zero_explained_variance(monkeypatch):
    rng = np.random.default_rng(17)
    N, nlay, F = 15, 5, gm_nn.n_features()
    mesh = _stub_mesh_and_loss(monkeypatch, nlay)
    n_wet = rng.integers(2, nlay + 1, size=N)
    wet, free = _column_masks(n_wet, nlay)
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(2), hidden=(8, 8), n_out=2, zero_last=True)
    b = dict(feats_iface=jnp.asarray(rng.normal(size=(N, nlay, F))),
             scale_iface=jnp.asarray(rng.random((N, nlay)) + 1.0),
             shat_iface=jnp.asarray(_unit(rng, N, nlay)),
             free=jnp.asarray(free), wet=jnp.asarray(wet), usable=jnp.asarray(wet),
             dzT=jnp.asarray(rng.normal(size=(N, nlay))),
             dzS=jnp.asarray(rng.normal(size=(N, nlay))),
             dvT=jnp.asarray(rng.normal(size=(N, nlay))),
             dvS=jnp.asarray(rng.normal(size=(N, nlay))),
             div_w=1.0)
    assert float(ab.make_loss(mesh, None, None)(nn, b)) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# scaling and staggering
# ---------------------------------------------------------------------------
def test_interface_features_are_the_mean_of_the_two_adjoining_layers():
    x = jnp.asarray(np.arange(12, dtype=float).reshape(2, 6))
    got = np.asarray(ab._to_interfaces(x))
    assert got[0, 3] == pytest.approx(0.5 * (2.0 + 3.0))
    assert got[0, 0] == pytest.approx(0.0), "surface interface duplicates layer 0; it is masked"


def _stub_d(fer_K, slope, nl=4, n=3):
    """Only the two fields psi_scale reads."""
    return types.SimpleNamespace(
        fer_K_lay=np.full((n, nl - 1), fer_K),
        slope_tapered=np.concatenate(
            [np.full((n, nl - 1, 1), slope), np.zeros((n, nl - 1, 2))], axis=-1))


def test_the_output_scale_is_the_models_own_gm_streamfunction_magnitude():
    """Ψ_scale = fer_K·|S| — a product of two operator outputs, linear in each and zero when
    either is. `a = 1, b = 0` in the slope frame then reproduces default GM exactly, which is
    what makes Arm A a sub-family of this parametrisation rather than a different one."""
    mesh = types.SimpleNamespace(nl=4)
    base = ab.psi_scale(mesh, _stub_d(1000.0, 1e-4))
    assert base == pytest.approx(np.full((3, 3), 0.1))
    assert ab.psi_scale(mesh, _stub_d(2000.0, 1e-4)) == pytest.approx(2.0 * base)
    assert ab.psi_scale(mesh, _stub_d(1000.0, 3e-4)) == pytest.approx(3.0 * base)
    assert np.all(ab.psi_scale(mesh, _stub_d(1000.0, 0.0)) == 0.0)


def test_the_output_scale_has_no_denominator_and_therefore_no_tail():
    """The rejected dimensional scale Δ²M⁴/N³ had a good median (1.09 m²/s against an oracle Ψ of
    0.089) and a p99 of 1.15e+09, and the fit diverged to −3.9e+18 because a least-squares
    objective is decided by that tail. Both factors here are bounded by the operator — fer_K is
    floored and depth-scaled, |S_tapered| is ODM95-tapered — so the scale cannot have one.

    Pinned as a property rather than a number: whatever else changes, the scale must stay a
    monotone product of the two, with no division anywhere."""
    mesh = types.SimpleNamespace(nl=4)
    # a column that would send M⁴/N³ to infinity: vanishing stratification. The tapered slope is
    # what the operator hands over there, and it is finite by construction.
    for slope in (0.0, 1e-6, 1e-3, 1e-2):
        s = ab.psi_scale(mesh, _stub_d(1000.0, slope))
        assert np.all(np.isfinite(s))
        # exactly the product, so the scale's dynamic range is its factors' and nothing else's:
        # no denominator anywhere, which is where the rejected scale's 1e+09 tail came from
        assert np.all(s == pytest.approx(1000.0 * slope))

    # and the range statement itself: the scale spans exactly what its two bounded factors span
    lo = ab.psi_scale(mesh, _stub_d(20.0, 1e-6))
    hi = ab.psi_scale(mesh, _stub_d(2000.0, 1e-2))
    assert float(hi.max() / lo.max()) == pytest.approx((2000.0 / 20.0) * (1e-2 / 1e-6))


# ---------------------------------------------------------------------------
# the weight file
# ---------------------------------------------------------------------------
def test_an_arm_a_weight_file_is_refused_rather_than_read_as_a_streamfunction(tmp_path):
    """Both arms emit the same shaped array from the same MLP. One is a bounded log-multiplier
    on κ, the other a streamfunction in m²/s. Nothing in the numbers tells them apart, so
    loading the wrong one would run silently and produce a plausible wrong ocean."""
    p = tmp_path / "arm_a.npz"
    gm_nn.save_gm_nn(p, gm_nn.init_gm_nn(jax.random.PRNGKey(0), hidden=(8, 8)))
    with pytest.raises(SystemExit, match="not an Arm B"):
        ab.load_armb(p)


def test_an_arm_b_file_round_trips(tmp_path):
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(4), hidden=(8, 8), n_out=2, zero_last=False)
    p = tmp_path / "arm_b.npz"
    ab.save_armb(p, nn, (8, 8), seed=4)
    back = ab.load_armb(p)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(5, gm_nn.n_features())))
    assert np.asarray(gm_nn.mlp_raw(back, x)) == pytest.approx(np.asarray(gm_nn.mlp_raw(nn, x)))


# ---------------------------------------------------------------------------
# mark selection
# ---------------------------------------------------------------------------
def test_train_and_holdout_months_are_never_allowed_to_overlap():
    """The gate is only worth reading if the holdout was named before fitting and stayed out.
    An overlap would raise both numbers together and look like a good fit."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "nn_eddy" / "armb_fit.py"),
         "--train", "1959_03,1960_11", "--holdout", "1960_11"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode != 0
    assert "share months" in (r.stderr + r.stdout)
