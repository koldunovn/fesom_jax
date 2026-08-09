"""Tests for the κ̂ projection (`scripts/nn_eddy/kappa_hat.py`, child plan M3).

The projection turns the teacher's eddy fluxes into the Stage-0 fitting target, and a wrong
one is not loud: it produces finite, plausible diffusivities that fit nothing, and the failure
surfaces months later as an offline fit with no online benefit. So the tests below check the
claims the algebra rests on rather than the outputs' shape:

* GM is recovered from DENSITY with no Redi contamination — the claim that makes the sequential
  projection legitimate at all;
* the two channels do not leak into each other — a projection that fitted one κ and reported it
  twice would pass a constant-κ test;
* the planted κ is recovered when the fluxes were made BY the operator, which is the only
  setting where "exactly" can be claimed (real products carry the coarse-graining's ~1e-2
  non-conservative floor);
* the guard rejects the two failures separately, because a single statistic hides one of them;
* ∇_h of a tracer really is ∇_h of that tracer — the `compute_sigma_xy` reuse is a sign trick
  and a coefficient trick, and both are the kind of thing that is right until someone edits it.

Runs on the packaged `pi` mesh, CPU, no /work data.
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
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.state import State  # noqa: E402

PI_MESH_DIR = _ROOT / "fesom_jax" / "data" / "mesh_pi"


def _load(name):
    path = _ROOT / "scripts" / "nn_eddy" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def kh_mod():
    _load("features_from_product")
    return _load("kappa_hat")


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(PI_MESH_DIR)


@pytest.fixture(scope="module")
def derived(kh_mod, mesh):
    """A stratified ocean with real lateral structure — flat fields make every projection
    0/0 and every test below vacuous."""
    import dataclasses
    fp = sys.modules["features_from_product"]
    nl = mesh.nl
    st = State.rest(mesh, T0=10.0, S0=35.0)
    zl = np.abs(np.asarray(mesh.Z))
    z = np.concatenate([zl, zl[-1:]])[None, :nl]
    lon = np.asarray(mesh.coord_nod2D)[:, 0][:, None]
    lat = np.asarray(mesh.coord_nod2D)[:, 1][:, None]
    T = 4.0 + 16.0 * np.exp(-z / 500.0) + 2.0 * np.cos(3.0 * lon) * np.cos(lat)
    S = 34.5 + 1.0 * (1.0 - np.exp(-z / 800.0)) + 0.4 * np.sin(2.0 * lon)
    wet = np.asarray(mesh.node_layer_mask, bool)
    st = dataclasses.replace(st, T=jnp.asarray(np.where(wet, T, 0.0)),
                             S=jnp.asarray(np.where(wet, S, 0.0)))
    return fp.derive_from_state(mesh, st, static_geometry=True)


# ---------------------------------------------------------------------------
# the gradient reuse
# ---------------------------------------------------------------------------

def test_tracer_gradient_really_is_the_tracer_gradient(kh_mod, mesh, derived):
    """`_tracer_gradient` gets ∇X out of `compute_sigma_xy` by setting α=0, β=1/ρ0. Check it
    against the density gradient the same function produces from the same state: with the real
    α and β, ρ0(−α∇T + β∇S) must equal −ρ0·α·∇T + ρ0·β·∇S built from the two single-tracer
    gradients. If the trick ever stops holding, this is what says so."""
    from fesom_jax import gm
    fp = sys.modules["features_from_product"]
    nl = mesh.nl
    gT = np.asarray(kh_mod._tracer_gradient(mesh, derived.T, fp.GM_CFG))
    gS = np.asarray(kh_mod._tracer_gradient(mesh, derived.S, fp.GM_CFG))
    a = np.asarray(derived.sw_alpha)[..., None]
    b = np.asarray(derived.sw_beta)[..., None]
    rho0 = fp.GM_CFG.rho_ref
    rebuilt = rho0 * (-a * gT + b * gS)
    ref = np.asarray(gm.compute_sigma_xy(mesh, derived.T, derived.S, derived.sw_alpha,
                                         derived.sw_beta, fp.GM_CFG))
    lay = np.asarray(derived.lay, bool)
    err = np.abs(rebuilt - ref)[:, :nl - 1][lay].max()
    scale = np.abs(ref)[:, :nl - 1][lay].max()
    assert err <= 1e-10 * max(scale, 1e-30), (err, scale)


def test_spice_gradient_is_the_sign_flip(kh_mod, mesh, derived):
    """∇χ is `compute_sigma_xy` called with −T. Verify against the explicit combination."""
    from fesom_jax import gm
    fp = sys.modules["features_from_product"]
    rho0 = fp.GM_CFG.rho_ref
    gT = np.asarray(kh_mod._tracer_gradient(mesh, derived.T, fp.GM_CFG))
    gS = np.asarray(kh_mod._tracer_gradient(mesh, derived.S, fp.GM_CFG))
    a = np.asarray(derived.sw_alpha)[..., None]
    b = np.asarray(derived.sw_beta)[..., None]
    want = rho0 * (a * gT + b * gS)
    got = np.asarray(gm.compute_sigma_xy(mesh, -derived.T, derived.S, derived.sw_alpha,
                                         derived.sw_beta, fp.GM_CFG))
    lay = np.asarray(derived.lay, bool)[:, :]
    sel = np.zeros(want.shape[:2], bool)
    sel[:, :mesh.nl - 1] = lay
    err = np.abs(got - want)[sel].max()
    scale = np.abs(want)[sel].max()
    assert err <= 1e-10 * max(scale, 1e-30), (err, scale)


# ---------------------------------------------------------------------------
# the physics claim the sequential projection rests on
# ---------------------------------------------------------------------------

def test_redi_alone_leaves_the_density_flux_almost_untouched(kh_mod, mesh, derived):
    """Plant Redi ONLY. The density flux it produces must be small compared with what the same
    coefficient would produce as GM — that is the ``∇σ + S ∂_zσ ≈ 0`` claim, and it is what
    lets κ̂_GM be read off density without subtracting anything."""
    nl = mesh.nl
    ones = jnp.ones((mesh.nod2D, nl - 1))
    F_T_r, F_S_r = kh_mod.gm_redi_flux(mesh, derived, 0.0 * ones, 500.0 * ones)
    F_T_g, F_S_g = kh_mod.gm_redi_flux(mesh, derived, 500.0 * ones, 0.0 * ones)
    a = np.asarray(derived.sw_alpha)[:, :nl - 1, None]
    b = np.asarray(derived.sw_beta)[:, :nl - 1, None]
    sig = lambda ft, fs: np.linalg.norm(  # noqa: E731
        (-a * np.asarray(ft) + b * np.asarray(fs)), axis=-1)
    lay = np.asarray(derived.lay, bool)
    ratio = np.median(sig(F_T_r, F_S_r)[lay]) / max(np.median(sig(F_T_g, F_S_g)[lay]), 1e-300)
    assert ratio < 0.05, (
        f"Redi's density flux is {100*ratio:.1f}% of GM's at the same coefficient — the "
        f"sequential projection needs this small; above a few percent it must become a joint "
        f"2x2 least squares")


def test_synthetic_recovery_of_both_planted_fields(kh_mod, mesh, derived):
    """The plan's required test: fluxes generated BY the operator from a known κ(x,z) must come
    back. The two planted fields have different spatial and vertical patterns, so a projection
    that recovered one and copied it, or that recovered a global constant, fails."""
    out = kh_mod.synthetic_check(mesh, derived, verbose=False)
    for name in ("kappa_gm", "kappa_redi"):
        r = out[name]
        assert r["n"] > 0, f"{name}: the guard rejected everything, nothing was tested"
        assert r["median_rel"] < 0.05, (name, r)
        assert r["p99_rel"] < 0.5, (name, r)


def test_the_channels_do_not_leak(kh_mod, mesh, derived):
    """Plant GM only, and κ̂_Redi must come back near zero (and vice versa). A projection that
    attributed GM's flux to Redi would still recover 'a' diffusivity and look healthy."""
    nl = mesh.nl
    ones = jnp.ones((mesh.nod2D, nl - 1))
    F_T, F_S = kh_mod.gm_redi_flux(mesh, derived, 800.0 * ones, 0.0 * ones)
    kh = kh_mod.project(mesh, derived, F_T, F_S, resid_max=1e9, verbose=False)
    ok = np.asarray(kh.ok_gm, bool) & np.asarray(kh.ok_redi, bool)
    assert ok.sum() > 0
    assert np.median(np.abs(np.asarray(kh.kappa_gm)[ok] - 800.0)) < 40.0
    assert np.median(np.abs(np.asarray(kh.kappa_redi)[ok])) < 80.0, "GM leaked into Redi"

    F_T, F_S = kh_mod.gm_redi_flux(mesh, derived, 0.0 * ones, 600.0 * ones)
    kh = kh_mod.project(mesh, derived, F_T, F_S, resid_max=1e9, verbose=False)
    ok = np.asarray(kh.ok_gm, bool) & np.asarray(kh.ok_redi, bool)
    assert np.median(np.abs(np.asarray(kh.kappa_redi)[ok] - 600.0)) < 60.0
    assert np.median(np.abs(np.asarray(kh.kappa_gm)[ok])) < 60.0, "Redi leaked into GM"


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------

def test_guard_rejects_ill_conditioning_and_inadequacy_separately(kh_mod, mesh, derived):
    """Two different failures, two different counters. One statistic would hide one of them."""
    nl = mesh.nl
    ones = jnp.ones((mesh.nod2D, nl - 1))
    F_T, F_S = kh_mod.gm_redi_flux(mesh, derived, 500.0 * ones, 300.0 * ones)

    loose = kh_mod.project(mesh, derived, F_T, F_S, grad_floor=0.0, slope_floor=0.0,
                           resid_max=1e9, verbose=False)
    assert loose.reject["gm_illconditioned"] == 0 and loose.reject["gm_inadequate"] == 0

    # conditioning only: an impossible gradient floor
    tight = kh_mod.project(mesh, derived, F_T, F_S, grad_floor=1e30, slope_floor=0.0,
                           resid_max=1e9, verbose=False)
    assert tight.reject["kept_gm"] == 0 and tight.reject["gm_illconditioned"] > 0
    assert tight.reject["gm_inadequate"] == 0, "an ill-conditioned cell must not also be " \
                                               "counted as inadequate — the reasons are exclusive"

    # adequacy only: a flux the GM+Redi ansatz cannot explain (rotate it 90 degrees)
    F_rot = jnp.stack([-F_T[..., 1], F_T[..., 0]], axis=-1)
    S_rot = jnp.stack([-F_S[..., 1], F_S[..., 0]], axis=-1)
    bad = kh_mod.project(mesh, derived, F_rot, S_rot, grad_floor=0.0, slope_floor=0.0,
                         resid_max=0.5, verbose=False)
    assert bad.reject["gm_inadequate"] > 0, "a flux orthogonal to ∇σ must fail adequacy"
    assert bad.reject["gm_illconditioned"] == 0


def test_the_decoupling_guard_is_what_removes_the_bad_tail(kh_mod, mesh, derived):
    """`b_sigma = ∇σ + S ∂_zσ` is zero in exact arithmetic and is what makes Redi blind to
    density. Where it is not small, Redi's density flux lands entirely in κ̂_GM — and because
    it is COLLINEAR with ∇σ, the perpendicular-residual test cannot see it. So this guard has
    to exist separately, and disabling it must visibly bring the bad tail back."""
    off = kh_mod.synthetic_check.__wrapped__ if hasattr(kh_mod.synthetic_check, "__wrapped__") \
        else None
    assert off is None  # (no decorator; keep the lookup honest if one is ever added)

    import jax.numpy as _jnp
    nl = mesh.nl
    zl = _jnp.abs(_jnp.asarray(mesh.Z))[None, :nl - 1]
    lat = _jnp.asarray(mesh.coord_nod2D)[:, 1][:, None]
    k_gm = _jnp.broadcast_to(300.0 + 700.0 * _jnp.exp(-zl / 800.0)
                             * (0.6 + 0.4 * _jnp.cos(2.0 * lat)), (mesh.nod2D, nl - 1))
    k_redi = _jnp.broadcast_to(200.0 + 400.0 * _jnp.exp(-zl / 1500.0)
                               * (0.5 + 0.5 * _jnp.sin(3.0 * lat)), (mesh.nod2D, nl - 1))
    F_T, F_S = kh_mod.gm_redi_flux(mesh, derived, k_gm, k_redi)

    def tail(b_max):
        kh = kh_mod.project(mesh, derived, F_T, F_S, b_sigma_max=b_max, verbose=False)
        m = np.asarray(kh.ok_gm, bool)
        rel = np.abs(np.asarray(kh.kappa_gm)[m] - np.asarray(k_gm)[m]) \
            / np.maximum(np.abs(np.asarray(k_gm)[m]), 1e-300)
        return float(np.percentile(rel, 99)), float(np.median(rel)), int(m.sum())

    p99_off, med_off, n_off = tail(1e30)
    p99_on, med_on, n_on = tail(kh_mod.B_SIGMA_MAX)
    assert p99_off > 1.0, "without the guard the tail must be visibly broken"
    assert p99_on < 0.1, f"with the guard the 99th percentile must be small, got {p99_on:.3e}"
    assert med_off < 0.05 and med_on < 0.05, "the MEDIAN is fine either way — which is " \
                                             "exactly why a median-only check would miss this"
    assert n_on > 0.5 * n_off, f"the guard must cost coverage, not most of it: " \
                               f"{n_on} of {n_off} kept"


def test_holes_never_reach_the_projection(kh_mod, mesh):
    """`usable` is the gate; a filled cell is a numerical convenience, never an observation."""
    import dataclasses
    fp = sys.modules["features_from_product"]
    nl = mesh.nl
    st = State.rest(mesh, T0=10.0, S0=35.0)
    zl = np.abs(np.asarray(mesh.Z))
    z = np.concatenate([zl, zl[-1:]])[None, :nl]
    lon = np.asarray(mesh.coord_nod2D)[:, 0][:, None]
    wet = np.asarray(mesh.node_layer_mask, bool)
    T = np.where(wet, 4.0 + 16.0 * np.exp(-z / 500.0) + 2.0 * np.cos(3.0 * lon), 0.0)
    S = np.where(wet, 34.5 + (1.0 - np.exp(-z / 800.0)), 0.0)
    have = wet[:, :nl - 1].copy()
    victim = int(np.argmax(have[:, 0]))
    have[victim, :] = False
    d = fp.derive(mesh, T[:, :nl - 1], S[:, :nl - 1], have, verbose=False)
    ones = jnp.ones((mesh.nod2D, nl - 1))
    F_T, F_S = kh_mod.gm_redi_flux(mesh, d, 500.0 * ones, 300.0 * ones)
    kh = kh_mod.project(mesh, d, F_T, F_S, grad_floor=0.0, slope_floor=0.0, resid_max=1e9,
                        verbose=False)
    assert not np.asarray(kh.ok_gm, bool)[victim].any()
    assert not np.asarray(kh.ok_redi, bool)[victim].any()
    assert kh.reject["not_usable_hole_ring"] > 0


def test_strata_report_covers_every_usable_cell(kh_mod, mesh, derived):
    """The rejected-fraction report is the plan's deliverable; it must not quietly drop cells."""
    nl = mesh.nl
    ones = jnp.ones((mesh.nod2D, nl - 1))
    F_T, F_S = kh_mod.gm_redi_flux(mesh, derived, 500.0 * ones, 300.0 * ones)
    kh = kh_mod.project(mesh, derived, F_T, F_S, verbose=False)
    rows = kh_mod.report_by_stratum(mesh, derived, kh, verbose=False)
    n_depth = sum(r["n"] for r in rows if r["kind"] == "depth")
    n_lat = sum(r["n"] for r in rows if r["kind"] == "lat")
    n_use = int(np.asarray(derived.usable, bool).sum())
    assert n_depth == n_use, (n_depth, n_use)
    assert n_lat == n_use, (n_lat, n_use)


# ---------------------------------------------------------------------------
# the offline fit's target construction and its baselines
# (offline_fit.py — same M3 chain, tested here to keep the fixture)
# ---------------------------------------------------------------------------

def _offline():
    import importlib.util
    path = _ROOT / "scripts" / "nn_eddy" / "offline_fit.py"
    spec = importlib.util.spec_from_file_location("offline_fit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["offline_fit"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_clipping_reports_unreachable_targets_rather_than_chasing_them():
    """A target outside [1/m_max, m_max] cannot be expressed by a bounded multiplier. Leaving
    it in would drag every weight toward saturation chasing something unreachable, and a
    saturated network has almost no gradient left for the columns it CAN fit. So it is clipped
    — and the clipped fraction is REPORTED, because it is the architecture's reachability and
    it feeds the parent's Arm-B decision."""
    of = _offline()
    lm = float(np.log(3.0))
    tgt = np.array([[0.0, 0.0], [0.5, -0.5], [5.0, 0.0], [0.0, -9.0]])
    out, frac = of.clip_targets(tgt, lm)
    assert np.isclose(frac, 0.5), "two of four rows are out of bounds"
    assert out.max() <= lm + 1e-12 and out.min() >= -lm - 1e-12
    assert np.array_equal(out[:2], tgt[:2]), "in-bounds targets must be untouched"


def test_skill_baselines_are_the_two_the_gate_needs():
    """The gate requires beating BOTH the formula and the best global constant. A closure that
    only rescaled kappa everywhere would beat the formula and mean nothing, so a test that the
    constant baseline is actually the mean-predictor is what keeps that check honest."""
    of = _offline()
    import jax
    from fesom_jax import gm_nn as _g
    X = np.zeros((64, _g.N_FEATURES_LOCAL))
    Y = np.stack([np.linspace(-0.4, 0.4, 64), np.linspace(0.2, -0.2, 64)], axis=-1)
    nn = _g.init_gm_nn(jax.random.PRNGKey(0), zero_last=True)
    s = of.skill(nn, X, Y, log_m_max=float(np.log(3.0)))
    # a zero_last network predicts log m == 0 everywhere, i.e. exactly the formula baseline
    assert np.isclose(s["nn"], s["formula"]), "an untrained net IS the default closure"
    # the constant baseline is the column mean; here the columns are symmetric about 0, so it
    # coincides with the formula — which is exactly the degenerate case the gate must survive
    assert s["constant"] >= s["formula"] - 1e-12
    Y2 = Y + 0.3                       # now the mean is NOT zero and the constant must win
    s2 = of.skill(nn, X, Y2, log_m_max=float(np.log(3.0)))
    assert s2["constant"] > s2["formula"], "with a nonzero mean, a global constant must beat m=1"
