"""The mEVP tangent mode-study script ``scripts/adjoint/ice_modes.py`` (Track I, Task 4).

Synthetic (no data): the power iteration, the power SVD and Arnoldi recover the eigenvalues of a
block-diagonal two-variable "sweep map" (per cell σ' = a·σ + k·u, u' = b·u + d·σ', loop gain
c = −d·k), the model of one mEVP sweep the study is built on:
  * c ≈ 4.0273 with a = b = 1 ⇒ an oscillating mode λ ≈ −1.1795 (the earlier inferred rate);
  * implicit damping at T_N = 7 (the whole stress update × 1/(1+1/T_N)) ⇒ a complex pair,
    |λ| = √0.875 ≈ 0.935;
  * explicit damping at T_N = 7 (a → a − 1/T_N) ⇒ a stronger oscillation, λ ≈ −1.65;
plus a complex-pair map, the zero start vector, an Arnoldi breakdown, the table round trip,
the correlation, the verdict rule, provenance and the refusal to write into the checkout.

CORE2 smoke (2013-01-01 restart, CPU): the captured mEVP inputs reproduce the model step's
``u_ice`` bitwise through the script's own 120 sweeps, every factor set leaves that forward
bitwise, and 2 power iterations of the step map are finite.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import ice_modes as IM  # noqa: E402

STATES = Path("/work/ab0995/a270088/fesom_jax_dev/states_2013")
C_OSC = 4.0273


def _cell(a, b, c, *, f_imp=1.0, shift=0.0, k=1.0):
    d = -c / k
    a2 = a - shift
    return np.array([[f_imp * a2, f_imp * k], [d * f_imp * a2, b + d * f_imp * k]])


def _block_map(n=40, *, f_imp=1.0, shift=0.0, seed=3):
    rng = np.random.default_rng(seed)
    blocks = [_cell(1.0, 1.0, C_OSC, f_imp=f_imp, shift=shift)]
    blocks += [_cell(0.95, 0.95, c, f_imp=f_imp, shift=shift) for c in rng.uniform(0.5, 3.0, n - 1)]
    A = np.zeros((2 * n, 2 * n))
    for i, B in enumerate(blocks):
        A[2 * i:2 * i + 2, 2 * i:2 * i + 2] = B
    perm = rng.permutation(2 * n)                    # hide the block structure
    return A[np.ix_(perm, perm)], blocks[0]


def _dominant_eig(A):
    w = np.linalg.eigvals(A)
    return w[np.argmax(np.abs(w))]


def _warm(A, v, n):
    for _ in range(n):
        v = A @ v
        v = v / np.linalg.norm(v)
    return v


def test_the_two_variable_model_reproduces_the_inferred_rate():
    lam = _dominant_eig(_cell(1.0, 1.0, C_OSC))
    assert abs(lam.imag) < 1e-12 and lam.real == pytest.approx(-1.1795, abs=1e-3)


def test_power_iteration_and_arnoldi_recover_the_oscillating_mode():
    A, _ = _block_map()
    want = _dominant_eig(A)
    v0 = np.random.default_rng(0).standard_normal(A.shape[0])
    rates, ray, v = IM.power_iteration(lambda x: A @ x, v0, 300)
    assert rates[-1] == pytest.approx(abs(want), rel=1e-6)
    assert ray[-1] < 0                                    # sign flip per sweep
    rz = IM.arnoldi(lambda x: A @ x, _warm(A, v0, 30), 20)
    dm = IM.dominant(rz)
    assert dm["type"] == "oscillating"
    assert dm["re"] == pytest.approx(want.real, rel=1e-6) and dm["re"] == pytest.approx(-1.1795, abs=1e-3)
    assert dm["residual"] < 1e-6
    re, im = rz["vector"](0)
    assert np.linalg.norm(A @ re - dm["re"] * re) < 1e-6 * np.linalg.norm(re)
    assert np.linalg.norm(im) < 1e-8


def test_implicit_damping_turns_the_oscillation_into_a_decaying_complex_pair():
    A, B = _block_map(f_imp=1.0 / (1.0 + 1.0 / 7.0))
    dm = IM.dominant(IM.arnoldi(lambda x: A @ x,
                                _warm(A, np.random.default_rng(1).standard_normal(A.shape[0]), 30), 20))
    assert dm["type"] == "complex pair"
    assert dm["abs"] == pytest.approx(math.sqrt(0.875), rel=1e-6)
    assert dm["abs"] == pytest.approx(0.935, abs=1e-3)
    assert dm["abs"] == pytest.approx(abs(_dominant_eig(B)), rel=1e-6)


def test_explicit_damping_makes_the_oscillation_worse():
    A, B = _block_map(shift=1.0 / 7.0)
    want = _dominant_eig(B)
    dm = IM.dominant(IM.arnoldi(lambda x: A @ x,
                                _warm(A, np.random.default_rng(2).standard_normal(A.shape[0]), 30), 20))
    assert dm["type"] == "oscillating"
    assert dm["re"] == pytest.approx(want.real, rel=1e-6)
    assert dm["re"] == pytest.approx(-1.655, abs=0.01)


def test_power_svd_is_the_top_singular_value_of_a_non_symmetric_map():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((30, 30)) + 3 * np.triu(rng.standard_normal((30, 30)))
    s, hist, v = IM.power_svd(lambda x: A @ x, lambda y: A.T @ y, rng.standard_normal(30), 400)
    assert s == pytest.approx(np.linalg.svd(A, compute_uv=False)[0], rel=1e-8)
    assert np.all(np.diff(hist) >= -1e-9 * hist[-1])     # monotone estimate


def test_a_complex_pair_map_is_classified_as_such():
    r, th = 1.3, 0.7
    R = r * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = np.zeros((6, 6))
    A[:2, :2] = R
    A[2:, 2:] = np.diag([0.5, -0.4, 0.3, 0.2])
    rz = IM.arnoldi(lambda x: A @ x, np.ones(6), 6)
    dm = IM.dominant(rz)
    assert dm["type"] == "complex pair"
    assert dm["abs"] == pytest.approx(r, rel=1e-10)
    assert abs(dm["im"]) == pytest.approx(r * np.sin(th), rel=1e-10)
    assert IM.classify(-2.0) == "oscillating" and IM.classify(2.0) == "steady"


def test_zero_start_vector_is_refused():
    A = np.eye(4)
    for call in (lambda: IM.power_iteration(lambda x: A @ x, np.zeros(4), 3),
                 lambda: IM.power_svd(lambda x: A @ x, lambda x: A.T @ x, np.zeros(4), 3),
                 lambda: IM.arnoldi(lambda x: A @ x, np.zeros(4), 3)):
        with pytest.raises(ValueError, match="start vector"):
            call()


def test_arnoldi_breakdown_on_a_small_invariant_subspace():
    A = np.diag([3.0, -2.0, 0.5])
    rz = IM.arnoldi(lambda x: A @ x, np.ones(3), 10)
    assert rz["m"] == 3
    np.testing.assert_allclose(sorted(np.real(rz["values"])), [-2.0, 0.5, 3.0], rtol=1e-12)


def test_ritz_values_come_sorted_by_modulus():
    rng = np.random.default_rng(5)
    for _ in range(20):
        A = rng.standard_normal((8, 8))
        rz = IM.arnoldi(lambda x: A @ x, rng.standard_normal(8), 8)
        mods = np.abs(rz["values"])
        assert np.all(np.diff(mods) <= 1e-12 * mods[0])
        assert mods[0] == pytest.approx(np.max(np.abs(np.linalg.eigvals(A))), rel=1e-10)


def test_table_round_trips():
    rows = [{"variant": "imp7", "form": "implicit", "tn": 7, "sigma1_step": 3.25e-5,
             "growth_day": 0.031, "mode_type": "complex pair λ=0.9|x", "r_lnp_1": 0.61,
             "c_lnp_1": 0.93, "c2_lnp_1": 0.88, "verdict": "PASS", "bwd_s": None},
            {"variant": "exact", "form": "exact", "tn": None, "sigma1_step": 4.1e8,
             "growth_day": float("inf"), "verdict": "FAIL", "chain_blowup": True}]
    md = IM.table_markdown(rows)
    back = IM.parse_markdown_table("# title\n\n" + md + "\ntrailing text\n")
    assert len(back) == 2
    for r, b in zip(rows, back):
        for key, _ in IM.TABLE_COLUMNS:
            assert b[key] == IM._fmt(r.get(key)), key
    assert back[0]["mode_type"] == "complex pair λ=0.9/x"        # a pipe cannot break the table


def test_weighted_correlation_and_the_verdict_rule():
    t = np.linspace(-1, 2, 50)
    w = np.linspace(1, 3, 50)
    assert IM.weighted_corr(t, 2 * t + 1, w) == pytest.approx(1.0)
    assert IM.weighted_corr(t, -t, w) == pytest.approx(-1.0)
    good = {"growth_day": 3.0, "chain_blowup": False, "r_lnp_1": 0.7, "c2_lnp_1": 0.9,
            "r_lncd_1": 1.5, "c2_lncd_1": 0.85}
    assert IM.verdict(good) == "PASS"
    for k, v in (("growth_day", 11.0), ("chain_blowup", True), ("r_lnp_1", 0.49),
                 ("r_lncd_1", 2.01), ("c2_lncd_1", 0.8), ("c2_lnp_1", float("nan"))):
        assert IM.verdict({**good, k: v}) == "FAIL", k
    # the verdict reads the correlation against the h = 1e-2 FD field, not the one of the
    # ratio's h (which is only reported)
    assert IM.VERDICT_FD_H == 1e-2
    assert IM.verdict({**good, "c_lnp_1": 0.1, "c_lncd_1": 0.1}) == "PASS"
    assert IM.verdict({**good, "c2_lnp_1": None}) == "FAIL"
    keys = [k for k, _ in IM.TABLE_COLUMNS]
    for c in IM.CONTROLS:
        for when in ("1", "d"):
            assert f"c2_{c}_{when}" in keys and f"c_{c}_{when}" in keys


def test_provenance_and_output_location():
    head = IM.git_head()
    assert len(head) == 40 and all(ch in "0123456789abcdef" for ch in head), head
    with pytest.raises(SystemExit, match="inside the checkout"):
        IM._refuse_inside_checkout(ROOT / "runs_here")
    IM._refuse_inside_checkout("/work/ab0995/a270088/fesom_jax_dev/runs/ice/modes")


# ---------------------------------------------------------------------------------------------
# CORE2 smoke


@pytest.fixture(scope="module")
def icemap():
    if not ((ROOT / "data" / "mesh_core2").is_dir() and (STATES / "fesom.2013.001.00000").is_dir()):
        pytest.skip("CORE2 mesh / 2013 States missing")
    return IM.IceMap("2013-01-01")


def test_core2_capture_reproduces_the_step_bitwise_for_every_factor_set(icemap):
    M = icemap
    assert M.n_sweeps == 120
    su, sv = (np.asarray(q) for q in M.step_u_ice)
    assert np.max(np.abs(su)) > 0.0
    sets = [M.ones(), M.factors("implicit", 7), M.factors("explicit", 3),
            M.factors("whole", 4), M.factors("truncate", k=8), M.factors("zero_stress")]
    for fac in sets:
        y = [np.asarray(q) for q in M.p_fwd0(M.x0, *M.air0, *fac)]
        assert y[0].tobytes() == su.tobytes() and y[1].tobytes() == sv.tobytes()
        for a, b in zip(y, M.mevp_out):
            assert a.tobytes() == np.asarray(b).tobytes()
    assert M.air_check["u_ice_arg_is_entry"] and M.air_check["stress_bitwise"]
    # the control program at c = 0 differs only by rounding (amplified by the forward itself)
    yc = [np.asarray(q) for q in M.p_fwd(M.x0, M.c0, *M.ones())]
    assert max(np.max(np.abs(yc[0] - su)), np.max(np.abs(yc[1] - sv))) < 1e-5


def test_core2_factor_sets(icemap):
    M = icemap
    with pytest.raises(ValueError, match="T_N"):
        M.factors("explicit", 1.0)
    cf = np.asarray(M.factors("truncate", k=8)[2])
    assert cf[120 - 8 - 1] == 0.0 and np.count_nonzero(cf == 0.0) == 1
    imp = np.asarray(M.factors("implicit", 7)[1])
    assert np.all(imp[M.ice_el] == pytest.approx(7.0 / 8.0)) and np.all(imp[~M.ice_el] == 1.0)
    assert list(np.asarray(M.factors("local", 0.5)[3])) == [0.5, 0.0, 0.0, 0.0]
    assert list(np.asarray(M.factors("local_cut", 4)[3])) == [0.0, 4.0, 0.0, 0.0]
    assert list(np.asarray(M.factors("frozen_alpha")[3])) == [0.0, 0.0, 1.0, 0.0]
    assert list(np.asarray(M.factors("tla", 0.25)[3])) == [0.0, 0.0, 0.0, 0.25]
    assert list(np.asarray(M.ones()[3])) == [0.0, 0.0, 0.0, 0.0]


def test_core2_local_factors_act_only_where_the_stability_number_is_large(icemap):
    """The local (fallback-B preview) forms: a cut at κ removes the new-stress tangent exactly on
    the elements whose γ/αβ exceeds κ at the sweep input, and nowhere else; (0, 0) is exact."""
    import jax
    import jax.numpy as jnp
    M = icemap
    x0f = M.flat(M.x0)
    o = M.offs
    rng = np.random.default_rng(1)
    v = jnp.asarray(rng.standard_normal(int(o[-1])))
    ones = M.ones()
    t1 = np.asarray(M.p_jvp_sweep(x0f, x0f, M.c0, ones[0], ones[1], 1.0, ones[3], v))
    fac = M.factors("local_cut", 4.0)
    tc = np.asarray(M.p_jvp_sweep(x0f, x0f, M.c0, fac[0], fac[1], 1.0, fac[3], v))
    rows, _, gamma = M.element_table(x0f, [0])
    ab = M.cfg.alpha_evp * M.cfg.beta_evp
    g = gamma / ab
    hit = M.ice_el & (g > 4.0)
    sure = np.abs(g - 4.0) > 1e-8 * 4.0          # rounding of γ may differ right at the threshold
    assert 0 < hit.sum() < M.ice_el.sum()
    for k in (2, 3, 4):
        a, b = t1[o[k]:o[k + 1]], tc[o[k]:o[k + 1]]
        assert np.all(b[hit & sure] == 0.0)
        np.testing.assert_array_equal(b[~hit & sure], a[~hit & sure])
    t0 = np.asarray(M.p_jvp_sweep(x0f, x0f, M.c0, ones[0], ones[1], 1.0, jnp.zeros(4), v))
    assert t0.tobytes() == t1.tobytes()
    del jax


def test_core2_two_power_iterations_are_finite(icemap):
    M = icemap
    import jax.numpy as jnp
    fac = M.ones()
    x0f = M.flat(M.x0)
    v0 = jnp.asarray(np.random.default_rng(0).standard_normal(int(M.offs[-1])))
    s, hist, _ = IM.power_svd(lambda v: M.p_jvp(x0f, M.c0, *fac, v, jnp.zeros(2))[1],
                              lambda w: M.p_vjp(x0f, M.c0, *fac, w), v0, 2)
    assert len(hist) == 2 and all(math.isfinite(h) and h > 0 for h in hist), hist


# ---------------------------------------------------------------------------------------------
# aEVP (FESOM2 whichEVP=2) forward of the study


def _alpha_beta_fortran(cfg, mesh, a_ice, m_ice, u, v, c_aevp, alpha_min):
    """Independent numpy transcription of ice_maEVP.F90 find_alpha_field_a (:986-1030) and
    find_beta_field_a (:1314-1325), element by element as written there."""
    en = np.asarray(mesh.elem_nodes)
    gs = np.asarray(mesh.gradient_sca)
    mf = np.asarray(mesh.metric_factor)
    area = np.asarray(mesh.elem_area)
    ulev = np.asarray(mesh.ulevels)
    val3 = 1.0 / 3.0
    vale = 1.0 / cfg.ellipse ** 2
    alpha = np.full(en.shape[0], cfg.alpha_evp)              # alpha_evp_array = alpha_evp
    for elem in range(en.shape[0]):
        nodes = en[elem]
        if ulev[elem] > 1:
            continue
        msum = m_ice[nodes].sum() * val3
        if msum <= 0.01:
            continue
        asum = a_ice[nodes].sum() * val3
        dx, dy = gs[elem, 0:3], gs[elem, 3:6]
        vsum, usum, meancos = v[nodes].sum(), u[nodes].sum(), mf[elem]
        eps11 = (dx * u[nodes]).sum() - val3 * vsum * meancos
        eps22 = (dy * v[nodes]).sum()
        eps12 = 0.5 * (dy * u[nodes] + dx * v[nodes]).sum() + 0.5 * val3 * usum * meancos
        eps1, eps2 = eps11 + eps22, eps11 - eps22
        delta = np.sqrt(eps1 ** 2 + vale * (eps2 ** 2 + 4.0 * eps12 ** 2))
        pressure = cfg.pstar * np.exp(-cfg.c_pressure * (1.0 - asum)) / (delta + cfg.delta_min)
        alpha[elem] = max(alpha_min, np.sqrt(cfg.ice_dt * c_aevp * pressure / cfg.rhoice
                                             / area[elem]))
    beta = np.full(int(mesh.nod2D), -np.inf)
    for elem in range(en.shape[0]):
        for n in en[elem]:
            beta[n] = max(beta[n], alpha[elem])
    return alpha, beta


def test_core2_aevp_alpha_beta_match_the_fortran_transcription(icemap):
    M = icemap
    a, m = (np.asarray(M.inp[k]) for k in ("a", "m"))
    u, v = (np.asarray(q) for q in M.x0[:2])
    al, be = (np.asarray(q) for q in M.p_alpha_beta(M.inp, M.x0[0], M.x0[1]))
    ral, rbe = _alpha_beta_fortran(M.cfg, M.mesh, a, m, u, v, IM.AEVP_C, IM.AEVP_ALPHA_MIN)
    # The strain sums are associated differently (the JAX metric term is mfac·ΣV/3, the Fortran
    # val3·ΣV·mfac): Δ differs in its last bits, which near-rigid elements (Δ ~ δmin, a
    # cancellation) carry into α at up to ~3e-12 relative.
    np.testing.assert_allclose(al, ral, rtol=1e-10, atol=0)
    np.testing.assert_allclose(be, rbe, rtol=1e-10, atol=0)
    # the captured state exercises all three branches: the floor, the formula, no ice
    assert (al[M.ice_el] == IM.AEVP_ALPHA_MIN).any() and (al[M.ice_el] > IM.AEVP_ALPHA_MIN).any()
    assert np.all(al[~M.ice_el] == M.cfg.alpha_evp)
    assert np.array_equal(al, M.alpha0) and np.array_equal(be, M.beta0)


def test_core2_aevp_with_a_constant_alpha_is_the_mevp_map(icemap):
    """c_aevp = 0 and α_min = alpha_evp give α ≡ β ≡ alpha_evp as FIELDS: the aEVP sweep is then
    the mEVP sweep up to XLA's rounding. Not bitwise: the setup values are identical (det1, det2,
    pressure_fac, ... compared eagerly, job 27987677), but the compiled sweep with α as an array
    rounds differently from the one with α a literal: 0.05-0.2 % of the entries differ by about
    one ulp after one sweep, and the forward amplifies that over 120 sweeps (as it does for
    the control programs, test above)."""
    M = IM.IceMap("2013-01-01", forward="aevp", aevp_c=0.0, aevp_alpha_min=icemap.cfg.alpha_evp)
    assert np.all(M.alpha0 == M.cfg.alpha_evp) and np.all(M.beta0 == M.cfg.alpha_evp)
    ia = [np.asarray(q) for q in M.p_iterates(M.x0, M.c0)]          # ia[p] = input of sweep p+1
    im = [np.asarray(q) for q in M.p_iterates_other(M.x0, M.c0)]
    for p, q in zip(ia, im):                                        # after one sweep: ~one ulp
        scale = np.max(np.abs(q[1]))
        assert np.max(np.abs(p[1] - q[1])) <= 1e-14 * scale
        assert np.count_nonzero(p[1] != q[1]) <= 1e-2 * q[1].size
    ones = M.ones()
    ya = [np.asarray(q) for q in M.p_fwd0(M.x0, *M.air0, *ones)]
    ym = [np.asarray(q) for q in M.p_fwd0_other(M.x0, *M.air0, *ones)]
    for q, r in zip(ym, icemap.mevp_out):                           # the mEVP program is the step's
        assert q.tobytes() == np.asarray(r).tobytes()
    assert max(np.max(np.abs(ya[0] - ym[0])), np.max(np.abs(ya[1] - ym[1]))) < 1e-5


# ---------------------------------------------------------------------------------------------
# TLa: the mEVP forward, linearized with tangent-only α_t / β_t


def test_core2_tla_keeps_the_forward_and_is_exact_at_infinite_kappa(icemap):
    """κ0 = ∞ gives α_t ≡ β_t ≡ alpha_evp: the TLa tangent is then the exact tangent, one sweep and
    one sweep to rounding, and its primal is the mEVP forward (to the programs' rounding)."""
    import jax.numpy as jnp
    M = icemap
    x0f = M.flat(M.x0)
    rng = np.random.default_rng(2)
    v = jnp.asarray(rng.standard_normal(int(M.offs[-1])))
    ones = M.ones()
    inf = jnp.asarray([0.0, 0.0, 0.0, np.inf])
    al, be = (np.asarray(q) for q in M.p_tla_alpha(M.x0, np.inf))
    assert np.all(al == M.cfg.alpha_evp) and np.all(be == M.cfg.alpha_evp)
    t_ex = np.asarray(M.p_jvp_sweep(x0f, x0f, M.c0, ones[0], ones[1], 1.0, ones[3], v))
    t_tl = np.asarray(M.p_jvp_sweep_tla(x0f, x0f, M.c0, ones[0], ones[1], 1.0, inf, v))
    np.testing.assert_allclose(t_tl, t_ex, rtol=0, atol=1e-12 * np.max(np.abs(t_ex)))
    y_ex, s_ex = (np.asarray(q) for q in M.p_jvp(x0f, M.c0, *ones, v, jnp.zeros(2)))
    for k0 in (np.inf, 0.25):
        y_tl, s_tl = (np.asarray(q) for q in M.p_jvp_tla(x0f, M.c0, *ones[:3],
                                                         jnp.asarray([0, 0, 0, k0]), v,
                                                         jnp.zeros(2)))
        # the primal is the mEVP sweep; the two compiled programs round differently, and the
        # forward amplifies that (as for the control programs above)
        nu = 2 * M.nod
        assert np.max(np.abs(y_tl[:nu] - y_ex[:nu])) < 1e-5                       # u, v [m/s]
        assert np.max(np.abs(y_tl[nu:] - y_ex[nu:])) < 1e-8 * np.max(np.abs(y_ex[nu:]))   # σ
    y_tl, s_tl = (np.asarray(q) for q in M.p_jvp_tla(x0f, M.c0, *ones[:3], inf, v, jnp.zeros(2)))
    # (no tolerance on the whole step: the exact tangent amplifies ~1e12 per step on this date,
    # so one-ulp differences per sweep would not stay small; recorded below)
    TLA_BITWISE.update(sweep=t_tl.tobytes() == t_ex.tobytes(), step=s_tl.tobytes() == s_ex.tobytes())
    # κ0 = 0.25 raises α_t only where the stability number exceeds κ0
    al25 = np.asarray(M.p_tla_alpha(M.x0, 0.25)[0])
    g = M.element_table(x0f, [])[2] / (M.cfg.alpha_evp * M.cfg.beta_evp)
    up = al25 > M.cfg.alpha_evp
    assert up.any() and M.ice_el[up].all() and np.all(g[up] > 0.25 * (1 - 1e-9))
    np.testing.assert_allclose(al25[up], M.cfg.alpha_evp * np.sqrt(g[up] / 0.25), rtol=1e-9)


TLA_BITWISE = {}


def test_core2_tla_bitwise_report(icemap):
    """Records (does not require) bitwise equality of the κ0 = ∞ TLa tangent with the exact one."""
    if not TLA_BITWISE:
        pytest.skip("the TLa test did not run")
    print("TLA_BITWISE", TLA_BITWISE)


def test_core2_tlai_takes_alpha_from_the_sweep_input_and_is_exact_at_infinite_kappa(icemap):
    """TLai: α_t from each sweep's input iterate. At κ0 = ∞ its one-sweep tangent is the exact one
    (to rounding); at finite κ0 it follows the iterate, not the entry state."""
    import jax.numpy as jnp
    M = icemap
    x0f = M.flat(M.x0)
    its = M.p_iterates(M.x0, M.c0)
    x30 = M.flat(tuple(q[29] for q in its))                    # the input of sweep 30
    del its
    rng = np.random.default_rng(3)
    v = jnp.asarray(rng.standard_normal(int(M.offs[-1])))
    ones = M.ones()
    t_ex = np.asarray(M.p_jvp_sweep(x30, x0f, M.c0, ones[0], ones[1], 1.0, ones[3], v))
    t_inf = np.asarray(M.p_jvp_sweep_tlai(x30, x0f, M.c0, ones[0], ones[1], 1.0,
                                          jnp.asarray([0.0, 0.0, 0.0, np.inf]), v))
    np.testing.assert_allclose(t_inf, t_ex, rtol=0, atol=1e-12 * np.max(np.abs(t_ex)))
    k = jnp.asarray([0.0, 0.0, 0.0, 0.25])
    t_i = np.asarray(M.p_jvp_sweep_tlai(x30, x0f, M.c0, ones[0], ones[1], 1.0, k, v))
    t_e = np.asarray(M.p_jvp_sweep_tla(x30, x0f, M.c0, ones[0], ones[1], 1.0, k, v))
    assert not np.allclose(t_i, t_e)                           # the coefficients differ
    a_i = np.asarray(M.p_tla_alpha(M.unflat(x30), 0.25)[0])
    a_e = np.asarray(M.p_tla_alpha(M.x0, 0.25)[0])
    assert not np.array_equal(a_i, a_e)


# ---------------------------------------------------------------------------------------------
# TLf: the frozen-coefficient tangent


def test_frozen_tangent_is_the_explicit_linear_recursion_and_its_transpose():
    """On a tiny nonlinear 'sweep' g(x, a, c): K applications of the map linearized once at x_N
    equal δx_{k+1} = J_x δx_k + J_a δa + J_c δc with the Jacobians at x_N, and the transpose of
    the whole K-fold map (δa = δx0[:2], δc = 0) is its matrix transpose."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(4)
    A = jnp.asarray(rng.standard_normal((4, 4)) * 0.4)
    B = jnp.asarray(rng.standard_normal((4, 2)))
    d = jnp.asarray(rng.standard_normal(4))

    def g(x, a, c):
        return jnp.tanh(A @ x) * (1.0 + 0.3 * c[0]) + B @ jnp.sin(a) + 0.1 * x * x * c[1]

    xN = jnp.asarray(rng.standard_normal(4))
    a0 = jnp.asarray(rng.standard_normal(2))
    c0 = jnp.asarray([0.2, -0.5])
    _, lin = jax.linearize(g, xN, a0, c0)
    Jx = jax.jacfwd(g, 0)(xN, a0, c0)
    Ja = jax.jacfwd(g, 1)(xN, a0, c0)
    Jc = jax.jacfwd(g, 2)(xN, a0, c0)
    dx0 = jnp.asarray(rng.standard_normal(4))
    da = jnp.asarray(rng.standard_normal(2))
    dc = jnp.asarray(rng.standard_normal(2))
    K = 7
    got = IM.frozen_tangent(lin, dx0, da, dc, K)
    ref = dx0
    for _ in range(K):
        ref = Jx @ ref + Ja @ da + Jc @ dc
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)

    def T(v):
        return IM.frozen_tangent(lin, v, v[:2], jnp.zeros(2), K)
    M_T = jax.jacfwd(T)(dx0)                                  # the K-fold map as a matrix
    P = np.eye(4)
    Mref = np.zeros((4, 4))
    for _ in range(K):                                       # δx_K = Σ terms, assembled explicitly
        Mref = np.asarray(Jx) @ Mref + np.asarray(Ja) @ np.eye(4)[:2]
        P = np.asarray(Jx) @ P
    np.testing.assert_allclose(np.asarray(M_T), P + Mref, rtol=1e-12, atol=1e-12)
    w = jnp.asarray(rng.standard_normal(4))
    tr = jax.linear_transpose(T, dx0)(w)[0]
    np.testing.assert_allclose(np.asarray(tr), (P + Mref).T @ np.asarray(w), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# symmetrized (S) and Picard (P) stress tangents


def _stress_jacobians(e11, e22, e12, pf, vale, dmin, mode):
    """Per-element Jacobian of (S11, S22, 2·S12) w.r.t. (ε11, ε22, ε12): the pairing in which a
    gradient of a potential has a symmetric Jacobian (σ:ε = σ11ε11 + σ22ε22 + 2σ12ε12)."""
    import jax
    import jax.numpy as jnp

    def f(e, p):
        t11, t12, t22 = IM.stress_terms(e[0], e[1], e[2], p, vale, dmin, mode)
        return jnp.stack([t11, t22, 2.0 * t12])
    E = jnp.stack([e11, e22, e12], axis=1)
    return np.asarray(jax.vmap(jax.jacfwd(f), in_axes=(0, 0))(E, pf))


def _check_stress_modes(e11, e22, e12, pf, vale, dmin):
    import jax.numpy as jnp
    eps1, eps2 = e11 + e22, e11 - e22
    delta = np.sqrt(eps1 ** 2 + vale * (eps2 ** 2 + 4 * e12 ** 2))
    ref11 = 0.5 * pf / (delta + dmin) * (eps1 - delta + eps2 * vale)       # the model's formula
    for mode in IM.STRESS_MODES:                                          # values: all modes
        t11, _, _ = IM.stress_terms(jnp.asarray(e11), jnp.asarray(e22), jnp.asarray(e12),
                                    jnp.asarray(pf), vale, dmin, mode)
        np.testing.assert_allclose(np.asarray(t11), ref11, rtol=1e-9, atol=1e-12 * np.abs(ref11).max())
    J = {m: _stress_jacobians(jnp.asarray(e11), jnp.asarray(e22), jnp.asarray(e12),
                              jnp.asarray(pf), vale, dmin, m) for m in IM.STRESS_MODES}
    for m in ("S", "P"):
        Jm = J[m]
        scale = np.abs(Jm).max(axis=(1, 2), keepdims=True)
        np.testing.assert_allclose(Jm, np.transpose(Jm, (0, 2, 1)), rtol=0, atol=1e-10 * scale.max())
        sym = 0.5 * (Jm + np.transpose(Jm, (0, 2, 1)))
        ev = np.linalg.eigvalsh(sym)
        assert np.all(ev >= -1e-10 * scale[:, :, 0])                       # PSD (convex potential)
    asym = np.abs(J["exact"] - np.transpose(J["exact"], (0, 2, 1))).max(axis=(1, 2))
    assert (asym > 1e-3 * np.abs(J["exact"]).max(axis=(1, 2))).any()       # the exact one is not


def test_symmetrized_and_picard_stress_jacobians_are_symmetric_psd():
    import jax
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(5)
    n = 400
    dmin = 1e-11
    mag = 10.0 ** rng.uniform(-12, -6, n)                                 # from rigid to deforming
    e11, e22, e12 = (mag * rng.standard_normal(n) for _ in range(3))
    pf = 10.0 ** rng.uniform(1, 2, n)
    _check_stress_modes(e11, e22, e12, pf, 0.25, dmin)


def test_core2_symmetrized_stress_jacobian_on_the_captured_state(icemap):
    """The same on the captured CORE2 state: the 200 ice elements with the smallest Δ (the
    near-rigid ones where the exact stress Jacobian is most asymmetric) and 200 at random."""
    M = icemap
    e11, e22, e12 = (np.asarray(q) for q in M._strain_rates(M.mesh, M.x0[0], M.x0[1]))
    eps1, eps2 = e11 + e22, e11 - e22
    delta = np.sqrt(eps1 ** 2 + M.cfg.vale * (eps2 ** 2 + 4 * e12 ** 2))
    ice = np.flatnonzero(M.ice_el)
    pick = np.concatenate([ice[np.argsort(delta[ice])[:200]],
                           np.random.default_rng(6).choice(ice, 200, replace=False)])
    pf = np.asarray(M.st0.pressure_fac)[pick]
    _check_stress_modes(e11[pick], e22[pick], e12[pick], pf, M.cfg.vale, M.cfg.delta_min)


def test_partial_viscosity_derivative_interpolates_picard_and_newton():
    """Mode "Tθ": θ = 0 is the Picard (P) stress Jacobian, θ = 1 the symmetrized Newton (S) one,
    and in between the convex combination (so symmetric PSD); the values never change."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(7)
    n = 300
    mag = 10.0 ** rng.uniform(-12, -6, n)
    e11, e22, e12 = (jnp.asarray(mag * rng.standard_normal(n)) for _ in range(3))
    pf = jnp.asarray(10.0 ** rng.uniform(1, 2, n))
    vale, dmin = 0.25, 1e-11
    J = {m: _stress_jacobians(e11, e22, e12, pf, vale, dmin, m)
         for m in ("P", "S", "T0", "T1", "T0.5")}
    scale = np.abs(J["S"]).max()
    np.testing.assert_allclose(J["T0"], J["P"], rtol=0, atol=1e-13 * scale)
    np.testing.assert_allclose(J["T1"], J["S"], rtol=0, atol=1e-13 * scale)
    np.testing.assert_allclose(J["T0.5"], 0.5 * (J["P"] + J["S"]), rtol=0, atol=1e-12 * scale)
    vals = {m: np.asarray(IM.stress_terms(e11, e22, e12, pf, vale, dmin, m)[0])
            for m in ("P", "T0.5", "exact")}
    np.testing.assert_array_equal(vals["T0.5"], vals["P"])
    with pytest.raises(ValueError):
        IM.stress_terms(e11, e22, e12, pf, vale, dmin, "Q")


# ---------------------------------------------------------------------------------------------
# Task 23: the damped frozen iteration (T2) and the steady operator (T1)


def _linear_lin(Jx, Ja, Jc):
    import jax.numpy as jnp
    Jx, Ja, Jc = (jnp.asarray(q) for q in (Jx, Ja, Jc))
    return lambda d, a, c: Jx @ d + Ja @ a + Jc @ c


def _rotation_block(r, phi):
    return r * np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def test_damped_frozen_tangent_is_the_averaged_recursion_with_the_same_fixed_point():
    """ω < 1: δx ← (1−ω)·δx + ω·lin(δx, …), exactly; ω = 1 is the plain map bitwise. On a map
    with a complex pair |λ| = 1.02 at 1.1 rad (Re λ < 1) the plain iteration grows and the
    damped one converges, to the plain map's fixed point (I − J)⁻¹·b."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(8)
    Jx = np.zeros((4, 4))
    Jx[:2, :2] = _rotation_block(1.02, 1.1)
    Jx[2:, 2:] = _rotation_block(0.9, 0.3)
    Ja, Jc = rng.standard_normal((4, 2)), rng.standard_normal((4, 2))
    lin = _linear_lin(Jx, Ja, Jc)
    dx0, da, dc = (jnp.asarray(rng.standard_normal(k)) for k in (4, 2, 2))
    for om in (0.5, 0.8):
        got = np.asarray(IM.frozen_tangent(lin, dx0, da, dc, 9, om))
        ref = np.asarray(dx0)
        for _ in range(9):
            ref = (1.0 - om) * ref + om * (Jx @ ref + Ja @ np.asarray(da) + Jc @ np.asarray(dc))
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(IM.frozen_tangent(lin, dx0, da, dc, 9, 1.0)),
                                  np.asarray(IM.frozen_tangent(lin, dx0, da, dc, 9)))
    b = Jx @ np.zeros(4) + Ja @ np.asarray(da) + Jc @ np.asarray(dc)
    xstar = np.linalg.solve(np.eye(4) - Jx, b)
    plain = np.asarray(IM.frozen_tangent(lin, dx0, da, dc, 600))
    damped = np.asarray(IM.frozen_tangent(lin, dx0, da, dc, 600, 0.5))
    assert np.linalg.norm(plain - xstar) > 1e3                    # 1.02^600 ≈ 1.4e5
    np.testing.assert_allclose(damped, xstar, rtol=1e-8, atol=1e-8)
    T = jax.linear_transpose(lambda v: IM.frozen_tangent(lin, v, v[:2], jnp.zeros(2), 9, 0.8),
                             dx0)
    Mt = jax.jacfwd(lambda v: IM.frozen_tangent(lin, v, v[:2], jnp.zeros(2), 9, 0.8))(dx0)
    w = jnp.asarray(rng.standard_normal(4))
    np.testing.assert_allclose(np.asarray(T(w)[0]), np.asarray(Mt).T @ np.asarray(w),
                               rtol=1e-12, atol=1e-12)


def test_tl_forms_parse_the_damping_suffix():
    assert IM.tl_kind_mode("tlfSw0.8") == ("tlf", "S") and IM.tl_omega("tlfSw0.8") == 0.8
    assert IM.tl_kind_mode("tlfw0.5") == ("tlf", "exact") and IM.tl_omega("tlfw0.5") == 0.5
    assert IM.tl_kind_mode("tlfT0.97w0.8") == ("tlf", "T0.97")
    assert IM.tl_omega("tlfT0.97w0.8") == 0.8
    assert IM.tl_kind_mode("tlfT0.95") == ("tlf", "T0.95") and IM.tl_omega("tlfT0.95") == 1.0
    assert IM.tl_omega("tlaiS") == 1.0 and IM.tl_omega("local") == 1.0
    names = {v for v, _, _ in IM.VARIANTS}
    assert {"TLfT0.97", "TLfT0.99", "TLfS1w0.5", "TLfS1w0.8", "TLf1w0.5", "TLf1w0.8", "TLfT0.97w0.8",
            "TLfT0.99w0.8"} <= names


def test_lanczos_brackets_the_spectrum_and_finds_an_isolated_negative_eigenvalue():
    rng = np.random.default_rng(9)
    n = 300
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.concatenate([[-5.0], rng.uniform(1.0, 2.0, n - 2), [1e3]])
    S = (Q * lam) @ Q.T
    rec, x = IM.lanczos_extremes(lambda v: S @ v, rng.standard_normal(n), 120,
                                 checkpoints=(10, 40, 120))
    assert [r["m"] for r in rec] == [10, 40, 120]
    assert abs(rec[-1]["ritz_min"][0] + 5.0) < 1e-6
    assert abs(rec[-1]["ritz_max"] - 1e3) < 1e-6
    assert abs(abs(x @ Q[:, 0]) - 1.0) < 1e-6
    for r in rec:                                       # Ritz values lie inside the spectrum
        assert r["ritz_min"][0] >= -5.0 - 1e-9 and r["ritz_max"] <= 1e3 + 1e-9


def test_element_pairing_jacobians_match_the_stress_test_helper():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(11)
    n = 200
    mag = 10.0 ** rng.uniform(-12, -6, n)
    e11, e22, e12 = (mag * rng.standard_normal(n) for _ in range(3))
    pf = 10.0 ** rng.uniform(1, 2, n)
    for mode in ("exact", "S", "P", "T0.97"):
        E = IM.element_pairing_jacobians(e11, e22, e12, pf, 0.25, 1e-11, mode)
        ref = _stress_jacobians(jnp.asarray(e11), jnp.asarray(e22), jnp.asarray(e12),
                                jnp.asarray(pf), 0.25, 1e-11, mode)
        np.testing.assert_allclose(E, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_strain_matrix_and_the_divergence_is_minus_its_transpose():
    """One element: element_strain_matrix reproduces ice_evp.strain_rates (metric term
    included), and the stress-divergence scatter is −A_e·Bᵀ·(σ11, σ22, 2σ12): the pairing in
    which the steady operator's stress part is symmetric for a symmetric stress Jacobian."""
    import types
    import jax
    import jax.numpy as jnp
    from fesom_jax.ice_evp import strain_rates, stress_div_scatter
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(12)
    gs = rng.standard_normal(6) * 1e-5
    mf, area = 3e-7, 2.5e9
    mesh = types.SimpleNamespace(gradient_sca=jnp.asarray(gs[None, :]),
                                 metric_factor=jnp.asarray([mf]),
                                 elem_nodes=jnp.asarray([[0, 1, 2]]), elem_area=jnp.asarray([area]),
                                 nod2D=3)
    u, v = rng.standard_normal(3), rng.standard_normal(3)
    B = IM.element_strain_matrix(gs, mf)
    got = np.array([float(q[0]) for q in strain_rates(mesh, jnp.asarray(u), jnp.asarray(v))])
    np.testing.assert_allclose(got, B @ np.concatenate([u, v]), rtol=1e-12, atol=0)
    s11, s12, s22 = rng.standard_normal(3) * 1e4
    ur, vr = stress_div_scatter(mesh, jnp.asarray([s11]), jnp.asarray([s12]), jnp.asarray([s22]),
                                jnp.asarray([True]))
    ref = -area * B.T @ np.array([s11, s22, 2.0 * s12])
    np.testing.assert_allclose(np.concatenate([np.asarray(ur), np.asarray(vr)]), ref,
                               rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_core2_steady_operator_transpose_and_the_symmetrized_floor(icemap):
    """CORE2 2013-01-01 at u_N: ⟨w, A d⟩ = ⟨Aᵀ w, d⟩ on the active ice nodes (exact and S),
    and with the symmetrized stress Jacobian the mass-weighted quadratic form of A is at least
    the inertia term, dᵀWAd ≥ dᵀWd, also along uniform translations (little strain)."""
    M = icemap
    jnp = M.jnp
    xN = M.p_fwd(M.x0, M.c0, *M.ones())
    uvN, x_e = (xN[0], xN[1]), (M.x0[0], M.x0[1])
    _, _, W = (np.asarray(q) for q in M.p_steady_coef(x_e, uvN))
    act = M.ice_nod & (np.asarray(M.st0.bc_index) > 0) & (W > 0)
    rng = np.random.default_rng(10)

    def field():
        return tuple(jnp.asarray(np.where(act, rng.standard_normal(M.nod), 0.0)) for _ in range(2))
    for mode in ("exact", "S"):
        A, AT, _ = M.steady_programs(mode)
        d, w = field(), field()
        Ad, ATw = A(x_e, uvN, d), AT(x_e, uvN, w)
        lhs = sum(float(np.sum((np.asarray(p) * np.asarray(q))[act])) for p, q in zip(w, Ad))
        rhs = sum(float(np.sum((np.asarray(p) * np.asarray(q))[act])) for p, q in zip(d, ATw))
        assert abs(lhs - rhs) <= 1e-9 * max(abs(lhs), abs(rhs))
    A, _, _ = M.steady_programs("S")
    one, zero = np.where(act, 1.0, 0.0), np.zeros(M.nod)
    for d in [field(), field(), (jnp.asarray(one), jnp.asarray(zero)),
              (jnp.asarray(zero), jnp.asarray(one))]:
        Ad = A(x_e, uvN, d)
        q = sum(float(np.sum((W * np.asarray(p) * np.asarray(a))[act])) for p, a in zip(d, Ad))
        n2 = sum(float(np.sum((W * np.asarray(p) ** 2)[act])) for p in d)
        assert q >= (1.0 - 1e-9) * n2


# ---------------------------------------------------------------------------------------------
# Task 25: the helpers of `why` (synthetic) and the constant-α forward (CORE2)


def test_fd_record_central_is_the_mean_of_the_one_sided_and_a_kink_splits_them():
    """f(c) = |c| + 2c has a kink at 0: forward 3, backward 1, central their mean 2; a smooth
    f(c) = c³ + 2c has the three within O(h)."""
    for h in (1e-2, 1e-4):
        r = IM.fd_record(0.0, 3.0 * h, -h, h)
        assert r["forward"] == pytest.approx(3.0) and r["backward"] == pytest.approx(1.0)
        assert r["central"] == pytest.approx(0.5 * (r["forward"] + r["backward"]))
        f = lambda c: c ** 3 + 2.0 * c                                  # noqa: E731
        s = IM.fd_record(f(0.0), f(h), f(-h), h)
        assert abs(s["forward"] - s["backward"]) <= 3.0 * h
        assert s["central"] == pytest.approx(2.0, abs=2.0 * h * h)


def test_node_extremes_over_masked_elements():
    en = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    vals = np.array([1.0, 5.0, -2.0])
    mx = IM.node_extreme_of_elements(vals, en, 6, np.array([True, False, True]))
    np.testing.assert_array_equal(mx[:5], [1.0, 1.0, 1.0, -2.0, -2.0])
    assert np.isnan(mx[5])                                   # no element at all
    mn = IM.node_extreme_of_elements(vals, en, 6, None, "min")
    np.testing.assert_array_equal(mn[:5], [1.0, 1.0, -2.0, -2.0, -2.0])


def test_binned_contributions_add_up_and_skip_unbinned_nodes():
    rng = np.random.default_rng(4)
    n = 500
    prop = rng.uniform(0.0, 30.0, n)
    prop[:7] = np.nan                                        # no ice element around the node
    mask = rng.uniform(size=n) < 0.8
    c = {"fd": rng.standard_normal(n), "excess": rng.standard_normal(n)}
    rows = IM.binned_contributions(c, prop, IM.WHY_EDGES["gab250"], mask)
    sel = mask & np.isfinite(prop)
    assert sum(r["n"] for r in rows) == int(sel.sum())
    for k in c:
        assert sum(r[k] for r in rows) == pytest.approx(float(np.sum(c[k][sel])))
    assert rows[-1]["hi"] == math.inf and rows[-1]["n"] == int((sel & (prop >= 20.0)).sum())


def test_concentration_of_a_single_dominant_entry():
    x = np.concatenate([[100.0], np.full(999, 0.1)])
    c = IM.concentration(x)
    assert c["0.001"]["k"] == 1
    assert c["0.001"]["signed_share"] == pytest.approx(100.0 / 199.9)
    assert c["0.1"]["abs_share"] == pytest.approx((100.0 + 99 * 0.1) / 199.9)


def test_why_parser_needs_the_reference_h_and_the_constant():
    with pytest.raises(SystemExit):
        IM._parse(["why", "--date", "2013-01-01", "--out", "/tmp/x", "--hs", "1e-3"])
    with pytest.raises(SystemExit):
        IM._parse(["why", "--date", "2013-01-01", "--out", "/tmp/x", "--forward", "const"])
    a = IM._parse(["why", "--date", "2013-01-01", "--out", "/tmp/x", "--forward", "const",
                   "--alpha-const", "3000", "--n-sweeps", "24000"])
    assert a.alpha_const == 3000.0 and a.n_sweeps == 24000 and IM.WHY_REF_H in a.hs


def test_core2_const_forward_at_250_is_the_mevp_map(icemap):
    """forward 'const' with α = β = alpha_evp is the model's mEVP sweep (bitwise: the same
    scalar config); its TLa coefficients equal mEVP's up to the rounding of γ/(αβ); at
    α = 3000 the sweep changes and the tangent rows are allowed."""
    M = IM.IceMap("2013-01-01", forward="const", alpha_const=icemap.cfg.alpha_evp)
    ones = M.ones()
    yc = [np.asarray(q) for q in M.p_fwd0(M.x0, *M.air0, *ones)]
    for q, r in zip(yc, icemap.mevp_out):
        assert q.tobytes() == np.asarray(r).tobytes()
    ac = np.asarray(M.p_tla_alpha(M.x0, 1.0)[0])
    am = np.asarray(icemap.p_tla_alpha(icemap.x0, 1.0)[0])
    np.testing.assert_allclose(ac, am, rtol=1e-12, atol=0)
    M3 = IM.IceMap("2013-01-01", forward="const", alpha_const=3000.0, n_sweeps=20)
    assert M3.factors("tlfS", 1.0)[3][3] == 1.0
    a3 = np.asarray(M3.p_tla_alpha(M3.x0, 1.0)[0])
    assert np.all(a3 >= 3000.0)
    y3 = [np.asarray(q) for q in M3.p_fwd(M3.x0, M3.c0, *M3.ones())]
    assert all(np.all(np.isfinite(q)) for q in y3)
    with pytest.raises(ValueError):
        IM.IceMap("2013-01-01", forward="const")


def test_core2_why_smoke_writes_its_tables_and_the_bins_add_up(icemap, tmp_path):
    """`why` end to end with 10 sweeps, two h and two forms; the a-binned sums of every
    contribution equal its box total (a has no NaN on box nodes); `why-summary` reads it."""
    import json as _json
    out = tmp_path / "why"
    rc = IM.main(["why", "--date", "2013-01-01", "--out", str(out), "--n-sweeps", "10",
                  "--hs", "1e-2", "1e-4", "--field-hs", "1e-4", "--forms", "tlfS", "tlfT0.97",
                  "--field-forms", "tlfS", "tlfT0.97", "--repeats", "1"])
    assert rc == 0
    doc = _json.loads((out / "why_20130101_mevp_n10.json").read_text())
    assert set(doc["fd"]) == {"lnp", "lncd"} and set(doc["tl"]) == {"tlfS", "tlfT0.97"}
    assert doc["tl"]["tlfS"]["lnp"]["finite"]
    tot = doc["attribution"]["totals"]
    for k, v in tot.items():
        assert sum(b[k] for b in doc["attribution"]["bins"]["a"]) == pytest.approx(v, rel=1e-9, abs=1e-15)
    assert tot["excess_tlfS"] == pytest.approx(tot["tlfS"] - tot["fd"], rel=1e-9, abs=1e-15)
    F = np.load(out / "why_20130101_mevp_n10_fields.npz")
    assert "c_excess_tlfS" in F.files and "prop_gab250" in F.files
    assert (out / "why_20130101_mevp_n10.md").exists()
    assert IM.main(["why-summary", "--dir", str(out)]) == 0
    assert (out / "why_summary.md").exists() and (out / "why_20130101_hscan.png").exists()
