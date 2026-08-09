"""``scripts/nn_eddy/apriori_eval.py`` — the scorer GATE M3 is read from.

The gate's whole value is that its two measures mean what the table says they mean, so the
things worth testing here are the ones a wrong answer would look plausible in:

* the streaming accumulator must give exactly what a direct computation over the same numbers
  gives — it never stores a cell, and a silent sign or normalisation error in the sums would
  produce a perfectly reasonable-looking R² that is simply not R²;
* an UNTRAINED network must score bit-identically to the formula baseline in both measures.
  ``init_gm_nn(zero_last=True)`` makes ``m ≡ 1``, so the two closures are the same closure
  reached by two different code paths — the NN path through ``mlp_raw``/``tanh``/``exp`` and
  the formula path through a lambda. If the feature assembly or the multiplier convention ever
  skews, that identity is the first thing to break, and this is the only place the scorer would
  notice;
* the stratum edges must put a cell in the band its depth/latitude/resolution names, because
  the gate's second condition is per depth band and an off-by-one in ``digitize`` moves cells
  across a band boundary without ever failing anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "nn_eddy"))

import fesom_jax  # noqa: E402,F401
import jax.numpy as jnp  # noqa: E402
from fesom_jax import gm_nn  # noqa: E402

import apriori_eval as ae  # noqa: E402


# ---------------------------------------------------------------------------
# the accumulator
# ---------------------------------------------------------------------------
def test_streaming_r2_equals_direct_r2():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(500, 2))
    p0 = y + 0.3 * rng.normal(size=y.shape)
    p1 = np.zeros_like(y)

    acc = ae.Acc(2)
    for lo in range(0, 500, 37):                 # fed in ragged chunks on purpose
        sl = slice(lo, lo + 37)
        acc.add_logm(y[sl], [p0[sl], p1[sl]])

    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
    want = [1.0 - float(np.sum((y - p) ** 2)) / ss_tot for p in (p0, p1)]
    got = acc.r2()
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_flux_ev_is_about_zero_not_about_the_mean():
    """A closure predicting nothing must score 0.0, not be flattered by a removed mean."""
    acc = ae.Acc(1)
    acc.add_flux(2, np.array([50.0, 18.0]), np.array([[50.0, 18.0]]), np.array([[0.0, 0.0]]))
    assert acc.ev() == [[0.0, 0.0]]

    # and a closure that reproduces the flux exactly must score 1.0
    acc2 = ae.Acc(1)
    acc2.add_flux(2, np.array([50.0, 18.0]), np.array([[0.0, 0.0]]), np.array([[50.0, 18.0]]))
    assert acc2.ev() == [[1.0, 1.0]]

    # a closure whose error EXCEEDS the signal scores below zero — the whole point of the
    # zero reference, and the reading the Arm A / Arm B decision depends on.
    acc3 = ae.Acc(1)
    acc3.add_flux(2, np.array([50.0, 18.0]), np.array([[200.0, 18.0]]), np.array([[0.0, 0.0]]))
    ev = acc3.ev()[0]
    assert ev[0] == pytest.approx(-3.0)
    assert ev[1] == pytest.approx(0.0)


def test_rms_ratio_separates_a_scale_error_from_a_skill_failure():
    """The guard that would have caught the untapered-slope operator immediately.

    Both closures below score a catastrophic explained variance. Only the RMS ratio says which
    one is a wrong closure (right magnitude, wrong direction) and which is a wrong operator
    (magnitude off by orders), and that distinction is the difference between a physics result
    and a bug."""
    acc = ae.Acc(2)
    # closure 0: predicts a flux of the same size as the teacher's but uncorrelated with it
    # closure 1: predicts a flux 1000x too big
    acc.add_flux(1, np.array([1.0, 1.0]),
                 np.array([[2.0, 2.0], [1.0e6, 1.0e6]]),
                 np.array([[1.0, 1.0], [1.0e6, 1.0e6]]))
    ev = acc.ev()
    assert ev[0] == [pytest.approx(-1.0), pytest.approx(-1.0)]
    assert ev[1][0] < -1e5 and ev[1][1] < -1e5
    rr = acc.rms_ratio()
    assert rr[0] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert rr[1] == [pytest.approx(1.0e3), pytest.approx(1.0e3)]


def test_divergence_measure_uses_the_same_zero_reference_as_the_flux_measure():
    """Measure 3 is the one a "the difference is rotational" objection cannot be raised
    against, so it must keep the zero reference — a mean-referenced version would let a closure
    that supplies no divergence at all score 0 for the wrong reason."""
    acc = ae.Acc(2)
    acc.add_div(3, np.array([10.0, 4.0]), np.array([[10.0, 4.0], [0.0, 0.0]]))
    assert acc.ev_div() == [[0.0, 0.0], [1.0, 1.0]]
    assert ae.Acc(1).ev_div() is None


def test_r2_needs_two_points_and_spread():
    assert ae.Acc(1).r2() is None                       # nothing accumulated
    acc = ae.Acc(1)
    y = np.ones((10, 2))
    acc.add_logm(y, [y])
    assert acc.r2() is None                             # zero variance: R² undefined, not 1.0


# ---------------------------------------------------------------------------
# the untrained network IS the formula
# ---------------------------------------------------------------------------
def test_untrained_network_scores_identically_to_the_formula():
    nn = gm_nn.init_gm_nn(jnp.zeros(2, dtype=jnp.uint32), hidden=(16, 16), zero_last=True)
    rng = np.random.default_rng(1)
    n = 256
    feats = rng.normal(size=(n, gm_nn.n_features(False)))
    bg = np.exp(rng.normal(size=n))
    br = np.exp(rng.normal(size=n))

    lm = np.asarray(nn.log_m_max * jnp.tanh(gm_nn.mlp_raw(nn, jnp.asarray(feats))))
    kg, kr = bg * np.exp(lm[..., 0]), br * np.exp(lm[..., 1])
    # bit-for-bit, not approximately: tanh(0) is exactly 0 and exp(0) exactly 1.
    assert np.array_equal(kg, bg)
    assert np.array_equal(kr, br)


def test_make_closures_puts_every_closure_through_one_interface(tmp_path):
    nn = gm_nn.init_gm_nn(jnp.zeros(2, dtype=jnp.uint32), hidden=(8,), zero_last=True)
    p = tmp_path / "w.npz"
    gm_nn.save_gm_nn(p, nn)

    closures = ae.make_closures([("D0", p)], [np.log(2.0), np.log(0.5)])
    names = [nm for nm, _ in closures]
    assert names[0] == "D0" and names[1] == "formula (m=1)"
    assert "const m=2.00/0.50" in names[2]

    feats = np.zeros((4, gm_nn.n_features(False)))
    bg = np.array([1.0, 2.0, 4.0, 8.0])
    br = np.array([1.0, 1.0, 1.0, 1.0])
    kg_nn, kr_nn = closures[0][1](feats, bg, br)
    kg_fm, kr_fm = closures[1][1](feats, bg, br)
    kg_c, kr_c = closures[2][1](feats, bg, br)
    assert np.array_equal(kg_nn, kg_fm) and np.array_equal(kr_nn, kr_fm)
    assert kg_c == pytest.approx(2.0 * bg)
    assert kr_c == pytest.approx(0.5 * br)


# ---------------------------------------------------------------------------
# the ceiling row
# ---------------------------------------------------------------------------
def _nnls2_ref(A, B, Y):
    """Brute-force reference: the constrained optimum found by dense search, no algebra."""
    best, arg = None, (0.0, 0.0)
    grid = np.concatenate([[0.0], np.geomspace(1e-4, 1e4, 4001)])
    for k1 in grid:
        r = Y - k1 * A
        # optimal k2 >= 0 given k1
        k2 = max(0.0, float(r @ B) / float(B @ B)) if float(B @ B) > 0 else 0.0
        v = float(np.sum((r - k2 * B) ** 2))
        if best is None or v < best:
            best, arg = v, (k1, k2)
    return arg, best


@pytest.mark.parametrize("seed", range(6))
def test_nnls2_matches_a_brute_force_constrained_optimum(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=4)
    B = rng.normal(size=4)
    Y = rng.normal(size=4)
    k1, k2 = ae.nnls2(np.array(A @ A), np.array(A @ B), np.array(B @ B),
                      np.array(A @ Y), np.array(B @ Y))
    got = float(np.sum((Y - k1 * A - k2 * B) ** 2))
    _, want = _nnls2_ref(A, B, Y)
    assert k1 >= 0.0 and k2 >= 0.0
    # the closed form must be at least as good as the search, up to the grid's resolution
    assert got <= want * (1.0 + 1e-6) + 1e-12


def test_nnls2_recovers_a_known_nonnegative_solution_exactly():
    A = np.array([1.0, 0.0, 2.0, 1.0])
    B = np.array([0.0, 1.0, 1.0, -1.0])
    Y = 3.0 * A + 5.0 * B
    k1, k2 = ae.nnls2(np.array(A @ A), np.array(A @ B), np.array(B @ B),
                      np.array(A @ Y), np.array(B @ Y))
    assert k1 == pytest.approx(3.0)
    assert k2 == pytest.approx(5.0)


def test_nnls2_clamps_rather_than_returning_a_negative_kappa():
    """kappa < 0 is up-gradient. Arm A cannot reach it, so the ceiling must not use it."""
    A = np.array([1.0, 0.0])
    B = np.array([0.0, 1.0])
    Y = np.array([2.0, -7.0])            # the unconstrained k2 is -7
    k1, k2 = ae.nnls2(np.array(A @ A), np.array(A @ B), np.array(B @ B),
                      np.array(A @ Y), np.array(B @ Y))
    assert k1 == pytest.approx(2.0)
    assert k2 == 0.0


def test_nnls2_is_not_a_clip_of_the_unconstrained_solution():
    """The case that separates exact NNLS from solve-then-clip: with correlated columns,
    clamping one coefficient must MOVE the other, and clipping leaves it wrong."""
    A = np.array([1.0, 1.0, 0.0])
    B = np.array([1.0, 1.02, 0.0])        # nearly collinear with A
    Y = np.array([1.0, 0.0, 0.0])
    a11, a12, a22 = A @ A, A @ B, B @ B
    b1, b2 = A @ Y, B @ Y
    k1, k2 = ae.nnls2(np.array(a11), np.array(a12), np.array(a22), np.array(b1), np.array(b2))
    det = a11 * a22 - a12 * a12
    u1 = (a22 * b1 - a12 * b2) / det
    u2 = (a11 * b2 - a12 * b1) / det
    assert min(u1, u2) < 0.0                       # the unconstrained solution is infeasible
    clipped = (max(u1, 0.0), max(u2, 0.0))
    r_exact = float(np.sum((Y - k1 * A - k2 * B) ** 2))
    r_clip = float(np.sum((Y - clipped[0] * A - clipped[1] * B) ** 2))
    assert r_exact < r_clip
    assert k1 >= 0.0 and k2 >= 0.0


def test_arm_b_ceiling_contains_the_gm_half_of_arm_a():
    """A pure GM flux must be recovered EXACTLY by the free streamfunction.

    This is the containment claim the whole comparison rests on: Arm A's GM part is the special
    case Psi = kappa*(-S_y, S_x), so if the teacher's flux happened to be exactly what GM
    produces, the Arm B ceiling has to score a perfect fit. If it did not, the two ceilings would
    not be nested and the difference between them would not be 'what freeing the direction
    buys'."""
    rng = np.random.default_rng(3)
    n = 200
    slope = rng.normal(size=(n, 2))
    dzT = rng.normal(size=n)
    dzS = rng.normal(size=n)
    kappa = np.abs(rng.normal(size=n)) + 0.1
    F_T = kappa[:, None] * slope * dzT[:, None]
    F_S = kappa[:, None] * slope * dzS[:, None]

    pT, pS = ae.oracle_psi(dzT, dzS, F_T, F_S, 1.0)
    assert np.allclose(pT, F_T, rtol=1e-10, atol=1e-12)
    assert np.allclose(pS, F_S, rtol=1e-10, atol=1e-12)


def test_arm_b_ceiling_reaches_what_arm_a_cannot():
    """An up-gradient flux: Arm A's non-negative kappa cannot produce it, Arm B can."""
    dzT = np.array([1.0, 1.0])
    dzS = np.array([2.0, -1.0])
    # a flux whose direction is unrelated to any slope, with a sign Arm A cannot reach
    F_T = np.array([[-3.0, 1.0], [0.5, -2.0]])
    F_S = 2.0 * F_T * (dzS / dzT)[:, None]
    pT, pS = ae.oracle_psi(dzT, dzS, F_T, F_S, 1.0)
    # the fit is overdetermined here (S is not a multiple of T's flux), so it will not be exact;
    # what must hold is that it captures a real fraction rather than being pinned at zero
    ev = 1.0 - np.sum((F_T - pT) ** 2) / np.sum(F_T ** 2)
    assert ev > 0.3
    assert np.any(np.sign(pT) != 0)


def test_oracle_psi_takes_only_derivatives_so_the_tracer_zero_point_cannot_enter():
    """Invariance to adding a constant to T is STRUCTURAL here, not a numerical property: the
    teacher's eddy flux is unchanged by the offset, and the skew form only ever sees d/dz. This
    asserts the structure, because a value-based test would pass no matter what the body did.
    The advective form u*·T would fail this — Celsius has an arbitrary zero."""
    import inspect
    params = list(inspect.signature(ae.oracle_psi).parameters)
    assert params == ["dzT", "dzS", "F_T", "F_S", "w"]
    assert not any(p in ("T", "S", "d", "mesh") for p in params)


def _psi_from_gamma(gamma, nwet):
    """Layer-centred Psi from an interface Gamma, the same map the constrained fit inverts."""
    psi = np.zeros((gamma.shape[0], gamma.shape[1]))
    psi[:, :-1] = 0.5 * (gamma[:, :-1] + gamma[:, 1:])
    psi[:, -1] = 0.5 * gamma[:, -1]
    lay = np.arange(psi.shape[1])[None, :]
    return np.where(lay < nwet[:, None], psi, 0.0)


def test_constrained_arm_b_recovers_a_planted_streamfunction_exactly():
    """A flux generated by a REAL streamfunction — zero at the surface, zero at the bottom — must
    come back exactly. If it did not, the constrained row would be scoring its own solver rather
    than the physics, and its gap below the free row would not mean what the table says."""
    rng = np.random.default_rng(11)
    N, nlay = 40, 12
    nwet = rng.integers(4, nlay + 1, size=N)
    lay = np.arange(nlay)[None, :]
    wet = lay < nwet[:, None]

    gy = rng.normal(size=(N, nlay))
    gx = rng.normal(size=(N, nlay))
    gy[:, 0] = 0.0                                  # Gamma = 0 at the surface
    gx[:, 0] = 0.0
    gy = np.where(lay < nwet[:, None], gy, 0.0)     # and below the bottom
    gx = np.where(lay < nwet[:, None], gx, 0.0)

    psi_y = _psi_from_gamma(gy, nwet)
    psi_x = _psi_from_gamma(gx, nwet)
    dzT = np.where(wet, rng.normal(size=(N, nlay)), 0.0)
    dzS = np.where(wet, rng.normal(size=(N, nlay)), 0.0)
    F_T = np.stack([psi_y * dzT, -psi_x * dzT], axis=-1)
    F_S = np.stack([psi_y * dzS, -psi_x * dzS], axis=-1)

    pT, pS = ae.oracle_psi_constrained(dzT, dzS, F_T, F_S, 1.0, nwet, ridge=0.0)
    assert np.allclose(pT[wet], F_T[wet], atol=1e-8)
    assert np.allclose(pS[wet], F_S[wet], atol=1e-8)


def test_constrained_arm_b_can_never_beat_the_free_one():
    """The constraint removes freedom, so its residual must be >= the free fit's on any data.
    A constrained row that scored HIGHER would mean the solver is wrong, and the sign of the
    Arm A / Arm B comparison rests on these two rows bracketing the truth."""
    rng = np.random.default_rng(12)
    N, nlay = 60, 10
    nwet = np.full(N, nlay)
    wet = np.ones((N, nlay), bool)
    dzT = rng.normal(size=(N, nlay))
    dzS = rng.normal(size=(N, nlay))
    F_T = rng.normal(size=(N, nlay, 2))            # arbitrary, NOT from any streamfunction
    F_S = rng.normal(size=(N, nlay, 2))

    fT, fS = ae.oracle_psi(dzT, dzS, F_T, F_S, 1.0)
    cT, cS = ae.oracle_psi_constrained(dzT, dzS, F_T, F_S, 1.0, nwet, ridge=0.0)
    res_free = np.sum((F_T[wet] - fT[wet]) ** 2) + np.sum((F_S[wet] - fS[wet]) ** 2)
    res_con = np.sum((F_T[wet] - cT[wet]) ** 2) + np.sum((F_S[wet] - cS[wet]) ** 2)
    assert res_con >= res_free * (1.0 - 1e-9)
    assert res_con > res_free            # on random data the constraint must actually bind


def test_ridge_default_is_off_because_the_deep_anomaly_is_not_understood():
    """The abyssal overshoot in the constrained row is MEASURED (2.41x the teacher's flux below
    1500 m, 2.83x below 3000 m, against 0.95/0.93 for the free fit) but not explained. A ridge was
    tried as a minimum-norm fix and does NOT behave as the explanation predicts: on a column built
    to that description the unregularised solve already returns exactly zero in the weakly-weighted
    layers, and the ridge introduces a small nonzero instead of suppressing anything.

    So the knob stays at zero and this test pins that, rather than shipping a fix whose premise
    failed its own check. Delete this test when the mechanism is reproduced."""
    import inspect
    assert inspect.signature(ae.oracle_psi_constrained).parameters["ridge"].default == 0.0

    nlay = 8
    nwet = np.array([nlay])
    dzT = np.array([[1.0, 1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1e-6]])
    dzS = dzT.copy()
    F_T = np.zeros((1, nlay, 2))
    F_T[0, :4, 0] = [1.0, 2.0, 1.5, 0.5]
    F_S = F_T.copy()
    p0, _ = ae.oracle_psi_constrained(dzT, dzS, F_T, F_S, 1.0, nwet, ridge=0.0)
    p1, _ = ae.oracle_psi_constrained(dzT, dzS, F_T, F_S, 1.0, nwet, ridge=1e-3)
    psi0 = p0[0, :, 0] / dzT[0]
    psi1 = p1[0, :, 0] / dzT[0]
    # the documented counter-result: unregularised is already exactly zero in the deep,
    # regularised is not. Recorded so a future fix has to explain this, not rediscover it.
    assert np.max(np.abs(psi0[4:])) == 0.0
    assert np.max(np.abs(psi1[4:])) > 0.0


def test_constrained_arm_b_matches_a_dense_least_squares():
    """The tridiagonal normal equations must give what an explicit dense solve gives. This is the
    check on the algebra itself — a wrong coefficient in the assembly would still produce a
    plausible, smooth, wrong answer that no other test here would catch."""
    rng = np.random.default_rng(13)
    nlay = 7
    nwet = np.array([nlay])
    dzT = rng.normal(size=(1, nlay))
    dzS = rng.normal(size=(1, nlay))
    F_T = rng.normal(size=(1, nlay, 2))
    F_S = rng.normal(size=(1, nlay, 2))
    w = 0.7

    # dense: unknowns are Gamma at interfaces 1..nlay-1 (Gamma_0 = 0, Gamma_nlay = 0)
    nunk = nlay - 1
    A = np.zeros((2 * nlay, nunk))
    for k in range(nlay):
        for j in (k, k + 1):
            if 1 <= j <= nunk:
                A[k, j - 1] += 0.5 * dzT[0, k]
                A[nlay + k, j - 1] += 0.5 * w * dzS[0, k]
    y = np.concatenate([F_T[0, :, 0], w * F_S[0, :, 0]])
    g_dense = np.linalg.lstsq(A, y, rcond=None)[0]
    psi_dense = np.zeros(nlay)
    g_full = np.concatenate([[0.0], g_dense, [0.0]])
    for k in range(nlay):
        psi_dense[k] = 0.5 * (g_full[k] + g_full[k + 1])

    pT, _ = ae.oracle_psi_constrained(dzT, dzS, F_T, F_S, w, nwet, ridge=0.0)
    psi_got = pT[0, :, 0] / dzT[0]
    assert np.allclose(psi_got, psi_dense, atol=1e-8)


# ---------------------------------------------------------------------------
# stratum edges
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("depth,band", [
    (50.0, "0-100m"), (100.0, "100-500m"), (99.999, "0-100m"), (250.0, "100-500m"),
    (1500.0, "1500-3000m"), (2999.0, "1500-3000m"), (3000.0, ">3000m"), (5500.0, ">3000m"),
])
def test_depth_bands_land_where_their_name_says(depth, band):
    assert ae.DEPTH_NAMES[int(np.digitize(depth, ae.DEPTH_EDGES))] == band


@pytest.mark.parametrize("lat,band", [
    (-75.0, "S of 60S"), (-60.0, "60S-30S"), (0.0, "30S-30N"), (30.0, "30N-60N"),
    (61.0, "N of 60N"),
])
def test_latitude_bands_land_where_their_name_says(lat, band):
    assert ae.LAT_NAMES[int(np.digitize(lat, ae.LAT_EDGES))] == band


@pytest.mark.parametrize("ratio,band", [
    (0.4, "d/Lr<1"), (1.0, "1-3"), (2.9, "1-3"), (3.0, "3-10"), (25.0, ">10"),
])
def test_resolution_bands_land_where_their_name_says(ratio, band):
    assert ae.RES_NAMES[int(np.digitize(ratio, ae.RES_EDGES))] == band


def test_band_names_and_edges_stay_in_step():
    """Every ``digitize`` output must index a name — one edge added without a name would put
    cells in a band that silently is not printed and therefore never gates."""
    for edges, names in ((ae.DEPTH_EDGES, ae.DEPTH_NAMES), (ae.LAT_EDGES, ae.LAT_NAMES),
                         (ae.RES_EDGES, ae.RES_NAMES)):
        assert len(names) == len(edges) + 1
        assert list(edges) == sorted(edges)


# ---------------------------------------------------------------------------
# the rotated grid
# ---------------------------------------------------------------------------
def test_latitude_bands_use_the_geographic_pole_not_the_rotated_one():
    """FESOM computes in a rotated frame whose pole is at (50E, 15N), and ``mesh.coord_nod2D``
    is that rotated coordinate. It is the attribute that comes to hand, it has the right shape
    and the right units, and using it for a band labelled "30N-60N" is wrong by up to 15 degrees
    — measured on core2, 22.8 % of nodes land in a different band. Nothing fails; the table just
    describes a different ocean than its headings claim.

    Also pinned: the Coriolis parameter behind the resolution stratification must be the mesh's
    own ``coriolis_node`` (already 2*Omega*sin(geo lat)) rather than a recomputation from the
    rotated latitude, which is off by more than 2x at 17.4 % of nodes."""
    import types
    import nn_data as nd
    rot = np.radians(np.array([10.0, 20.0, 30.0]))
    geo = np.radians(np.array([40.0, 50.0, 70.0]))
    mesh = types.SimpleNamespace(coord_nod2D=np.stack([np.zeros(3), rot], axis=-1),
                                 geo_coord_nod2D=np.stack([np.zeros(3), geo], axis=-1))
    got = nd.geo_latitude_deg(mesh)
    assert got == pytest.approx([40.0, 50.0, 70.0])
    assert np.abs(got - np.degrees(rot)).min() > 1.0, "the fixture must actually differ"


def test_strata_reads_latitude_and_coriolis_from_the_geographic_fields():
    """Guards the call sites rather than the helper: `strata` must not reach for
    ``coord_nod2D`` or rebuild f from it."""
    import inspect
    src = inspect.getsource(ae.strata)
    assert "coord_nod2D" not in src.replace("geo_coord_nod2D", ""), \
        "strata must label regions with the geographic coordinate"
    assert "coriolis_node" in src, "the resolution band must use the mesh's own Coriolis"
