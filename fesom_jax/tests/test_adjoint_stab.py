"""Tests for the backward-pass stabilizers (NN-GM Task 2 follow-on).

Runs on the packaged `pi` mesh — small, real FESOM geometry, CPU-fast; no model steps.

What these check is the property that makes each stabilizer trustworthy rather than merely
plausible. A stabilizer is inserted into a backward pass whose output nobody can eyeball,
so the failure mode is silent: an operator that quietly rescaled the gradient, or that
applied the filter instead of its transpose, would look exactly like a stabilizer that
worked. Every claim in the module docstring gets a test here.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import adjoint_stab, coarsen
from fesom_jax.mesh import load_mesh

PI_MESH_DIR = Path(__file__).resolve().parents[1] / "data" / "mesh_pi"


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(PI_MESH_DIR)


# ---------------------------------------------------------------------------
# the primitive

def test_cotangent_op_is_the_identity_in_the_forward():
    """The whole design rests on this: inserting a stabilizer must not move the forward."""
    f = adjoint_stab.make_discounter(0.5)
    x = jnp.array([1.0, -2.0, 3.5])
    out = adjoint_stab.cotangent_op(f, x)
    assert np.array_equal(np.asarray(out), np.asarray(x))


def test_cotangent_op_applies_the_map_in_the_backward():
    f = adjoint_stab.make_discounter(0.25)

    def loss(x):
        return jnp.sum(adjoint_stab.cotangent_op(f, x) ** 2)

    x = jnp.array([1.0, 2.0, 3.0])
    g = jax.grad(loss)(x)
    assert np.allclose(np.asarray(g), 0.25 * 2.0 * np.asarray(x))


def test_discount_compounds_over_a_scan():
    """gamma applied once per step must weight a step k back by gamma**k — the property the
    whole 'discounted adjoint' interpretation depends on."""
    gamma, nstep = 0.5, 6
    f = adjoint_stab.make_discounter(gamma)

    def loss(x):
        def body(c, _):
            return adjoint_stab.cotangent_op(f, c), None
        out, _ = jax.lax.scan(body, x, xs=None, length=nstep)
        return jnp.sum(out)

    g = float(jax.grad(loss)(jnp.array(1.0)))
    assert np.isclose(g, gamma ** nstep)


def test_freeze_increment_keeps_the_value_but_not_the_derivative():
    """freeze_increment must be exact in the forward and see only `base` in the backward."""
    def base_fn(x):
        return 2.0 * x

    def full_fn(x):
        return 2.0 * x + 10.0 * x ** 2          # the "stage" adds a stiff term

    def f(x):
        return adjoint_stab.freeze_increment(base_fn(x), full_fn(x))

    x = jnp.array(3.0)
    assert np.isclose(float(f(x)), float(full_fn(x)))        # value: the full stage
    assert np.isclose(float(jax.grad(f)(x)), 2.0)            # slope: base only, not 2+20x


def test_clipper_caps_the_norm_and_keeps_the_direction():
    cap = 1.0
    f = adjoint_stab.make_clipper(cap)
    big = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([0.0])}
    out = f(big)
    nrm = float(adjoint_stab.cotangent_norm(out))
    assert np.isclose(nrm, cap, rtol=1e-12)
    # direction preserved
    assert np.allclose(np.asarray(out["a"]), np.asarray(big["a"]) / 5.0)
    # below the cap it is the identity
    small = {"a": jnp.array([0.3, 0.4]), "b": jnp.array([0.0])}
    assert np.allclose(np.asarray(f(small)["a"]), np.asarray(small["a"]))


def test_cotangent_norm_skips_float0():
    """jax hands back float0 cotangents for integer leaves; arithmetic on them raises, so the
    norm and every scalar map must step over them rather than blow up."""
    tree = {"x": jnp.array([3.0, 4.0]),
            "i": np.zeros((2,), dtype=jax.dtypes.float0)}
    assert np.isclose(float(adjoint_stab.cotangent_norm(tree)), 5.0)
    out = adjoint_stab.make_discounter(0.5)(tree)
    assert np.allclose(np.asarray(out["x"]), [1.5, 2.0])
    assert out["i"].dtype == jax.dtypes.float0


# ---------------------------------------------------------------------------
# the adjoint diffusion

def test_smoother_is_the_exact_transpose_of_the_filter(mesh):
    """<F x, y> == <x, F^T y>. This is the test that catches applying the filter itself to
    the cotangent: on a mesh whose cell areas span orders of magnitude, F and F^T differ by
    the area metric, and the mistake is invisible in any single field."""
    factor = 2.0
    apply_T, (n_steps, kappa_dt) = adjoint_stab.make_nodal_smoother(mesh, factor)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(mesh.nod2D,)))
    y = jnp.asarray(rng.normal(size=(mesh.nod2D,)))
    mask = jnp.asarray(mesh.node_layer_mask, jnp.float64)[:, 0]
    Fx = coarsen.filter_field(mesh, x, n_steps=n_steps, kappa_dt=kappa_dt,
                              node_mask=mask, check_stability=False)
    FTy = apply_T(y)
    lhs, rhs = float(jnp.sum(Fx * y)), float(jnp.sum(x * FTy))
    assert np.isclose(lhs, rhs, rtol=1e-9), f"<Fx,y>={lhs} != <x,F^T y>={rhs}"


def test_smoother_conserves_total_sensitivity(mesh):
    """F preserves constants, so F^T preserves the SUM of the cotangent. A stabilizer that
    could rescale the gradient total would be indistinguishable from a bug."""
    apply_T, _ = adjoint_stab.make_nodal_smoother(mesh, 2.0)
    rng = np.random.default_rng(1)
    g = jnp.asarray(rng.normal(size=(mesh.nod2D,)))
    out = apply_T(g)
    assert np.isclose(float(jnp.sum(out)), float(jnp.sum(g)), rtol=1e-9)


def test_smoother_actually_smooths(mesh):
    """Grid-scale structure must come down. Uses the roughness measure the filter targets:
    the norm of the discrete Laplacian."""
    apply_T, _ = adjoint_stab.make_nodal_smoother(mesh, 3.0)
    rng = np.random.default_rng(2)
    g = jnp.asarray(rng.normal(size=(mesh.nod2D,)))
    rough = lambda v: float(jnp.linalg.norm(coarsen.laplacian(mesh, v)))
    assert rough(apply_T(g)) < 0.5 * rough(g)


def test_smoother_handles_3d_and_leaves_non_nodal_fields_alone(mesh):
    """A State cotangent is a mixed bag: (N,), (N,nl), (N,nl,2) and element fields. The
    smoother must act on the nodal ones and pass the rest through untouched — silently
    smoothing an element field with a nodal operator would be a shape-lucky disaster."""
    apply_T, _ = adjoint_stab.make_nodal_smoother(mesh, 2.0)
    nl = int(mesh.nl)
    rng = np.random.default_rng(3)
    for shape in [(mesh.nod2D,), (mesh.nod2D, nl), (mesh.nod2D, nl, 2)]:
        out = apply_T(jnp.asarray(rng.normal(size=shape)))
        assert out.shape == shape
    elem = jnp.asarray(rng.normal(size=(mesh.elem2D, nl)))
    assert np.array_equal(np.asarray(apply_T(elem)), np.asarray(elem))
    assert float(apply_T(jnp.asarray(2.0))) == 2.0                 # scalar passes through


# ---------------------------------------------------------------------------
# assembly

def test_stabilizer_is_none_when_nothing_is_asked_for(mesh):
    """Off must mean absent, not an identity wrapper — the traced graph of an unstabilized
    run has to stay exactly what it was."""
    assert adjoint_stab.make_state_stabilizer(mesh, adjoint_stab.StabConfig()) is None
    assert not adjoint_stab.StabConfig().active
    assert adjoint_stab.StabConfig().describe() == "none"


def test_stabilizer_composes_and_leaves_the_forward_alone(mesh):
    cfg = adjoint_stab.StabConfig(gamma=0.5, filter_factor=2.0)
    stab = adjoint_stab.make_state_stabilizer(mesh, cfg)
    assert stab is not None and "gamma=0.5" in cfg.describe() and "filter=2dx" in cfg.describe()
    tree = {"nodal": jnp.ones((mesh.nod2D,)), "elem": jnp.ones((mesh.elem2D,))}
    out = stab(tree)
    for k in tree:
        assert np.array_equal(np.asarray(out[k]), np.asarray(tree[k])), "forward must be identity"

    def loss(t):
        return jnp.sum(stab(t)["nodal"]) + jnp.sum(stab(t)["elem"])

    g = jax.grad(loss)(tree)
    # element field: discount only. nodal: discount * F^T, whose sum is conserved.
    assert np.allclose(np.asarray(g["elem"]), 0.5)
    assert np.isclose(float(jnp.sum(g["nodal"])), 0.5 * mesh.nod2D, rtol=1e-9)


# ---------------------------------------------------------------------------
# the amplification screen (M6's per-window reject rule)
#
# The two populations below are the MEASURED ones, with the network entirely absent in both,
# so they are properties of the seed state. If a change to the screen makes either of these
# come out the other way, the change is wrong, not these numbers.

def _trace(rate_per_step, n_chunks, k=48, start=1.0):
    """Synthetic reverse-pass boundary-norm trace with a constant per-step growth rate."""
    tr = [start]
    for _ in range(n_chunks):
        tr.append(tr[-1] * rate_per_step ** k)
    return tr


def test_amplification_accepts_the_foundation_seed_rate():
    """1.0008/step over 30 daily chunks — the clean 30- and 60-day records."""
    amp = adjoint_stab.amplification(_trace(1.0008, 30), 48)
    assert amp.clean and amp.reason == ""
    assert np.isclose(amp.median, 1.0008, rtol=1e-6)
    assert amp.log_spread < 1e-9


def test_amplification_rejects_the_student_seed_rate():
    """1.0539/step over 5 daily chunks — the seed that failed the M1 gate for a day."""
    amp = adjoint_stab.amplification(_trace(1.0539, 5), 48)
    assert not amp.clean
    assert "median" in amp.reason and "1.0539" in amp.reason
    assert np.isclose(amp.median, 1.0539, rtol=1e-6)


def test_amplification_rejects_a_switch_firing_at_a_normal_median():
    """A near-flat trace with one chunk multiplying the norm by 8371x — the seam gate's own
    chunk-0 signature. The MEDIAN rate stays innocuous, so a median-only screen passes it."""
    tr = [1.0, 1.01, 1.02, 1.03, 1.04]
    tr.append(tr[-1] * 8371.0)
    amp = adjoint_stab.amplification(tr, 48)
    assert amp.median <= adjoint_stab.AMP_CLEAN_MAX, "the median alone does not see this"
    assert not amp.clean and "log-spread" in amp.reason
    assert amp.worst_chunk == 0, "the growth is in the chunk nearest the seed state"


def _february_trace():
    """The MEASURED shape of job 26819415: a 60-day window ending 1961-02-14, seeded
    1960-12-16. Eight chunks of moderate growth, ONE chunk multiplying the norm by 1406 on
    1961-02-06, then a plateau of ~1.0 ratios for the rest of the sweep. Its finite difference
    puts the true gradient at -9.80e-01; the adjoint returned +5.69e+06."""
    tr = [1.1964e-01, 9.6264e-01, 1.3803e+01, 2.2866e+01, 2.2766e+02,
          3.1132e+02, 3.1286e+02, 4.4013e+05, 5.2202e+06, 7.8367e+07]
    # the plateau: fifty chunks that neither grow nor decay
    for i in range(50):
        tr.append(tr[-1] * (1.02 if i % 2 else 0.98))
    return tr


def test_amplification_rejects_the_february_window_the_median_lets_through():
    """The case the third bar was added for. Over the FULL trace the median is innocuous and
    the log-spread only just fires, because fifty flat chunks outvote the one that exploded."""
    amp = adjoint_stab.amplification(_february_trace(), 48)
    assert amp.median <= adjoint_stab.AMP_CLEAN_MAX, (
        f"median {amp.median:.5f} — a median-only screen passes this window outright")
    assert amp.sustained > adjoint_stab.AMP_SUSTAINED_MAX, (
        f"worst 3-chunk stretch {amp.sustained:.5f} must exceed "
        f"{adjoint_stab.AMP_SUSTAINED_MAX}")
    assert not amp.clean and "stretch" in amp.reason


# The counterexample that killed the first design. A single-chunk bar looked obvious and would
# have been unsafe: this trace is clean by every independent check yet contains one chunk at
# x6.02 — because the very next chunk is x0.184 and the norm comes straight back.
_SEAM_30D_CLEAN = [
    0.00679151, 0.00813544, 0.00844541, 0.00859426, 0.00879829, 0.00933287, 0.00991297,
    0.0104272, 0.0110983, 0.011754, 0.0123324, 0.0128601, 0.0134079, 0.0140431, 0.014758,
    0.015685, 0.0944593, 0.0173923, 0.0179669, 0.0185726, 0.0184976, 0.0179418, 0.0182649,
    0.0189935, 0.0197747, 0.0205579, 0.0207511, 0.0213167, 0.0222801, 0.0234785, 0.024167,
]


def test_a_transient_spike_that_reverses_is_not_a_blow_up():
    """30-day seam-gate trace, FD-confirmed to 7.9 % with the gradient reproducing to 6.5e-4 —
    clean by everything that does not look at the trace. Its worst SINGLE chunk is 1.03811,
    which is why a single-chunk bar has no usable gap against the blown population (least
    blown 1.07108). Over three chunks the spike cancels against its own recovery and the rate
    falls to 1.01332, which is where the gap is."""
    amp = adjoint_stab.amplification(_SEAM_30D_CLEAN, 48)
    assert np.isclose(amp.max, 1.03811, atol=1e-5), f"single-chunk max moved: {amp.max:.5f}"
    assert np.isclose(amp.sustained, 1.01332, atol=1e-5), (
        f"3-chunk rate moved: {amp.sustained:.5f}")
    assert amp.sustained < adjoint_stab.AMP_SUSTAINED_MAX, (
        "an FD-confirmed run must not be rejected — if this fires, the bar is too tight")
    assert amp.clean, f"this window is clean; screen said: {amp.reason}"


def test_the_worst_chunk_bar_is_the_one_that_does_not_dilute():
    """Why a third statistic was needed, measured rather than argued: hold ONE x1200 event
    fixed and lengthen the quiet plateau behind it.

        flat chunks      median    log-spread       max     old screen
              2        1.000413      0.0696       1.15918     rejects
             50        1.000413      0.0205       1.15918     rejects (by 1.02x)
            100        1.000413      0.0146       1.15918     PASSES
            400        1.000413      0.0074       1.15918     PASSES

    The median never sees it at all — dilution is immediate, not gradual, because the plateau
    outnumbers the event from the second chunk on. The log-spread decays monotonically and
    crosses its bar somewhere between 50 and 100 quiet chunks, i.e. the old screen stopped
    catching a single-event blow-up on windows longer than about 100 days. The campaign's
    target is a YEAR. The worst-chunk rate does not move at all.
    """
    def trace_with_plateau(n_flat):
        tr = [1.0, 1.0 * 1200.0]
        for i in range(n_flat):
            tr.append(tr[-1] * (1.02 if i % 2 else 0.98))
        return tr

    seen = {}
    for n_flat in (2, 50, 100, 400):
        amp = adjoint_stab.amplification(trace_with_plateau(n_flat), 48)
        seen[n_flat] = amp
        assert not amp.clean, "a x1200 single-chunk event must reject at any plateau length"
        assert amp.median <= adjoint_stab.AMP_CLEAN_MAX, (
            "the median never fires on a single-chunk event — if it does here, the "
            "construction has drifted and this test no longer demonstrates anything")

    assert seen[2].log_spread > seen[50].log_spread > seen[100].log_spread > seen[400].log_spread, (
        "the log-spread must decay with plateau length — that decay is the whole problem")
    assert seen[100].log_spread <= adjoint_stab.AMP_SPREAD_MAX < seen[50].log_spread, (
        f"the old two-bar screen must PASS at 100 flat chunks ({seen[100].log_spread:.4f}) and "
        f"reject at 50 ({seen[50].log_spread:.4f}); if that crossing moved, requote the table")
    assert abs(seen[2].sustained - seen[400].sustained) < 1e-12, "the sustained rate must not dilute"
    assert "stretch" in seen[400].reason, (
        "at 400 flat chunks the worst-chunk bar must be the only thing rejecting this window")


def test_the_sustained_bar_sits_in_a_measured_empty_gap():
    """Both populations, as measured over every trace on record. The bar must clear the whole
    clean side and reject the whole blown side; if a future run lands between them, this test
    is where that shows up rather than in a training curve."""
    clean_3chunk = (1.00123, 1.00137, 1.00168, 1.00240, 1.00284, 1.00296, 1.00340, 1.01332)
    blown_3chunk = (1.07108, 1.07642, 1.09016, 1.17174, 1.23807, 1.40121)
    bar = adjoint_stab.AMP_SUSTAINED_MAX
    assert max(clean_3chunk) < bar < min(blown_3chunk), (
        f"bar {bar} no longer separates clean (worst {max(clean_3chunk)}) from blown "
        f"(least {min(blown_3chunk)})")
    # and it is not hugging either side: geometric middle of the gap, in excess over 1
    assert 1.8 < (bar - 1.0) / (max(clean_3chunk) - 1.0) , "too close to the clean population"
    assert 1.8 < (min(blown_3chunk) - 1.0) / (bar - 1.0), "too close to the blown population"


def test_sustained_falls_back_to_a_single_chunk_on_a_short_trace():
    """A 2-boundary trace cannot hold a 3-chunk window. It must degenerate to the single-chunk
    rate, not return NaN and quietly pass."""
    amp = adjoint_stab.amplification([1.0, 1.0 * 1200.0], 48)
    assert np.isfinite(amp.sustained) and np.isclose(amp.sustained, amp.max)
    assert not amp.clean


def test_amplification_reports_worst_chunk_in_forward_time():
    """`trace` runs backwards from the loss mark; `worst_chunk` must come back forwards."""
    tr = [1.0] * 6
    tr[4] = 1.0 * 50.0          # growth across the step from index 3 to 4
    tr[5] = tr[4]
    amp = adjoint_stab.amplification(tr, 48)
    assert amp.worst_chunk == (5 - 1) - 3


def test_amplification_refuses_to_pass_a_degenerate_trace():
    """Silence is not success: too short, or an all-zero cotangent, must REJECT."""
    for bad in ([], [1.0], [0.0, 0.0]):
        amp = adjoint_stab.amplification(bad, 48)
        assert not amp.clean and amp.reason
