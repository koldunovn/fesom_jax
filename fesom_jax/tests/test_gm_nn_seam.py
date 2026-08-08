"""NN-GM Task 7 gate — the ``gm_nn`` seam inside the GM/Redi coefficient block.

The network itself is gated by :mod:`test_gm_nn` (bounds, ``zero_last``, treedef). This module
gates the **wiring**: that a trained closure reaches the model, that an untrained one cannot
perturb it, and that the deployment path survives a file and a restart.

Consumption-site audit (the reason ONE multiplication suffices)
--------------------------------------------------------------
``grep -rn 'fer_K\\|\\bKi\\b' --include='*.py'`` over the library, excluding docstrings:

* ``fer_K`` is built in ``gm.init_redi_gm`` and read by exactly one consumer,
  ``gm.fer_solve_gamma`` (the Γ streamfunction RHS), called from ``gm.gm_diagnostics``
  immediately below the seam. It is also returned in ``GMDiag`` as a diagnostic.
* ``Ki`` is built in ``gm.init_redi_gm`` and read by ``gm_redi.diff_ver_part_redi_expl``,
  ``gm_redi.diff_part_hor_redi`` and ``gm_redi.k33_augmentation`` — all three called from
  ``step.py`` (~467-477), and all three receive it as ``gm_diag.Ki``, i.e. from the ``GMDiag``
  the seam has already multiplied. ``integrate_sharded.run_gm_diag_sharded`` returns the same
  three fields.

So multiplying inside ``gm_diagnostics`` covers every consumer, and doing it BEFORE the halo
exchanges (``gm.py`` ~385) is required rather than merely convenient: the features derive from
``sigma_xy``, an element→node scatter that is incomplete on halo nodes, so a multiplier applied
after the exchange would be wrong precisely on the halo. That claim is what
``test_gm_nn_sharded.py`` checks; it cannot be seen on one device.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import gm, gm_nn, ops
from fesom_jax.io_dump import load_gm_dump
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
GM_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "gm_dump_core2"

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir() and (GM_DUMP_DIR / "gm_meta.txt").is_file()),
    reason=f"CORE2 mesh / GM dump missing ({GM_DUMP_DIR}); run jax_gm_dump_core2.sh",
)

KEY = jax.random.PRNGKey(7)
M_MAX = 3.0            # the default bound: m ∈ [1/3, 3]


@pytest.fixture(scope="module")
def setup():
    """Mesh + the C GM dump's step-1 ocean state (a real stratified CORE2 field)."""
    mesh = load_mesh(CORE2_MESH_DIR)
    f, _ = load_gm_dump(GM_DUMP_DIR)
    cfg = gm.GMConfig()
    helem = ops.gather_nodes_to_elem(jnp.asarray(f["hnode"]), mesh.elem_nodes).mean(axis=1)
    helem = jnp.where(mesh.elem_layer_mask, helem, 0.0)
    args = (jnp.asarray(f["T"]), jnp.asarray(f["S"]), jnp.asarray(f["bvfreq"]),
            jnp.asarray(f["hnode_new"]), helem)
    return mesh, cfg, args


def _diag(mesh, cfg, args, nn):
    """``gm_diagnostics`` with ``params.gm_nn = nn`` (``None`` ⇒ the default-GM path)."""
    p = dataclasses.replace(Params.defaults(), gm_nn=nn)
    return gm.gm_diagnostics(mesh, *args[:4], args[4], p, cfg)


# ---------------------------------------------------------------------------
# 1. The two bit-identity invariants — DIFFERENT claims, both required
# ---------------------------------------------------------------------------
def test_default_params_have_no_gm_nn():
    """``Params.defaults().gm_nn is None`` — the seam is OFF unless a run asks for it, so every
    existing config keeps the default closure without being edited."""
    assert Params.defaults().gm_nn is None
    assert dataclasses.replace(Params.defaults(), gm_nn=None).gm_nn is None


def test_gm_nn_none_leaves_the_trace_untouched(setup):
    """Invariant (1): ``gm_nn is None`` is a **Python-level** branch, so the default path has no
    trace change at all — not "multiplied by one", but absent from the graph.

    Asserted structurally, by equation count: the ``None`` jaxpr must be strictly SHORTER than
    the NN jaxpr. A ``where(nn is None, 1.0, multiplier(...))`` implementation would produce
    byte-identical output (so the next test would still pass) while adding the MLP and its
    feature tensor to every existing run's graph and backward. This is the test that would
    catch that, and the ``step.py`` ``gm_cfg is None`` precedent is the pattern being enforced.
    """
    mesh, cfg, args = setup
    nn = gm_nn.init_gm_nn(KEY, zero_last=False)

    def f(nn_arg):
        return jax.make_jaxpr(lambda: _diag(mesh, cfg, args, nn_arg))()

    n_none = len(f(None).jaxpr.eqns)
    n_nn = len(f(nn).jaxpr.eqns)
    print(f"\njaxpr eqns: gm_nn=None {n_none}, gm_nn=MLP {n_nn} (+{n_nn - n_none})")
    assert n_none < n_nn, (
        f"the gm_nn=None path has as many equations as the NN path ({n_none} vs {n_nn}) — the "
        f"seam is being traced when it should be skipped by a Python branch")


def test_zero_last_nn_reproduces_the_default_path(setup):
    """Invariant (2): a ``zero_last`` NN gives multipliers of exactly 1.0, so the GM fields are
    bit-identical to the ``None`` path even though the MLP IS in the graph.

    Weaker than invariant (1) — it says the arithmetic is a no-op, not that the graph is
    untouched — and it is the property that makes an untrained network safe to deploy and makes
    "the seam is wired in" undetectable by any regression test until training starts."""
    mesh, cfg, args = setup
    base = _diag(mesh, cfg, args, None)
    zero = _diag(mesh, cfg, args, gm_nn.init_gm_nn(KEY, zero_last=True))
    for name in ("fer_K", "Ki", "slope_tapered", "fer_uv", "fer_C"):
        a, b = np.asarray(getattr(base, name)), np.asarray(getattr(zero, name))
        d = float(np.max(np.abs(a - b)))
        print(f"  {name:14s} max|Δ| = {d:.3e}")
        assert d == 0.0, f"{name}: zero_last NN perturbed the GM fields by {d:.3e}"


# ---------------------------------------------------------------------------
# 2. The wiring is real, and it is bounded
# ---------------------------------------------------------------------------
def test_trained_nn_scales_both_coefficients_within_bounds(setup):
    """A non-trivial NN must actually move ``fer_K`` AND ``Ki``, by a spatially varying factor
    inside ``[1/m_max, m_max]``.

    The complement of the two bit-identity tests: those prove the seam cannot break the default
    closure, this proves it is connected at all. ``Ki`` is checked exactly (``Ki_nn/Ki_base``
    IS ``m_Redi``, since the multiplication is the last thing that happens to it), so a
    bound violation here is a bound violation in the network."""
    mesh, cfg, args = setup
    base = _diag(mesh, cfg, args, None)
    got = _diag(mesh, cfg, args, gm_nn.init_gm_nn(KEY, zero_last=False))
    for name, iface in (("fer_K", True), ("Ki", False)):
        a, b = np.asarray(getattr(base, name)), np.asarray(getattr(got, name))
        wet = a > 0.0
        ratio = b[wet] / a[wet]
        lo, hi = float(ratio.min()), float(ratio.max())
        spread = float(np.std(np.log(ratio)))
        print(f"  {name:6s} ratio ∈ [{lo:.4f}, {hi:.4f}]  std(log m)={spread:.4f}")
        assert lo >= 1.0 / M_MAX - 1e-12 and hi <= M_MAX + 1e-12, \
            f"{name}: multiplier escaped [1/{M_MAX}, {M_MAX}] — got [{lo}, {hi}]"
        assert spread > 1e-3, f"{name}: multiplier is spatially constant (std log m = {spread})"
        # masked lanes stay exactly zero: a multiplier can only scale, never populate.
        assert np.all(b[~wet] == a[~wet]), f"{name}: the seam wrote into masked lanes"


@pytest.mark.parametrize("case", ["zero_n2", "huge_n2", "sharp_front"])
def test_bounds_hold_under_pathological_inputs(setup, case):
    """The seam stays finite and bounded on states the ocean should never reach.

    ``zero_n2`` (no stratification ⇒ the neutral-slope denominator hits its ``slope_eps²``
    floor and the ODM95 taper is forced to zero), ``huge_n2``, and a front with a 20 °C / 5 psu
    step. The bound is structural — ``exp(log_m_max·tanh(·))`` saturates rather than clipping —
    so what this really tests is that no feature reaches the MLP as a NaN or an infinity, which
    a bad masked divide would do and which would poison the whole reverse pass."""
    mesh, cfg, (T, S, bv, hn, helem) = setup
    if case == "zero_n2":
        bv = jnp.zeros_like(bv)
    elif case == "huge_n2":
        bv = jnp.full_like(bv, 1e3)
    else:
        lat_like = jnp.arange(mesh.nod2D)[:, None] % 2 == 0
        T = jnp.where(lat_like, T + 20.0, T - 20.0)
        S = jnp.where(lat_like, S + 5.0, jnp.maximum(S - 5.0, 0.0))
    args = (T, S, bv, hn, helem)
    base = _diag(mesh, cfg, args, None)
    got = _diag(mesh, cfg, args, gm_nn.init_gm_nn(KEY, zero_last=False))
    for name in ("fer_K", "Ki", "fer_uv"):
        b = np.asarray(getattr(got, name))
        assert np.all(np.isfinite(b)), f"{case}/{name}: {int((~np.isfinite(b)).sum())} non-finite"
    a, b = np.asarray(base.Ki), np.asarray(got.Ki)
    wet = a > 0.0
    if wet.any():
        ratio = b[wet] / a[wet]
        print(f"\n{case}: Ki ratio ∈ [{ratio.min():.4f}, {ratio.max():.4f}]")
        assert ratio.min() >= 1.0 / M_MAX - 1e-12 and ratio.max() <= M_MAX + 1e-12


# ---------------------------------------------------------------------------
# 3. The gradient — an FD agreement check, not a finiteness check
# ---------------------------------------------------------------------------
def _scaled(nn, alpha):
    """``nn`` with its LAST layer uniformly scaled by ``alpha`` — the scalar probe direction."""
    return gm_nn.GmNN(Ws=nn.Ws[:-1] + (nn.Ws[-1] * alpha,),
                      bs=nn.bs[:-1] + (nn.bs[-1] * alpha,),
                      log_m_max=nn.log_m_max)


def test_fd_agreement_on_the_scalar_probe(setup):
    """d(Σ fer_uv²)/dα for a uniform scaling α of the last layer, AD vs central differences.

    "Finite and nonzero" is NOT the check here: this project has a recorded trap where a
    gradient was finite, nonzero and structurally wrong (a dead path), so the standard is
    quantitative agreement with a difference quotient. One GM block only — the plan's 5-day
    window FD check at the same ≤10% bar is a GPU job, not a suite test."""
    mesh, cfg, args = setup
    nn = gm_nn.init_gm_nn(KEY, zero_last=False)

    def loss(alpha):
        return jnp.sum(_diag(mesh, cfg, args, _scaled(nn, alpha)).fer_uv ** 2)

    g_ad = float(jax.grad(loss)(jnp.asarray(1.0, jnp.float64)))
    h = 1e-5
    g_fd = float((loss(jnp.asarray(1.0 + h)) - loss(jnp.asarray(1.0 - h))) / (2 * h))
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-300)
    print(f"\nFD probe: AD {g_ad:.8e}  FD {g_fd:.8e}  rel {100 * rel:.4f}%")
    assert g_ad != 0.0, "the probe direction has no effect on the loss — the seam is dead"
    assert rel < 1e-2, f"AD/FD disagree by {100 * rel:.3f}% (bar 1%)"


def test_gradient_reaches_the_live_feature_rows(setup):
    """d(loss)/d(weights) is finite for every array, and every LIVE feature row of the first
    layer carries gradient — while the strain row carries exactly none.

    That asymmetry is the deferred feature made visible: the seam passes ``strain = 0.0``, so
    ``W0[6]`` is multiplied by zero forever and can never learn. The row is kept so a weight
    file trained now still loads if strain arrives at the D2 screen; this test is what stops
    that trade-off from being silently forgotten (and what would fail loudly if strain is later
    threaded but the seam is not updated to pass it)."""
    mesh, cfg, args = setup
    nn = gm_nn.init_gm_nn(KEY, zero_last=False)

    def loss(n):
        return jnp.sum(_diag(mesh, cfg, args, n).fer_uv ** 2)

    g = jax.grad(loss)(nn)
    for i, (W, b) in enumerate(zip(g.Ws, g.bs)):
        assert np.all(np.isfinite(np.asarray(W))), f"W{i} gradient has non-finite entries"
        assert np.all(np.isfinite(np.asarray(b))), f"b{i} gradient has non-finite entries"
        assert np.max(np.abs(np.asarray(W))) > 0.0, f"W{i} gradient identically zero"
    row_norms = np.linalg.norm(np.asarray(g.Ws[0]), axis=1)
    print("\nW0 row |grad| per feature: " + " ".join(f"{v:.2e}" for v in row_norms))
    live, strain_row = row_norms[:6], row_norms[6]
    assert np.all(live > 0.0), f"a live feature row has zero gradient: {row_norms}"
    assert strain_row == 0.0, (
        f"the strain row has gradient {strain_row:.3e} — the seam is no longer passing "
        f"strain = 0.0, so update this test and gm_nn.column_features together")


# ---------------------------------------------------------------------------
# 4. The deployment path: weight file → config → run, across a restart leg
# ---------------------------------------------------------------------------
def test_weight_file_round_trip_is_exact_and_keeps_the_treedef(tmp_path):
    """``save_gm_nn``/``load_gm_nn`` round-trip with an IDENTICAL treedef and exact values.

    The treedef is the real assertion: ``log_m_max`` is a ``meta`` field (so the optimiser
    cannot train the safety bound away), which means it is not one of the saved arrays by
    default and a weights-only file would silently rebuild at the class default. A network
    trained under ``m_max=2.5`` and reloaded at 3.0 is a different closure wearing the same
    weights, and nothing downstream would notice."""
    nn = gm_nn.init_gm_nn(KEY, hidden=(8, 8), m_max=2.5, zero_last=False)
    path = tmp_path / "gm_nn_stage0.npz"
    gm_nn.save_gm_nn(path, nn)
    back = gm_nn.load_gm_nn(path)

    assert jax.tree_util.tree_structure(nn) == jax.tree_util.tree_structure(back)
    assert back.log_m_max == nn.log_m_max == float(np.log(2.5))
    for a, b in zip(jax.tree_util.tree_leaves(nn), jax.tree_util.tree_leaves(back)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # the same weights ⇒ the same multipliers, bit for bit
    feats = jax.random.normal(KEY, (64, gm_nn.n_features()), jnp.float64)
    np.testing.assert_array_equal(np.asarray(gm_nn.multiplier(nn, feats)),
                                  np.asarray(gm_nn.multiplier(back, feats)))


def test_weight_file_rejects_inconsistent_shape_metadata(tmp_path):
    """A file whose recorded ``hidden``/``two_scale`` disagree with its weight shapes must raise
    at load, not at the first matmul inside a jitted step (where the error would surface as an
    opaque shape failure a long way from the cause)."""
    nn = gm_nn.init_gm_nn(KEY, hidden=(8, 8))
    path = tmp_path / "bad.npz"
    gm_nn.save_gm_nn(path, nn)
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["hidden"] = np.asarray([8, 16], np.int64)          # lie about the second hidden layer
    np.savez(path, **d)
    with pytest.raises(ValueError, match="hidden sizes"):
        gm_nn.load_gm_nn(path)

    d["hidden"] = np.asarray([8, 8], np.int64)
    d["two_scale"] = np.asarray(True, bool)              # lie about the input width
    np.savez(path, **d)
    with pytest.raises(ValueError, match="two_scale"):
        gm_nn.load_gm_nn(path)


def test_params_file_config_key_survives_yaml_and_reload(tmp_path):
    """``RunConfig.params_file`` → ``load_params()`` → a ``Params`` carrying the trained network,
    surviving a YAML round-trip and reloaded identically on a second leg.

    This is the restart-chain property: each leg of a resubmitting chain re-reads the file, so
    the weights a run used are a property of its config rather than of the shell that launched
    it. Loading twice must give the same closure — otherwise leg 2 of a chain quietly runs a
    different model from leg 1."""
    from fesom_jax.run_config import RunConfig, load_yaml

    nn = gm_nn.init_gm_nn(KEY, hidden=(8, 8), m_max=2.0, zero_last=False)
    wpath = tmp_path / "weights.npz"
    gm_nn.save_gm_nn(wpath, nn)

    cfg = RunConfig(params_file=str(wpath))
    ypath = tmp_path / "run.yaml"
    cfg.to_yaml(ypath)
    cfg2 = load_yaml(ypath)
    assert cfg2.params_file == str(wpath)

    leg1, leg2 = cfg2.load_params(), cfg2.load_params()     # two chain legs
    assert leg1.gm_nn is not None
    assert jax.tree_util.tree_structure(leg1) == jax.tree_util.tree_structure(leg2)
    feats = jax.random.normal(KEY, (32, gm_nn.n_features()), jnp.float64)
    m1 = np.asarray(gm_nn.multiplier(leg1.gm_nn, feats))
    m2 = np.asarray(gm_nn.multiplier(leg2.gm_nn, feats))
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(m1, np.asarray(gm_nn.multiplier(nn, feats)))
    # every other Params leaf keeps its config-constant default
    ref = Params.defaults()
    for f in dataclasses.fields(Params):
        if f.name != "gm_nn":
            np.testing.assert_array_equal(np.asarray(getattr(leg1, f.name) or 0.0),
                                          np.asarray(getattr(ref, f.name) or 0.0))


def test_no_params_file_means_no_params_object():
    """No ``params_file`` ⇒ ``load_params()`` returns ``None``, not ``Params.defaults()``.

    Deliberate: the runners already treat ``params=None`` as the defaults, so returning None
    keeps an unconfigured run on the exact pre-existing code path instead of a new-but-equal
    one — the same reasoning as the Python-level ``gm_nn is None`` branch."""
    from fesom_jax.run_config import RunConfig
    assert RunConfig().load_params() is None


def test_build_params_passes_the_nn_through():
    """``calibrate.build_params`` must route ``gm_nn`` through ``_NN_LEAVES`` untouched — a
    ``jnp.asarray`` on a dataclass of weight tuples raises, so a missing set entry is an
    immediate failure rather than a subtle one."""
    from fesom_jax.calibrate import _NN_LEAVES, build_params
    assert {"tke_nn", "gm_nn"} <= _NN_LEAVES
    nn = gm_nn.init_gm_nn(KEY)
    p = build_params({"gm_nn": nn, "k_gm": 1500.0})
    assert p.gm_nn is nn
    assert float(p.k_gm) == 1500.0
