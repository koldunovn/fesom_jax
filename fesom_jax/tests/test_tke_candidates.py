"""Candidate derivative-only settings of the classical-TKE adjoint (``AdjointConfig``
``tke_pr_cut``/``tke_pr_mode``/``tke_pr_ratio``/``tke_pr_raw_cut``/``tke_pr_ice_cut``, ``tke_kme_tke``/
``tke_kme_scope``/``tke_kme_term``, ``tke_ml_cut``/``tke_ml_ratio``/``tke_ml_raw_cut``,
``freeze_mixing_state``) on the column core (hand-built
columns of ``test_tke_cuts.py``; ``ci`` group). ``freeze_mixing_state`` on the assembled
``mixing_tke`` (pi mesh) is in ``test_tke_candidates_pi.py`` (``unit``), the CORE2 step in
``test_tke_candidates_core2.py``.

Properties:

* off (``None`` / ``False``) ⇒ the identical program (the jaxpr hashes pinned before the cut
  points existed) and ``AdjointConfig.tke_options()`` is ``None``;
* on ⇒ the forward bitwise (column core with and without diagnostics, ``mixing_tke``);
* each setting changes the VJP on a column where it is live;
* limits: ``tke_pr_cut = -inf`` ⇒ exact; ``+inf`` ⇒ the ``tke.pr`` diagnostic cut bitwise; the
  mask acts point-wise on the Prandtl number (KappaH tangent = the exact one where N² ≥ N²*, the
  ``tke.pr`` one below); ``"at"`` with N²* below every N² ⇒ exact, with a huge N²* ⇒ the
  ``tke.pr`` cut; ``tke_pr_ratio`` cuts exactly where ``|raw N²| > R*·|N²|``, joined with the
  N²* mask when both are set, ``R* -> inf`` ⇒ exact, ``R* -> 0`` ⇒ the ``tke.pr`` cut, and it
  refuses to run without the raw N²; ``tke_pr_raw_cut`` cuts exactly where the raw N² is below
  N²_raw*, joined with the N²* mask, a threshold below every raw N² ⇒ exact, above every one ⇒
  the ``tke.pr`` cut, and it refuses to run without the raw N²; ``tke_pr_ice_cut`` is the
  ``tke.pr`` cut on the iced columns and exact on the others, exact for an a* no column
  exceeds, and refuses to run without the ice concentration; ``tke_pr_lat_cut`` likewise on the
  columns poleward of phi* (either hemisphere), and refuses to run without the latitude;
  ``tke_pr_tu_cut`` likewise at the interfaces in the diffusive Turner regime, and refuses to
  run without the Turner angle, whose regimes :func:`fesom_jax.eos.turner_angle` gets right; ``tke_kme_tke = 0`` ⇒ exact;
  ``+inf`` (column, km_e) ⇒ the ``tke.km_e`` cut bitwise; unflagged columns keep the exact VJP
  bitwise; ``iface1`` changes the KappaM tangent at interface 1 only; ``ke_diff`` leaves the
  KappaM/KappaH tangents exact; ``tke_ml_cut = -inf`` ⇒ exact, ``+inf`` ⇒ the ``tke.ml_n2``
  cut bitwise, and an N² tangent on cells below N²* propagates as under ``tke.ml_n2``, one on
  cells at or above it as in the exact core; ``tke_ml_ratio`` / ``tke_ml_raw_cut`` do the same
  on the cells where ``|raw N²| > R*·|N²|`` / raw N² < N²_raw*, are exact for a threshold no cell
  passes and refuse to run without the raw N²;
* ``validate()`` rejects malformed settings; ``describe()`` renders them;
* every VJP finite on every column (padding and dry lanes included).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import cvmix_tke, eos
from fesom_jax.adjoint_stab import (FREEZE_FLAGS, TKE_CANDIDATE_FIELDS, AdjointConfig,
                                    TkeAdjoint)
from fesom_jax.tests.test_tke_cuts import (COL_NEUTRAL, COL_RI, COL_STRAT, COL_WALLTOP, KW,
                                           NCOL, NL, PINNED, _inputs, _sha, check_pin)


def _opts(**kw):
    return AdjointConfig(**kw).validate().tke_options()


def _raw(x):
    """The fixture's unsmoothed N² for the ``tke_pr_ratio`` / ``tke_pr_raw_cut`` cuts: equal to
    ``Nsqr`` except at every second interface of the RI column, where it is ``-300·Nsqr`` (a
    locally inverted cell whose smoothed N² is the small positive residual: R = 300)."""
    f = np.ones((NCOL, NL))
    f[COL_RI, 1::2] = -300.0
    return x["Nsqr"] * jnp.asarray(f)


def _ice():
    """The fixture's sea-ice concentration for the ``tke_pr_ice_cut`` cut: 0.9 on the RI
    column (the one whose Prandtl number is live), 0 elsewhere."""
    a = np.zeros(NCOL)
    a[COL_RI] = 0.9
    return jnp.asarray(a)


def _lat():
    """The fixture's latitude (radians) for the ``tke_pr_lat_cut`` cut: 80S on the RI column,
    10N elsewhere."""
    a = np.full(NCOL, math.radians(10.0))
    a[COL_RI] = math.radians(-80.0)
    return jnp.asarray(a)


def _tu():
    """The fixture's interface Turner angle (degrees) for the ``tke_pr_tu_cut`` cut: -60 (the
    diffusive regime) on the RI column, +30 (both components stable) elsewhere."""
    a = np.full((NCOL, NL), 30.0)
    a[COL_RI] = -60.0
    return jnp.asarray(a)


def _core(x, nlev, adj=None, cuts=None, with_diags=False):
    extra = {} if cuts is None else {"cuts": frozenset(cuts)}
    if adj is not None and (adj.pr_ratio is not None or adj.pr_raw_cut is not None
                            or adj.ml_ratio is not None or adj.ml_raw_cut is not None):
        extra["n2_raw"] = _raw(x)
    if adj is not None and adj.pr_ice is not None:
        extra["ice_conc"] = _ice()
    if adj is not None and adj.pr_lat is not None:
        extra["lat"] = _lat()
    if adj is not None and adj.pr_tu is not None:
        extra["turner"] = _tu()
    return cvmix_tke.integrate_tke_column(
        x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
        c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"],
        with_diags=with_diags, adj=adj, **KW, **extra)


# the settings under test (name -> AdjointConfig kwargs); N²* = 1e-5 splits the RI column
# (N² ~ 1e-5 ± 10 %, S² ~ 4e-5: Pr ~ 1.65, inside the clip band, so its derivative is live on
# both sides of the threshold), tke* = 2e-6 flags the WALLTOP column only (tke_old[1] = 1.2e-6;
# every other column has tke_old[1] >= 1e-5)
N2_STAR = 1.0e-5
TKE_STAR = 2.0e-6
R_STAR = 10.0
RAW_STAR = 0.0     # cuts the RI column's inverted interfaces (raw = -300·N²)
ICE_STAR = 0.15    # flags the RI column only (ice 0.9)
LAT_STAR = 70.0    # flags the RI column only (80S)
TU_STAR = -45.0    # flags the RI column only (Tu -60)
VARIANTS = {
    "pr_all": dict(tke_pr_cut=math.inf),
    "pr_below": dict(tke_pr_cut=N2_STAR),
    "pr_at": dict(tke_pr_cut=N2_STAR, tke_pr_mode="at"),
    "pr_ratio": dict(tke_pr_ratio=R_STAR),
    "pr_below_ratio": dict(tke_pr_cut=N2_STAR, tke_pr_ratio=R_STAR),
    "pr_raw": dict(tke_pr_raw_cut=RAW_STAR),
    "pr_below_raw": dict(tke_pr_cut=N2_STAR, tke_pr_raw_cut=RAW_STAR),
    "pr_ice": dict(tke_pr_ice_cut=ICE_STAR),
    "pr_lat": dict(tke_pr_lat_cut=LAT_STAR),
    "pr_tu": dict(tke_pr_tu_cut=TU_STAR),
    "kme_column": dict(tke_kme_tke=TKE_STAR),
    "kme_iface1": dict(tke_kme_tke=TKE_STAR, tke_kme_scope="iface1"),
    "kme_ke_diff": dict(tke_kme_tke=TKE_STAR, tke_kme_term="ke_diff"),
    "kme_all_columns": dict(tke_kme_tke=math.inf),
    "ml_below": dict(tke_ml_cut=N2_STAR),
    "ml_all": dict(tke_ml_cut=math.inf),
    "ml_ratio": dict(tke_ml_ratio=R_STAR),
    "ml_raw": dict(tke_ml_raw_cut=RAW_STAR),
    "ml_below_raw": dict(tke_ml_cut=N2_STAR, tke_ml_raw_cut=RAW_STAR),
}


def _ys(nlev):
    wet = (jnp.arange(NL)[None, :] <= nlev[:, None])
    rng = np.random.default_rng(11)
    return tuple(jnp.where(wet, jnp.asarray(rng.standard_normal((NCOL, NL))), 0.0)
                 for _ in range(3))


_VJP: dict = {}


def _vjp(adj=None, cuts=None):
    key = (adj, None if cuts is None else frozenset(cuts))
    if key not in _VJP:
        x, nlev = _inputs()
        _, vjp = jax.vjp(lambda x: _core(x, nlev, adj, cuts), x)
        _VJP[key] = jax.device_get(vjp(_ys(nlev))[0])
    return _VJP[key]


def _equal(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def _rel(a, b):
    num = sum(float(np.sum((np.asarray(a[k]) - np.asarray(b[k])) ** 2)) for k in a)
    den = sum(float(np.sum(np.asarray(a[k]) ** 2)) for k in a)
    return math.sqrt(num / max(den, 1e-300))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_off_is_none_and_the_pinned_core_program():
    assert AdjointConfig().tke_options() is None
    assert AdjointConfig(freeze_mixing=True, diag_cuts=frozenset({"tke.pr"})).tke_options() is None
    x, nlev = _inputs()
    for with_diags, key in ((False, "core"), (True, "core_diags")):
        got = _sha(str(jax.make_jaxpr(lambda x: _core(x, nlev, None, None, with_diags))(x)))
        check_pin(PINNED, key, got)


def test_every_candidate_changes_the_traced_core():
    x, nlev = _inputs()
    ref = str(jax.make_jaxpr(lambda x: _core(x, nlev))(x))
    for name, kw in VARIANTS.items():
        got = str(jax.make_jaxpr(lambda x: _core(x, nlev, _opts(**kw)))(x))
        assert got != ref, f"{name} traced the plain program"


@pytest.mark.parametrize("bad, err", [
    (dict(tke_pr_mode="at"), ValueError),                        # a mode without N²*
    (dict(tke_pr_cut=math.inf, tke_pr_mode="at"), ValueError),   # "at" needs a finite N²*
    (dict(tke_pr_cut=1e-8, tke_pr_mode="below"), ValueError),
    (dict(tke_pr_cut=math.nan), ValueError),
    (dict(tke_pr_cut="1e-8"), TypeError),
    (dict(tke_pr_cut=True), TypeError),
    (dict(tke_kme_tke=-1e-6), ValueError),
    (dict(tke_kme_scope="column"), ValueError),                  # scope without tke*
    (dict(tke_kme_tke=3e-6, tke_kme_scope="cell"), ValueError),
    (dict(tke_kme_tke=3e-6, tke_kme_term="km"), ValueError),
    (dict(tke_pr_ratio=0.0), ValueError),
    (dict(tke_pr_ratio=-10.0), ValueError),
    (dict(tke_pr_ratio=math.inf), ValueError),
    (dict(tke_pr_ratio=math.nan), ValueError),
    (dict(tke_pr_ratio="10"), TypeError),
    (dict(tke_pr_cut=1e-8, tke_pr_mode="at", tke_pr_ratio=10.0), ValueError),  # a cut, not "at"
    (dict(tke_pr_raw_cut=math.inf), ValueError),
    (dict(tke_pr_raw_cut=math.nan), ValueError),
    (dict(tke_pr_raw_cut="0"), TypeError),
    (dict(tke_pr_cut=1e-8, tke_pr_mode="at", tke_pr_raw_cut=0.0), ValueError),
    (dict(tke_ml_cut=math.nan), ValueError),
    (dict(tke_ml_cut="1e-8"), TypeError),
    (dict(tke_ml_ratio=0.0), ValueError),
    (dict(tke_ml_ratio=-10.0), ValueError),
    (dict(tke_ml_ratio=math.inf), ValueError),
    (dict(tke_ml_ratio=math.nan), ValueError),
    (dict(tke_ml_ratio="10"), TypeError),
    (dict(tke_ml_raw_cut=math.inf), ValueError),
    (dict(tke_ml_raw_cut=math.nan), ValueError),
    (dict(tke_ml_raw_cut="0"), TypeError),
    (dict(tke_pr_ice_cut=1.0), ValueError),
    (dict(tke_pr_ice_cut=-0.1), ValueError),
    (dict(tke_pr_ice_cut=math.nan), ValueError),
    (dict(tke_pr_ice_cut="0.15"), TypeError),
    (dict(tke_pr_lat_cut=0.0), ValueError),
    (dict(tke_pr_lat_cut=90.0), ValueError),
    (dict(tke_pr_lat_cut=math.nan), ValueError),
    (dict(tke_pr_lat_cut="70"), TypeError),
    (dict(tke_pr_tu_cut=-90.0), ValueError),
    (dict(tke_pr_tu_cut=0.0), ValueError),
    (dict(tke_pr_tu_cut=math.nan), ValueError),
    (dict(tke_pr_tu_cut="-45"), TypeError),
])
def test_validate_rejects_malformed_settings(bad, err):
    with pytest.raises(err):
        AdjointConfig(**bad).validate()


def test_describe_active_and_options():
    c = AdjointConfig(freeze_mixing_state=True, tke_pr_cut=1e-8, tke_pr_mode="at",
                      tke_kme_tke=3e-6).validate()
    assert c.active
    assert c.describe() == ("mixing_state+tke_pr_cut=1e-08+tke_pr_mode='at'"
                            "+tke_kme_tke=3e-06")
    assert c.tke_options() == TkeAdjoint(pr_n2_star=1e-8, pr_mode="at", kme_tke=3e-6,
                                         kme_scope="column", kme_term="km_e")
    assert AdjointConfig(tke_kme_tke=0.0).active                 # 0.0 is a value, not "off"
    r = AdjointConfig(tke_pr_cut=1e-8, tke_pr_ratio=10.0).validate()
    assert r.describe() == "tke_pr_cut=1e-08+tke_pr_ratio=10.0"
    assert r.tke_options() == TkeAdjoint(pr_n2_star=1e-8, pr_mode="cut", kme_tke=None,
                                         kme_scope="column", kme_term="km_e", pr_ratio=10.0)
    alone = AdjointConfig(tke_pr_ratio=10.0).validate().tke_options()
    assert alone is not None and alone.pr_n2_star is None and alone.pr_ratio == 10.0
    w = AdjointConfig(tke_pr_cut=1e-8, tke_pr_ratio=100.0, tke_pr_raw_cut=0.0).validate()
    assert w.describe() == "tke_pr_cut=1e-08+tke_pr_ratio=100.0+tke_pr_raw_cut=0.0"
    assert w.tke_options().pr_raw_cut == 0.0 and w.tke_options().pr_ratio == 100.0
    raw = AdjointConfig(tke_pr_raw_cut=1e-8).validate().tke_options()
    assert raw is not None and raw.pr_n2_star is None and raw.pr_raw_cut == 1e-8
    m = AdjointConfig(tke_ml_cut=1e-8, tke_ml_ratio=100.0, tke_ml_raw_cut=-1e-6).validate()
    assert m.describe() == "tke_ml_cut=1e-08+tke_ml_ratio=100.0+tke_ml_raw_cut=-1e-06"
    o = m.tke_options()
    assert (o.ml_n2_star, o.ml_ratio, o.ml_raw_cut) == (1e-8, 100.0, -1e-6)
    assert o.pr_ratio is None and o.pr_raw_cut is None
    alone = AdjointConfig(tke_ml_raw_cut=-1e-6).validate().tke_options()
    assert alone is not None and alone.ml_n2_star is None and alone.ml_raw_cut == -1e-6
    assert "freeze_mixing_state" in FREEZE_FLAGS
    assert set(TKE_CANDIDATE_FIELDS) <= set(AdjointConfig._fields)


# ---------------------------------------------------------------------------
# forward bitwise, effect, finiteness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_diags", [False, True])
@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_candidate_leaves_the_core_forward_bitwise(name, with_diags):
    x, nlev = _inputs()
    ref = jax.jit(lambda x: _core(x, nlev, None, None, with_diags))(x)
    got = jax.jit(lambda x: _core(x, nlev, _opts(**VARIANTS[name]), None, with_diags))(x)
    for u, v in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got)):
        assert np.array_equal(np.asarray(u), np.asarray(v)), f"{name} moved the forward"


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_candidate_changes_the_column_vjp_and_stays_finite(name, capsys):
    exact = _vjp()
    got = _vjp(_opts(**VARIANTS[name]))
    assert all(np.all(np.isfinite(np.asarray(v))) for v in got.values()), name
    r = _rel(exact, got)
    with capsys.disabled():
        print(f"\n  {name}: column VJP relative change {r:.3e}")
    assert r > 1e-10, f"{name} did not change the column VJP"


# ---------------------------------------------------------------------------
# limits: the Prandtl-number cut
# ---------------------------------------------------------------------------
def test_pr_cut_below_every_n2_is_exact():
    assert _equal(_vjp(_opts(tke_pr_cut=-math.inf)), _vjp())


def test_pr_cut_everywhere_equals_the_diagnostic_cut():
    """``inf`` is the stop_gradient itself; the masked path with every cell below N²* (1e30)
    gives the same column VJP bitwise too (on the column core; on the CORE2 step XLA fuses the
    masked zeros differently, see test_tke_candidates_core2.py)."""
    assert _equal(_vjp(_opts(tke_pr_cut=math.inf)), _vjp(cuts={"tke.pr"}))
    assert _equal(_vjp(_opts(tke_pr_cut=1e30)), _vjp(cuts={"tke.pr"}))


def _kappaH_tangent_in_n2(adj=None, cuts=None):
    """The tangent of KappaH for a unit tangent on every N² (point-wise in the column)."""
    x, nlev = _inputs()
    dx = {k: jnp.zeros_like(v) for k, v in x.items()}
    dx["Nsqr"] = jnp.ones_like(x["Nsqr"])
    _, (_, _, dKh) = jax.jvp(lambda x: _core(x, nlev, adj, cuts), (x,), (dx,))
    return np.asarray(dKh), np.asarray(x["Nsqr"])


def test_pr_cut_acts_pointwise_below_the_threshold():
    exact, n2 = _kappaH_tangent_in_n2()
    prcut, _ = _kappaH_tangent_in_n2(cuts={"tke.pr"})
    got, _ = _kappaH_tangent_in_n2(_opts(tke_pr_cut=N2_STAR))
    below = n2 < N2_STAR
    assert np.array_equal(got[below], prcut[below])
    assert np.array_equal(got[~below], exact[~below])
    # live on both sides: the Pr derivative matters in the RI column (above N²*) and nowhere
    # does the cut change a cell above the threshold
    assert np.any(exact[COL_RI] != prcut[COL_RI])
    assert np.any(below & (exact != prcut)), "the cut is not live below N²*"


def test_pr_at_below_every_n2_is_exact():
    got = _vjp(_opts(tke_pr_cut=-1.0, tke_pr_mode="at"))
    assert _rel(_vjp(), got) < 1e-14


def test_pr_at_with_a_huge_threshold_is_the_diagnostic_cut():
    got = _vjp(_opts(tke_pr_cut=1e30, tke_pr_mode="at"))
    assert _rel(_vjp(cuts={"tke.pr"}), got) < 1e-14


def test_pr_at_differs_from_the_plain_cut_where_shear_is_strong():
    """Where N² < N²* but S² is large enough that max(N², N²*)/S² stays in the clip band, the
    "at" variant keeps a (moved) derivative that the plain cut drops."""
    x, nlev = _inputs()
    n2, s2 = np.asarray(x["Nsqr"]), np.asarray(x["Ssqr"])
    ri_alt = np.maximum(n2, N2_STAR) / np.maximum(s2, 1e-12)
    live = (n2 < N2_STAR) & (6.6 * ri_alt > 1.0) & (6.6 * ri_alt < 10.0)
    at, _ = _kappaH_tangent_in_n2(_opts(tke_pr_cut=N2_STAR, tke_pr_mode="at"))
    cut, _ = _kappaH_tangent_in_n2(_opts(tke_pr_cut=N2_STAR))
    exact, _ = _kappaH_tangent_in_n2()
    above = n2 >= N2_STAR
    assert np.array_equal(at[above], exact[above])
    assert np.any(live), "no cell where the regularised Ri stays in the clip band"
    assert np.any(at[live] != cut[live])
    dead = (n2 < N2_STAR) & ~live
    np.testing.assert_allclose(at[dead], cut[dead], rtol=1e-14, atol=0.0)


def test_pr_ratio_acts_pointwise_where_the_smoothed_n2_is_a_residual():
    exact, n2 = _kappaH_tangent_in_n2()
    prcut, _ = _kappaH_tangent_in_n2(cuts={"tke.pr"})
    got, _ = _kappaH_tangent_in_n2(_opts(tke_pr_ratio=R_STAR))
    x, _ = _inputs()
    cancel = np.abs(np.asarray(_raw(x))) > R_STAR * np.abs(n2)
    assert np.array_equal(got[cancel], prcut[cancel])
    assert np.array_equal(got[~cancel], exact[~cancel])
    # live, and live where the N²* threshold would keep the derivative
    assert np.any(cancel & (n2 >= N2_STAR) & (exact != prcut)), "the ratio cut is not live"


def test_pr_ratio_joins_the_n2_threshold():
    exact, n2 = _kappaH_tangent_in_n2()
    prcut, _ = _kappaH_tangent_in_n2(cuts={"tke.pr"})
    got, _ = _kappaH_tangent_in_n2(_opts(tke_pr_cut=N2_STAR, tke_pr_ratio=R_STAR))
    x, _ = _inputs()
    m = (n2 < N2_STAR) | (np.abs(np.asarray(_raw(x))) > R_STAR * np.abs(n2))
    assert np.array_equal(got[m], prcut[m])
    assert np.array_equal(got[~m], exact[~m])
    alone, _ = _kappaH_tangent_in_n2(_opts(tke_pr_cut=N2_STAR))
    assert np.any(got != alone), "the ratio added nothing to the N²* cut"


def test_pr_ratio_limits():
    """A ratio nothing exceeds is exact; one every nonzero cell exceeds is the ``tke.pr`` cut
    (cells with N² = 0 have Ri = 0, below the clip band, where the Pr derivative is zero)."""
    assert _equal(_vjp(_opts(tke_pr_ratio=1e30)), _vjp())
    assert _equal(_vjp(_opts(tke_pr_ratio=1e-30)), _vjp(cuts={"tke.pr"}))


def test_pr_raw_cut_acts_pointwise_where_the_raw_n2_is_low():
    exact, n2 = _kappaH_tangent_in_n2()
    prcut, _ = _kappaH_tangent_in_n2(cuts={"tke.pr"})
    got, _ = _kappaH_tangent_in_n2(_opts(tke_pr_raw_cut=RAW_STAR))
    x, _ = _inputs()
    low = np.asarray(_raw(x)) < RAW_STAR
    assert np.array_equal(got[low], prcut[low])
    assert np.array_equal(got[~low], exact[~low])
    # live, and live where the smoothed N² is above the N²* threshold
    assert np.any(low & (n2 >= N2_STAR) & (exact != prcut)), "the raw-N² cut is not live"


def test_pr_raw_cut_joins_the_n2_threshold():
    exact, n2 = _kappaH_tangent_in_n2()
    prcut, _ = _kappaH_tangent_in_n2(cuts={"tke.pr"})
    got, _ = _kappaH_tangent_in_n2(_opts(tke_pr_cut=N2_STAR, tke_pr_raw_cut=RAW_STAR))
    x, _ = _inputs()
    m = (n2 < N2_STAR) | (np.asarray(_raw(x)) < RAW_STAR)
    assert np.array_equal(got[m], prcut[m])
    assert np.array_equal(got[~m], exact[~m])
    alone, _ = _kappaH_tangent_in_n2(_opts(tke_pr_cut=N2_STAR))
    assert np.any(got != alone), "the raw-N² cut added nothing to the N²* cut"


def test_pr_raw_cut_limits():
    """A threshold below every raw N² is exact; one above every raw N² is the ``tke.pr`` cut."""
    assert _equal(_vjp(_opts(tke_pr_raw_cut=-1e30)), _vjp())
    assert _equal(_vjp(_opts(tke_pr_raw_cut=1e30)), _vjp(cuts={"tke.pr"}))


def test_pr_raw_cut_needs_the_raw_n2():
    x, nlev = _inputs()
    with pytest.raises(ValueError, match="unsmoothed"):
        cvmix_tke.integrate_tke_column(
            x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
            c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"],
            adj=_opts(tke_pr_raw_cut=RAW_STAR), **KW)


def test_pr_ratio_needs_the_raw_n2():
    x, nlev = _inputs()
    with pytest.raises(ValueError, match="unsmoothed"):
        cvmix_tke.integrate_tke_column(
            x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
            c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"],
            adj=_opts(tke_pr_ratio=R_STAR), **KW)


def test_pr_ice_cut_is_the_diagnostic_cut_on_iced_columns_only():
    """Columns are independent in the core: the iced RI column gets the ``tke.pr`` column VJP,
    every other column keeps the exact one, both bitwise."""
    iced = np.asarray(_ice()) > ICE_STAR
    assert iced[COL_RI] and iced.sum() == 1
    exact, prcut = _vjp(), _vjp(cuts={"tke.pr"})
    got = _vjp(_opts(tke_pr_ice_cut=ICE_STAR))
    for k in ("tke_old", "Ssqr", "Nsqr", "dzw", "dzt"):
        assert np.array_equal(np.asarray(got[k])[~iced], np.asarray(exact[k])[~iced]), k
        assert np.array_equal(np.asarray(got[k])[iced], np.asarray(prcut[k])[iced]), k
    assert not np.array_equal(np.asarray(exact["Nsqr"])[COL_RI], np.asarray(got["Nsqr"])[COL_RI])


def test_pr_ice_cut_limits_and_join():
    """An a* no column exceeds is exact; joined with an N² mask it cuts where either holds."""
    assert _equal(_vjp(_opts(tke_pr_ice_cut=0.95)), _vjp())
    both = _vjp(_opts(tke_pr_cut=N2_STAR, tke_pr_ice_cut=ICE_STAR))
    below = _vjp(_opts(tke_pr_cut=N2_STAR))
    iced = np.asarray(_ice()) > ICE_STAR
    for k in ("tke_old", "Ssqr", "Nsqr"):
        assert np.array_equal(np.asarray(both[k])[~iced], np.asarray(below[k])[~iced]), k
    assert _equal({k: np.asarray(both[k])[iced] for k in ("Nsqr",)},
                  {k: np.asarray(_vjp(cuts={"tke.pr"})[k])[iced] for k in ("Nsqr",)})


def test_pr_ice_cut_needs_the_ice_concentration():
    x, nlev = _inputs()
    with pytest.raises(ValueError, match="ice concentration"):
        cvmix_tke.integrate_tke_column(
            x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
            c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"],
            adj=_opts(tke_pr_ice_cut=ICE_STAR), **KW)


def test_pr_lat_cut_is_the_diagnostic_cut_poleward_only():
    """The RI column (80S) gets the ``tke.pr`` column VJP, the 10N columns keep the exact one,
    both bitwise; a phi* above every column is exact."""
    polar = np.abs(np.degrees(np.asarray(_lat()))) > LAT_STAR
    assert polar[COL_RI] and polar.sum() == 1
    exact, prcut = _vjp(), _vjp(cuts={"tke.pr"})
    got = _vjp(_opts(tke_pr_lat_cut=LAT_STAR))
    for k in ("tke_old", "Ssqr", "Nsqr", "dzw", "dzt"):
        assert np.array_equal(np.asarray(got[k])[~polar], np.asarray(exact[k])[~polar]), k
        assert np.array_equal(np.asarray(got[k])[polar], np.asarray(prcut[k])[polar]), k
    assert _equal(_vjp(_opts(tke_pr_lat_cut=85.0)), exact)


def test_pr_lat_cut_needs_the_latitude():
    x, nlev = _inputs()
    with pytest.raises(ValueError, match="latitude"):
        cvmix_tke.integrate_tke_column(
            x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
            c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"],
            adj=_opts(tke_pr_lat_cut=LAT_STAR), **KW)


def test_pr_tu_cut_is_the_diagnostic_cut_in_the_diffusive_regime_only():
    """The RI column (Tu -60) gets the ``tke.pr`` column VJP, the Tu +30 columns keep the exact
    one, both bitwise; a Tu* below -60 is exact."""
    diffusive = (np.asarray(_tu()) < TU_STAR).all(1)
    assert diffusive[COL_RI] and diffusive.sum() == 1
    exact, prcut = _vjp(), _vjp(cuts={"tke.pr"})
    got = _vjp(_opts(tke_pr_tu_cut=TU_STAR))
    for k in ("tke_old", "Ssqr", "Nsqr", "dzw", "dzt"):
        assert np.array_equal(np.asarray(got[k])[~diffusive],
                              np.asarray(exact[k])[~diffusive]), k
        assert np.array_equal(np.asarray(got[k])[diffusive], np.asarray(prcut[k])[diffusive]), k
    assert _equal(_vjp(_opts(tke_pr_tu_cut=-75.0)), exact)


def test_pr_tu_cut_needs_the_turner_angle():
    x, nlev = _inputs()
    with pytest.raises(ValueError, match="Turner angle"):
        cvmix_tke.integrate_tke_column(
            x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
            c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"],
            adj=_opts(tke_pr_tu_cut=TU_STAR), **KW)


def test_turner_angle_regimes():
    """One column, constant alpha/beta: interface 1 diffusive (cold fresh over warm salty),
    2 salt fingering, 3 both stable, 4 statically unstable; the surface interface is 0."""
    al, be = 2.0e-4, 7.6e-4
    T = jnp.asarray([[4.0, 6.0, 3.0, 2.0, 3.0]])
    S = jnp.asarray([[34.0, 35.0, 34.9, 35.0, 35.0]])
    tu = np.asarray(eos.turner_angle(T, S, jnp.full_like(T, al), jnp.full_like(T, be)))[0]
    dT, dS = np.array([-2.0, 3.0, 1.0, -1.0]), np.array([-1.0, 0.1, -0.1, 0.0])
    a, b = al * dT, be * dS
    np.testing.assert_allclose(tu[1:], np.degrees(np.arctan2(a + b, a - b)), rtol=1e-12)
    assert tu[0] == 0.0
    assert -90.0 < tu[1] < -45.0 and 45.0 < tu[2] < 90.0 and abs(tu[3]) < 45.0
    assert abs(tu[4]) > 90.0


# ---------------------------------------------------------------------------
# limits: the near-floor km_e cut
# ---------------------------------------------------------------------------
def test_kme_threshold_zero_is_exact():
    assert _equal(_vjp(_opts(tke_kme_tke=0.0)), _vjp())


def test_kme_everywhere_equals_the_diagnostic_cut():
    assert _equal(_vjp(_opts(tke_kme_tke=math.inf)), _vjp(cuts={"tke.km_e"}))
    assert _equal(_vjp(_opts(tke_kme_tke=1e30)), _vjp(cuts={"tke.km_e"}))     # masked path


def test_kme_acts_on_flagged_columns_only():
    """Columns are independent in the core, so an unflagged column keeps its exact VJP:
    bitwise for the km_e term; to rounding for the ke_diff term (KappaM then reaches the
    cotangent through two separately accumulated copies)."""
    x, _ = _inputs()
    t1 = np.asarray(x["tke_old"])[:, 1]
    flagged = t1 < TKE_STAR
    assert flagged[COL_WALLTOP] and not flagged[COL_RI]
    exact = _vjp()
    for name in ("kme_column", "kme_iface1", "kme_ke_diff"):
        got = _vjp(_opts(**VARIANTS[name]))
        for k in ("tke_old", "Ssqr", "Nsqr", "dzw", "dzt"):
            a, b = np.asarray(exact[k])[~flagged], np.asarray(got[k])[~flagged]
            if name == "kme_ke_diff":
                np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12 * np.max(np.abs(a)),
                                           err_msg=f"{name} moved unflagged {k}")
            else:
                assert np.array_equal(a, b), f"{name} moved unflagged {k}"
        assert not np.array_equal(np.asarray(exact["tke_old"])[COL_WALLTOP],
                                  np.asarray(got["tke_old"])[COL_WALLTOP]), name


def _kappa_tangents_in_tke(adj=None):
    x, nlev = _inputs()
    dx = {k: jnp.zeros_like(v) for k, v in x.items()}
    dx["tke_old"] = jnp.ones_like(x["tke_old"])
    _, (dtke, dKm, dKh) = jax.jvp(lambda x: _core(x, nlev, adj), (x,), (dx,))
    return np.asarray(dtke), np.asarray(dKm), np.asarray(dKh)


def test_kme_scope_and_term():
    _, km0, kh0 = _kappa_tangents_in_tke()
    _, km1, _ = _kappa_tangents_in_tke(_opts(**VARIANTS["kme_iface1"]))
    _, kmc, _ = _kappa_tangents_in_tke(_opts(**VARIANTS["kme_column"]))
    tk2, km2, kh2 = _kappa_tangents_in_tke(_opts(**VARIANTS["kme_ke_diff"]))
    tk0, _, _ = _kappa_tangents_in_tke()
    c = COL_WALLTOP
    x, _ = _inputs()
    flagged = np.asarray(x["tke_old"])[:, 1] < TKE_STAR
    # iface1: interface 1 of flagged columns only; column: more than interface 1
    rows, cols = np.nonzero(km1 != km0)
    assert set(cols.tolist()) == {1} and flagged[rows].all() and c in rows, (rows, cols)
    assert np.sum(kmc[c] != km0[c]) > 1
    # ke_diff: KappaM/KappaH tangents untouched, tke_new's tangent moved
    assert np.array_equal(km2, km0) and np.array_equal(kh2, kh0)
    assert not np.array_equal(tk2[c], tk0[c])


def test_kme_keeps_the_constant_paths():
    """c_k multiplies sqrt(tke) in KappaM; the cut removes d/d sqrt(tke) only, so the column
    cotangent of every constant stays nonzero (it changes, through the solve)."""
    got = _vjp(_opts(tke_kme_tke=math.inf))
    for k in ("c_k", "c_eps", "cd", "alpha_tke"):
        assert float(np.asarray(got[k])) != 0.0, k


# ---------------------------------------------------------------------------
# limits: the mixing-length cut
# ---------------------------------------------------------------------------
def test_ml_cut_limits():
    """A threshold below every N² is exact; ``inf`` or one above every N² is ``tke.ml_n2``."""
    assert _equal(_vjp(_opts(tke_ml_cut=-math.inf)), _vjp())
    assert _equal(_vjp(_opts(tke_ml_cut=math.inf)), _vjp(cuts={"tke.ml_n2"}))
    assert _equal(_vjp(_opts(tke_ml_cut=1e30)), _vjp(cuts={"tke.ml_n2"}))


def _tangent_from_n2(where, adj=None, cuts=None):
    """The core's output tangent for a unit N² tangent on the cells ``where`` only."""
    x, nlev = _inputs()
    dx = {k: jnp.zeros_like(v) for k, v in x.items()}
    dx["Nsqr"] = jnp.asarray(where, dtype=x["Nsqr"].dtype)
    _, t = jax.jvp(lambda x: _core(x, nlev, adj, cuts), (x,), (dx,))
    return [np.asarray(v) for v in t]


def test_ml_cut_acts_on_the_cells_below_the_threshold():
    """The cut is linear in the N² tangent: one on cells below N²* propagates as under the
    ``tke.ml_n2`` cut, one on cells at or above N²* as in the exact core (the walls and scans
    may carry either to other interfaces)."""
    x, _ = _inputs()
    below = np.asarray(x["Nsqr"]) < N2_STAR
    adj = _opts(tke_ml_cut=N2_STAR)
    for a, b in zip(_tangent_from_n2(below, adj), _tangent_from_n2(below, cuts={"tke.ml_n2"})):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)
    for a, b in zip(_tangent_from_n2(~below, adj), _tangent_from_n2(~below)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)
    live = [np.any(a != b) for a, b in zip(_tangent_from_n2(below),
                                           _tangent_from_n2(below, cuts={"tke.ml_n2"}))]
    assert any(live), "the mixing length's N² derivative is not live below N²*"


def _ml_masked_cells_propagate_as_the_diagnostic_cut(adj, mask):
    for a, b in zip(_tangent_from_n2(mask, adj), _tangent_from_n2(mask, cuts={"tke.ml_n2"})):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)
    for a, b in zip(_tangent_from_n2(~mask, adj), _tangent_from_n2(~mask)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)
    live = [np.any(a != b) for a, b in zip(_tangent_from_n2(mask),
                                           _tangent_from_n2(mask, cuts={"tke.ml_n2"}))]
    assert any(live), "the mixing length's N² derivative is not live on the masked cells"


def test_ml_ratio_acts_on_the_cells_where_the_smoothed_n2_is_a_residual():
    x, _ = _inputs()
    n2 = np.asarray(x["Nsqr"])
    cancel = np.abs(np.asarray(_raw(x))) > R_STAR * np.abs(n2)
    assert np.any(cancel & (n2 >= N2_STAR)), "no residual cell above the N²* threshold"
    _ml_masked_cells_propagate_as_the_diagnostic_cut(_opts(tke_ml_ratio=R_STAR), cancel)


def test_ml_raw_cut_acts_on_the_cells_where_the_raw_n2_is_low():
    x, _ = _inputs()
    low = np.asarray(_raw(x)) < RAW_STAR
    _ml_masked_cells_propagate_as_the_diagnostic_cut(_opts(tke_ml_raw_cut=RAW_STAR), low)


def test_ml_raw_cut_joins_the_n2_threshold():
    x, _ = _inputs()
    n2 = np.asarray(x["Nsqr"])
    m = (n2 < N2_STAR) | (np.asarray(_raw(x)) < RAW_STAR)
    _ml_masked_cells_propagate_as_the_diagnostic_cut(
        _opts(tke_ml_cut=N2_STAR, tke_ml_raw_cut=RAW_STAR), m)
    assert not _equal(_vjp(_opts(tke_ml_cut=N2_STAR, tke_ml_raw_cut=RAW_STAR)),
                      _vjp(_opts(tke_ml_cut=N2_STAR))), "the raw-N² mask added nothing"


def test_ml_masked_cut_limits():
    """A ratio or a raw threshold no cell passes is exact; a raw threshold above every raw N²
    is ``tke.ml_n2``."""
    assert _equal(_vjp(_opts(tke_ml_ratio=1e30)), _vjp())
    assert _equal(_vjp(_opts(tke_ml_raw_cut=-1e30)), _vjp())
    assert _equal(_vjp(_opts(tke_ml_raw_cut=1e30)), _vjp(cuts={"tke.ml_n2"}))


@pytest.mark.parametrize("kw", [dict(tke_ml_ratio=R_STAR), dict(tke_ml_raw_cut=RAW_STAR)])
def test_ml_masked_cut_needs_the_raw_n2(kw):
    x, nlev = _inputs()
    with pytest.raises(ValueError, match="unsmoothed"):
        cvmix_tke.integrate_tke_column(
            x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
            c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"],
            adj=_opts(**kw), **KW)


def test_pr_cut_keeps_the_constant_paths():
    got = _vjp(_opts(tke_pr_cut=math.inf))
    for k in ("c_k", "c_eps", "cd", "alpha_tke"):
        assert float(np.asarray(got[k])) != 0.0, k


def test_fixture_columns_are_live():
    x, _ = _inputs()
    n2, t = np.asarray(x["Nsqr"]), np.asarray(x["tke_old"])
    assert np.any(n2[COL_RI] > N2_STAR) and np.any((n2[COL_RI] > 0) & (n2[COL_RI] < N2_STAR))
    assert np.any((n2[COL_WALLTOP] < N2_STAR)
                                                   & (n2[COL_WALLTOP] > 0))
    assert t[COL_WALLTOP, 1] < TKE_STAR
    assert all(t[i, 1] > TKE_STAR for i in (COL_STRAT, COL_RI, COL_NEUTRAL))
