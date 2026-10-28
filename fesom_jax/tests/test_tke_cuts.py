"""TKE derivative cut points (``tke.*`` in ``AdjointConfig.diag_cuts``), the column-input
assembly ``tke.tke_column_inputs`` and the regime logger ``cvmix_tke.tke_regimes``.

The pure column core on hand-built synthetic columns (no mesh; ``ci`` group). The assembled
``mixing_tke`` on the pi mesh is tested in ``test_tke_cuts_pi.py``, the CORE2 step in
``test_tke_cuts_core2.py``.

Properties:

* no cuts ⇒ the identical program: the jaxpr text of the column core hashes to the value
  pinned from the parent commit (cd2ba05, before any cut point existed), with and without
  an explicit empty set;
* every cut leaves the forward bitwise;
* each cut changes the column VJP w.r.t. the input(s) it acts on, on a column where its term
  is live (``CUT_ACTS_ON``); cuts that sever an input's only path zero that cotangent;
* ``tke.deep`` cuts the derivative below 500 m only;
* ``tke_regimes``: analytic ``dKm/dtke`` equals the Jacobian diagonal to 1e-12 in every
  regime; the off-diagonal term of a length that inherits the neighbour's tke; labels on
  hand-built cells; the floor flag; finite everywhere (dry and padding columns included).
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import cvmix_tke
from fesom_jax.adjoint_stab import KNOWN_CUTS, AdjointConfig

# ---------------------------------------------------------------------------
# synthetic columns
# ---------------------------------------------------------------------------
NL = 24                       # interfaces; layers 0..NL-2
KW = dict(mxl_min=1.0e-8, tke_min=1.0e-6, kappaM_max=100.0, dt=1800.0)
CONSTS = dict(c_k=0.1, c_eps=0.7, cd=3.75, alpha_tke=30.0)
TKE_CUTS = sorted(c for c in KNOWN_CUTS if c.startswith("tke."))

# column index -> purpose (the hand-built cells tested below refer to these)
COL_STRAT = 0      # strong stratification, small tke: free length in the interior, Pr = 10
COL_RI = 1         # moderate Ri (Pr unclipped), free length
COL_NEUTRAL = 2    # N² <= 0: wall-bound lengths (k=1 wall_top, below wall_other)
COL_WALLTOP = 3    # tke at the floor, weak N²: first interface wall-bound (the small-tke case)
COL_INHERIT = 4    # k=1 free (tiny mxl0), k=2 bound by mxl[1]+dzw[1] (inherits tke[1])
COL_MXLFLOOR = 5   # an interior interface with mxl0 < mxl_min (the length floor)
COL_DEEP = 6       # deep neutral column, large tke: KappaM cap and interfaces below 500 m
COL_SINK = 7       # stratified, tke at the floor, no shear: the solve undershoots tke_min
COL_SHALLOW = 8    # nlev = 2
COL_PAD = 9        # padding column (nlev = -1, all zero), as sharded padding nodes
NCOL = 10


def synthetic_columns():
    """Inputs of ``integrate_tke_column`` for ``NCOL`` hand-built columns ``[NCOL, NL]``."""
    rng = np.random.default_rng(7)
    nlev = np.full(NCOL, NL - 1, np.int64)
    nlev[COL_SHALLOW] = 2
    nlev[COL_PAD] = -1
    nlev[COL_WALLTOP] = 12
    kk = np.arange(NL)
    # layer thicknesses: 10 m growing with depth (~1370 m total), deep column 200 m layers
    dz = 10.0 * 1.12 ** np.arange(NL)
    dzw = np.tile(dz, (NCOL, 1))
    dzw[COL_DEEP] = 200.0
    jit = 1.0 + 0.01 * rng.standard_normal((NCOL, NL))      # break exact ties
    dzw = dzw * jit
    wet_layer = kk[None, :] <= nlev[:, None] - 1
    dzw = np.where(wet_layer, dzw, 0.0)
    # interface spacing: interior mean of adjacent layers, half layers at the ends
    dzt = np.zeros((NCOL, NL))
    for i in range(NCOL):
        n = nlev[i]
        if n < 0:
            continue
        dzt[i, 0] = dzw[i, 0] / 2.0
        for k in range(1, n):
            dzt[i, k] = 0.5 * (dzw[i, k - 1] + dzw[i, k])
        dzt[i, n] = dzw[i, n - 1] / 2.0 if n >= 1 else 0.0

    r = lambda s: 1.0 + 0.1 * rng.standard_normal(s)          # noqa: E731
    tke = np.zeros((NCOL, NL))
    N2 = np.zeros((NCOL, NL))
    S2 = np.zeros((NCOL, NL))
    tke[COL_STRAT], N2[COL_STRAT], S2[COL_STRAT] = 1e-5 * r(NL), 1e-4 * r(NL), 1e-5 * r(NL)
    tke[COL_RI], N2[COL_RI], S2[COL_RI] = 1e-4 * r(NL), 1e-5 * r(NL), 4e-5 * r(NL)
    tke[COL_NEUTRAL], N2[COL_NEUTRAL], S2[COL_NEUTRAL] = (
        3e-4 * r(NL), -1e-6 * r(NL), 2e-5 * r(NL))
    tke[COL_WALLTOP], N2[COL_WALLTOP], S2[COL_WALLTOP] = (
        1.0e-6 * r(NL), 1e-9 * r(NL), 1e-6 * r(NL))
    tke[COL_WALLTOP, 1] = 1.2e-6
    tke[COL_INHERIT], N2[COL_INHERIT], S2[COL_INHERIT] = (
        1e-4 * r(NL), 1e-10 * r(NL), 1e-5 * r(NL))
    tke[COL_INHERIT, 1], N2[COL_INHERIT, 1] = 1.0e-6, 1.0e-4
    tke[COL_MXLFLOOR], N2[COL_MXLFLOOR], S2[COL_MXLFLOOR] = (
        1e-4 * r(NL), 1e-5 * r(NL), 3e-5 * r(NL))
    tke[COL_MXLFLOOR, 5], N2[COL_MXLFLOOR, 5] = 1.0e-6, 1.0e12     # mxl0 = 1.4e-9 < mxl_min
    tke[COL_DEEP], N2[COL_DEEP], S2[COL_DEEP] = 0.5 * r(NL), -1e-7 * r(NL), 1e-5 * r(NL)
    tke[COL_SINK], N2[COL_SINK], S2[COL_SINK] = 1.0e-6 * np.ones(NL), 5e-4 * r(NL), 0.0
    tke[COL_SHALLOW], N2[COL_SHALLOW], S2[COL_SHALLOW] = 1e-4 * r(NL), 1e-5 * r(NL), 1e-5 * r(NL)

    wet = kk[None, :] <= nlev[:, None]
    interior = (kk[None, :] >= 1) & (kk[None, :] <= nlev[:, None] - 1)
    tke = np.where(wet, tke, 0.0)
    N2 = np.where(interior, N2, 0.0)          # zeroed at the ends, as the driver does
    S2 = np.where(interior, S2, 0.0)
    forc = 1e-4 * np.abs(r(NCOL))
    forc[COL_PAD] = 0.0
    f = lambda x: jnp.asarray(x, jnp.float64)                   # noqa: E731
    return dict(tke_old=f(tke), Ssqr=f(S2), Nsqr=f(N2), dzw=f(dzw), dzt=f(dzt),
                forc_tke_surf=f(forc), nlev=jnp.asarray(nlev, jnp.int32))


DIFF_INPUTS = ("tke_old", "Ssqr", "Nsqr", "dzw", "dzt", "forc_tke_surf",
               "c_k", "c_eps", "cd", "alpha_tke")


def _core(x, nlev, cuts=None, with_diags=False):
    """The column core as a function of the dict ``x`` of differentiable inputs."""
    extra = {} if cuts is None else {"cuts": cuts}
    return cvmix_tke.integrate_tke_column(
        x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
        c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"],
        with_diags=with_diags, **KW, **extra)


def _inputs():
    cols = synthetic_columns()
    x = {k: cols[k] for k in DIFF_INPUTS if k in cols}
    x.update({k: jnp.asarray(v, jnp.float64) for k, v in CONSTS.items()})
    return x, cols["nlev"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def core_jaxpr_sha(with_diags=False, cuts=None) -> str:
    x, nlev = _inputs()
    return _sha(str(jax.make_jaxpr(lambda x: _core(x, nlev, cuts, with_diags))(x)))


# sha256 of the jaxpr text of the cut-free column core, jax 0.10.1, CPU, x64, computed by ``core_jaxpr_sha`` with
# PYTHONPATH pointing at a checkout without the cut points. First pinned from cd2ba05 (7ff11206 / dd62d994);
# re-pinned on perf/core2-adj-fast from perf/core2-gpu 0378994, whose column-kernel dispatch moves the scans'
# moveaxis after their init (the same operations, CPU path unchanged; the merge traces 0378994's program exactly).
# If JAX is upgraded, or the TKE physics is changed on purpose, re-pin from the commit before the change.
PINNED_JAX = "0.10.1"
PINNED = {
    "core": "0d6f8c48893fcd780f0ab07dfe4216787c768f0867853f3e63f3376d599429c4",
    "core_diags": "64532f033d6ed458c510ab112e87921e1524c9dabf9843f86a90e805446a5338",
}


def check_pin(pins, key, got):
    assert jax.__version__ == PINNED_JAX, (
        f"the pinned jaxpr hashes were taken with jax {PINNED_JAX}, this is {jax.__version__}: "
        f"re-pin from the parent commit (see the comment above PINNED)")
    assert got == pins[key], (
        f"{key}: the traced program changed ({got[:12]} != pinned {pins[key][:12]}); with "
        f"no cuts the TKE closure must trace exactly the pre-cut program")


# ---------------------------------------------------------------------------
# no cuts => the identical program
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_diags", [False, True])
def test_no_cuts_traces_the_pinned_core_program(with_diags):
    key = "core_diags" if with_diags else "core"
    ref = core_jaxpr_sha(with_diags)
    assert core_jaxpr_sha(with_diags, cuts=frozenset()) == ref
    assert core_jaxpr_sha(with_diags, cuts=set()) == ref
    check_pin(PINNED, key, ref)
    # control: a live cut changes the program
    assert core_jaxpr_sha(with_diags, cuts=frozenset({"tke.km_e"})) != ref


def test_unknown_cut_is_rejected_by_the_core():
    x, nlev = _inputs()
    with pytest.raises(ValueError, match="unknown AdjointConfig.diag_cuts"):
        _core(x, nlev, cuts={"tke.bogus"})


# ---------------------------------------------------------------------------
# forward bitwise
# ---------------------------------------------------------------------------
def _assert_tree_equal(a, b, what):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for i, (u, v) in enumerate(zip(la, lb)):
        assert np.array_equal(np.asarray(u), np.asarray(v)), f"{what}: leaf {i} moved"


@pytest.mark.parametrize("with_diags", [False, True])
def test_all_cuts_leave_the_core_forward_bitwise(with_diags):
    x, nlev = _inputs()
    ref = jax.jit(lambda x: _core(x, nlev, None, with_diags))(x)
    got = jax.jit(lambda x: _core(x, nlev, frozenset(TKE_CUTS), with_diags))(x)
    _assert_tree_equal(ref, got, "column core under all tke cuts (jit)")
    _assert_tree_equal(_core(x, nlev, None, with_diags),
                       _core(x, nlev, frozenset(TKE_CUTS), with_diags),
                       "column core under all tke cuts (eager)")


# ---------------------------------------------------------------------------
# effect: each cut changes the column VJP w.r.t. the input(s) it acts on
# ---------------------------------------------------------------------------
# cut -> the inputs whose cotangent must change (on the synthetic columns, where the term is
# live: tke > 0 everywhere wet; free lengths in COL_STRAT/COL_RI; unclipped Pr in COL_RI;
# forc > 0; interfaces below 500 m in COL_DEEP and the 23-layer columns)
CUT_ACTS_ON = {
    "tke.km_e": ("tke_old",),
    "tke.ml_e": ("tke_old",),
    "tke.ml_n2": ("Nsqr",),
    "tke.ml_wall": ("tke_old", "Nsqr", "dzw"),
    "tke.shear": ("Ssqr",),
    "tke.n2": ("Nsqr",),
    "tke.pr": ("Ssqr", "Nsqr"),
    "tke.prod": ("Ssqr", "c_k"),
    "tke.buoy": ("Nsqr", "c_k"),
    "tke.diss": ("c_eps", "tke_old"),
    "tke.ke_diff": ("alpha_tke", "dzw"),
    "tke.surf": ("forc_tke_surf", "cd"),
    "tke.deep": ("tke_old", "c_k"),
    "tke.kh_floor": ("tke_old", "c_k"),
}
# cut (set) -> inputs whose ONLY path it severs: their cotangent must become exactly zero
CUT_ZEROES = {
    ("tke.surf",): ("forc_tke_surf", "cd"),
    ("tke.diss",): ("c_eps",),
    ("tke.ke_diff",): ("alpha_tke",),
    ("tke.shear",): ("Ssqr",),
    ("tke.ml_n2", "tke.n2"): ("Nsqr",),
}

_VJP: dict = {}


def _column_vjp(cuts):
    key = frozenset(cuts)
    if key not in _VJP:
        x, nlev = _inputs()
        wet = (jnp.arange(NL)[None, :] <= nlev[:, None])
        rng = np.random.default_rng(11)
        ys = tuple(jnp.where(wet, jnp.asarray(rng.standard_normal((NCOL, NL))), 0.0)
                   for _ in range(3))
        _, vjp = jax.vjp(lambda x: _core(x, nlev, key), x)
        _VJP[key] = jax.device_get(vjp(ys)[0])
    return _VJP[key]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))


def test_cut_table_covers_every_tke_cut():
    assert set(CUT_ACTS_ON) == set(TKE_CUTS)


@pytest.mark.parametrize("cut", TKE_CUTS)
def test_each_cut_changes_the_column_vjp(cut, capsys):
    base, got = _column_vjp(()), _column_vjp((cut,))
    for k in DIFF_INPUTS:
        assert np.all(np.isfinite(np.asarray(got[k]))), f"{cut}: non-finite d/d{k}"
    changes = {k: _rel(base[k], got[k]) for k in DIFF_INPUTS}
    with capsys.disabled():
        print(f"\n  {cut}: relative change of the column VJP: "
              + ", ".join(f"{k} {v:.2e}" for k, v in changes.items() if v > 0))
    for k in CUT_ACTS_ON[cut]:
        assert changes[k] > 1e-12, f"{cut} did not change d/d{k} — is its term live?"


@pytest.mark.parametrize("cuts", list(CUT_ZEROES), ids=lambda c: "+".join(c))
def test_cut_of_the_only_path_zeroes_the_cotangent(cuts):
    base, got = _column_vjp(()), _column_vjp(cuts)
    for k in CUT_ZEROES[cuts]:
        assert np.any(np.asarray(base[k]) != 0.0), f"d/d{k} is zero without the cut"
        assert np.all(np.asarray(got[k]) == 0.0), f"{cuts} left a nonzero d/d{k}"


def test_deep_cut_acts_below_500m_only():
    """Tangent of the three outputs w.r.t. c_k: zero at interfaces deeper than 500 m under
    ``tke.deep``, bitwise unchanged above."""
    x, nlev = _inputs()
    dx = jax.tree.map(jnp.zeros_like, x)
    dx["c_k"] = jnp.asarray(1.0)
    _, t0 = jax.jvp(lambda x: _core(x, nlev), (x,), (dx,))
    _, t1 = jax.jvp(lambda x: _core(x, nlev, frozenset({"tke.deep"})), (x,), (dx,))
    depth = np.asarray(cvmix_tke._iface_depth(x["dzw"]))
    deep = depth > cvmix_tke.TKE_DEEP_CUT_Z
    wet = np.arange(NL)[None, :] <= np.asarray(nlev)[:, None]
    assert np.count_nonzero(deep & wet) > 0 and np.count_nonzero(~deep & wet) > 0
    for a, b in zip(t0, t1):
        a, b = np.asarray(a), np.asarray(b)
        assert np.all(b[deep] == 0.0)
        assert np.array_equal(a[~deep], b[~deep])
        assert np.any(a[deep & wet] != 0.0)          # the cut did something there


# ---------------------------------------------------------------------------
# A3: the regime logger
# ---------------------------------------------------------------------------
def _regimes(x, nlev):
    return cvmix_tke.tke_regimes(
        x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
        c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"], **KW)


def _km_jacobian(x, nlev, cuts=None):
    """``jac[k][:, j] = d KappaM[:, j] / d tke_old[:, k]`` for every column: one JVP per
    one-hot level tangent (columns are independent, so column i's response is its own),
    vmapped over the levels in one compiled program."""
    def one(e):
        dx = jax.tree.map(jnp.zeros_like, x)
        dx["tke_old"] = jnp.broadcast_to(e, (NCOL, NL))
        return jax.jvp(lambda x: _core(x, nlev, cuts)[1], (x,), (dx,))[1]
    out = np.asarray(jax.jit(jax.vmap(one))(jnp.eye(NL)))
    return [out[k] for k in range(NL)]


@pytest.fixture(scope="module")
def regime_data():
    x, nlev = _inputs()
    reg = jax.device_get(jax.jit(lambda x: _regimes(x, nlev))(x))
    jac = _km_jacobian(x, nlev)                             # jac[k][:, j] = dKm[j]/dtke[k]
    diag = np.stack([jac[k][:, k] for k in range(NL)], axis=1)
    return x, nlev, reg, jac, diag


def test_regime_labels_on_hand_built_cells(regime_data):
    _, nlev, reg, _, _ = regime_data
    r = np.asarray(reg["regime"])
    name = {i: n for i, n in enumerate(cvmix_tke.REGIMES)}
    lab = lambda i, k: name.get(int(r[i, k]), "dry")        # noqa: E731
    assert lab(COL_STRAT, 5) == "free"
    assert lab(COL_RI, 5) == "free"
    assert lab(COL_NEUTRAL, 1) == "wall_top"
    assert lab(COL_NEUTRAL, 5) == "wall_other"
    assert lab(COL_WALLTOP, 1) == "wall_top"
    assert lab(COL_INHERIT, 1) == "free" and lab(COL_INHERIT, 2) == "wall_other"
    assert lab(COL_MXLFLOOR, 5) == "floor"
    for i in range(NCOL - 1):                                # surface + bottom: floor
        assert lab(i, 0) == "floor" and lab(i, int(nlev[i])) == "floor"
    assert np.all(r[COL_PAD] == -1)
    assert lab(COL_SHALLOW, 3) == "dry"
    # every regime is present, so the Jacobian test below covers all of them
    assert set(np.unique(r[r >= 0])) == {0, 1, 2, 3}
    assert bool(np.asarray(reg["at_cap"])[COL_DEEP].any()), "no KappaM-cap cell in COL_DEEP"


def test_analytic_dkm_dtke_equals_the_jacobian_diagonal(regime_data, capsys):
    """Per regime, max |analytic − diag| / max(|diag|, tiny) ≤ 1e-12 (the regime's own
    scale: dKm/dtke spans ~1e-3 .. 1e3 across the columns)."""
    _, nlev, reg, _, diag = regime_data
    r = np.asarray(reg["regime"])
    ana = np.asarray(reg["dKm_dtke"])
    wet = np.arange(NL)[None, :] <= np.asarray(nlev)[:, None]
    lines = []
    for code, nm in enumerate(cvmix_tke.REGIMES):
        m = wet & (r == code) & ~np.asarray(reg["at_cap"])
        assert m.any(), nm
        err = np.abs(ana[m] - diag[m]) / np.maximum(np.abs(diag[m]), 1e-300)
        lines.append(f"{nm}: n={m.sum()} max rel {err.max():.2e} "
                     f"(|dKm/dtke| {np.abs(diag[m]).min():.2e}..{np.abs(diag[m]).max():.2e})")
        assert err.max() <= 1e-12, f"{nm}: analytic dKm/dtke off by {err.max():.3e}"
    cap = wet & np.asarray(reg["at_cap"])
    assert np.all(ana[cap] == 0.0) and np.all(diag[cap] == 0.0)
    assert np.all(ana[~wet] == 0.0) and np.all(diag[~wet] == 0.0)
    with capsys.disabled():
        print("\n  dKm/dtke analytic vs Jacobian diagonal: " + "; ".join(lines)
              + f"; cap n={cap.sum()}")


def test_wall_other_inherits_the_upper_neighbours_tke(regime_data, capsys):
    """COL_INHERIT k=2: mxl[2] = mxl[1] + dzw[1] with mxl[1] = mxl0[1] free, so KappaM[2]
    depends on tke[1]: ∂KappaM[2]/∂tke[1] = c_k·√tke[2]·√2/(N[1]·2√tke[1]) — an
    OFF-diagonal term the diagonal dKm/dtke does not contain."""
    x, _, reg, jac, diag = regime_data
    i = COL_INHERIT
    t = np.asarray(x["tke_old"])[i]
    n2 = np.asarray(x["Nsqr"])[i]
    c_k = float(x["c_k"])
    off = jac[1][i, 2]
    ana = c_k * np.sqrt(t[2]) * np.sqrt(2.0) / (np.sqrt(max(1e-12, n2[1])) * 2 * np.sqrt(t[1]))
    rel = abs(off - ana) / abs(ana)
    with capsys.disabled():
        print(f"\n  inheritance COL_INHERIT k=2: dKm[2]/dtke[1] = {off:.6e} (analytic "
              f"{ana:.6e}, rel {rel:.1e}); diagonal dKm[2]/dtke[2] = {diag[i, 2]:.6e}")
    assert rel <= 1e-12
    assert abs(off) > 0.1 * abs(diag[i, 2])            # not negligible next to the diagonal
    # a wall_top cell has no off-diagonal dependence at all (its length is the wall's)
    j = COL_NEUTRAL
    assert int(np.asarray(reg["regime"])[j, 1]) == 1
    assert all(jac[k][j, 1] == 0.0 for k in range(NL) if k != 1)


def test_wall_cut_fixes_the_wall_bound_lengths(regime_data, capsys):
    """``tke.ml_wall``: a wall-bound length (wall_top, wall_other) is treated as fixed, so
    KappaM there keeps only its √tke factor — COL_INHERIT's inherited dKm[2]/dtke[1] becomes
    exactly 0, every off-diagonal of a wall-bound cell is 0, and its diagonal (which never had
    a length term) is unchanged; free cells keep their Jacobian."""
    x, nlev, reg, jac, diag = regime_data
    cut = _km_jacobian(x, nlev, frozenset({"tke.ml_wall"}))
    regime = np.asarray(reg["regime"])
    wall = (regime == 1) | (regime == 2)
    free = regime == 0
    i = COL_INHERIT
    with capsys.disabled():
        print(f"\n  COL_INHERIT dKm[2]/dtke[1]: {jac[1][i, 2]:.3e} -> {cut[1][i, 2]:.3e}; "
              f"wall-bound cells {int(wall.sum())}, free {int(free.sum())}")
    assert jac[1][i, 2] != 0.0 and cut[1][i, 2] == 0.0
    for k in range(NL):
        off = wall.copy()
        off[:, k] = False
        assert np.all(cut[k][off] == 0.0), f"wall-bound off-diagonal left at tke level {k}"
    cdiag = np.stack([cut[k][:, k] for k in range(NL)], axis=1)
    np.testing.assert_allclose(cdiag[wall], diag[wall], rtol=1e-13, atol=0.0)
    for k in range(NL):
        np.testing.assert_allclose(cut[k][free], jac[k][free], rtol=1e-13, atol=0.0)


def test_kh_floor_cut_acts_below_the_floor_only():
    """``tke.kh_floor``: the KappaH tangent is zero where KappaH < TKE_KH_FLOOR and the uncut
    one elsewhere; tke_new and KappaM are untouched (the floor acts on what leaves for Kv)."""
    x, nlev = _inputs()
    rng = np.random.default_rng(5)
    dx = {k: jnp.asarray(rng.standard_normal(np.shape(v))) for k, v in x.items()}
    (_, _, kh), t0 = jax.jvp(lambda x: _core(x, nlev), (x,), (dx,))
    _, t1 = jax.jvp(lambda x: _core(x, nlev, frozenset({"tke.kh_floor"})), (x,), (dx,))
    low = np.asarray(kh) < cvmix_tke.TKE_KH_FLOOR
    t0, t1 = [np.asarray(a) for a in t0], [np.asarray(a) for a in t1]
    assert np.any(low & (t0[2] != 0.0)), "no live cell below the floor"
    assert np.all(t1[2][low] == 0.0)
    assert np.array_equal(t1[2][~low], t0[2][~low])
    assert np.array_equal(t1[0], t0[0]) and np.array_equal(t1[1], t0[1])


def test_floor_flag_is_the_pre_floor_solve(regime_data):
    """Independent check: with the tke floor disabled the core returns the pre-floor solve."""
    x, nlev, reg, _, _ = regime_data
    kw = dict(KW, tke_min=-np.inf)
    solve = np.asarray(cvmix_tke.integrate_tke_column(
        x["tke_old"], x["Ssqr"], x["Nsqr"], x["dzw"], x["dzt"], x["forc_tke_surf"], nlev,
        c_k=x["c_k"], c_eps=x["c_eps"], cd=x["cd"], alpha_tke=x["alpha_tke"], **kw)[0])
    wet = np.arange(NL)[None, :] <= np.asarray(nlev)[:, None]
    expect = wet & (solve < KW["tke_min"])
    got = np.asarray(reg["floor_hit"])
    assert np.array_equal(got, expect)
    assert got[COL_SINK].any() and not got.all()


def test_regimes_finite_everywhere_and_forward_only(regime_data):
    x, nlev, reg, _, _ = regime_data
    for k, v in reg.items():
        assert np.all(np.isfinite(np.asarray(v, np.float64))), f"{k} has non-finite values"
    F = np.asarray(reg["F"])
    assert np.all(F >= 0.0) and F.max() > 0.0
    assert np.all(F[COL_PAD] == 0.0)
    # Ri, Pr, F by their definitions on wet interfaces
    wet = np.arange(NL)[None, :] <= np.asarray(nlev)[:, None]
    S2, N2 = np.asarray(x["Ssqr"]), np.asarray(x["Nsqr"])
    Pr = np.clip(6.6 * N2 / np.maximum(S2, 1e-12), 1.0, 10.0)
    np.testing.assert_allclose(np.asarray(reg["Pr"])[wet], Pr[wet], rtol=1e-14)
    Fd = KW["dt"] * np.asarray(reg["dKm_dtke"]) * np.abs(S2 - N2 / Pr)
    np.testing.assert_allclose(F[wet], Fd[wet], rtol=1e-14, atol=0.0)
    # a logger: no derivative flows through it
    g = jax.grad(lambda t: jnp.sum(_regimes(dict(x, tke_old=t), nlev)["F"]))(x["tke_old"])
    assert np.all(np.asarray(g) == 0.0)
