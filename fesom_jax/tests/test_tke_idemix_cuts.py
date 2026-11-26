"""The IDEMIX-coupled TKE column under the derivative cuts (idemix-jax host patch 0001, as ported
onto the cut machinery of upstream c110cd8).

With IDEMIX on, ``integrate_tke_column`` caps the Richardson number by
``KappaM·N²/max(1e-12, alpha_c·E_iw²)`` (``cvmix_tke.F90:710-712``), so ``E_iw`` and ``alpha_c``
reach the column outputs only through the Prandtl number. The cuts must keep the forward bitwise
on the coupled path, every Prandtl cut (``tke_pr_cut`` / ``tke_pr_ratio`` / ``tke_pr_raw_cut`` /
``tke.pr``) must wrap the CAPPED Prandtl number, ``tke_pr_mode="at"`` (whose straight-through
derivative is that of the uncapped Prandtl function) is refused, and the mixing-length cut leaves
the ``iw_diss`` forcing path alone.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from fesom_jax import cvmix_tke
from fesom_jax.adjoint_stab import AdjointConfig

KW = dict(c_k=0.1, c_eps=0.7, cd=3.75, alpha_tke=30.0, mxl_min=1e-8, tke_min=1e-6,
          kappaM_max=100.0, dt=1800.0)


def _columns(seed=20261028, N=24, nl=16):
    rng = np.random.default_rng(seed)
    nlev = rng.integers(nl // 2, nl - 1, N).astype(np.int32)
    k = np.arange(nl)[None, :]
    interior = (k >= 1) & (k <= nlev[:, None] - 1)
    wet = k <= nlev[:, None]
    Nsqr = np.where(interior, 10.0 ** rng.uniform(-9.0, -4.0, (N, nl)), 0.0)
    Ssqr = np.where(interior, 10.0 ** rng.uniform(-9.0, -4.0, (N, nl)), 0.0)
    tke = 10.0 ** rng.uniform(-6.0, -3.0, (N, nl)) * wet
    dzw = 10.0 + 40.0 * rng.uniform(size=(N, nl))
    dzt = 10.0 + 40.0 * rng.uniform(size=(N, nl))
    forc = 10.0 ** rng.uniform(-6.0, -4.0, N)
    iw_diss = 10.0 ** rng.uniform(-11.0, -8.0, (N, nl)) * wet
    alpha_c = 10.0 ** rng.uniform(-2.0, 2.0, (N, nl)) * wet
    E_iw = 10.0 ** rng.uniform(-5.0, -3.0, (N, nl)) * wet
    args = tuple(jnp.asarray(x) for x in (tke, Ssqr, Nsqr, dzw, dzt, forc, nlev))
    iw = {n: jnp.asarray(x) for n, x in
          (("iw_diss", iw_diss), ("alpha_c", alpha_c), ("E_iw", E_iw))}
    return args, iw, interior


def _raw(args):
    """An unsmoothed N² for the ratio / raw cuts: the column N² scaled by a factor in [-2, 4]."""
    rng = np.random.default_rng(3)
    n2 = np.asarray(args[2])
    return jnp.asarray(n2 * rng.uniform(-2.0, 4.0, n2.shape))


def _cap_binds(args, iw, interior):
    """Interior interfaces where the IDEMIX Rinum cap is below the plain Ri (the cap acts)."""
    Ssqr, Nsqr = np.asarray(args[1]), np.asarray(args[2])
    _, KM, _ = cvmix_tke.integrate_tke_column(*args, **KW, **iw)   # KappaM (wet interfaces)
    ri = Nsqr / np.maximum(Ssqr, 1e-12)
    cap = np.asarray(KM) * Nsqr / np.maximum(1e-12, np.asarray(iw["alpha_c"])
                                             * np.asarray(iw["E_iw"]) ** 2)
    return interior & (cap < ri) & (6.6 * cap > 1.0) & (6.6 * cap < 10.0)


ADJ = {
    "pr_cut": AdjointConfig(tke_pr_cut=1e-6),
    "pr_inf": AdjointConfig(tke_pr_cut=float("inf")),
    "pr_ratio": AdjointConfig(tke_pr_ratio=3.0),
    "pr_raw": AdjointConfig(tke_pr_raw_cut=1e-7),
    "kme": AdjointConfig(tke_kme_tke=float("inf")),
    "ml": AdjointConfig(tke_ml_cut=1e-6),
    # upstream d7e4e5d's cuts (b4130bc, 6fbccec): near-floor tke_old, buoyancy N² (cut / band)
    "floor": AdjointConfig(tke_floor_cut=1e-4),
    "n2": AdjointConfig(tke_n2_cut=1e-6),
    "n2_band": AdjointConfig(tke_n2_band=1e-6),
}


@pytest.mark.parametrize("name", sorted(ADJ))
def test_coupled_forward_bitwise_under_candidate_cuts(name):
    args, iw, _ = _columns()
    ref = cvmix_tke.integrate_tke_column(*args, **KW, **iw)
    got = cvmix_tke.integrate_tke_column(*args, **KW, **iw, n2_raw=_raw(args),
                                         adj=ADJ[name].validate().tke_options())
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("cut", ["tke.pr", "tke.n2", "tke.km_e", "tke.buoy"])
def test_coupled_forward_bitwise_under_diag_cuts(cut):
    args, iw, _ = _columns()
    ref = cvmix_tke.integrate_tke_column(*args, **KW, **iw)
    got = cvmix_tke.integrate_tke_column(*args, **KW, **iw, cuts=frozenset({cut}))
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _kh_jvp(args, iw, wrt, adj=None, cuts=frozenset(), out=2, n2_raw=None):
    """d KappaH · e for a unit tangent e on the input ``wrt`` (a key of ``iw`` or ``"Nsqr"``)."""
    rng = np.random.default_rng(7)

    def f(x):
        a = list(args)
        w = dict(iw)
        if wrt == "Nsqr":
            a[2] = x
        else:
            w[wrt] = x
        return cvmix_tke.integrate_tke_column(*a, **KW, **w, adj=adj, cuts=cuts,
                                              n2_raw=n2_raw)[out]
    x0 = args[2] if wrt == "Nsqr" else iw[wrt]
    t = jnp.asarray(rng.standard_normal(x0.shape)) * x0
    return np.asarray(jax.jvp(f, (x0,), (t,))[1])


def _kh_diag(args, iw, wrt, adj=None, n2_raw=None):
    """The diagonal d KappaH[n,k] / d wrt[n,k] (one JVP with a unit tangent: KappaH[n,k] depends
    on alpha_c / E_iw only at its own interface, through the capped Prandtl number)."""
    def f(x):
        w = dict(iw)
        w[wrt] = x
        return cvmix_tke.integrate_tke_column(*args, **KW, **w, adj=adj, n2_raw=n2_raw)[2]
    x0 = iw[wrt]
    t = jnp.ones_like(x0)
    return np.asarray(jax.jvp(f, (x0,), (t,))[1])


def test_cap_binds_in_the_fixture():
    args, iw, interior = _columns()
    b = _cap_binds(args, iw, interior)
    assert b.any() and (interior & ~b).any()


@pytest.mark.parametrize("wrt", ["E_iw", "alpha_c"])
def test_coupled_pr_cut_wraps_the_cap(wrt):
    """``E_iw`` and ``alpha_c`` reach KappaH only through the capped Prandtl number: the Prandtl
    cut everywhere (``tke_pr_cut=inf`` or the diagnostic ``tke.pr``) leaves no such derivative,
    the masked cut leaves it only where N² >= N²*."""
    args, iw, _ = _columns()
    assert np.abs(_kh_jvp(args, iw, wrt)).max() > 0.0
    inf = AdjointConfig(tke_pr_cut=float("inf")).validate().tke_options()
    assert not np.any(_kh_jvp(args, iw, wrt, adj=inf))
    assert not np.any(_kh_jvp(args, iw, wrt, cuts=frozenset({"tke.pr"})))
    n2s = 1e-6
    masked = AdjointConfig(tke_pr_cut=n2s).validate().tke_options()
    got = _kh_jvp(args, iw, wrt, adj=masked)
    low = np.asarray(args[2]) < n2s
    assert not np.any(got[low])
    assert np.any(got[~low])


def test_coupled_dkh_dalpha_c_is_the_capped_formula():
    """Where the cap binds and 1 < Pr < 10: KappaH = KappaM/(6.6·KappaM·N²/(alpha_c·E_iw²)) =
    alpha_c·E_iw²/(6.6·N²), so d KappaH / d alpha_c = E_iw²/(6.6·N²) exactly."""
    args, iw, interior = _columns()
    b = _cap_binds(args, iw, interior)
    got = _kh_diag(args, iw, "alpha_c")
    want = np.asarray(iw["E_iw"]) ** 2 / (6.6 * np.where(b, np.asarray(args[2]), 1.0))
    np.testing.assert_allclose(got[b], want[b], rtol=1e-12)


@pytest.mark.parametrize("name, kw", [
    ("pr_inf", dict(tke_pr_cut=float("inf"))),
    ("pr_cut", dict(tke_pr_cut=1e-30)),          # N² >= N²* nowhere below: cuts nothing interior
    ("pr_ratio", dict(tke_pr_ratio=1e-30)),      # |raw| > R*·|N²| everywhere N² != 0: cuts all
    ("pr_raw", dict(tke_pr_raw_cut=1.0)),        # raw N² < 1 everywhere: cuts all
])
def test_each_pr_mask_wraps_the_capped_path(name, kw):
    """Each Prandtl mask acts on the capped Prandtl number: where it cuts, d KappaH / d alpha_c
    (which exists only through the cap) is 0; where it does not, it is the capped formula."""
    args, iw, interior = _columns()
    b = _cap_binds(args, iw, interior)
    adj = AdjointConfig(**kw).validate().tke_options()
    got = _kh_diag(args, iw, "alpha_c", adj=adj, n2_raw=_raw(args))
    want = np.asarray(iw["E_iw"]) ** 2 / (6.6 * np.where(b, np.asarray(args[2]), 1.0))
    if name == "pr_cut":
        np.testing.assert_allclose(got[b], want[b], rtol=1e-12)
    else:
        assert b.any() and not np.any(got[b])


def test_coupled_n2_cut_reaches_the_cap():
    """The buoyancy-N² cut (``tke_n2_cut``, upstream 6fbccec) acts on the N² the Rinum cap reads
    (``Nsqr_b``): with ``tke_n2_cut=inf`` no derivative of KappaH with respect to N² is left
    where the cap binds (there KappaH = alpha_c·E_iw²/(6.6·N²) depends on N² only through
    ``Nsqr_b``; KappaM, which also reads N² through the mixing length, cancels in KappaM/Pr up to
    rounding, so the cut derivative is zero to a relative 1e-9 of the uncut one)."""
    args, iw, interior = _columns()
    b = _cap_binds(args, iw, interior)
    def kh_diag_n2(adj):
        def f(x):
            a = list(args)
            a[2] = x
            return cvmix_tke.integrate_tke_column(*a, **KW, **iw, adj=adj)[2]
        x0 = args[2]
        return np.asarray(jax.jvp(f, (x0,), (jnp.ones_like(x0),))[1])
    ref = kh_diag_n2(None)
    assert np.all(ref[b] != 0.0)
    cut = kh_diag_n2(AdjointConfig(tke_n2_cut=float("inf")).validate().tke_options())
    assert np.all(np.abs(cut[b]) <= 1e-9 * np.abs(ref[b]))


def test_pr_mode_at_is_refused_under_coupling():
    args, iw, _ = _columns()
    adj = AdjointConfig(tke_pr_cut=1e-6, tke_pr_mode="at").validate().tke_options()
    with pytest.raises(ValueError, match="tke_pr_mode='at' is not supported with IDEMIX"):
        cvmix_tke.integrate_tke_column(*args, **KW, **iw, adj=adj)
    # the same setting without the IDEMIX inputs still runs (the only_tke path is unchanged)
    cvmix_tke.integrate_tke_column(*args, **KW, adj=adj)


def test_ml_cut_leaves_the_iw_diss_path():
    """The mixing-length cut acts on N² inside the mixing length only: on columns whose interior
    N² is everywhere >= N²* it cuts nothing, so d tke_new / d iw_diss (the forcing
    ``forc += iw_diss`` through the implicit solve; KappaM/KappaH of the step read tke_old and do
    not depend on iw_diss) is bitwise the uncut one."""
    out = 0
    args, iw, interior = _columns()
    n2s = 1e-6
    n2 = np.asarray(args[2]).copy()
    n2[:4] = np.where(interior[:4], np.maximum(n2[:4], 1e-5), 0.0)   # 4 columns with N² >= N²*
    args = args[:2] + (jnp.asarray(n2),) + args[3:]
    adj = AdjointConfig(tke_ml_cut=n2s).validate().tke_options()
    ref = _kh_jvp(args, iw, "iw_diss", out=out)
    got = _kh_jvp(args, iw, "iw_diss", adj=adj, out=out)
    n2 = np.asarray(args[2])
    above = np.all(np.where(interior, n2 >= n2s, True), axis=1)
    assert above.any() and (~above).any()
    np.testing.assert_array_equal(got[above], ref[above])
    assert np.abs(ref).max() > 0.0


def test_mixing_tke_refuses_idemix_with_cavity():
    """Upstream 4fe45cc's ``mixing_tke(cavity=True)`` shifts every TKE column up by
    ``ulevels_nod2D - 1``; the IDEMIX triple is on the global interface levels and is not shifted,
    so the combination is refused (ValueError) rather than run misaligned under the shelves. The
    same call with ``idemix=None`` still runs (upstream's cavity path), and ``cavity=False`` with
    the triple runs the coupled column."""
    from fesom_jax import tke as tkemod
    from fesom_jax.params import Params
    from fesom_jax.tests.test_tke_cuts_pi import _pi_driver
    mesh, a = _pi_driver()
    shape = (mesh.nod2D, mesh.nl)
    triple = (jnp.full(shape, 1e-10), jnp.full(shape, 1.0), jnp.full(shape, 1e-4))

    def run(**kw):
        return tkemod.mixing_tke(mesh, a["uv"], a["bvfreq"], a["tke"], a["stress_node_surf"],
                                 a["hnode"], tkemod.TkeConfig(), Params.defaults(), dt=1800.0,
                                 Z3d=a["Z3d"], **kw)

    with pytest.raises(ValueError, match="ice-shelf cavities"):
        run(cavity=True, idemix=triple)
    run(cavity=True)
    run(idemix=triple)


def test_interface_spacing_is_the_tke_dzt_bitwise():
    """``tke.interface_spacing`` (the IDEMIX column's ``dzt``) recomputes the interface masks and
    the layer-centre spacing itself; ``tke_column_inputs`` keeps its inline masks and calls the
    shared ``_dz_trr`` (so the TKE program is upstream's). On pi (no cavities) the two give the
    same ``dzt`` bitwise; on a cavity mesh they differ at the cavity nodes' surface interface
    unless ``mixing_tke(cavity=True)`` re-caps it (test_tke_interface_spacing.py)."""
    from fesom_jax import tke as tkemod
    from fesom_jax.tests.test_tke_cuts_pi import _pi_driver
    mesh, a = _pi_driver()
    d = tkemod.tke_column_inputs(mesh, a["uv"], a["bvfreq"], a["tke"], a["stress_node_surf"],
                                 a["hnode"], Z3d=a["Z3d"])
    got = tkemod.interface_spacing(mesh, a["hnode"], a["Z3d"])
    np.testing.assert_array_equal(np.asarray(got), np.asarray(d["dzt"]))
