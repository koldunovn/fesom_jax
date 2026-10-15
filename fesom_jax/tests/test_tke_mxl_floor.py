"""``TkeConfig.mxl_nsqr_floor``: the N² floor of the TKE mixing length (idemix-jax patch 0009).

The column core computes the unconstrained mixing length ``√2·√tke/√max(floor, N²)``
(``cvmix_tke.F90:671`` ``max(1d-12,Nsqr)``). The floor is a static ``TkeConfig`` field whose
default ``1e-12`` is the Fortran/C literal: the default program is bitwise the one before the
field existed. A larger floor caps the length in weakly stratified water (a research knob that
changes the forward; it departs from FESOM).
"""

import numpy as np
import pytest

import jax.numpy as jnp

from fesom_jax import cvmix_tke
from fesom_jax.tke import TkeConfig

RNG = np.random.default_rng(20261015)
KW = dict(c_k=0.1, c_eps=0.7, cd=3.75, alpha_tke=30.0, mxl_min=1e-8, tke_min=1e-6,
          kappaM_max=100.0, dt=1800.0)


def _columns(N=16, nl=20):
    """Columns whose interior N² straddles 1e-8 (some between 1e-12 and 1e-8, some above)."""
    nlev = RNG.integers(nl // 2, nl - 1, N).astype(np.int32)
    k = np.arange(nl)[None, :]
    interior = (k >= 1) & (k <= nlev[:, None] - 1)
    Nsqr = np.where(interior, 10.0 ** RNG.uniform(-11.0, -5.0, (N, nl)), 0.0)
    Ssqr = np.where(interior, 10.0 ** RNG.uniform(-9.0, -5.0, (N, nl)), 0.0)
    tke = 10.0 ** RNG.uniform(-6.0, -3.0, (N, nl)) * (k <= nlev[:, None])
    dzw = np.full((N, nl), 25.0)
    dzt = np.full((N, nl), 25.0)
    forc = 10.0 ** RNG.uniform(-6.0, -4.0, N)
    return tuple(jnp.asarray(x) for x in (tke, Ssqr, Nsqr, dzw, dzt, forc, nlev)), Nsqr, interior


def test_default_is_the_fortran_literal_and_bitwise_unchanged():
    """The field defaults to 1e-12, and passing it explicitly or not gives the same bits (the
    diagnostics included)."""
    assert TkeConfig().mxl_nsqr_floor == 1.0e-12
    assert TkeConfig._fields[-1] == "mxl_nsqr_floor"        # positional order unchanged
    args, _, _ = _columns()
    for wd in (False, True):
        a = cvmix_tke.integrate_tke_column(*args, **KW, with_diags=wd)
        b = cvmix_tke.integrate_tke_column(*args, **KW, with_diags=wd, mxl_nsqr_floor=1e-12)
        la = a[:3] + (tuple(a[3].values()) if wd else ())
        lb = b[:3] + (tuple(b[3].values()) if wd else ())
        for x, y in zip(la, lb):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_raised_floor_caps_the_unconstrained_length_where_n2_is_below_it():
    """``mixing_length`` with floor 1e-8: the unconstrained length of an interior interface is
    ``√2·√tke/√max(1e-8, N²)``, so it changes exactly where N² < 1e-8 and nowhere else before
    the wall constraints; through the column, KappaM changes only in columns that have such an
    interface."""
    (tke, Ssqr, Nsqr, dzw, dzt, forc, nlev), N2, interior = _columns()
    nl = tke.shape[1]
    k = jnp.arange(nl)[None, :]
    nlc = jnp.asarray(nlev)[:, None]
    masks = dict(is_surf=(k == 0), is_bot=(k == nlc), is_interior=(k >= 1) & (k <= nlc - 1),
                 is_wet_iface=(k <= nlc))
    # no wall constraint: a huge dzw leaves the unconstrained length (floored at mxl_min)
    big = jnp.full_like(dzw, 1e30)
    m12, sq = cvmix_tke.mixing_length(tke, Nsqr, big, **masks, mxl_min=1e-8)
    m8, _ = cvmix_tke.mixing_length(tke, Nsqr, big, **masks, mxl_min=1e-8, mxl_nsqr_floor=1e-8)
    want = np.where(interior, np.sqrt(2.0) * np.asarray(sq) / np.sqrt(np.maximum(1e-8, N2)), 0.0)
    want = np.where(interior, np.maximum(want, 1e-8), np.asarray(m12))
    np.testing.assert_allclose(np.asarray(m8), want, rtol=1e-14, atol=0.0)
    low = interior & (N2 < 1e-8)
    assert low.any() and (interior & (N2 >= 1e-8)).any()
    assert np.all(np.asarray(m8)[low] < np.asarray(m12)[low])
    np.testing.assert_array_equal(np.asarray(m8)[~low], np.asarray(m12)[~low])

    _, KM12, _ = cvmix_tke.integrate_tke_column(tke, Ssqr, Nsqr, dzw, dzt, forc, nlev, **KW)
    _, KM8, _ = cvmix_tke.integrate_tke_column(tke, Ssqr, Nsqr, dzw, dzt, forc, nlev, **KW,
                                               mxl_nsqr_floor=1e-8)
    changed = np.any(np.asarray(KM8) != np.asarray(KM12), axis=1)
    assert changed.any()
    assert not np.any(changed & ~low.any(axis=1))


@pytest.mark.parametrize("bad", [0.0, -1e-12, float("nan"), float("inf")])
def test_validate_refuses_a_bad_floor(bad):
    with pytest.raises(ValueError, match="mxl_nsqr_floor"):
        TkeConfig(mxl_nsqr_floor=bad).validate()
    assert TkeConfig(mxl_nsqr_floor=1e-8).validate().mxl_nsqr_floor == 1e-8


@pytest.mark.parametrize("floor, n2_star", [(1e-8, 1e-9), (1e-12, 1e-8), (1e-8, 1e-6)])
def test_floor_sits_inside_the_mixing_length_cut(floor, n2_star):
    """The floor replaces the 1e-12 of ``√max(1e-12, n2_ml)`` (``cvmix_tke.F90:671``), where
    ``n2_ml`` is the N² after the mixing-length cut (``AdjointConfig.tke_ml_cut`` = N²*): the
    unconstrained length then has no N² derivative where N² < max(floor, N²*) (the floor's max,
    or the cut) and the exact one, -mxl0/(2N²), where N² >= max(floor, N²*)."""
    import jax
    from fesom_jax.adjoint_stab import AdjointConfig
    (tke, Ssqr, Nsqr, dzw, dzt, forc, nlev), N2, interior = _columns()
    nl = tke.shape[1]
    k = jnp.arange(nl)[None, :]
    nlc = jnp.asarray(nlev)[:, None]
    masks = dict(is_surf=(k == 0), is_bot=(k == nlc), is_interior=(k >= 1) & (k <= nlc - 1),
                 is_wet_iface=(k <= nlc))
    big = jnp.full_like(dzw, 1e30)                      # no wall constraint: mxl = max(mxl0, mxl_min)
    adj = AdjointConfig(tke_ml_cut=n2_star).validate().tke_options()

    def mxl(n2):
        return cvmix_tke.mixing_length(tke, n2, big, **masks, mxl_min=1e-8,
                                       ml_n2_star=adj.ml_n2_star, mxl_nsqr_floor=floor)[0]
    m, dm = jax.jvp(mxl, (Nsqr,), (jnp.ones_like(Nsqr),))
    m, dm = np.asarray(m), np.asarray(dm)
    low = interior & (N2 < max(floor, n2_star))
    free = interior & ~low & (m > 1e-8)
    assert low.any() and free.any()
    assert not np.any(dm[low])
    np.testing.assert_allclose(dm[free], -m[free] / (2.0 * N2[free]), rtol=1e-12)
