"""Stage 1 pieces beyond the G2 dump gate (``test_recom_sms.py``): the refresh schedule, the solar angle, the
alkalinity restoring reduction, the cavity guard, ``dt_bgc`` as an argument, and a gradient check with respect
to REcoM parameters through one stage-1 step (central finite differences, from a dumped bloom state)."""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from fesom_jax.recom import forcing as F                                         # noqa: E402
from fesom_jax.recom import params as P                                          # noqa: E402
from fesom_jax.recom.sms import check_no_cavities, krill_resp, refresh_periods  # noqa: E402
from fesom_jax.recom.stage1 import recom_sources_sinks                           # noqa: E402
from fesom_jax.tests import test_recom_sms as G                                  # noqa: E402


def test_refresh_periods():
    assert refresh_periods(2700.0 / 86400.0) == (224, 960)            # pi, step_per_day 32
    assert refresh_periods(2 * 2700.0 / 86400.0) == (112, 480)        # BGC every 2nd step
    assert refresh_periods(2400.0 / 86400.0) == (252, 1080)           # bin_2p3z2d, 36 steps/day
    a, b = refresh_periods(jnp.asarray(2700.0 / 86400.0))
    assert int(a) == 224 and int(b) == 960


def test_solar_angle():
    # the last day of the year is year fraction 0 (mod(daynew, ndpyr))
    assert float(F.solar_declination(365, 365)) == pytest.approx(float(F.solar_declination(0, 365)), abs=0)
    lat = jnp.deg2rad(jnp.array([-60.0, 0.0, 45.0]))
    c = np.asarray(F.cobeta(172, 365, lat))
    assert np.all((c > 0.6) & (c <= 1.0))                             # refracted: cos >= sqrt(1 - 1/1.33^2)
    assert c[2] > c[0]                                                # northern summer


def test_krill_resp_branches():
    lat = jnp.array([-1.0, 1.0])
    r = lambda d: np.asarray(krill_resp(d, lat))
    assert np.allclose(r(1), [0.0, -0.5]) and np.allclose(r(200), [-0.5, 0.0])
    assert np.allclose(r(300), [0.0, 57.0 / 18.0 - 300.0 / 90.0]) and np.allclose(r(340), [0.0, -0.5])


def test_bio_fluxes_global_mean_removed():
    rng = np.random.default_rng(0)
    alk = 2300 + rng.random(50)
    area = rng.random(50) * 1e9
    r = F.bio_fluxes(alk, jnp.full(50, 2350.0), area, area.sum(), 3.2e-7)
    assert abs(float(jnp.sum(r * area))) < 1e-12 * float(jnp.sum(jnp.abs(r) * area))
    calls = []
    r2 = F.bio_fluxes(alk, jnp.full(50, 2350.0), area, area.sum(), 3.2e-7,
                      reduce_sum=lambda x: calls.append(1) or jnp.sum(x))
    assert calls == [1] and np.array_equal(np.asarray(r), np.asarray(r2))


def test_no_cavities_guard():
    check_no_cavities(np.ones(4, int))
    with pytest.raises(NotImplementedError):
        check_no_cavities(np.array([1, 2, 1]))


@pytest.mark.skipif(not (G.REF / "pi_30d_jan" / "dumps").is_dir(), reason="REcoM reference dumps absent")
def test_dt_bgc_and_gradient():
    f, meta = G._load("pi_30d_jan", 960)
    x, _ = G._inputs(f, meta)
    _, p0 = P.load_namelist(G.NML)
    names = ("res_phy", "alfa", "k_din")    # (grazing is negligible after 30 days from the tiny cold start)

    def npp(theta, dt_bgc):
        p = p0._replace(**dict(zip(names, theta)))
        r = recom_sources_sinks(x["bgc"], x["carb"], x["benthos"], x["alk_surf"], x["temp"], x["salt"],
                                x["geom"], x["inputs"], dt_bgc, p, n_iter=20)
        npp = jnp.sum((r.sms.diag2d["NPPn"] + r.sms.diag2d["NPPd"]) * x["geom"].area_top)
        # the PhyC source-minus-sink inventory (the increment, without the large state offset)
        phyc = jnp.sum(r.sms.sms[..., 4] * x["geom"].hnode * x["geom"].area_top[:, None])
        return npp, phyc

    theta = jnp.array([getattr(p0, n) for n in names])
    j1 = jax.jit(lambda t: npp(t, 2700.0))
    j2 = jax.jit(lambda t: npp(t, 5400.0))
    # dt_bgc enters as dt_b: the NPP rate [per day] does not depend on it, the updated tracers do
    (n1, c1), (n2, c2) = j1(theta), j2(theta)
    assert float(n2) == pytest.approx(float(n1), rel=1e-12) and float(c2) != float(c1)
    jc = jax.jit(lambda t: npp(t, 2700.0)[1])
    g = np.asarray(jax.jit(jax.grad(lambda t: npp(t, 2700.0)[1]))(theta))
    assert np.all(np.isfinite(g)) and np.all(g != 0)
    for i in range(len(names)):
        h = 1e-4 * float(theta[i])
        e = jnp.zeros(len(names)).at[i].set(h)
        fd = (float(jc(theta + e)) - float(jc(theta - e))) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-4), names[i]
