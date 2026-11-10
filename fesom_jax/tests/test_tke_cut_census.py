"""The derivative-cut census ``cvmix_tke.cut_census`` on the hand-built synthetic columns of
``test_tke_cuts.py`` (no mesh; ``ci`` group).

Properties:

* every per-criterion count equals the criterion recomputed in numpy from the inputs
  (``N² < N²*``, ``|raw| > R·|N²|``, ``raw < raw*``, ``|lat| > phi*``, ``conc > a*``), over the
  wet interior interfaces of each region;
* ``<family>.any`` is the union of its criteria; every ``.live`` count is the part of its mask
  inside ``pr_band`` (``1 < 6.6·Ri < 10``) resp. ``ml_free`` (the free mixing length,
  ``N² > 1e-12``);
* ``adj=None`` counts only the interfaces; the census is forward-only (no derivative of it is
  needed or traced into the model) and the column core is unchanged by it.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import cvmix_tke
from fesom_jax.adjoint_stab import AdjointConfig
from fesom_jax.tests.test_tke_cuts import CONSTS, KW, NCOL, NL, synthetic_columns

SETTING = dict(tke_pr_cut=1e-6, tke_pr_ratio=100.0, tke_pr_raw_cut=-1e-6, tke_ml_cut=1e-6,
               tke_ml_ratio=100.0, tke_ml_raw_cut=-1e-6, tke_pr_lat_cut=70.0,
               tke_pr_ice_cut=0.15)


def _case():
    """Synthetic columns plus an unsmoothed N² that triggers the ratio and raw cuts on a few
    planted interfaces, column latitudes on both sides of 70° and an ice concentration."""
    cols = synthetic_columns()
    n2 = np.asarray(cols["Nsqr"])
    rng = np.random.default_rng(3)
    raw = n2 * (1.0 + 0.1 * rng.standard_normal(n2.shape))
    raw[1, 3] = 500.0 * abs(n2[1, 3])            # cancellation: |raw| > 100 |N²|
    raw[0, 5] = -3.0e-6                          # inverted raw N²
    raw[4, 7] = -5.0e-7                          # inverted, but above raw* = -1e-6
    raw = np.where(n2 == 0.0, 0.0, raw)
    lat = np.radians(np.linspace(-80.0, 80.0, NCOL))
    conc = np.linspace(0.0, 1.0, NCOL)
    return cols, raw, lat, conc


def _census(adj, cols, raw, lat, conc, regions=None):
    out = cvmix_tke.cut_census(
        cols["tke_old"], cols["Ssqr"], cols["Nsqr"], cols["dzw"], cols["dzt"],
        cols["forc_tke_surf"], cols["nlev"], adj=adj, **CONSTS, **KW,
        n2_raw=jnp.asarray(raw), ice_conc=jnp.asarray(conc), lat=jnp.asarray(lat),
        regions=regions)
    return {r: {k: int(v) for k, v in row.items()} for r, row in out.items()}


def _expected_masks(cols, raw, lat, conc):
    n2 = np.asarray(cols["Nsqr"])
    s = SETTING
    pr = {"small": n2 < s["tke_pr_cut"],
          "cancel": np.abs(raw) > s["tke_pr_ratio"] * np.abs(n2),
          "low": raw < s["tke_pr_raw_cut"],
          "ice": np.broadcast_to((conc > s["tke_pr_ice_cut"])[:, None], n2.shape),
          "lat": np.broadcast_to((np.abs(lat) > math.radians(s["tke_pr_lat_cut"]))[:, None],
                                 n2.shape)}
    ml = {"small": n2 < s["tke_ml_cut"],
          "cancel": np.abs(raw) > s["tke_ml_ratio"] * np.abs(n2),
          "low": raw < s["tke_ml_raw_cut"]}
    return pr, ml


def _interior(cols):
    k = np.arange(NL)[None, :]
    nlev = np.asarray(cols["nlev"])[:, None]
    return (k >= 1) & (k <= nlev - 1)


@pytest.fixture(scope="module")
def case():
    cols, raw, lat, conc = _case()
    adj = AdjointConfig(**SETTING).validate().tke_options()
    regions = {"global": jnp.ones(NCOL, bool),
               "north": jnp.asarray(lat > 0.0)}
    return cols, raw, lat, conc, adj, regions, _census(adj, cols, raw, lat, conc, regions)


def test_counts_equal_the_criteria(case):
    cols, raw, lat, conc, adj, regions, out = case
    interior = _interior(cols)
    pr, ml = _expected_masks(cols, raw, lat, conc)
    for rname, rmask in regions.items():
        r = interior & np.asarray(rmask)[:, None]
        row = out[rname]
        assert row["interior"] == int(r.sum())
        for fam, parts in (("pr", pr), ("ml", ml)):
            union = np.zeros_like(r)
            for name, m in parts.items():
                assert row[f"{fam}.{name}"] == int((r & m).sum()), (rname, fam, name)
                union |= m
            assert row[f"{fam}.any"] == int((r & union).sum()), (rname, fam)
    # the planted interfaces are counted (the fixture is not vacuous)
    g = out["global"]
    assert g["pr.cancel"] >= 1 and g["pr.low"] >= 1 and g["pr.lat"] > 0 and g["pr.ice"] > 0
    assert 0 < g["pr.small"] < g["interior"]


def test_live_counts_are_the_mask_inside_the_derivative_band(case):
    cols, raw, lat, conc, adj, regions, out = case
    interior = _interior(cols)
    n2, s2 = np.asarray(cols["Nsqr"]), np.asarray(cols["Ssqr"])
    x = cvmix_tke.TKE_C66 * (n2 / np.maximum(s2, 1e-12))
    band = interior & (x > 1.0) & (x < 10.0)
    reg = cvmix_tke.tke_regimes(*(cols[k] for k in ("tke_old", "Ssqr", "Nsqr", "dzw", "dzt",
                                                    "forc_tke_surf", "nlev")), **CONSTS, **KW)
    free = interior & (np.asarray(reg["regime"]) == 0) & (n2 > 1e-12)
    pr, ml = _expected_masks(cols, raw, lat, conc)
    g = out["global"]
    assert g["pr_band"] == int(band.sum()) and g["ml_free"] == int(free.sum())
    assert 0 < g["pr_band"] < g["interior"] and 0 < g["ml_free"] < g["interior"]
    for fam, parts, live in (("pr", pr, band), ("ml", ml, free)):
        for name, m in parts.items():
            assert g[f"{fam}.{name}.live"] == int((interior & m & live).sum()), (fam, name)
            assert g[f"{fam}.{name}.live"] <= g[f"{fam}.{name}"]


def test_no_adjoint_settings_count_only_the_interfaces():
    cols, raw, lat, conc = _case()
    out = _census(None, cols, raw, lat, conc)
    assert set(out) == {"global"}
    assert set(out["global"]) == {"interior", "pr_band", "ml_free"}
    assert out["global"]["interior"] == int(_interior(cols).sum())


def test_masks_are_the_ones_the_core_applies(case):
    """The core's mask builders are the census's: on the same inputs, ``n2_cut_masks`` /
    ``pr_region_masks`` give the criteria above, in the core's union order."""
    cols, raw, lat, conc, adj, regions, out = case
    pr, ml = _expected_masks(cols, raw, lat, conc)
    got = cvmix_tke.n2_cut_masks(cols["Nsqr"], jnp.asarray(raw), n2_star=adj.pr_n2_star,
                                 ratio=adj.pr_ratio, raw_cut=adj.pr_raw_cut)
    assert list(got) == ["small", "cancel", "low"]
    for k in got:
        np.testing.assert_array_equal(np.asarray(got[k]), pr[k])
    reg = cvmix_tke.pr_region_masks(adj, ice_conc=jnp.asarray(conc), lat=jnp.asarray(lat))
    assert list(reg) == ["ice", "lat"]
    for k in reg:
        np.testing.assert_array_equal(np.broadcast_to(np.asarray(reg[k]), pr[k].shape), pr[k])
    with pytest.raises(ValueError, match="need the unsmoothed"):
        cvmix_tke.n2_cut_masks(cols["Nsqr"], None, ratio=100.0)
    with pytest.raises(ValueError, match="column latitude"):
        cvmix_tke.pr_region_masks(adj, ice_conc=jnp.asarray(conc))
