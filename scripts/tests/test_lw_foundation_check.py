"""Long-window Task A1 — CPU sanity test for the foundation drift-metric reducer.

Pure-numpy (no JAX): the area/volume-weighted reducer is finite, mask-safe (non-finite
values excluded as below-bottom/land), exact on analytic cases, and the annual/slope/
deceleration helpers behave. Token: LW_DRIFT_SEAM_OK. Runs standalone (`pytest scripts/tests/`),
separate from the JAX suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/ on the path
import core2_lw_foundation_check as fc   # noqa: E402


# ---- weighted_mean: area/volume-weighted, finite, mask-safe -------------------------------
def test_weighted_mean_uniform_field_returns_that_value():
    # a constant field with arbitrary positive weights → exactly the constant
    v = np.full(50, 3.7)
    w = np.abs(np.linspace(0.1, 9.0, 50))
    assert fc.weighted_mean(v, w) == pytest.approx(3.7, abs=1e-12)


def test_weighted_mean_is_area_weighted_not_plain_mean():
    # value 0 with weight 1, value 10 with weight 3 → 7.5, NOT the unweighted 5
    v = np.array([0.0, 10.0])
    w = np.array([1.0, 3.0])
    assert fc.weighted_mean(v, w) == pytest.approx(7.5, abs=1e-12)


def test_weighted_mean_excludes_nan_values():
    # NaN entries (below-bottom / land) are masked out, weighting renormalizes over valid
    v = np.array([2.0, np.nan, 4.0, np.nan])
    w = np.array([1.0, 5.0, 1.0, 9.0])
    out = fc.weighted_mean(v, w)
    assert np.isfinite(out)
    assert out == pytest.approx(3.0, abs=1e-12)   # mean of 2 and 4 (equal weight)


def test_weighted_mean_ignores_nonpositive_weights():
    v = np.array([1.0, 100.0, 5.0])
    w = np.array([2.0, 0.0, 2.0])                  # zero weight drops the 100
    assert fc.weighted_mean(v, w) == pytest.approx(3.0, abs=1e-12)


def test_weighted_mean_empty_is_finite_zero():
    # all-masked input → finite 0.0 (the project masked-reduction contract), never nan/inf
    out = fc.weighted_mean(np.array([np.nan, np.nan]), np.array([1.0, 1.0]))
    assert np.isfinite(out) and out == 0.0
    out2 = fc.weighted_mean(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert np.isfinite(out2) and out2 == 0.0


def test_weighted_mean_2d_volume_weighting():
    # a (nz, nod) layer field with a known vertical gradient + per-column areas
    field = np.array([[10.0, 10.0], [20.0, 20.0]])   # 2 levels, 2 columns
    w = np.array([[1.0, 1.0], [3.0, 3.0]])           # deep layer 3× thicker
    # (10*1+10*1+20*3+20*3)/(1+1+3+3) = (20+120)/8 = 17.5
    assert fc.weighted_mean(field, w) == pytest.approx(17.5, abs=1e-12)


# ---- weighted_total: a sum, not a mean (sea-ice area/volume) -------------------------------
def test_weighted_total_is_sum():
    a_ice = np.array([0.5, 1.0, 0.0])
    area = np.array([2.0, 4.0, 8.0])
    assert fc.weighted_total(a_ice, area) == pytest.approx(0.5 * 2 + 1.0 * 4, abs=1e-12)


def test_weighted_total_skips_nan():
    out = fc.weighted_total(np.array([1.0, np.nan, 2.0]), np.array([1.0, 9.0, 1.0]))
    assert np.isfinite(out) and out == pytest.approx(3.0, abs=1e-12)


# ---- annual_means / slope / decel ---------------------------------------------------------
def test_annual_means_reduces_months_to_years():
    monthly = np.concatenate([np.full(12, 5.0), np.full(12, 8.0)])
    ann = fc.annual_means(monthly)
    assert ann.tolist() == pytest.approx([5.0, 8.0])


def test_lin_slope_per_year_recovers_known_trend():
    # +0.5 per year over 3 years of monthly samples
    t = np.arange(36) / 12.0
    monthly = 1.0 + 0.5 * t
    assert fc.lin_slope_per_year(monthly) == pytest.approx(0.5, abs=1e-9)


def test_decel_flags_leveling_vs_accelerating():
    # decelerating: year steps 1.0 then 0.3 → leveling
    lvl = np.concatenate([np.full(12, 0.0), np.full(12, 1.0), np.full(12, 1.3)])
    d = fc.decel(lvl)
    assert d["decelerating"] is True
    assert d["drift_y1y2"] == pytest.approx(1.0)
    assert d["drift_y2y3"] == pytest.approx(0.3)
    # accelerating: steps 0.3 then 1.0 → runaway
    run = np.concatenate([np.full(12, 0.0), np.full(12, 0.3), np.full(12, 1.3)])
    assert fc.decel(run)["decelerating"] is False


def test_decel_handles_short_series():
    d = fc.decel(np.full(12, 1.0))   # only 1 year
    assert d["decelerating"] is None and np.isfinite(d["lin_slope_per_year"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
