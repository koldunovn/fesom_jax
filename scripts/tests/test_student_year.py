"""Pure-function tests for the Stage-1 feasibility pair (student_year.py / signal_eval.py).

Same scope rule as test_horizon_scan.py: only what these scripts can get wrong on their own —
date/mark bookkeeping and the loss-form usage (w_eta=0 path, teacher-hole exclusion). The
model run itself is covered by the driver pieces horizon_scan already validated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nn_eddy"))
import horizon_scan as hs  # noqa: E402
import signal_eval as se  # noqa: E402
import student_year as sy  # noqa: E402


def test_mult_tags_stable():
    assert sy.mult_tag(1.0 / 3.0) == "m0333"
    assert sy.mult_tag(1.0) == "m1000"
    assert sy.mult_tag(3.0) == "m3000"


def test_branch_start_dates_1958():
    # hand-checked against the 1958 (non-leap) calendar
    assert sy.branch_start_date(1958, 60) == (3, 2)
    assert sy.branch_start_date(1958, 150) == (5, 31)
    assert sy.branch_start_date(1958, 240) == (8, 29)
    assert sy.branch_start_date(1958, 330) == (11, 27)


def test_mark_index_absorbs_one_dt_offset():
    # teacher records sit one fine-dt SHORT of the round mark; both ramp (120 s) and
    # production (240 s) offsets must land on the same mark as the exact time
    assert se.mark_index(432000.0) == 1
    assert se.mark_index(431880.0) == 1          # dt=120 s stage
    assert se.mark_index(2 * 432000.0 - 240.0) == 2   # dt=240 s stage
    assert se.mark_index(73 * 432000.0 - 240.0) == 73  # last mark of the year


def _toy(nn=5, nl=3, seed=0):
    rng = np.random.default_rng(seed)
    T = jnp.asarray(rng.normal(10.0, 1.0, (nn, nl)))
    S = jnp.asarray(rng.normal(35.0, 0.1, (nn, nl)))
    wet = jnp.asarray(rng.random((nn, nl)) > 0.2, jnp.float64)
    area = jnp.asarray(rng.random((nn, nl)) + 0.5) * wet
    return T, S, wet, area


def test_misfit_weta0_is_pure_TS():
    """With w_eta=0 the misfit must equal the hand-built T,S sum — eta drops out exactly."""
    T, S, wet, area = _toy()
    Th, Sh, _, _ = _toy(seed=1)
    w_T = jnp.asarray([1.0, 2.0, 0.5])
    w_S = jnp.asarray([3.0, 1.0, 1.0])
    e_a = jnp.asarray(np.random.default_rng(2).normal(0, 1, T.shape[0]))
    e_b = jnp.asarray(np.random.default_rng(3).normal(0, 1, T.shape[0]))
    got = hs.stage1_misfit(T, S, e_a, Th, Sh, e_b, w_T, w_S, jnp.asarray(0.0), wet, area)
    aw = np.asarray(area * wet)
    den = aw.sum(axis=0)
    lev = den / den.sum()
    mseT = (aw * np.asarray((T - Th)) ** 2).sum(axis=0) / den
    mseS = (aw * np.asarray((S - Sh)) ** 2).sum(axis=0) / den
    want = (lev * (np.asarray(w_T) * mseT + np.asarray(w_S) * mseS)).sum()
    np.testing.assert_allclose(float(got), float(want), rtol=1e-12)


def test_teacher_hole_exclusion():
    """A cell masked out (teacher cell_area==0 ⇒ wet=0) must not feel ANY student value."""
    T, S, wet, area = _toy()
    Th, Sh, _, _ = _toy(seed=1)
    wet = np.asarray(wet).copy()
    wet[2, 1] = 0.0                       # the "hole"
    area = jnp.asarray(np.asarray(area) * wet)
    wet = jnp.asarray(wet)
    w = jnp.ones(T.shape[1])
    e = jnp.zeros(T.shape[0])
    base = float(hs.stage1_misfit(T, S, e, Th, Sh, e, w, w, jnp.asarray(0.0), wet, area))
    T2 = np.asarray(T).copy()
    T2[2, 1] = 1.0e6                      # junk where there is no target
    poked = float(hs.stage1_misfit(jnp.asarray(T2), S, e, Th, Sh, e, w, w,
                                   jnp.asarray(0.0), wet, area))
    assert base == poked
