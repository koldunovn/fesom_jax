"""The forcing's mid-step instant under a dt ramp (``run_config.DtRamp``).

FESOM2 interpolates the forcing to ``timenew - dt/2`` with the dt of the step; fesom-jax adds ``dt/2`` to the step
start (drift fix 2). A run with a ramp (the CORE2 runs from the dt-1800 v3 restart: ``dt: 1800`` +
``dt_ramp: {after_step: 561024, dt: 2700}``) takes the steps after the switch with the ramp dt, so their offset is
half of it (:func:`fesom_jax.surface_forcing.set_midstep`); before, every step used half the base dt.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fesom_jax import jra55
from fesom_jax.run import _chunk_dates
from fesom_jax.run_config import DtRamp
from fesom_jax.surface_forcing import set_midstep

RAMP = DtRamp(after_step=48 * 10, dt=2700.0)         # 1800 s for 10 days from 1958-01-01, then 2700 s


def _reader_like():
    return SimpleNamespace(midstep_days=0.0, midstep_switch=None, midstep_days_after=0.0)


def test_midstep_offsets():
    r = _reader_like()
    set_midstep(r, 1800.0, True)
    assert r.midstep_days == 900.0 / 86400.0 and r.midstep_switch is None
    off = lambda d: jra55.JRA55Reader._midstep(r, *d[:3])           # noqa: E731
    assert off((1958, 40, 0.0)) == 900.0 / 86400.0
    r = _reader_like()
    set_midstep(r, 1800.0, True, RAMP, 1958)
    assert r.midstep_switch == (1958, 11, 0.0)
    dates = _chunk_dates(1958, 1800.0, 478, 4, RAMP)                 # two steps before, two after the switch
    assert [round(off(d) * 86400.0, 9) for d in dates] == [900.0, 900.0, 1350.0, 1350.0]
    assert off((1959, 1, 0.0)) == 1350.0 / 86400.0                   # a later year
    r = _reader_like()
    set_midstep(r, 1800.0, False, RAMP, 1958)                        # the legacy step-start evaluation stays 0
    assert r.midstep_days == 0.0 and r.midstep_switch is None
    with pytest.raises(ValueError, match="year0"):
        set_midstep(_reader_like(), 1800.0, True, RAMP)


@pytest.mark.skipif(not Path(jra55.DEFAULT_JRA_DIR).is_dir(), reason="JRA55-do forcing not reachable")
def test_ramped_reader_equals_constant_dt_reader():
    """After the switch a ramped reader gives the fields of a reader built with the ramp dt, bitwise; before it,
    those of the base-dt reader."""
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh()
    readers = {}
    for name, dt, ramp in (("ramp", 1800.0, RAMP), ("d1800", 1800.0, None), ("d2700", 2700.0, None)):
        r = jra55.JRA55Reader(mesh, 1958)
        set_midstep(r, dt, True, ramp, 1958)
        readers[name] = r
    for d, ref in [(d, "d1800") for d in _chunk_dates(1958, 1800.0, 476, 4, RAMP)] + \
                  [(d, "d2700") for d in _chunk_dates(1958, 1800.0, 480, 40, RAMP)]:
        a, b = readers["ramp"].step(*d[:3]), readers[ref].step(*d[:3])
        for f in dataclasses.fields(a):
            fa, fb = getattr(a, f.name), getattr(b, f.name)
            if fa is not None:
                np.testing.assert_array_equal(np.asarray(fa), np.asarray(fb), err_msg=f"{f.name} at {d}")
