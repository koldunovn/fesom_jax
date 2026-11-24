"""``SurfaceForcing.step_forcing`` rolls the JRA reader to the year of the date it is asked for. The reader holds one
year and clamps a date outside it to its first or last record, so a forcing built for the run's start year and stepped
at a later year (the REcoM gradient and column-dump drivers on a year-2000 restart, plan 20261105, 2026-11-24)
returned the 31 December record at every step. A stub reader, no data.
"""

from __future__ import annotations

import numpy as np

from fesom_jax import surface_forcing as sf
from fesom_jax.jra55 import JRAFields


class _Reader:
    def __init__(self, year):
        self.year, self.rolls = year, []

    def reopen_year(self, year):
        self.rolls.append(year)
        self.year = year
        return self

    def step(self, year, day, sec, *, point_slope=False):
        assert year == self.year, "stepped outside the year the reader holds"
        v = np.full(3, float(year) + day / 1000.0)
        return JRAFields(u_wind=v, v_wind=v, shum=v, shortwave=v, longwave=v, Tair=v, prec_rain=v, prec_snow=v,
                         press_air=None)


class _SSS:
    def month(self, m):
        return np.zeros(3)


def _forcing(year):
    f = object.__new__(sf.SurfaceForcing)
    f.jra, f.sss, f.chl_clim, f.bgc, f.point_slope = _Reader(year), _SSS(), np.zeros((12, 3)), None, False
    return f


def test_step_forcing_rolls_the_reader_year():
    f = _forcing(1998)
    assert float(f.step_forcing(1998, 10, 0.0, 1, xp=np).shortwave[0]) == 1998.010
    assert f.jra.rolls == []                                   # the held year: nothing reopened
    assert float(f.step_forcing(2000, 183, 43200.0, 7, xp=np).shortwave[0]) == 2000.183
    assert f.jra.rolls == [2000]
    # a stack across a year boundary rolls once, in order
    st = f.stack([(2000, 365, 0.0, 12), (2000, 366, 0.0, 12), (2001, 1, 0.0, 1), (2001, 2, 0.0, 1)], xp=np)
    assert f.jra.rolls == [2000, 2001]
    assert np.allclose(st.shortwave[:, 0], [2000.365, 2000.366, 2001.001, 2001.002])
