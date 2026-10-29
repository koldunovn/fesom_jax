"""``forcing.repeat_year`` (FESOM2's ``y_perpetual``): the JRA55-do reader on a repeat-year climatology
(``{var}.clim61.nc``, a 365-day calendar) for every model year, and the run config's keys for it and for the
constant chlorophyll (``chl_const``). Needs the clim61 files of the REcoM spectral runs on Albedo; skips otherwise."""

from __future__ import annotations

import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fesom_jax import jra55  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.run_config import RunConfig  # noqa: E402

CLIM = Path(os.environ.get("FESOM_JRA_CLIM_DIR", "/albedo/work/user/anbros001/Maresys"))
need_clim = pytest.mark.skipif(not (CLIM / "uas.clim61.nc").is_file(), reason=f"no repeat-year JRA files in {CLIM}")


def test_config_keys_round_trip():
    cfg = RunConfig.from_dict(dict(forcing=dict(kind="core2", start_year=1998, jra_dir="/d", repeat_year="clim61",
                                                chl_const=0.1)))
    fp = cfg.forcing_paths()
    assert fp["jra_tag"] == "clim61" and fp["chl_const"] == 0.1 and fp["jra_dir"] == "/d"
    back = RunConfig.from_dict(cfg.to_dict())
    assert back.forcing_paths() == fp
    assert "jra_tag" not in RunConfig.from_dict(dict(forcing=dict(kind="core2", start_year=1958))).forcing_paths()


@need_clim
def test_repeat_year_reader():
    mesh = load_mesh()
    jra = jra55.JRA55Reader(mesh, 1958, CLIM, mslp=True, tag="clim61")
    f0 = jra.fields[0]
    assert f0.path.endswith("uas.clim61.nc") and f0.calendar not in jra55._GREG_CALS and f0.Ntime == 2920
    assert abs(f0.nc_time[0] - 365 * 1958) < 1.0              # the file's year rebased to the model year
    a = jra.step(1958, 1, 0.0).as_array()                     # (fields, nod2D)
    jra.reopen_year(1959)
    assert jra.fields[0].path.endswith("uas.clim61.nc") and abs(jra.fields[0].nc_time[0] - 365 * 1959) < 1.0
    b = jra.step(1959, 1, 0.0).as_array()
    assert np.array_equal(a, b)                                # the same day of every year reads the same data
    jra.reopen_year(1960)                                      # a leap model year: day 366 holds the last record
    c = jra.step(1960, 366, 43200.0).as_array()
    d = jra.step(1960, 365, 82800.0).as_array()
    assert np.all(np.isfinite(c))
    scale = np.max(np.abs(d), axis=1, keepdims=True)
    assert np.all(np.abs(c - d) <= 0.2 * scale + 1e-12)
    jra.close()
