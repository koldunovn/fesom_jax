"""REcoM's atmospheric CO2 time series (``constant_CO2`` off, FESOM2 ``gen_surface_forcing.F90:1986-2041``) and the
present-day DIC IC set.

* :class:`AtmCO2Series`: FESOM2's CO2-year formula (one cycle: the model year; two cycles), the values as stored;
* :meth:`BgcForcing.co2_atm`: ``AtmCO2(month)`` at the ``call recom`` clock (the last step of a month keeps its
  month, the last step of a year its year);
* host stack == on-device tables == local stack (``co2_atm`` is a replicated per-step scalar);
* the plumbing gate: pi, 3 steps of full REcoM from the twin's step 0 with a series of 278 ppm everywhere equal
  ``constant_CO2`` (``CO2_for_spinup`` = 278) bitwise; with 369 ppm the air-sea CO2 flux changes;
* the sharded runner hands ``co2_atm`` to ``step()`` on both forcing paths (pi ``dist_4``, 4 host devices);
* RunConfig: ``forcing.co2_file``; a relative namelist file raises; ``constant_CO2`` off and ``ic_set:
  core2_present`` validate.

CPU, double precision; the sharded test needs ``XLA_FLAGS=--xla_force_host_platform_device_count=4``. Compute node only.
"""

from __future__ import annotations

import dataclasses
import re

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fesom_jax import partit, shard_mesh, ssh, surface_forcing  # noqa: E402
from fesom_jax import integrate_sharded as ish  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.forcing_local import build_local_forcing  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.recom import ic, wiring  # noqa: E402
from fesom_jax.recom.config import RecomConfig  # noqa: E402
from fesom_jax.run_config import RunConfig  # noqa: E402
from fesom_jax.state import State  # noqa: E402
from fesom_jax.surface_forcing import AtmCO2Series, BgcForcing  # noqa: E402
from fesom_jax.tests import test_recom_step as TS  # noqa: E402
from fesom_jax.tests.test_recom_inputs import (DT, NML, PI_DIST, PI_DUST, YEAR, _dates, have_dist4,  # noqa: E402
                                               have_dust, have_jra)

REAL_CO2 = "/work/ab0246/a270092/input/fesom2/recom/MonthlyAtmCO2_gcb2023.nc"


def _co2_file(path, years, value=None):
    """A CO2 file in FESOM2's layout: ``AtmCO2_<year>(12)``, ``year + month / 100`` or the constant ``value``."""
    import netCDF4
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("month", 12)
        for y in years:
            v = ds.createVariable(f"AtmCO2_{y}", "f8", ("month",))
            v[:] = np.full(12, float(value)) if value is not None else y + np.arange(1, 13) / 100.0
    return path


def _series(path, **kw):
    return AtmCO2Series(path, first_year=1958, last_year=2019, **kw)


def test_series_cycle_years_and_values(tmp_path):
    s = _series(_co2_file(tmp_path / "co2.nc", range(1890, 2021)))
    assert [s.co2_year(y) for y in (1958, 2000, 2019)] == [1958, 2000, 2019]        # one cycle: the model year
    assert s.value(2000, 1) == 2000 + 1 / 100.0 and s.value(2000, 12) == 2000 + 12 / 100.0
    # two cycles of the 62 forcing years 1958-2019 (gen_surface_forcing.F90:2001-2004): the first takes 1896-1957
    s1, s2 = _series(s.path, n_cycles=2, cycle=1), _series(s.path, n_cycles=2, cycle=2)
    assert (s1.co2_year(1958), s1.co2_year(2019)) == (1896, 1957)
    assert (s2.co2_year(1958), s2.co2_year(2019)) == (1958, 2019)
    with pytest.raises(ValueError, match="AtmCO2_2030"):
        s.value(2030, 1)


@pytest.mark.skipif(not __import__("os").path.isfile(REAL_CO2), reason=f"{REAL_CO2} absent")
def test_real_file_reads():
    s = _series(REAL_CO2)
    v = s.year_values(2000)
    assert v.shape == (12,) and v.dtype == np.float64 and abs(v[0] - 369.39) < 1e-3
    assert 405.0 < s.value(2019, 6) < 415.0


def test_co2_follows_the_recom_clock(tmp_path):
    """``AtmCO2(month)`` with FESOM2's ``month`` at the recom call (the step end): the last step of 31 January ends at
    24:00 of that day and keeps January (the held dust already switches there, the CO2 does not); the last step of
    a year keeps its year."""
    bf = BgcForcing(np.zeros((12, 3)), DT, (2000, 1, 0.0), co2=_series(_co2_file(tmp_path / "co2.nc", (2000, 2001))))
    for start, want in [((2000, 31, 86400.0 - DT), 2000.01), ((2000, 32, 0.0), 2000.02),
                        ((2000, 366, 86400.0 - DT), 2000.12), ((2001, 1, 0.0), 2001.01)]:
        c, held = bf.inputs(*start)
        y, m = divmod(round(want * 100), 100)
        assert bf.co2_atm(c) == y + m / 100.0, (start, c)
    assert bf.inputs(2000, 31, 86400.0 - DT)[1] == 2          # the dust switch on the same step, for contrast


@have_jra
@have_dust
@pytest.mark.skipif(not (PI_DIST / "dist_4").is_dir(), reason="pi dist_4 absent")
def test_host_local_tables_agree(tmp_path):
    mesh = load_mesh()
    co2 = _series(_co2_file(tmp_path / "co2.nc", (YEAR,)))
    dates = _dates((YEAR, 31, 86400.0 - 2 * DT), 4)                  # the last two steps of January + 1 Feb
    host = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, bgc_dust_path=str(PI_DUST), bgc_co2=co2)
    st = host.stack(dates, xp=np)
    want = np.array([YEAR + 0.01, YEAR + 0.01, YEAR + 0.02, YEAR + 0.02])
    np.testing.assert_array_equal(st.co2_atm, want)
    assert st.co2_atm.dtype == np.float64
    dev = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, bgc_dust_path=str(PI_DUST), bgc_co2=co2)
    tables, sched = dev.stack_tables(dates, nb=4)
    np.testing.assert_array_equal(sched.co2_atm, want)
    const = surface_forcing.forcing_device_const(dev)
    for i in range(len(dates)):
        got = surface_forcing.combine_step_forcing(tables, sched.rdate[i], sched.fidx[i], sched.midx[i], const,
                                                   didx=sched.didx[i], daynew=sched.daynew[i],
                                                   ndpyr=sched.ndpyr[i], co2_atm=sched.co2_atm[i])
        assert float(got.co2_atm) == want[i]
    part = partit.read_partition(PI_DIST, 4)
    loc = build_local_forcing(mesh, YEAR, part, 4, local_parts=list(range(4)), static=host.static, dt=DT,
                              bgc_dust_path=str(PI_DUST), bgc_co2=co2)
    np.testing.assert_array_equal(loc.stack_partitioned(dates).co2_atm, want)
    np.testing.assert_array_equal(loc.stack_tables_partitioned(dates, nb=4)[1].co2_atm, want)
    np.testing.assert_array_equal(shard_mesh.partition_step_forcing(st, part).co2_atm, want)


@TS.need_twin
def test_constant_series_is_constant_co2(tmp_path):
    """The plumbing gate: 3 steps of the pi twin with the series at 278 ppm everywhere == ``CO2_for_spinup`` 278
    (``constant_CO2``), bitwise in every field; at 369 ppm the air-sea CO2 flux of step 1 changes."""
    fx = TS._setup()
    assert float(fx["cfg"].recom.load()[1].CO2_for_spinup) == 278.0
    mesh = fx["mesh"]

    def forcing(value):
        co2 = _series(_co2_file(tmp_path / f"co2_{value}.nc", (1958,), value=value))
        return surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(fx["st"].T[:, 0]), dt=fx["cfg"].dt,
                                                     mslp=True, bgc_dust_path=fx["cfg"].bgc_dust_path(), bgc_co2=co2,
                                                     **fx["cfg"].forcing_paths())

    ref = TS._run(fx, 3)
    got = TS._run(dict(fx, cf=forcing(278.0)), 3)
    for f in dataclasses.fields(State):
        a, b = getattr(ref[-1], f.name), getattr(got[-1], f.name)
        if a is None:
            assert b is None, f.name
            continue
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a), err_msg=f.name)
    hi = TS._run(dict(fx, cf=forcing(369.0)), 1)
    k = wiring.DIAG2D.index("co2flux")
    wet = np.asarray(mesh.nlevels_nod2D) > 1
    d = np.asarray(hi[-1].bgc_diag)[:mesh.nod2D, k][wet] - np.asarray(ref[1].bgc_diag)[:mesh.nod2D, k][wet]
    print(f"CO2 369 vs 278 ppm, step 1: CO2 flux change mean {d.mean():.4e}, max {np.abs(d).max():.4e}")
    assert np.abs(d).max() > 0.0


@have_jra
@have_dust
@have_dist4
def test_sharded_co2_reaches_step(tmp_path, monkeypatch):
    """pi ``dist_4``, 4 host devices, three forced steps across the month end: inside ``step()`` the sharded runner
    hands ``co2_atm`` with the host stack's values, on the per-step stack and on the tables path."""
    mesh = load_mesh()
    co2 = _series(_co2_file(tmp_path / "co2.nc", (YEAR,)))
    dates = _dates((YEAR, 31, 86400.0 - 2 * DT), 3)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, bgc_dust_path=str(PI_DUST), bgc_co2=co2)
    st_host = cf.stack(dates, xp=np)
    orig = stepmod.step

    def probe(state, *a, step_forcing=None, **kw):
        out = orig(state, *a, step_forcing=step_forcing, **kw)
        return dataclasses.replace(out, hbar_old=jnp.full_like(out.hbar_old, step_forcing.co2_atm))

    monkeypatch.setattr(stepmod, "step", probe)
    npes = 4
    part = partit.read_partition(PI_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(jax.tree.map(np.asarray, State.rest(mesh)), part)
    sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=DT), part)
    fs_p = shard_mesh.partition_forcing_static(cf.static, part)
    stress_p = jnp.zeros((npes, sm.Lmax["elem"], 2))
    for n in (2, 3):                    # the probe holds the last step's value: January, then February
        for path in ("host", "tables"):
            if path == "host":
                seq = jax.tree.map(lambda x: x[:n], st_host)
                out = ish.run_steps_sharded_forced(sm, state_p, sop, stress_p,
                                                   shard_mesh.partition_step_forcing(seq, part), fs_p, n, dt=DT,
                                                   npes=npes)
            else:
                dev = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, bgc_dust_path=str(PI_DUST),
                                                            bgc_co2=co2)
                tables, sched = dev.stack_tables(dates[:n], nb=4)
                out = ish.run_steps_sharded_forced(
                    sm, state_p, sop, stress_p, None, fs_p, n, dt=DT, npes=npes,
                    forcing_tables=shard_mesh.partition_forcing_tables(tables, part), forcing_sched=sched,
                    forcing_const=shard_mesh.partition_forcing_const(surface_forcing.forcing_device_const(dev),
                                                                     part))
            hb = np.asarray(out.hbar_old)
            for p in range(npes):
                md = int(part.myDim_nod2D[p])
                assert np.all(hb[p, :md] == st_host.co2_atm[n - 1]), (path, n, p)
            print(f"SHARDED {path} n={n}: co2_atm {st_host.co2_atm[n - 1]} reaches step()")


def test_run_config_co2(tmp_path):
    if not NML.is_file():
        pytest.skip("the reference namelist.recom is absent")
    text = NML.read_text().replace("@RECOM_RESTART@", ".false.")
    nml = tmp_path / "namelist.recom"
    nml.write_text(text)
    base = {"kind": "core2", "mslp": True, "dust_file": str(PI_DUST)}
    assert RunConfig.from_dict({"recom": {"namelist": str(nml)}, "forcing": base}).bgc_co2() is None
    off = tmp_path / "namelist_co2.recom"
    off.write_text(re.sub(r"(constant_CO2\s*=\s*)\.true\.", r"\1.false.", text))
    cfg = RunConfig.from_dict({"recom": {"namelist": str(off)}, "forcing": base})
    with pytest.raises(ValueError, match="co2_file"):                  # the namelist names a relative file
        cfg.bgc_co2()
    f = _co2_file(tmp_path / "co2.nc", (2000,))
    cfg = RunConfig.from_dict({"recom": {"namelist": str(off), "ic_set": "core2_present"},
                               "forcing": dict(base, co2_file=str(f))})
    s = cfg.bgc_co2()
    assert isinstance(s, AtmCO2Series) and s.path == str(f) and (s.first_year, s.last_year) == (1958, 2024)
    assert cfg.bgc_dust_path() == str(PI_DUST)
    RecomConfig(namelist=str(off), ic_set="core2_present").validate()
    assert ic.CORE2_PRESENT_IC_FILES[1002] == ("GLODAPv2.2016b.TCO2_fesom2_mmol_fix_z_Fillvalue.nc", "TCO2_mmol")
    assert {k: v for k, v in ic.CORE2_PRESENT_IC_FILES.items() if k != 1002} == \
        {k: v for k, v in ic.CORE2_IC_FILES.items() if k != 1002}
