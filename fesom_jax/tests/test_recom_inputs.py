"""REcoM plan Task 7 (lane P part 2): REcoM's per-step inputs through the forcing paths.

``StepForcing.daynew``/``ndpyr`` (FESOM2's clock at the ``call recom`` point) and ``StepForcing.fe_dust`` (the held
Albani dust, ``GloFeDust``), built by :class:`fesom_jax.surface_forcing.BgcForcing`:

* the gate: against the ``recom_in`` records of lane F's dumps (``recom_in.calendar`` = daynew, month, mstep, ndpyr,
  yearold; ``recom_in.GloFeDust``) through the host stack, for a January and a July cold start and the second
  segment of the split run (the ``mstep == 1`` dust rule);
* FESOM2's switch rule across month and year ends (synthetic dates), incl. a segment starting on a month's last step;
* host stack == local (per-rank) stack == on-device tables, bitwise; on the sharded pi ``dist_4`` path the fields
  reach ``step()`` with the values of the host stack, on the per-step stack and on the tables path;
* REcoM off: no new leaf (the DP lock, ``test_dp_lock.py``, is the program check).

CPU, double precision; the sharded test needs ``XLA_FLAGS=--xla_force_host_platform_device_count=4``. Compute node only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import jra55, partit, shard_mesh, ssh, surface_forcing
from fesom_jax import integrate_sharded as ish
from fesom_jax import step as stepmod
from fesom_jax.forcing_local import build_local_forcing
from fesom_jax.io_dump import load_recom_dump
from fesom_jax.mesh import load_mesh
from fesom_jax.run_config import RunConfig
from fesom_jax.state import State
from fesom_jax.surface_forcing import BgcForcing, recom_clock

REPO = Path(__file__).resolve().parents[2]
DT = 2700.0
YEAR = 1958
NDEV = len(jax.devices())
PI_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/pi")
PI_DUST = Path("/work/ab0995/a270088/fesom2_sp_zstar/test/input/recom/DustClimMonthlyAlbani_pimesh.nc")
NML = REPO / "scripts" / "recom" / "fortran" / "namelists" / "namelist.recom"
REF = Path("/work/ab0995/a270088/recom_ref")
# (dump directory, segment start (year, day of year, sec) of its first step, dumped msteps)
GATE_RUNS = [
    (REF / "pi_30d_jan" / "dumps", (1958, 1, 0.0), (1, 2, 17, 224, 225, 960)),
    (REF / "pi_1d_jul" / "dumps", (1958, 182, 0.0), (1, 17)),
    # segment 2 of the split run: fesom.clock "86400 15 1958" -> the first step starts 1958-01-16 00:00
    (REF / "pi_30d_split" / "dumps", (1958, 16, 0.0), (1, 2)),
]

have_jra = pytest.mark.skipif(not Path(jra55.DEFAULT_JRA_DIR).is_dir(), reason="JRA55-do forcing not reachable")
have_dust = pytest.mark.skipif(not PI_DUST.is_file(), reason=f"pi Albani dust file absent ({PI_DUST})")
have_dist4 = pytest.mark.skipif(NDEV < 4 or not (PI_DIST / "dist_4").is_dir(),
                                reason="needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4) and pi dist_4")


def _dates(start, n, dt=DT):
    """``n`` step starts (year, day, sec, month) from the step start ``start`` = (year, day, sec)."""
    import datetime
    base = datetime.datetime(start[0], 1, 1) + datetime.timedelta(days=start[1] - 1, seconds=start[2])
    out = []
    for i in range(n):
        d = base + datetime.timedelta(seconds=i * dt)
        out.append((d.year, d.timetuple().tm_yday, d.hour * 3600.0 + d.minute * 60.0 + d.second, d.month))
    return out


def _held(bf, dates):
    return [bf.inputs(*d[:3])[1] for d in dates]


@pytest.fixture(scope="module")
def mesh():
    return load_mesh()


# ---------------------------------------------------------------------------------------------------------------
# the clock and the dust rule
def test_recom_clock_is_fesom2s_step_end():
    c = recom_clock(1958, 1, 0.0, DT)                                  # the first step of 1958
    assert (c.yearnew, c.daynew, c.timenew, c.ndpyr, c.month, c.day_in_month) == (1958, 1, 2700.0, 365, 1, 1)
    c = recom_clock(1958, 31, 86400.0 - DT, DT)                        # the last step of January: 24:00, same day
    assert (c.daynew, c.timenew, c.month, c.day_in_month) == (31, 86400.0, 1, 31)
    c = recom_clock(1958, 32, 0.0, DT)                                 # the first step of February
    assert (c.daynew, c.timenew, c.month, c.day_in_month) == (32, 2700.0, 2, 1)
    c = recom_clock(1958, 365, 86400.0 - DT, DT)                       # 31 Dec 24:00 stays in 1958
    assert (c.yearnew, c.daynew, c.timenew, c.month) == (1958, 365, 86400.0, 12)
    c = recom_clock(1960, 60, 0.0, DT)                                 # 29 Feb of a leap year
    assert (c.yearnew, c.ndpyr, c.month, c.day_in_month) == (1960, 366, 2, 29)
    c = recom_clock(1958, 1, 86400.0 - 1000.0, DT)                     # a step that crosses midnight: the next day
    assert (c.daynew, c.timenew) == (2, 1700.0)


def test_dust_hold_month_ends():
    dust = np.arange(12.0)[:, None] * np.ones((1, 3))                  # month m -> m-1
    # cold start 1 Jan: January until the last step of 31 Jan, which already holds February; February until its last
    # step, which holds March
    bf = BgcForcing(dust, DT)
    d = _dates((1958, 1, 0.0), 32 * 59 + 1)                            # through 1 Mar 00:00
    held = _held(bf, d)
    i_jan_last = 32 * 31 - 1
    assert set(held[:i_jan_last]) == {1} and held[i_jan_last] == 2
    i_feb_last = 32 * 59 - 1
    assert set(held[i_jan_last:i_feb_last]) == {2} and held[i_feb_last] == 3 and held[-1] == 3
    # a segment starting on January's last step: mstep == 1 loads January, which is then held through February
    bf = BgcForcing(dust, DT)
    d = _dates((1958, 31, 86400.0 - DT), 32 * 28 + 1)
    held = _held(bf, d)
    assert set(held[:-1]) == {1} and held[-1] == 3
    # the year end wraps 12 -> 1
    bf = BgcForcing(dust, DT, start=(1958, 365, 0.0))
    d = _dates((1958, 365, 0.0), 33)
    held = _held(bf, d)
    assert set(held[:31]) == {12} and held[31] == 1 and held[32] == 1
    # a dt that never ends a step at 24:00 never switches after mstep 1 (FESOM2's timenew == 86400 test)
    bf = BgcForcing(dust, 2500.0)
    d = _dates((1958, 1, 0.0), 2000, dt=2500.0)                        # ~58 days
    assert set(_held(bf, d)) == {1}
    # queries out of order give the same answers; before the segment start they raise
    bf = BgcForcing(dust, DT)
    d = _dates((1958, 1, 0.0), 32 * 32)
    fwd = _held(bf, d)
    assert _held(bf, d[::-1]) == fwd[::-1]
    with pytest.raises(ValueError, match="precedes"):
        bf.inputs(1957, 365, 0.0)


def test_dust_hold_dt_ramp():
    """A run with a dt ramp (the CORE2 runs from the dt-1800 v3 restart: dt 1800, ``dt_ramp`` 2700 at the restart
    step): the steps follow the run driver's piecewise clock, each step's clock reads its own dt, and the answers
    after the switch are those of a constant-dt object started there."""
    from fesom_jax.run import _chunk_dates
    from fesom_jax.run_config import DtRamp
    dust = np.arange(12.0)[:, None] * np.ones((1, 3))
    ramp = DtRamp(after_step=48 * 10, dt=2700.0)                      # 1800 s for 10 days, then 2700 s
    dates = _chunk_dates(1958, 1800.0, 0, 48 * 10 + 32 * 25, ramp)     # through 4 Feb
    bf = BgcForcing(dust, 1800.0, dt_ramp=ramp, year0=1958)
    got = [bf.inputs(*d[:3]) for d in dates]
    assert dates[48 * 10][:3] == (1958, 11, 0.0)
    assert [c.timenew for c, _ in got[48 * 10 - 1:48 * 10 + 1]] == [86400.0, 2700.0]
    ref = BgcForcing(dust, 2700.0, start=(1958, 11, 0.0))
    for d, (c, m) in zip(dates[48 * 10:], got[48 * 10:]):
        assert ref.inputs(*d[:3]) == (c, m)
    held = [m for _, m in got]
    i_jan_last = 48 * 10 + 32 * 21 - 1                                 # the step ending at 24:00 on 31 Jan
    assert set(held[:i_jan_last]) == {1} and set(held[i_jan_last:]) == {2}
    # a segment that starts after the switch (a restart at the ramp step) steps on the 2700 s grid
    bf = BgcForcing(dust, 1800.0, dt_ramp=ramp, year0=1958)
    assert bf.inputs(1958, 11, 0.0)[1] == 1 and bf.inputs(1958, 11, 2700.0)[0].timenew == 5400.0
    # without the ramp the 2700 s steps are off the 1800 s grid (the failure of job 28430951)
    bf = BgcForcing(dust, 1800.0, start=(1958, 11, 0.0))
    with pytest.raises(ValueError, match="grid"):
        bf.inputs(1958, 11, 2700.0)


# ---------------------------------------------------------------------------------------------------------------
# the gate against the Fortran dumps
_GATE = [(r, s, m) for r, s, ms in GATE_RUNS if r.is_dir() for m in ms]


@have_jra
@have_dust
@pytest.mark.skipif(not _GATE, reason="REcoM reference dumps absent")
@pytest.mark.parametrize("run,start", sorted({(r, s) for r, s, _ in _GATE}),
                         ids=[r.parent.name for r, _ in sorted({(r, s) for r, s, _ in _GATE})])
def test_gate_calendar_and_dust(mesh, run, start):
    """Through the host stack of the segment: daynew, ndpyr (and the clock's month) = ``recom_in.calendar``,
    fe_dust = ``recom_in.GloFeDust`` bitwise, on every dumped step."""
    msteps = [m for r, s, m in _GATE if r == run]
    cf = surface_forcing.build_surface_forcing(mesh, start[0], dt=DT, bgc_dust_path=str(PI_DUST))
    dates = _dates(start, max(msteps))
    st = cf.stack(dates, xp=np)
    assert st.daynew.dtype == np.int32 and st.daynew.shape == (len(dates),)
    for m in msteps:
        f, meta = load_recom_dump(run, m, names=["recom_in.calendar", "recom_in.GloFeDust"])
        daynew, month, mstep, ndpyr, _ = (int(x) for x in f["recom_in.calendar"])
        assert mstep == m
        c = recom_clock(*dates[m - 1][:3], DT)
        assert (int(st.daynew[m - 1]), int(st.ndpyr[m - 1]), c.month) == (daynew, ndpyr, month), f"mstep {m}"
        assert (c.daynew, c.timenew) == (meta["daynew"], meta["timenew"]), f"mstep {m}: clock"
        np.testing.assert_array_equal(st.fe_dust[m - 1], f["recom_in.GloFeDust"], err_msg=f"mstep {m}")
        print(f"GATE {run.parent.name} mstep {m}: daynew {daynew} ndpyr {ndpyr} month {month} dust month "
              f"{cf.bgc.inputs(*dates[m - 1][:3])[1]}: equal")


# ---------------------------------------------------------------------------------------------------------------
# REcoM off and the three host paths
@have_jra
def test_off_adds_no_leaf(mesh):
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT)
    d = _dates((YEAR, 1, 0.0), 3)
    sf = cf.step_forcing(*d[0])
    assert sf.fe_dust is None and sf.daynew is None and sf.ndpyr is None and sf.co2_atm is None
    assert len(jax.tree.leaves(sf)) == len(surface_forcing.StepForcing._fields) - 5     # + press_air off
    tables, sched = cf.stack_tables(d, nb=3)
    assert tables.dust is None and sched.didx is None and sched.daynew is None
    with pytest.raises(ValueError, match="needs dt"):
        surface_forcing.build_surface_forcing(mesh, YEAR, bgc_dust_path=str(PI_DUST))


@have_jra
@have_dust
@pytest.mark.skipif(not (PI_DIST / "dist_4").is_dir(), reason="pi dist_4 absent")
def test_host_local_tables_agree(mesh):
    """Over 30 Jan - 1 Feb 1958 (the dust switch on 31 Jan's last step, the sss/chl month change at 1 Feb 00:00):
    the host stack, the per-rank local stack and the on-device tables give the same REcoM inputs, bitwise."""
    dates = _dates((YEAR, 30, 0.0), 96)
    host = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, bgc_dust_path=str(PI_DUST))
    st = host.stack(dates, xp=np)
    held = [host.bgc.inputs(*d[:3])[1] for d in dates]
    assert held.count(1) == 63 and held.count(2) == 33                 # 30 Jan + 31 Jan up to its last step
    part = partit.read_partition(PI_DIST, 4)
    loc = build_local_forcing(mesh, YEAR, part, 4, local_parts=list(range(4)), static=host.static, dt=DT,
                              bgc_dust_path=str(PI_DUST))
    lp = loc.stack_partitioned(dates)
    gp = shard_mesh.partition_step_forcing(st, part)
    for name in ("daynew", "ndpyr"):
        np.testing.assert_array_equal(lp_ := getattr(lp, name), getattr(st, name))
        np.testing.assert_array_equal(getattr(gp, name), lp_)
    for p, ml in enumerate(part.myList_nod2D):
        n = len(ml)
        np.testing.assert_array_equal(lp.fe_dust[p, :, :n], gp.fe_dust[p, :, :n], err_msg=f"part {p}")
    dev = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, bgc_dust_path=str(PI_DUST))
    tables, sched = dev.stack_tables(dates, nb=int(96 * DT) // 10800 + 2)
    const = surface_forcing.forcing_device_const(dev)
    for i in range(len(dates)):
        got = surface_forcing.combine_step_forcing(tables, sched.rdate[i], sched.fidx[i], sched.midx[i], const,
                                                   didx=sched.didx[i], daynew=sched.daynew[i], ndpyr=sched.ndpyr[i])
        np.testing.assert_array_equal(np.asarray(got.fe_dust), st.fe_dust[i], err_msg=f"step {i}")
        assert int(got.daynew) == int(st.daynew[i]) and int(got.ndpyr) == int(st.ndpyr[i])
    ltab, lsched = loc.stack_tables_partitioned(dates, nb=int(96 * DT) // 10800 + 2)
    gtab = shard_mesh.partition_forcing_tables(tables, part)
    for p, ml in enumerate(part.myList_nod2D):
        np.testing.assert_array_equal(ltab.dust[p, :, :len(ml)], gtab.dust[p, :, :len(ml)])
    np.testing.assert_array_equal(lsched.didx, sched.didx)


# ---------------------------------------------------------------------------------------------------------------
# sharded: the fields reach step()
@have_jra
@have_dust
@have_dist4
def test_sharded_fields_reach_step(mesh, monkeypatch):
    """pi ``dist_4``, 4 host devices, two forced steps across the dust switch: inside ``step()`` the sharded runner
    hands ``fe_dust`` (node lanes), ``daynew`` and ``ndpyr`` (scalars) with the host stack's values, on the per-step
    stack and on the tables path. A probe around ``step`` writes them into two state fields."""
    dates = _dates((YEAR, 31, 86400.0 - 2 * DT), 3)                    # the last two steps of January + 1 Feb
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, bgc_dust_path=str(PI_DUST))
    st_host = cf.stack(dates, xp=np)
    held = [cf.bgc.inputs(*d[:3])[1] for d in dates]
    assert held == [1, 2, 2]
    orig = stepmod.step

    def probe(state, *a, step_forcing=None, **kw):
        out = orig(state, *a, step_forcing=step_forcing, **kw)
        cal = (step_forcing.daynew + 1000 * step_forcing.ndpyr).astype(out.d_eta.dtype)
        return dataclasses.replace(out, d_eta=step_forcing.fe_dust, hbar_old=jnp.full_like(out.hbar_old, cal))

    monkeypatch.setattr(stepmod, "step", probe)
    npes = 4
    part = partit.read_partition(PI_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    st0 = jax.tree.map(np.asarray, State.rest(mesh))
    state_p = shard_mesh.partition_state(st0, part)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    sop = ssh.partition_ssh_operator(op, part)
    fs_p = shard_mesh.partition_forcing_static(cf.static, part)
    stress_p = jnp.zeros((npes, sm.Lmax["elem"], 2))
    for n in (1, 3):                    # one step (mstep 1), then all three (the switch on the second)
        seq = jax.tree.map(lambda x: x[:n], st_host)
        for path in ("host", "tables"):
            if path == "host":
                out = ish.run_steps_sharded_forced(sm, state_p, sop, stress_p,
                                                   shard_mesh.partition_step_forcing(seq, part), fs_p, n, dt=DT,
                                                   npes=npes)
            else:
                dev = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, bgc_dust_path=str(PI_DUST))
                tables, sched = dev.stack_tables(dates[:n], nb=4)
                out = ish.run_steps_sharded_forced(
                    sm, state_p, sop, stress_p, None, fs_p, n, dt=DT, npes=npes,
                    forcing_tables=shard_mesh.partition_forcing_tables(tables, part), forcing_sched=sched,
                    forcing_const=shard_mesh.partition_forcing_const(surface_forcing.forcing_device_const(dev),
                                                                     part))
            k = n - 1                       # the probe holds the last step's inputs
            want_cal = float(st_host.daynew[k]) + 1000.0 * float(st_host.ndpyr[k])
            d_eta, hb = np.asarray(out.d_eta), np.asarray(out.hbar_old)
            for p, ml in enumerate(part.myList_nod2D):
                md = int(part.myDim_nod2D[p])
                np.testing.assert_array_equal(d_eta[p, :md], st_host.fe_dust[k][np.asarray(ml)[:md]],
                                              err_msg=f"{path} n={n} part {p}")
                assert np.all(hb[p, :md] == want_cal), (path, n, p)
            print(f"SHARDED {path} n={n}: fe_dust (month {held[k]}) and daynew/ndpyr reach step()")


# ---------------------------------------------------------------------------------------------------------------
def test_run_config_dust_path(tmp_path):
    assert RunConfig().bgc_dust_path() is None
    assert RunConfig.from_dict({"recom": {"passive_tracers": 3}}).bgc_dust_path() is None
    if not NML.is_file():
        pytest.skip("the reference namelist.recom is absent")
    nml = tmp_path / "namelist.recom"
    nml.write_text(NML.read_text().replace("@RECOM_RESTART@", ".false."))
    cfg = RunConfig.from_dict({"recom": {"namelist": str(nml)}, "forcing": {"kind": "core2", "mslp": True}})
    with pytest.raises(ValueError, match="dust_file"):                 # the namelist names a relative file
        cfg.bgc_dust_path()
    cfg = RunConfig.from_dict({"recom": {"namelist": str(nml)},
                               "forcing": {"kind": "core2", "mslp": True, "dust_file": str(PI_DUST)}})
    assert cfg.bgc_dust_path() == str(PI_DUST)
