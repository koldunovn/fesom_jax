"""FESOM2's ``l_mslp`` in fesom-jax (REcoM plan Task 5, lane P): the JRA55-do sea-level pressure ``psl`` read like
the other JRA55-do fields into ``StepForcing.press_air`` [Pa], and its gradient in the momentum RHS
(``oce_ale_vel_rhs.F90:208``: ``p_air = press_air/1000``, ``pre = -(g·eta + p_air)``).

* GP gate: ``press_air`` on the nodes and the SLP momentum term against the Fortran FESOM2 + REcoM dumps
  (``recom_in.press_air``, ``vel_rhs.slp_force``; ``docs/recom/FORTRAN_REFERENCE.md``).
* ``mslp`` off adds no leaf and leaves the program unchanged (the DP lock, ``test_dp_lock.py``, is the full check).
* the host, local (per-process) and on-device forcing paths and the serial/sharded step agree.

CPU, double precision. The sharded tests need ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` and FESOM2's pi
``dist_4``; the gate needs the reference dumps; every test needs the JRA55-do files. Compute node only (XLA).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ice, jra55, momentum, partit, shard_mesh, ssh, surface_forcing
from fesom_jax import integrate_sharded as ish
from fesom_jax import step as stepmod
from fesom_jax.forcing_local import build_local_forcing
from fesom_jax.io_dump import load_recom_dump
from fesom_jax.mesh import load_mesh
from fesom_jax.recom.config import RecomConfig
from fesom_jax.run_config import RunConfig, load_yaml
from fesom_jax.state import State

REPO = Path(__file__).resolve().parents[2]
PI_RECOM = REPO / "configs" / "pi_recom.yaml"
DT = 2700.0
YEAR = 1958
NDEV = len(jax.devices())
PI_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/pi")
REF = Path("/work/ab0995/a270088/recom_ref")
# lane F's 30-day run (l_mslp on, dumps at mstep 1, 2, 17, 224, 225, 960) and the physics-matched twin (1, 2, 3, 17,
# 960); the forcing is the same in both
DUMP_RUNS = [REF / "pi_30d_jan" / "dumps", REF / "pi_30d_jan_jaxphys" / "dumps"]
# GP tolerance (DP). press_air: the JAX reader evaluates rdate·coef_a + coef_b at an absolute Julian day (~2.4e6) where
# FESOM2 e96a2b13 evaluates the point-slope form coef_b + (rdate - t0)·coef_a (FESOM2 #940) and forms rdate from
# timenew - dt/2 instead of the step start + dt/2: both differ in the last bits (measured below).
GP_RTOL = 1e-12
# The JAX press_air itself: the evaluation-form difference above, measured 1.5e-11..4.3e-11 relative (up to 4e-6 Pa)
# on the dumped steps past the first JRA55 record (steps 1-2 sit before it, on the clamped bracket: 4e-16); the same
# bracket evaluated in FESOM2's form meets GP_RTOL (asserted), so the residual is the form.
GP_PA_RTOL = 1e-10


def _have_jra():
    return Path(jra55.DEFAULT_JRA_DIR).is_dir() and (Path(jra55.DEFAULT_JRA_DIR) / f"psl.{YEAR}.nc").is_file()


have_forcing = pytest.mark.skipif(not _have_jra(), reason="JRA55-do forcing (with psl) not reachable")
have_dist4 = pytest.mark.skipif(NDEV < 4 or not (PI_DIST / "dist_4").is_dir(),
                                reason="needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4) and pi dist_4")


def _dump_steps():
    out = []
    for run in DUMP_RUNS:
        if run.is_dir():
            out += [(run, int(p.name.split("_")[1])) for p in sorted(run.glob("step_*"))
                    if (p / "index.txt").is_file()]
    return out


@pytest.fixture(scope="module")
def mesh():
    return load_mesh()


@pytest.fixture(scope="module")
def cf_pair(mesh):
    """The pi forcing without and with l_mslp (mid-step instant, as the run driver builds it)."""
    off = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT)
    on = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, mslp=True)
    return off, on


# ---------------------------------------------------------------------------------------------------------------
# configuration
def test_config_mslp_key_and_recom_requirement():
    cfg = load_yaml(PI_RECOM)
    assert cfg.forcing_mslp() and cfg.recom is None
    assert "mslp" not in cfg.forcing_paths()                       # not an input path: the readers never see it
    assert not RunConfig().forcing_mslp()
    assert not RunConfig.from_dict({"forcing": {"kind": "core2", "start_year": 1958}}).forcing_mslp()
    with pytest.raises(TypeError, match="mslp"):
        RunConfig.from_dict({"forcing": {"kind": "core2", "mslp": "yes"}}).validate()
    # REcoM proper (no passive tracers) without l_mslp: refused with the reason
    for forcing in (None, {"kind": "core2", "start_year": 1958}, {"kind": "core2", "mslp": False}):
        with pytest.raises(ValueError, match="forcing.mslp"):
            RunConfig.from_dict({"recom": {}, "forcing": forcing}).validate()
    try:                                                           # with it, the mslp check passes (the rest is
        RunConfig.from_dict({"recom": {}, "forcing": {"kind": "core2", "mslp": True}}).validate()   # Task 6+)
    except NotImplementedError:
        pass
    # the passive-tracer transport mode runs no chemistry and keeps its physics
    RunConfig.from_dict({"recom": {"passive_tracers": 3}}).validate()
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


# ---------------------------------------------------------------------------------------------------------------
# the reader and the forcing pytrees
@have_forcing
def test_reader_off_unchanged_on_adds_psl(mesh, cf_pair):
    off, on = cf_pair
    assert [f.var for f in off.jra.fields] == list(jra55.JRA_VARS)
    assert [f.var for f in on.jra.fields] == list(jra55.JRA_VARS) + ["psl"]
    for date in surface_forcing.dates_for_steps(YEAR, DT, 9):      # crosses three 3-hourly brackets
        a = off.jra.step(*date[:3])
        b = on.jra.step(*date[:3])
        assert a.press_air is None
        np.testing.assert_array_equal(b.as_array(), a.as_array())  # the eight fields are untouched
        assert b.press_air.shape == (mesh.nod2D,)
        assert 8.5e4 < b.press_air.min() and b.press_air.max() < 1.1e5


@have_forcing
def test_stepforcing_off_adds_no_leaf(mesh, cf_pair):
    off, on = cf_pair
    dates = surface_forcing.dates_for_steps(YEAR, DT, 3)
    sf_off, sf_on = off.step_forcing(*dates[0]), on.step_forcing(*dates[0])
    assert sf_off.press_air is None
    n_core = len(jax.tree.leaves(sf_off))                          # every non-optional field
    assert n_core == sum(getattr(sf_off, f) is not None for f in surface_forcing.StepForcing._fields)
    assert len(jax.tree.leaves(sf_on)) == n_core + 1
    st_off, st_on = off.stack(dates, xp=np), on.stack(dates, xp=np)
    assert st_off.press_air is None and st_on.press_air.shape == (3, sf_on.press_air.shape[0])
    for name in surface_forcing.StepForcing._fields:
        if name != "press_air":
            np.testing.assert_array_equal(getattr(st_on, name), getattr(st_off, name), err_msg=name)
    # the sharded fold: a None field stays None in the data and in the spec tree
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    for st, has in ((st_off, False), (st_on, True)):
        p = shard_mesh.partition_step_forcing(st, ser)
        seq, spec = ish._fold_forcing_seq(p)
        assert (seq.press_air is not None) == has and (spec.press_air is not None) == has
        one = jax.tree.map(lambda x: x[:, 0], p)
        f1, s1 = ish._fold_forcing(one)
        assert (f1.press_air is not None) == has and (s1.press_air is not None) == has
        assert len(jax.tree.leaves(seq)) == len(jax.tree.leaves(st))


@have_forcing
def test_tables_combine_press_air(mesh, cf_pair):
    """The on-device combine (forcing.on_device) gives the host reader's press_air, eagerly bitwise, across bracket
    rolls; a reader without mslp gives eight table rows and press_air None."""
    dates = surface_forcing.dates_for_steps(YEAR, DT, 12)
    for mslp in (False, True):
        host = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, mslp=mslp)
        dev = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, mslp=mslp)   # its own bracket cache
        tables, sched = dev.stack_tables(dates, nb=int(12 * DT) // 10800 + 2)
        assert tables.coef_a.shape[0] == jra55.N_FLD + int(mslp)
        const = surface_forcing.forcing_device_const(dev)
        for i, d in enumerate(dates):
            got = surface_forcing.combine_step_forcing(tables, sched.rdate[i], sched.fidx[i], sched.midx[i], const)
            ref = host.step_forcing(*d, xp=np)
            if mslp:
                np.testing.assert_array_equal(np.asarray(got.press_air), ref.press_air, err_msg=f"step {i}")
            else:
                assert got.press_air is None


@have_forcing
@pytest.mark.skipif(not (PI_DIST / "dist_4").is_dir(), reason="pi dist_4 absent")
def test_local_forcing_press_air(mesh, cf_pair):
    """The per-process local forcing (NG5 path) scatters press_air like the other fields: bitwise the global stack's
    partition; mslp off ⇒ None."""
    off, on = cf_pair
    part = partit.read_partition(PI_DIST, 4)
    dates = surface_forcing.dates_for_steps(YEAR, DT, 5)
    for cf in (off, on):
        fresh = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, mslp=cf.jra.mslp)
        glob = shard_mesh.partition_step_forcing(fresh.stack(dates, xp=np), part)
        loc = build_local_forcing(mesh, YEAR, part, 4, local_parts=list(range(4)), static=cf.static, dt=DT,
                                  mslp=cf.jra.mslp).stack_partitioned(dates)
        if not cf.jra.mslp:
            assert loc.press_air is None and glob.press_air is None
            continue
        for p, ml in enumerate(part.myList_nod2D):
            n = len(ml)
            np.testing.assert_array_equal(loc.press_air[p, :, :n], glob.press_air[p, :, :n], err_msg=f"part {p}")


# ---------------------------------------------------------------------------------------------------------------
# the momentum term
def _rhs_inputs(mesh, seed=0):
    rng = np.random.default_rng(seed)
    st = State.rest(mesh)
    lm = np.asarray(mesh.elem_layer_mask)[..., None]
    uv = jnp.asarray(np.where(lm, 0.1 * rng.standard_normal(st.uv.shape), 0.0))
    eta = jnp.asarray(0.3 * rng.standard_normal(mesh.nod2D))
    z = jnp.zeros_like(st.pgf_x)
    return dict(uv=uv, uv_rhsAB=jnp.zeros_like(st.uv_rhsAB), eta_n=eta, pgf_x=z, pgf_y=z, w_e=st.w_e,
                hnode=st.hnode)


def test_vel_rhs_zero_pressure_is_unchanged(mesh):
    """press_air ≡ 0 gives the l_mslp-off RHS bitwise (-(g·eta + 0) = -g·eta), so the term is wired additively into
    the SSH pressure exactly where FESOM2 adds it."""
    kw = _rhs_inputs(mesh)
    a = momentum.compute_vel_rhs(mesh, **kw, is_first_step=True, dt=DT)
    b = momentum.compute_vel_rhs(mesh, **kw, is_first_step=True, dt=DT, press_air=jnp.zeros(mesh.nod2D))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _slp_force(mesh, press_air):
    """FESOM2's l_mslp momentum term per element, -grad(press_air/1000) [m/s2] (the dump hook's arithmetic,
    ``rdh_slp_force``: ``pre = -p_air``, ``Fx = sum(gradient_sca(1:3)*pre)``)."""
    p = np.asarray(press_air)[np.asarray(mesh.elem_nodes)] / 1000.0
    pre = -p
    gs = np.asarray(mesh.gradient_sca)
    fx = (gs[:, 0] * pre[:, 0] + gs[:, 1] * pre[:, 1]) + gs[:, 2] * pre[:, 2]
    fy = (gs[:, 3] * pre[:, 0] + gs[:, 4] * pre[:, 1]) + gs[:, 5] * pre[:, 2]
    return np.stack([fx, fy], axis=-1)


def test_vel_rhs_slp_term(mesh):
    """With eta = 0 the RHS difference on/off is dt·F_slp on every wet layer of the first step (no AB history)."""
    kw = _rhs_inputs(mesh, seed=1)
    kw["eta_n"] = jnp.zeros(mesh.nod2D)
    rng = np.random.default_rng(2)
    pa = 1.0e5 + 2.0e3 * rng.standard_normal(mesh.nod2D)
    a, _ = momentum.compute_vel_rhs(mesh, **kw, is_first_step=True, dt=DT)
    b, _ = momentum.compute_vel_rhs(mesh, **kw, is_first_step=True, dt=DT, press_air=jnp.asarray(pa))
    want = DT * _slp_force(mesh, pa)                                  # (elem2D, 2)
    lm = np.asarray(mesh.elem_layer_mask)
    got = np.asarray(b - a)
    exp = np.where(lm[..., None], want[:, None, :], 0.0)
    np.testing.assert_allclose(got, exp, rtol=0, atol=1e-9 * np.max(np.abs(want)))


# ---------------------------------------------------------------------------------------------------------------
# GP gate against the Fortran dumps
_GP = _dump_steps()


@have_forcing
@pytest.mark.skipif(not _GP, reason=f"REcoM reference dumps absent ({[str(r) for r in DUMP_RUNS]})")
@pytest.mark.parametrize("run,mstep", _GP, ids=[f"{r.parent.name}-{s}" for r, s in _GP])
def test_gp_press_air_and_slp_force(mesh, cf_pair, run, mstep):
    """GP: press_air from the JAX reader and the SLP momentum term against the Fortran dump of the same step (a cold
    start on 1958-01-01, so mstep = the model step)."""
    f, meta = load_recom_dump(run, mstep, names=["mesh.gid_nod", "recom_in.press_air", "vel_rhs.slp_force"])
    assert meta["mstep"] == mstep and np.array_equal(f["mesh.gid_nod"], np.arange(1, mesh.nod2D + 1))
    fresh = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, mslp=True)
    date = surface_forcing.dates_for_steps(YEAR, DT, mstep)[-1]
    pa = fresh.step_forcing(*date, xp=np).press_air
    ref = f["recom_in.press_air"]
    pscale = np.max(np.abs(ref))
    rel = np.max(np.abs(pa - ref)) / pscale
    # FESOM2's own evaluation from the same reader state: its instant timenew - dt/2 (timenew = the step end) and the
    # point-slope form coef_b + (rdate - t0)·coef_a with coef_b = the bracket-start field (gen_surface_forcing.F90:
    # 1171-1175). This isolates the time axis, the bracket and the mesh interpolation from the evaluation form.
    fp = fresh.jra.fields[jra55.I_MSLP]
    rdate_f = (float(jra55._julday(YEAR, 1, 1, fp.calendar)) + float(meta["daynew"] - 1)
               + meta["timenew"] / 86400.0 - DT / 86400.0 / 2.0)
    if fp.t_indx != fp.t_indx_p1:          # an open bracket must hold FESOM2's instant (the clamped one, before the
        assert fp.nc_time[fp.t_indx - 1] <= rdate_f <= fp.nc_time[fp.t_indx_p1 - 1]    # first record, has slope 0)
    pa_ps = fp.d1 + (rdate_f - fp.t0) * fp.coef_a
    rel_ps = np.max(np.abs(pa_ps - ref)) / pscale
    print(f"GP {run.parent.name} mstep {mstep}: press_air max rel {rel:.2e} (JAX), {rel_ps:.2e} (FESOM2's form)")
    assert rel_ps <= GP_RTOL
    assert rel <= GP_PA_RTOL
    slp_ref = f["vel_rhs.slp_force"]                                 # (elem2D, 2)
    scale = np.max(np.abs(slp_ref))
    rel_f = np.max(np.abs(_slp_force(mesh, ref) - slp_ref)) / scale    # the arithmetic on Fortran's own pressure
    rel_j = np.max(np.abs(_slp_force(mesh, pa) - slp_ref)) / scale     # the JAX pressure end to end
    print(f"GP {run.parent.name} mstep {mstep}: slp_force rel {rel_f:.2e} (Fortran press_air), {rel_j:.2e} (JAX)")
    assert rel_f <= GP_RTOL
    assert rel_j <= GP_PA_RTOL * pscale / 1000.0 * np.max(np.abs(np.asarray(mesh.gradient_sca))) / scale * 3


# ---------------------------------------------------------------------------------------------------------------
# the forced step: serial, sharded, on-device
def _pi_state(mesh, cfg):
    """A rest zstar state with T/S fronts and the seeded ice (pi has no packaged forcing-ready IC of the REcoM run)."""
    lat = np.asarray(mesh.geo_coord_nod2D[:, 1])
    lon = np.asarray(mesh.geo_coord_nod2D[:, 0])
    Z = np.asarray(mesh.Z)
    z = np.concatenate([Z, Z[-1:]])[None, :]
    decay = np.exp(z / 800.0)
    T = np.clip(28.0 * np.cos(lat)[:, None] - 4.0, -1.8, 29.0) * decay + 2.0 * (1.0 - decay)
    S = 34.6 + 0.8 * (np.cos(lat) * np.sin(3.0 * lon))[:, None] * decay + 0.2 * (1.0 - decay)
    st = jax.tree.map(np.asarray, State.rest(mesh, ale_cfg=cfg.ale_cfg()))
    st = dataclasses.replace(st, T=T, S=S, T_old=T, S_old=S)
    return ice.seed_ice(st, mesh, st.T[:, 0], xp=np)


@pytest.fixture(scope="module")
def pi_recom_setup(mesh, cf_pair):
    cfg = load_yaml(PI_RECOM)
    st = _pi_state(mesh, cfg)
    _, on = cf_pair
    dates = surface_forcing.dates_for_steps(YEAR, DT, 2)
    fresh = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, sst_ic=st.T[:, 0], mslp=True)
    return dict(cfg=cfg, phys=cfg.physics_kwargs(), state=st, cf=fresh, dates=dates,
                stack=fresh.stack(dates, xp=np), op=ssh.build_ssh_operator(mesh, dt=DT))


def _dense(fx, mesh, sf_stack, n=1):
    st = jax.tree.map(jnp.asarray, fx["state"])
    for i in range(n):
        sf = jax.tree.map(lambda x: jnp.asarray(x[i]), sf_stack)
        st = stepmod.step_jit(st, mesh, fx["op"], jnp.zeros((mesh.elem2D, 2)), dt=DT, is_first_step=(i == 0),
                              step_forcing=sf, forcing_static=fx["cf"].static, **fx["phys"])
    return st


@have_forcing
def test_dense_step_mslp_moves_only_through_momentum(mesh, pi_recom_setup):
    """One pi_recom step with and without press_air: the velocity changes (the SLP term is live), the state stays
    finite, and the step-0 forcing fields other than press_air are the ones the off run uses."""
    fx = pi_recom_setup
    on = _dense(fx, mesh, fx["stack"])
    off = _dense(fx, mesh, fx["stack"]._replace(press_air=None))
    du = np.max(np.abs(np.asarray(on.uv) - np.asarray(off.uv)))
    assert np.all(np.isfinite(np.asarray(on.uv))) and np.all(np.isfinite(np.asarray(on.T)))
    assert du > 1e-6, f"the SLP term did not move uv ({du:.2e})"
    print(f"pi_recom step 1: max|uv(mslp) - uv(no mslp)| = {du:.3e} m/s")


@have_forcing
@have_dist4
def test_sharded_step_matches_serial(mesh, pi_recom_setup):
    """pi dist_4 on 4 host devices: one forced pi_recom step with press_air equals the dense step on the owned
    entities (clean fields at reassociation level, the FCT tracers at their documented flip floor); the on-device
    tables path equals the host stack path on the sharded run."""
    from fesom_jax.tests.test_forcing_sharded import _slow_field_rel_check
    from fesom_jax.tests.test_step_sharded import _ICE_FIELDS, _global_boundary_node_p, _owned_match
    fx = pi_recom_setup
    npes = 4
    part = partit.read_partition(PI_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    dense = _dense(fx, mesh, fx["stack"], n=1)
    state_p = shard_mesh.partition_state(fx["state"], part)
    sop = ssh.partition_ssh_operator(fx["op"], part)
    fs_p = shard_mesh.partition_forcing_static(fx["cf"].static, part)
    stress_p = jnp.zeros((npes, sm.Lmax["elem"], 2))
    seq1 = jax.tree.map(lambda x: x[:1], fx["stack"])
    bn_p = _global_boundary_node_p(mesh, part)
    host = ish.run_steps_sharded_forced(sm, state_p, sop, stress_p, shard_mesh.partition_step_forcing(seq1, part),
                                        fs_p, 1, dt=DT, npes=npes, boundary_node_p=bn_p, **fx["phys"])
    _owned_match(dense, host, mesh, part, npes, tag="mslp-host", fct_atol=2e-2, extra_fct=_ICE_FIELDS)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, dt=DT, sst_ic=fx["state"].T[:, 0], mslp=True)
    tables, sched = cf.stack_tables(fx["dates"][:1], nb=2)
    assert tables.coef_a.shape[0] == jra55.N_FLD + 1
    tab = ish.run_steps_sharded_forced(
        sm, state_p, sop, stress_p, None, fs_p, 1, dt=DT, npes=npes,
        forcing_tables=shard_mesh.partition_forcing_tables(tables, part), forcing_sched=sched,
        forcing_const=shard_mesh.partition_forcing_const(surface_forcing.forcing_device_const(cf), part),
        boundary_node_p=bn_p, **fx["phys"])
    _slow_field_rel_check(host, tab, "mslp-tables")
