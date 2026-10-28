"""Speed plan Task 12 gate: sea ice every N ocean steps (``IceConfig.ice_ave_steps``), FESOM2's
scheme (``fesom_module.F90:1100-1166``, ``ice_oce_coupling.F90:146-196``, e96a2b13).

* the running means of ``ocean2ice`` and the update schedule == a hand-rolled numpy copy of the
  Fortran loop, bitwise, the upstream ``u_w``/``v_w`` quirk included (``uw/(k+1)``);
* the optional State fields: absent for N = 1 (no new pytree leaves), the upstream init values for
  N > 1, carried through the partition / fold / restart plumbing;
* CORE2, N = 2: the ice updates on alternate steps, ``ice_dt = 2*dt``, the means in the model
  match the loop, and the fluxes on the in-between step come from the held ice state;
* dense == sharded (4 fake CPU devices, dist_4) at the existing sharded-step tolerance: in
  ``test_step_sharded.py`` (``test_ice_ave_steps_sharded_owned_matches``, the sharding group);
* a restart written mid-cycle (odd step) resumes like the uninterrupted run, and chunk lengths
  that give odd scan lengths give the same trajectory.

  PY -m pytest fesom_jax/tests/test_ice_ave_steps.py -x
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ice_coupling, ice_step, partit, shard_mesh, ssh
from fesom_jax import integrate_sharded as ish
from fesom_jax.ice import IceConfig
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.state import ICE_AVG_FIELDS, State

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())
DT = 1800.0
YEAR = 1958
ICE2 = IceConfig(whichEVP=1, ice_ave_steps=2)
_PROG = ("a_ice", "m_ice", "m_snow", "u_ice", "v_ice", "t_skin", "sigma11", "sigma12", "sigma22")
_HELD = ("ice_flx_h", "ice_flx_fw", "ice_evaporation", "ice_sublimation", "ice_thdgr",
         "ice_thdgrsn", "ice_real_salt_flux", "ice_a_old")


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


have_forcing = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists() or not CORE2_MESH.is_dir() or not _have_jra(),
    reason="CORE2 PHC IC / mesh / JRA55 forcing missing (compute node only)")


# ==========================================================================
# 1. ocean2ice running means + the update schedule == the Fortran loop (no data)
# ==========================================================================
def _fortran_loop(cur, nave):
    """numpy copy of fesom_module.F90:600-601 + ocean2ice (ice_oce_coupling.F90:146-196) +
    the decision (:1129-1135), step by step. ``cur[s]`` = this step's (T, S, hbar, uw, vw)."""
    k, upd = nave - 1, True
    srf = {n: np.zeros_like(cur[0][0]) for n in ("T", "S", "e", "u", "v")}
    rec = []
    for (T, S, e, uw, vw) in cur:
        if upd:
            srf["T"], srf["S"], srf["e"] = T.copy(), S.copy(), e.copy()
        else:
            srf["T"] = (srf["T"] * float(k) + T) / float(k + 1)
            srf["S"] = (srf["S"] * float(k) + S) / float(k + 1)
            srf["e"] = (srf["e"] * float(k) + e) / float(k + 1)
        srf["u"] = np.zeros_like(uw)               # u_w(n) = 0 for every node (:170-174)
        srf["v"] = np.zeros_like(vw)
        if upd:
            srf["u"], srf["v"] = uw.copy(), vw.copy()
        else:
            srf["u"] = (srf["u"] * float(k) + uw) / float(k + 1)
            srf["v"] = (srf["v"] * float(k) + vw) / float(k + 1)
        if k >= nave - 1:
            upd, k = True, 0
        else:
            upd, k = False, k + 1
        rec.append(({n: a.copy() for n, a in srf.items()}, upd, k))
    return rec


def _same(got, want, msg, exact):
    """Bitwise, except that XLA:CPU divides a float64 array by the scalar k+1 as a multiply by its
    reciprocal (measured: 1 ulp at k+1 = 3 on 18 of 37 nodes; job 28295379); k+1 = 1, 2 are exact
    either way, so N = 2 is checked bitwise."""
    got = np.asarray(got)
    if exact or np.array_equal(got, want):
        np.testing.assert_array_equal(got, want, err_msg=msg)
        return
    np.testing.assert_allclose(got, want, rtol=2.3e-16, atol=0, err_msg=msg)


@pytest.mark.parametrize("nave", [2, 3])
def test_ocean2ice_avg_matches_fortran_loop(nave):
    rng = np.random.default_rng(nave)
    n, nl, steps = 37, 3, 3 * nave + 2
    cur = [tuple(rng.normal(size=n) for _ in range(5)) for _ in range(steps)]
    ref = _fortran_loop(cur, nave)
    k = jnp.full((n,), nave - 1, jnp.int32)
    upd = jnp.ones((n,), jnp.int32)
    srf = {f: jnp.zeros((n,)) for f in ("srfoce_temp", "srfoce_salt", "srfoce_ssh")}
    updates = []
    for s, ((T, S, e, uw, vw), (rsrf, rupd, rk)) in enumerate(zip(cur, ref)):
        T3 = np.repeat(T[:, None], nl, 1)
        S3 = np.repeat(S[:, None], nl, 1)
        uvn = np.zeros((n, nl, 2))
        uvn[:, 0, 0], uvn[:, 0, 1] = uw, vw
        st = SimpleNamespace(T=jnp.asarray(T3), S=jnp.asarray(S3), hbar=jnp.asarray(e),
                             uvnode=jnp.asarray(uvn), **srf)
        out = ice_coupling.ocean2ice_avg(st, k[0], upd[0] != 0)
        u_now, k_now = ice_step.update_decision(k[0], nave)
        for got, want in ((out.temp, rsrf["T"]), (out.salt, rsrf["S"]), (out.ssh, rsrf["e"]),
                          (out.u, rsrf["u"]), (out.v, rsrf["v"])):
            _same(got, want, f"step {s + 1}", nave == 2)
        assert bool(u_now) == rupd and int(k_now) == rk, f"step {s + 1}"
        if not bool(upd[0]):
            # the quirk: not a running mean of the velocity, the current velocity / (k+1)
            _same(out.u, uw / float(int(k[0]) + 1), f"step {s + 1} quirk", nave == 2)
        updates.append(bool(u_now))
        k = jnp.full((n,), k_now, jnp.int32)
        upd = jnp.full((n,), u_now, jnp.int32)
        srf = dict(srfoce_temp=out.temp, srfoce_salt=out.salt, srfoce_ssh=out.ssh)
    # the ice updates on step 1, 1+N, 1+2N, ...
    assert updates == [(s % nave) == 0 for s in range(steps)]


# ==========================================================================
# 2. The optional State fields: N = 1 adds nothing; N > 1 init values; plumbing (pi mesh)
# ==========================================================================
def test_init_ice_avg_fields():
    mesh = load_mesh()
    st = State.rest(mesh)
    n_leaves = len(jax.tree_util.tree_leaves(st))
    assert ice_step.init_ice_avg(st, IceConfig()) is st                  # N = 1: unchanged
    assert ice_step.init_ice_avg(st, None) is st
    s2 = ice_step.init_ice_avg(st, IceConfig(ice_ave_steps=3))
    assert len(jax.tree_util.tree_leaves(s2)) == n_leaves + len(ICE_AVG_FIELDS)
    assert np.all(np.asarray(s2.ice_steps_since_upd) == 2)               # ice_ave_steps - 1
    assert np.all(np.asarray(s2.ice_update) == 1)                        # .true.
    assert s2.ice_steps_since_upd.dtype == jnp.int32 and s2.ice_flx_h.dtype == jnp.float64
    assert ice_step.init_ice_avg(s2, IceConfig(ice_ave_steps=3)) is s2   # present: unchanged
    s1 = ice_step.init_ice_avg(s2, IceConfig())                          # back to N = 1: stripped
    assert all(getattr(s1, f) is None for f in ICE_AVG_FIELDS)
    assert len(jax.tree_util.tree_leaves(s1)) == n_leaves
    sh = ice_step.init_ice_avg(st, IceConfig(ice_ave_steps=2), xp=np)    # host build
    assert isinstance(sh.ice_update, np.ndarray)


def _signed(mesh, nave=2):
    """A pi State with the ICE_AVG fields filled with a distinct per-node signature."""
    st = ice_step.init_ice_avg(State.rest(mesh), IceConfig(ice_ave_steps=nave), xp=np)
    st = jax.tree.map(np.asarray, st)
    ids = np.arange(mesh.nod2D, dtype=np.float64)
    rep = {f: ids * (i + 1) + 0.25 for i, f in enumerate(ICE_AVG_FIELDS[2:])}
    rep["ice_steps_since_upd"] = np.zeros(mesh.nod2D, np.int32)
    rep["ice_update"] = np.ones(mesh.nod2D, np.int32)
    return dataclasses.replace(st, **rep)


def test_partition_fold_restart_roundtrip(tmp_path):
    """The new node fields go through partition_state -> fold -> device -> write_restart ->
    read_restart (onto another device count) unchanged; a restart without them reads as None and
    init_ice_avg gives the upstream init values (a new cycle, like FESOM2's netCDF restart)."""
    from fesom_jax import halo, zarr_output as zo
    mesh = load_mesh()
    g0 = _signed(mesh)
    npes = min(2, NDEV)
    part = (partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D) if npes == 1 else
            partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, npes))
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sp = shard_mesh.partition_state(g0, part)
    fs, spec = ish.folded_state(sp)
    jmesh = halo.device_mesh("p", devices=jax.devices()[:npes])
    placed = ish._to_global_sharded(fs, spec, jmesh)
    d = tmp_path / "r_avg"
    zo.write_restart(d, placed, sm, part, step=3, calendar_date="1958-01-01", dt_stage=DT)
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    loaded, meta = zo.read_restart(d, mesh, ser)
    for f in ICE_AVG_FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(loaded, f))[: mesh.nod2D],
                                      getattr(g0, f), err_msg=f)
    assert ice_step.init_ice_avg(loaded, ICE2) is loaded                  # mid-cycle: kept
    # a restart without the fields (an ice_ave_steps == 1 writer)
    g1 = ice_step.init_ice_avg(g0, IceConfig())
    sp1 = shard_mesh.partition_state(g1, ser)
    fs1, spec1 = ish.folded_state(sp1)
    placed1 = ish._to_global_sharded(fs1, spec1, halo.device_mesh("p", devices=jax.devices()[:1]))
    sm1 = shard_mesh.build_sharded_mesh(mesh, ser)
    d1 = tmp_path / "r_n1"
    zo.write_restart(d1, placed1, sm1, ser, step=3, calendar_date="1958-01-01", dt_stage=DT)
    import zarr
    assert not set(ICE_AVG_FIELDS) & set(zarr.open_group(str(d1), mode="r").array_keys())
    l1, _ = zo.read_restart(d1, mesh, ser)
    assert all(getattr(l1, f) is None for f in ICE_AVG_FIELDS)
    i1 = ice_step.init_ice_avg(l1, ICE2)
    assert np.all(np.asarray(i1.ice_steps_since_upd) == 1) and np.all(np.asarray(i1.ice_update) == 1)


# ==========================================================================
# 3. CORE2 dense, N = 2: schedule, ice_dt, means in the model, fluxes from the held ice
# ==========================================================================
@pytest.fixture(scope="module")
def core2():
    from fesom_jax import surface_forcing
    from fesom_jax.phc_ic import cold_start_state
    mesh = load_mesh(CORE2_MESH)
    state = cold_start_state(mesh, IC_DIR)                  # PHC IC + the seeded sea ice
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    op = ssh.build_ssh_operator(mesh, dt=DT)
    n = 5
    dates = surface_forcing.dates_for_steps(YEAR, DT, n)
    stack = cf.stack(dates)
    return dict(mesh=mesh, state=state, cf=cf, op=op, stack=stack, n=n, fs=cf.static)


def _sf(fx, i):
    return jax.tree.map(lambda x: x[i], fx["stack"])


def _dense_run(fx, cfg, n):
    """n dense jitted steps (step 1 bootstraps AB2); returns [state_0, ..., state_n]."""
    from fesom_jax import step as stepmod
    st = ice_step.init_ice_avg(fx["state"], cfg)
    out = [st]
    zero = jnp.zeros((fx["mesh"].elem2D, 2))
    for i in range(n):
        st = stepmod.step_jit(st, fx["mesh"], fx["op"], zero, dt=DT, is_first_step=(i == 0),
                              step_forcing=_sf(fx, i), forcing_static=fx["fs"], ice_cfg=cfg,
                              kpp_cfg=KppConfig())
        out.append(st)
    return out


@have_forcing
def test_core2_n2_schedule_and_means(core2):
    fx = core2
    traj = _dense_run(fx, ICE2, fx["n"])
    for s in range(1, fx["n"] + 1):
        prev, cur = traj[s - 1], traj[s]
        upd_now = (s % 2) == 1                                 # updates on steps 1, 3, 5
        assert np.all(np.asarray(cur.ice_update) == int(upd_now)), f"step {s}"
        assert np.all(np.asarray(cur.ice_steps_since_upd) == (0 if upd_now else 1)), f"step {s}"
        for f in _PROG + _HELD:
            a, b = np.asarray(getattr(prev, f)), np.asarray(getattr(cur, f))
            if upd_now and s > 1 and f in ("a_ice", "m_ice", "u_ice", "ice_flx_h"):
                assert not np.array_equal(a, b), f"step {s}: {f} did not change on an update step"
            elif not upd_now:
                np.testing.assert_array_equal(b, a, err_msg=f"step {s}: {f} not held")
        # the means: the previous step's flag decides (ocean2ice runs before the decision)
        upd_prev = s == 1 or ((s - 1) % 2) == 1
        T_in = np.asarray(prev.T[:, 0])
        uw = np.asarray(cur.uvnode[:, 0, 0])      # the step returns the uvnode of its input uv
        if upd_prev:
            np.testing.assert_array_equal(np.asarray(cur.srfoce_temp), T_in)
            np.testing.assert_allclose(np.asarray(cur.srfoce_u), uw, rtol=1e-13, atol=1e-18)
        else:
            want = (np.asarray(prev.srfoce_temp) * 1.0 + T_in) / 2.0
            np.testing.assert_array_equal(np.asarray(cur.srfoce_temp), want)
            np.testing.assert_allclose(np.asarray(cur.srfoce_u), uw / 2.0, rtol=1e-13, atol=1e-18)
    assert np.isfinite(np.asarray(traj[-1].T)).all()


def _ice_call(fx, state, cfg, i, dt, use_virt_salt):
    from fesom_jax import ice_evp
    bn = ice_evp.boundary_node_mask(fx["mesh"])
    fn = jax.jit(lambda s, sf: ice_step.ice_surface_step(
        cfg, fx["mesh"], s, sf, fx["fs"], dt=dt, boundary_node=bn, use_virt_salt=use_virt_salt))
    return fn(state, _sf(fx, i))


def _close(a, b, what, rtol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    d = float(np.max(np.abs(a - b)))
    assert d <= rtol * scale, f"{what}: max|diff| {d:.3e} > {rtol:.0e} x {scale:.3e}"
    return d


@have_forcing
@pytest.mark.parametrize("use_virt_salt", [True, False])
def test_core2_ice_dt_and_held_fluxes(core2, use_virt_salt):
    """(a) The first step of a cycle (an update, ocean2ice takes the current values) at N = 2 and
    dt is the N = 1 ice step at 2*dt: ice_dt = N*dt. (b) The step after it (no update) returns the
    held ice state and builds the ocean fluxes from the held thermo outputs: heat = shortwave
    penetration of -flx_h, water = -flx_fw (+ the zstar balance from the held evaporation /
    sublimation / growth / a_old and this step's precipitation), stress = the blend of the held
    u_ice / a_ice with the bulk of the running means."""
    from fesom_jax import forcing as _forcing
    fx = core2
    mesh, fs = fx["mesh"], fx["fs"]
    s0 = ice_step.init_ice_avg(fx["state"], ICE2)
    o2 = _ice_call(fx, s0, ICE2, 0, DT, use_virt_salt)
    o1 = _ice_call(fx, fx["state"], IceConfig(whichEVP=1), 0, 2 * DT, use_virt_salt)
    o1_dt = _ice_call(fx, fx["state"], IceConfig(whichEVP=1), 0, DT, use_virt_salt)
    # N = 2 at dt must be the N = 1 step at 2*dt up to round-off (the update branch is compiled
    # inside a conditional, so XLA may round it differently, and the mEVP iteration amplifies
    # that), and far from the N = 1 step at dt (the ice_dt a missing N*dt would give)
    for f in ("a_ice", "m_ice", "m_snow", "u_ice", "v_ice", "water_flux", "heat_flux"):
        a, b, c = (np.asarray(getattr(o, f)) for o in (o2, o1, o1_dt))
        same, wrong = float(np.max(np.abs(a - b))), float(np.max(np.abs(c - b)))
        print(f"{f:12s} |N2(dt) - N1(2dt)| {same:.3e}   |N1(dt) - N1(2dt)| {wrong:.3e}")
        assert wrong > 0 and same <= 1e-6 * wrong, f"{f}: ice_dt is not 2*dt"
    # (b) the in-between step, from the state the first step leaves
    s1 = dataclasses.replace(s0, **{f: getattr(o2, f) for f in _PROG + ICE_AVG_FIELDS})
    assert int(s1.ice_steps_since_upd[0]) == 0 and int(s1.ice_update[0]) == 1
    o = _ice_call(fx, s1, ICE2, 1, DT, use_virt_salt)
    assert int(o.ice_update[0]) == 0 and int(o.ice_steps_since_upd[0]) == 1
    for f in _PROG + _HELD:
        np.testing.assert_array_equal(np.asarray(getattr(o, f)), np.asarray(getattr(s1, f)),
                                      err_msg=f"{f} not held")
    sf = _sf(fx, 1)
    open_water = fs.open_water
    heat, sw3 = _forcing.cal_shortwave_rad(mesh, -s1.ice_flx_h, sf.shortwave, sf.chl,
                                           open_water & (s1.a_ice <= 0.0))
    _close(o.heat_flux, heat, "heat_flux")
    _close(o.sw_3d, sw3, "sw_3d")
    wf = -s1.ice_flx_fw
    if not use_virt_salt:
        wf = ice_coupling.fresh_water_balance_zstar(
            wf, s1.ice_evaporation, s1.ice_sublimation, sf.prec_rain, sf.prec_snow, s1.ice_a_old,
            fs.runoff_node, s1.ice_thdgr, s1.ice_thdgrsn, fs.areasvol_surf, fs.ocean_area, ICE2)
    _close(o.water_flux, wf, "water_flux")
    bulk = _forcing.bulk_surface_fluxes(mesh, sf.u_air, sf.v_air, sf.shum, sf.shortwave,
                                        sf.longwave, sf.Tair, sf.prec_rain, sf.prec_snow,
                                        o.srfoce_temp, o.srfoce_u, o.srfoce_v)
    ss, sns = ice_coupling.ice_oce_fluxes_mom(mesh, s1.a_ice, s1.u_ice, s1.v_ice, o.srfoce_u,
                                              o.srfoce_v, bulk.stress_node_surf, open_water, ICE2)
    _close(o.stress_surf, ss, "stress_surf")
    _close(o.stress_node_surf, sns, "stress_node_surf")
    # the in-between step's ocean2ice took the current values (the previous step updated)
    np.testing.assert_array_equal(np.asarray(o.srfoce_temp), np.asarray(s1.T[:, 0]))


# ==========================================================================
# 4. Dense == sharded (4 fake CPU devices, dist_4) at N = 2: test_step_sharded.py
#    (test_ice_ave_steps_sharded_owned_matches; it needs the 4-device sharding group)
# ==========================================================================


# ==========================================================================
# 5. Restart mid-cycle (odd step) and odd scan lengths, through run_from_config
# ==========================================================================
def _serial_setup(fx):
    part = partit.synth_serial(fx["mesh"].nod2D, fx["mesh"].elem2D, fx["mesh"].edge2D)
    sm = shard_mesh.build_sharded_mesh(fx["mesh"], part)
    sop = ssh.partition_ssh_operator(fx["op"], part)
    return dict(mesh=fx["mesh"], part=part, sm=sm, sop=sop, year=YEAR)


def _worst(a, b, part, sm):
    from fesom_jax.integrate_sharded import unfold_state
    a, b = unfold_state(a, 1), unfold_state(b, 1)
    worst, exact = 0.0, True
    for fld in dataclasses.fields(State):
        A = getattr(a, fld.name)
        if A is None:
            assert getattr(b, fld.name) is None, fld.name
            continue
        A, B = np.asarray(A), np.asarray(getattr(b, fld.name))
        md = int((part.myDim_nod2D if A.shape[1] == sm.Lmax["nod"] else part.myDim_elem2D)[0])
        d = float(np.max(np.abs(A[0, :md] - B[0, :md])))
        if fld.name in ("ice_steps_since_upd", "ice_update"):
            assert d == 0, fld.name
        exact = exact and d == 0
        worst = max(worst, d)
    return worst, exact


@have_forcing
def test_core2_n2_restart_mid_cycle(core2, tmp_path):
    """5 steps continuous vs 3 steps -> restart (step 3 = an update step; the restart carries
    counter 0, flag 1 and the held fluxes) -> 2 steps. Tolerance: the driver's restart-seam gate
    (test_run_entry.py, 5e-3: the seam step is a bootstrap call, not a scan iteration)."""
    import zarr
    from fesom_jax.run import run_from_config
    from fesom_jax.run_config import RunConfig
    fx = core2
    common = _serial_setup(fx)
    kw = dict(kpp=KppConfig(), ice=ICE2, dt=DT)
    n, m = fx["n"], 3
    cont = run_from_config(RunConfig(n_steps=n, **kw), state0=fx["state"], forcing=fx["cf"],
                           forcing_stack=fx["stack"], **common)
    r1 = tmp_path / "r_mid"
    run_from_config(RunConfig(n_steps=m, restart_out=str(r1), **kw), state0=fx["state"],
                    forcing=fx["cf"], forcing_stack=jax.tree.map(lambda x: x[:m], fx["stack"]),
                    **common)
    g = zarr.open_group(str(r1), mode="r")
    assert set(ICE_AVG_FIELDS) <= set(g.array_keys())
    assert np.all(g["ice_steps_since_upd"][:] == 0) and np.all(g["ice_update"][:] == 1)
    chained = run_from_config(RunConfig(n_steps=n - m, restart_in=str(r1), **kw),
                              forcing=fx["cf"], forcing_stack=jax.tree.map(lambda x: x[m:], fx["stack"]),
                              **common)
    assert chained.step == n
    worst, exact = _worst(cont.state_p, chained.state_p, common["part"], common["sm"])
    print(f"ICE2_RESTART_MID_CYCLE owned max|diff| {worst:.3e} (bitwise: {exact})")
    assert worst < 5e-3


_DIAG = ("sigma11", "sigma12", "sigma22")    # the VP-kink diagnostic (test_step_sharded.py)


def _per_field(a, b, part, sm):
    from fesom_jax.integrate_sharded import unfold_state
    a, b = unfold_state(a, 1), unfold_state(b, 1)
    out = {}
    for fld in dataclasses.fields(State):
        A = getattr(a, fld.name)
        if A is None:
            continue
        A, B = np.asarray(A), np.asarray(getattr(b, fld.name))
        md = int((part.myDim_nod2D if A.shape[1] == sm.Lmax["nod"] else part.myDim_elem2D)[0])
        out[fld.name] = float(np.max(np.abs(A[0, :md] - B[0, :md])))
    return out


@have_forcing
def test_core2_n2_odd_scan_lengths(core2):
    """5 steps as one chunk vs chunks of 2 (2+2+1: scan lengths 1, 1, 0) and of 3 (3+2: scan
    lengths 2, 1). A chunk boundary turns a scan iteration into a separately compiled bootstrap
    call, which XLA rounds differently, and with the mEVP on the round-off grows; so the bar is
    the same comparison with ice_ave_steps = 1 (the existing driver, same physics): no field may
    differ by more than 10x its N = 1 counterpart (or 1e-6, the driver's multichunk gate), sigma
    printed, not gated. The counter and the flag must be equal."""
    from fesom_jax.run import run_from_config
    from fesom_jax.run_config import RunConfig
    fx = core2
    common = _serial_setup(fx)
    res = {}
    for ice in (IceConfig(whichEVP=1), ICE2):
        cfg = RunConfig(n_steps=fx["n"], kpp=KppConfig(), ice=ice, dt=DT)
        for k in (fx["n"], 2, 3):
            res[ice.ice_ave_steps, k] = run_from_config(
                cfg, state0=fx["state"], forcing=fx["cf"], forcing_stack=fx["stack"],
                chunk_steps=k, **common).state_p
    for k in (2, 3):
        d1 = _per_field(res[1, fx["n"]], res[1, k], common["part"], common["sm"])
        d2 = _per_field(res[2, fx["n"]], res[2, k], common["part"], common["sm"])
        print(f"\nICE2_CHUNKS chunk {k} vs {fx['n']}: owned max|diff| N=2 (N=1)")
        for f in sorted(d2, key=lambda f: -d2[f]):
            if d2[f] or d1.get(f):
                print(f"   {f:20s} {d2[f]:.3e} ({d1.get(f, float('nan')):.3e})")
        assert d2["ice_steps_since_upd"] == 0 and d2["ice_update"] == 0
        bad = [f for f in d2 if f not in _DIAG and d2[f] > max(1e-6, 10 * d1.get(f, 0.0))]
        assert not bad, f"chunk {k}: {[(f, d2[f], d1.get(f)) for f in bad]}"
