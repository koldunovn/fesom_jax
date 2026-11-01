"""The coupled ocean API: init, one coupling interval, the fields for the atmosphere, restart, monthly output.

Design (docs/coupling/awicm.md):

* The ocean runs through the production forced runner (:func:`fesom_jax.integrate_sharded.run_steps_sharded_forced`)
  on ONE device (partition ``serial``: one shard whose lanes are the global node order, so every ``(nod2D,)`` array
  of this API is in fesom_jax's global node order without a permutation). The runner's compiled program for one
  interval of ``nsteps`` steps is built once (:meth:`OceanStatic.executable`) and then called directly with the
  device State and a device forcing stack: nothing goes through the host between intervals.
* Mode B (``"bulk"``): the atmosphere's near-surface fields become the :class:`~fesom_jax.surface_forcing.StepForcing`
  the file-driven run builds from JRA55-do (:func:`bulk_step_forcing`, the same operations in the same order as
  :meth:`fesom_jax.jra55.JRA55Reader.step`), so the standalone bulk formulae, sea ice and SSS/runoff path run
  unchanged. Held constant over the interval when given as ``(nod2D,)``; ``(nsteps, nod2D)`` gives one row a step
  (the bitwise check against the file-driven run).
* Mode F (``"flux"``): :mod:`fesom_jax.coupled.flux` (FESOM2's coupled flux path).
* :func:`ocean_interval` keeps the calendar (``static.step``, steps since ``year0``-01-01 as the CORE2 restarts
  count them), the month of the SSS/chlorophyll climatologies and the monthly means on the host; the device work of
  an interval is one call of the compiled program (``static.executable(...)``), which is a pure function
  ``(state, forcing stack) -> (state, per-step field sums)``.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import jra55, partit, shard_mesh, ssh, sss_runoff, surface_forcing, ushow_output, zarr_output
from .. import precision as PR
from ..mesh import load_mesh
from ..run import _chunk_dates
from ..run_config import RunConfig, load_yaml

#: mode B: the atmosphere's near-surface state, interval means on the FESOM nodes (the plan's contract):
#: 10 m wind [m s-1] east/north, 2 m temperature [K] and specific humidity [kg kg-1], downward short- and longwave
#: at the surface [W m-2], rain and snow [kg m-2 s-1], mean sea-level pressure [Pa] (accepted, not used: FESOM2's
#: l_mslp is off in AWI-ESM3 v3.5.1 and in the CORE2 configuration)
MODE_B_FIELDS = ("u10", "v10", "t2m", "q2m", "swdn", "lwdn", "rain", "snow", "msl")
#: mode F: the namcouple fields FESOM2 receives (cpl_driver.F90 cpl_recv, __oifs), after OASIS (fesom_jax.coupled.flux)
MODE_F_FIELDS = ("taux_oce", "tauy_oce", "taux_ico", "tauy_ico", "prec_oce", "snow_oce", "evap_oce", "subl_oce",
                 "heat_oce", "heat_ico", "heat_swo", "hydr_oce", "calv_oce", "u10w_oce", "v10w_oce", "tsk_ico")
#: ocean -> atmosphere, in cpl_driver's send order (cpl_send, __oifs)
OCEAN_TO_ATM = ("sst_feom", "sie_feom", "snt_feom", "ist_feom", "sia_feom", "u_feom", "v_feom", "sit_feom")

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "core2_cpl.yaml")
DEFAULT_MESH = "/work/ab0995/a270088/port_jax/data/mesh_core2"


def _monthly_fields(s) -> dict:
    """The monthly-mean stream of the coupled ocean (run.py's monthly fields + snow), summed every step."""
    return {"temp": s.T, "salt": s.S, "u": s.uvnode[:, :, 0], "v": s.uvnode[:, :, 1], "ssh": s.eta_n,
            "a_ice": s.a_ice, "m_ice": s.m_ice, "m_snow": s.m_snow}


@dataclasses.dataclass
class OceanStatic:
    """Everything of the coupled ocean that is not the State: configuration, mesh, partition, operators, the
    climatologies, the rotation tables, the compiled interval programs, the calendar and the monthly means.
    Host object; :func:`ocean_interval` advances its calendar."""
    cfg: RunConfig
    mesh: Any
    part: Any
    sm: Any
    sop: Any
    gather: Any
    fs_p: Any                 # ForcingStatic partitioned [1, Lmax] (runoff, areas, open water)
    bn_p: Any                 # the coastal boundary mask of the sea ice, partitioned
    stress_p: Any
    phys: dict                # the physics configs (one set of objects: the executable cache keys on their ids)
    precision: str
    dt: float
    nsteps: int
    mode: str
    year0: int                # the calendar origin: static.step counts steps from year0-01-01 00:00
    step: int                 # the absolute step of the State
    sss_clim: np.ndarray      # [12, nod2D] SSS restoring target (unused with sss_restoring off)
    chl_clim: np.ndarray      # [12, nod2D] chlorophyll (shortwave penetration)
    trig: tuple               # the g2r/r2g per-node sin/cos tables (device, float64)
    M: Any                    # the geographic -> rotated Euler matrix (device, [9] float64)
    sss_restoring: bool
    bootstrap: bool           # the next interval starts AB2 (cold start / dt change)
    monthly_out: str | None = None
    sample_fn: Any = None
    _exe: dict = dataclasses.field(default_factory=dict)
    _mon_key: Any = None
    _mon_sum: Any = None
    _mon_count: int = 0
    flux: Any = None          # mode F constants (fesom_jax.coupled.flux.FluxStatic), None in mode B

    @property
    def nod2D(self) -> int:
        return int(self.mesh.nod2D)

    def dates(self, nsteps: int, start: int | None = None):
        """``(year, day of year, second of day, month)`` of each step start of the next interval."""
        return _chunk_dates(self.year0, self.dt, self.step if start is None else start, nsteps)

    def calendar(self) -> str:
        y, d, s, _ = self.dates(1)[0]
        return f"{y:04d}-doy{d:03d}-{int(s):05d}"

    def executable(self, nsteps: int, bootstrap: bool, state):
        """The compiled interval program ``(jfn, const_args)`` of ``nsteps`` steps (built once per
        ``(nsteps, bootstrap)``): ``jfn(const_args[0], state, const_args[2], const_args[3], const_args[4], seq,
        *const_args[6:]) -> (state, sums)`` with ``seq`` a :class:`StepForcing` of ``[nsteps, nod2D]`` leaves."""
        key = (int(nsteps), bool(bootstrap))
        if key not in self._exe:
            from ..integrate_sharded import run_steps_sharded_forced
            n = self.nod2D
            z = np.zeros((nsteps, n), np.float64)
            seq = surface_forcing.StepForcing(u_air=z, v_air=z, shum=z, shortwave=z, longwave=z, Tair=z,
                                              prec_rain=z, prec_snow=z, Ssurf_month=z, chl=z)
            seq_p = shard_mesh.partition_step_forcing(seq, self.part)
            jfn, args, _ = run_steps_sharded_forced(
                self.sm, state, self.sop, self.stress_p, seq_p, self.fs_p, int(nsteps), dt=self.dt, npes=1,
                bootstrap_ab2=bool(bootstrap), state_is_folded=True, return_folded=True,
                boundary_node_p=self.bn_p, reuse_executable=True, return_executable=True,
                sample_fn=self.sample_fn, ssh_gather=self.gather, precision=self.precision, **self.phys)
            self._exe[key] = (jfn, args)
        return self._exe[key]


def _env_setup(cfg: RunConfig, sss_restoring: bool) -> None:
    """The config's ``gpu:`` switches in the environment for the life of the process (the step reads them at trace
    time; the coupled ocean is the process's only fesom_jax user), and the SSS restoring switch."""
    for k, v in cfg.gpu_env().items():
        os.environ.setdefault(k, v)
    if not sss_restoring:
        os.environ["FESOM_SURF_RELAX_S"] = "0"


def init_ocean(config=DEFAULT_CONFIG, *, mesh_dir: str | None = None, restart_in: str | None = "config",
               ic_dir: str | None = None, start: tuple | None = None, year0: int = 1958, nsteps: int = 4,
               mode: str = "bulk", sss_restoring: bool = False, monthly_out: str | None = None, log=print):
    """Build the coupled ocean. Returns ``(state, static)``: the folded device :class:`~fesom_jax.state.State`
    (working precision) and the :class:`OceanStatic`.

    ``config``: a RunConfig YAML path or object (default ``configs/core2_cpl.yaml``). ``restart_in``: a restart
    store (``zarr_output.write_restart``), ``"config"`` (default) = the config's ``restart_in``, ``None`` = the
    PHC cold start with FESOM's initial sea ice (:func:`fesom_jax.phc_ic.cold_start_state`, ``ic_dir``) at
    ``start = (year, month, day)``. The calendar counts steps from ``year0``-01-01 (the CORE2 hindcast restarts are
    stamped that way: fesom.1989.001.00000 is step 543504 from 1958). ``nsteps``: ocean steps per coupling interval
    (4 = 7200 s at dt 1800). ``mode``: ``"bulk"`` (B) or ``"flux"`` (F). ``sss_restoring`` (default off, as
    AWI-ESM3 v3.5.1's namelist.tra surf_relax_s = 0): the SSS restoring of the standalone forcing. ``monthly_out``:
    a directory for the monthly means (``<dir>/<YYYY>_<MM>``, run.py's layout)."""
    cfg = load_yaml(config) if isinstance(config, (str, os.PathLike)) else config
    if restart_in != "config":
        cfg = dataclasses.replace(cfg, restart_in=restart_in)
    cfg.validate()
    if mode not in ("bulk", "flux"):
        raise ValueError(f"mode must be 'bulk' or 'flux', got {mode!r}")
    if cfg.ice is None or int(cfg.ice.ice_ave_steps) != 1:
        raise NotImplementedError("the coupled ocean runs the sea ice every ocean step (ice: {..., ice_ave_steps: 1})")
    if cfg.split_explicit() is not None:
        raise NotImplementedError("the coupled ocean uses the CG SSH solve (ssh.split_explicit is not wired)")
    if cfg.dt_ramp:
        raise NotImplementedError("dt_ramp is not supported by the coupled ocean")
    _env_setup(cfg, sss_restoring)
    precision = cfg.precision
    dt = float(cfg.dt)

    mdir = mesh_dir or (cfg.mesh if cfg.mesh and os.path.isdir(str(cfg.mesh)) else DEFAULT_MESH)
    mesh = load_mesh(mdir)
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    if sm.Lmax["nod"] != mesh.nod2D or not np.array_equal(np.asarray(part.myList_nod2D[0]),
                                                           np.arange(mesh.nod2D)):
        raise AssertionError("the serial partition is expected to hold the nodes in global order")
    from .. import ops
    if ops.ordered_sums_enabled():
        sm = shard_mesh.with_inverse_maps(sm)
    op = ssh.build_ssh_operator(mesh, dt=dt, precond_variant=1 if precision == "single" else 0)
    deg, kap = cfg.ssh_cheb()
    if deg:
        op = ssh.enable_cheb_precond(op, deg, kappa_guess=kap)
    sop = ssh.partition_ssh_operator(op, part)
    if ssh.ssh_ell_enabled():
        sop = ssh.to_ell_sharded(sop)
    gather = None
    if cfg.ssh_gathered():
        op_g = op
        if cfg.ale is not None and ssh.ssh_inc_ell_enabled():
            op_g = ssh.with_increment(op_g, mesh)
        if ssh.ssh_ell_enabled():
            op_g = ssh.to_ell(op_g)
        gather = ssh.build_ssh_gather(op_g, mesh, part)

    # --- initial State (folded [nod2D, ...] on the device) ---
    from ..integrate_sharded import folded_state
    from ..ice_step import init_ice_avg
    if cfg.restart_in is not None:
        state, meta = zarr_output.read_restart(cfg.restart_in, mesh, part)
        step0 = int(meta["step"])
        bootstrap = float(meta.get("dt_stage", dt)) != dt
        log(f"[ocean] restart {cfg.restart_in}: step {step0} ({meta.get('calendar_date')}), dt_stage "
            f"{meta.get('dt_stage')}{' -> AB2 bootstrap' if bootstrap else ''}")
        if precision == "double":
            state = PR.drop_carry(state)
    else:
        from ..phc_ic import cold_start_state
        y, m, d = start if start is not None else (1989, 1, 1)
        el = (datetime.datetime(int(y), int(m), int(d)) - datetime.datetime(int(year0), 1, 1)).total_seconds()
        if el % dt:
            raise ValueError(f"start {start} is not on the dt={dt} grid from {year0}-01-01")
        step0 = int(el // dt)
        st0 = cold_start_state(mesh, ic_dir, xp=np)
        state = jax.tree.map(jnp.asarray, folded_state(shard_mesh.partition_state(st0, part))[0])
        bootstrap = True
        log(f"[ocean] cold start (PHC + FESOM's initial sea ice) at {y:04d}-{m:02d}-{d:02d}, step {step0}")
    state = init_ice_avg(state, cfg.ice)
    if precision == "single":
        state = PR.cast_state(state)

    # --- climatologies + static forcing (runoff, areas), the coastal mask, the rotation tables ---
    fpaths = cfg.forcing_paths()
    sss = sss_runoff.build_reader(mesh, fpaths.get("sss_path"), fpaths.get("runoff_path"))
    chl_clim = (sss_runoff.build_chl_clim(mesh, fpaths.get("chl_path")))
    sss_clim = np.stack([np.asarray(sss.month(m)) for m in range(1, 13)])
    fs = surface_forcing._static_from_mesh(mesh, sss.runoff_node, np.zeros(int(mesh.nod2D)))
    fs_p = shard_mesh.partition_forcing_static(fs, part)
    from .. import ice_evp
    bn = np.asarray(ice_evp.boundary_node_mask(mesh))
    bn_p = shard_mesh._shard_along_axis(bn, part.myList_nod2D, sm.Lmax["nod"], 0, False)
    stress_p = np.zeros((1, sm.Lmax["elem"], 2))
    geo = np.asarray(mesh.geo_coord_nod2D, np.float64)
    rot = np.asarray(mesh.coord_nod2D, np.float64)
    trig = tuple(jnp.asarray(t) for t in jra55._g2r_trig(geo[:, 0], geo[:, 1], rot[:, 0], rot[:, 1]))

    static = OceanStatic(
        cfg=cfg, mesh=mesh, part=part, sm=sm, sop=sop, gather=gather, fs_p=fs_p, bn_p=bn_p, stress_p=stress_p,
        phys=cfg.physics_kwargs(), precision=precision, dt=dt, nsteps=int(nsteps), mode=mode, year0=int(year0),
        step=step0, sss_clim=sss_clim, chl_clim=np.asarray(chl_clim), trig=trig,
        M=jnp.asarray(jra55._rotation_matrix()), sss_restoring=bool(sss_restoring), bootstrap=bootstrap,
        monthly_out=monthly_out, sample_fn=_monthly_fields if monthly_out is not None else None)
    if mode == "flux":
        from . import flux
        state, static = flux.init_flux(state, static)
    log(f"[ocean] CORE2 nod2D={mesh.nod2D} dt={dt} precision={precision} mode={mode} nsteps={nsteps} "
        f"sss_restoring={sss_restoring} calendar {static.calendar()}")
    return state, static


# ---------------------------------------------------------------------------------------------------------------
# mode B: the atmosphere's near-surface state -> the standalone StepForcing
# ---------------------------------------------------------------------------------------------------------------
def g2r(u, v, trig, M):
    """Geographic (east, north) -> the model's rotated frame, :func:`fesom_jax.jra55._vector_g2r` (FESOM2
    vector_g2r) op for op. Eager element-wise calls keep it bitwise the host numpy version (no FMA contraction)."""
    return jra55._vector_g2r(u, v, None, None, None, None, M, trig=trig)


def r2g(u, v, trig, M):
    """The model's rotated frame -> geographic (east, north): FESOM2 ``vector_r2g`` (gen_modules_rotate_grid.F90:
    164-202, the transposed Euler matrix)."""
    sgl, cgl, sgo, cgo, srl, crl, sro, cro = trig
    txg = -v * srl * cro - u * sro
    tyg = -v * srl * sro + u * cro
    tzg = v * crl
    txr = M[0] * txg + M[3] * tyg + M[6] * tzg
    tyr = M[1] * txg + M[4] * tyg + M[7] * tzg
    tzr = M[2] * txg + M[5] * tyg + M[8] * tzg
    v_g = -sgl * cgo * txr - sgl * sgo * tyr + cgl * tzr
    u_g = -sgo * txr + cgo * tyr
    return u_g, v_g


def _rows(forcing: dict, name: str, nsteps: int, n: int):
    a = forcing[name]
    a = a if isinstance(a, jax.Array) else jnp.asarray(np.asarray(a))
    if a.shape == (n,):
        return jnp.broadcast_to(a, (nsteps, n))
    if a.shape == (nsteps, n):
        return a
    raise ValueError(f"forcing[{name!r}] has shape {a.shape}; need ({n},) or ({nsteps}, {n})")


def bulk_step_forcing(forcing: dict, static: OceanStatic, dates) -> surface_forcing.StepForcing:
    """Mode B: the :class:`StepForcing` stack ``[nsteps, nod2D]`` of one interval from the atmosphere's fields
    (:data:`MODE_B_FIELDS`, each ``(nod2D,)`` held over the interval or ``(nsteps, nod2D)``), with the conversions
    of :meth:`fesom_jax.jra55.JRA55Reader.step` in its order: the wind rotated into the model frame (FESOM2
    force_rotation), ``Tair = t2m - 273.15`` [degC], ``prec = rain / 1000`` [m s-1]; the SSS target and the
    chlorophyll of each step's month. Eager element-wise device ops (bitwise the host reader on the same input);
    the result in the working precision."""
    n, T = static.nod2D, len(dates)
    missing = [k for k in MODE_B_FIELDS if k != "msl" and k not in forcing]
    if missing:
        raise KeyError(f"mode B forcing lacks {missing}")
    u, v = _rows(forcing, "u10", T, n), _rows(forcing, "v10", T, n)
    if jra55.FORCE_ROTATION:
        u, v = g2r(u, v, static.trig, static.M)
    months = np.array([d[3] for d in dates]) - 1
    seq = surface_forcing.StepForcing(
        u_air=u, v_air=v,
        shum=_rows(forcing, "q2m", T, n),
        shortwave=_rows(forcing, "swdn", T, n),
        longwave=_rows(forcing, "lwdn", T, n),
        Tair=_rows(forcing, "t2m", T, n) - 273.15,
        prec_rain=_rows(forcing, "rain", T, n) / 1000.0,
        prec_snow=_rows(forcing, "snow", T, n) / 1000.0,
        Ssurf_month=jnp.asarray(static.sss_clim[months]),
        chl=jnp.asarray(static.chl_clim[months]))
    return PR.cast_tree(seq) if static.precision == "single" else seq


# ---------------------------------------------------------------------------------------------------------------
# one coupling interval
# ---------------------------------------------------------------------------------------------------------------
def ocean_interval(state, static: OceanStatic, forcing: dict, nsteps: int | None = None, mode: str | None = None):
    """Run one coupling interval: ``nsteps`` (default ``static.nsteps``) ocean steps from ``state`` with the
    atmosphere's ``forcing`` (``{name: (nod2D,) or (nsteps, nod2D)}`` on the FESOM nodes in global node order:
    :data:`MODE_B_FIELDS` for ``mode="bulk"``, :data:`MODE_F_FIELDS` for ``"flux"``). Returns the new State (on the
    device) and advances ``static``'s calendar; adds the interval to the monthly means and writes a month when the
    interval starts a new one."""
    nsteps = static.nsteps if nsteps is None else int(nsteps)
    mode = static.mode if mode is None else mode
    if mode != static.mode:
        raise ValueError(f"the ocean was built for mode {static.mode!r}, not {mode!r}")
    dates = static.dates(nsteps)
    if mode == "bulk":
        seq = bulk_step_forcing(forcing, static, dates)
    else:
        from . import flux
        state, seq = flux.flux_step_forcing(state, forcing, static, dates)
    jfn, a = static.executable(nsteps, static.bootstrap, state)
    out = jfn(a[0], state, a[2], a[3], a[4], seq, *a[6:])
    state, sums = out if static.sample_fn is not None else (out, None)
    static.bootstrap = False
    if sums is not None:
        key = (int(dates[0][0]), int(dates[0][3]))
        if static._mon_key is not None and key != static._mon_key:
            flush_monthly(static)
        if static._mon_sum is None:
            static._mon_key, static._mon_sum, static._mon_count = key, sums, 0
        else:
            static._mon_sum = jax.tree.map(jnp.add, static._mon_sum, sums)
        static._mon_count += nsteps
    static.step += nsteps
    return state


def flush_monthly(static: OceanStatic) -> str | None:
    """Write the monthly mean accumulated so far (``<monthly_out>/<YYYY>_<MM>``) and reset; None if empty."""
    if static._mon_sum is None or static.monthly_out is None:
        return None
    (yr, mo), cnt = static._mon_key, static._mon_count
    mean = jax.tree.map(lambda x: x / cnt, static._mon_sum)
    path = f"{static.monthly_out}/{yr:04d}_{mo:02d}"
    ushow_output.write_global_zarr(path, mean, static.sm, static.part, static.mesh,
                                   attrs={"calendar_month": f"{yr:04d}-{mo:02d}", "n_samples": int(cnt),
                                          "coupled_mode": static.mode},
                                   time_days=ushow_output.cf_days(datetime.date(yr, mo, 15)))
    static._mon_key, static._mon_sum, static._mon_count = None, None, 0
    return path


# ---------------------------------------------------------------------------------------------------------------
# ocean -> atmosphere
# ---------------------------------------------------------------------------------------------------------------
TMELT = 273.15
AIMIN = 0.001          # ice_thermo_cpl.F90 Aimin


def tfrez_k(S):
    """The freezing point of seawater [K] as ice_thermo_cpl.F90 writes ice_temp where A <= Aimin."""
    return -0.0575 * S + 1.7105e-3 * jnp.sqrt(jnp.maximum(S, 0.0) ** 3) - 2.155e-4 * (S * S) + TMELT


def ice_albedo_cpl(h, hsn, t, cfg):
    """ice_thermo_cpl.F90 ``ice_albedo`` without melt ponds, ``alb_tramp = 0`` and ``h_snowscale = 0`` (the
    AWI-ESM3 v3.5.1 namelist.ice): frozen (t < 273.15 K) snow/ice albedo albsn/albi, melting albsnm/albim; snow
    when hsn > 1 mm; 0.066 (ocean) where h = 0."""
    frozen = t < TMELT
    alb_snow = jnp.where(frozen, cfg.albsn, cfg.albsnm)
    alb_bare = jnp.where(frozen, cfg.albi, cfg.albim)
    return jnp.where(h > 0.0, jnp.where(hsn > 0.001, alb_snow, alb_bare), 0.066)


def ocean_to_atm(state, static: OceanStatic) -> dict:
    """The fields FESOM2 sends to OpenIFS (gen_forcing_couple.F90 update_atm_forcing, ``__oifs``), each
    ``(nod2D,)`` in global node order, INSTANT at the call:

    * ``sst_feom`` = T(1) + tmelt [K];  ``sie_feom`` = a_ice [0..1];  ``snt_feom`` = m_snow [m, grid-cell mean];
    * ``ist_feom`` = ice_temp * a_ice [K x fraction: the concentration-weighted ice surface temperature, which the
      atmosphere divides by the received ice fraction];
    * ``sia_feom`` = ice albedo [0..1];  ``sit_feom`` = m_ice [m, grid-cell mean = effective thickness];
    * ``u_feom``/``v_feom`` = the surface node velocity UVnode(:,1,:) rotated to geographic east/north [m s-1].

    ice_temp/albedo: mode F carries FESOM2's coupled ice_temp/ice_alb (:mod:`fesom_jax.coupled.flux`); mode B has the
    standalone thermodynamics, so ice_temp = t_skin + 273.15 where a_ice > Aimin and the freezing point of the
    surface salinity elsewhere (ice_thermo_cpl's rule), and the albedo is ice_thermo_cpl's ``ice_albedo`` of that
    temperature (:func:`ice_albedo_cpl`). The working precision (float32 in single precision)."""
    from ..pp import compute_vel_nodes
    if static.mode == "flux":
        from . import flux
        ice_temp, ice_alb = flux.ice_temp_alb(state, static)
    else:
        ice_temp = jnp.where(state.a_ice > AIMIN, state.t_skin + TMELT, tfrez_k(state.S[:, 0]))
        ice_alb = ice_albedo_cpl(state.m_ice, state.m_snow, ice_temp, static.cfg.ice)
    # FESOM2 sends UVnode as compute_vel_nodes made it at the top of the step from the carried velocity (step.py
    # drift fix 1); state.uvnode is one step older, so recompute it from state.uv
    uvn = compute_vel_nodes(static.mesh, state.uv)[:, 0, :]
    u_g, v_g = r2g(uvn[:, 0].astype(jnp.float64), uvn[:, 1].astype(jnp.float64), static.trig, static.M)
    wp = state.T.dtype
    return {"sst_feom": state.T[:, 0] + jnp.asarray(TMELT, wp),
            "sie_feom": state.a_ice,
            "snt_feom": state.m_snow,
            "ist_feom": (ice_temp * state.a_ice).astype(wp),
            "sia_feom": ice_alb.astype(wp),
            "u_feom": u_g.astype(wp), "v_feom": v_g.astype(wp),
            "sit_feom": state.m_ice}


# ---------------------------------------------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------------------------------------------
def write_ocean_restart(state, static: OceanStatic, path: str) -> str:
    """The full State as a portable restart store (zarr_output.write_restart, global layout, float64 on disk) with
    the step and calendar of ``static``; reload it with ``init_ocean(..., restart_in=path)``."""
    y, d, _, _ = static.dates(1)[0]
    return zarr_output.write_restart(path, state, static.sm, static.part, step=static.step,
                                     calendar_date=f"{y}-doy{d:03d}", dt_stage=static.dt, layout="global",
                                     attrs={"coupled_mode": static.mode})
