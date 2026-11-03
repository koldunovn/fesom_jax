"""The coupled ocean API: init, one coupling interval, the fields for the atmosphere, restart, monthly output.

Design (docs/coupling/awicm.md):

* The ocean runs through the production forced runner (:func:`fesom_jax.integrate_sharded.run_steps_sharded_forced`)
  on ONE device (partition ``serial``: one shard whose lanes are the global node order, so every ``(nod2D,)`` array
  of this API is in fesom_jax's global node order without a permutation), or on the first N devices of the process
  (partition ``dist_N``, :func:`init_ocean`'s ``partition``): the State is then folded ``[N*Lmax]`` over the devices,
  and the API still speaks global node order -- the forcing of :func:`ocean_interval` is laid onto the lanes inside
  (:func:`to_lanes`, a gather; halo lanes take their node's value, pad lanes 1.0 as ``partition_step_forcing``), and
  :func:`ocean_to_atm` / :func:`global_view` gather the owner lanes back to global node order (replicated on the
  devices). Restarts and monthly means are written in global order either way (``layout="global"``), so a restart
  is interchangeable between 1 and N devices. The runner's compiled program for one
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
import functools
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import halo, jra55, partit, shard_mesh, ssh, sss_runoff, surface_forcing, ushow_output, zarr_output
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
#: where the FESOM2 ``dist_<N>`` partitions of the CORE2 mesh are (FESOM_DIST_DIR overrides)
DEFAULT_DIST_DIR = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2"


def _monthly_fields(s) -> dict:
    """The monthly-mean stream of the coupled ocean (run.py's monthly fields + snow), summed every step."""
    return {"temp": s.T, "salt": s.S, "u": s.uvnode[:, :, 0], "v": s.uvnode[:, :, 1], "ssh": s.eta_n,
            "a_ice": s.a_ice, "m_ice": s.m_ice, "m_snow": s.m_snow}


#: optional fields of the monthly stream (init_ocean's ``monthly_extra``; FESOM2's io names): ``w`` the vertical
#: velocity at the interfaces (nod2D, nl) [m s-1] (the overturning from w), ``fw`` the step's water_flux (nod2D,)
#: [m s-1, + = the ocean loses fresh water; basal melt under the shelves], mode F only (State.cpl_fw)
MONTHLY_EXTRA = {"w": lambda s: s.w, "fw": lambda s: s.cpl_fw}


def _monthly_fields_with(extra):
    extra = tuple(extra)

    def f(s):
        out = _monthly_fields(s)
        out.update({k: MONTHLY_EXTRA[k](s) for k in extra})
        return out
    return f


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
    flux: Any = None          # mode F constants (fesom_jax.coupled.flux.FluxConfig), None in mode B
    _o2a: Any = None          # the jitted ocean_to_atm
    # None, or a writer's submit(fn, *args) (one background thread, single process): the monthly means' store, with
    # their copy to the host, is written by that call (awicm_jax's run_coupled)
    submit: Any = None
    mevp_wide: Any = None     # gpu.mevp_wide's MevpWideHost (dist_N only), None = the per-subcycle exchange
    _lanes: Any = None        # the jitted global -> lanes gather of the forcing (dist_N)
    _views: dict = dataclasses.field(default_factory=dict)   # the jitted lanes -> global gathers (dist_N)

    @property
    def nod2D(self) -> int:
        return int(self.mesh.nod2D)

    @property
    def npes(self) -> int:
        return int(self.part.npes)

    def device_mesh(self):
        """The runner's device mesh (``halo.device_mesh`` over ``halo.first_devices(npes)``, axis ``"p"``)."""
        from jax.sharding import Mesh
        return Mesh(np.asarray(halo.first_devices(self.npes)), ("p",))

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
            if self.mode == "flux":
                from .flux import dummy_seq
                ro = getattr(self.flux, "runoff", "climatology")
                seq_p = dummy_seq(int(nsteps), n, runoff=ro) if self.npes == 1 else \
                    dummy_seq(int(nsteps), int(self.sm.Lmax["nod"]), self.npes, runoff=ro)
            else:
                z = np.zeros((nsteps, n), np.float64)
                seq = surface_forcing.StepForcing(u_air=z, v_air=z, shum=z, shortwave=z, longwave=z, Tair=z,
                                                  prec_rain=z, prec_snow=z, Ssurf_month=z, chl=z)
                seq_p = shard_mesh.partition_step_forcing(seq, self.part)
            jfn, args, _ = run_steps_sharded_forced(
                self.sm, state, self.sop, self.stress_p, seq_p, self.fs_p, int(nsteps), dt=self.dt, npes=self.npes,
                bootstrap_ab2=bool(bootstrap), state_is_folded=True, return_folded=True,
                boundary_node_p=self.bn_p, reuse_executable=True, return_executable=True,
                sample_fn=self.sample_fn, ssh_gather=self.gather, precision=self.precision,
                **({} if self.mevp_wide is None else {"mevp_wide": self.mevp_wide}), **self.phys)
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
               mode: str = "bulk", sss_restoring: bool = False, monthly_out: str | None = None, log=print,
               partition: str | None = None, dist_dir: str | None = None, cpl_runoff: str = "climatology",
               monthly_extra: tuple = (), cpl_cavity: bool | None = None):
    """Build the coupled ocean. Returns ``(state, static)``: the folded device :class:`~fesom_jax.state.State`
    (working precision) and the :class:`OceanStatic`.

    ``config``: a RunConfig YAML path or object (default ``configs/core2_cpl.yaml``). ``restart_in``: a restart
    store (``zarr_output.write_restart``), ``"config"`` (default) = the config's ``restart_in``, ``None`` = the
    PHC cold start with FESOM's initial sea ice (:func:`fesom_jax.phc_ic.cold_start_state`, ``ic_dir``) at
    ``start = (year, month, day)``. The calendar counts steps from ``year0``-01-01 (the CORE2 hindcast restarts are
    stamped that way: fesom.1989.001.00000 is step 543504 from 1958). ``nsteps``: ocean steps per coupling interval
    (4 = 7200 s at dt 1800). ``mode``: ``"bulk"`` (B) or ``"flux"`` (F). ``sss_restoring`` (default off, as
    AWI-ESM3 v3.5.1's namelist.tra surf_relax_s = 0): the SSS restoring of the standalone forcing. ``monthly_out``:
    a directory for the monthly means (``<dir>/<YYYY>_<MM>``, run.py's layout). ``partition`` (default: the
    config's): ``serial`` (one device) or ``dist_N`` (the FESOM2 partition of N ranks from ``dist_dir``, default
    $FESOM_DIST_DIR or :data:`DEFAULT_DIST_DIR`, over the first N devices of the process; ``gpu.lane_order`` and
    ``gpu.mevp_wide`` of the config apply as in the standalone runner). ``cpl_runoff`` (mode F): ``"climatology"``
    (default: the CORE2 runoff file, hydr_oce/calv_oce ignored) or ``"coupled"`` (the runoff mapper's hydr_oce +
    calv_oce and the calving's enthalpy of fusion, :mod:`fesom_jax.coupled.flux`). ``monthly_extra``: more fields
    of the monthly means (:data:`MONTHLY_EXTRA`: ``"w"``, ``"fw"``; default none: the stream unchanged).
    ``cpl_cavity`` (mode F): the cavity branches of the coupled flux (flux.CAVITY_ENV, :mod:`fesom_jax.coupled.flux`);
    None = follow the ocean's cavity switch (the config's ``cavity`` block of lane L-OCN, when it has one; off
    otherwise)."""
    monthly_extra = tuple(monthly_extra or ())
    bad = [k for k in monthly_extra if k not in MONTHLY_EXTRA]
    if bad or ("fw" in monthly_extra and mode != "flux"):
        raise ValueError(f"monthly_extra {monthly_extra}: known {tuple(MONTHLY_EXTRA)}, fw in mode 'flux' only")
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
    if cpl_cavity is None:
        cpl_cavity = getattr(cfg, "cavity", None) is not None
    if cpl_cavity and mode != "flux":
        raise ValueError("cpl_cavity is a mode F ('flux') switch")
    from .flux import CAVITY_ENV
    if cpl_cavity:
        os.environ[CAVITY_ENV] = "1"           # read at trace time by the coupled ice step (the process's only user)
    else:
        os.environ.pop(CAVITY_ENV, None)
    precision = cfg.precision
    dt = float(cfg.dt)

    mdir = mesh_dir or (cfg.mesh if cfg.mesh and os.path.isdir(str(cfg.mesh)) else DEFAULT_MESH)
    mesh = load_mesh(mdir)
    pspec = str(cfg.partition if partition is None else partition)
    npes = 1 if pspec in ("None", "serial", "1", "dist_1") else int(pspec.replace("dist_", ""))
    if npes == 1:
        part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    else:
        halo.first_devices(npes)                  # raises when the process has too few devices
        ddir = dist_dir or os.environ.get("FESOM_DIST_DIR") or DEFAULT_DIST_DIR
        part = partit.read_partition(ddir, npes)
        if cfg.lane_order():          # the lanes inside the shards in a cache-friendly order (global ids unchanged)
            from ..lane_order import reorder_lanes
            part = reorder_lanes(part, mesh, cfg.lane_order())
        log(f"[ocean] partition dist_{npes} ({ddir}), lane_order {cfg.lane_order()}")
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    if npes == 1 and (sm.Lmax["nod"] != mesh.nod2D or not np.array_equal(np.asarray(part.myList_nod2D[0]),
                                                                         np.arange(mesh.nod2D))):
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
        state = folded_state(shard_mesh.partition_state(st0, part))[0]
        if npes == 1:
            state = jax.tree.map(jnp.asarray, state)
        else:
            from jax.sharding import Mesh, NamedSharding, PartitionSpec
            sh = NamedSharding(Mesh(np.asarray(halo.first_devices(npes)), ("p",)), PartitionSpec("p"))
            state = jax.tree.map(lambda x: jax.device_put(np.asarray(x), sh), state)
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
    stress_p = np.zeros((npes, sm.Lmax["elem"], 2))
    mevp_wide = None
    if npes >= 2 and cfg.ice is not None:
        from .. import ice_wide
        kw = ice_wide.mevp_wide_from_env()
        if kw >= 2:
            if cfg.ice.whichEVP != 1:
                raise ValueError(f"gpu.mevp_wide={kw} needs the mEVP sea ice (ice.whichEVP: 1)")
            mevp_wide = ice_wide.build_mevp_wide(mesh, part, sm, kw)
            log(f"[ocean] mevp_wide: K={kw}")
    geo = np.asarray(mesh.geo_coord_nod2D, np.float64)
    rot = np.asarray(mesh.coord_nod2D, np.float64)
    trig = tuple(jnp.asarray(t) for t in jra55._g2r_trig(geo[:, 0], geo[:, 1], rot[:, 0], rot[:, 1]))

    static = OceanStatic(
        cfg=cfg, mesh=mesh, part=part, sm=sm, sop=sop, gather=gather, fs_p=fs_p, bn_p=bn_p, stress_p=stress_p,
        phys=cfg.physics_kwargs(), precision=precision, dt=dt, nsteps=int(nsteps), mode=mode, year0=int(year0),
        step=step0, sss_clim=sss_clim, chl_clim=np.asarray(chl_clim), trig=trig,
        M=jnp.asarray(jra55._rotation_matrix()), sss_restoring=bool(sss_restoring), bootstrap=bootstrap,
        monthly_out=monthly_out, sample_fn=(None if monthly_out is None else
                                            _monthly_fields_with(monthly_extra) if monthly_extra else _monthly_fields),
        mevp_wide=mevp_wide)
    if mode == "flux":
        from . import flux
        state, static = flux.init_flux(state, static, runoff=cpl_runoff)
        if "fw" in monthly_extra and getattr(state, "cpl_fw", None) is None:
            state = dataclasses.replace(state, cpl_fw=jnp.zeros_like(state.a_ice))
    elif cpl_runoff != "climatology":
        raise ValueError("cpl_runoff 'coupled' is a mode F ('flux') switch")
    log(f"[ocean] CORE2 nod2D={mesh.nod2D} npes={npes} dt={dt} precision={precision} mode={mode} nsteps={nsteps} "
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


def _div1000(x):
    """``x / 1000`` correctly rounded in ``x``'s precision (the reader's numpy division). The A100's float32 division
    is not (1 ULP off on ~half the operands, measured, job 28643779), so a float32 ``x`` is divided in float64 (IEEE
    on the GPU) and rounded once to float32: double rounding is exact for a quotient of float32 operands (53 >= 2*24
    + 2). The divisor is an array operand, so XLA cannot rewrite the division as a multiplication by 0.001."""
    return (x.astype(jnp.float64) / jnp.asarray(1000.0, jnp.float64)).astype(x.dtype)


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
    rain, snow = _rows(forcing, "rain", T, n), _rows(forcing, "snow", T, n)
    seq = surface_forcing.StepForcing(
        u_air=u, v_air=v,
        shum=_rows(forcing, "q2m", T, n),
        shortwave=_rows(forcing, "swdn", T, n),
        longwave=_rows(forcing, "lwdn", T, n),
        Tair=_rows(forcing, "t2m", T, n) - 273.15,
        prec_rain=_div1000(rain),
        prec_snow=_div1000(snow),
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
    if static.npes > 1:
        seq = to_lanes(seq, static)
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
    write = functools.partial(ushow_output.write_global_zarr,
                              attrs={"calendar_month": f"{yr:04d}-{mo:02d}", "n_samples": int(cnt),
                                     "coupled_mode": static.mode},
                              time_days=ushow_output.cf_days(datetime.date(yr, mo, 15)))
    if static.submit is None:
        write(path, mean, static.sm, static.part, static.mesh)
    else:
        static.submit(write, path, mean, static.sm, static.part, static.mesh)
    static._mon_key, static._mon_sum, static._mon_count = None, None, 0
    return path


# ---------------------------------------------------------------------------------------------------------------
# dist_N: global node order <-> the folded lanes
# ---------------------------------------------------------------------------------------------------------------
def _replicated(static):
    from jax.sharding import NamedSharding, PartitionSpec
    return NamedSharding(static.device_mesh(), PartitionSpec())


def to_lanes(seq, static: OceanStatic):
    """A forcing stack with ``[nsteps, nod2D]`` leaves in global node order -> ``[nsteps, P*Lmax]`` on the runner's
    lanes, sharded ``PartitionSpec(None, "p")``: a gather by each lane's global node id (halo lanes take their
    node's value, pad lanes 1.0: the values :func:`fesom_jax.shard_mesh.partition_step_forcing` gives them)."""
    if static._lanes is None:
        from jax.sharding import NamedSharding, PartitionSpec
        from ..zarr_output import _folded_gid_owned
        gid, _ = _folded_gid_owned(static.part, static.sm, "nod")
        idx, pad = jnp.asarray(np.where(gid >= 0, gid, 0), jnp.int32), jnp.asarray(gid < 0)

        def f(seq):
            return jax.tree.map(lambda a: jnp.where(pad, jnp.asarray(1, a.dtype), a[:, idx]), seq)
        static._lanes = jax.jit(f, out_shardings=NamedSharding(static.device_mesh(), PartitionSpec(None, "p")))
    return static._lanes(seq)


def _lanes_to_global_fn(static: OceanStatic):
    """gather(state, names) -> {name: the State field in global entity order}: each node's (element's) unique owner
    lane, by index (traceable; the caller's jit decides where the result lives)."""
    from ..zarr_output import _folded_gid_owned
    pos = {}
    for kind, n in (("nod", static.sm.nod2D), ("elem", static.sm.elem2D)):
        gid, owned = _folded_gid_owned(static.part, static.sm, kind)
        p = np.full(int(n), -1, np.int64)
        p[gid[owned]] = np.nonzero(owned)[0]
        if (p < 0).any():
            raise AssertionError(f"{kind}: {(p < 0).sum()} entities without an owner lane")
        pos[(kind, int(static.npes) * int(static.sm.Lmax[kind]))] = jnp.asarray(p, jnp.int32)

    def gather(state, names):
        out = {}
        for k in names:
            a = getattr(state, k)
            idx = [v for (kind, lead), v in pos.items() if lead == a.shape[0]]
            if len(idx) != 1:
                raise ValueError(f"State.{k}: leading extent {a.shape[0]} is not a folded node or element axis")
            out[k] = a[idx[0]]
        return out
    return gather


def global_view(state, static: OceanStatic, names) -> dict:
    """``{name: State field}`` in global node (element) order: the fields themselves with one device (serial), else
    the owner lanes gathered (replicated over the devices, one jitted program per ``names``)."""
    names = tuple(names)
    if static.npes == 1:
        return {k: getattr(state, k) for k in names}
    if names not in static._views:
        gather = _lanes_to_global_fn(static)
        static._views[names] = jax.jit(lambda s: gather(s, names), out_shardings=_replicated(static))
    return static._views[names](state)


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
    """The fields FESOM2 sends to OpenIFS (see :func:`_ocean_to_atm`); one jitted program per ``static``."""
    if static._o2a is None:
        if static.npes == 1:
            static._o2a = jax.jit(lambda s: _ocean_to_atm(s, static))
        else:                         # the owner lanes in global node order (replicated), then the same function
            import types
            names = ("T", "S", "a_ice", "m_ice", "m_snow", "t_skin", "uv") + \
                (("cpl_ice_temp", "cpl_ice_alb") if static.mode == "flux" else ())
            gather = _lanes_to_global_fn(static)

            def f(s):
                return _ocean_to_atm(types.SimpleNamespace(**gather(s, names)), static)
            static._o2a = jax.jit(f, out_shardings=_replicated(static))
    return static._o2a(state)


def _ocean_to_atm(state, static: OceanStatic) -> dict:
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
    return _restart_writer(static)(path, state, static.sm, static.part)


def ocean_restart_writer(state, static: OceanStatic):
    """:func:`write_ocean_restart` in two halves (single process): the State widened to float64 (on the device) and
    the step and calendar of ``static``, now; returns ``write(path) -> path``, the store with the copy to the host,
    which may run later on another thread while the ocean steps on (the State's arrays are not donated). The copy is
    not started here: on the step's thread it held up the coupled run's own small copies behind it."""
    return functools.partial(_restart_writer(static), state_folded=PR.cast_tree(state, keep64=True), sm=static.sm,
                             part=static.part)


def _restart_writer(static):
    y, d, _, _ = static.dates(1)[0]
    return functools.partial(zarr_output.write_restart, step=static.step, calendar_date=f"{y}-doy{d:03d}",
                             dt_stage=static.dt, layout="global", attrs={"coupled_mode": static.mode})
