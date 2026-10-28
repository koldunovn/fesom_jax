"""Surface-forcing DRIVER — assembles the per-step ocean surface boundary conditions.

**The dataset is JRA55-do.** This module drives **JRA55-do** (v1.4.0). It does *not* read
CORE-II / "CORE forcing" (the Large & Yeager reanalysis-forcing dataset), despite the legacy
module name ``core2_forcing`` — kept only as a deprecation shim. In this repository
**"CORE2" names a MESH, never a forcing dataset.** (The L&Y09 *bulk formulae* — an air-sea
flux parameterisation, a different thing from the CORE-II dataset — are used, and live in
:mod:`fesom_jax.forcing`.)

**Mesh-agnostic.** Nothing here is CORE2-specific: the same driver runs unchanged on pi,
farc, dars, CORE2 and NG5. It adapts to whatever :class:`~fesom_jax.mesh.Mesh` it is handed
and resolves its inputs through :mod:`fesom_jax.paths` (or the run YAML's ``forcing:`` block).

**Driver vs. kernels.** This module is the *driver*: it assembles the input datasets (JRA55
atmosphere + SSS restoring + river runoff + chlorophyll) into the per-step forcing the ocean
step consumes. Its sibling :mod:`fesom_jax.forcing` holds the differentiable *kernels* it
calls — the L&Y09 bulk formulae, the wind stress, the shortwave penetration.

Ties the per-step **JRA55 atmosphere** (host reader, :mod:`fesom_jax.jra55`), the
**L&Y09 bulk + shortwave penetration** (:mod:`fesom_jax.forcing`), and the
**SSS-restoring / runoff** salt-water balance (:mod:`fesom_jax.sss_runoff`) into the
surface boundary conditions the ocean step consumes: the element wind ``stress_surf``
(momentum, substep 7) and the per-node ``bc_T``/``bc_S`` + ``sw_3d`` (tracer diffusion,
substep 15). Mirrors the C forcing block in ``fesom_main.c:1021-1122``:

    jra55_step → bulk_compute → sss_runoff_step → oce_fluxes_mom → cal_shortwave_rad
    → fesom_timestep

Two layers, mirroring the underlying modules:

* **Host SETUP** (non-differentiable): :func:`build_surface_forcing` builds the readers +
  climatologies once; :meth:`SurfaceForcing.step_forcing` / :meth:`SurfaceForcing.stack` run
  the numpy JRA reader to produce per-step **device-constant** atmosphere arrays.
* **Device per-step math** (AD-safe, runs inside ``step``/``scan``):
  :func:`compute_surface_fluxes` — ``bulk(SST, current)`` → ``sss_runoff(S_top,
  water_flux)`` → ``cal_shortwave_rad(heat_flux)`` → ``bc_T``/``bc_S``/``sw_3d``/
  ``stress_surf``, differentiable w.r.t. the model state (the SST→flux / current→stress
  seam for hybrid ML).

linfs notes (verified by reading the C): the surface salinity BC is
``bc_S = dt·(virtual_salt + relax_salt)`` — the balanced ``water_flux`` is **inert**
under linfs (the ``ssh_rhs`` ``water_flux`` term is the non-linfs branch,
``fesom_ssh.c:322-324``; ``virtual_salt`` already uses the *pre*-balance bulk
``water_flux``). So runoff has no effect on the Phase-5 linfs trajectory beyond the
(unused) balanced ``water_flux`` — consistent with the Task-5.5 finding.

chl source: the C default (``FESOM_CHL_SOURCE`` unset) and the production CORE2 PP run
both use the **Sweeney monthly climatology** (``Sweeney_2005.nc``), read by the same
``read_other_NetCDF`` routine as the SSS climatology. Pass ``chl_const=0.1`` to
:func:`build_surface_forcing` for the ``FESOM_CHL_SOURCE=None`` /
``FESOM_PHASE1_CHL_CONST`` alternative.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import NamedTuple

import jax.numpy as jnp

import numpy as np

from . import forcing as _forcing
from . import jra55, sss_runoff
from .config import DENSITY_0, DT_DEFAULT, VCPW
from .mesh import Mesh
from .state import State

# Ocean-ice drag for the momentum stress blend (fesom_ice_coupling.c:23,
# fesom_ice.c:66 cd_oce_ice=5.5e-3, FESOM_DENSITY_0=1030 → 5.665).
CD_OCE_ICE = 5.5e-3
RHO_CD = DENSITY_0 * CD_OCE_ICE


# ==========================================================================
# Pytrees threaded through step / scan
# ==========================================================================
class ForcingStatic(NamedTuple):
    """Per-mesh forcing constants (closed over the time loop — NOT scanned)."""
    runoff_node: jnp.ndarray        # (nod2D,)  CORE2 runoff [m/s] (/1000 applied)
    areasvol_surf: jnp.ndarray      # (nod2D,)  surface-layer CV area [m²]
    ocean_area: jnp.ndarray         # scalar    Σ areasvol_surf over ocean [m²]
    open_water: jnp.ndarray         # (nod2D,)  bool: ulevels<=1 (no cavity)
    a_ice: jnp.ndarray              # (nod2D,)  STATIC sea-ice conc (=0.9 where IC SST<0)
    # Full REcoM's per-run node constants (fesom_jax.recom.wiring.with_recom_static, built once on the host; None
    # otherwise: no leaf, so the REcoM-off program is unchanged). Node-leading, so they are partitioned, folded and
    # placed like the fields above on every path.
    bgc_seafloor: jnp.ndarray = None    # (nod2D, nl-1) seafloor area under each layer [m2] (sinking.seafloor_area)
    bgc_saar_h: jnp.ndarray = None      # (nod2D, 45) gsw_saar's horizontally interpolated SAAR (sinking.saar_columns)
    bgc_saar_pcap: jnp.ndarray = None   # (nod2D,) p_ref(ndepth_max) [dbar]
    bgc_saar_data: jnp.ndarray = None   # (nod2D,) bool: the SAAR table has data here
    bgc_saar_baltic: jnp.ndarray = None # (nod2D,) bool: gsw_sa_from_sp_baltic applies


class StepForcing(NamedTuple):
    """Per-step device-constant forcing (the scanned ``xs``). Each field ``[nod2D]``;
    the wind is g2r-rotated to the model frame, ``Tair`` is °C, ``prec_*`` m/s."""
    u_air: jnp.ndarray
    v_air: jnp.ndarray
    shum: jnp.ndarray
    shortwave: jnp.ndarray
    longwave: jnp.ndarray
    Tair: jnp.ndarray
    prec_rain: jnp.ndarray
    prec_snow: jnp.ndarray
    Ssurf_month: jnp.ndarray        # SSS restoring target for this step's month [psu]
    chl: jnp.ndarray                # chlorophyll for this step's month [mg/m³]
    # FESOM2's l_mslp (forcing.mslp): the JRA55-do sea-level pressure [Pa] (press_air,
    # gen_forcing_couple.F90:680), read by the momentum RHS (oce_ale_vel_rhs.F90:208) and REcoM.
    # None = l_mslp off: an empty pytree node, so no leaf, shape or program changes.
    press_air: jnp.ndarray = None
    # REcoM's per-step inputs (full REcoM only; None otherwise, so no leaf): FESOM2's clock at the `call recom` point
    # of this step (fesom_module.F90:1203, after `call clock`) and the Albani dust it holds then
    # (:class:`BgcForcing`). daynew/ndpyr are int32 SCALARS per step ([T] in a stack): replicated, never partitioned.
    fe_dust: jnp.ndarray = None     # GloFeDust [nod2D], DustClim of the held month [umol m-2 s-1, as in the file]
    daynew: jnp.ndarray = None      # () int32: day of the year at the step END (FESOM2 daynew)
    ndpyr: jnp.ndarray = None       # () int32: days in that year (365/366)
    # () float64: AtmCO2(month) [ppm], REcoM's prescribed atmospheric CO2 (recom_main.F90:375) from the time series
    # (constant_CO2 off, :class:`AtmCO2Series`); None with constant_CO2 (CO2_for_spinup, no leaf)
    co2_atm: jnp.ndarray = None


#: StepForcing fields that are per-step SCALARS (REcoM's calendar): replicated on every device, never partitioned
STEP_SCALARS = ("daynew", "ndpyr", "co2_atm")


class ForcingTables(NamedTuple):
    """Per-CHUNK forcing coefficient tables for the ON-DEVICE combine
    (``docs/plans/20260715-fesom-jax-ondevice-forcing.md``): the few bracket
    coefficient fields a chunk touches, node axis LAST (shardable). Built by
    :meth:`SurfaceForcing.stack_tables`; consumed with a :class:`ForcingSched` by
    :func:`combine_step_forcing`. Replaces the ``[n_steps, nod2D]`` per-step stack
    (H2D volume ÷ ~n_steps/n_brackets)."""
    coef_a: jnp.ndarray             # [N_FLD, nb, nod2D] getcoeffld linear-in-time slope
    coef_b: jnp.ndarray             # [N_FLD, nb, nod2D] getcoeffld intercept
    sss: jnp.ndarray                # [nm, nod2D] month table: SSS restoring target [psu]
    chl: jnp.ndarray                # [nm, nod2D] month table: chlorophyll [mg/m³]
    dust: jnp.ndarray = None        # [nm, nod2D] held-dust table (full REcoM; None otherwise)


class ForcingSched(NamedTuple):
    """Per-STEP schedule for :class:`ForcingTables` (tiny, replicated scan ``xs``)."""
    rdate: jnp.ndarray              # [T] f64 — the reader's Julian step date
    fidx: jnp.ndarray               # [T, N_FLD] i32 — per-field bracket row in the tables
    midx: jnp.ndarray               # [T] i32 — month row in the sss/chl tables
    elapsed: jnp.ndarray = None     # [T, N_FLD] f32 — single precision only: rdate − t0 per field (#940)
    didx: jnp.ndarray = None        # [T] i32 — full REcoM: row in the dust table
    daynew: jnp.ndarray = None      # [T] i32 — full REcoM: FESOM2 daynew at the step end
    ndpyr: jnp.ndarray = None       # [T] i32 — full REcoM: FESOM2 ndpyr
    co2_atm: jnp.ndarray = None     # [T] f64 — full REcoM with the CO2 time series: AtmCO2(month) [ppm]


class ForcingDeviceConst(NamedTuple):
    """Per-RUN device constants for :func:`combine_step_forcing`: the g2r rotation
    trig table (mesh constants — the jra55 ROTCACHE, now device-resident) + the Euler
    rotation matrix. Built once by :func:`forcing_device_const`; partitioned once."""
    sgl: jnp.ndarray                # [nod2D] sin(glat)   } _g2r_trig order
    cgl: jnp.ndarray                # [nod2D] cos(glat)
    sgo: jnp.ndarray                # [nod2D] sin(glon)
    cgo: jnp.ndarray                # [nod2D] cos(glon)
    srl: jnp.ndarray                # [nod2D] sin(rlat)
    crl: jnp.ndarray                # [nod2D] cos(rlat)
    sro: jnp.ndarray                # [nod2D] sin(rlon)
    cro: jnp.ndarray                # [nod2D] cos(rlon)
    M: jnp.ndarray                  # [9] geographic→rotated Euler matrix (replicated)
    prec_div: jnp.ndarray           # () 1000.0 — a RUNTIME operand so no backend can
    #   strength-reduce the prec unit conversion into x·(1/1000.0) at compile time.
    #   ⚠️ Even so, XLA *CPU*'s f64 divide is not correctly rounded (measured: 1 ULP
    #   off vs numpy on ~13 % of operands, flags don't help) ⇒ prec_rain/prec_snow are
    #   bit-exact vs the host reader on GPU (CUDA div.rn.f64 is IEEE) but ≤1 ULP on
    #   CPU. The gates probe the platform divide and demand accordingly.


def forcing_device_const(sf: "SurfaceForcing") -> ForcingDeviceConst:
    """The :class:`ForcingDeviceConst` for this forcing's reader (host numpy — the
    driver partitions + uploads it once)."""
    return ForcingDeviceConst(*sf.jra._trig, M=np.asarray(sf.jra.M, dtype=np.float64),
                              prec_div=np.float64(1000.0))


def combine_step_forcing(tables: ForcingTables, rdate, fidx, midx,
                         const: ForcingDeviceConst, elapsed=None, *, didx=None, daynew=None,
                         ndpyr=None, co2_atm=None) -> StepForcing:
    """ONE step's :class:`StepForcing` from the chunk tables — the device-side twin of
    :meth:`SurfaceForcing.step_forcing`, run INSIDE the scan (all elementwise; fuses
    into the step). Same expression order as the host reader
    (:meth:`fesom_jax.jra55.JRA55Reader.step`) on bit-identical inputs (the table rows
    are ``_getcoeffld``'s own outputs — the host ``_gather`` is untouched); executed
    EAGERLY this is bit-identical to the host (gated, ``test_jra55``). ⚠️ INSIDE jit,
    XLA FMA-contracts ``rdate·coef_a + coef_b`` (single rounding; ~1e-9 rel at the
    time-interp's cancellation magnitudes — measured on CPU; barriers/bitcasts don't
    reliably prevent it) ⇒ the in-scan path is VALUE-equivalent, not bit-identical, to
    the host reader. ``forcing.on_device`` therefore defaults OFF, and the step-level
    gates budget the roundoff-seed growth (`_TABLES_ATOL` in test_forcing_sharded).
    Tables with a ninth row (a reader built with ``mslp``) also give ``press_air``; eight rows ⇒ None.
    The REcoM inputs (``didx``/``daynew``/``ndpyr``/``co2_atm``, the schedule's fields; ``tables.dust``) pass through
    when set."""
    def val(fi):
        if elapsed is not None:   # single precision: point-slope form, coef_b = bracket-start value (#940)
            return tables.coef_b[fi][fidx[fi]] + elapsed[fi] * tables.coef_a[fi][fidx[fi]]
        return rdate * tables.coef_a[fi][fidx[fi]] + tables.coef_b[fi][fidx[fi]]

    u = val(jra55.I_XWIND)
    v = val(jra55.I_YWIND)
    if jra55.FORCE_ROTATION:
        trig = (const.sgl, const.cgl, const.sgo, const.cgo,
                const.srl, const.crl, const.sro, const.cro)
        u, v = jra55._vector_g2r(u, v, None, None, None, None, const.M, trig=trig)
    return StepForcing(
        u_air=u, v_air=v,
        shum=val(jra55.I_HUMI),
        shortwave=val(jra55.I_QSR),
        longwave=val(jra55.I_QLW),
        Tair=val(jra55.I_TAIR) - 273.15,
        prec_rain=val(jra55.I_PREC) / const.prec_div,
        prec_snow=val(jra55.I_SNOW) / const.prec_div,
        Ssurf_month=tables.sss[midx],
        chl=tables.chl[midx],
        press_air=val(jra55.I_MSLP) if tables.coef_a.shape[0] > jra55.N_FLD else None,
        fe_dust=None if tables.dust is None else tables.dust[didx],
        daynew=daynew, ndpyr=ndpyr, co2_atm=co2_atm)


class SurfaceFluxes(NamedTuple):
    """Output of :func:`compute_surface_fluxes`. The step consumes ``stress_surf``
    (momentum) and ``bc_T``/``bc_S``/``sw_3d`` (tracer diffusion); the rest are
    diagnostics retained for the per-substep dump gate."""
    stress_surf: jnp.ndarray        # (elem2D, 2) element wind stress [N/m²]
    bc_T: jnp.ndarray               # (nod2D,)  −dt·heat_flux/vcpw  [°C·... surface incr]
    bc_S: jnp.ndarray               # (nod2D,)  dt·(virtual_salt+relax_salt)
    sw_3d: jnp.ndarray              # (nod2D, nl) shortwave temperature flux [K·m/s]
    stress_node_surf: jnp.ndarray   # (nod2D, 2) ice-blended NODE wind stress (KPP ustar)
    heat_flux: jnp.ndarray          # (nod2D,)  post-shortwave-penetration [W/m²]
    water_flux: jnp.ndarray         # (nod2D,)  balanced (inert in linfs) [m/s]
    virtual_salt: jnp.ndarray       # (nod2D,)  [psu·m/s]
    relax_salt: jnp.ndarray         # (nod2D,)  [psu·m/s]


# ==========================================================================
# Device per-step forcing math (AD-safe; runs inside step / lax.scan)
# ==========================================================================
def compute_surface_fluxes(mesh: Mesh, state: State, sf: StepForcing,
                           fs: ForcingStatic, *, dt: float = DT_DEFAULT,
                           owned_mask=None, axis_name=None,
                           use_virt_salt: bool = True, zbar3=None,
                           legacy_sw_depth: bool = False) -> SurfaceFluxes:
    """Compute the surface BCs from the start-of-step model ``state`` + this step's
    atmosphere ``sf`` + the static constants ``fs``. Pure & differentiable w.r.t.
    ``state`` (the bulk taps ``state.T[:,0]`` / ``state.uvnode[:,0]``).

    Order mirrors the C (``fesom_main.c``): bulk → sss_runoff → oce_fluxes_mom (ice stress
    blend) → cal_shortwave_rad. The static ``a_ice`` mask (0.9 where the IC SST<0; ice
    physics off ⇒ never updated) gates BOTH the stress blend and the shortwave penetration —
    the only two ``a_ice`` couplings in the no-ice path."""
    # state.uvnode is refreshed from state.uv at the top of step() (drift fix 1).
    u_w = state.uvnode[:, 0, 0]
    v_w = state.uvnode[:, 0, 1]

    # 1 — L&Y09 bulk (SST + surface-current taps). Open-water atm-ocean fluxes at all nodes.
    bulk = _forcing.bulk_surface_fluxes(
        mesh, sf.u_air, sf.v_air, sf.shum, sf.shortwave, sf.longwave, sf.Tair,
        sf.prec_rain, sf.prec_snow, state.T[:, 0], u_w, v_w)

    # 2 — SSS restoring + runoff balance (consumes the bulk water_flux + S_top).
    sss = sss_runoff.sss_runoff_fluxes(
        state.S[:, 0], bulk.water_flux, sf.Ssurf_month, fs.runoff_node,
        fs.areasvol_surf, fs.ocean_area, fs.open_water,
        owned_mask=owned_mask, axis_name=axis_name)

    # 3 — oce_fluxes_mom ice stress blend (fesom_ice_coupling.c:234-264). At ice nodes the
    #     wind stress is replaced by stress = ice_drag·a_ice + atm·(1−a_ice); the ice-ocean
    #     drag uses u_ice=0 (static) and u_w = surface current (a current→stress AD seam):
    #     ice_drag = ρ·Cd·|u_ice−u_w|·(u_ice−u_w). Then element stress = mean-of-3 of nodes.
    a = fs.a_ice
    du, dv = -u_w, -v_w                                   # u_ice − u_w, u_ice = 0
    aux = _forcing._safe_speed(du, dv) * RHO_CD
    ice_on = a > 0.001                                    # static mask (C threshold)
    sic_x = jnp.where(ice_on, aux * du, 0.0)
    sic_y = jnp.where(ice_on, aux * dv, 0.0)
    sns = bulk.stress_node_surf
    blend_x = jnp.where(fs.open_water, sic_x * a + sns[:, 0] * (1.0 - a), sns[:, 0])
    blend_y = jnp.where(fs.open_water, sic_y * a + sns[:, 1] * (1.0 - a), sns[:, 1])
    sns_b = jnp.stack([blend_x, blend_y], axis=-1)
    ev = mesh.elem_nodes
    stress_surf = (sns_b[ev[:, 0]] + sns_b[ev[:, 1]] + sns_b[ev[:, 2]]) / 3.0

    # 4 — shortwave penetration (heat_flux += visible band; build sw_3d). NO penetration
    #     under ice (a_ice>0) or cavity — fesom_bulk.c:381-382.
    pene_open = fs.open_water & (a <= 0.0)
    heat_flux, sw_3d = _forcing.cal_shortwave_rad(
        mesh, bulk.heat_flux, sf.shortwave, sf.chl, pene_open, zbar3=zbar3, legacy_depth=legacy_sw_depth)

    # 5 — surface BCs (fesom_tracer_diff.c:43-75). bc_T base here; the zstar −dt·sval·wf term
    #     is added in step.py (post-advection sval). Under zstar (use_virt_salt=False) the
    #     no-ice path has no ice thermo ⇒ no real_salt_flux, and virtual_salt≡0 ⇒
    #     bc_S = dt·relax_salt. linfs (default) ⇒ byte-identical.
    virtual_salt = sss.virtual_salt if use_virt_salt else jnp.zeros_like(sss.virtual_salt)
    bc_T = -dt * heat_flux / VCPW
    bc_S = dt * (virtual_salt + sss.relax_salt)

    return SurfaceFluxes(
        stress_surf=stress_surf, bc_T=bc_T, bc_S=bc_S, sw_3d=sw_3d,
        stress_node_surf=sns_b, heat_flux=heat_flux, water_flux=sss.water_flux,
        virtual_salt=virtual_salt, relax_salt=sss.relax_salt)


# ==========================================================================
# Host SETUP — readers, climatologies, per-step atmosphere
# ==========================================================================
def ice_ic_aice(mesh: Mesh, sst_ic) -> jnp.ndarray:
    """Static sea-ice concentration from ``fesom_ice_initial_state`` (``fesom_ice.c``):
    ``a_ice = 0.9`` where (non-cavity & PHC IC SST < 0), else 0. With ice dyn/adv/thermo
    off it is never updated, so it is a constant surface mask the C surface couplings read
    (the shortwave-penetration gate + the momentum stress blend). Returns ``[nod2D]``."""
    non_cavity = np.asarray(mesh.ulevels_nod2D) <= 1
    cold = np.asarray(sst_ic) < 0.0
    return jnp.asarray(np.where(non_cavity & cold, 0.9, 0.0))


def _static_from_mesh(mesh: Mesh, runoff_node, a_ice) -> ForcingStatic:
    open_water = jnp.asarray(mesh.ulevels_nod2D) <= 1
    return ForcingStatic(
        runoff_node=jnp.asarray(runoff_node),
        areasvol_surf=jnp.asarray(mesh.areasvol[:, 0]),
        ocean_area=jnp.asarray(float(mesh.ocean_area)),
        open_water=open_water,
        a_ice=jnp.asarray(a_ice))


@dataclasses.dataclass
class SurfaceForcing:
    """Host driver bundling the readers + climatologies + the device static constants.

    Build with :func:`build_surface_forcing`; produce per-step forcing with
    :meth:`step_forcing` (one date) or :meth:`stack` (a list of dates → a scannable
    :class:`StepForcing` with a leading step axis)."""
    jra: jra55.JRA55Reader
    sss: sss_runoff.SSSRunoffReader
    chl_clim: "object"              # np.ndarray [12, nod2D]
    static: ForcingStatic
    # single precision (fesom_jax.precision, FESOM2 #940): per-step values in the point-slope form, float32.
    # Set by the run driver for a precision: single run; False (default) keeps every double-precision value.
    point_slope: bool = False
    # full REcoM: the per-step REcoM inputs (calendar at the `call recom` point, the held Albani dust); None = off
    bgc: "BgcForcing | None" = None

    def step_forcing(self, year: int, day: int, sec: float, month: int,
                     *, xp=jnp) -> StepForcing:
        """Per-step :class:`StepForcing` for a model date (1-based ``day`` of year,
        ``sec`` of day, 1-based ``month``). Runs the host JRA reader + selects the
        month's SSS target + chl.

        ``xp`` selects the array backend (mirrors :meth:`State.zeros`): the default ``jnp``
        is byte-identical to before (device arrays); ``xp=np`` builds the StepForcing on the
        HOST. The readers already return host numpy, so ``xp=np`` is a pure host op — used by
        the run driver so a big-mesh per-chunk stack is never staged on GPU 0 (the NG5 forcing
        OOM: a global ``[n_steps, nod2D]`` stack is ~2.65 GB/field at 7.4 M × 48 steps)."""
        f = self.jra.step(int(year), int(day), float(sec), point_slope=self.point_slope)
        return StepForcing(
            u_air=xp.asarray(f.u_wind), v_air=xp.asarray(f.v_wind),
            shum=xp.asarray(f.shum), shortwave=xp.asarray(f.shortwave),
            longwave=xp.asarray(f.longwave), Tair=xp.asarray(f.Tair),
            prec_rain=xp.asarray(f.prec_rain), prec_snow=xp.asarray(f.prec_snow),
            Ssurf_month=xp.asarray(self.sss.month(int(month))),
            chl=xp.asarray(self.chl_clim[int(month) - 1]),
            press_air=None if f.press_air is None else xp.asarray(f.press_air),
            **self._bgc_fields(year, day, sec, xp=xp))

    def _bgc_fields(self, year, day, sec, *, xp=jnp) -> dict:
        """The REcoM per-step fields of :class:`StepForcing` (empty when REcoM is off)."""
        if self.bgc is None:
            return {}
        c, m = self.bgc.inputs(year, day, sec)
        co2 = {} if self.bgc.co2 is None else dict(co2_atm=xp.asarray(np.float64(self.bgc.co2_atm(c))))
        return dict(fe_dust=xp.asarray(self.bgc.dust[m - 1]), daynew=xp.asarray(np.int32(c.daynew)),
                    ndpyr=xp.asarray(np.int32(c.ndpyr)), **co2)

    def stack(self, dates, *, xp=jnp) -> StepForcing:
        """Stack per-step forcings for an iterable of ``(year, day, sec, month)`` into a
        single :class:`StepForcing` with leading axis ``[n_steps]`` (for :func:`lax.scan`
        / :func:`fesom_jax.integrate.integrate`).

        ``xp=np`` builds the whole stack on the HOST (the NG5 forcing-OOM fix — the driver
        passes ``xp=np`` so the ``[n_steps, nod2D]`` per-chunk forcing AND the intermediate
        list of ``n_steps`` per-date forcings are never materialized on GPU 0 before
        :func:`shard_mesh.partition_step_forcing` shards them). Default ``jnp`` = unchanged."""
        steps = [self.step_forcing(*d, xp=xp) for d in dates]
        return StepForcing(*[None if leaves[0] is None else xp.stack(leaves, axis=0)
                             for leaves in zip(*steps)])

    def stack_tables(self, dates, *, nb: int, nm: int = 2
                     ) -> "tuple[ForcingTables, ForcingSched]":
        """The on-device-forcing twin of :meth:`stack`: per-chunk coefficient TABLES +
        per-step schedule instead of an ``[n_steps, nod2D]`` stack. Host numpy, pure
        host op (the tables are partitioned/uploaded by the driver);
        :func:`combine_step_forcing` turns row ``i`` back into :meth:`step_forcing`'s
        :class:`StepForcing` bit-exactly (same coefficient fields, same op order).

        ``nb``/``nm`` are the STATIC table depths (JRA brackets / months); fixed per
        (chunk length, dt) so the compiled chunk executable is reusable. A chunk is a
        few days ⇒ ``nm=2`` covers any single month roll (asserted)."""
        dates = list(dates)
        elapsed = None
        if self.point_slope:
            coef_a, coef_b, rdates, fidx, elapsed = self.jra.bracket_schedule(dates, nb, point_slope=True)
        else:
            coef_a, coef_b, rdates, fidx = self.jra.bracket_schedule(dates, nb)
        N = coef_a.shape[-1]
        months: list[int] = []
        midx = np.empty(len(dates), dtype=np.int32)
        for i, d in enumerate(dates):
            mo = int(d[3])
            if not months or months[-1] != mo:
                if mo in months:
                    raise ValueError(f"non-monotonic month sequence in chunk dates ({months + [mo]})")
                months.append(mo)
            midx[i] = len(months) - 1
        if len(months) > nm:
            raise ValueError(f"chunk crosses {len(months) - 1} month boundaries but nm={nm}")
        bgc_tab = {}
        bgc_sched = {}
        if self.bgc is not None:              # the held dust as a month table, the calendar as [T] scalars
            dmonths: list[int] = []
            didx = np.empty(len(dates), dtype=np.int32)
            dayn = np.empty(len(dates), dtype=np.int32)
            ndp = np.empty(len(dates), dtype=np.int32)
            co2 = None if self.bgc.co2 is None else np.empty(len(dates), dtype=np.float64)
            for i, d in enumerate(dates):
                c, m = self.bgc.inputs(d[0], d[1], d[2])
                if not dmonths or dmonths[-1] != m:
                    dmonths.append(m)
                didx[i], dayn[i], ndp[i] = len(dmonths) - 1, c.daynew, c.ndpyr
                if co2 is not None:
                    co2[i] = self.bgc.co2_atm(c)
            if len(dmonths) > nm:
                raise ValueError(f"chunk holds {len(dmonths)} dust months but nm={nm}")
            dust_tab = np.zeros((nm, N), dtype=self.bgc.dust.dtype)
            for j, m in enumerate(dmonths):
                dust_tab[j] = self.bgc.dust[m - 1]
            bgc_tab = dict(dust=dust_tab)
            bgc_sched = dict(didx=didx, daynew=dayn, ndpyr=ndp, co2_atm=co2)
        # dtype mirrors step_forcing's (no cast) so the ON/OFF paths agree byte-for-byte
        sss_rows = [np.asarray(self.sss.month(mo)) for mo in months]
        chl_rows = [np.asarray(self.chl_clim[mo - 1]) for mo in months]
        sss_tab = np.zeros((nm, N), dtype=sss_rows[0].dtype)
        chl_tab = np.zeros((nm, N), dtype=chl_rows[0].dtype)
        for j in range(len(months)):
            sss_tab[j] = sss_rows[j]
            chl_tab[j] = chl_rows[j]
        return (ForcingTables(coef_a=coef_a, coef_b=coef_b, sss=sss_tab, chl=chl_tab, **bgc_tab),
                ForcingSched(rdate=rdates, fidx=fidx, midx=midx, elapsed=elapsed, **bgc_sched))

    def reopen_year(self, year: int):
        """Roll the JRA reader to ``year`` IN PLACE (keeps the year-independent interpolation
        stencil; see :meth:`fesom_jax.jra55.JRA55Reader.reopen_year`). SSS-restoring and chl are
        monthly CLIMATOLOGIES (year-independent), so only the JRA reader rolls. Returns ``self``."""
        self.jra.reopen_year(year)
        return self


def _midstep_days(dt, forcing_midstep: bool) -> float:
    """Drift fix 2: the forcing instant is the MIDDLE of the step (Fortran sbc_do,
    ``timenew - dt/2``). Needs ``dt``; ``dt=None`` or ``forcing_midstep=False`` keeps the
    legacy step-start evaluation (a uniform dt/2 lag of every forcing field) and says so."""
    if not forcing_midstep or dt is None:
        print("[fesom_jax] forcing evaluated at the step START (legacy); pass dt= to "
              "build_surface_forcing for the Fortran mid-step instant", flush=True)
        return 0.0
    return 0.5 * float(dt) / 86400.0


def set_midstep(jra, dt, forcing_midstep: bool, dt_ramp=None, year0=None) -> None:
    """Set ``jra``'s mid-step offset (:func:`_midstep_days`); with the run's ``dt_ramp`` (``year0`` = the
    calendar's start year, ``run._elapsed_seconds``) the steps from the switch on take half the ramp dt, as FESOM2's
    ``timenew - dt/2`` with the dt of that step."""
    jra.midstep_days = _midstep_days(dt, forcing_midstep)
    if dt_ramp is None or jra.midstep_days == 0.0:
        return
    if year0 is None:
        raise ValueError("a dt_ramp mid-step offset needs year0, the calendar's start year")
    t = datetime.datetime(int(year0), 1, 1) + datetime.timedelta(seconds=int(dt_ramp.after_step) * float(dt))
    jra.midstep_switch = (t.year, t.timetuple().tm_yday, float(t.hour * 3600 + t.minute * 60 + t.second))
    jra.midstep_days_after = 0.5 * float(dt_ramp.dt) / 86400.0


def build_surface_forcing(mesh: Mesh, year: int, *, sst_ic=None,
                       jra_dir: str | None = None,
                       sss_path: str | None = None,
                       runoff_path: str | None = None,
                       chl_path: str | None = None,
                       chl_const: float | None = None,
                       dt: float | None = None,
                       forcing_midstep: bool = True,
                       mslp: bool = False,
                       bgc_dust_path: str | None = None,
                       dt_ramp=None, bgc_co2=None) -> SurfaceForcing:
    """Build the surface-forcing driver for a JRA55-do ``year`` (host setup, ~seconds).
    Mesh-agnostic: works on any :class:`~fesom_jax.mesh.Mesh` (pi, farc, dars, CORE2, NG5).

    The four input-path kwargs default to ``None`` ⇒ each reader resolves its own path
    through :mod:`fesom_jax.paths` (``$FESOM_JRA_DIR`` / ``$FESOM_SSS_PATH`` /
    ``$FESOM_RUNOFF_PATH`` / ``$FESOM_CHL_PATH``, else the Levante default). Pass a string
    (e.g. from the run YAML's ``forcing:`` block) to override.

    ``sst_ic`` (``[nod2D]``, the PHC IC surface temperature) builds the static ``a_ice``
    mask (:func:`ice_ic_aice`); pass ``state.T[:, 0]`` from
    :func:`fesom_jax.phc_ic.phc_initial_state`. ``None`` ⇒ ``a_ice≡0`` (truly ice-free;
    e.g. constant-T sanity runs).

    ``chl_const`` (default ``None``) selects the chl source: ``None`` ⇒ the Sweeney
    monthly climatology (the C default / production config); a float ⇒ a constant chl
    everywhere (the ``FESOM_CHL_SOURCE=None`` alternative, ``FESOM_PHASE1_CHL_CONST``).

    ``dt`` (the ocean timestep, s) places the JRA55 interpolation instant at the MIDDLE of
    each step, as the Fortran's ``sbc_do`` does (drift fix 2). ``dt=None`` or
    ``forcing_midstep=False`` ⇒ the legacy step-start instant (a uniform ``dt/2`` lag; what
    the pre-2026-09-18 C dumps and fesom-jax ≤ v1.0.2 use), announced with a printed line.

    ``mslp`` (FESOM2's ``l_mslp``, the run YAML's ``forcing.mslp``) also reads the JRA55-do sea-level
    pressure ``psl`` into :attr:`StepForcing.press_air`; off (the default) the forcing is unchanged.

    ``bgc_dust_path`` (full REcoM only: the Albani ``DustClim(months, nod2d)`` file) adds REcoM's per-step inputs
    (:class:`BgcForcing`: ``fe_dust``, ``daynew``, ``ndpyr``, and ``co2_atm`` with ``bgc_co2``, an
    :class:`AtmCO2Series`); it needs ``dt``, and the run's ``dt_ramp`` when it has one (the calendar's start year is
    ``year``). None (the default) adds nothing. ``dt_ramp`` also sets the
    forcing's mid-step offset after the switch (:func:`set_midstep`)."""
    jra = jra55.JRA55Reader(mesh, year, jra_dir, mslp=mslp)
    set_midstep(jra, dt, forcing_midstep, dt_ramp, year)
    sss = sss_runoff.build_reader(mesh, sss_path, runoff_path)
    if chl_const is None:
        chl_clim = sss_runoff.build_chl_clim(mesh, chl_path)
    else:
        chl_clim = np.full((12, int(mesh.nod2D)), float(chl_const), dtype=np.float64)
    a_ice = (ice_ic_aice(mesh, sst_ic) if sst_ic is not None
             else jnp.zeros(int(mesh.nod2D)))
    bgc = None
    if bgc_dust_path is not None:
        if dt is None:
            raise ValueError("bgc_dust_path needs dt: FESOM2's clock at the recom call is the step end")
        bgc = BgcForcing(read_dust_clim(bgc_dust_path, int(mesh.nod2D)), float(dt), dt_ramp=dt_ramp, year0=year,
                         co2=bgc_co2)
    return SurfaceForcing(jra=jra, sss=sss, chl_clim=chl_clim,
                       static=_static_from_mesh(mesh, sss.runoff_node, a_ice), bgc=bgc)


# ==========================================================================
# REcoM per-step inputs: FESOM2's clock at the recom call + the Albani dust hold
# ==========================================================================
class RecomClock(NamedTuple):
    """FESOM2's clock (``g_clock``) after ``call clock`` of a step, i.e. at the ``call recom`` point."""
    yearnew: int
    daynew: int
    timenew: float          # seconds into ``daynew``, in (0, 86400]
    ndpyr: int
    month: int
    day_in_month: int


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


_DAYS_IN_MONTH = ((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
                  (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))


def recom_clock(year: int, day: int, sec: float, dt: float) -> RecomClock:
    """FESOM2's clock at the ``call recom`` point of the step that STARTS at ``(year, day, sec)`` (the
    :func:`dates_for_steps` convention). ``fesom_module.F90`` calls ``clock`` first, so the clock reads the step
    END, by ``gen_modules_clock.F90:clock``: ``timenew = timenew + dt``; only ``timenew > 86400`` (strict) moves
    ``daynew`` on (the last step of a day ends at ``timenew == 86400`` of that day); ``daynew > ndpyr`` starts the
    next year (``ndpyr`` = 365 + leap of the new year, ``include_fleapyear`` true); ``month`` and
    ``day_in_month`` from ``daynew``. FESOM2 accumulates ``timenew`` by additions, exact for an integer ``dt``."""
    year, day = int(year), int(day)
    timenew = float(sec) + float(dt)
    ndpyr = 365 + int(_is_leap(year))
    if timenew > 86400.0:
        day += 1
        timenew -= 86400.0
    if day > ndpyr:
        day = 1
        year += 1
        ndpyr = 365 + int(_is_leap(year))
    dim = _DAYS_IN_MONTH[int(_is_leap(year))]
    aux = 0
    for m in range(12):
        if aux < day <= aux + dim[m]:
            return RecomClock(year, day, timenew, ndpyr, m + 1, day - aux)
        aux += dim[m]
    raise AssertionError(f"day {day} outside year {year}")


def read_dust_clim(path: str, nod2D: int) -> np.ndarray:
    """``DustClim(months, nod2d)`` of an Albani file on the mesh nodes (``read_2ddata_on_grid_NetCDF``: record
    ``i`` read as stored, global node order), ``[12, nod2D]`` float64, units as in the file (umol m-2 s-1)."""
    import netCDF4
    with netCDF4.Dataset(str(path)) as ds:
        v = ds.variables["DustClim"]
        v.set_auto_maskandscale(False)
        a = np.asarray(v[:], dtype=np.float64)
    if a.shape != (12, nod2D):
        raise ValueError(f"{path}: DustClim has shape {a.shape}, the mesh needs (12, {nod2D})")
    return a


class AtmCO2Series:
    """REcoM's prescribed atmospheric CO2 with ``constant_CO2`` off (``gen_surface_forcing.F90:1986-2041``): the 12
    monthly values ``AtmCO2_<year>`` [ppm] of ``nm_co2_data_file`` for the CO2 year ``currentCO2year =
    firstyearofcurrentCO2cycle + (yearnew - firstyearoffesomcycle) + 1``, with ``firstyearofcurrentCO2cycle =
    lastyearoffesomcycle - numofCO2cycles * totnumyear + (currentCO2cycle - 1) * totnumyear`` and ``totnumyear =
    lastyearoffesomcycle - firstyearoffesomcycle + 1`` (one cycle: the model year); ``recom`` takes
    ``AtmCO2(month)`` (recom_main.F90:375).

    ⚠️ FESOM2 reads a year's values once, at ``mstep == 1`` (the first step of a run segment), so once a year in
    one-year segments. Here every model year takes its own values: FESOM2 run in one-year segments (a multi-year
    FESOM2 segment keeps its first year's values throughout; that is not mirrored)."""

    def __init__(self, path, *, first_year: int, last_year: int, n_cycles: int = 1, cycle: int = 1):
        self.path = str(path)
        self.first_year, self.last_year = int(first_year), int(last_year)
        self.n_cycles, self.cycle = int(n_cycles), int(cycle)
        self._years: dict = {}

    def co2_year(self, yearnew: int) -> int:
        """``currentCO2year`` of model year ``yearnew``."""
        tot = self.last_year - self.first_year + 1
        first = self.last_year - self.n_cycles * tot + (self.cycle - 1) * tot
        return first + (int(yearnew) - self.first_year) + 1

    def year_values(self, yearnew: int) -> np.ndarray:
        """``AtmCO2(1:12)`` [ppm] for model year ``yearnew``, float64 (read once per CO2 year)."""
        y = self.co2_year(yearnew)
        if y not in self._years:
            import netCDF4
            name = f"AtmCO2_{y:4d}"
            with netCDF4.Dataset(self.path) as ds:
                if name not in ds.variables:
                    raise ValueError(f"{self.path}: no {name} (model year {yearnew})")
                v = ds.variables[name]
                v.set_auto_maskandscale(False)
                self._years[y] = np.asarray(v[:12], dtype=np.float64)
        return self._years[y]

    def value(self, yearnew: int, month: int) -> float:
        """``AtmCO2(month)`` [ppm] in model year ``yearnew``."""
        return float(self.year_values(yearnew)[int(month) - 1])


class BgcForcing:
    """REcoM's per-step inputs as FESOM2 has them at the ``call recom`` point.

    * the calendar: :func:`recom_clock` (``daynew``, ``ndpyr``; ``month``/``day_in_month``/``timenew`` for the
      dust rule below);
    * ``GloFeDust``, the Albani monthly dust HELD between updates (``gen_surface_forcing.F90:sbc_do_recom``,
      called in ``update_atm_forcing`` before ``recom``): an update happens when ``update_monthly_flag =
      (day_in_month == num_day_in_month(month) .and. timenew == 86400) .or. mstep == 1``, and loads month
      ``i = month`` at ``mstep == 1``, else ``i = month + 1`` (12 wraps to 1). So the last step of a month (ending
      at 24:00 of its last day) already gets the next month's dust, and a run segment starts with the current
      month's: a segment starting on a month's last step keeps that month's dust through the next month (FESOM2's
      ``mstep`` restarts at 1 in every run segment).

    ``mstep == 1`` is the first step this object is asked for (``start``, a ``(year, day, sec)`` step start; set
    on the first query unless given): build one per run segment, as the run driver does per job. Later queries
    replay FESOM2's rule step by step from the last answered step (in step order a chunk costs one rule per step);
    a query before ``start`` raises. ``GloNDust`` is zero (``useAeolianN`` off). ``AtmCO2``: with ``co2`` (an
    :class:`AtmCO2Series`, ``constant_CO2`` off) ``AtmCO2(month)`` at the step's clock (:meth:`co2_atm`); without,
    the constant ``CO2_for_spinup`` (``constant_CO2``), which the caller builds.

    ``dt_ramp`` (the run's :class:`~fesom_jax.run_config.DtRamp`, with ``year0`` the calendar's start year): the steps
    starting at or after ``year0``-01-01 + ``after_step * dt`` are taken with ``dt_ramp.dt`` (the run driver's
    piecewise clock, ``run._elapsed_seconds``), so a segment that starts after the switch, or straddles it, steps on
    its own grid and each step's clock reads its own dt. None: one ``dt`` throughout."""

    def __init__(self, dust: np.ndarray, dt: float, start=None, *, dt_ramp=None, year0=None, co2=None):
        self.dust = np.asarray(dust)
        self.dt = float(dt)
        self.co2 = co2
        self.start = None if start is None else self._key(*start[:3])
        self._switch, self._dt1 = None, None
        if dt_ramp is not None:
            if year0 is None:
                raise ValueError("BgcForcing with dt_ramp needs year0, the calendar's start year")
            self._switch = (datetime.datetime(int(year0), 1, 1)
                            + datetime.timedelta(seconds=int(dt_ramp.after_step) * self.dt))
            self._dt1 = float(dt_ramp.dt)
        self._held: dict = {}            # step start -> held month, the answered steps
        self._front = None               # the latest answered step start

    def co2_atm(self, c: RecomClock) -> float:
        """``AtmCO2(month)`` [ppm] at the ``call recom`` clock ``c`` of a step (``co2`` must be set)."""
        return self.co2.value(c.yearnew, c.month)

    @staticmethod
    def _key(year, day, sec):
        return datetime.datetime(int(year), 1, 1) + datetime.timedelta(days=int(day) - 1, seconds=float(sec))

    def _dt_at(self, k) -> float:
        """The dt of the step starting at ``k``."""
        return self._dt1 if self._switch is not None and k >= self._switch else self.dt

    def _rule(self, k, prev_month):
        c = recom_clock(k.year, k.timetuple().tm_yday, k.hour * 3600.0 + k.minute * 60.0 + k.second
                        + k.microsecond * 1e-6, self._dt_at(k))
        first = k == self.start
        last_step_of_month = (c.day_in_month == _DAYS_IN_MONTH[int(_is_leap(c.yearnew))][c.month - 1]
                              and c.timenew == 86400.0)
        if first or last_step_of_month:
            i = c.month if first else c.month + 1
            return c, (1 if i > 12 else i)
        return c, prev_month

    def inputs(self, year, day, sec):
        """``(RecomClock, held dust month 1..12)`` for the step starting at ``(year, day, sec)``."""
        k = self._key(year, day, sec)
        if self.start is None:
            self.start = k
        if k < self.start:
            raise ValueError(f"step {k} precedes the run segment start {self.start} (mstep would be < 1)")
        if k not in self._held:
            nxt = lambda t: t + datetime.timedelta(seconds=self._dt_at(t))        # noqa: E731
            cur, month = (self.start, None) if self._front is None or self._front > k else \
                (nxt(self._front), self._held[self._front])
            while cur <= k:
                _, month = self._rule(cur, month)
                self._held[cur] = month
                cur = nxt(cur)
            self._front = max(self._front or k, k)
            if k not in self._held:
                raise ValueError(f"step {k} is not on the dt={self.dt} s grid of the segment start {self.start}"
                                 + ("" if self._switch is None else f" (dt {self._dt1} s from {self._switch})"))
        c, _ = self._rule(k, self._held[k])
        return c, self._held[k]


# ==========================================================================
# Calendar — model date at step n (mirrors fesom_jra55_step_cal)
# ==========================================================================
def dates_for_steps(start_year: int, dt: float, n_steps: int, *,
                    start_month: int = 1, start_day: int = 1):
    """Model ``(year, day_of_year, sec_of_day, month)`` for steps ``1..n_steps``.

    The C model calendar starts at ``(jra_year, 1, 1, 00:00:00)`` and step ``n`` reads
    the calendar at elapsed ``(n−1)·dt`` (``fesom_main.c``: the forcing read precedes the
    post-step ``fesom_io_step`` calendar advance). Proleptic-Gregorian via
    :mod:`datetime` (matches ``FESOM_CAL_GREGORIAN`` + the JRA ``gregorian`` calendar for
    these dates). Returns a list of length ``n_steps``."""
    base = datetime.datetime(int(start_year), int(start_month), int(start_day))
    out = []
    for n in range(n_steps):
        d = base + datetime.timedelta(seconds=(n) * float(dt))
        doy = (d - datetime.datetime(d.year, 1, 1)).days + 1
        sec = d.hour * 3600.0 + d.minute * 60.0 + d.second + d.microsecond * 1e-6
        out.append((d.year, doy, sec, d.month))
    return out
