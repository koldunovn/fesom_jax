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
                           use_virt_salt: bool = True, zbar3=None) -> SurfaceFluxes:
    """Compute the surface BCs from the start-of-step model ``state`` + this step's
    atmosphere ``sf`` + the static constants ``fs``. Pure & differentiable w.r.t.
    ``state`` (the bulk taps ``state.T[:,0]`` / ``state.uvnode[:,0]``).

    Order mirrors the C (``fesom_main.c``): bulk → sss_runoff → oce_fluxes_mom (ice stress
    blend) → cal_shortwave_rad. The static ``a_ice`` mask (0.9 where the IC SST<0; ice
    physics off ⇒ never updated) gates BOTH the stress blend and the shortwave penetration —
    the only two ``a_ice`` couplings in the no-ice path."""
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
        mesh, bulk.heat_flux, sf.shortwave, sf.chl, pene_open, zbar3=zbar3)

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
        f = self.jra.step(int(year), int(day), float(sec))
        return StepForcing(
            u_air=xp.asarray(f.u_wind), v_air=xp.asarray(f.v_wind),
            shum=xp.asarray(f.shum), shortwave=xp.asarray(f.shortwave),
            longwave=xp.asarray(f.longwave), Tair=xp.asarray(f.Tair),
            prec_rain=xp.asarray(f.prec_rain), prec_snow=xp.asarray(f.prec_snow),
            Ssurf_month=xp.asarray(self.sss.month(int(month))),
            chl=xp.asarray(self.chl_clim[int(month) - 1]))

    def stack(self, dates, *, xp=jnp) -> StepForcing:
        """Stack per-step forcings for an iterable of ``(year, day, sec, month)`` into a
        single :class:`StepForcing` with leading axis ``[n_steps]`` (for :func:`lax.scan`
        / :func:`fesom_jax.integrate.integrate`).

        ``xp=np`` builds the whole stack on the HOST (the NG5 forcing-OOM fix — the driver
        passes ``xp=np`` so the ``[n_steps, nod2D]`` per-chunk forcing AND the intermediate
        list of ``n_steps`` per-date forcings are never materialized on GPU 0 before
        :func:`shard_mesh.partition_step_forcing` shards them). Default ``jnp`` = unchanged."""
        steps = [self.step_forcing(*d, xp=xp) for d in dates]
        return StepForcing(*[xp.stack(leaves, axis=0)
                             for leaves in zip(*steps)])

    def reopen_year(self, year: int):
        """Roll the JRA reader to ``year`` IN PLACE (keeps the year-independent interpolation
        stencil; see :meth:`fesom_jax.jra55.JRA55Reader.reopen_year`). SSS-restoring and chl are
        monthly CLIMATOLOGIES (year-independent), so only the JRA reader rolls. Returns ``self``."""
        self.jra.reopen_year(year)
        return self


def build_surface_forcing(mesh: Mesh, year: int, *, sst_ic=None,
                       jra_dir: str | None = None,
                       sss_path: str | None = None,
                       runoff_path: str | None = None,
                       chl_path: str | None = None,
                       chl_const: float | None = None) -> SurfaceForcing:
    """Build the surface-forcing driver for a JRA55-do ``year`` (host setup, ~seconds).
    Mesh-agnostic: works on any :class:`~fesom_jax.mesh.Mesh` (pi, farc, dars, CORE2, NG5).

    The four input-path kwargs default to ``None`` ⇒ each reader resolves its own path
    through :mod:`fesom_jax.paths` (``$FESOM_JRA_DIR`` / ``$FESOM_SSS_PATH`` /
    ``$FESOM_RUNOFF_PATH`` / ``$FESOM_CHL_PATH``, else the Levante default). Pass a string
    (e.g. from the run YAML's ``forcing:`` block) to override.

    ``sst_ic`` (``[nod2D]``, the PHC IC surface temperature) builds the static ``a_ice``
    mask (:func:`ice_ic_aice`); pass ``state.T[:, 0]`` from
    :func:`fesom_jax.phc_ic.core2_initial_state`. ``None`` ⇒ ``a_ice≡0`` (truly ice-free;
    e.g. constant-T sanity runs).

    ``chl_const`` (default ``None``) selects the chl source: ``None`` ⇒ the Sweeney
    monthly climatology (the C default / production config); a float ⇒ a constant chl
    everywhere (the ``FESOM_CHL_SOURCE=None`` alternative, ``FESOM_PHASE1_CHL_CONST``)."""
    jra = jra55.JRA55Reader(mesh, year, jra_dir)
    sss = sss_runoff.build_reader(mesh, sss_path, runoff_path)
    if chl_const is None:
        chl_clim = sss_runoff.build_chl_clim(mesh, chl_path)
    else:
        chl_clim = np.full((12, int(mesh.nod2D)), float(chl_const), dtype=np.float64)
    a_ice = (ice_ic_aice(mesh, sst_ic) if sst_ic is not None
             else jnp.zeros(int(mesh.nod2D)))
    return SurfaceForcing(jra=jra, sss=sss, chl_clim=chl_clim,
                       static=_static_from_mesh(mesh, sss.runoff_node, a_ice))


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
