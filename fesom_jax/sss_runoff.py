"""SSS restoring (PHC2_salx) + CORE2 runoff (Task 5.5) — a faithful **numpy** port of
``fesom_sss_runoff.c`` (``interp_2d_field`` + ``read_other_NetCDF`` + the
``fesom_sss_runoff_step`` oce_fluxes salt/water balance) for the single-rank,
no-cavity CORE2 case.

Two parts live here, mirroring the bulk (:mod:`fesom_jax.forcing`) split:

* **Host-numpy readers** (:class:`SSSRunoffReader`) — non-differentiable SETUP, like
  :mod:`fesom_jax.phc_ic` / :mod:`fesom_jax.jra55`. Read the monthly SSS climatology
  (12 months → ``Ssurf_clim[12, nod2D]``) and the constant CORE2 runoff
  (``runoff_node[nod2D]``, ``(kg/s)/m² → /1000 → m/s``) off disk, bilinearly
  interpolating each onto the mesh nodes. Verified bit-for-bit-class against the C
  ``sss_dump_*`` dumps.
* **AD-safe JAX flux math** (:func:`sss_runoff_fluxes`) — the per-step
  ``oce_fluxes`` salt + water balance (``fesom_sss_runoff_step:382-440``). A pure,
  differentiable function of ``(S_top, water_flux, Ssurf_month, runoff_node,
  areasvol_surf, ocean_area)``: it consumes the bulk ``water_flux`` (Task 5.4) and the
  model surface salinity ``S[:,0]`` and produces ``(virtual_salt, relax_salt,
  water_flux_balanced)``. ``d(virtual_salt)/d(SST)`` flows through ``water_flux``.

Pipeline (``fesom_sss_runoff.c``):
  1. ``read_other_NetCDF`` (``:122``): read a (time,lat,lon) slice; fill missing values
     (30-cell expanding-neighbourhood mean when ``check_dummy``, else 0); bilinearly
     interpolate to nodes via ``interp_2d_field``.
  2. ``interp_2d_field`` (``:34``): bilinear interp from a regular (lon,lat) grid to model
     nodes — clamp in latitude, **cyclic-wrap** in longitude. This is a DIFFERENT routine
     than the JRA55 stencil (``gen_interpolation.F90`` vs the bulk reader's), so it is
     ported on its own.
  3. ``fesom_sss_runoff_step`` (``:341``): ``rsss = S_top`` (``ref_sss_local=1``);
     ``virtual_salt = rsss·water_flux``; ``relax_salt = surf_relax_S·(Ssurf − S_top)``;
     each has its **area-weighted global mean** subtracted (non-cavity); then
     ``water_flux += mean(water_flux + runoff)`` (all nodes — the asymmetry is real).

⚠️ ``ref_sss_local = 1`` ⇒ ``rsss`` is the LOCAL surface salinity, not the constant 34.7.
⚠️ Month index has **no legacy +1** (``:351-359``) — our flag fires on the first step of
the new month where ``month_now`` is already the new month. ⚠️ The runoff missing value is
``1e30`` (``check_dummy=0`` → land → 0); the SSS missing value is ``-99`` (``check_dummy=1``
→ 30-cell expanding fill).
"""

from __future__ import annotations

import dataclasses
import os
from typing import NamedTuple

import numpy as np
import netCDF4
from scipy.ndimage import uniform_filter

import jax.numpy as jnp
from jax import lax

from . import paths
from .config import RAD

# Import-time snapshots of the resolved defaults ($FESOM_SSS_PATH / $FESOM_RUNOFF_PATH /
# $FESOM_CHL_PATH → the Levante defaults), kept importable for the callers that use the
# names. The readers below re-resolve at CALL time via :mod:`fesom_jax.paths`.
DEFAULT_SSS_PATH = paths.resolve("sss_path")
DEFAULT_RUNOFF_PATH = paths.resolve("runoff_path")
DEFAULT_CHL_PATH = paths.resolve("chl_path")

# surf_relax_S = 10 / (60·3600·24) = 1.929e-6 s⁻¹ (fesom_sss_runoff_init:305).
SURF_RELAX_S = 10.0 / (60.0 * 3600.0 * 24.0)
_MISS_DEFAULT = -1.0e30                     # read_other_NetCDF:199 (absent-attr sentinel)
_MISS99 = -99.0                             # the Fortran also treats -99 as missing (:211)
_FILL_MAX_K = 30                            # expanding-neighbourhood radius (:215)


# --------------------------------------------------------------------------
# interp_2d_field — gen_interpolation.F90:145-288  (fesom_sss_runoff.c:34-113)
# --------------------------------------------------------------------------
def _interp_2d_field(lon_reg, lat_reg, data_reg, lon_mod, lat_mod):
    """Bilinear interp from a regular (lon,lat) grid to model nodes — vectorized
    port of ``interp_2d_field`` (``fesom_sss_runoff.c:34``). ``data_reg`` is
    ``[num_lat, num_lon]`` (row-major lat,lon). Latitude is **clamped** to the grid
    range; longitude **cyclic-wraps** across the 0/360 seam. The corner blend order
    matches the C exactly (``(s·rt_lon1 + s·rt_lon2)·rt_lat`` per lat row, summed)."""
    lon_reg = np.asarray(lon_reg, dtype=np.float64)
    lat_reg = np.asarray(lat_reg, dtype=np.float64)
    if lon_reg[0] < 0.0 or lon_reg[-1] > 360.0:
        raise ValueError("interp_2d_field: regular grid lon out of [0,360]")
    num_lat = lat_reg.shape[0]
    num_lon = lon_reg.shape[0]

    # --- latitude (clamp + linear-search bracket, C:55-76) ---
    y = np.clip(np.asarray(lat_mod, dtype=np.float64), lat_reg[0], lat_reg[-1])
    jh = np.clip(np.searchsorted(lat_reg, y, side="left"), 1, num_lat - 1)
    jl = jh - 1
    diff_lat = lat_reg[jh] - lat_reg[jl]
    rt_lat1 = (lat_reg[jh] - y) / diff_lat
    rt_lat2 = 1.0 - rt_lat1

    # --- longitude (cyclic wrap when out of range, C:78-103) ---
    x = np.asarray(lon_mod, dtype=np.float64)
    below = x < lon_reg[0]
    above = x > lon_reg[-1]
    in_range = ~(below | above)
    ih_ir = np.clip(np.searchsorted(lon_reg, x, side="left"), 1, num_lon - 1)
    il_ir = ih_ir - 1
    # wrap branch: ind_lon_h = 0 (first), ind_lon_l = num_lon-1 (last), across the seam.
    ih = np.where(in_range, ih_ir, 0)
    il = np.where(in_range, il_ir, num_lon - 1)
    diff_wrap = lon_reg[0] + (360.0 - lon_reg[-1])
    diff_ir = lon_reg[ih_ir] - lon_reg[il_ir]
    rt_lon1_ir = (lon_reg[ih_ir] - x) / diff_ir          # weight toward lon_l (C:101)
    rt_lon1_below = (lon_reg[0] - x) / diff_wrap         # C:83
    rt_lon2_above = (x - lon_reg[-1]) / diff_wrap        # C:89
    rt_lon1 = np.where(in_range, rt_lon1_ir,
                       np.where(below, rt_lon1_below, 1.0 - rt_lon2_above))
    rt_lon2 = 1.0 - rt_lon1

    # corners: data_<lon><lat> — ll=(lon_l,lat_l), hl=(lon_h,lat_l), lh=(lon_l,lat_h)…
    data_ll = data_reg[jl, il]
    data_hl = data_reg[jl, ih]
    data_lh = data_reg[jh, il]
    data_hh = data_reg[jh, ih]
    return ((data_ll * rt_lon1 + data_hl * rt_lon2) * rt_lat1
            + (data_lh * rt_lon1 + data_hh * rt_lon2) * rt_lat2)


def _fill_missing_expand(data, valid):
    """Expanding-neighbourhood missing-value fill — port of ``read_other_NetCDF``'s
    ``check_dummy`` branch (``fesom_sss_runoff.c:212-233``). Each missing cell is set to
    the mean of the non-missing cells in the **smallest** ``(2k+1)×(2k+1)`` box (k≤30,
    clamped at the grid edge, NOT cyclic) that contains any data — reading from the
    *original* field (the C uses ``ncdata_temp``, so fills do not cascade ⇒ vectorizable,
    order-independent). The box mean is a reduction (~1e-14, the ``/count`` crushes the
    sum reassociation), so coastal cells match the C to reduction-class, not bit-for-bit."""
    data = np.asarray(data, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out = data.copy()
    remaining = ~valid
    if not remaining.any():
        return out
    data0 = np.where(valid, data, 0.0)
    cnt0 = valid.astype(np.float64)
    for k in range(1, _FILL_MAX_K + 1):
        if not remaining.any():
            break
        size = 2 * k + 1
        win = float(size * size)
        s = uniform_filter(data0, size=size, mode="constant", cval=0.0) * win
        c = uniform_filter(cnt0, size=size, mode="constant", cval=0.0) * win
        fill = remaining & (c > 0.5)                      # ≥1 valid cell in the box
        out[fill] = s[fill] / c[fill]
        remaining &= ~fill
    return out                                            # deep-land cells keep the sentinel


def _read_other_netcdf(path, vari, itime, lon_mod, lat_mod, check_dummy):
    """Port of ``fesom_read_other_NetCDF`` (``fesom_sss_runoff.c:122``) for npes=1: read the
    1-based ``itime`` (time,lat,lon) slice, fill missing values, bilinear-interp to nodes.
    Returns ``model_2D[nod2D]`` (float64)."""
    ds = netCDF4.Dataset(path)
    try:
        for v in ds.variables.values():
            v.set_auto_maskandscale(False)               # raw float (C nc_get_vara_float)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.where(lon < 0.0, lon + 360.0, lon)      # range fix (:166-168)
        ncvar = ds.variables[vari]
        raw = np.asarray(ncvar[itime - 1, :, :], dtype=np.float64)   # (lat, lon)
        miss = _MISS_DEFAULT
        if "missing_value" in ncvar.ncattrs():           # read but tolerate absence (:199-201)
            miss = float(np.float64(np.float32(ncvar.missing_value)))
    finally:
        ds.close()

    bad = (raw == miss) | (raw == _MISS99)               # the C also treats -99 as missing
    if check_dummy:
        data = _fill_missing_expand(raw, ~bad)
    else:
        data = raw.copy()
        data[bad] = 0.0
    return _interp_2d_field(lon, lat, data, lon_mod, lat_mod)


# --------------------------------------------------------------------------
# Host-numpy reader
# --------------------------------------------------------------------------
@dataclasses.dataclass
class SSSRunoffReader:
    """SSS climatology + CORE2 runoff interpolated to mesh nodes (host numpy).

    ``Ssurf_clim``: ``[12, nod2D]`` monthly surface-salinity restoring target (PSU).
    ``runoff_node``: ``[nod2D]`` constant runoff (m/s, already ``/1000``)."""
    Ssurf_clim: np.ndarray
    runoff_node: np.ndarray

    def month(self, m: int) -> np.ndarray:
        """Surface-salinity target for **1-based** month ``m`` (``:362`` reads slice ``m``)."""
        return self.Ssurf_clim[m - 1]


def build_reader(mesh, sss_path: str | None = None,
                 runoff_path: str | None = None) -> SSSRunoffReader:
    """Read the SSS climatology (12 months) + CORE2 runoff and interpolate to ``mesh``
    nodes (npes=1). Mirrors ``fesom_sss_runoff_init`` (runoff once, ``/1000``) + the
    per-month ``SALT`` reads in ``fesom_sss_runoff_step``.

    ``sss_path``/``runoff_path`` ``None`` ⇒ resolved from ``$FESOM_SSS_PATH`` /
    ``$FESOM_RUNOFF_PATH``, else the Levante defaults (:mod:`fesom_jax.paths`)."""
    sss_path = paths.require("sss_path", sss_path)
    runoff_path = paths.require("runoff_path", runoff_path)
    geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64) / RAD
    lon_mod = np.where(geo[:, 0] < 0.0, geo[:, 0] + 360.0, geo[:, 0])   # <0 wrap (:250)
    lat_mod = geo[:, 1].copy()

    # runoff: single record, (kg/s)/m² → /1000 → m/s; check_dummy=0 (land → 0).
    runoff_node = _read_other_netcdf(runoff_path, "Foxx_o_roff", 1, lon_mod, lat_mod,
                                     check_dummy=False) / 1000.0
    # SSS: 12 monthly slices, check_dummy=1 (30-cell expanding fill).
    Ssurf_clim = np.stack(
        [_read_other_netcdf(sss_path, "SALT", m, lon_mod, lat_mod, check_dummy=True)
         for m in range(1, 13)], axis=0)
    return SSSRunoffReader(Ssurf_clim=Ssurf_clim, runoff_node=runoff_node)


def build_chl_clim(mesh, chl_path: str | None = None) -> np.ndarray:
    """Chlorophyll monthly climatology interpolated to mesh nodes — host numpy port of
    the ``chl`` read in ``fesom_main.c:1111`` (``fesom_read_other_NetCDF(chl_file, "chl",
    mi, …, /*check_dummy=*/1, /*do_onvert=*/1)``). The CORE2 default source is
    ``Sweeney_2005.nc`` (``TIME=12, lat=180, lon=360`` — same format/routine as the SSS
    climatology); ``FESOM_CHL_SOURCE`` defaults to Sweeney, so this is the matched config.

    Returns ``chl_clim[12, nod2D]`` (mg/m³); index by ``m-1`` for 1-based month ``m``.
    For the constant-chl alternative (``FESOM_CHL_SOURCE=None`` ⇒ ``FESOM_PHASE1_CHL_CONST
    = 0.1``) just pass ``jnp.full(nod2D, 0.1)`` to :func:`fesom_jax.forcing.cal_shortwave_rad`
    instead of this climatology — the seam is the ``chl`` argument.

    ``chl_path=None`` ⇒ resolved from ``$FESOM_CHL_PATH``, else the Levante default
    (:mod:`fesom_jax.paths`)."""
    chl_path = paths.require("chl_path", chl_path)
    geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64) / RAD
    lon_mod = np.where(geo[:, 0] < 0.0, geo[:, 0] + 360.0, geo[:, 0])
    lat_mod = geo[:, 1].copy()
    return np.stack(
        [_read_other_netcdf(chl_path, "chl", m, lon_mod, lat_mod, check_dummy=True)
         for m in range(1, 13)], axis=0)


# --------------------------------------------------------------------------
# AD-safe JAX flux math — fesom_sss_runoff_step:382-440
# --------------------------------------------------------------------------
class SSSFluxes(NamedTuple):
    """Per-step salt/water-balance output (a pytree), each ``[nod2D]``."""
    virtual_salt: jnp.ndarray       # rsss·water_flux − ⟨·⟩ (PSU·m/s)
    relax_salt: jnp.ndarray         # surf_relax_S·(Ssurf − S_top) − ⟨·⟩ (PSU·m/s)
    water_flux: jnp.ndarray         # bulk water_flux + ⟨water_flux + runoff⟩ (m/s)


def _area_mean(x, areasvol_surf, ocean_area, *, owned_mask=None, axis_name=None):
    """``integrate_nod_2D(x)/ocean_area`` (``:270-284``, ``:393``): area-weighted global
    mean ``Σ(x·areasvol_surf)/ocean_area``. A reduction (~1e-12). Summed over **all**
    interior nodes (the C integrate carries no cavity mask).

    Phase 8: when ``owned_mask`` is given (sharded path, S.7 threads it in) the sum is
    an owned-node sum + ``psum`` over devices via :func:`reductions.global_sum`. The
    default (``owned_mask=None``) is the **exact** single-device ``jnp.sum`` ⇒ the
    ``npes==1`` path is byte-identical to ``v1.0``.

    A ``lax.optimization_barrier`` on ``x`` (DEFAULT ON; ``FESOM_BALANCE_BARRIER=0``
    opts out, trace-time static) stops XLA fusing this reduction into its producer —
    semantically the identity, and verified bitwise-identical on CPU and CUDA at the
    exact affected shape.  Root-cause fix for the ng5 x P=128 cliff (HANDOFF-20260724):
    at exactly those shapes (Lmax_nod=59637) XLA's multi_output_fusion merged this
    reduce + the whole vmapped ice-thermodynamics producer + the post-subtract consumers
    into one 1,425-op kind=kInput fusion whose kernel runs 383 ms once per step (nsys
    job 1036137) — the entire 2.7x cliff (647.1 -> 236.2 ms/step with the barrier,
    job 1036604).  The barrier costs one [Lmax_nod] f64 materialisation (~0.5 MB) per
    call; P=64 measured unchanged (barrier64 leg).  ``=0`` exists to reproduce the sick
    behaviour deliberately."""
    if os.environ.get("FESOM_BALANCE_BARRIER", "1") != "0":
        x = lax.optimization_barrier(x)
    if owned_mask is None:
        return jnp.sum(x * areasvol_surf) / ocean_area
    from . import reductions
    return reductions.global_sum(x * areasvol_surf, owned_mask, axis_name) / ocean_area


def sss_runoff_fluxes(S_top, water_flux, Ssurf_month, runoff_node,
                      areasvol_surf, ocean_area, open_water=None,
                      balance_water_flux=True, *, owned_mask=None,
                      axis_name=None) -> SSSFluxes:
    """AD-safe port of the ``oce_fluxes`` salt + water balance
    (``fesom_sss_runoff_step:382-440``). Pure, differentiable function:

    * ``virtual_salt = S_top·water_flux``, minus its area-weighted global mean
      (``ref_sss_local=1`` ⇒ ``rsss = S_top``; ``use_virt_salt=1`` for linfs).
    * ``relax_salt = surf_relax_S·(Ssurf_month − S_top)``, minus its global mean.
    * ``water_flux += ⟨water_flux + runoff_node⟩`` over **all** nodes (no cavity skip —
      the asymmetry vs the salt terms is real, ``:422-438``).

    The global-mean subtractions skip cavity nodes (``open_water=False``); on CORE2 there
    are none, so ``open_water`` defaults to all-True. ``virtual_salt`` uses the **bulk**
    (pre-balance) ``water_flux``, matching the C ordering (balance is step 4, last).

    ``balance_water_flux`` — the no-ice (Phase-5 standalone ``fesom_sss_runoff_step``) path
    adds the ``⟨water_flux + runoff⟩`` term (default ``True``). The **ice-on** path
    (``fesom_ice_oce_fluxes``, ``fesom_ice_coupling.c:125-179``) does the virtual_salt +
    relax_salt balancing but **NOT** this term — runoff is already inside ``flx_fw`` and
    ``water_flux = -flx_fw`` is taken as-is. Pass ``False`` from the ice coupling. The
    returned ``water_flux`` is then the input unchanged.

    All ops are smooth (multiply / sum / divide-by-constant ``ocean_area``); ``areasvol_surf``
    and ``ocean_area`` are strictly positive, so no AD guards are needed — ``d/d(water_flux)``
    (hence ``d/d(SST)`` via the bulk) and ``d/d(S_top)`` flow cleanly."""
    water_flux = jnp.asarray(water_flux)
    S_top = jnp.asarray(S_top)
    if open_water is None:
        open_water = jnp.ones_like(water_flux, dtype=bool)

    # virtual_salt (C:384-400) — rsss = S_top (ref_sss_local=1).
    virtual_salt = S_top * water_flux
    virtual_salt = jnp.where(open_water,
                             virtual_salt - _area_mean(virtual_salt, areasvol_surf, ocean_area,
                                       owned_mask=owned_mask, axis_name=axis_name),
                             virtual_salt)

    # relax_salt (C:404-417).
    relax_salt = SURF_RELAX_S * (Ssurf_month - S_top)
    relax_salt = jnp.where(open_water,
                           relax_salt - _area_mean(relax_salt, areasvol_surf, ocean_area,
                                       owned_mask=owned_mask, axis_name=axis_name),
                           relax_salt)

    # water balance (C:419-440) — add ⟨water_flux + runoff⟩ to every node. The ice-on path
    # (balance_water_flux=False) skips it (runoff is already inside flx_fw → water_flux).
    if balance_water_flux:
        flux = water_flux + runoff_node
        water_flux = water_flux + _area_mean(flux, areasvol_surf, ocean_area,
                                             owned_mask=owned_mask, axis_name=axis_name)

    return SSSFluxes(virtual_salt=virtual_salt, relax_salt=relax_salt,
                     water_flux=water_flux)
