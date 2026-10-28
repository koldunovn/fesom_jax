"""REcoM's cold-start state, exactly as FESOM2 + REcoM build it (parent plan Task 13, lane I).

A one-time, host-side numpy setup (as :mod:`fesom_jax.phc_ic`), outside the autodiff path. The entry point is
:func:`bgc_initial_state`; :func:`with_bgc_ic` puts the tracers into a :class:`~fesom_jax.state.State`.
Sources: FESOM2 e96a2b13 (``src/``), REcoM 692f804. Lines cited as ``file:line``.

The sequence (cold start, ``r_restart = .false.``, no cavities, no partial cells):

1. ``tracer_init`` (``oce_setup_step.F90:460-475``): every tracer ``values = 0``, ``valuesold = 0``.
2. ``ocean_setup`` -> ``init_ale`` (``oce_ale.F90:202-236``): the IC-time mid-layer depths ``Z_3d_n``: the
   top layer ``zbar(1) + (zbar(2) - zbar(1))/2``, the interior ``Z``, the bottom wet layer
   ``zbar(nzmax-1) + (zbar(nzmax) - zbar(nzmax-1))/2`` (:func:`ic_layer_depths`).
3. ``oce_initial_state`` -> ``do_ic3d`` (``gen_ic3d.F90:548-620``), per ``tracer_init3d`` entry
   (:func:`interp_ic_field`):

   * ``nc_readGrid`` (``gen_ic3d.F90:90-260``): dimensions ``LAT|lat|latitude``, ``LON|longitude|lon``,
     ``depth``; the longitudes padded by one cyclic column on each side, ``nc_lon(1) - 360``,
     ``nc_lon(N) + 360``; ``nc_depth`` negated if ``nc_depth(2) < 0``.
   * ``nc_ic3d_ini`` (``:306-325``): per node ``x = lon/rad`` (+360 if negative), ``y = lat/rad``,
     ``binarysearch`` (``:933-970``) for the bracket, ``-1`` outside the grid.
   * ``getcoeffld`` (``:327-546``): ``values = dummy``; the raw data (no scale/offset) with NaN, the variable's
     fill value, ``< -0.99 dummy`` and ``> dummy`` set to ``dummy``; per node: skipped if any surface corner is
     dummy, else ``((v*(x2-x))*(y2-y) + ...)/denom`` for all depths with a dummy corner giving dummy, then per wet
     layer the depth bracket of ``-Z_3d_n``: linear ``-cf_a*Z + cf_b`` between two non-dummy depths, the first
     depth's value above the first file depth, dummy below the last.
   * non-finite -> 0, then ``extrap_nod`` = ``extrap_nod3D`` (``gen_support.F90:387-507``; ``ic_extrap_det``
     defaults to ``.false.``, ``gen_support.F90:25``, and ``namelist.tra`` does not set it): only if the SURFACE
     layer has a dummy somewhere, per layer an in-place Gauss-Seidel sweep in node order repeated until nothing is
     filled (a dummy wet node takes the mean of its valid wet neighbours over the elements reaching one level
     deeper than their bottom, ``nz > nlevels(el)`` is the skip test), then a top-down vertical fill of the wet
     layers. One rank: the serial order is the Fortran order (the reference ran on ``dist_1``).
   * cleanup over ALL tracers (``:583-607``): non-finite, ``> 0.9 dummy`` -> 0; levels at and below
     ``nlevels_nod2D - 1`` -> 0. (The K -> degC and ``insitu2pot`` steps touch tracer 1 only.)

4. ``oce_initial_state`` (``oce_setup_step.F90:1251-1281``): ``Alk_surf = values(1, :)`` of tracer 1003 when
   ``restore_alkalinity`` (else it keeps its ``arrays_init`` zero, ``:1108``); before any restart read.
   Then (``:1300-1420``) every BGC ID without an IC file is set to 0 (1004:1017, 1020:1021, 1023:1037).
5. ``ocean_setup`` (``oce_setup_step.F90:264-268``): ``valuesold = values``. So ``bgc_old`` holds the do_ic3d
   fields WITHOUT REcoM's changes below: Fe in the file's unit (no x1e9, no cap, no floor), zero for the tracers
   REcoM sets to constants. This is FESOM2's AB2 history at step 1.
6. ``recom_init`` (``fesom_module.F90:562``, ``recom_init.F90:47-144``), after ``ocean_setup``, on ``values``
   only (``tracers_info%data_pointers => values``, ``fesom_module.F90:557``):

   * Fe (1019) ``* 1.e9`` on all levels (``:120-124``);
   * every ID ``> 1003`` except 1018, 1022 and DICremin (1037) -> ``get_tracer_init_value``
     (``:126-130``, ``:647-708``; :func:`fesom_jax.recom.tracers.init_value`), on ALL levels, the ones below
     the bottom included (the dump shows them there);
   * ``mask_hydrothermal_vents`` (``:710-754``): Fe = ``min(0.3, Fe)`` (``-r8``: 0.3d0) on the wet layers with
     ``|Z_3d_n| >= 2000`` m of the nodes in the box ``-12.5 < lat < 9.5``, ``-106 < lon < -72`` (degrees x
     ``rad``, geographic radians with lon in ``(-pi, pi]``); then ``Fe = max(tiny, Fe)`` on all levels, so below
     the bottom Fe is ``tiny`` (2.23e-16);
   * ``initialize_memory`` (``:146-363``): ``Benthos = 0`` (4 pools), ``GloHplus = exp(-8 log 10)``; every
     other array (the carbonate cache included) starts at 0. ``ciso`` is off, so ``initialize_ciso*`` do nothing.

What :mod:`fesom_jax.phc_ic` provides and what changed since its port (it mirrors ``fesom_phc.c``, the C port of
an earlier FESOM2 ``gen_ic3d``):

* reused unchanged: :func:`fesom_jax.phc_ic._binarysearch_d` (= Fortran ``binarysearch`` minus one: the same
  1e-9 match and the rounded midpoint ``nint((l+r)/2)``); the bilinear association ``((v*wx)*wy)`` summed left
  to right; the per-depth corner-dummy rule and the vertical interpolation formula;
* changed here: the vertical bracket is per node from ``-Z_3d_n`` (IC-time depths, which differ from ``Z`` by
  rounding in the bottom layer) instead of ``-Z``; ``nc_depth`` sign flip; the fill-value test; the extrapolation
  runs only while the surface layer has a dummy (``do while (glob_max > 0.99 dummy)``) and has no 200-sweep cap;
  ``rad = pi/180`` with the full ``pi`` (``oce_modules.F90:34-35``), not :data:`fesom_jax.config.RAD`
  (0.017453292519943278, the C port's truncated pi);
* not ported: cavities (``use_cavity``), partial cells, ``ic_extrap_det``, the IC perturbation (``lperturb``).

Division as the reference binary does it. FESOM2 is built with ``-fp-model precise -no-prec-div`` and no ``-x``
flag (SSE2, ``docs/recom/FORTRAN_REFERENCE.md``). ``-no-prec-div`` lets ifort replace a division by a
loop-invariant divisor with a multiplication by its reciprocal. Matching the step-1 dump bit for bit takes:
``x = geo/rad`` as ``geo * (1/rad)`` (the constant folded), ``data1d(:) = (...)/denom`` as ``(...) * (1/denom)``
for the depths the 2-wide vector loop covers and a true division for the last depth of an odd-length column (the
scalar remainder: the 33-depth Fe file shows it at 19 points, a 1-ulp change that the deep vertical
interpolation turns into 1 ulp in the field); ``(d2-d1)/delta_d`` and the extrapolation mean ``val/cnt`` stay
true divisions. This is a property of that build, inferred from the dump, not of the source; with plain
divisions everywhere the fields agree to <= 2e-14 relative.

Input coordinates: the FESOM2 geographic node coordinates. The JAX meshes carry their own ``geo_coord_nod2D``
(exported by the C port), which on pi differ from Fortran's by up to ~1e-14 rad (lon in ``[0, 2 pi)`` instead
of ``(-pi, pi]``, and the rotation computed differently); ``geo_coord`` overrides them (the gate passes the
dumped Fortran coordinates to compare the algorithm bitwise).

Not mirrored: Fortran's run-stops (an IC ID missing from the tracer list aborts; a surface hole that no
neighbour can reach makes ``extrap_nod3D`` loop forever) raise here instead.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Mapping, NamedTuple, Optional, Tuple, Union

import netCDF4
import numpy as np

from ..phc_ic import _binarysearch_d
from . import params as P
from . import tracers as T

DUMMY = 1.0e10                       # g_config dummy (gen_modules_config.F90:262)
RAD = math.pi / 180.0                # o_PARAM rad = pi/180, pi to 36 digits (oce_modules.F90:34-35)
INV_RAD = 1.0 / RAD                  # the folded constant of x/rad under -no-prec-div
HPLUS0 = math.exp(-8.0 * math.log(10.0))   # GloHplus at allocation (recom_init.F90:226)
VENT_CAP = 0.3                       # mask_hydrothermal_vents (recom_init.F90:742; -r8 => 0.3d0)
VENT_DEPTH = 2000.0                  # recom_init.F90:739
BENTHOS_NUM = 4                      # benthic N, C, Si, Calc (recom_modules.F90 benthos_num)
FE_FACTOR = 1.0e9                    # recom_init.F90:122-123 (comment: mol/L => umol/m3)

# The CORE2 (1-degree) IC set of upstream config/bin_2p3z2d namelist.tra (&tracer_init3d), BGC entries only.
CORE2_IC_FILES = {
    1001: ("woa13_all_n00_01_fesom2.nc", "n_an"),
    1002: ("GLODAPv2.2016b.PI_TCO2_fesom2_mmol_fix_z_Fillvalue.nc", "PI_TCO2_mmol"),
    1003: ("GLODAPv2.2016b.TAlk_fesom2_mmol_fix_z_Fillvalue.nc", "TAlk_mmol"),
    1018: ("woa13_all_i00_01_fesom2.nc", "i_an"),
    1019: ("fe_pisces_opa_eq_init_3D_changed_name.nc", "Fe"),
    1022: ("woa18_all_o00_01_mmol_fesom2.nc", "oxygen_mmol"),
}
# The same with the present-day DIC that config/bin_2p3z2d namelist.tra names for present-day runs (its comment at
# &tracer_init3d: "for present-day runs use GLODAPv2.2016b.TCO2_fesom2_mmol_fix_z_Fillvalue.nc with varlist TCO2_mmol")
CORE2_PRESENT_IC_FILES = {**CORE2_IC_FILES, 1002: ("GLODAPv2.2016b.TCO2_fesom2_mmol_fix_z_Fillvalue.nc", "TCO2_mmol")}
PI_IC_FILES = dict(T.IC_FILES)       # the 5-degree pi set (test/input/recom), lane F's reference run

_LAT_NAMES = ("LAT", "lat", "latitude")       # gen_ic3d.F90:122-128, 162-169
_LON_NAMES = ("LON", "longitude", "lon")      # gen_ic3d.F90:132-138, 171-178

PathLike = Union[str, Path]


class BgcIC(NamedTuple):
    """The cold-start REcoM state (levels are FESOM's ``nl-1`` layers, index 0 = the surface layer)."""
    bgc: np.ndarray        # (nod2D, nl-1, 31) BGC tracers after recom_init, tracers.BGC_IDS slot order
    bgc_old: np.ndarray    # (nod2D, nl-1, 31) valuesold = the do_ic3d fields before recom_init
    alk_surf: np.ndarray   # (nod2D,) Alk_surf, the alkalinity restoring target
    benthos: np.ndarray    # (nod2D, 4) Benthos [mmol m-2], zero
    hplus: np.ndarray      # (nod2D,) GloHplus at allocation, exp(-8 ln 10)


# --------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------
def _binarysearch_vec(arr: np.ndarray, values: np.ndarray) -> np.ndarray:
    """:func:`_binarysearch_d` (= ``gen_ic3d.F90:933`` ``binarysearch`` minus one) for many values in lockstep."""
    arr = np.asarray(arr, np.float64)
    v = np.asarray(values, np.float64)
    left = np.zeros(v.shape, np.int64)
    right = np.full(v.shape, arr.size - 1, np.int64)
    ind = np.full(v.shape, -2, np.int64)
    live = left <= right
    while live.any():
        mid = (left + right + 1) // 2
        a = arr[np.where(live, mid, 0)]
        hit = live & (np.abs(a - v) <= 1e-9)
        ind[hit] = mid[hit]
        go_left = live & ~hit & (a > v)
        go_right = live & ~hit & ~(a > v)
        right = np.where(go_left, mid - 1, right)
        left = np.where(go_right, mid + 1, left)
        live = live & ~hit & (left <= right)
    return np.where(ind == -2, right, ind)


def ic_layer_depths(mesh) -> np.ndarray:
    """``Z_3d_n`` as ``init_ale`` sets it before the IC is read (``oce_ale.F90:202-236``, no cavity, full
    bottom cells: ``zbar_n_srf = zbar(1)``, ``zbar_n_bot = zbar(nlevels_nod2D)``, ``oce_ale.F90:449-452``).
    Returns ``(nod2D, nl-1)``, zero below the wet layers. Statements in Fortran order (the bottom wins)."""
    zbar = np.asarray(mesh.zbar, np.float64)
    Z = np.asarray(mesh.Z, np.float64)
    nlev = np.asarray(mesh.nlevels_nod2D, np.int64)
    ulev = np.asarray(mesh.ulevels_nod2D, np.int64)
    if np.any(ulev != 1):
        raise NotImplementedError("REcoM IC with cavities (ulevels_nod2D > 1) is not ported")
    N, nl = nlev.size, zbar.size
    z3 = np.zeros((N, nl - 1))
    k = np.arange(nl - 1)[None, :]
    nzmax = nlev[:, None]                                   # Fortran nzmax = nlevels_nod2D (1-based)
    # Z_3d_n(nzmin) = zbar_3d_n(nzmin) + (zbar_3d_n(nzmin+1) - zbar_n_srf)/2, nzmin = 1 -> index 0
    z3[:, 0] = zbar[0] + (zbar[1] - zbar[0]) / 2
    # Z_3d_n(nzmin+1:nzmax-2) = Z(nzmin+1:nzmax-2) -> 0-based 1 .. nzmax-3
    inner = (k >= 1) & (k <= nzmax - 3)
    z3 = np.where(inner, Z[None, :], z3)
    # Z_3d_n(nzmax-1) = zbar_3d_n(nzmax-1) + (zbar_n_bot - zbar_3d_n(nzmax-1))/2 -> 0-based nzmax-2
    kb = nlev - 2
    z3[np.arange(N), kb] = zbar[kb] + (zbar[nlev - 1] - zbar[kb]) / 2
    return z3


def _canonical_lon(lon: np.ndarray) -> np.ndarray:
    """Geographic longitude in Fortran's ``(-pi, pi]`` (the JAX meshes store ``[0, 2 pi)``); an input already in
    that range is returned unchanged (bit for bit)."""
    return np.where(lon > math.pi, lon - 2.0 * math.pi, lon)


def _find(names, pool, what, path):
    for n in names:
        if n in pool:
            return n
    raise KeyError(f"{path}: no {what} among {names} (gen_ic3d.F90 nc_readGrid)")


def _read_ic_file(path: PathLike, varname: str):
    """``nc_readGrid`` + the read and dummy replacement of ``getcoeffld`` (``gen_ic3d.F90:90-260, 361-414``).
    Returns ``(nc_lon (Nlon+2,), nc_lat, nc_depth, ncdata (Nlon+2, Nlat, Ndepth))``."""
    nc = netCDF4.Dataset(str(path))
    try:
        nc.set_auto_maskandscale(False)            # Fortran nf_get_vara_double: no mask, no scale/offset
        dlat = _find(_LAT_NAMES, nc.dimensions, "latitude dimension", path)
        dlon = _find(_LON_NAMES, nc.dimensions, "longitude dimension", path)
        if "depth" not in nc.dimensions:
            raise KeyError(f"{path}: no 'depth' dimension (gen_ic3d.F90:156)")
        vlat = _find(_LAT_NAMES, nc.variables, "latitude variable", path)
        vlon = _find(_LON_NAMES, nc.variables, "longitude variable", path)
        nc_lat = np.asarray(nc.variables[vlat][:], np.float64)
        lon_real = np.asarray(nc.variables[vlon][:], np.float64)
        nc_depth = np.asarray(nc.variables["depth"][:], np.float64)
        var = nc.variables[varname]
        if tuple(var.dimensions) != ("depth", dlat, dlon):
            raise ValueError(f"{path}:{varname} has dimensions {var.dimensions}; getcoeffld reads a "
                             f"(depth, {dlat}, {dlon}) array (gen_ic3d.F90:374-392)")
        raw = np.asarray(var[:])
        attrs = var.ncattrs()
        fill = var.getncattr("_FillValue") if "_FillValue" in attrs else \
            netCDF4.default_fillvals.get(var.dtype.str[1:])
    finally:
        nc.close()
    Nlon_r, Nlat, Nd = lon_real.size, nc_lat.size, nc_depth.size
    if raw.shape != (Nd, Nlat, Nlon_r):
        raise ValueError(f"{path}:{varname} shape {raw.shape} != (depth, lat, lon) = {(Nd, Nlat, Nlon_r)}")
    if Nd > 1 and nc_depth[1] < 0.0:                     # gen_ic3d.F90:233
        nc_depth = -nc_depth
    nc_lon = np.empty(Nlon_r + 2)
    nc_lon[1:-1] = lon_real
    nc_lon[0] = lon_real[-1] - 360.0                     # :221-222 + ic_cyclic :252-255
    nc_lon[-1] = lon_real[0] + 360.0

    data = np.transpose(raw.astype(np.float64), (2, 1, 0))     # ncdata(lon, lat, depth)
    ncdata = np.empty((Nlon_r + 2, Nlat, Nd))
    ncdata[1:-1] = data
    ncdata[0] = data[-1]                                  # :402-403
    ncdata[-1] = data[0]
    bad = np.isnan(ncdata) | (ncdata < -0.99 * DUMMY) | (ncdata > DUMMY)
    if fill is not None:
        is_fill = ncdata == np.float64(fill)
        if raw.dtype != np.float64 and np.any(is_fill & ~bad):
            # nf_inq_var_fill writes the fill in the variable's own type into a real(8) (gen_ic3d.F90:368);
            # for a non-double variable that comparison is not verified here, and it matters only for a fill
            # value the dummy tests above do not catch.
            raise NotImplementedError(f"{path}:{varname}: {raw.dtype} fill value {fill!r} inside +-dummy")
        bad |= is_fill
    ncdata[bad] = DUMMY
    return nc_lon, nc_lat, nc_depth, ncdata


def _brackets(geo_deg_x, geo_deg_y, nc_lon, nc_lat):
    """``nc_ic3d_ini`` (``gen_ic3d.F90:306-325``): 0-based ``bilin_indx_i/j`` (-1 outside)."""
    N = geo_deg_x.size
    bi = np.full(N, -1, np.int64)
    bj = np.full(N, -1, np.int64)
    for n in range(N):
        x, y = geo_deg_x[n], geo_deg_y[n]
        if nc_lon[0] <= x <= nc_lon[-1]:
            bi[n] = _binarysearch_d(nc_lon, x)
        if nc_lat[0] <= y <= nc_lat[-1]:
            bj[n] = _binarysearch_d(nc_lat, y)
    # x == nc_lon(nc_Nlon) (or y == nc_lat(nc_Nlat)) makes Fortran read ip1/jp1 out of bounds
    if np.any(bi >= nc_lon.size - 1) or np.any(bj >= nc_lat.size - 1):
        raise ValueError("a node sits exactly on the last IC grid longitude/latitude: Fortran reads out of bounds")
    return bi, bj


def _getcoeffld(ncdata, nc_lon, nc_lat, nc_depth, x, y, bi, bj, z3, nlev):
    """The interpolation of ``getcoeffld`` (``gen_ic3d.F90:416-541``, non-cavity branch). ``(nod2D, nl-1)``,
    ``DUMMY`` where nothing was set."""
    N, nlm1 = z3.shape
    Nd = nc_depth.size
    out = np.full((N, nlm1), DUMMY)
    ok = (bi >= 0) & (bj >= 0)                              # min(i,j) > 0 (1-based)
    i = np.where(ok, bi, 0)
    j = np.where(ok, bj, 0)
    v00, v10 = ncdata[i, j, :], ncdata[i + 1, j, :]
    v01, v11 = ncdata[i, j + 1, :], ncdata[i + 1, j + 1, :]
    hi = 0.99 * DUMMY
    srf_bad = ((v00[:, 0] > hi) | (v10[:, 0] > hi) | (v01[:, 0] > hi) | (v11[:, 0] > hi)
               | ~np.isfinite(v00[:, 0]) | ~np.isfinite(v10[:, 0])
               | ~np.isfinite(v01[:, 0]) | ~np.isfinite(v11[:, 0]))       # :447-448 -> cycle
    ok &= ~srf_bad
    x1, x2 = nc_lon[i], nc_lon[i + 1]
    y1, y2 = nc_lat[j], nc_lat[j + 1]
    denom = ((x2 - x1) * (y2 - y1))[:, None]
    wx2, wx1 = (x2 - x)[:, None], (x - x1)[:, None]
    wy2, wy1 = (y2 - y)[:, None], (y - y1)[:, None]
    acc = (v00 * wx2) * wy2 + (v10 * wx1) * wy2 + (v01 * wx2) * wy1 + (v11 * wx1) * wy1
    # ``data1d(:) = (...) / denom`` as the reference binary computes it (-no-prec-div, see the module docstring):
    # times the reciprocal, except the last depth of an odd-length column, which is a true division
    data1d = acc * (1.0 / denom)
    if Nd % 2 == 1:
        data1d[:, -1] = acc[:, -1] / denom[:, 0]
    cdum = ((v00 > hi) | (v10 > hi) | (v01 > hi) | (v11 > hi)
            | ~np.isfinite(v00) | ~np.isfinite(v10) | ~np.isfinite(v01) | ~np.isfinite(v11))
    data1d = np.where(cdum, DUMMY, data1d)                  # :458-463

    wet = np.arange(nlm1)[None, :] < (nlev[:, None] - 1)    # k = ul1 .. nl1 (1-based)
    di = _binarysearch_vec(nc_depth, -z3)                   # (N, nl-1), 0-based = Fortran d_indx - 1
    rows = np.arange(N)[:, None]
    interior = (di >= 0) & (di < Nd - 1)
    dc = np.clip(di, 0, Nd - 2)
    d1 = data1d[rows, dc]
    d2 = data1d[rows, dc + 1]
    both = (d1 < hi) & (d2 < hi)
    delta_d = nc_depth[dc + 1] - nc_depth[dc]
    cf_a = (d2 - d1) / delta_d
    cf_b = d1 - cf_a * nc_depth[dc]
    val = -cf_a * z3 + cf_b
    sel = ok[:, None] & wet
    out = np.where(sel & interior & both, val, out)
    out = np.where(sel & (di == -1), data1d[:, :1], out)    # d_indx == 0: data1d(1)
    return out


def _neighbour_table(mesh):
    """Per node, the (neighbour node, element nlevels) pairs in ``extrap_nod3D``'s loop order: the node's
    elements in ``nod_in_elem2D`` order, each element's three vertices in ``elem2D_nodes`` order."""
    off = np.asarray(mesh.nod_in_elem2D_offsets, np.int64)
    flat = np.asarray(mesh.nod_in_elem2D, np.int64)
    en = np.asarray(mesh.elem_nodes, np.int64)
    elev = np.asarray(mesh.nlevels, np.int64)
    nb = en[flat].reshape(-1)                               # 3 per (node, element)
    nb_elev = np.repeat(elev[flat], 3)
    return off * 3, nb, nb_elev


def _extrap_nod3D(arr, nlev, table):
    """``extrap_nod3D`` (``gen_support.F90:387-507``) on one rank, in place on ``arr (nod2D, nl-1)``."""
    off, nb, nb_elev = table
    N, nlm1 = arr.shape
    hi = 0.99 * DUMMY
    if arr[:, 0].max() > hi:                                # do while (glob_max > 0.99 dummy)
        for nz in range(nlm1):                              # Fortran nz = nz0 + 1
            col = arr[:, nz]
            active = np.nonzero((col > hi) & (nlev > nz + 1))[0]
            if active.size == 0:
                continue
            # the candidates of each dummy node, in loop order: elements reaching this level
            # (skip if nz > nlevels(el)) and vertices that have it (nlevels_nod2D(v) > nz)
            keep = (nb_elev >= nz + 1) & (nlev[nb] > nz + 1)
            cand = [nb[off[n]:off[n + 1]][keep[off[n]:off[n + 1]]].tolist() for n in active]
            work = col.tolist()                             # work_array (Python floats = IEEE doubles)
            todo = list(zip(active.tolist(), cand))
            success = True
            while success:                                  # do while (success)
                success = False
                still = []
                for n, cs in todo:
                    val = 0.0
                    cnt = 0
                    for v in cs:
                        w = work[v]
                        if w < hi:
                            val += w
                            cnt += 1
                    if cnt > 0:
                        work[n] = val / float(cnt)          # in place: visible to the later nodes
                        success = True
                    else:
                        still.append((n, cs))
                todo = still
            arr[:, nz] = np.asarray(work)
        if arr[:, 0].max() > hi:
            raise RuntimeError("extrap_nod3D: a surface node has no path to valid IC data "
                               "(Fortran loops forever here)")
    for nz in range(1, nlm1):                               # vertical fill, do nz = 2, nl1
        fill = (arr[:, nz] > hi) & (nz < nlev - 1)
        arr[:, nz] = np.where(fill, arr[:, nz - 1], arr[:, nz])


def _cleanup(arr, nlev):
    """``do_ic3d`` cleanup (``gen_ic3d.F90:583-607``), in place."""
    arr[~np.isfinite(arr)] = 0.0
    arr[arr > 0.9 * DUMMY] = 0.0
    below = np.arange(arr.shape[1])[None, :] >= (nlev[:, None] - 1)
    arr[below] = 0.0


def _geo(mesh, geo_coord):
    g = np.asarray(mesh.geo_coord_nod2D if geo_coord is None else geo_coord, np.float64)
    if g.shape != (int(mesh.nod2D), 2):
        raise ValueError(f"geo_coord shape {g.shape} != (nod2D, 2)")
    return _canonical_lon(g[:, 0]), g[:, 1]


# --------------------------------------------------------------------------------------------------------------
# public
# --------------------------------------------------------------------------------------------------------------
def interp_ic_field(mesh, path: PathLike, varname: str, *, geo_coord=None, z3=None,
                    _table=None) -> np.ndarray:
    """One ``do_ic3d`` entry: the file field on the mesh nodes after ``getcoeffld``, ``extrap_nod`` and the
    cleanup, ``(nod2D, nl-1)``. ``geo_coord`` (nod2D, 2) geographic radians overrides
    ``mesh.geo_coord_nod2D``; ``z3`` overrides :func:`ic_layer_depths`."""
    lon, lat = _geo(mesh, geo_coord)
    nlev = np.asarray(mesh.nlevels_nod2D, np.int64)
    if z3 is None:
        z3 = ic_layer_depths(mesh)
    x = lon * INV_RAD                                       # geo_coord/rad (-no-prec-div: x * (1/rad))
    x = np.where(x < 0.0, x + 360.0, x)
    x = np.where(x > 360.0, x - 360.0, x)
    y = lat * INV_RAD
    nc_lon, nc_lat, nc_depth, ncdata = _read_ic_file(path, varname)
    bi, bj = _brackets(x, y, nc_lon, nc_lat)
    arr = _getcoeffld(ncdata, nc_lon, nc_lat, nc_depth, x, y, bi, bj, z3, nlev)
    arr[~np.isfinite(arr)] = 0.0                            # gen_ic3d.F90:569-571
    _extrap_nod3D(arr, nlev, _neighbour_table(mesh) if _table is None else _table)
    _cleanup(arr, nlev)
    return arr


def resolve_ic_files(ic_files: Mapping[int, Tuple[PathLike, str]],
                     ic_dir: Optional[PathLike] = None) -> dict:
    """``filename = trim(ClimateDataPath)//trim(filelist(n))`` (``gen_ic3d.F90:561``): relative names are
    joined to ``ic_dir``. Checks that every ID is a BGC tracer of this configuration."""
    out = {}
    for tid, (f, var) in ic_files.items():
        tid = int(tid)
        if tid not in T.SLOT:
            raise KeyError(f"IC file for tracer {tid}, which is not a BGC tracer {T.BGC_IDS} "
                           "(do_ic3d aborts: 'idlist contains tracer which is not listed in tracer_id')")
        p = Path(f)
        if not p.is_absolute():
            if ic_dir is None:
                raise ValueError(f"relative IC file {f!r} needs ic_dir (FESOM's ClimateDataPath)")
            p = Path(ic_dir) / p
        out[tid] = (p, str(var))
    return out


def bgc_initial_state(mesh, ic_files: Mapping[int, Tuple[PathLike, str]], switches: P.RecomSwitches,
                      params: P.RecomParams, *, ic_dir: Optional[PathLike] = None,
                      geo_coord=None) -> BgcIC:
    """The cold-start REcoM state of FESOM2 + REcoM (module docstring, steps 1-6).

    ``ic_files`` maps BGC tracer ID -> ``(file, variable)`` (``namelist.tra`` ``&tracer_init3d`` without T and
    S; :data:`PI_IC_FILES`, :data:`CORE2_IC_FILES`), relative files under ``ic_dir``. ``switches`` must be the
    ported configuration (:func:`fesom_jax.recom.params.check_supported`); ``params`` gives the constant initial
    values. ``geo_coord`` overrides the mesh's geographic coordinates (see the module docstring)."""
    P.check_supported(switches)
    files = resolve_ic_files(ic_files, ic_dir)
    N, nl = int(mesh.nod2D), int(mesh.nl)
    nlev = np.asarray(mesh.nlevels_nod2D, np.int64)
    z3 = ic_layer_depths(mesh)
    table = _neighbour_table(mesh)

    # do_ic3d (+ tracer_init's zeros, + oce_initial_state's zeros for the BGC IDs without a file)
    vals = np.zeros((N, nl - 1, T.NTR))
    for tid in T.BGC_IDS:                                   # the tracer loop order of do_ic3d's search
        if tid in files:
            path, var = files[tid]
            vals[:, :, T.slot(tid)] = interp_ic_field(mesh, path, var, geo_coord=geo_coord, z3=z3,
                                                      _table=table)
    alk_surf = vals[:, 0, T.ialk].copy() if switches.restore_alkalinity else np.zeros(N)
    bgc_old = vals.copy()                                   # ocean_setup: valuesold = values

    # recom_init
    bgc = vals
    for tid in T.BGC_IDS:
        s = T.slot(tid)
        if tid == 1019:
            bgc[:, :, s] = bgc[:, :, s] * FE_FACTOR
        elif tid > 1003 and tid not in (1018, 1022, 1037):
            bgc[:, :, s] = T.init_value(tid, params)
    lon, lat = _geo(mesh, geo_coord)
    box = (lat > -12.5 * RAD) & (lat < 9.5 * RAD) & (lon > -106.0 * RAD) & (lon < -72.0 * RAD)
    wet = np.arange(nl - 1)[None, :] < (nlev[:, None] - 1)
    deep = ~(np.abs(np.asarray(mesh.Z, np.float64))[None, :] < VENT_DEPTH)   # recom_init's Z_3d_n == Z
    cap = box[:, None] & wet & deep
    fe = bgc[:, :, T.ife]
    fe = np.where(cap, np.minimum(VENT_CAP, fe), fe)
    bgc[:, :, T.ife] = np.maximum(T.TINY, fe)

    return BgcIC(bgc=bgc, bgc_old=bgc_old, alk_surf=alk_surf, benthos=np.zeros((N, BENTHOS_NUM)),
                 hplus=np.full(N, HPLUS0))


def with_bgc_ic(state, ic: BgcIC, *, xp=None):
    """``state`` with ``bgc``/``bgc_old`` from ``ic``, padded from FESOM's ``nl-1`` layers to the State's
    ``nl`` levels with a zero bottom row. ``alk_surf``, ``benthos`` and ``hplus`` are not State fields on this
    branch; the caller keeps them from ``ic``. ``xp`` (default ``jax.numpy``) as in
    :func:`fesom_jax.phc_ic.phc_initial_state`."""
    if xp is None:
        import jax.numpy as xp
    nl = state.T.shape[1]
    if ic.bgc.shape[:2] != (state.T.shape[0], nl - 1):
        raise ValueError(f"IC shape {ic.bgc.shape} does not fit a State with T {state.T.shape}")
    pad = lambda a: xp.asarray(np.concatenate([a, np.zeros((a.shape[0], 1, a.shape[2]))], axis=1))
    return dataclasses.replace(state, bgc=pad(ic.bgc), bgc_old=pad(ic.bgc_old))
