"""JRA55-do v1.4.0 daily-forcing reader (Task 5.3) — a faithful **numpy** port of
``fesom_jra55.c`` (``nc_read_time_grid`` + ``build_bilin_indices`` + ``getcoeffld`` +
``fesom_jra55_step``) for the single-rank, no-cavity CORE2 case.

Like :mod:`fesom_jax.phc_ic`, this is **host-side, non-differentiable setup**: it reads
the atmosphere off disk and bilinearly interpolates it to the mesh nodes, producing the
8 time-interpolated physics-unit fields the bulk formulae (Task 5.4) consume. The
*differentiable* SST→flux / current→stress feedback lives in the bulk, **not** here —
this reader's output is a per-step device constant. We mirror the C bit-for-bit
(verified against the C ``jra_dump_*`` dumps), so the JAX CORE2 run sees the exact same
forcing as the C reference.

Pipeline (``fesom_jra55.c``):
  1. ``nc_read_time_grid`` (``:123``): per field, read ``lon``/``lat``/``time`` + the
     ``calendar`` attribute; pad ``lon`` by +2 cyclic-halo columns; transform the time
     axis to "julian days since year 0001 of *yearnew*" (julday rebase + per-field
     **mid-interval shift**, ``nm_nc_tmid=0``); flip ``lat`` if stored north→south.
  2. ``build_bilin_indices`` (``:295``): per node, a 1-based lon/lat bracket via the
     literal ``binarysearch`` on **geographic** node coords (deg). Built **once** — the
     8 fields share one (Nlon×Nlat) grid — into a 4-corner gather stencil ``(idx4,w4)``.
  3. ``getcoeffld`` (``:386``): for a date ``rdate``, locate the two bracketing time
     slices, bilinear-interp each to the nodes (``d1``/``d2``), form the linear-in-time
     ``coef_a``/``coef_b``. Cached per field (refresh only when ``rdate`` leaves the
     bracket — ``fesom_jra55_step:651``).
  4. ``step`` (``:638``): ``field = rdate·coef_a + coef_b``; **rotate the wind**
     geographic→model-rotated (``fesom_vector_g2r``, Euler 50/15/−90); unit conversions
     (Tair K→°C, prec /1000 → m/s).

⚠️ Field order is **uas,vas,huss,rsds,rlds,tas,prra,prsn** (``fesom_jra55.h:50`` — ``tas``
is 6th, not 3rd). ⚠️ The interpolation grid is **geographic**, but the wind is rotated
into the **model** frame afterwards. ⚠️ Each field gets its **own** mid-shifted time axis
(instantaneous fields sampled on the 3-h marks, flux fields on the half-marks → different
``nc_time`` after the shift), so the per-field ``getcoeffld`` is not shared.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import netCDF4

from .config import (RAD, ALPHA_EULER_DEG, BETA_EULER_DEG, GAMMA_EULER_DEG,
                     FORCE_ROTATION)

DEFAULT_JRA_DIR = "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0"

# Field order — fesom_jra55.h:50-59 (FESOM_JRA_XWIND=0 … FESOM_JRA_SNOW=7).
JRA_VARS = ("uas", "vas", "huss", "rsds", "rlds", "tas", "prra", "prsn")
N_FLD = len(JRA_VARS)
(I_XWIND, I_YWIND, I_HUMI, I_QSR, I_QLW, I_TAIR, I_PREC, I_SNOW) = range(N_FLD)

# Namelist scalars (fesom_jra55_init, fesom_jra55.c:578-583).
NM_NC_IYEAR = 1900
NM_NC_IMM = 1
NM_NC_IDD = 1
NM_NC_FREQ = 1
NM_NC_TMID = 0
_GREG_CALS = ("julian", "gregorian", "proleptic_gregorian", "standard")


# --------------------------------------------------------------------------
# Scalar helpers (literal ports of the C)
# --------------------------------------------------------------------------
def _julday(yyyy: int, mm: int, dd: int, calendar: str) -> int:
    """``fesom_jra_julday`` (``fesom_jra55.c:27``). Julian day at noon for
    (yyyy,mm,dd); ``int()`` truncates toward zero exactly like C ``(int)``."""
    if calendar in _GREG_CALS:
        IGREG = 15 + 31 * (10 + 12 * 1582)        # Oct 15, 1582 → 588829
        jy = yyyy
        if jy == 0:
            raise ValueError("julday: there is no year zero")
        if jy < 0:
            jy += 1
        if mm > 2:
            jm = mm + 1
        else:
            jy -= 1
            jm = mm + 13
        jul = int(365.25 * jy) + int(30.6001 * jm) + dd + 1720995
        if dd + 31 * (mm + 12 * yyyy) >= IGREG:
            ja = int(0.01 * jy)
            jul = jul + 2 - ja + int(0.25 * ja)
        return jul
    return 365 * yyyy


def _binarysearch(arr: np.ndarray, value: float) -> int:
    """``fesom_jra_binarysearch`` (``fesom_jra55.c:70``). **1-based** index of the
    largest element ≤ value (1e-9 exact-match short-circuit, ``nint`` midpoint).
    Returns 0 if value < arr[0]; ``len`` if value > arr[-1]."""
    d = 1e-9
    left, right = 1, len(arr)
    while left <= right:
        middle = int(math.floor((left + right) / 2.0 + 0.5))   # Fortran nint()
        am = arr[middle - 1]
        if abs(am - value) <= d:
            return middle
        if am > value:
            right = middle - 1
        else:
            left = middle + 1
    return right


def _calendar_year_orig(time0: float, calendar: str) -> int:
    """Year of ``time0`` (julian days since year 0001) — the ``calendar_date``
    inverse used to rebase the time axis (``fesom_jra55.c:226-255``)."""
    if calendar in _GREG_CALS:
        julian = int(time0)
        IGREG = 2299161
        if julian >= IGREG:
            x = ((julian - 1867216) - 0.25) / 36524.25
            ja = julian + 1 + int(x) - int(0.25 * x)
        else:
            ja = julian
        jb = ja + 1524
        jc = int(6680 + ((jb - 2439870) - 122.1) / 365.25)
        jd = int(365 * jc + 0.25 * jc)
        je = int((jb - jd) / 30.6001)
        mm_o = je - 1
        if mm_o > 12:
            mm_o -= 12
        yyyy = jc - 4715
        if mm_o > 2:
            yyyy -= 1
        if yyyy <= 0:
            yyyy -= 1
        return yyyy
    return int((time0 + 1.0e-12) / 365.0)


def _rotation_matrix() -> np.ndarray:
    """``build_rotation_matrix`` (``fesom_mesh.c:105``): row-major flat 3×3 mapping
    geographic Cartesian → rotated Cartesian, from Euler (α,β,γ)=(50,15,−90)°.
    Uses the truncated-π ``RAD`` (config.py) — load-bearing for fidelity."""
    al = ALPHA_EULER_DEG * RAD
    be = BETA_EULER_DEG * RAD
    ga = GAMMA_EULER_DEG * RAD
    ca, sa = math.cos(al), math.sin(al)
    cb, sb = math.cos(be), math.sin(be)
    cg, sg = math.cos(ga), math.sin(ga)
    return np.array([
        cg * ca - sg * cb * sa,
        cg * sa + sg * cb * ca,
        sg * sb,
        -sg * ca - cg * cb * sa,
        -sg * sa + cg * cb * ca,
        cg * sb,
        sb * sa,
        -sb * ca,
        cb,
    ], dtype=np.float64)


def _vector_g2r(u, v, glon, glat, rlon, rlat, M):
    """``fesom_vector_g2r`` (``fesom_mesh.c:169``), vectorized over nodes. Rotate a
    geographic (east,north) vector into the model-rotated frame; magnitude-preserving.
    Returns ``(u_rot, v_rot)``."""
    sgl, cgl = np.sin(glat), np.cos(glat)
    sgo, cgo = np.sin(glon), np.cos(glon)
    txg = -v * sgl * cgo - u * sgo
    tyg = -v * sgl * sgo + u * cgo
    tzg = v * cgl
    txr = M[0] * txg + M[1] * tyg + M[2] * tzg
    tyr = M[3] * txg + M[4] * tyg + M[5] * tzg
    tzr = M[6] * txg + M[7] * tyg + M[8] * tzg
    srl, crl = np.sin(rlat), np.cos(rlat)
    sro, cro = np.sin(rlon), np.cos(rlon)
    v_rot = -srl * cro * txr - srl * sro * tyr + crl * tzr
    u_rot = -sro * txr + cro * tyr
    return u_rot, v_rot


# --------------------------------------------------------------------------
# Output container
# --------------------------------------------------------------------------
@dataclasses.dataclass
class JRAFields:
    """The 8 time-interpolated physics-unit fields, each ``[nod2D]`` numpy float64.
    ``u_wind``/``v_wind`` are post-g2r-rotation (model frame); ``Tair`` is °C;
    ``prec_rain``/``prec_snow`` are m/s."""
    u_wind: np.ndarray
    v_wind: np.ndarray
    shum: np.ndarray
    shortwave: np.ndarray
    longwave: np.ndarray
    Tair: np.ndarray
    prec_rain: np.ndarray
    prec_snow: np.ndarray

    def as_array(self) -> np.ndarray:
        """``[nod2D, 8]`` in the JRA field order (uas,vas,huss,rsds,rlds,tas,prra,prsn)."""
        return np.stack([self.u_wind, self.v_wind, self.shum, self.shortwave,
                         self.longwave, self.Tair, self.prec_rain, self.prec_snow],
                        axis=-1)


# --------------------------------------------------------------------------
# Per-field NetCDF state
# --------------------------------------------------------------------------
class _Field:
    """Per-field grid + time axis + open Dataset (mirrors ``fesom_jra55_field``)."""

    def __init__(self, var: str, path: str, yearnew: int):
        self.var = var
        self.path = path
        ds = netCDF4.Dataset(path)
        # Match the C nc_get_vara_* reads exactly: no auto mask/scale (JRA has none).
        for v in ds.variables.values():
            v.set_auto_maskandscale(False)
        self.ds = ds
        self.ncvar = ds.variables[var]

        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        time = np.asarray(ds.variables["time"][:], dtype=np.float64)
        cal = "none"
        if hasattr(ds.variables["time"], "calendar"):
            cal = str(ds.variables["time"].calendar).lower()
        self.calendar = cal

        Nlon_real = lon.shape[0]
        self.Nlat = lat.shape[0]
        self.Nlon = Nlon_real + 2                 # +2 cyclic-halo columns (C :154)
        self.Ntime = time.shape[0]
        self.Nlon_real = Nlon_real

        # nc_lon[1..Nlon-2] = file lon; halo at [0],[Nlon-1] (C :184-188).
        nc_lon = np.empty(self.Nlon, dtype=np.float64)
        nc_lon[1:self.Nlon - 1] = lon
        nc_lon[0] = nc_lon[self.Nlon - 2]
        nc_lon[self.Nlon - 1] = nc_lon[1]
        nc_lat = lat.copy()

        # --- time-axis transform (C :219-268) ---
        jd_origin = _julday(NM_NC_IYEAR, NM_NC_IMM, NM_NC_IDD, cal)
        nc_time = time / float(NM_NC_FREQ) + float(jd_origin)
        year_orig = _calendar_year_orig(nc_time[0], cal)
        jd_year_orig = _julday(year_orig, 1, 1, cal)
        jd_yearnew = _julday(yearnew, 1, 1, cal)
        nc_time = nc_time - float(jd_year_orig) + float(jd_yearnew)
        # mid-interval shift (nm_nc_tmid != 1). The in-place C loop reads the
        # *original* nc_time[i+1] (modified only on the next iteration) → a plain
        # vectorized half-sum reproduces it; the last point uses the *already
        # shifted* nc_time[-2], which the in-place assignment below also does.
        if NM_NC_TMID != 1 and self.Ntime > 1:
            nc_time[:-1] = 0.5 * (nc_time[1:] + nc_time[:-1])
            nc_time[-1] = nc_time[-1] + (nc_time[-1] - nc_time[-2]) * 0.5

        # flip_lat: file stores north→south → flip so nc_lat ascends (C :270-279).
        self.flip_lat = bool(self.Nlat > 1 and nc_lat[0] > nc_lat[-1])
        if self.flip_lat:
            nc_lat = nc_lat[::-1].copy()

        # ic_cyclic: extend the halo by ±360 (C :282-283).
        nc_lon[0] = nc_lon[0] - 360.0
        nc_lon[self.Nlon - 1] = nc_lon[self.Nlon - 1] + 360.0

        self.nc_lon = nc_lon
        self.nc_lat = nc_lat
        self.nc_time = nc_time

        # getcoeffld cache (per-field linear-in-time coefficients + bracket).
        self.t_indx = -1
        self.t_indx_p1 = -1
        self.coef_a = None
        self.coef_b = None

    def read_slice(self, t_indx_1based: int) -> np.ndarray:
        """``read_one_time_slice`` (``fesom_jra55.c:336``): the (Nlat×Nlon) slice at
        the 1-based time index, with optional lat-flip + cyclic halo. Returns a
        flat ``[Nlat*Nlon]`` float64 array (row-major, matching ``idx4``)."""
        raw = np.asarray(self.ncvar[t_indx_1based - 1, :, :], dtype=np.float64)
        if self.flip_lat:
            raw = raw[::-1, :]
        sd = np.empty((self.Nlat, self.Nlon), dtype=np.float64)
        sd[:, 1:self.Nlon - 1] = raw
        sd[:, 0] = sd[:, self.Nlon - 2]            # cyclic halo (C :374-376)
        sd[:, self.Nlon - 1] = sd[:, 1]
        return sd.reshape(-1)

    def close(self):
        if self.ds is not None:
            self.ds.close()
            self.ds = None


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------
class JRA55Reader:
    """Host-numpy JRA55-do reader. Build once per (mesh, year), then call
    :meth:`step` per model date. Output is a :class:`JRAFields` of ``[nod2D]`` numpy
    arrays (convert to ``jnp`` at the device boundary in the step driver)."""

    def __init__(self, mesh, year: int, jra_dir: str | Path = DEFAULT_JRA_DIR):
        self.year = int(year)
        self.N = int(mesh.nod2D)
        self.jra_dir = Path(jra_dir)

        # geographic node coords (deg), with only the <0 wrap (C :306-308).
        geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64) / RAD
        self.x = np.where(geo[:, 0] < 0.0, geo[:, 0] + 360.0, geo[:, 0])
        self.y = geo[:, 1].copy()
        # rotated node coords (rad), for the wind g2r rotation.
        self.glon = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64)[:, 0]
        self.glat = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64)[:, 1]
        self.rlon = np.asarray(mesh.coord_nod2D, dtype=np.float64)[:, 0]
        self.rlat = np.asarray(mesh.coord_nod2D, dtype=np.float64)[:, 1]
        self.M = _rotation_matrix()

        # Open the 8 per-year fields; build the gather stencil ONCE from their shared (lon,lat)
        # grid. That grid is FIXED across years, so a multi-year run rolls the year via
        # `reopen_year` (swap the file handles) and REUSES idx4/weights — never rebuilt.
        self.fields = self._open_fields(self.year)
        f0 = self.fields[0]
        self.idx4, self.dx4, self.dy4, self.denom = self._build_stencil(
            f0.nc_lon, f0.nc_lat, f0.Nlon, f0.Nlat)
        self._grid_lon, self._grid_lat = f0.nc_lon, f0.nc_lat   # reopen_year invariance guard

    def _open_fields(self, year: int) -> "list[_Field]":
        """Open the 8 JRA fields for ``year`` (``{var}.{year:04d}.nc``) and verify they share one
        (lon,lat) grid — the precondition for the single shared bilinear stencil."""
        fields = [_Field(var, str(self.jra_dir / f"{var}.{int(year):04d}.nc"), int(year))
                  for var in JRA_VARS]
        f0 = fields[0]
        for f in fields[1:]:
            if not (np.array_equal(f.nc_lon, f0.nc_lon)
                    and np.array_equal(f.nc_lat, f0.nc_lat)):
                raise ValueError("JRA55 fields do not share a common lon/lat grid; "
                                 "the shared-stencil assumption is violated.")
        return fields

    def reopen_year(self, year: int):
        """Switch the forcing year IN PLACE — swap ONLY the 8 per-year file handles + time axes.

        The bilinear stencil (``idx4``/``dx4``/``dy4``/``denom``) and the wind-rotation factors
        depend only on (mesh, JRA grid), which is IDENTICAL every year, so they are KEPT — not
        rebuilt. This is the multi-year-run fix: a full reader rebuild would re-pay the per-node
        stencil setup (~seconds–tens-of-seconds at NG5 scale) every January 1 and discard the
        per-device interpolation knowledge we paid to build. Cheap here: open 8 files + read their
        1-D time axes (the field data is still read lazily per step). Returns ``self``."""
        year = int(year)
        if year == self.year:
            return self
        new_fields = self._open_fields(year)
        g0 = new_fields[0]
        if not (np.array_equal(g0.nc_lon, self._grid_lon)
                and np.array_equal(g0.nc_lat, self._grid_lat)):
            raise ValueError(f"JRA55 year {year}'s grid differs from the build year's — the "
                             "shared interpolation stencil cannot be reused.")
        for f in self.fields:                  # release the old year's open Datasets
            f.close()
        self.fields = new_fields
        self.year = year
        return self

    # ---- stencil -------------------------------------------------------
    def _build_stencil(self, nc_lon, nc_lat, Nlon, Nlat):
        """``build_bilin_indices`` (``:295``) + the ``getcoeffld`` corner/weight branch
        logic (``:457-513``), precomputed per node into 4-corner flat indices
        ``idx4[N,4]`` plus per-corner ``(dx4,dy4)`` factors and a ``denom[N]`` such that
        the gather reproduces the C **bit-for-bit**:

            d[n] = ( (sA·dxA)·dyA + (sB·dxB)·dyB + (sC·dxC)·dyC + (sD·dxD)·dyD ) / denom

        i.e. each corner term is ``(s·dx)·dy`` in the C's evaluation order and the
        ``/denom`` is applied to the *sum* (not folded into the weights — folding it
        reassociates ~1e-13, which the C's large-Julian-day time-interp formula then
        amplifies to ~1e-8). Corner order A,B,C,D = (j0,i0),(j0,ip1),(jp1,i0),(jp1,ip1).
        Unused corners (extrp≠0) carry ``dx=dy=0`` (term 0, added exactly); the
        single-factor extrp 1/2/3 branches set ``dy=1`` (×1 is exact)."""
        N = self.N
        x, y = self.x, self.y
        lon_lo, lon_hi = nc_lon[0], nc_lon[Nlon - 1]
        lat_lo, lat_hi = nc_lat[0], nc_lat[Nlat - 1]

        # --- 1-based brackets via the literal binarysearch (one-time node loop) ---
        bi = np.empty(N, dtype=np.int64)
        bj = np.empty(N, dtype=np.int64)
        for n in range(N):
            xn = x[n]
            if xn < lon_hi and xn >= lon_lo:
                bi[n] = _binarysearch(nc_lon, xn)
            elif xn < lon_lo:
                bi[n] = -1
            else:
                bi[n] = 0
            yn = y[n]
            if yn < lat_hi and yn >= lat_lo:
                bj[n] = _binarysearch(nc_lat, yn)
            elif yn < lat_lo:
                bj[n] = -1
            else:
                bj[n] = 0

        # --- extrp adjustment (C :467-474), vectorized off the ORIGINAL bi/bj ---
        i = bi.copy()
        ip1 = bi + 1
        j = bj.copy()
        jp1 = bj + 1
        extrp = np.zeros(N, dtype=np.int64)

        m = (bi == 0)
        i = np.where(m, Nlon, i); ip1 = np.where(m, i, ip1); extrp = np.where(m, extrp + 1, extrp)
        m = (bi == -1)
        i = np.where(m, 1, i);    ip1 = np.where(m, i, ip1); extrp = np.where(m, extrp + 1, extrp)
        m = (bj == 0)
        j = np.where(m, Nlat, j); jp1 = np.where(m, j, jp1); extrp = np.where(m, extrp + 2, extrp)
        m = (bj == -1)
        j = np.where(m, 1, j);    jp1 = np.where(m, j, jp1); extrp = np.where(m, extrp + 2, extrp)

        i0 = i - 1; ip10 = ip1 - 1            # 0-based column indices
        j0 = j - 1; jp10 = jp1 - 1            # 0-based row indices

        x1 = nc_lon[i0]; x2 = nc_lon[ip10]
        y1 = nc_lat[j0]; y2 = nc_lat[jp10]
        dxa = x2 - x; dxb = x - x1            # (x2-x), (x-x1)
        dya = y2 - y; dyb = y - y1            # (y2-y), (y-y1)
        one = np.ones(N); zero = np.zeros(N)

        # default = extrp==0 (full bilinear): term_k = (s·dx_k)·dy_k.
        dxA, dyA = dxa, dya
        dxB, dyB = dxb, dya
        dxC, dyC = dxa, dyb
        dxD, dyD = dxb, dyb
        denom = (x2 - x1) * (y2 - y1)

        # extrp==1 (lon wrapped): d = (sA·(y2-y) + sD·(y-y1)) / (y2-y1).
        m = (extrp == 1)
        dxA = np.where(m, dya, dxA); dyA = np.where(m, one, dyA)
        dxB = np.where(m, zero, dxB); dyB = np.where(m, zero, dyB)
        dxC = np.where(m, zero, dxC); dyC = np.where(m, zero, dyC)
        dxD = np.where(m, dyb, dxD); dyD = np.where(m, one, dyD)
        denom = np.where(m, y2 - y1, denom)

        # extrp==2 (lat wrapped): d = (sA·(x2-x) + sD·(x-x1)) / (x2-x1).
        m = (extrp == 2)
        dxA = np.where(m, dxa, dxA); dyA = np.where(m, one, dyA)
        dxB = np.where(m, zero, dxB); dyB = np.where(m, zero, dyB)
        dxC = np.where(m, zero, dxC); dyC = np.where(m, zero, dyC)
        dxD = np.where(m, dxb, dxD); dyD = np.where(m, one, dyD)
        denom = np.where(m, x2 - x1, denom)

        # extrp==3 (both wrapped): d = sA.
        m = (extrp == 3)
        dxA = np.where(m, one, dxA); dyA = np.where(m, one, dyA)
        dxB = np.where(m, zero, dxB); dyB = np.where(m, zero, dyB)
        dxC = np.where(m, zero, dxC); dyC = np.where(m, zero, dyC)
        dxD = np.where(m, zero, dxD); dyD = np.where(m, zero, dyD)
        denom = np.where(m, one, denom)

        # flat (row-major) corner indices into a [Nlat*Nlon] slice.
        A = j0 * Nlon + i0
        B = j0 * Nlon + ip10
        C = jp10 * Nlon + i0
        D = jp10 * Nlon + ip10
        idx4 = np.stack([A, B, C, D], axis=-1).astype(np.int64)
        dx4 = np.stack([dxA, dxB, dxC, dxD], axis=-1).astype(np.float64)
        dy4 = np.stack([dyA, dyB, dyC, dyD], axis=-1).astype(np.float64)
        # The C reads these corners unguarded; CORE2 never lands a node within 1e-9
        # of the last lon-halo point, so the brackets stay in-bounds. Assert it.
        assert idx4.min() >= 0 and idx4.max() < Nlat * Nlon, \
            "JRA55 bilinear corner index out of bounds (a node hit the lon-halo edge)"
        return idx4, dx4, dy4, denom

    def _gather(self, slice_flat: np.ndarray) -> np.ndarray:
        """Bilinear interp of one flat time slice to all nodes, bit-for-bit with the C:
        ``( (sA·dxA)·dyA + (sB·dxB)·dyB + (sC·dxC)·dyC + (sD·dxD)·dyD ) / denom``,
        summing corners A→D left-to-right (``getcoeffld`` evaluation order)."""
        t = (slice_flat[self.idx4] * self.dx4) * self.dy4     # [N,4] = (s·dx)·dy
        num = ((t[:, 0] + t[:, 1]) + t[:, 2]) + t[:, 3]
        return num / self.denom

    # ---- per-field time coefficients (getcoeffld) ----------------------
    def _getcoeffld(self, f: _Field, rdate: float):
        """``getcoeffld`` (``fesom_jra55.c:386``): locate the bracketing slices around
        ``rdate``, bilinear-interp each, set ``f.coef_a``/``f.coef_b`` + the bracket."""
        Ntime = f.Ntime
        t_indx = _binarysearch(f.nc_time, rdate)        # 1-based
        if 0 < t_indx < Ntime:
            t_indx_p1 = t_indx + 1
            delta_t = f.nc_time[t_indx_p1 - 1] - f.nc_time[t_indx - 1]
        elif t_indx > 0:                                 # t_indx == Ntime: no future extrap
            t_indx = Ntime; t_indx_p1 = Ntime; delta_t = 1.0
        else:                                            # t_indx <= 0: no past extrap
            t_indx = 1; t_indx_p1 = 1; delta_t = 1.0

        d1 = self._gather(f.read_slice(t_indx))
        d2 = d1 if t_indx_p1 == t_indx else self._gather(f.read_slice(t_indx_p1))
        f.coef_a = (d2 - d1) / delta_t
        f.coef_b = d1 - f.coef_a * f.nc_time[t_indx - 1]
        f.t_indx = t_indx
        f.t_indx_p1 = t_indx_p1

    # ---- per-step update (fesom_jra55_step) ----------------------------
    def step(self, year: int, day: int, sec: float) -> JRAFields:
        """``fesom_jra55_step`` (``:638``): refresh per-field coefficients as needed,
        time-interpolate, distribute to physics units, rotate the wind. ``day`` is
        1-based day-of-year, ``sec`` is seconds-into-day."""
        cal0 = self.fields[0].calendar
        rdate = float(_julday(int(year), 1, 1, cal0)) + float(day - 1) + sec / 86400.0

        vals = []
        for f in self.fields:
            need = (f.t_indx <= 0)
            if not need:
                lo = f.nc_time[f.t_indx - 1]
                hi = f.nc_time[f.t_indx_p1 - 1]
                need = (rdate < lo or rdate > hi)
            if need:
                self._getcoeffld(f, rdate)
            vals.append(rdate * f.coef_a + f.coef_b)

        u = vals[I_XWIND]
        v = vals[I_YWIND]
        if FORCE_ROTATION:
            u, v = _vector_g2r(u, v, self.glon, self.glat, self.rlon, self.rlat, self.M)
        return JRAFields(
            u_wind=u,
            v_wind=v,
            shum=vals[I_HUMI],
            shortwave=vals[I_QSR],
            longwave=vals[I_QLW],
            Tair=vals[I_TAIR] - 273.15,
            prec_rain=vals[I_PREC] / 1000.0,
            prec_snow=vals[I_SNOW] / 1000.0,
        )

    def close(self):
        for f in self.fields:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
