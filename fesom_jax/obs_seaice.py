"""Sea-ice observations as super-observations on the model nodes (numpy + scipy, host only).

The ice-parameter calibration compares monthly means of the model's ``a_ice`` and ``m_ice``
with two satellite products on the 25 km EASE2 grid (432 x 432 Lambert azimuthal equal-area
cells per hemisphere):

* OSI-SAF sea-ice concentration CDR v3.0 (osi-450-a), daily, both hemispheres;
* ESA-CCI CryoSat-2 sea-ice thickness v2.0, monthly, Arctic, October-April.

The observations are moved to the model, not the model to the grid: every observed EASE2 cell
goes to its nearest wet surface node (great-circle distance, KD-tree on unit vectors), if that
node lies within a cap (30 km north, 50 km south). A node's super-observation is the mean of its
cells, and its cost weight is the OBSERVED area it carries, ``n_cells * 625 km^2 / A_corr`` with
``A_corr = (100 km)^2``. With roughly one cell per node in the Arctic many nodes receive no cell;
node-area weights would then cover only part of the observed area.

Flag handling (decided for the calibration plan; never a generic "flag != 0 -> drop"):

* OSI-SAF ``status_flag`` is a BIT MASK. Dropped: ``land`` (1), ``lake`` (2),
  ``land_spill_over`` (8), ``high_t2m`` (16), ``spatial_interp`` (32, the interpolated pole hole)
  and the ``ice_conc`` fill value. Kept: ``open_water_filtered`` (4) and ``max_ice_climo`` (128),
  which mark VALID zero-concentration cells (the open-water side of every ice edge; every such
  cell of 2013 is exactly 0), and ``temporal_interp`` (64). ``high_t2m`` is the product's caution
  flag ("this might be false ice"), not a zero: 14 % of the NH high_t2m cell-days of 2013 are above
  0 (up to 100 %); it is dropped, zeros included (decided 2026-10-10). A cell value combining a
  kept and a dropped bit is dropped.
* CCI ``status_flag`` is an ENUMERATION 0-5; only 0 (nominal retrieval) is used. Flag 2 ("outside
  the concentration mask") is NOT a zero-thickness observation: the concentration term carries the
  ice edge.

Units: OSI-SAF stores ``ice_conc`` and ``total_standard_uncertainty`` as integers with
``scale_factor`` 0.01 in percent, so fraction = raw * 1e-4. CCI stores floats with NaN as the
missing value (no ``_FillValue``); its ``sea_ice_concentration`` is in percent.

Monthly means: a cell's OSI-SAF monthly mean exists when at least half of the month's AVAILABLE
daily files are valid for it (before mid-1987 the record has a file every other day only; a month
without files has no mean). The cell's monthly sigma is sqrt(mean of the daily sigma^2) over the
valid days that carry an uncertainty (some kept zero-concentration and temporally interpolated
cells have none; they add nothing to the sigma, and the floor below dominates anyway).

Super-observation sigma^2 = mean of the cell sigma^2 + floor^2, with floors 0.1 (concentration)
and 0.3 m (effective thickness). The effective thickness compared with ``m_ice`` is
``sea_ice_thickness * sea_ice_concentration / 100`` (sigma likewise); negative CCI thicknesses
(retrieval noise over thin ice, kept by the product for unbiased averaging) are kept.

The arrays the cost reads are ``[12, nod2D]`` (calendar months of one year): ``C_obs, sC, nC,
H_obs, sH, nH`` with NaN (``n = 0``) where there is no observation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .config import R_EARTH

# --- constants of the observation operator ------------------------------------------------
CELL_AREA_M2 = 625.0e6          # one 25 km x 25 km EASE2 cell
A_CORR_M2 = 1.0e10              # (100 km)^2: the weight unit of the cost
CAP_KM = {"nh": 30.0, "sh": 50.0}   # nearest-node distance cap per hemisphere
FLOOR_C = 0.1                   # concentration sigma floor (fraction)
FLOOR_H = 0.3                   # effective-thickness sigma floor [m]
CCI_MONTHS = (1, 2, 3, 4, 10, 11, 12)   # calendar months with CryoSat-2 winter retrievals

# OSI-SAF CDR v3.0 status_flag bits (flag_masks / flag_meanings of the files)
OSI_BITS = {
    "land": 1, "lake": 2, "open_water_filtered": 4, "land_spill_over": 8,
    "high_t2m": 16, "spatial_interp": 32, "temporal_interp": 64, "max_ice_climo": 128,
}
OSI_DROP_BITS = ("land", "lake", "land_spill_over", "high_t2m", "spatial_interp")
OSI_KEEP_BITS = ("open_water_filtered", "temporal_interp", "max_ice_climo")
OSI_DROP_MASK = sum(OSI_BITS[b] for b in OSI_DROP_BITS)
OSI_FILL = -32767               # ice_conc / total_standard_uncertainty _FillValue
OSI_FLAG_FILL = -32768          # status_flag _FillValue
OSI_RAW_TO_FRACTION = 1.0e-4    # raw int * scale_factor 0.01 -> percent; / 100 -> fraction

CCI_FLAG_NOMINAL = 0
EXTENT_THRESHOLD = 0.15         # ice edge: concentration > 15 %


def obs_weight(n_cells):
    """Cost weight of a super-observation carrying ``n_cells`` EASE2 cells: its observed area in
    units of ``A_corr`` (``n * 625 km^2 / (100 km)^2 = n / 16``). Zero where ``n`` is 0."""
    return np.asarray(n_cells, dtype=np.float64) * (CELL_AREA_M2 / A_CORR_M2)


# --- model nodes ---------------------------------------------------------------------------
def mesh_surface_nodes(mesh_dir: str | Path):
    """Geographic lon/lat [deg] and the wet-surface mask of a C-exported mesh directory
    (``geo_coord_nod2D.npy`` in radians, ``ulevels_nod2D.npy``, ``nlevels_nod2D.npy``).

    A node's surface layer (0-based level 0) exists when ``ulevels - 1 <= 0 < nlevels - 1``,
    i.e. ``ulevels == 1`` and ``nlevels >= 2`` (fesom_jax.mesh conventions). Read with numpy
    only, in mesh node order, which is the node order of the Fortran output too."""
    d = Path(mesh_dir)
    geo = np.load(d / "geo_coord_nod2D.npy")
    ulev = np.load(d / "ulevels_nod2D.npy")
    nlev = np.load(d / "nlevels_nod2D.npy")
    lon = np.degrees(np.asarray(geo[:, 0], dtype=np.float64))
    lat = np.degrees(np.asarray(geo[:, 1], dtype=np.float64))
    wet = (np.asarray(ulev) == 1) & (np.asarray(nlev) >= 2)
    return lon, lat, wet


def northern_nodes(node_lat):
    """Node hemisphere as ``ice.py`` splits it: lat > 0 is north."""
    return np.asarray(node_lat) > 0.0


def unit_vectors(lon_deg, lat_deg):
    """[n, 3] unit vectors of geographic points given in degrees."""
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64)).ravel()
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64)).ravel()
    c = np.cos(lat)
    return np.stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)], axis=1)


def assign_cells(cell_lon, cell_lat, node_lon, node_lat, node_ok, cap_km,
                 radius_m: float = R_EARTH):
    """Nearest eligible node of every cell centre by great-circle distance.

    ``node_ok`` selects the eligible nodes (wet surface nodes of the cell grid's hemisphere).
    Returns ``(node, dist_km)``, both of the cells' flattened shape: ``node`` is the global node
    index, or -1 where the nearest eligible node is farther than ``cap_km``. Distances use the
    model's Earth radius (``config.R_EARTH``)."""
    node_ok = np.asarray(node_ok, dtype=bool)
    idx = np.flatnonzero(node_ok)
    if idx.size == 0:
        raise ValueError("assign_cells: no eligible node")
    tree = cKDTree(unit_vectors(np.asarray(node_lon)[idx], np.asarray(node_lat)[idx]))
    chord, j = tree.query(unit_vectors(cell_lon, cell_lat), k=1)
    dist_km = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)) * radius_m / 1000.0
    node = np.where(dist_km <= cap_km, idx[j], -1).astype(np.int64)
    return node, dist_km


# --- OSI-SAF daily fields --------------------------------------------------------------------
def osisaf_valid(conc_raw, flag_raw):
    """Cells of one daily OSI-SAF field that are observations (bit-mask decisions above)."""
    conc_raw = np.asarray(conc_raw)
    flag_raw = np.asarray(flag_raw).astype(np.int64)
    return ((conc_raw != OSI_FILL) & (flag_raw != OSI_FLAG_FILL)
            & ((flag_raw & OSI_DROP_MASK) == 0))


def osisaf_fraction(raw):
    """OSI-SAF raw integer (percent / scale_factor 0.01) -> fraction; fill -> NaN."""
    raw = np.asarray(raw)
    return np.where(raw == OSI_FILL, np.nan, raw.astype(np.float64) * OSI_RAW_TO_FRACTION)


def osisaf_flag_counts(conc_raw, flag_raw) -> dict:
    """Cell counts of one daily field: every bit (cells carrying it, alone or combined),
    ``nominal`` (flag 0), ``conc_fill``, ``flag_fill``, ``kept`` and ``dropped``."""
    conc_raw = np.asarray(conc_raw)
    fl = np.asarray(flag_raw).astype(np.int64)
    present = fl != OSI_FLAG_FILL
    out = {name: int(np.count_nonzero(present & ((fl & bit) != 0)))
           for name, bit in OSI_BITS.items()}
    out["nominal"] = int(np.count_nonzero(fl == 0))
    out["conc_fill"] = int(np.count_nonzero(conc_raw == OSI_FILL))
    out["flag_fill"] = int(np.count_nonzero(~present))
    kept = osisaf_valid(conc_raw, fl)
    out["kept"] = int(np.count_nonzero(kept))
    out["dropped"] = int(kept.size - out["kept"])
    return out


class MonthlyCellMean:
    """Running monthly mean of daily cell fields (one hemisphere, one month).

    ``add(value, valid, sigma)`` per available daily file (``value``/``sigma`` as fractions,
    ``sigma`` NaN where the file has no uncertainty); ``finalize(n_available)`` returns
    ``(mean, sigma, n_days)`` with NaN where fewer than half of the available days are valid."""

    def __init__(self, n_cells: int):
        self.sum = np.zeros(n_cells)
        self.sum_s2 = np.zeros(n_cells)
        self.n = np.zeros(n_cells, dtype=np.int64)
        self.n_s = np.zeros(n_cells, dtype=np.int64)
        self.n_files = 0

    def add(self, value, valid, sigma):
        value = np.asarray(value, dtype=np.float64).ravel()
        valid = np.asarray(valid, dtype=bool).ravel()
        sigma = np.asarray(sigma, dtype=np.float64).ravel()
        self.n_files += 1
        self.sum += np.where(valid, value, 0.0)
        self.n += valid
        has_s = valid & np.isfinite(sigma)
        self.sum_s2 += np.where(has_s, sigma, 0.0) ** 2
        self.n_s += has_s

    def finalize(self, n_available: int | None = None):
        n_av = self.n_files if n_available is None else int(n_available)
        ok = (self.n > 0) & (2 * self.n >= n_av) if n_av > 0 else np.zeros_like(self.n, bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(ok, self.sum / np.maximum(self.n, 1), np.nan)
            s2 = np.where(self.n_s > 0, self.sum_s2 / np.maximum(self.n_s, 1), 0.0)
        sigma = np.where(ok, np.sqrt(s2), np.nan)
        return mean, sigma, np.where(ok, self.n, 0)


# --- CCI monthly thickness -------------------------------------------------------------------
def cci_effective_thickness(sit, sit_unc, sic_pct, flag):
    """Effective thickness ``SIT * SIC / 100`` and its sigma ``SIT_unc * SIC / 100`` of one
    monthly CCI field; valid only where ``status_flag == 0`` and all three inputs are finite.
    Returns ``(H, sH, valid)`` with NaN where invalid."""
    sit = np.asarray(sit, dtype=np.float64)
    unc = np.asarray(sit_unc, dtype=np.float64)
    sic = np.asarray(sic_pct, dtype=np.float64)
    valid = ((np.asarray(flag).astype(np.int64) == CCI_FLAG_NOMINAL)
             & np.isfinite(sit) & np.isfinite(unc) & np.isfinite(sic))
    frac = np.where(valid, sic, 0.0) / 100.0
    H = np.where(valid, np.where(valid, sit, 0.0) * frac, np.nan)
    sH = np.where(valid, np.where(valid, unc, 0.0) * frac, np.nan)
    return H, sH, valid


# --- super-observations ----------------------------------------------------------------------
def super_obs(cell_node, cell_value, cell_sigma, cell_valid, nod2D: int, floor: float):
    """Per-node super-observations of one month from valid cells assigned to a node.

    Returns ``(obs, sigma, n)`` of shape [nod2D]: the mean of the node's cell values, ``sigma =
    sqrt(mean cell sigma^2 + floor^2)``, the cell count; NaN / 0 where the node got no cell."""
    node = np.asarray(cell_node).ravel()
    val = np.asarray(cell_value, dtype=np.float64).ravel()
    sig = np.asarray(cell_sigma, dtype=np.float64).ravel()
    use = np.asarray(cell_valid, dtype=bool).ravel() & (node >= 0)
    nd = node[use]
    n = np.bincount(nd, minlength=nod2D)
    s = np.bincount(nd, weights=val[use], minlength=nod2D)
    s2 = np.bincount(nd, weights=sig[use] ** 2, minlength=nod2D)
    have = n > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = np.where(have, s / np.maximum(n, 1), np.nan)
        sigma = np.where(have, np.sqrt(s2 / np.maximum(n, 1) + floor ** 2), np.nan)
    return obs, sigma, n.astype(np.int32)


def count_above(cell_node, cell_value, cell_valid, nod2D: int, thresh: float = EXTENT_THRESHOLD):
    """Per node, the number of its valid cells whose value exceeds ``thresh`` (``nE``: the
    cell-level extent a super-observation carries, ``nE * 625 km^2``)."""
    node = np.asarray(cell_node).ravel()
    val = np.asarray(cell_value, dtype=np.float64).ravel()
    use = np.asarray(cell_valid, dtype=bool).ravel() & (node >= 0)
    with np.errstate(invalid="ignore"):
        icy = use & (np.where(use, val, 0.0) > thresh)
    return np.bincount(node[icy], minlength=nod2D).astype(np.int32)


OBS_KEYS = ("C_obs", "sC", "nC", "H_obs", "sH", "nH")


def empty_obs(nod2D: int) -> dict:
    """``[12, nod2D]`` observation arrays with no observation anywhere."""
    nan = np.full((12, nod2D), np.nan)
    zero = np.zeros((12, nod2D), dtype=np.int32)
    return {"C_obs": nan.copy(), "sC": nan.copy(), "nC": zero.copy(), "nE": zero.copy(),
            "H_obs": nan.copy(), "sH": nan.copy(), "nH": zero.copy()}


def merge_into(target: np.ndarray, month: int, values: np.ndarray, have: np.ndarray):
    """Write ``values[have]`` into ``target[month]``, refusing to overwrite an existing value
    (the two hemispheres' cells go to disjoint node sets)."""
    clash = have & (np.isfinite(target[month]) if target.dtype.kind == "f"
                    else target[month] != 0)
    if clash.any():
        raise ValueError(f"super-observations of two grids land on {int(clash.sum())} common "
                         f"nodes in month {month + 1}")
    target[month][have] = values[have]


def check_obs(obs: Mapping[str, np.ndarray], nod2D: int | None = None):
    """Shape and consistency checks of an observation dict (``[12, nod2D]``, NaN exactly where
    the count is 0, sigma > 0 where valid)."""
    shp = np.shape(obs["C_obs"])
    if len(shp) != 2 or shp[0] != 12 or (nod2D is not None and shp[1] != nod2D):
        raise ValueError(f"observation arrays must be [12, nod2D], got {shp}")
    for k in OBS_KEYS:
        if np.shape(obs[k]) != shp:
            raise ValueError(f"{k} has shape {np.shape(obs[k])}, expected {shp}")
    for v, s, n in (("C_obs", "sC", "nC"), ("H_obs", "sH", "nH")):
        have = np.asarray(obs[n]) > 0
        if not np.array_equal(np.isfinite(obs[v]), have) or not np.array_equal(
                np.isfinite(obs[s]), have):
            raise ValueError(f"{v}/{s} must be finite exactly where {n} > 0")
        if np.any(np.asarray(obs[s])[have] <= 0):
            raise ValueError(f"{s} must be positive where {n} > 0")
    if "nE" in obs:                       # optional (files written before 2026-10-10 lack it)
        nE = np.asarray(obs["nE"])
        if nE.shape != shp or np.any(nE < 0) or np.any(nE > np.asarray(obs["nC"])):
            raise ValueError("nE must be [12, nod2D] with 0 <= nE <= nC")
