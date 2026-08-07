"""NN-GM Task 4 — mesh diagnostics: the w-based meridional overturning streamfunction.

Analysis-only (numpy, no JAX, no library surface — the long-window plan's precedent):
these read model output and produce numbers/figures, they are never in the model's
gradient path.

The overturning follows pyfesom2's ``xmoc_data`` (Sidorenko et al. 2020, GMD 13, 3337)
exactly, so results are comparable with everything AWI publishes from FESOM:

  1. mask the nodal vertical velocity w by a basin mask,
  2. average w over each triangle's three nodes and weight by the triangle area,
  3. blank levels below each triangle's own bottom,
  4. bin triangles by the latitude of their centroid,
  5. cumulatively sum from the north and negate.

Basin masks come from **the same GeoJSON polygons pyfesom2 ships**, so the region
definition is identical by construction rather than by reimplementation; point-in-polygon
uses ``matplotlib.path`` (shapely is not in the fesom-jax environment).

Works on any source that can hand over w on nodes plus the mesh arrays: the Fortran
teacher's netCDF, the JAX student, or the Fortran baseline.

Usage as a script (writes a figure and a .npz):
  python scripts/nn_eddy/mesh_diag.py --w-file w.fesom.1958.nc --mesh data/mesh_core2 \\
      --basin Atlantic_MOC --out amoc_1958
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GEOJSON_DIR = Path("/home/a/a270088/PYTHON/pyfesom2/pyfesom2/geojson")
MOC_BASINS = ("Atlantic_MOC", "IndoPacific_MOC", "Pacific_MOC", "Indian_MOC")


# ---------------------------------------------------------------------------
# basin masks

def basin_mask(lon_deg, lat_deg, basin: str, geojson_dir: Path = GEOJSON_DIR) -> np.ndarray:
    """Boolean node mask for a named basin, from pyfesom2's own GeoJSON polygons.

    ``lon_deg``/``lat_deg``: (nod2D,) GEOGRAPHIC degrees, lon in [-180, 180].
    ``basin``: "Global" (everything), or any feature name in MOCBasins.geojson /
    oceanBasins.geojson.
    """
    if basin.lower() in ("global", "global ocean", "all"):
        return np.ones(lon_deg.shape, dtype=bool)
    from matplotlib.path import Path as MplPath

    fname = "MOCBasins.geojson" if basin in MOC_BASINS else "oceanBasins.geojson"
    feats = json.loads((Path(geojson_dir) / fname).read_text())["features"]
    for feat in feats:
        if feat["properties"]["name"] == basin:
            geom = feat["geometry"]
            rings = ([geom["coordinates"]] if geom["type"] == "Polygon"
                     else geom["coordinates"])            # MultiPolygon
            pts = np.column_stack([lon_deg, lat_deg])
            mask = np.zeros(lon_deg.shape, dtype=bool)
            for poly in rings:
                outer = np.asarray(poly[0])
                inside = MplPath(outer).contains_points(pts)
                for hole in poly[1:]:                      # subtract interior rings
                    inside &= ~MplPath(np.asarray(hole)).contains_points(pts)
                mask |= inside
            return mask
    raise ValueError(f"basin {basin!r} not found in {fname}; "
                     f"available: {[f['properties']['name'] for f in feats]}")


# ---------------------------------------------------------------------------
# overturning

def moc_from_w(w, elem_nodes, elem_area, elem_nlevels, elem_lat_deg, *,
               mask=None, nlats: int = 91):
    """Meridional overturning streamfunction in Sv from nodal vertical velocity.

    Parameters
    ----------
    w : (nod2D, nl) vertical velocity at layer interfaces [m/s].
    elem_nodes : (elem2D, 3) node indices of each triangle (0-based).
    elem_area : (elem2D,) triangle area [m²].
    elem_nlevels : (elem2D,) number of valid levels of each triangle (its bottom).
    elem_lat_deg : (elem2D,) GEOGRAPHIC latitude of the triangle centroid [degrees].
    mask : (nod2D,) optional node mask (bool or 0/1); None = global.
    nlats : number of latitude bins spanning -90..90 (pyfesom2 default 91 ⇒ 2° bins).

    Returns
    -------
    lats : (nlats,) bin latitudes [degrees].
    moc : (nlats, nl) overturning streamfunction [Sv]; positive = clockwise in a
        standard latitude-depth plot, matching pyfesom2 / Sidorenko et al. (2020).
    """
    w = np.asarray(w, dtype=np.float64)
    nl = w.shape[1]
    if mask is not None:
        w = w * np.asarray(mask, dtype=np.float64)[:, None]

    lats = np.linspace(-90.0, 90.0, nlats)
    dlat = lats[1] - lats[0]
    # ⚠️ Round the centroid before binning. On the CORE2 mesh two triangles have centroid
    # latitudes of exactly 86.0 and 74.0 degrees — right on a bin edge — and our .npy mesh
    # export differs from the mesh ASCII in the last bits (~1e-13 deg), which is enough to
    # flip which bin they land in and to move ~7e-3 Sv of overturning by one bin. Rounding
    # to 1e-9 deg (~0.1 mm) makes the assignment depend on the coordinate as written, not
    # on accumulated round-off, so any copy of the mesh gives the same answer.
    pos = ((np.round(np.asarray(elem_lat_deg, dtype=np.float64), 9) - lats[0]) / dlat).astype(int)
    pos = np.clip(pos, 0, nlats - 1)

    # triangle-mean w, converted to Sv, weighted by triangle area
    elem_mean = w[elem_nodes, :].sum(axis=1) / 3.0 * 1.0e-6      # (elem2D, nl)
    weighted = elem_mean * np.asarray(elem_area, dtype=np.float64)[:, None]

    # blank levels at or below each triangle's bottom
    lev = np.arange(nl)[None, :]
    weighted = np.where(lev >= np.asarray(elem_nlevels)[:, None], np.nan, weighted)

    binned = np.zeros((nl, nlats))
    for k in range(pos.min(), pos.max() + 1):
        sel = pos == k
        if sel.any():
            binned[:, k] = np.nansum(weighted[sel, :], axis=0)

    # accumulate from the north, then flip the sign (pyfesom2 xmoc convention)
    moc = np.cumsum(binned[:, ::-1], axis=1)[:, ::-1].T * -1.0
    return lats, moc


def elem_centroid_lat(elem_nodes, lat_deg) -> np.ndarray:
    """Geographic latitude of each triangle's centroid [degrees]."""
    return np.asarray(lat_deg)[np.asarray(elem_nodes)].mean(axis=1)


def amoc_strength(lats, moc, zbar, *, lat_range=(20.0, 60.0), depth_range=(500.0, 3000.0)):
    """Peak overturning in a latitude/depth box — the usual "AMOC strength" scalar [Sv].

    Defaults are the RAPID-like North Atlantic box (20-60°N, 500-3000 m depth).
    ``zbar`` is the interface-depth array [m], negative downward as FESOM stores it.
    """
    depth = np.abs(np.asarray(zbar))
    jl = (lats >= lat_range[0]) & (lats <= lat_range[1])
    jd = (depth >= depth_range[0]) & (depth <= depth_range[1])
    box = moc[np.ix_(jl, jd)]
    return float(np.nanmax(box)) if box.size and np.isfinite(box).any() else np.nan


# ---------------------------------------------------------------------------
# section volume transport (Drake / ACC and friends)
#
# DECISION (plan Task 4, taken 2026-08-08): a direct line integral of the normal velocity, NOT
# a port of pyfesom2's broken-line machinery.
#
# pyfesom2's `transport.py` spends ~1000 lines building a "broken line": walking an arbitrary
# multi-segment user path, finding every triangle edge it crosses, computing the intersection
# geometry and orienting each crossing. That machinery exists to serve arbitrary interactively
# drawn paths and to plot the section. What this project needs is one number per section per
# snapshot for a handful of FIXED sections, compared between runs and over time.
#
# For that, the honest discretization is the definition itself:
#
#     T = ∫∫ u·n  dl dz
#
# Chop the great circle from A to B into sub-segments much shorter than an element, find the
# element each sub-segment's midpoint falls in, and sum `u_elem·n × ds × dz`. First-order
# accurate, converges as the sub-segment length shrinks (tested), and needs no edge topology at
# all. FESOM already stores u, v on ELEMENTS, so there is no collocation step and therefore none
# of the non-conservative interpolation that `coarsen.elem_to_node` carries.
#
# Three things this must get right, and all three are the same lesson as the remap:
#
# * **The section is a GREAT CIRCLE, interpolated on the unit sphere**, not a straight line in
#   lon/lat. Drake Passage is at 68°W so it happens not to care, but a zonal section at high
#   latitude or anything crossing the dateline does, and a lon/lat interpolation there is wrong
#   in a way that produces a plausible number.
# * **The normal is defined explicitly**: n = r × t, i.e. the LEFT of the direction of travel
#   A→B. For a section running north→south that is EASTWARD, so Drake Passage entered as
#   (South America → Antarctica) gives the ACC a positive transport. Asserted in the tests
#   rather than left to the reader.
# * **The bottom is masked per element** from `nlevels-1`, the same subtraction `load_mesh_arrays`
#   documents. Without it the deepest layer contributes either fill values or water that is not
#   there.

def _unit_xyz(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def _xyz_to_lonlat(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])), np.degrees(np.arcsin(v[..., 2]))


def great_circle_points(lon0, lat0, lon1, lat1, n):
    """``n`` points along the great circle from A to B inclusive, by spherical interpolation.

    Straight interpolation of the Cartesian endpoints and renormalising ("nlerp") lands on the
    same great circle as the exact slerp — only the spacing differs — and for a section that is
    then subdivided finely, uneven spacing is handled by using each sub-segment's own length.
    """
    a, b = _unit_xyz(lon0, lat0), _unit_xyz(lon1, lat1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    v = (1.0 - t) * a[None, :] + t * b[None, :]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


EARTH_R = 6371000.0


def element_centroids_lonlat(elem_nodes, lon_deg, lat_deg):
    """Element centroids as (lon, lat) in degrees, averaged **in 3-D on the unit sphere**.

    ⚠️ Averaging longitudes directly puts the centroid of a triangle straddling the dateline in
    the middle of the Pacific on the wrong side. The 3-D mean has no seam.
    """
    v = _unit_xyz(lon_deg, lat_deg)[np.asarray(elem_nodes)]         # (elem, 3, 3)
    return _xyz_to_lonlat(v.mean(axis=1))


def section_transport(u_elem, v_elem, *, lon0, lat0, lon1, lat1, elem_nodes, lon_deg, lat_deg,
                      elem_nlevels, zbar, n_sub=2000, return_profile=False,
                      centroid_xyz=None):
    """Volume transport across the great-circle section A→B, in Sv.

    ``u_elem``, ``v_elem``: (elem2D, nz1) eastward/northward velocity on elements [m/s].
    ``elem_nlevels``: the ALREADY-SUBTRACTED level count (see :func:`load_mesh_arrays`).
    ``n_sub``: number of sub-segments. Increase until the answer stops moving; on CORE2 (~1°)
    a Drake-Passage section is ~900 km, so 2000 sub-segments is ~450 m each, far below the
    element size, and the result is converged.

    Positive = flow to the LEFT of A→B (see the module comment: north→south ⇒ eastward).

    Returns the total in Sv, or ``(total, per_level)`` if ``return_profile``.
    """
    u_elem = np.asarray(u_elem, dtype=np.float64)
    v_elem = np.asarray(v_elem, dtype=np.float64)
    zbar = np.asarray(zbar, dtype=np.float64)
    dz = -np.diff(zbar)                                            # (nz1,) positive thickness
    nz1 = dz.size
    if u_elem.shape[1] != nz1:
        raise ValueError(f"u_elem has {u_elem.shape[1]} layers, zbar implies {nz1}")

    pts = great_circle_points(lon0, lat0, lon1, lat1, n_sub + 1)    # (n_sub+1, 3) unit vectors
    mid = pts[:-1] + pts[1:]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    seg = pts[1:] - pts[:-1]
    # chord vs arc: identical to O(ds^2/R^2), and ds/R ~ 7e-5 here
    ds = np.linalg.norm(seg, axis=1) * EARTH_R                     # (n_sub,) metres

    # local east/north at each midpoint, and the section tangent expressed in them
    lon_m, lat_m = _xyz_to_lonlat(mid)
    lam, phi = np.radians(lon_m), np.radians(lat_m)
    east = np.stack([-np.sin(lam), np.cos(lam), np.zeros_like(lam)], axis=1)
    north = np.stack([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
                     axis=1)
    t_e = np.einsum("ij,ij->i", seg, east)
    t_n = np.einsum("ij,ij->i", seg, north)
    tn = np.hypot(t_e, t_n)
    tn[tn == 0.0] = 1.0
    t_e, t_n = t_e / tn, t_n / tn
    # n = r x t in the local frame: (up) x (t_e*east + t_n*north) = t_e*north - t_n*east,
    # i.e. rotate the tangent 90 degrees anticlockwise = the LEFT of the direction of travel.
    n_e, n_n = -t_n, t_e

    # which element each midpoint sits in: nearest centroid on the unit sphere. Exact
    # point-in-triangle would be the pedantic version; with sub-segments two orders of magnitude
    # shorter than an element, nearest-centroid differs only within half an element of a
    # boundary and the convergence test shows it does not matter.
    from scipy.spatial import cKDTree

    if centroid_xyz is None:
        clon, clat = element_centroids_lonlat(elem_nodes, lon_deg, lat_deg)
        centroid_xyz = _unit_xyz(clon, clat)
    _, idx = cKDTree(centroid_xyz).query(mid, k=1, workers=-1)

    valid = np.arange(nz1)[None, :] < np.asarray(elem_nlevels)[idx][:, None]
    un = (u_elem[idx] * n_e[:, None] + v_elem[idx] * n_n[:, None]) * valid
    per_level = (un * ds[:, None]).sum(axis=0) * dz / 1.0e6        # Sv per layer
    total = float(per_level.sum())
    return (total, per_level) if return_profile else total


# Fixed sections. Each is entered so that the CLIMATOLOGICALLY POSITIVE direction comes out
# positive under the "left of A→B" convention, which for a north→south section is eastward.
SECTIONS = {
    # Drake Passage at 68W, Tierra del Fuego to the Antarctic Peninsula. Observed ~134-174 Sv
    # (Donohue et al. 2016 puts the total at 173.3 +/- 10.7 Sv; older estimates ~134).
    "Drake": dict(lon0=-68.0, lat0=-54.5, lon1=-68.0, lat1=-63.5),
    # ACC south of Africa, 20E, Cape Agulhas to Antarctica.
    "ACC_20E": dict(lon0=20.0, lat0=-34.0, lon1=20.0, lat1=-69.5),
    # ACC south of Tasmania, 147E.
    "ACC_147E": dict(lon0=147.0, lat0=-43.5, lon1=147.0, lat1=-67.0),
}


# ---------------------------------------------------------------------------

def load_mesh_arrays(mesh_dir: Path):
    """The handful of exported mesh arrays the overturning needs.

    ⚠️ ``elem_nlevels`` is returned as the exported ``nlevels`` MINUS ONE — the number of
    interface levels that carry data. FESOM's mesh diagnostic stores the interface count,
    and its deepest entry is where the netCDF fill value starts; pyfesom2's ``xmoc_data``
    makes the same subtraction. Blanking at the unsubtracted count leaves one row of
    9.97e36 fill values in the sum and the answer comes out at 1e43 Sv.
    """
    mesh_dir = Path(mesh_dir)
    geo = np.load(mesh_dir / "geo_coord_nod2D.npy")           # GEOGRAPHIC radians
    lon = np.degrees(geo[:, 0])
    lon = (lon + 180.0) % 360.0 - 180.0                        # -> [-180, 180)
    lat = np.degrees(geo[:, 1])
    return dict(lon_deg=lon, lat_deg=lat,
                elem_nodes=np.load(mesh_dir / "elem_nodes.npy"),
                elem_area=np.load(mesh_dir / "elem_area.npy"),
                elem_nlevels=np.load(mesh_dir / "nlevels.npy") - 1,
                zbar=np.load(mesh_dir / "zbar.npy"))


def load_w(path, var: str = "w", time_index: int | None = None) -> np.ndarray:
    """Read w as (nod2D, nl) with missing data zeroed.

    ⚠️ FESOM writes ``9.96921e+36`` (the netCDF float fill) wherever a cell is dry **or**
    its value rounds below 1e-30, so the raw array is not usable arithmetic:
    ``np.asarray(ds.variables['w'][:])`` strips netCDF4's mask and silently hands back
    those 1e36 values. Missing means "contributes nothing", so they are set to zero here.
    """
    from netCDF4 import Dataset

    ds = Dataset(str(path))
    v = ds.variables[var][:]                       # netCDF4 returns a MaskedArray
    w = np.ma.filled(np.ma.masked_invalid(np.ma.masked_greater(np.ma.asarray(v), 1e30)), 0.0)
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 3:
        w = w[time_index] if time_index is not None else w.mean(axis=0)
    return w


def load_elem_field(path, var: str, n_elem: int, time_index: int | None = None) -> np.ndarray:
    """Read an ELEMENTAL field as (elem2D, nz1), fills zeroed and orientation fixed.

    Same fill trap as :func:`load_w`: FESOM writes 9.96921e+36 where a cell is dry, and
    ``np.asarray(masked_array)`` hands those straight back as numbers.
    """
    from netCDF4 import Dataset

    with Dataset(str(path)) as ds:
        v = ds.variables[var][:]
    a = np.ma.filled(np.ma.masked_invalid(np.ma.masked_greater(np.ma.asarray(v), 1e30)), 0.0)
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3:
        a = a[time_index] if time_index is not None else a.mean(axis=0)
    if a.shape[0] != n_elem:                       # (nz1, elem) -> (elem, nz1)
        a = a.T
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-file", help="netCDF with variable w (time, nod2D, nl) or (nod2D, nl)")
    ap.add_argument("--u-file", help="netCDF with ELEMENTAL u; with --v-file switches to "
                                    "section-transport mode instead of the overturning")
    ap.add_argument("--v-file")
    ap.add_argument("--sections", default="Drake,ACC_20E,ACC_147E",
                    help=f"comma-separated names from {sorted(SECTIONS)}")
    ap.add_argument("--n-sub", type=int, default=2000)
    ap.add_argument("--w-var", default="w")
    ap.add_argument("--mesh", required=True, help="exported mesh dir (data/mesh_core2)")
    ap.add_argument("--basin", default="Atlantic_MOC")
    ap.add_argument("--nlats", type=int, default=91)
    ap.add_argument("--time", type=int, default=None, help="time index; default = mean over all records")
    ap.add_argument("--out", default="moc")
    args = ap.parse_args()

    m = load_mesh_arrays(args.mesh)

    if args.u_file and args.v_file:
        n_elem = m["elem_nodes"].shape[0]
        u = load_elem_field(args.u_file, "u", n_elem, args.time)
        v = load_elem_field(args.v_file, "v", n_elem, args.time)
        clon, clat = element_centroids_lonlat(m["elem_nodes"], m["lon_deg"], m["lat_deg"])
        cxyz = _unit_xyz(clon, clat)
        out = {}
        for name in args.sections.split(","):
            name = name.strip()
            if name not in SECTIONS:
                raise SystemExit(f"unknown section {name!r}; have {sorted(SECTIONS)}")
            total, prof = section_transport(
                u, v, **SECTIONS[name], elem_nodes=m["elem_nodes"], lon_deg=m["lon_deg"],
                lat_deg=m["lat_deg"], elem_nlevels=m["elem_nlevels"], zbar=m["zbar"],
                n_sub=args.n_sub, return_profile=True, centroid_xyz=cxyz)
            print(f"{name:10s} {total:9.2f} Sv   (top 500 m: "
                  f"{prof[m['zbar'][1:] > -500.0].sum():7.2f} Sv)")
            out[name] = total
            out[name + "_profile"] = prof
        np.savez(f"{args.out}_sections.npz", zbar=m["zbar"], **out)
        print(f"wrote {args.out}_sections.npz")
        return 0

    if not args.w_file:
        raise SystemExit("need either --w-file (overturning) or --u-file + --v-file (sections)")
    w = load_w(args.w_file, args.w_var, args.time)
    if w.shape[0] != m["lon_deg"].size:      # (nl, nod2D) -> (nod2D, nl)
        w = w.T

    mask = basin_mask(m["lon_deg"], m["lat_deg"], args.basin)
    lats, moc = moc_from_w(w, m["elem_nodes"], m["elem_area"], m["elem_nlevels"],
                           elem_centroid_lat(m["elem_nodes"], m["lat_deg"]),
                           mask=mask, nlats=args.nlats)
    strength = amoc_strength(lats, moc, m["zbar"])
    print(f"{args.basin}: {mask.sum()} of {mask.size} nodes; "
          f"peak overturning in the 20-60N / 500-3000 m box = {strength:.2f} Sv")
    np.savez(f"{args.out}.npz", lats=lats, moc=moc, zbar=m["zbar"], basin=args.basin,
             strength=strength)
    print(f"wrote {args.out}.npz")


if __name__ == "__main__":
    raise SystemExit(main())
