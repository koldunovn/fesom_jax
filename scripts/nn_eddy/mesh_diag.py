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
    pos = ((np.asarray(elem_lat_deg) - lats[0]) / dlat).astype(int)
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

def load_mesh_arrays(mesh_dir: Path):
    """The handful of exported mesh arrays the overturning needs."""
    mesh_dir = Path(mesh_dir)
    geo = np.load(mesh_dir / "geo_coord_nod2D.npy")           # radians
    lon = np.degrees(geo[:, 0])
    lon = (lon + 180.0) % 360.0 - 180.0                        # -> [-180, 180)
    lat = np.degrees(geo[:, 1])
    return dict(lon_deg=lon, lat_deg=lat,
                elem_nodes=np.load(mesh_dir / "elem_nodes.npy"),
                elem_area=np.load(mesh_dir / "elem_area.npy"),
                elem_nlevels=np.load(mesh_dir / "nlevels.npy"),
                zbar=np.load(mesh_dir / "zbar.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-file", required=True, help="netCDF with variable w (time, nod2D, nl) or (nod2D, nl)")
    ap.add_argument("--w-var", default="w")
    ap.add_argument("--mesh", required=True, help="exported mesh dir (data/mesh_core2)")
    ap.add_argument("--basin", default="Atlantic_MOC")
    ap.add_argument("--nlats", type=int, default=91)
    ap.add_argument("--time", type=int, default=None, help="time index; default = mean over all records")
    ap.add_argument("--out", default="moc")
    args = ap.parse_args()

    from netCDF4 import Dataset
    m = load_mesh_arrays(args.mesh)
    ds = Dataset(args.w_file)
    w = np.asarray(ds.variables[args.w_var][:])
    if w.ndim == 3:
        w = w[args.time] if args.time is not None else w.mean(axis=0)
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
