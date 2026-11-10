#!/usr/bin/env python
"""Monthly sea-ice super-observations on the CORE2 nodes for the ice-parameter calibration.

    python scripts/icecal/prep_obs.py --years 2013 --out <dir> [--mesh-dir data/mesh_core2]

For every year writes ``<out>/obs_<YYYY>.npz`` (``C_obs, sC, nC, H_obs, sH, nH`` as ``[12,
nod2D]``, NaN / 0 where there is no observation; ``nE`` = the node's cells with a monthly mean
above 15 %, for the cell-level extent; plus ``nh`` = the northern-node mask and the node lon/lat) and ``<out>/obs_<YYYY>.json`` (provenance: inputs, decisions, per-month flag counts,
cell and node counts, the npz sha256). The cell -> node assignment is cached in
``<out>/assign.npz`` and reused only if the mesh, the grids and the caps are the same.

Inputs: OSI-SAF CDR v3.0 daily concentration (both hemispheres) and ESA-CCI CryoSat-2 v2.0
monthly thickness (north, October-April). The observation operator and the flag decisions are in
``fesom_jax/obs_seaice.py``.

Nothing is overwritten: every output is opened with mode ``"x"``, and the run refuses to start if
a year's output exists. Years after 2019 are refused (2019 is the last compared year).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fesom_jax import obs_seaice as O  # noqa: E402

OSI_ROOT = "/pool/data/ICDC/ice_and_snow/osisaf_iceconc/DATA"
CCI_ROOT = "/pool/data/ICDC/ice_and_snow/esa_cci_sicci/CRDP2_SIT/DATA/north"
OSI_PATTERN = "{y:04d}/{m:02d}/ice_conc_{hem}_ease2-250_cdr-v3p0_{y:04d}{m:02d}??1200.nc"
CCI_PATTERN = "{y:04d}/ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-{y:04d}{m:02d}-fv2.0.nc"
LAST_YEAR = 2019
HEMS = ("nh", "sh")
GRID_MATCH_DEG = 1e-4      # CCI vs OSI-SAF NH cell centres (measured 4e-6 deg in 2013)
# A month needs at least this many daily files (every-other-day months before mid-1987 have
# 10-16). It turns the near-empty months of the record into gaps: 1986-06 (4 files) and
# 1987-12 (2 files, the 1st and 2nd); 1986-04/05 have none.
MIN_FILES = 8


def parse_years(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:          # noqa: BLE001 — provenance must never kill a run
        return ""


# --- file access ---------------------------------------------------------------------------
def osisaf_files(root, year, month, hem) -> list[str]:
    return sorted(glob.glob(os.path.join(root, OSI_PATTERN.format(y=year, m=month, hem=hem))))


def cci_file(root, year, month) -> str | None:
    p = os.path.join(root, CCI_PATTERN.format(y=year, m=month))
    return p if os.path.exists(p) else None


def read_grid(path):
    """Cell-centre lat/lon [deg] (flattened, float64) and the projection axes of one file."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        lat = np.asarray(ds["lat"][:], dtype=np.float64)
        lon = np.asarray(ds["lon"][:], dtype=np.float64)
        xc = np.asarray(ds["xc"][:], dtype=np.float64)
        yc = np.asarray(ds["yc"][:], dtype=np.float64)
    return lat.ravel(), lon.ravel(), xc, yc, lat.shape


def _check_attr(var, name, want):
    got = getattr(var, name, None)
    ok = (np.allclose(np.asarray(got, dtype=np.float64), want) if isinstance(want, float)
          else (list(np.atleast_1d(got)) == list(want) if isinstance(want, (list, tuple))
                else got == want))
    if not ok:
        raise ValueError(f"{var.name}.{name} = {got!r}, expected {want!r}")


def read_osisaf_day(path, xc0, yc0):
    """Raw ``ice_conc``, ``total_standard_uncertainty`` and ``status_flag`` of one daily file
    (flattened), after checking the attributes the decoding relies on, the grid axes and the
    file's date against its name."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        for v in ("ice_conc", "total_standard_uncertainty"):
            _check_attr(ds[v], "scale_factor", 0.01)
            _check_attr(ds[v], "units", "%")
            _check_attr(ds[v], "_FillValue", O.OSI_FILL)
        fl = ds["status_flag"]
        _check_attr(fl, "flag_masks", list(O.OSI_BITS.values()))
        _check_attr(fl, "flag_meanings", " ".join(O.OSI_BITS))
        if not (np.array_equal(ds["xc"][:], xc0) and np.array_equal(ds["yc"][:], yc0)):
            raise ValueError(f"{path}: grid axes differ from the first file of the hemisphere")
        t = ds["time"]
        d = netCDF4.num2date(t[0], t.units, getattr(t, "calendar", "standard"))
        stamp = os.path.basename(path).rsplit("_", 1)[1][:8]
        if f"{d.year:04d}{d.month:02d}{d.day:02d}" != stamp:
            raise ValueError(f"{path}: time {d} does not match the file name")
        conc = np.asarray(ds["ice_conc"][0]).ravel()
        unc = np.asarray(ds["total_standard_uncertainty"][0]).ravel()
        flag = np.asarray(fl[0]).ravel()
    return conc, unc, flag


def read_cci_month(path):
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        fl = ds["status_flag"]
        _check_attr(fl, "flag_values", [0, 1, 2, 3, 4, 5])
        _check_attr(ds["sea_ice_concentration"], "units", "percent")
        _check_attr(ds["sea_ice_thickness"], "units", "m")
        _check_attr(ds["sea_ice_thickness_uncertainty"], "units", "m")
        out = {k: np.asarray(ds[k][0], dtype=np.float64).ravel()
               for k in ("sea_ice_thickness", "sea_ice_thickness_uncertainty",
                         "sea_ice_concentration")}
        out["status_flag"] = np.asarray(fl[0]).astype(np.int64).ravel()
        out["lat"] = np.asarray(ds["lat"][:], dtype=np.float64).ravel()
        out["lon"] = np.asarray(ds["lon"][:], dtype=np.float64).ravel()
    return out


# --- assignment ----------------------------------------------------------------------------
def get_assignment(path: Path, grids: dict, node_lon, node_lat, node_wet, caps) -> dict:
    """Cell -> node assignment per hemisphere, cached in ``path`` (created with mode "x" if it does
    not exist; an existing cache is reused only if its mesh/grid hashes and caps match)."""
    nh = O.northern_nodes(node_lat)
    meta = {"mesh_sha": _sha(node_lon, node_lat, node_wet), "R_earth_m": O.R_EARTH}
    for hem, (lat, lon) in grids.items():
        meta[f"{hem}_grid_sha"] = _sha(lat, lon)
        meta[f"{hem}_cap_km"] = float(caps[hem])
    if path.exists():
        with np.load(path) as z:
            for k, v in meta.items():
                got = z[k].item() if z[k].ndim == 0 else z[k]
                if got != v:
                    raise ValueError(f"{path}: cached {k} = {got!r} differs from {v!r}; "
                                     "use another --assign file")
            return {hem: (z[f"{hem}_node"], z[f"{hem}_dist_km"]) for hem in grids}
    res, arrays = {}, {}
    for hem, (lat, lon) in grids.items():
        ok = node_wet & (nh if hem == "nh" else ~nh)
        node, dist = O.assign_cells(lon, lat, node_lon, node_lat, ok, caps[hem])
        res[hem] = (node, dist)
        arrays[f"{hem}_node"] = node
        arrays[f"{hem}_dist_km"] = dist
    with open(path, "xb") as f:
        np.savez(f, **arrays, **{k: np.asarray(v) for k, v in meta.items()})
    return res


# --- one year ------------------------------------------------------------------------------
def process_year(year: int, *, osi_root, cci_root, node_lon, node_lat, node_wet, assign_path,
                 caps=None, min_files=MIN_FILES, log=print):
    caps = dict(O.CAP_KM if caps is None else caps)
    nod2D = node_lon.size
    nh = O.northern_nodes(node_lat)
    obs = O.empty_obs(nod2D)
    prov_months = []

    # grids: the first available file of each hemisphere in the year
    grids, axes = {}, {}
    for hem in HEMS:
        first = next((f for m in range(1, 13) for f in osisaf_files(osi_root, year, m, hem)), None)
        if first is None:
            continue
        lat, lon, xc, yc, _ = read_grid(first)
        grids[hem] = (lat, lon)
        axes[hem] = (xc, yc)
    if not grids:
        raise FileNotFoundError(f"no OSI-SAF files for {year} under {osi_root}")
    assign = get_assignment(Path(assign_path), grids, node_lon, node_lat, node_wet, caps)

    for month in range(1, 13):
        mp = {"month": month}
        for hem in grids:
            files = osisaf_files(osi_root, year, month, hem)
            n_found = len(files)
            if n_found < min_files:          # a near-empty month is a gap, not a monthly mean
                files = []
            node, _ = assign[hem]
            acc = O.MonthlyCellMean(node.size)
            counts: dict = {}
            no_sigma = 0
            for f in files:
                conc, unc, flag = read_osisaf_day(f, *axes[hem])
                valid = O.osisaf_valid(conc, flag)
                sig = O.osisaf_fraction(unc)
                no_sigma += int(np.count_nonzero(valid & ~np.isfinite(sig)))
                acc.add(O.osisaf_fraction(conc), valid, sig)
                for k, v in O.osisaf_flag_counts(conc, flag).items():
                    counts[k] = counts.get(k, 0) + v
            mean, sigma, ndays = acc.finalize(len(files))
            cell_ok = np.isfinite(mean)
            C, sC, nC = O.super_obs(node, mean, sigma, cell_ok, nod2D, O.FLOOR_C)
            have = nC > 0
            O.merge_into(obs["C_obs"], month - 1, C, have)
            O.merge_into(obs["sC"], month - 1, sC, have)
            O.merge_into(obs["nC"], month - 1, nC, have)
            O.merge_into(obs["nE"], month - 1, O.count_above(node, mean, cell_ok, nod2D), have)
            mp[f"C_{hem}"] = {
                "n_files": n_found, "skipped_fewer_than_min_files": 0 < n_found < min_files,
                "flag_cell_days": counts,
                "kept_cell_days_without_sigma": no_sigma,
                "cells_valid_month": int(cell_ok.sum()),
                "cells_valid_beyond_cap": int((cell_ok & (node < 0)).sum()),
                "cells_assigned": int(nC.sum()),
                "nodes_valid": int(have.sum()),
                "nodes_C_gt_0": int((have & (np.nan_to_num(C) > 0)).sum()),
                "nodes_C_ge_0.15": int((have & (np.nan_to_num(C) >= 0.15)).sum()),
                "sum_weight": float(O.obs_weight(nC).sum()),
            }
        f = cci_file(cci_root, year, month) if month in O.CCI_MONTHS else None
        if f is not None:
            d = read_cci_month(f)
            if "nh" not in grids:
                raise ValueError("CCI thickness without an OSI-SAF NH grid to assign it with")
            glat, glon = grids["nh"]
            dlat = np.abs(d["lat"] - glat).max()
            dlon = np.abs((d["lon"] - glon + 180.0) % 360.0 - 180.0).max()
            if max(dlat, dlon) > GRID_MATCH_DEG:
                raise ValueError(f"{f}: CCI cell centres differ from OSI-SAF NH by "
                                 f"{max(dlat, dlon):.2e} deg")
            H, sH, valid = O.cci_effective_thickness(
                d["sea_ice_thickness"], d["sea_ice_thickness_uncertainty"],
                d["sea_ice_concentration"], d["status_flag"])
            node, _ = assign["nh"]
            Hn, sHn, nHn = O.super_obs(node, H, sH, valid, nod2D, O.FLOOR_H)
            have = nHn > 0
            O.merge_into(obs["H_obs"], month - 1, Hn, have)
            O.merge_into(obs["sH"], month - 1, sHn, have)
            O.merge_into(obs["nH"], month - 1, nHn, have)
            fl = d["status_flag"]
            mp["H_nh"] = {
                "file": os.path.basename(f),
                "flag_cells": {int(k): int(v) for k, v in zip(*np.unique(fl, return_counts=True))},
                "flag0_without_finite_inputs": int(((fl == 0) & ~valid).sum()),
                "cells_valid": int(valid.sum()),
                "cells_valid_negative_sit": int((valid & (d["sea_ice_thickness"] < 0)).sum()),
                "cells_valid_beyond_cap": int((valid & (node < 0)).sum()),
                "cells_assigned": int(nHn.sum()),
                "nodes_valid": int(have.sum()),
                "sum_weight": float(O.obs_weight(nHn).sum()),
            }
        prov_months.append(mp)
        log(f"{year}-{month:02d}: " + "  ".join(
            f"{k} nodes {v['nodes_valid']}" for k, v in mp.items() if isinstance(v, dict)))
    O.check_obs(obs, nod2D)
    assign_stats = {hem: {"cells": int(assign[hem][0].size),
                          "cells_within_cap": int((assign[hem][0] >= 0).sum()),
                          "cap_km": caps[hem]} for hem in grids}
    return obs, {"months": prov_months, "assignment": assign_stats}


def write_year(out_dir: Path, year: int, obs: dict, node_lon, node_lat, prov: dict):
    npz = out_dir / f"obs_{year:04d}.npz"
    js = out_dir / f"obs_{year:04d}.json"
    with open(npz, "xb") as f:
        np.savez(f, **obs, nh=O.northern_nodes(node_lat), node_lon=node_lon, node_lat=node_lat,
                 year=np.int64(year))
    prov = dict(prov, npz=npz.name, npz_sha256=_file_sha(npz))
    with open(js, "x") as f:
        json.dump(prov, f, indent=1)
    return npz, js


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--years", required=True, help="2013 | 1980-2019 | 2013,2015")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mesh-dir", type=Path, default=ROOT / "data" / "mesh_core2")
    ap.add_argument("--osi-root", default=OSI_ROOT)
    ap.add_argument("--cci-root", default=CCI_ROOT)
    ap.add_argument("--assign", type=Path, default=None,
                    help="cell -> node assignment cache (default <out>/assign.npz; an existing "
                         "file is reused read-only if its hashes match)")
    ap.add_argument("--min-files", type=int, default=MIN_FILES,
                    help="daily files a month needs to get a monthly mean (default %(default)s)")
    a = ap.parse_args(argv)

    years = parse_years(a.years)
    late = [y for y in years if y > LAST_YEAR]
    if late:
        ap.error(f"years after {LAST_YEAR} are not compared: {late}")
    a.out.mkdir(parents=True, exist_ok=True)
    existing = [p for y in years for p in (a.out / f"obs_{y:04d}.npz", a.out / f"obs_{y:04d}.json")
                if p.exists()]
    if existing:
        print(f"refusing to overwrite existing output: {[str(p) for p in existing]}",
              file=sys.stderr)
        return 3

    node_lon, node_lat, node_wet = O.mesh_surface_nodes(a.mesh_dir)
    common = {
        "script": "scripts/icecal/prep_obs.py", "argv": sys.argv if argv is None else argv,
        "git_head": _git_head(), "mesh_dir": str(a.mesh_dir), "nod2D": int(node_lon.size),
        "wet_surface_nodes": int(node_wet.sum()),
        "osi_root": a.osi_root, "cci_root": a.cci_root,
        "decisions": {
            "osi_drop_bits": list(O.OSI_DROP_BITS), "osi_keep_bits": list(O.OSI_KEEP_BITS),
            "osi_fill": O.OSI_FILL, "cci_flag_used": O.CCI_FLAG_NOMINAL,
            "cci_months": list(O.CCI_MONTHS), "cap_km": O.CAP_KM,
            "floor_C": O.FLOOR_C, "floor_H_m": O.FLOOR_H,
            "cell_area_m2": O.CELL_AREA_M2, "A_corr_m2": O.A_CORR_M2,
            "monthly_rule": "cell mean if valid on >= half of the month's available daily files",
            "min_files_per_month": a.min_files,
            "assign_file": str(a.assign or a.out / "assign.npz"),
            "H_obs": "sea_ice_thickness * sea_ice_concentration / 100 (negative SIT kept)",
        },
    }
    for y in years:
        t0 = time.time()
        obs, prov = process_year(y, osi_root=a.osi_root, cci_root=a.cci_root,
                                 node_lon=node_lon, node_lat=node_lat, node_wet=node_wet,
                                 assign_path=a.assign or a.out / "assign.npz",
                                 min_files=a.min_files)
        prov = dict(common, year=y, created=datetime.datetime.now().isoformat(timespec="seconds"),
                    wall_s=round(time.time() - t0, 1), **prov)
        npz, js = write_year(a.out, y, obs, node_lon, node_lat, prov)
        print(f"wrote {npz} and {js} ({prov['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
