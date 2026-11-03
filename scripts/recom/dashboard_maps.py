"""The 1-degree surface maps and zonal means of the REcoM spectral comparison dashboard (the cold 1958-59 pair; lane W
of docs/plans/completed/20261028-recom-spectral-port.md). The tables come from ``dashboard_data.py``.

Writes into ``OUT`` one ``map_<YYYY_MM>.csv`` per month (``lat, lon`` of the 1-degree cell centre, then surface Chl
[mg m-3], NPP of the three groups both sides write [mmol C m-2 d-1] and the CO2 flux [mmol C m-2 d-1, into the ocean],
``_jax`` and ``_fortran`` each) and ``zonal.csv`` (the same fields per month and 1-degree latitude band).

Regridding from the CORE2 mesh nodes: a cell holding nodes takes their area-weighted mean (``area`` of the mesh
bundle); a cell holding none whose centre lies in a mesh triangle takes the linear (barycentric) interpolation of the
triangle's nodes at the centre; any other cell is land and is left out. The zonal means are area-weighted over the
nodes of each band. CORE2 node spacing is ~25 km in the Arctic and North Atlantic, 40-50 km in the Southern Ocean and
the tropics and up to ~100 km in the subtropical gyres, so 1 degree keeps what the mesh resolves almost everywhere;
the empty cells (~5 % of the ocean cells) sit in the gyres.

    python scripts/recom/dashboard_maps.py OUT [--jax <run> --fortran <ResultPath> --mesh <bundle>]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_fortran_jax import FORTRAN_CHL, Fortran, Mesh, jax_month  # noqa: E402

RUNS = Path("/albedo/work/projects/p_specbic/sloza/fesom_jax_runs")
FOUT = Path("/albedo/work/projects/p_specbic/sloza/tmp_recom-fesom_output")
MESH = Path("/albedo/work/projects/p_specbic/sloza/fesom_jax_data/core2_mesh_ic/mesh_core2")
NPP_STREAMS = ("NPPn", "NPPd", "NPPp")
NLAT, NLON = 180, 360


def unit_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


class Regrid:
    """Mesh nodes -> the ocean cells of a 1-degree grid (node-area mean, else barycentric at the cell centre)."""

    def __init__(self, m: Mesh, ncand: int = 12):
        self.A = m.area
        lon = np.where(m.lon >= 180, m.lon - 360, m.lon)
        lat = m.lat
        self.ilat = np.clip(np.floor(lat + 90).astype(int), 0, NLAT - 1)
        ilon = np.clip(np.floor(lon + 180).astype(int), 0, NLON - 1)
        self.cell = self.ilat * NLON + ilon
        self.wsum = np.bincount(self.cell, weights=self.A, minlength=NLAT * NLON)
        self.has = self.wsum > 0
        # the empty cells: search the elements whose centroids lie nearest the cell centre
        empty = np.nonzero(~self.has)[0]
        clat = empty // NLON - 90 + 0.5
        clon = empty % NLON - 180 + 0.5
        E = m.elem
        cent = np.stack([unit_xyz(lat[E[:, k]], lon[E[:, k]]) for k in range(3)], 0).mean(0)
        _, cand = cKDTree(cent).query(unit_xyz(clat, clon), k=ncand)
        cells, nodes, weights = [], [], []
        for j in range(len(empty)):
            for e in cand[j]:
                v = E[e]
                x = ((lon[v] - clon[j] + 180) % 360) - 180                      # unwrapped about the centre
                y = lat[v] - clat[j]
                den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
                if den == 0:
                    continue
                l1 = ((y[1] - y[2]) * -x[2] + (x[2] - x[1]) * -y[2]) / den
                l2 = ((y[2] - y[0]) * -x[2] + (x[0] - x[2]) * -y[2]) / den
                l3 = 1 - l1 - l2
                if min(l1, l2, l3) >= -1e-9:
                    cells.append(empty[j]); nodes.append(v); weights.append((l1, l2, l3))
                    break
        self.icell = np.array(cells, dtype=int)
        self.inodes = np.array(nodes, dtype=int).reshape(-1, 3)
        self.iw = np.array(weights).reshape(-1, 3)
        self.cells = np.sort(np.concatenate([np.nonzero(self.has)[0], self.icell]))
        self.lat = self.cells // NLON - 90 + 0.5
        self.lon = self.cells % NLON - 180 + 0.5
        self.zw = np.bincount(self.ilat, weights=self.A, minlength=NLAT)
        print(f"[maps] cells with nodes {self.has.sum()}, interpolated {len(self.icell)}", flush=True)

    def grid(self, f):
        g = np.full(NLAT * NLON, np.nan)
        g[self.has] = np.bincount(self.cell, weights=f * self.A, minlength=NLAT * NLON)[self.has] / self.wsum[self.has]
        if len(self.icell):
            g[self.icell] = np.sum(f[self.inodes] * self.iw, 1)
        return g[self.cells]

    def zonal(self, f):
        return np.bincount(self.ilat, weights=f * self.A, minlength=NLAT) / np.where(self.zw > 0, self.zw, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--jax", type=Path, default=RUNS / "core2_recom_spectral_kdind15_groups",
                    help="the fesom-jax run (its monthly/ stores, with the per-group outputs)")
    ap.add_argument("--fortran", type=Path, default=FOUT / "fesom2.6_spectral_kdind15")
    ap.add_argument("--mesh", type=Path, default=MESH)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m, F = Mesh(a.mesh), Fortran(a.fortran)
    rg = Regrid(m)
    zrows = []
    for name in sorted(p.name for p in (a.jax / "monthly").iterdir()):
        y, mo = map(int, name.split("_"))
        if not F.has("CO2f", y):
            continue
        J = jax_month(a.jax / "monthly" / name)
        f = {"chl_jax": J["Chl"][0], "chl_fortran": sum(F.field(v, y, mo)[0] for v in FORTRAN_CHL),
             "npp_jax": J["npp_phy"] + J["npp_dia"] + J["npp_phaeo"],
             "npp_fortran": sum(F.field(v, y, mo) for v in NPP_STREAMS),
             "co2_jax": J["co2flux"], "co2_fortran": F.field("CO2f", y, mo)}
        gs = {k: rg.grid(v) for k, v in f.items()}
        path = a.out / f"map_{name}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["lat", "lon"] + list(f))
            for i in range(len(rg.cells)):
                w.writerow([f"{rg.lat[i]:g}", f"{rg.lon[i]:g}"] + [f"{gs[k][i]:.4g}" for k in f])
        zs = {k: rg.zonal(v) for k, v in f.items()}
        for b in np.nonzero(rg.zw > 0)[0]:
            zrows.append({"month": f"{y}-{mo:02d}", "lat": b - 90 + 0.5, **{k: float(f"{zs[k][b]:.5g}") for k in f}})
        print(f"[maps] {name}: {path.stat().st_size} bytes", flush=True)
    with open(a.out / "zonal.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(zrows[0]))
        w.writeheader()
        w.writerows(zrows)
    print(f"[maps] wrote {a.out}")


if __name__ == "__main__":
    main()
