"""REcoM plan Task 15: the summary of a CORE2 + full REcoM run from its monthly mean output (nereus env: zarr, nereus).

Per month: the global inventories of DIC, Alk, DIN, DSi, DFe, O2 [Pmol] (``sum X dz_nominal areasvol`` over the wet
layers, the monthly mean fields), the global air-sea CO2 flux, NPP and the 100 m POC export [Pg C yr-1] (``sum F
area_surface``, F in mmol C m-2 d-1) and the area-mean surface pCO2 [uatm]; the relative inventory change from the
first month. Maps (nereus, regridded, never scatter): the annual-mean CO2 flux and NPP.

    python scripts/recom/core2_year_summary.py <run>/monthly --out <dir> [--mesh data/mesh_core2]

Writes ``summary.json`` and ``maps.png`` into ``--out``; prints the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
INV = ("DIC", "Alk", "DIN", "DSi", "DFe", "O2")
FLUX = ("co2flux", "npp", "export100")
MMOL_D_TO_PGC_YR = 365.0 * 12.011e-3 / 1e15          # mmol C d-1 -> Pg C yr-1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("monthly", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, default=REPO / "data" / "mesh_core2")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import zarr

    m = args.mesh
    area = np.load(m / "area.npy")[:, 0]
    areasvol = np.load(m / "areasvol.npy")                     # (nod2D, nl)
    zbar = np.load(m / "zbar.npy")
    nlev = np.load(m / "nlevels_nod2D.npy").astype(int)
    lon, lat = np.rad2deg(np.load(m / "geo_coord_nod2D.npy")).T
    dz = zbar[:-1] - zbar[1:]                                  # (nl-1,) nominal layer thickness
    nl1 = dz.size
    wet = np.arange(nl1)[None, :] < (nlev - 1)[:, None]        # (nod2D, nl-1)
    vol = np.where(wet, dz[None, :] * areasvol[:, :nl1], 0.0)

    rows, sums = [], {k: 0.0 for k in FLUX}
    months = sorted(p for p in args.monthly.iterdir() if (p / "DIC").exists())
    for p in months:
        g = zarr.open_group(str(p), mode="r")
        row = {"month": p.name}
        for k in INV:
            x = np.asarray(g[k][0], dtype=np.float64).T[:, :nl1]   # (nod2D, nl-1)
            row[k] = float(np.sum(np.where(wet, x, 0.0) * vol)) / 1e18     # mmol -> Pmol
        for k in FLUX:
            f = np.asarray(g[k][0], dtype=np.float64)
            row[k] = float(np.sum(f * area)) * MMOL_D_TO_PGC_YR
            sums[k] = sums[k] + f
        pc = np.asarray(g["pco2surf"][0], dtype=np.float64)
        row["pco2surf"] = float(np.sum(pc * area) / np.sum(area))
        rows.append(row)
    for r in rows:
        for k in INV:
            r[k + "_rel"] = r[k] / rows[0][k] - 1.0

    hdr = ["month"] + list(INV) + list(FLUX) + ["pco2surf"]
    print(" ".join(f"{h:>10s}" for h in hdr))
    for r in rows:
        print(" ".join([f"{r['month']:>10s}"] + [f"{r[k]:10.4f}" for k in INV]
                       + [f"{r[k]:10.3f}" for k in FLUX] + [f"{r['pco2surf']:10.2f}"]))
    print("inventory change last/first month:", {k: f"{rows[-1][k + '_rel']:+.2e}" for k in INV})
    ann = {k: float(np.mean([r[k] for r in rows])) for k in FLUX}
    print("annual means [Pg C yr-1]:", {k: round(v, 3) for k, v in ann.items()})
    (args.out / "summary.json").write_text(json.dumps({"months": rows, "annual_PgC_yr": ann}, indent=1))

    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nereus

    proj = ccrs.Robinson()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), layout="constrained", subplot_kw={"projection": proj})
    interp = None
    for ax, k, title, cmap, sym in ((axes[0], "co2flux", "air-sea CO$_2$ flux (into the ocean)", "RdBu_r", True),
                                    (axes[1], "npp", "net primary production", "viridis", False)):
        v = sums[k] / len(months)
        hi = float(np.percentile(np.abs(v), 99.0))
        kw = dict(vmin=-hi, vmax=hi) if sym else dict(vmin=0.0, vmax=hi)
        im, _, interp = nereus.plot(v, lon, lat, projection=proj, resolution=1.0, interpolator=interp,
                                    method="nearest", influence_radius=150e3, cmap=cmap, land=True,
                                    colorbar=False, ax=ax, rasterized=True, **kw)
        ax.set_global()
        ax.set_title(f"{title}, annual mean ({months[0].name}..{months[-1].name}): {ann[k]:.2f} Pg C yr$^{{-1}}$",
                     loc="left", fontsize=8)
        from matplotlib.collections import QuadMesh
        mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
        cb = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.03, pad=0.01)
        cb.set_label("mmol C m$^{-2}$ d$^{-1}$", fontsize=7)
    fig.savefig(args.out / "maps.png", dpi=150)
    print("wrote", args.out / "maps.png")


if __name__ == "__main__":
    main()
