"""Observations on the CORE2 mesh for the REcoM figures (fesom-jax env, CPU, a compute node).

* WOA13 nitrate and silicate, WOA18 oxygen, GLODAPv2 DIC (present-day and preindustrial) and alkalinity: the files of
  REcoM's initial conditions, interpolated to the model levels exactly as the model's cold start does
  (:func:`fesom_jax.recom.ic.interp_ic_field` = FESOM2 ``do_ic3d``), ``(nod2D, nl-1)``;
* GlobColour merged monthly surface chlorophyll (CHL1, 1 degree): the monthly climatology of ``--chl-years``, taken at
  each mesh node from the 1-degree cell that holds it (no fill: NaN where the satellite has no data), ``(12, nod2D)``.

    python scripts/recom/page/prep_obs.py --out <dir> [--mesh data/mesh_core2] [--chl-years 2003 2010]

Writes ``<dir>/obs_on_mesh.npz`` (float32).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
IC_DIR = Path("/work/ab0246/a270092/input/fesom2/recom")
GLOBCOLOUR = Path("/pool/data/ICDC/ocean/globcolour_chloro/DATA/100")
FIELDS = {
    "DIN": ("woa13_all_n00_01_fesom2.nc", "n_an"),
    "DSi": ("woa13_all_i00_01_fesom2.nc", "i_an"),
    "O2": ("woa18_all_o00_01_mmol_fesom2.nc", "oxygen_mmol"),
    "DIC": ("GLODAPv2.2016b.TCO2_fesom2_mmol_fix_z_Fillvalue.nc", "TCO2_mmol"),
    "DIC_PI": ("GLODAPv2.2016b.PI_TCO2_fesom2_mmol_fix_z_Fillvalue.nc", "PI_TCO2_mmol"),
    "Alk": ("GLODAPv2.2016b.TAlk_fesom2_mmol_fix_z_Fillvalue.nc", "TAlk_mmol"),
}


def chl_climatology(lon, lat, years):
    """GlobColour monthly CHL1 climatology over ``years`` at the nodes ``lon``/``lat`` [deg]: ``(12, n)`` mean and
    count of the years with data."""
    import netCDF4
    files = sorted(GLOBCOLOUR.glob("L3m_*_CHL1_MO_00.nc"))
    n = lon.size
    tot, cnt = np.zeros((12, n)), np.zeros((12, n))
    ii = jj = None
    for f in files:
        y, m = int(f.name[4:8]), int(f.name[8:10])
        if not years[0] <= y <= years[1]:
            continue
        with netCDF4.Dataset(str(f)) as ds:
            if ii is None:
                glat, glon = np.asarray(ds["lat"][:], float), np.asarray(ds["lon"][:], float)
                ii = np.clip(np.rint((lat - glat[0]) / (glat[1] - glat[0])).astype(int), 0, glat.size - 1)
                x = ((lon - glon[0]) % 360.0)
                jj = np.clip(np.rint(x / (glon[1] - glon[0])).astype(int) % glon.size, 0, glon.size - 1)
            v = np.ma.filled(ds["CHL1_mean"][:].astype(float), np.nan)[ii, jj]
        ok = np.isfinite(v) & (v > 0.0)
        tot[m - 1, ok] += v[ok]
        cnt[m - 1, ok] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), np.nan)
    return clim, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, default=REPO / "data" / "mesh_core2")
    ap.add_argument("--chl-years", type=int, nargs=2, default=(2003, 2010))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from fesom_jax.mesh import load_mesh
    from fesom_jax.recom.ic import _neighbour_table, ic_layer_depths, interp_ic_field

    mesh = load_mesh(args.mesh)
    lon, lat = np.rad2deg(np.asarray(mesh.geo_coord_nod2D, float)).T
    z3, table = ic_layer_depths(mesh), _neighbour_table(mesh)
    out = {}
    for k, (f, var) in FIELDS.items():
        out[k] = interp_ic_field(mesh, IC_DIR / f, var, z3=z3, _table=table).astype(np.float32)
        print(f"{k}: {f} {var} -> {out[k].shape}, surface mean {np.nanmean(out[k][:, 0]):.2f}", flush=True)
    clim, cnt = chl_climatology(lon, lat, tuple(args.chl_years))
    print(f"GlobColour {args.chl_years}: nodes with data per month {[int(np.isfinite(c).sum()) for c in clim]}")
    np.savez(args.out / "obs_on_mesh.npz", **out, chl_clim=clim.astype(np.float32), chl_count=cnt.astype(np.int16),
             chl_years=np.asarray(args.chl_years), lon=lon.astype(np.float32), lat=lat.astype(np.float32))
    print("wrote", args.out / "obs_on_mesh.npz")


if __name__ == "__main__":
    main()
