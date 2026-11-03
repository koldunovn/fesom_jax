"""A fesom-jax REcoM run against a FESOM2-REcoM Fortran run on the same mesh, month by month (lane W of
docs/plans/20261028-recom-spectral-port.md: the CORE2 spectral runs against the user's ``fesom2.6_spectral_*``).

Inputs: the JAX monthly-mean stores (``<run>/monthly/<YYYY_MM>``, zarr: the ``recom.output`` fields on ``(1, nl,
nod2D)`` plus ``temp``/``salt``/``a_ice``) and the Fortran monthly means (``<ResultPath>/<var>.fesom.<year>.nc``, 12
records of ``(nz1, nod2)`` or ``(nod2)``), the mesh bundle for the areas, levels and coordinates. For every month both
have:

* global NPP of the four groups [Pg C yr-1] (``npp`` vs the sum of the ``NPPn``, ``NPPd``, ``NPPc``, ``NPPp`` streams
  the Fortran run wrote; ``summary.json`` names them: kdind15 has no ``NPPc``), the global air-sea CO2 flux
  [Pg C yr-1] (``co2flux`` vs ``CO2f``, positive into the ocean), the area-mean surface Chl [mg m-3] (``Chl`` vs
  ``PhyChl + DiaChl + CoccoChl + PhaeoChl``), the area-mean surface pCO2 [uatm], the area-mean SST and sea-ice area,
* the global inventories of DIN, DSi, DFe, DIC, Alk, O2 [Pmol] (``sum X dz areasvol`` over the wet layers, the nominal
  thicknesses on both sides),
* the pattern agreement of the surface fields: the area-weighted correlation and the RMS difference of surface Chl,
  NPP and the CO2 flux.

Writes ``summary.json`` and ``compare_<YYYY_MM>.png`` (surface Chl, NPP, CO2 flux: JAX, Fortran, difference; tripcolor
on the mesh) into ``--out``; prints the table. Units: REcoM's outputs are mmol C m-2 d-1 for NPP and the CO2 flux, mg
Chl m-3, mmol m-3, uatm, on both sides.

    python scripts/recom/compare_fortran_jax.py --jax <run>/monthly --fortran <ResultPath> --mesh <bundle> --out <dir>
        [--months 1998_01 1998_02 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np

MMOL_D_TO_PGC_YR = 365.0 * 12.011e-3 / 1e15          # mmol C d-1 -> Pg C yr-1
INVENTORY = ("DIN", "DSi", "DFe", "DIC", "Alk", "O2")
FORTRAN_CHL = ("PhyChl", "DiaChl", "CoccoChl", "PhaeoChl")
FORTRAN_NPP = ("NPPn", "NPPd", "NPPc", "NPPp")


class Mesh:
    def __init__(self, bundle: Path):
        ld = lambda n: np.load(bundle / f"{n}.npy")                                # noqa: E731
        self.geo = ld("geo_coord_nod2D")                                            # (nod, 2) radians
        self.lon, self.lat = np.degrees(self.geo[:, 0]), np.degrees(self.geo[:, 1])
        self.area = ld("area")[:, 0]                                                # (nod,) surface control area
        self.areasvol = ld("areasvol")                                              # (nod, nl)
        self.zbar = ld("zbar")
        self.dz = -np.diff(self.zbar)                                               # (nl-1,) nominal thicknesses
        self.nlev = ld("nlevels_nod2D").astype(int)
        self.elem = ld("elem_nodes")
        nl1 = len(self.dz)
        self.wet = np.arange(nl1)[None, :] < (self.nlev - 1)[:, None]               # (nod, nl-1)
        self.vol = np.where(self.wet, self.areasvol[:, :nl1] * self.dz[None, :], 0.0)


def jax_month(store: Path):
    import zarr
    g = zarr.open_group(str(store), mode="r")
    out = {}
    for k in g.array_keys():
        a = np.asarray(g[k][...])
        out[k] = a[0] if a.ndim >= 2 and a.shape[0] == 1 else a
    return out                                   # 3-D fields (nl, nod), 2-D (nod,)


class Fortran:
    def __init__(self, path: Path):
        self.path = path
        self._cache = {}

    def field(self, var: str, year: int, month: int):
        key = (var, year)
        if key not in self._cache:
            d = netCDF4.Dataset(self.path / f"{var}.fesom.{year}.nc")
            self._cache[key] = d
        d = self._cache[key]
        v = d.variables[var]
        assert v.shape[0] == 12, (var, v.shape)
        # FESOM 2.8 masks the levels below the bottom with _FillValue (2.6 wrote zeros): fill them with 0
        return np.ma.filled(v[month - 1], 0.0).astype(np.float64)  # (nz1, nod) or (nod,)

    def has(self, var: str, year: int):
        return (self.path / f"{var}.fesom.{year}.nc").is_file()


def wcorr(a, b, w):
    w = w / w.sum()
    ma, mb = np.sum(w * a), np.sum(w * b)
    ca, cb = a - ma, b - mb
    den = np.sqrt(np.sum(w * ca * ca) * np.sum(w * cb * cb))
    return float(np.sum(w * ca * cb) / den) if den > 0 else float("nan")


def compare_month(m: Mesh, J: dict, F: Fortran, year: int, month: int) -> dict:
    nl1 = len(m.dz)
    A = m.area
    r = {}
    # the 2-D rates
    npp_j = J["npp"]
    have = [v for v in FORTRAN_NPP if F.has(v, year)]            # kdind15's streams carry NPPn, NPPd, NPPp, no NPPc
    r["fortran_npp_groups"] = have
    npp_f = sum(F.field(v, year, month) for v in have)
    co2_j, co2_f = J["co2flux"], F.field("CO2f", year, month)
    chl_j = J["Chl"][0]
    chl_f = sum(F.field(v, year, month)[0] for v in FORTRAN_CHL)
    for name, a, b, unit, scale in (("npp", npp_j, npp_f, "Pg C yr-1", MMOL_D_TO_PGC_YR),
                                    ("co2flux", co2_j, co2_f, "Pg C yr-1", MMOL_D_TO_PGC_YR)):
        r[f"{name}_global_jax"] = float(np.sum(a * A) * scale)
        r[f"{name}_global_fortran"] = float(np.sum(b * A) * scale)
        r[f"{name}_corr"] = wcorr(a, b, A)
        r[f"{name}_rms"] = float(np.sqrt(np.sum(A * (a - b) ** 2) / A.sum()))
    r["chl_surf_mean_jax"] = float(np.sum(chl_j * A) / A.sum())
    r["chl_surf_mean_fortran"] = float(np.sum(chl_f * A) / A.sum())
    r["chl_corr"] = wcorr(chl_j, chl_f, A)
    r["chl_rms"] = float(np.sqrt(np.sum(A * (chl_j - chl_f) ** 2) / A.sum()))
    r["pco2_surf_mean_jax"] = float(np.sum(J["pco2surf"] * A) / A.sum())
    r["pco2_surf_mean_fortran"] = float(np.sum(F.field("pCO2s", year, month) * A) / A.sum())
    r["sst_mean_jax"] = float(np.sum(J["temp"][0] * A) / A.sum())
    r["sst_mean_fortran"] = float(np.sum(F.field("sst", year, month) * A) / A.sum())
    r["ice_area_jax_Mkm2"] = float(np.sum(J["a_ice"] * A) / 1e12)
    r["ice_area_fortran_Mkm2"] = float(np.sum(F.field("a_ice", year, month) * A) / 1e12)
    # the inventories
    for name in INVENTORY:
        xj = np.asarray(J[name])[:nl1].T                                            # (nod, nl-1)
        xf = F.field(name, year, month).T
        r[f"inv_{name}_jax_Pmol"] = float(np.sum(xj * m.vol) / 1e18)
        r[f"inv_{name}_fortran_Pmol"] = float(np.sum(xf * m.vol) / 1e18)
    fields = dict(chl=(chl_j, chl_f), npp=(npp_j, npp_f), co2flux=(co2_j, co2_f))
    return r, fields


def figure(m: Mesh, fields: dict, tag: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("surface Chl [mg m-3]", "chl", 0.0, 1.5), ("NPP [mmol C m-2 d-1]", "npp", 0.0, 60.0),
            ("CO2 flux [mmol C m-2 d-1]", "co2flux", -30.0, 30.0)]
    fig, axs = plt.subplots(len(rows), 3, figsize=(18, 4.2 * len(rows)), constrained_layout=True)
    tri = m.elem
    lon = np.where(m.lon > 180.0, m.lon - 360.0, m.lon)
    keep = np.ptp(lon[tri], axis=1) < 180.0                                        # drop the dateline-crossing elements
    for i, (title, key, lo, hi) in enumerate(rows):
        a, b = fields[key]
        d = a - b
        lim = float(np.percentile(np.abs(d), 99)) or 1e-9
        for j, (x, t, vmin, vmax, cmap) in enumerate(((a, "fesom-jax", lo, hi, "viridis"), (b, "Fortran", lo, hi, "viridis"),
                                                      (d, "fesom-jax - Fortran", -lim, lim, "RdBu_r"))):
            ax = axs[i, j]
            pc = ax.tripcolor(lon, m.lat, tri[keep], x, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}: {t}")
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            fig.colorbar(pc, ax=ax, shrink=0.8)
    fig.suptitle(tag)
    fig.savefig(out / f"compare_{tag}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jax", type=Path, required=True, help="the JAX run's monthly directory")
    ap.add_argument("--fortran", type=Path, required=True, help="the Fortran run's ResultPath")
    ap.add_argument("--mesh", type=Path, required=True, help="the mesh bundle")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--months", nargs="*", default=None, help="YYYY_MM (default: every month both have)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = Mesh(args.mesh)
    F = Fortran(args.fortran)
    months = args.months or sorted(p.name for p in args.jax.iterdir() if p.is_dir() and len(p.name) == 7)
    rows = {}
    for tag in months:
        year, month = int(tag[:4]), int(tag[5:7])
        if not F.has("CO2f", year):
            print(f"[compare] {tag}: no Fortran output for {year}, skipped")
            continue
        J = jax_month(args.jax / tag)
        r, fields = compare_month(m, J, F, year, month)
        rows[tag] = r
        print(f"[compare] {tag}: NPP {r['npp_global_jax']:.2f} vs {r['npp_global_fortran']:.2f} Pg C/yr (corr "
              f"{r['npp_corr']:.3f}); CO2 flux {r['co2flux_global_jax']:.2f} vs {r['co2flux_global_fortran']:.2f} "
              f"Pg C/yr (corr {r['co2flux_corr']:.3f}); surface Chl {r['chl_surf_mean_jax']:.3f} vs "
              f"{r['chl_surf_mean_fortran']:.3f} mg/m3 (corr {r['chl_corr']:.3f}); pCO2 {r['pco2_surf_mean_jax']:.1f} vs "
              f"{r['pco2_surf_mean_fortran']:.1f}; SST {r['sst_mean_jax']:.3f} vs {r['sst_mean_fortran']:.3f}; ice "
              f"{r['ice_area_jax_Mkm2']:.2f} vs {r['ice_area_fortran_Mkm2']:.2f} Mkm2", flush=True)
        print("[compare]   inventories [Pmol] " + ", ".join(
            f"{n} {r[f'inv_{n}_jax_Pmol']:.4f}/{r[f'inv_{n}_fortran_Pmol']:.4f}" for n in INVENTORY), flush=True)
        if not args.no_figures:
            figure(m, fields, tag, args.out)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"[compare] wrote {args.out}/summary.json ({len(rows)} months)")


if __name__ == "__main__":
    main()
