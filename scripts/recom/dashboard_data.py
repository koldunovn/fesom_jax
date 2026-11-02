"""The monthly global tables of the REcoM spectral comparison dashboard (lane W of
docs/plans/20261028-recom-spectral-port.md; the dashboard is a claude.ai artifact, its data are these CSV files).

Writes into ``OUT``:

* ``cold.csv``: the cold 1958-59 pair, fesom-jax (``--cold``, a run with the per-group outputs and its
  ``compare/summary.json`` and ``compare/groups_summary.json`` from ``compare_fortran_jax.py`` and
  ``compare_fortran_jax_groups.py``) against ``--cold-fortran``, a row a month: global NPP (fesom-jax four groups,
  both sides' three common groups, every group), CO2 flux, surface Chl, pCO2, SST, ice area, the pattern correlations
  (NPP of the three common groups), the six inventories and the fesom-jax export at 100 m;
* ``warm1998.csv``: the four warm 1998 runs, a row a month: Fortran ``--wjp`` (matched build and physics) and ``--w``
  (production), fesom-jax ``--jx`` (matched, per-group outputs) and ``--nc`` (own physics, no per-group outputs);
  columns ``<metric>_<run>``, plus the jx-vs-wjp pattern correlations.

Units: NPP and CO2 flux Pg C yr-1, Chl mg m-3, pCO2 uatm, SST degC, ice area Mkm2, inventories Pmol.
The 1-degree maps come from ``dashboard_maps.py``.

    python scripts/recom/dashboard_data.py OUT [--cold ... --cold-fortran ... --wjp ... --w ... --jx ... --nc ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_fortran_jax import (FORTRAN_CHL, MMOL_D_TO_PGC_YR, Fortran, Mesh, jax_month,  # noqa: E402
                                 wcorr)

RUNS = Path("/albedo/work/projects/p_specbic/sloza/fesom_jax_runs")
FOUT = Path("/albedo/work/projects/p_specbic/sloza/tmp_recom-fesom_output")
MESH = Path("/albedo/work/projects/p_specbic/sloza/fesom_jax_data/core2_mesh_ic/mesh_core2")
GROUPS = (("phy", "NPPn"), ("dia", "NPPd"), ("phaeo", "NPPp"))     # the groups both sides write NPP for
INVENTORY = ("DIN", "DSi", "DFe", "DIC", "Alk", "O2")


def r6(x):
    return None if x is None else float(f"{x:.6g}")


def write(path: Path, rows: list[dict]):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})


def cold_table(m: Mesh, run: Path, F: Fortran) -> list[dict]:
    A = m.area
    S = json.load(open(run / "compare/summary.json"))
    GS = {r["month"]: r for r in json.load(open(run / "compare/groups_summary.json"))["monthly"]}
    rows = []
    for name in sorted(S):
        s, g = S[name], GS[name]
        y, mo = map(int, name.split("_"))
        J = jax_month(run / "monthly" / name)
        r = {"month": f"{y}-{mo:02d}", "npp_jax": r6(s["npp_global_jax"]), "npp3_jax": r6(g["npp_common_jax"]),
             "npp3_fortran": r6(g["npp_common_fortran"]), "npp_cocco_jax": r6(g["npp_cocco_jax"])}
        for gg, _ in GROUPS:
            r[f"npp_{gg}_jax"] = r6(g[f"npp_{gg}_jax"])
            r[f"npp_{gg}_fortran"] = r6(g[f"npp_{gg}_fortran"])
        r.update({"co2flux_jax": r6(s["co2flux_global_jax"]), "co2flux_fortran": r6(s["co2flux_global_fortran"]),
                  "chl_jax": r6(s["chl_surf_mean_jax"]), "chl_fortran": r6(s["chl_surf_mean_fortran"]),
                  "pco2_jax": r6(s["pco2_surf_mean_jax"]), "pco2_fortran": r6(s["pco2_surf_mean_fortran"]),
                  "sst_jax": r6(s["sst_mean_jax"]), "sst_fortran": r6(s["sst_mean_fortran"]),
                  "ice_jax": r6(s["ice_area_jax_Mkm2"]), "ice_fortran": r6(s["ice_area_fortran_Mkm2"]),
                  "co2flux_corr": r6(s["co2flux_corr"]), "chl_corr": r6(s["chl_corr"]),
                  "export100_jax": r6(float(np.sum(J["export100"] * A) * MMOL_D_TO_PGC_YR))})
        n3j = sum(J[f"npp_{gg}"] for gg, _ in GROUPS)
        n3f = sum(F.field(fv, y, mo) for _, fv in GROUPS)
        r["npp_corr"] = r6(wcorr(n3j, n3f, A))
        for t in INVENTORY:
            r[f"inv_{t}_jax"] = r6(s[f"inv_{t}_jax_Pmol"])
            r[f"inv_{t}_fortran"] = r6(s[f"inv_{t}_fortran_Pmol"])
        rows.append(r)
        print(f"[dashboard] cold {name}: NPP 3 groups {r['npp3_jax']} vs {r['npp3_fortran']}", flush=True)
    return rows


def warm_table(m: Mesh, fortran: dict, jax: dict, year: int) -> list[dict]:
    A = m.area
    glob = lambda x: float(np.sum(x * A) * MMOL_D_TO_PGC_YR)                       # noqa: E731
    amean = lambda x: float(np.sum(x * A) / A.sum())                               # noqa: E731
    ice = lambda x: float(np.sum(x * A) / 1e12)                                    # noqa: E731
    rows = []
    for mo in range(1, 13):
        name = f"{year}_{mo:02d}"
        r = {"month": f"{year}-{mo:02d}"}
        for k, Fk in fortran.items():
            for gg, fv in GROUPS:
                r[f"npp_{gg}_{k}"] = r6(glob(Fk.field(fv, year, mo)))
            r[f"npp3_{k}"] = r6(sum(r[f"npp_{gg}_{k}"] for gg, _ in GROUPS))
            r[f"chl_{k}"] = r6(amean(sum(Fk.field(v, year, mo)[0] for v in FORTRAN_CHL)))
            r[f"co2flux_{k}"] = r6(glob(Fk.field("CO2f", year, mo)))
            r[f"pco2_{k}"] = r6(amean(Fk.field("pCO2s", year, mo)))
            r[f"sst_{k}"] = r6(amean(Fk.field("sst", year, mo)))
            r[f"ice_{k}"] = r6(ice(Fk.field("a_ice", year, mo)))
        Js = {k: jax_month(p / name) for k, p in jax.items()}
        for k, J in Js.items():
            r[f"npp_total_{k}"] = r6(glob(J["npp"]))
            if "npp_phy" in J:                                                     # runs since b178eb6
                for gg, _ in GROUPS:
                    r[f"npp_{gg}_{k}"] = r6(glob(J[f"npp_{gg}"]))
                r[f"npp3_{k}"] = r6(sum(r[f"npp_{gg}_{k}"] for gg, _ in GROUPS))
                r[f"npp_cocco_{k}"] = r6(glob(J["npp_cocco"]))
            r[f"chl_{k}"] = r6(amean(J["Chl"][0]))
            r[f"co2flux_{k}"] = r6(glob(J["co2flux"]))
            r[f"pco2_{k}"] = r6(amean(J["pco2surf"]))
            r[f"sst_{k}"] = r6(amean(J["temp"][0]))
            r[f"ice_{k}"] = r6(ice(J["a_ice"]))
        Fw, J = fortran["wjp"], Js["jx"]
        r["chl_corr_jx"] = r6(wcorr(J["Chl"][0], sum(Fw.field(v, year, mo)[0] for v in FORTRAN_CHL), A))
        r["co2flux_corr_jx"] = r6(wcorr(J["co2flux"], Fw.field("CO2f", year, mo), A))
        r["npp3_corr_jx"] = r6(wcorr(sum(J[f"npp_{gg}"] for gg, _ in GROUPS),
                                     sum(Fw.field(fv, year, mo) for _, fv in GROUPS), A))
        rows.append(r)
        print(f"[dashboard] warm {name}: NPP 3 groups jx {r['npp3_jx']} wjp {r['npp3_wjp']} w {r['npp3_w']}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--mesh", type=Path, default=MESH)
    ap.add_argument("--cold", type=Path, default=RUNS / "core2_recom_spectral_kdind15_groups")
    ap.add_argument("--cold-fortran", type=Path, default=FOUT / "fesom2.6_spectral_kdind15")
    ap.add_argument("--wjp", type=Path, default=FOUT / "fesom2.6_spectral_kdind15wjp")
    ap.add_argument("--w", type=Path, default=FOUT / "fesom2.6_spectral_kdind15w")
    ap.add_argument("--jx", type=Path, default=RUNS / "core2_recom_spectral_warm1998_jaxphys_groups/monthly")
    ap.add_argument("--nc", type=Path, default=RUNS / "core2_recom_spectral_warm1998_noco2lim/monthly")
    ap.add_argument("--year", type=int, default=1998)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = Mesh(a.mesh)
    write(a.out / "cold.csv", cold_table(m, a.cold, Fortran(a.cold_fortran)))
    write(a.out / "warm1998.csv", warm_table(m, {"wjp": Fortran(a.wjp), "w": Fortran(a.w)},
                                             {"jx": a.jx, "nc": a.nc}, a.year))
    print(f"[dashboard] wrote {a.out}/cold.csv, {a.out}/warm1998.csv")


if __name__ == "__main__":
    main()
