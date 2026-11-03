"""The production-gap bisect table (docs/plans/20261103-production-gap-bisect.md): global monthly and 1998-mean values
of Fortran CORE2 runs, and each run's share of the gap between the production run kdind15w and the matched run
kdind15wjp.

For every run (``name=ResultPath``; defaults: the two end points ``w`` and ``wjp``, plus every
``fesom2.6_bisect_<name>`` found): NPP of the three groups the runs write (``NPPn``, ``NPPd``, ``NPPp``) and their sum
[Pg C yr-1], surface Chl (four groups, area mean, mg m-3), CO2 flux [Pg C yr-1], surface pCO2 [uatm], SST [degC], sea-ice
area north and south [Mkm2]. ``position`` = (run - wjp) / (w - wjp) for each metric's mean over the months the run has:
1 = like production, 0 = like the matched run.

    python scripts/recom/fortran/bisect_table.py OUT [--run name=PATH ...] [--year 1998]

Writes ``OUT/bisect_monthly.csv`` and ``OUT/bisect_means.csv`` and prints the means table.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_fortran_jax import FORTRAN_CHL, MMOL_D_TO_PGC_YR, Fortran, Mesh  # noqa: E402

FOUT = Path("/albedo/work/projects/p_specbic/sloza/tmp_recom-fesom_output")
MESH = Path("/albedo/work/projects/p_specbic/sloza/fesom_jax_data/core2_mesh_ic/mesh_core2")
GROUPS = (("phy", "NPPn"), ("dia", "NPPd"), ("phaeo", "NPPp"))
METRICS = ("npp3", "npp_phy", "npp_dia", "npp_phaeo", "chl", "co2flux", "pco2", "sst", "ice_n", "ice_s")


def month_values(m: Mesh, F: Fortran, year: int, mo: int) -> dict:
    A = m.area
    glob = lambda x: float(np.sum(x * A) * MMOL_D_TO_PGC_YR)                       # noqa: E731
    amean = lambda x: float(np.sum(x * A) / A.sum())                               # noqa: E731
    r = {f"npp_{g}": glob(F.field(v, year, mo)) for g, v in GROUPS}
    r["npp3"] = sum(r[f"npp_{g}"] for g, _ in GROUPS)
    r["chl"] = amean(sum(F.field(v, year, mo)[0] for v in FORTRAN_CHL))
    r["co2flux"] = glob(F.field("CO2f", year, mo))
    r["pco2"] = amean(F.field("pCO2s", year, mo))
    r["sst"] = amean(F.field("sst", year, mo))
    a = F.field("a_ice", year, mo) * A / 1e12
    r["ice_n"] = float(np.sum(a[m.lat > 0]))
    r["ice_s"] = float(np.sum(a[m.lat < 0]))
    return r


def months_of(F: Fortran, year: int) -> int:
    import netCDF4
    if not F.has("NPPn", year):
        return 0
    with netCDF4.Dataset(F.path / f"NPPn.fesom.{year}.nc") as d:
        return len(d.dimensions["time"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--run", action="append", default=[], help="name=ResultPath (repeatable)")
    ap.add_argument("--year", type=int, default=1998)
    ap.add_argument("--mesh", type=Path, default=MESH)
    a = ap.parse_args()
    runs = {"w": FOUT / "fesom2.6_spectral_kdind15w", "wjp": FOUT / "fesom2.6_spectral_kdind15wjp"}
    runs.update({p.name.removeprefix("fesom2.6_bisect_"): p for p in sorted(FOUT.glob("fesom2.6_bisect_*"))})
    runs.update(dict(s.split("=", 1) for s in a.run))
    m = Mesh(a.mesh)
    rows, means = [], {}
    for name, path in runs.items():
        F = Fortran(Path(path))
        n = months_of(F, a.year)
        if n == 0:
            print(f"[bisect] {name}: no {a.year} output yet")
            continue
        vals = []
        for mo in range(1, n + 1):
            try:
                v = month_values(m, F, a.year, mo)
            except AssertionError:                                                 # a year file still being written
                break
            vals.append(v)
            rows.append({"run": name, "month": f"{a.year}-{mo:02d}", **{k: f"{v[k]:.6g}" for k in METRICS}})
        means[name] = {"n_months": len(vals), **{k: float(np.mean([v[k] for v in vals])) for k in METRICS}}
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "bisect_monthly.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "month", *METRICS])
        w.writeheader()
        w.writerows(rows)
    # position against the end points over the same months (the common prefix)
    out = []
    for name, mv in means.items():
        n = mv["n_months"]
        ends = {}
        for e in ("w", "wjp"):
            ev = [r for r in rows if r["run"] == e][:n]
            ends[e] = {k: float(np.mean([float(r[k]) for r in ev])) for k in METRICS} if len(ev) == n else None
        rec = {"run": name, "n_months": n}
        for k in METRICS:
            rec[k] = mv[k]
            if ends["w"] and ends["wjp"] and ends["w"][k] != ends["wjp"][k]:
                rec[f"pos_{k}"] = (mv[k] - ends["wjp"][k]) / (ends["w"][k] - ends["wjp"][k])
        out.append(rec)
    cols = ["run", "n_months", *METRICS, *[f"pos_{k}" for k in METRICS]]
    with open(a.out / "bisect_means.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in cols} for r in out])
    show = ("npp3", "npp_dia", "chl", "co2flux", "ice_n", "ice_s")
    print(f"{'run':16s} {'mo':>3s} " + " ".join(f"{k:>9s} {'pos':>5s}" for k in show))
    for r in out:
        print(f"{r['run']:16s} {r['n_months']:3d} " + " ".join(
            f"{r[k]:9.4g} {r.get('pos_' + k, float('nan')):5.2f}" for k in show))
    print(f"[bisect] wrote {a.out}/bisect_monthly.csv, bisect_means.csv")


if __name__ == "__main__":
    main()
