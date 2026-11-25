"""G2 (partial-cells plan, Task 8): the response of the model to partial cells, JAX against Fortran FESOM2.

For each model X (JAX, Fortran) and each window (day 1, month 1, year 1958), Δ_X = X_pc − X_fc of three fields:
bottom-layer temperature, bottom-layer speed (|u|, invariant to the Fortran output's rotation to geographic axes) and
SSH. Weights: the surface area of each node's control volume (areasvol[:, 0]). Mask: every node, minus those whose
level counts differ between our mesh export and the Fortran run's /pool levels and the nodes of elements whose levels
differ (memory note core2-mesh-prefix-levels).

Pass bars (fixed in the plan before any run):
  direct   (day 1, month 1): rms(JAX_pc − F_pc) ≤ 1.2·rms(JAX_fc − F_fc), and both ≤ 0.5·rms(Δ_F)
  response (day 1, year)   : corr(Δ_JAX, Δ_F) ≥ 0.9 at day 1, ≥ 0.7 for the year; std(Δ_JAX)/std(Δ_F) in [0.7, 1.4]

    <env-py> scripts/partial_cells/g2_compare.py <runs root> [mesh dir]
The runs root holds fortran/{zstar,linfs}_{fc,pc}_1m (daily output), fortran/zstar_pc_1y, and jax/<ale>_<fc|pc>_<1d|1m|1y>.
The full-cell Fortran year is fesom2_icefw_runs/A_main_ias2_year.
"""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
import zarr

from fesom_jax.mesh import load_mesh

ROOT = Path(sys.argv[1])
MESH = sys.argv[2] if len(sys.argv) > 2 else "data/mesh_core2"
F_FC_YEAR = Path("/work/ab0995/a270088/fesom2_icefw_runs/A_main_ias2_year")

mesh = load_mesh(MESH)
nln = np.asarray(mesh.nlevels_nod2D)
kb = nln - 2                                                  # 0-based bottom layer of each node
rows = np.arange(mesh.nod2D)
w = np.asarray(mesh.areasvol)[:, 0]


def level_mask(fortran_dir):
    d = netCDF4.Dataset(fortran_dir / "fesom.mesh.diag.nc")
    f_nlev, f_nlevn = np.asarray(d["nlevels"][:]), np.asarray(d["nlevels_nod2D"][:])
    keep = f_nlevn == nln
    en = np.asarray(mesh.elem_nodes)
    keep[en[f_nlev != np.asarray(mesh.nlevels)].ravel()] = False
    return keep


def _nodes_last(a, n_lev):
    """Fortran 3-D output as (time, nod2D, nz): FESOM2 writes either (time, nz1, nod2) or (time, nod2, nz1)."""
    a = np.asarray(a, np.float64)
    if a.ndim == 3 and a.shape[1] == n_lev and a.shape[2] == mesh.nod2D:
        a = a.transpose(0, 2, 1)
    return a


def fortran_fields(run, recs):
    """Mean over the records ``recs`` (a slice) of the run's output; returns the three fields."""
    def var(name):
        with netCDF4.Dataset(run / f"{name}.fesom.1958.nc") as d:
            v = d[name]
            a = v[recs]
            return np.asarray(a, np.float64)
    T = _nodes_last(var("temp"), 47).mean(axis=0)
    u = _nodes_last(var("unod"), 47).mean(axis=0)
    v = _nodes_last(var("vnod"), 47).mean(axis=0)
    ssh = var("ssh").mean(axis=0)
    return {"bottom_T": T[rows, kb], "bottom_speed": np.hypot(u[rows, kb], v[rows, kb]), "ssh": ssh}


def jax_fields(stores):
    """Mean over the monthly-mean stores ``stores`` (each one calendar period) of the three fields."""
    acc = None
    for s in stores:
        g = zarr.open_group(str(s), mode="r")
        def a(name):
            x = np.asarray(g[name][...], np.float64)
            return x[0] if x.shape[0] == 1 and x.ndim >= 2 and x.shape[1] == mesh.nod2D else x
        T, u, v, ssh = a("temp"), a("u"), a("v"), a("ssh")
        f = {"bottom_T": T[rows, kb], "bottom_speed": np.hypot(u[rows, kb], v[rows, kb]), "ssh": ssh}
        acc = f if acc is None else {k: acc[k] + f[k] for k in acc}
    return {k: v / len(stores) for k, v in acc.items()}


def rms(x, m):
    return float(np.sqrt(np.sum(w[m] * x[m] ** 2) / np.sum(w[m])))


def corr(a, b, m):
    wa, wb = a[m] - np.average(a[m], weights=w[m]), b[m] - np.average(b[m], weights=w[m])
    return float(np.sum(w[m] * wa * wb) / np.sqrt(np.sum(w[m] * wa ** 2) * np.sum(w[m] * wb ** 2)))


def std(a, m):
    return float(np.sqrt(np.average((a[m] - np.average(a[m], weights=w[m])) ** 2, weights=w[m])))


def months(run):
    return sorted(p for p in (run / "monthly").iterdir() if p.is_dir())


cases = []
for ale in ("zstar", "linfs"):
    f_fc, f_pc = ROOT / f"fortran/{ale}_fc_1m", ROOT / f"fortran/{ale}_pc_1m"
    if not (f_fc / "temp.fesom.1958.nc").exists() or not (f_pc / "temp.fesom.1958.nc").exists():
        continue
    keep = level_mask(f_pc)
    j = lambda c, n: ROOT / f"jax/{ale}_{c}_{n}"
    if (j("fc", "1d") / "monthly").is_dir() and (j("pc", "1d") / "monthly").is_dir():
        cases.append((f"{ale} day 1", keep, fortran_fields(f_fc, slice(0, 1)), fortran_fields(f_pc, slice(0, 1)),
                      jax_fields(months(j("fc", "1d"))), jax_fields(months(j("pc", "1d"))), "day1"))
    mlen = "1y" if ale == "zstar" else "1m"
    if (j("fc", mlen) / "monthly").is_dir() and (j("pc", mlen) / "monthly").is_dir():
        cases.append((f"{ale} month 1", keep, fortran_fields(f_fc, slice(0, 31)), fortran_fields(f_pc, slice(0, 31)),
                      jax_fields(months(j("fc", mlen))[:1]), jax_fields(months(j("pc", mlen))[:1]), "month1"))
    if ale == "zstar" and (ROOT / "fortran/zstar_pc_1y/temp.fesom.1958.nc").exists() \
            and (j("fc", "1y") / "monthly").is_dir() and (j("pc", "1y") / "monthly").is_dir():
        mf, mp = months(j("fc", "1y")), months(j("pc", "1y"))
        if len(mf) == 12 and len(mp) == 12:
            cases.append(("zstar year 1958", keep, fortran_fields(F_FC_YEAR, slice(0, 12)),
                          fortran_fields(ROOT / "fortran/zstar_pc_1y", slice(0, 12)),
                          jax_fields(mf), jax_fields(mp), "year"))

ok_all = True
report = []
for name, keep, Ffc, Fpc, Jfc, Jpc, kind in cases:
    print(f"\n=== {name} (nodes {keep.sum()} of {keep.size}) ===")
    print(f"{'field':14s} {'rms dF':>10s} {'rms dJ':>10s} {'corr':>7s} {'amp':>6s} {'rms Jpc-Fpc':>12s} "
          f"{'rms Jfc-Ffc':>12s}  verdict")
    for fld in ("bottom_T", "bottom_speed", "ssh"):
        dF, dJ = Fpc[fld] - Ffc[fld], Jpc[fld] - Jfc[fld]
        r = dict(case=name, field=fld, rms_dF=rms(dF, keep), rms_dJ=rms(dJ, keep), corr=corr(dJ, dF, keep),
                 amp=std(dJ, keep) / std(dF, keep), rms_pc=rms(Jpc[fld] - Fpc[fld], keep),
                 rms_fc=rms(Jfc[fld] - Ffc[fld], keep))
        fails = []
        if kind in ("day1", "month1"):
            if r["rms_pc"] > 1.2 * r["rms_fc"]:
                fails.append("pc>1.2fc")
            if max(r["rms_pc"], r["rms_fc"]) > 0.5 * r["rms_dF"]:
                fails.append("model diff>0.5 dF")
        if kind in ("day1", "year"):
            if r["corr"] < (0.9 if kind == "day1" else 0.7):
                fails.append("corr")
            if not 0.7 <= r["amp"] <= 1.4:
                fails.append("amp")
        r["fails"] = fails
        ok_all &= not fails
        report.append(r)
        print(f"{fld:14s} {r['rms_dF']:10.3e} {r['rms_dJ']:10.3e} {r['corr']:7.3f} {r['amp']:6.2f} "
              f"{r['rms_pc']:12.3e} {r['rms_fc']:12.3e}  {'PASS' if not fails else 'MISS ' + ','.join(fails)}")
(ROOT / "g2_report.json").write_text(json.dumps(report, indent=1))
print(f"\ncases: {[c[0] for c in cases]}")
print("G2_RESPONSE_OK" if ok_all and cases else "G2_RESPONSE_MISS")
