#!/usr/bin/env python
"""Multi-year sea-ice score of a model run against the monthly super-observations.

    python scripts/icecal/score.py --run <dir> --obs-dir <dir> --years 1980-2019 --out <new dir>
                                   [--mesh-dir data/mesh_core2] [--kind auto|fortran|jax]

Scores a Fortran FESOM2 run directory or a JAX fields directory against the ``obs_<YYYY>.npz``
files of ``scripts/icecal/prep_obs.py`` and writes ``<out>/score.json`` and three line plots.
``<out>`` must not exist; nothing is overwritten.

Model input (monthly means on the mesh nodes, mesh node order):

* **Fortran** (``--kind fortran``): ``a_ice.fesom.<YYYY>.nc`` and ``m_ice.fesom.<YYYY>.nc``, variable
  ``(time, nod2)``, one record per month. Record i is calendar month i + 1 (the convention of the
  FESOM monthly output; the record's ``time`` stamp, the start of the month's last step, must fall
  in that month and year, else the file is refused). FESOM writes the netCDF fill value
  (9.96921e36) at ICE-FREE nodes: the reader turns it into 0 ice (``a_ice`` 0, ``m_ice`` 0 m); it
  is never averaged or dropped. A year with fewer than 12 records is scored on the months it has.
* **JAX** (``--kind jax``): ``fields_<YYYY>.npz`` files, or one ``fields.npz``, each with
  ``C_mean`` [12, nod2D] (monthly mean ``a_ice``), ``H_mean`` [12, nod2D] (monthly mean ``m_ice``,
  the effective thickness), ``n_mon`` [12] (model steps in each month; a month is scored iff
  ``n_mon > 0``) and ``year`` (int). This is the format the calibration driver writes.

What is computed (areas in 10^6 km^2; extent and the ice edge use concentration > 15 %):

* J per year and term (``seaice_cost.cost_terms_np``: J_C_NH, J_C_SH, J_H_NH, J_data, per month,
  and the weight sums N_* of all scored and Nact_* of the ACTIVE super-observations, obs > 0 or
  model > 0, which the chi^2/N_eff scaling counts; totals over the years in ``total``);
  months without model data or without observations are skipped (every-other-day OSI-SAF years
  and the Dec 1987 - Jan 1988 gap are handled by the per-year observation files).
* Hemispheric monthly area and extent: the model on its node areas over all wet nodes of the
  hemisphere (``area_model``, ``extent_model``); the observations on their 625 km^2 cells,
  ``sum C_obs * nC`` and ``sum nE`` (cells above 15 %; files without ``nE`` fall back to
  ``nC * [C_obs > 0.15]``). Same-mask companions: the model sampled in observation space,
  ``sum Cbar * nC`` and ``sum nC * [Cbar > 0.15]`` over the observed nodes, i.e. the observed
  cells carrying the value of their node (``*_model_obsspace``).
* The integrated ice-edge error per month and hemisphere in node space: each observed node's cells
  carry the model's ice/no-ice verdict of the node; IIEE = (cells where the model says ice and
  the observation not) + (the reverse), x 625 km^2 (``iiee``, with its ``over``/``under`` parts).
* NH March and September extent series (model node-area extent vs observed cell extent): trend
  per decade (least squares) of each, RMSE between them.
* SH seasonal cycle: monthly climatologies of extent and area over the scored years, RMSE.
* CCI winters (October Y - April Y+1): J_H and the mean thickness bias (model - obs, weighted by
  the observed area).
* Per-decade means (1980s, ...) of each term and of NH = J_C_NH + J_H_NH and SH = J_C_SH.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fesom_jax import obs_seaice as O  # noqa: E402
from fesom_jax import seaice_cost as S  # noqa: E402

THRESH = O.EXTENT_THRESHOLD
MKM2 = 1e12                    # m^2 per 10^6 km^2
CCI_WINTER_MONTHS = O.CCI_MONTHS


def parse_years(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def mesh_nodes(mesh_dir):
    """Node lon/lat [deg], wet-surface mask and surface node area [m^2] (``area.npy`` level 0)."""
    lon, lat, wet = O.mesh_surface_nodes(mesh_dir)
    area = np.load(Path(mesh_dir) / "area.npy")
    area = np.asarray(area[:, 0] if area.ndim == 2 else area, dtype=np.float64)
    return lon, lat, wet, area


# --- model readers ---------------------------------------------------------------------------
def read_fortran_var(path, var: str, year: int, nod2D: int):
    """``[12, nod2D]`` monthly field of one Fortran output year (NaN rows for missing months) and
    the present-month mask. Fill (ice-free) -> 0."""
    with netCDF4.Dataset(path) as ds:
        v = ds[var]
        v.set_auto_maskandscale(False)
        if v.dimensions != ("time", "nod2") or v.shape[1] != nod2D:
            raise ValueError(f"{path}: {var}{v.dimensions}{v.shape}, expected (time, nod2={nod2D})")
        nrec = v.shape[0]
        if nrec > 12:
            raise ValueError(f"{path}: {nrec} records, expected at most 12 monthly records")
        data = np.asarray(v[:], dtype=np.float64)
        fill = getattr(v, "_FillValue", None)
        if "time" in ds.variables:
            t = ds["time"]
            dates = netCDF4.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
            for i, d in enumerate(np.atleast_1d(dates)):
                if (d.year, d.month) != (year, i + 1):
                    raise ValueError(f"{path}: record {i} is stamped {d}, expected month "
                                     f"{year}-{i + 1:02d}")
    ice_free = np.abs(data) > 1e30
    if fill is not None:
        ice_free |= data == np.float64(fill)
    if np.isnan(data).any():
        raise ValueError(f"{path}: NaN in {var} (expected the fill value at ice-free nodes)")
    data = np.where(ice_free, 0.0, data)
    out = np.full((12, nod2D), np.nan)
    present = np.zeros(12, dtype=bool)
    for i in range(nrec):              # record i -> calendar month i + 1
        out[i] = data[i]
        present[i] = True
    return out, present


def read_fortran_year(run_dir, year: int, nod2D: int):
    pa = Path(run_dir) / f"a_ice.fesom.{year:04d}.nc"
    pm = Path(run_dir) / f"m_ice.fesom.{year:04d}.nc"
    if not pa.exists() and not pm.exists():
        return None
    if not (pa.exists() and pm.exists()):
        raise FileNotFoundError(f"{run_dir}: only one of a_ice/m_ice exists for {year}")
    C, pc = read_fortran_var(pa, "a_ice", year, nod2D)
    H, ph = read_fortran_var(pm, "m_ice", year, nod2D)
    if not np.array_equal(pc, ph):
        raise ValueError(f"{run_dir} {year}: a_ice and m_ice hold different months")
    return C, H, pc


def read_jax_fields(run_dir, nod2D: int) -> dict:
    """``{year: (C, H, present)}`` from ``fields_<YYYY>.npz`` / ``fields.npz``."""
    files = sorted(glob.glob(os.path.join(run_dir, "fields_[0-9][0-9][0-9][0-9].npz")))
    if os.path.exists(os.path.join(run_dir, "fields.npz")):
        files.append(os.path.join(run_dir, "fields.npz"))
    out = {}
    for f in files:
        with np.load(f) as z:
            for k in ("C_mean", "H_mean", "n_mon", "year"):
                if k not in z.files:
                    raise KeyError(f"{f}: missing key {k!r} (expected C_mean, H_mean, n_mon, year)")
            C = np.asarray(z["C_mean"], dtype=np.float64)
            H = np.asarray(z["H_mean"], dtype=np.float64)
            n = np.asarray(z["n_mon"])
            y = int(z["year"])
        m = re.search(r"fields_(\d{4})\.npz$", f)
        if m and int(m.group(1)) != y:
            raise ValueError(f"{f}: year key {y} differs from the file name")
        if C.shape != (12, nod2D) or H.shape != (12, nod2D) or n.shape != (12,):
            raise ValueError(f"{f}: shapes {C.shape} {H.shape} {n.shape}, expected [12, {nod2D}]")
        if y in out:
            raise ValueError(f"{run_dir}: two field files for {y}")
        out[y] = (C, H, n > 0)
    return out


def detect_kind(run_dir) -> str:
    if glob.glob(os.path.join(run_dir, "a_ice.fesom.*.nc")):
        return "fortran"
    if glob.glob(os.path.join(run_dir, "fields*.npz")):
        return "jax"
    raise FileNotFoundError(f"{run_dir}: neither a_ice.fesom.*.nc nor fields*.npz")


def load_model(run_dir, kind: str, years, nod2D: int) -> dict:
    if kind == "auto":
        kind = detect_kind(run_dir)
    if kind == "jax":
        allf = read_jax_fields(run_dir, nod2D)
        return {y: allf[y] for y in years if y in allf}
    out = {}
    for y in years:
        r = read_fortran_year(run_dir, y, nod2D)
        if r is not None:
            out[y] = r
    return out


# --- diagnostics -----------------------------------------------------------------------------
def model_area_extent(C_row, area, hem):
    a = np.clip(C_row, 0.0, 1.0)
    return float(np.sum(a[hem] * area[hem])), float(np.sum(area[hem & (C_row > THRESH)]))


def obs_counts(obs, m):
    """``(nC, nE, valid)`` of month m (``nE`` from the file, or the node-level fallback)."""
    nC = np.asarray(obs["nC"][m], dtype=np.int64)
    valid = nC > 0
    if "nE" in obs:
        nE = np.asarray(obs["nE"][m], dtype=np.int64)
    else:
        nE = np.where(valid & (np.nan_to_num(obs["C_obs"][m]) > THRESH), nC, 0)
    return nC, nE, valid


def month_diagnostics(C_row, obs, m, area, hem) -> dict:
    nC, nE, valid = obs_counts(obs, m)
    v = valid & hem
    a_mod, e_mod = model_area_extent(C_row, area, hem)
    out = {"area_model": a_mod / MKM2, "extent_model": e_mod / MKM2}
    if not v.any():                     # no observation in this month and hemisphere
        for k in ("area_obs", "extent_obs", "area_model_obsspace", "extent_model_obsspace",
                  "iiee", "iiee_over", "iiee_under"):
            out[k] = None
        return out
    cell = O.CELL_AREA_M2 / MKM2
    icy = C_row > THRESH
    out["area_obs"] = float(np.sum(np.nan_to_num(obs["C_obs"][m])[v] * nC[v])) * cell
    out["extent_obs"] = float(np.sum(nE[v])) * cell
    out["area_model_obsspace"] = float(np.sum(np.clip(C_row, 0, 1)[v] * nC[v])) * cell
    out["extent_model_obsspace"] = float(np.sum(nC[v & icy])) * cell
    over = float(np.sum((nC - nE)[v & icy])) * cell
    under = float(np.sum(nE[v & ~icy])) * cell
    out.update(iiee=over + under, iiee_over=over, iiee_under=under)
    return out


def trend_per_decade(years, values):
    x = np.asarray(years, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(x[ok], y[ok], 1)[0] * 10.0)


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else None


def _f(x):
    return np.nan if x is None else x


def score(model: dict, obs_dir, years, area, lat) -> dict:
    nod2D = area.size
    nh = O.northern_nodes(lat)
    res = {"years": list(years), "scored_years": [], "missing_model_years": [],
           "missing_obs_years": [], "J": {}, "monthly": [], "cci_winters": []}
    winters: dict = {}
    for y in years:
        if y not in model:
            res["missing_model_years"].append(y)
            continue
        C, H, present = model[y]
        p = Path(obs_dir) / f"obs_{y:04d}.npz"
        obs = None
        if p.exists():
            with np.load(p) as z:
                obs = {k: z[k] for k in z.files}
            O.check_obs(obs, nod2D)
            if not np.array_equal(obs["nh"], nh):
                raise ValueError(f"{p}: node hemisphere mask differs from the mesh's")
        else:
            res["missing_obs_years"].append(y)
        if obs is not None:
            ob = S.cost_obs(obs)
            terms = S.cost_terms_np(C, H, ob, months=np.flatnonzero(present))
            res["J"][str(y)] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in terms.items()}
            res["scored_years"].append(y)
            for m in range(12):
                if not present[m] or (m + 1) not in CCI_WINTER_MONTHS:
                    continue
                v = ob.validH[m]
                if not v.any():
                    continue
                key = f"{y}/{y + 1}" if m + 1 >= 10 else f"{y - 1}/{y}"
                w = winters.setdefault(key, {"J_H": 0.0, "sw": 0.0, "swb": 0.0, "months": 0})
                w["J_H"] += float(terms["J_H_NH_month"][m])
                w["sw"] += float(ob.wH[m][v].sum())
                w["swb"] += float(np.sum(ob.wH[m][v] * (H[m][v] - ob.H[m][v])))
                w["months"] += 1
        for m in range(12):
            if not present[m]:
                continue
            row = {"year": y, "month": m + 1}
            for hem_name, hem in (("nh", nh), ("sh", ~nh)):
                if obs is None:
                    a_mod, e_mod = model_area_extent(C[m], area, hem)
                    row[hem_name] = {"area_model": a_mod / MKM2, "extent_model": e_mod / MKM2}
                else:
                    row[hem_name] = month_diagnostics(C[m], obs, m, area, hem)
            res["monthly"].append(row)
    for key in sorted(winters):
        w = winters[key]
        res["cci_winters"].append({"winter": key, "J_H": w["J_H"], "n_months": w["months"],
                                   "mean_bias_m": w["swb"] / w["sw"] if w["sw"] > 0 else None})

    def series(month, hem, field):
        out = {}
        for r in res["monthly"]:
            if r["month"] == month:
                out[r["year"]] = _f(r[hem].get(field))
        return out

    for name, month in (("nh_march_extent", 3), ("nh_september_extent", 9)):
        mod = series(month, "nh", "extent_model")
        obs_s = series(month, "nh", "extent_obs")
        ys = sorted(mod)
        mv = [mod[y] for y in ys]
        ov = [obs_s.get(y, np.nan) for y in ys]
        res[name] = {"years": ys, "model": mv, "obs": [None if not np.isfinite(v) else v for v in ov],
                     "trend_model_per_decade": trend_per_decade(ys, mv),
                     "trend_obs_per_decade": trend_per_decade(ys, ov), "rmse": rmse(mv, ov)}

    clim = {}
    for field in ("extent", "area"):
        mod_c, obs_c = [], []
        for month in range(1, 13):
            pairs = [(_f(r["sh"].get(f"{field}_model")), _f(r["sh"].get(f"{field}_obs")))
                     for r in res["monthly"] if r["month"] == month]
            pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
            mod_c.append(float(np.mean([a for a, _ in pairs])) if pairs else None)
            obs_c.append(float(np.mean([b for _, b in pairs])) if pairs else None)
        clim[f"{field}_model_clim"] = mod_c
        clim[f"{field}_obs_clim"] = obs_c
        clim[f"rmse_{field}"] = rmse([_f(v) for v in mod_c], [_f(v) for v in obs_c])
    res["sh_seasonal_cycle"] = clim

    dec: dict = {}
    for ys, t in res["J"].items():
        d = dec.setdefault(f"{int(ys) // 10 * 10}s", {k: [] for k in S.TERMS + ("J_data",)})
        for k in d:
            d[k].append(t[k])
    res["decades"] = {}
    for k, d in sorted(dec.items()):
        mean = {t: float(np.mean(v)) for t, v in d.items()}
        mean["NH"] = mean["J_C_NH"] + mean["J_H_NH"]
        mean["SH"] = mean["J_C_SH"]
        mean["n_years"] = len(d["J_data"])
        res["decades"][k] = mean
    sums = S.TERMS + ("J_data",) + tuple(f"{p}_{t}" for p in ("N", "Nact")
                                          for t in ("C_NH", "C_SH", "H_NH"))
    res["total"] = {t: float(sum(v[t] for v in res["J"].values())) for t in sums}
    return res


# --- output ----------------------------------------------------------------------------------
def figures(res: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    def save(fig, name):
        with open(out_dir / name, "xb") as f:
            fig.savefig(f, format="png", dpi=120)
        plt.close(fig)
        written.append(name)

    t = [r["year"] + (r["month"] - 0.5) / 12 for r in res["monthly"]]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, hem in zip(axs, ("nh", "sh")):
        ax.plot(t, [_f(r[hem].get("extent_model")) for r in res["monthly"]], label="model extent")
        ax.plot(t, [_f(r[hem].get("extent_obs")) for r in res["monthly"]], label="obs extent")
        ax.plot(t, [_f(r[hem].get("area_model")) for r in res["monthly"]], "--", label="model area")
        ax.plot(t, [_f(r[hem].get("area_obs")) for r in res["monthly"]], "--", label="obs area")
        ax.set_ylabel(f"{hem.upper()} [10$^6$ km$^2$]")
        ax.legend(fontsize=8, ncol=4)
    axs[-1].set_xlabel("year")
    save(fig, "extent_area_monthly.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, lab in (("nh_march_extent", "March"), ("nh_september_extent", "September")):
        s = res[name]
        ax.plot(s["years"], s["model"], "-o", ms=3, label=f"model {lab}")
        ax.plot(s["years"], [_f(v) for v in s["obs"]], "-s", ms=3, label=f"obs {lab}")
    ax.set_ylabel("NH extent [10$^6$ km$^2$]")
    ax.legend(fontsize=8)
    save(fig, "nh_mar_sep_extent.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ys = [int(y) for y in res["J"]]
    for term in S.TERMS:
        ax.plot(ys, [res["J"][str(y)][term] for y in ys], "-o", ms=3, label=term)
    ax.set_ylabel("J per year")
    ax.legend(fontsize=8)
    save(fig, "J_per_year.png")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--years", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mesh-dir", type=Path, default=ROOT / "data" / "mesh_core2")
    ap.add_argument("--kind", choices=("auto", "fortran", "jax"), default="auto")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)

    if a.out.exists():
        print(f"refusing to write into an existing directory: {a.out}", file=sys.stderr)
        return 3
    years = parse_years(a.years)
    lon, lat, wet, area = mesh_nodes(a.mesh_dir)
    kind = detect_kind(a.run) if a.kind == "auto" else a.kind
    model = load_model(a.run, kind, years, area.size)
    if not model:
        print(f"no model data for {a.years} in {a.run}", file=sys.stderr)
        return 2
    res = score(model, a.obs_dir, years, area, lat)
    res["provenance"] = {"run": str(a.run), "kind": kind, "obs_dir": str(a.obs_dir),
                         "mesh_dir": str(a.mesh_dir), "argv": sys.argv if argv is None else argv,
                         "created": datetime.datetime.now().isoformat(timespec="seconds"),
                         "extent_threshold": THRESH, "units_area": "1e6 km^2"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(a.out)
    if not a.no_figures:
        res["figures"] = figures(res, a.out)
    with open(a.out / "score.json", "x") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}/score.json: {len(res['scored_years'])} scored years, "
          f"J_data total {res['total']['J_data']:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
