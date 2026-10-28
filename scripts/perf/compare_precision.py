#!/usr/bin/env python
"""Single-precision sub-plan, Task 9: compare the runs of a triplet (``scripts/perf/sp_triplet.sbatch``).

    compare_precision.py <triplet dir> [--mesh data/mesh_core2] [--json out.json]

For each pair (sp − dp, dpic32 − dp) at the final restart: volume-mean T and S (the carry counted as content),
area-mean SST and SSS, global-mean SSH, sea-ice area and volume NH/SH, pointwise area-weighted rms and max of SST, SSS,
SSH, a_ice, m_ice. From the ``[budget]`` lines of each run: the change of the ocean volume and of the total salt
(ocean + the salt held by the sea ice) between the first and the last chunk — the residual analogous to FESOM2's
``fesom_conservation`` CONS lines (salt restoring is balanced to a zero global integral, water fluxes too). The
thresholds T1/T2 of docs/plans/20261020-core2-single-precision.md are applied by :func:`verdict`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_restart(path) -> dict:
    import zarr
    g = zarr.open_group(str(path), mode="r")
    return {k: np.asarray(g[k][:]) for k in g.array_keys()}


def state_stats(st: dict, av: np.ndarray, lat: np.ndarray) -> dict:
    """Global quantities of one canonical (global node order) restart; ``av`` = areasvol [nod2D, nl]."""
    m = av > 0
    h = st["hnode"]
    V = np.where(m, h * av, 0.0)
    out = {"volume": V.sum()}
    for X, C in (("T", "carry_T"), ("S", "carry_S")):
        c = (st[X] * V).sum() + (np.where(m, st[C] * av, 0.0).sum() if C in st else 0.0)
        out[f"{X}_vol_mean"] = c / V.sum()
    a0 = av[:, 0]
    out["SST_area_mean"] = (st["T"][:, 0] * a0).sum() / a0.sum()
    out["SSS_area_mean"] = (st["S"][:, 0] * a0).sum() / a0.sum()
    out["ssh_global_mean"] = (st["eta_n"] * a0).sum() / a0.sum()
    nh, sh = lat > 0, lat < 0
    out["ice_area_NH"] = (st["a_ice"] * a0)[nh].sum() / 1e12
    out["ice_area_SH"] = (st["a_ice"] * a0)[sh].sum() / 1e12
    out["ice_vol_NH"] = (st["m_ice"] * a0)[nh].sum() / 1e12
    out["ice_vol_SH"] = (st["m_ice"] * a0)[sh].sum() / 1e12
    return out


def pointwise(a: dict, b: dict, av: np.ndarray) -> dict:
    a0 = av[:, 0]
    w = a0 / a0.sum()
    out = {}
    for name, get in (("SST", lambda s: s["T"][:, 0]), ("SSS", lambda s: s["S"][:, 0]),
                      ("SSH", lambda s: s["eta_n"]), ("a_ice", lambda s: s["a_ice"]),
                      ("m_ice", lambda s: s["m_ice"])):
        d = np.where(a0 > 0, get(b) - get(a), 0.0)
        out[name] = {"max": float(np.max(np.abs(d))), "rms": float(np.sqrt((w * d * d).sum()))}
    return out


def parse_budget(log) -> list[dict]:
    rows = []
    for line in Path(log).read_text().splitlines():
        if line.startswith("[budget]"):
            step = int(re.search(r"step (\d+)", line).group(1))
            d = {k: float(v) for k, v in re.findall(r"(\w+)=([-+0-9.eE]+)", line)}
            d["step"] = step
            rows.append(d)
    return rows


def budget_drift(rows: list[dict]) -> dict:
    """First → last chunk: change of ocean volume [m³] and of total salt (ocean + ice) [psu m³]."""
    a, b = rows[0], rows[-1]
    tot = lambda r: r["S_oce"] + r.get("S_ice", 0.0)       # noqa: E731
    return {"steps": b["step"] - a["step"], "dV": b["V"] - a["V"], "dS_total": tot(b) - tot(a),
            "V": b["V"], "deta_mean": b["eta_mean"] - a["eta_mean"]}


def compare(triplet: Path, mesh_dir: Path) -> dict:
    av = np.load(mesh_dir / "areasvol.npy")
    lat = np.load(mesh_dir / "geo_coord_nod2D.npy")
    lat = lat[:, 1] if lat.shape[-1] == 2 else lat[1]
    runs = {r: load_restart(triplet / r / "restart") for r in ("dp", "dpic32", "sp")
            if (triplet / r / "restart").exists()}
    stats = {r: state_stats(s, av, lat) for r, s in runs.items()}
    out = {"stats": stats, "pairs": {}, "budget": {}}
    for r in ("dpic32", "sp"):
        if r in stats and "dp" in stats:
            out["pairs"][f"{r}-dp"] = {"delta": {k: stats[r][k] - stats["dp"][k] for k in stats["dp"]},
                                       "pointwise": pointwise(runs["dp"], runs[r], av)}
    for r in runs:
        rows = parse_budget(triplet / r / "run.log")
        if len(rows) > 1:
            out["budget"][r] = budget_drift(rows)
    return out


def verdict(res: dict, *, days: float) -> dict:
    """The plan's thresholds (T1 for <= 31 days, T2 for a year); scaled linearly in between for the budgets."""
    year = days >= 300
    sp = res["pairs"].get("sp-dp")
    env = res["pairs"].get("dpic32-dp")
    if sp is None:
        return {}
    d = sp["delta"]
    out = {}
    out["S_vol_mean"] = (abs(d["S_vol_mean"]), 1.5e-7 if year else 2e-8)
    t_env = 3 * abs(env["delta"]["T_vol_mean"]) if env else 0.0
    out["T_vol_mean"] = (abs(d["T_vol_mean"]), max(1.3e-6 if year else 1e-6, t_env))
    if year:
        out["ssh_global_mean"] = (abs(d["ssh_global_mean"]), 5e-7)
    frac = days / 365.0
    b = res["budget"]
    if "sp" in b:
        out["budget_salt"] = (abs(b["sp"]["dS_total"] - b.get("dp", {}).get("dS_total", 0.0)), 1.3e10 * frac)
        out["budget_volume"] = (abs(b["sp"]["dV"] - b.get("dp", {}).get("dV", 0.0)), 3.7e8 * frac)
    if year and env:
        for k, v in sp["pointwise"].items():
            out[f"rms_{k}"] = (v["rms"], 3 * env["pointwise"][k]["rms"])
    # the budget entries are a DIAGNOSTIC until the user decides the criterion (the State-based budget does not close
    # in double precision either; sub-plan, "Budget criterion")
    return {k: {"value": float(v), "bound": float(bd), "pass": bool(v <= bd),
                "diagnostic": k.startswith("budget_")} for k, (v, bd) in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("triplet", type=Path)
    ap.add_argument("--mesh", type=Path, default=ROOT / "data" / "mesh_core2")
    ap.add_argument("--days", type=float, required=True, help="simulated days (selects T1 or T2)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    res = compare(a.triplet, a.mesh)
    res["verdict"] = verdict(res, days=a.days)
    for pair, p in res["pairs"].items():
        print(f"== {pair}")
        for k, v in p["delta"].items():
            print(f"  {k:18s} {v:+.3e}")
        for k, v in p["pointwise"].items():
            print(f"  pointwise {k:6s} max {v['max']:.3e} rms {v['rms']:.3e}")
    for r, b in res["budget"].items():
        print(f"== budget {r}: steps {b['steps']} dV {b['dV']:+.3e} m3  dS_total {b['dS_total']:+.3e} psu m3  "
              f"deta_mean {b['deta_mean']:+.3e} m")
    for k, v in res["verdict"].items():
        tag = "diagnostic" if v.get("diagnostic") else ("PASS" if v["pass"] else "FAIL")
        print(f"VERDICT {k:18s} {v['value']:.3e} <= {v['bound']:.3e}  {tag}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
