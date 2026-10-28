#!/usr/bin/env python
"""Tables of the TKE-adjoint diagnosis runs (``scripts/adjoint/tke_diag.py``) from their
``results.jsonl`` rows and screen map files. Host-side numpy only (no model, no device).

    python scripts/adjoint/tke_diag_summary.py RUN_DIR [RUN_DIR ...] [--md out.md] [--json out.json]

Growth of a screen is reported as the OVERALL per-step rate over the window,
``(trace[-1] / trace[0]) ** (1 / n_steps)``, next to the worst stretches — the worst 4-chunk
stretch and the worst lag-day (per-step rates) — and the largest single chunk, never the chunk
median alone: on these windows the median sits near 1.02 while the norm grows by 1e3-1e6 in
bursts. Repeats of the same window and setting are grouped; every number is shown as the median
over repeats with the [min, max] range (the repeat spread).

The trace is the Euclidean norm of the whole State cotangent at each chunk boundary, seed
first (so index ``i`` is a lag of ``i · chunk_steps`` steps before the window end). Per-field
rates use ``field_trace``; a field whose cotangent is zero at the seed (``tke`` for a
temperature cost) is measured from its first non-zero boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
FIELDS = ("tke", "T", "S", "uv", "eta_n")
DT = 1800.0


# ---------------------------------------------------------------------------
# loading

_NONFINITE = {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan"),
              "Infinity": float("inf"), "-Infinity": float("-inf"), "NaN": float("nan")}


def _numify(x):
    """The driver writes non-finite floats as the strings "inf" / "-inf" / "nan"; read them back
    as floats (recursively)."""
    if isinstance(x, str):
        return _NONFINITE.get(x, x)
    if isinstance(x, list):
        return [_numify(v) for v in x]
    if isinstance(x, dict):
        return {k: _numify(v) for k, v in x.items()}
    return x


def load_rows(dirs) -> list:
    rows = []
    for d in dirs:
        d = Path(d)
        files = [d] if d.is_file() else sorted(d.rglob("results.jsonl"))
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    r = _numify(json.loads(line))
                    r["_file"] = str(f)
                    rows.append(r)
    return rows


def mesh_lonlat(mesh_dir=MESH_DIR):
    """``(lon, lat)`` of the nodes in degrees (lon in [-180, 180])."""
    g = np.load(Path(mesh_dir) / "geo_coord_nod2D.npy")
    if g.shape[0] == 2 and g.shape[1] != 2:
        g = g.T
    lon, lat = np.degrees(g[:, 0]), np.degrees(g[:, 1])
    return np.where(lon > 180.0, lon - 360.0, lon), lat


# ---------------------------------------------------------------------------
# growth statistics of one trace

def _rates(tr, k: int, w: int):
    """Per-step rates of all stretches of ``w`` chunks (``k`` steps each)."""
    out = []
    for i in range(len(tr) - w):
        a, b = tr[i], tr[i + w]
        out.append((b / a) ** (1.0 / (k * w)) if a > 0 and b > 0 and math.isfinite(b)
                   else float("nan"))
    return out


def trace_stats(tr, k: int) -> dict:
    """Growth statistics of a seed-first norm trace with ``k`` steps per chunk."""
    tr = [float(x) for x in tr]
    i0 = next((i for i, x in enumerate(tr) if x > 0), None)
    if i0 is None or len(tr) - i0 < 2:
        return {"overall": float("nan"), "total": float("nan"), "n_steps": 0}
    tr = tr[i0:]
    n = (len(tr) - 1) * k
    per_chunk = _rates(tr, k, 1)
    fin = [r for r in per_chunk if np.isfinite(r)]
    day = max(1, int(round(86400.0 / DT / k)))           # chunks per day
    per_day = [(tr[min(i + day, len(tr) - 1)] / tr[i]) ** (1.0 / (k * (min(i + day, len(tr) - 1) - i)))
               for i in range(0, len(tr) - 1, day) if tr[i] > 0]
    w4 = _rates(tr, k, min(4, len(tr) - 1))
    wd = _rates(tr, k, min(day, len(tr) - 1))
    return {"overall": (tr[-1] / tr[0]) ** (1.0 / n) if tr[-1] > 0 else float("nan"),
            "total": tr[-1] / tr[0], "n_steps": n, "first_nonzero": i0,
            "median": float(np.median(fin)) if fin else float("nan"),
            "max_chunk": float(np.max(fin)) if fin else float("nan"),
            "worst4": float(np.nanmax(w4)) if w4 else float("nan"),
            "worst_day": float(np.nanmax(wd)) if wd else float("nan"),
            "worst_day_lag": (float(np.nanargmax(wd)) * k * DT / 86400.0) if wd else float("nan"),
            "per_day": per_day, "per_chunk": per_chunk}


def screen_stats(row) -> dict:
    k = int(row["chunk_steps"])
    s = trace_stats(row["trace"], k)
    s["status"] = row.get("status")
    if row.get("status") == "overflow":
        s["overflow"] = row.get("overflow")
    ft = [f for f in (row.get("field_trace") or []) if f is not None]
    s["fields"] = {}
    for f in FIELDS:
        if ft and f in ft[0]:
            s["fields"][f] = trace_stats([x[f] for x in ft], k)
    return s


def divergence_lag(traces, k: int, rel: float = 1e-2):
    """First lag (days) at which two repeats' traces differ by more than ``rel`` (relative)."""
    if len(traces) < 2:
        return None
    n = min(len(t) for t in traces)
    a = np.asarray([t[:n] for t in traces], float)
    ref = np.maximum(np.abs(a).max(axis=0), 1e-300)
    d = (a.max(axis=0) - a.min(axis=0)) / ref
    idx = np.nonzero(d > rel)[0]
    return float(idx[0] * k * DT / 86400.0) if idx.size else float("inf")


# ---------------------------------------------------------------------------
# tables

def _fmt(x, p=4):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return str(x)
    a = abs(x)
    if a != 0 and (a >= 1e4 or a < 1e-3):
        return f"{x:.{max(p - 2, 1)}e}"
    return f"{x:.{p}g}"


def _mrange(vals, p=4):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    if len(v) == 1:
        return _fmt(v[0], p)
    return f"{_fmt(float(np.median(v)), p)} [{_fmt(min(v), p)}, {_fmt(max(v), p)}]"


def _setting(row):
    s = row.get("setting") or "none"
    return "exact" if s == "none" else s


def screen_groups(rows) -> dict:
    g = defaultdict(list)
    for r in rows:
        if r.get("action") == "tke_screen":
            g[(r["start"], r["n_steps"], r["chunk_steps"], _setting(r))].append(r)
    return g


def screen_table(rows) -> tuple[str, list]:
    out = ["| start | days | setting | n | overall /step | total | worst 4-chunk | worst day "
           "| max chunk | median | tke overall | T overall | uv overall | repeats diverge at lag (d) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    recs = []
    for (start, n, k, st), rs in sorted(screen_groups(rows).items()):
        ss = [screen_stats(r) for r in rs]
        dl = divergence_lag([r["trace"] for r in rs], k)
        rec = {"start": start, "days": n * DT / 86400.0, "setting": st, "n": len(rs),
               "stats": ss, "diverge_lag_days": dl, "slurm_jobs": sorted({r.get("slurm_job") for r in rs})}
        recs.append(rec)

        def col(key, sub=None):
            if sub is None:
                return _mrange([s.get(key) for s in ss])
            return _mrange([s["fields"].get(sub, {}).get(key) for s in ss])
        out.append(f"| {start[:10]} | {n * DT / 86400.0:g} | {st} | {len(rs)} | {col('overall')} | "
                   f"{col('total')} | {col('worst4')} | {col('worst_day')} | "
                   f"{col('max_chunk')} | {col('median')} | {col('overall', 'tke')} | "
                   f"{col('overall', 'T')} | {col('overall', 'uv')} | "
                   f"{_fmt(dl) if dl is not None else '-'} |")
    return "\n".join(out), recs


def per_day_table(rows) -> str:
    """Per-lag-day rates (per step) of every screen repeat (seed first: day 1 = the last
    model day of the window)."""
    out = []
    for (start, n, k, st), rs in sorted(screen_groups(rows).items()):
        for r in rs:
            s = screen_stats(r)
            pd = " ".join(_fmt(x, 3) for x in s.get("per_day", []))
            out.append(f"- {start[:10]} {n * DT / 86400.0:g} d {st} r{r.get('repeat', 0)} "
                       f"(job {r.get('slurm_job')}): {pd}")
    return "\n".join(out)


def map_sites(npz_path, lonlat, field: str = "tke", top: int = 8) -> list:
    """Per map (lag day): share of the field's cotangent norm² in the top 0.1 % / 1 % of
    nodes, and the top nodes (lon, lat, value)."""
    z = np.load(npz_path)
    if field not in z.files or z[field].ndim != 2:
        return []
    lon, lat = lonlat
    a = np.asarray(z[field], float)
    day = z["day"] if "day" in z.files else np.arange(a.shape[0])
    out = []
    for i in range(a.shape[0]):
        m = a[i] ** 2
        tot = m.sum()
        if tot <= 0:
            out.append({"day": float(day[i]), "total": 0.0})
            continue
        srt = np.sort(m)[::-1]
        n1, n01 = max(1, m.size // 100), max(1, m.size // 1000)
        idx = np.argsort(-m)[:top]
        out.append({"day": float(day[i]), "norm": float(np.sqrt(tot)),
                    "share_top1pct": float(srt[:n1].sum() / tot),
                    "share_top01pct": float(srt[:n01].sum() / tot),
                    "top": [{"node": int(j), "lon": float(lon[j]), "lat": float(lat[j]),
                             "value": float(np.sqrt(m[j]))} for j in idx]})
    return out


def sites_table(rows) -> tuple[str, list]:
    rs = [r for r in rows if r.get("action") in ("tke_sites", "tke_svd")]
    out = ["| site | start | steps | weights | setting | σ | σ/step | ratio to none | iters | "
           "converged | top v cell (regime, tke, F) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    base = {}
    for r in rs:
        if _setting(r) == "exact" and r.get("repeat", 0) == 0:
            base[(r.get("site"), r["start"], r["n_steps"], r["weights"])] = r["sigma"]
    recs = []
    for r in rs:
        key = (r.get("site"), r["start"], r["n_steps"], r["weights"])
        s0 = base.get(key)
        ratio = r["sigma"] / s0 if s0 else None
        tv = [c for c in (r.get("top_v") or []) if c.get("field") == "tke"]
        top = (f"{tv[0].get('regime')} k={tv[0].get('level')} tke={_fmt(tv[0].get('tke'), 3)} "
               f"F={_fmt(tv[0].get('F'), 3)} ({_fmt(tv[0].get('lon'), 4)}, {_fmt(tv[0].get('lat'), 4)})"
               if tv else "-")
        recs.append(dict(site=r.get("site"), start=r["start"], steps=r["n_steps"],
                         weights=r["weights"], setting=_setting(r), repeat=r.get("repeat", 0),
                         sigma=r["sigma"], ratio=ratio, job=r.get("slurm_job")))
        out.append(f"| {r.get('site', '-')} | {r['start'][:16]} | {r['n_steps']} | {r['weights']} | "
                   f"{_setting(r)}{' r' + str(r.get('repeat')) if r.get('repeat') else ''} | "
                   f"{_fmt(r['sigma'])} | {_fmt(r.get('sigma_per_step'))} | {_fmt(ratio)} | "
                   f"{len(r.get('history') or [])} | {r.get('converged')} | {top} |")
    return "\n".join(out), recs


def regime_census(rows, top: int = 20) -> str:
    """Regimes of the top tke cells of the leading input (v) and output (u) vectors of the
    exact-model power iterations."""
    out = []
    for r in rows:
        if r.get("action") not in ("tke_sites", "tke_svd") or _setting(r) != "exact":
            continue
        for which in ("top_v", "top_u"):
            cells = [c for c in (r.get(which) or []) if c.get("field") == "tke"][:top]
            if not cells:
                continue
            cnt = defaultdict(int)
            for c in cells:
                cnt[c.get("regime")] += 1
            lev = sorted({c.get("level") for c in cells})
            tk = [c.get("tke") for c in cells if c.get("tke") is not None]
            F = [c.get("F") for c in cells if c.get("F") is not None]
            ri = [c.get("Ri") for c in cells if c.get("Ri") is not None]
            flo = sum(1 for c in cells if c.get("floor_hit"))
            out.append(f"- {r.get('site')} {r['start'][:16]} {r['n_steps']} steps {r['weights']} "
                       f"{which}: regimes {dict(cnt)}, levels {lev}, tke {_mrange(tk, 3)}, "
                       f"F {_mrange(F, 3)}, Ri {_mrange(ri, 3)}, floor-hit {flo}/{len(cells)}")
    return "\n".join(out)


def _cell(c) -> str:
    if not c:
        return "-"
    return (f"({_fmt(c.get('lon'), 4)}, {_fmt(c.get('lat'), 4)}) k={c.get('level')} "
            f"{c.get('regime')} tke={_fmt(c.get('tke'), 2)} F={_fmt(c.get('F'), 2)} "
            f"Ri={_fmt(c.get('Ri'), 2)}")


def _top_tke(r, which="top_v"):
    cells = [c for c in (r.get(which) or []) if c.get("field") == "tke"]
    return cells[0] if cells else None


def scan_table(rows) -> str:
    rs = sorted([r for r in rows if r.get("action") == "tke_scan"], key=lambda r: r["start"])
    if not rs:
        return ""
    out = ["| window start | steps | σ₁ (tke norm) | σ₁ per step | iters | top input cell (v) | "
           "top output cell (u) |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        out.append(f"| {r['start'][:16]} | {r['n_steps']} | {_fmt(r['sigma'])} | "
                   f"{_fmt(r.get('sigma_per_step'))} | {len(r.get('history') or [])} | "
                   f"{_cell(_top_tke(r))} | {_cell(_top_tke(r, 'top_u'))} |")
    return "\n".join(out)


def _census_str(c) -> str:
    if not c or not c.get("norm2"):
        return "-"
    reg = c.get("regime", {})
    lev = c.get("level", {})
    return (f"free {reg.get('free', 0):.2f} / wall_top {reg.get('wall_top', 0):.2f} / wall_other "
            f"{reg.get('wall_other', 0):.2f} / floor {reg.get('floor', 0):.2f}; k=1 "
            f"{lev.get('1', 0):.2f}, k=2 {lev.get('2', 0):.2f}, k3-5 {lev.get('3-5', 0):.2f}, "
            f"k>5 {lev.get('6-10', 0) + lev.get('>10', 0):.2f}; floor-hit "
            f"{c.get('floor_hit', 0):.2f}; excess<1e-5 {c.get('tke_excess<1e-5', 0):.2f}; "
            f"N2<=0 {c.get('N2<=0', 0):.2f}; Pr=1 {c.get('Pr==1', 0):.2f}, Pr=10 "
            f"{c.get('Pr==10', 0):.2f}; F>10 {c.get('F>10', 0):.2f}")


def site_table(rows) -> str:
    rs = [r for r in rows if r.get("action") == "tke_site"]
    if not rs:
        return ""
    out = ["| site | start | box | MLD median / max (m) | steps | norm | σ₁ | σ₁ per step | iters | "
           "top v cell |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rs, key=lambda r: (r["site"], r["n_steps"], r["weights"])):
        m = r.get("mld_box") or {}
        b = r.get("box")
        bs = (", ".join(_fmt(x, 3) for x in b[:4]) if isinstance(b, list) else str(b))
        out.append(f"| {r['site']} | {r['start'][:16]} | {bs} | {_fmt(m.get('median'), 3)} / "
                   f"{_fmt(m.get('max'), 3)} | {r['n_steps']} | {r['weights']} | "
                   f"{_fmt(r['sigma'])} | {_fmt(r.get('sigma_per_step'))} | "
                   f"{len(r.get('history') or [])} | {_cell(_top_tke(r))} |")
    out += ["", "Where the leading vectors live (share of the tke part of the vector, squared):", ""]
    for r in sorted(rs, key=lambda r: (r["site"], r["n_steps"], r["weights"])):
        if r["weights"] != "tke":
            continue
        out.append(f"- {r['site']} {r['n_steps']} steps, input v: {_census_str(r.get('census_v'))}")
        out.append(f"- {r['site']} {r['n_steps']} steps, output u: {_census_str(r.get('census_u'))}")
    return "\n".join(out)


def cuts_table(rows) -> str:
    rs = [r for r in rows if r.get("action") == "tke_cuts"]
    if not rs:
        return ""
    out = []
    for site in sorted({r["site"] for r in rs}):
        sr = [r for r in rs if r["site"] == site]
        steps = sorted({r["n_steps"] for r in sr})
        base = {n: [r for r in rows if r.get("action") == "tke_site" and r.get("site") == site
                    and r["n_steps"] == n and r["weights"] == "tke"] for n in steps}
        head = f"**{site}** ({sr[0]['start'][:16]}, box {sr[0].get('box')}, MLD median " \
               f"{_fmt((sr[0].get('mld_box') or {}).get('median'), 3)} m)"
        out += [head, ""]
        cols = " | ".join(f"σ₁ {n} steps | ratio" for n in steps)
        out += [f"| setting | {cols} |", "|---|" + "---|---|" * len(steps)]
        b0 = {n: (base[n][0]["sigma"] if base[n] else None) for n in steps}
        line = " | ".join(f"{_fmt(b0[n])} | 1" for n in steps)
        out.append(f"| none (site run) | {line} |")
        order = []
        for r in sr:
            key = (r["setting"], str(r.get("repeat", 0)))
            if key not in order:
                order.append(key)
        for sp, rep_ in order:
            cells = []
            for n in steps:
                m = [r for r in sr if r["setting"] == sp and str(r.get("repeat", 0)) == rep_
                     and r["n_steps"] == n]
                if m:
                    cells.append(f"{_fmt(m[0]['sigma'])} | {_fmt(m[0].get('ratio'))}")
                else:
                    cells.append("- | -")
            name = sp if rep_ in ("0", "None") else f"{sp} (repeat {rep_})"
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        rd = [r for r in sr if r.get("realisation_diff")]
        for r in rd:
            d = r["realisation_diff"]
            last = d[-1]
            out += ["", f"Two forwards of the same program from the same State ({r['n_steps']} "
                        f"steps, job {r.get('slurm_job')}), per step (global / box): conv flips "
                    + ", ".join(f"{x['global']['conv_flips']}/{x.get('box', {}).get('conv_flips', '-')}"
                                for x in d)
                    + f"; tke floor flips at the end {last['global']['tke_floor_flips']}/"
                      f"{last.get('box', {}).get('tke_floor_flips', '-')}; t_skin > 0.1 K "
                      f"{last['global']['t_skin_gt_0.1K']} (max {_fmt(last['global']['t_skin_max'], 3)} K); "
                      f"max |ΔT| {_fmt(last['global']['T_max'], 3)} K (box "
                      f"{_fmt(last.get('box', {}).get('T_max'), 3)})"]
        out.append("")
    return "\n".join(out)


def column_table(rows) -> str:
    out = ["| job | start | nodes | max rel diff (block vs jacfwd) | max σ rel | σ₁ core per column | "
           "max diag | max |sub|,|super| |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("action") != "tke_column":
            continue
        cols = r.get("columns") or []
        sig = ", ".join(_fmt(c["sigma_core"], 3) for c in cols)
        dg = max((max(np.abs(c["diag"])) for c in cols if c["diag"]), default=float("nan"))
        off = max((max(np.abs(c["sub"] + c["super"])) for c in cols if c["sub"]), default=float("nan"))
        out.append(f"| {r.get('slurm_job')} | {r['start'][:16]} | {r.get('nodes')} | "
                   f"{_fmt(r.get('max_rel_diff'))} | {_fmt(r.get('max_sigma_rel'))} | {sig} | "
                   f"{_fmt(dg)} | {_fmt(off)} |")
    return "\n".join(out)


def fdlin_table(rows) -> str:
    g = defaultdict(list)
    for r in rows:
        if r.get("action") == "tke_fdlin":
            g[(r.get("slurm_job"), r.get("label"), r["start"], r["n_steps"], r.get("weights"),
               r.get("ssh_forward_tol"), r.get("platform"))].append(r)
    out = []
    for (job, label, start, n, w, tol, plat), rs in sorted(g.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: -r["h"])
        r0 = rs[0]
        out.append(f"**{label}** — job {job}, {plat}, start {start[:16]}, {n} steps, norm {w}, "
                   f"SSH forward tol {tol}; predicted linear range h ≤ {_fmt(r0.get('predicted_h'))}; "
                   f"noise ({r0.get('noise')}) {_fmt(r0.get('noise_floor'))}; |TL| {_fmt(r0.get('tl_norm'))}")
        out.append("")
        out.append("| h | rel err ‖FD − TL‖/‖TL‖ | noise rel | regime |")
        out.append("|---|---|---|---|")
        for r in rs:
            out.append(f"| {_fmt(r['h'], 2)} | {_fmt(r['rel_err'], 3)} | {_fmt(r.get('noise_rel'), 3)} | "
                       f"{r.get('regime')} |")
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--maps", action="store_true", help="also summarise the screen maps")
    ap.add_argument("--mesh", default=str(MESH_DIR))
    a = ap.parse_args(argv)
    rows = load_rows(a.dirs)
    parts, js = [], {}
    t, recs = screen_table(rows)
    if recs:
        parts += ["## Screens", "", t, "", "### Per-lag-day rates (per step; lag day 1 first)", "",
                  per_day_table(rows), ""]
        js["screens"] = recs
    if a.maps:
        ll = mesh_lonlat(a.mesh)
        js["maps"] = {}
        lines = []
        for r in rows:
            if r.get("action") == "tke_screen" and r.get("maps") and Path(r["maps"]).exists():
                ms = map_sites(r["maps"], ll)
                js["maps"][r["maps"]] = ms
                for m in ms:
                    if "top" in m:
                        t0 = m["top"][0]
                        lines.append(f"- {r['start'][:10]} {_setting(r)} r{r.get('repeat', 0)} "
                                     f"map day {m['day']:g}: |tke cot| {_fmt(m['norm'], 3)}, top 1 % "
                                     f"{m['share_top1pct']:.2f}, top 0.1 % {m['share_top01pct']:.2f}; "
                                     f"max at ({t0['lon']:.1f}, {t0['lat']:.1f})")
        if lines:
            parts += ["## Where (screen maps, tke cotangent column norm)", "", *lines, ""]
    for title, fn in (("## Scan (global seed, tke norm)", scan_table),
                      ("## Sites (seed box around the scan peak)", site_table),
                      ("## Cuts at the top site", cuts_table)):
        txt = fn(rows)
        if txt:
            parts += [title, "", txt, ""]
    t, recs = sites_table(rows)
    if recs:
        parts += ["## Power iteration and cuts", "", t, "", "### Regimes of the leading vectors", "",
                  regime_census(rows), ""]
        js["sites"] = recs
    if any(r.get("action") == "tke_column" for r in rows):
        parts += ["## Column zoom", "", column_table(rows), ""]
    if any(r.get("action") == "tke_fdlin" for r in rows):
        parts += ["## FD vs TL", "", fdlin_table(rows), ""]
    text = "\n".join(parts)
    if a.md:
        Path(a.md).write_text(text)
    else:
        print(text)
    if a.json:
        Path(a.json).write_text(json.dumps(js, default=float, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
