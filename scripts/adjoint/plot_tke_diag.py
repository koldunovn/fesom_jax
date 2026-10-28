#!/usr/bin/env python
"""Figures of the TKE-adjoint diagnosis runs (``scripts/adjoint/tke_diag.py``), host-side
numpy + matplotlib (no model, no device).

    python scripts/adjoint/plot_tke_diag.py RUN_DIR [RUN_DIR ...] --out FIGDIR

Writes (each only when the runs contain the rows it needs):

* ``screen_traces_<start>_<days>d.png`` — the reverse-pass norm of the State cotangent against
  the lag (days before the window end), one line per setting and repeat (log scale), and the
  per-step rate per chunk below it;
* ``screen_maps_<start>_<setting>_r<k>.png`` — the tke cotangent column norm at each map day
  (log colour scale, top of the range = the day's maximum), on longitude/latitude;
* ``cuts_<site>_<steps>.png`` — σ₁ of the window map under each cut relative to no cut;
* ``fdlin_<label>_<job>.png`` — ‖FD − TL‖/‖TL‖ against h, with the predicted linear range and
  the forward noise floor.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import tke_diag_summary as S  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "lines.linewidth": 1.6, "font.size": 9,
                     "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False})


def _tag(s: str) -> str:
    return s.replace("+", "__").replace(":", "").replace("/", "_")


def plot_screens(rows, out: Path) -> list:
    made = []
    groups = defaultdict(list)
    for r in rows:
        if r.get("action") == "tke_screen":
            groups[(r["start"][:10], r["n_steps"])].append(r)
    for (start, n), rs in sorted(groups.items()):
        settings = sorted({S._setting(r) for r in rs}, key=lambda s: (s != "exact", s))
        colour = {s: SERIES[i % len(SERIES)] for i, s in enumerate(settings)}
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
        seen = set()
        for r in sorted(rs, key=lambda r: (S._setting(r), r.get("repeat", 0))):
            k = int(r["chunk_steps"])
            tr = np.asarray(r["trace"], float)
            lag = np.arange(tr.size) * k * S.DT / 86400.0
            st = S._setting(r)
            lab = st if st not in seen else None
            seen.add(st)
            ls = ["-", "--", ":", "-."][int(r.get("repeat", 0)) % 4]
            a1.semilogy(lag, tr / tr[0], ls, color=colour[st], label=lab)
            rate = np.asarray(S._rates(list(tr), k, 1))
            a2.plot(lag[1:], rate, ls, color=colour[st], lw=1.0)
        a1.set_ylabel("‖state cotangent‖ / seed")
        a1.set_title(f"Reverse-pass growth, {n * S.DT / 86400.0:g}-day window from {start} "
                     f"(line style = repeat)", color=INK, loc="left")
        a1.legend(fontsize=8, ncol=2)
        a2.axhline(1.0, color=MUTED, lw=0.8)
        a2.set_ylabel("per-step rate (chunk)")
        a2.set_xlabel("lag before the window end (days)")
        fig.tight_layout()
        p = out / f"screen_traces_{start}_{n * S.DT / 86400.0:g}d.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


def plot_maps(rows, out: Path, mesh_dir, field: str = "tke", max_maps: int = 12) -> list:
    made = []
    lon, lat = S.mesh_lonlat(mesh_dir)
    for r in rows:
        if r.get("action") != "tke_screen" or not r.get("maps") or not Path(r["maps"]).exists():
            continue
        z = np.load(r["maps"])
        if field not in z.files or z[field].ndim != 2 or z[field].shape[0] == 0:
            continue
        a = np.asarray(z[field], float)
        days = z["day"]
        order = np.argsort(days)                       # the reverse sweep records late days first
        sel = np.linspace(0, a.shape[0] - 1, min(max_maps, a.shape[0])).round().astype(int)
        idx = [int(order[i]) for i in sorted(set(sel.tolist()))]
        nc = min(4, len(idx))
        nr = int(np.ceil(len(idx) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc + 0.8, 2.0 * nr + 0.5), squeeze=False,
                                layout="constrained")
        sc = None
        for ax, i in zip(axs.ravel(), idx):
            m = a[i]
            over = not np.all(np.isfinite(m))          # float32 maps overflow above 3.4e38
            m = np.where(np.isfinite(m), m, np.finfo(np.float64).max)
            vmax = m.max()
            if vmax <= 0:
                ax.set_axis_off()
                continue
            o = np.argsort(m)
            sc = ax.scatter(lon[o], lat[o], c=np.log10(np.maximum(m[o] / vmax, 1e-6)), s=1.2,
                            cmap="Blues", vmin=-6.0, vmax=0.0, linewidths=0)
            ax.set_title(f"model day {days[i]:g}: max " + ("> 3.4e38 (float32 map)" if over
                                                             else f"{vmax:.2g}"),
                         fontsize=8, color=INK, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-180, 180)
            ax.set_ylim(-80, 90)
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True)
        for ax in axs.ravel()[len(idx):]:
            ax.set_axis_off()
        if sc is not None:
            fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.7,
                         label=f"log10 (|{field} cot| / panel max)")
        fig.suptitle(f"{field} cotangent, {r['start'][:10]} {S._setting(r)} repeat "
                     f"{r.get('repeat', 0)} (job {r.get('slurm_job')})", fontsize=9, color=INK,
                     x=0.02, ha="left")
        p = out / f"screen_maps_{r['start'][:10]}_{r['n_steps']}_{_tag(S._setting(r))}_r{r.get('repeat', 0)}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)
    return made


def plot_cuts(rows, out: Path) -> list:
    made = []
    g = defaultdict(list)
    for r in rows:
        if r.get("action") in ("tke_sites", "tke_cuts", "tke_site"):
            g[(r.get("site", "svd"), r["start"][:16], r["n_steps"], r.get("weights", "tke"))].append(r)
    for (site, start, n, wts), rs in sorted(g.items()):
        if not any(r.get("action") != "tke_site" for r in rs):
            continue
        base = [r["sigma"] for r in rs if S._setting(r) == "exact" and r.get("action") != "tke_site"
                or r.get("action") == "tke_site"]
        if not base or len(rs) < 3:
            continue
        s0 = base[0]
        cut = [r for r in rs if S._setting(r) != "exact"
               and not str(r.get("repeat", 0)).startswith("pins")]
        cut = sorted(cut, key=lambda r: r["sigma"])
        names = [S._setting(r) for r in cut]
        vals = np.asarray([max(r["sigma"] / s0, 1e-30) for r in cut])
        fig, ax = plt.subplots(figsize=(7.0, 0.28 * len(cut) + 1.3))
        ax.barh(np.arange(len(cut)), np.log10(vals), color=SERIES[0], height=0.6)
        for b in base[1:]:
            ax.axvline(np.log10(b / s0), color=MUTED, ls="--", lw=0.8)
        ax.axvline(0.0, color=INK, lw=0.8)
        ax.set_yticks(np.arange(len(cut)), names, fontsize=8)
        ax.set_xlabel("log10 σ₁(cut) / σ₁(no cut)   (dashed: repeats of no cut)")
        ax.set_title(f"{site}: {n}-step window from {start}, {wts} norm, σ₁(no cut) = {s0:.3g}",
                     color=INK, loc="left", fontsize=9)
        fig.tight_layout()
        p = out / f"cuts_{_tag(str(site))}_{n}_{wts}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


def plot_fdlin(rows, out: Path) -> list:
    made = []
    g = defaultdict(list)
    for r in rows:
        if r.get("action") == "tke_fdlin":
            g[(r.get("slurm_job"), r.get("label"))].append(r)
    for (job, label), rs in sorted(g.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: r["h"])
        h = np.asarray([r["h"] for r in rs])
        e = np.asarray([r["rel_err"] for r in rs])
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.loglog(h, e, "o-", color=SERIES[0], ms=5, label="‖FD − TL‖ / ‖TL‖")
        nz = [r.get("noise_rel") for r in rs]
        if any(x is not None for x in nz):
            ax.loglog(h, [x if x is not None else np.nan for x in nz], "s--", color=SERIES[1],
                      ms=4, lw=1.0, label=f"noise floor ({rs[0].get('noise')})")
        ph = rs[0].get("predicted_h")
        if ph is not None and np.isfinite(ph):
            ax.axvline(ph, color=MUTED, ls=":", lw=1.0)
            ax.text(ph, ax.get_ylim()[1], " predicted range", color=MUTED, fontsize=7, va="top")
        ax.set_xlabel("h")
        ax.set_ylabel("relative error")
        ax.set_title(f"{label}: {rs[0]['n_steps']} steps from {rs[0]['start'][:16]}, "
                     f"SSH tol {rs[0].get('ssh_forward_tol')}, {rs[0].get('platform')}",
                     color=INK, loc="left", fontsize=8)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out / f"fdlin_{_tag(str(label))}_{job}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


def plot_setting_rates(rows, out: Path, min_settings: int = 4) -> list:
    """Overall per-step growth rate of every setting of a screen window (median and the range
    over its repeats), sorted, against the band spanned by the exact repeats."""
    made = []
    groups = defaultdict(list)
    for r in rows:
        if r.get("action") == "tke_screen":
            groups[(r["start"][:10], r["n_steps"])].append(r)
    for (start, n), rs in sorted(groups.items()):
        by = defaultdict(list)
        for r in rs:
            st = S.trace_stats(r["trace"], int(r["chunk_steps"]))["overall"]
            by[S._setting(r)].append(st)
        if len(by) < min_settings:
            continue
        names = sorted(by, key=lambda k: np.median(by[k]))
        fig, ax = plt.subplots(figsize=(7.0, 0.3 * len(names) + 1.4), layout="constrained")
        if "exact" in by:
            ax.axvspan(min(by["exact"]), max(by["exact"]), color="#e6e5e0", zorder=0,
                       label=f"exact, {len(by['exact'])} repeats")
        for i, k in enumerate(names):
            v = np.asarray(by[k])
            col = SERIES[0] if k != "exact" else INK
            ax.plot([v.min(), v.max()], [i, i], color=col, lw=2.0)
            ax.plot(v, [i] * v.size, "o", color=col, ms=4)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.axvline(1.0, color=MUTED, lw=0.8)
        ax.set_xlabel("overall growth per 30-min step of the reverse pass (each dot one run)")
        ax.set_title(f"{n * S.DT / 86400.0:g}-day window from {start}: every setting", color=INK,
                     loc="left", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        p = out / f"setting_rates_{start}_{n * S.DT / 86400.0:g}d.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


def plot_rho_maps(rows, out: Path) -> list:
    """The spectral radius ρ of each column's TKE-only recursion (``colcensus``): columns with
    ρ > 1 coloured by log10 ρ, the others grey."""
    made = []
    for r in rows:
        if r.get("action") != "tke_colcensus" or r.get("setting") != "none" or not r.get("rho_map"):
            continue
        if not Path(r["rho_map"]).exists():
            continue
        z = np.load(r["rho_map"])
        rho, lon, lat = z["rho"].astype(float), z["lon"], z["lat"]
        fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
        q = rho <= 1.0
        ax.scatter(lon[q], lat[q], s=0.4, c="#e6e5e0", linewidths=0)
        u = ~q
        o = np.argsort(rho[u])
        sc = ax.scatter(lon[u][o], lat[u][o], s=3.0, c=np.log10(rho[u][o]), cmap="Reds",
                        vmin=0.0, vmax=max(0.5, float(np.log10(rho[u].max()))), linewidths=0)
        fig.colorbar(sc, ax=ax, shrink=0.8, label="log10 ρ (columns with ρ > 1)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-80, 90)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"TKE-only recursion, {r['start'][:16]}: {r['n_rho_gt_1']} of {r['n_wet']} "
                     f"columns with ρ > 1 (max {r['rho_max']:.3g})", color=INK, loc="left",
                     fontsize=9)
        p = out / f"rho_map_{r['start'][:13].replace(':', '')}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mesh", default=str(S.MESH_DIR))
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = S.load_rows(a.dirs)
    made = (plot_screens(rows, out) + plot_cuts(rows, out) + plot_fdlin(rows, out)
            + plot_rho_maps(rows, out) + plot_setting_rates(rows, out))
    if not a.no_maps:
        made += plot_maps(rows, out, a.mesh)
    for p in made:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
