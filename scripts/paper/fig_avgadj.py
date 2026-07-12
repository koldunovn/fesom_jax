"""Fig D1 — the ENSEMBLE-AVERAGED climate sensitivity maps (long-window Task D1).

Reads one or more ``lw_avgadj_{target}_{mode}_map.npz`` (written by ``core2_lw_avgadj.py``) and, per
map, draws TWO panels:
  * LEFT — the signed ``[nod2D]`` climate sensitivity map on the CORE2 mesh (wet nodes scattered on
    lon/lat, diverging RdBu_r, symmetric limits at the 99th |percentile| so a few convection-site
    extrema don't wash out the basin-scale structure). This is the climate-timescale companion to the
    paper's instantaneous Fig 2 — adjoint: ``d(global diagnostic)/d(param(x))``; tlm: the response
    field ``d(diagnostic(x))/d(global param)``.
  * RIGHT — the across-burst CONVERGENCE curve: the running mean of the (MAD-kept) per-burst scalars
    vs burst count, with the final ensemble value ± across-burst SE band. Shows how many short bursts
    the climate sensitivity took to stabilise (the uncertainty the paper plan demands).

Degrades cleanly if matplotlib is absent (prints the numeric summary + AVGADJ_FIG_OK regardless).

Usage:
  python scripts/paper/fig_avgadj.py --maps scripts/lw_avgadj_mld_ck_map.npz scripts/lw_avgadj_t100_kgm_map.npz \\
      --out scripts/fig_avgadj_climate.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _summary(npz):
    g = npz["mean_map"]
    wet = npz["node_wet"].astype(bool) if "node_wet" in npz else np.ones_like(g, bool)
    sel = wet & np.isfinite(g)
    gg = g[sel]
    return dict(
        label=str(npz["label"]), unit=str(npz["unit"]),
        mode=(str(npz["mode"]) if "mode" in npz else "adjoint"),
        target=str(npz["target"]) if "target" in npz else "?",
        scalar=float(npz["scalar_grad"]), se=float(npz["scalar_se"]),
        K=int(npz["K"]), N=int(npz["N"]),
        nwet=int(sel.sum()), frac_pos=float(np.mean(gg > 0)) if gg.size else float("nan"),
        absmax=float(np.abs(gg).max()) if gg.size else float("nan"),
    )


def _map_panel(ax, npz, s, symlog=False):
    import matplotlib.colors as mcolors
    lon = npz["lon"]; lat = npz["lat"]; g = npz["mean_map"]
    wet = npz["node_wet"].astype(bool) if "node_wet" in npz else np.ones_like(g, bool)
    sel = wet & np.isfinite(g)
    ag = np.abs(g[sel])
    if symlog:
        # heavy-tailed (TLM) response field: a signed-log scale reveals the broad fingerprint UNDER
        # the few extreme convection nodes; capture more of the tail (99.9th) + a robust linthresh.
        vmax = float(np.percentile(ag, 99.9)) if sel.any() else 1.0
        pos = ag[ag > 0]
        lt = float(np.percentile(pos, 75)) if pos.size else 1e-6
        norm = mcolors.SymLogNorm(linthresh=max(lt, vmax * 1e-4), vmin=-vmax, vmax=vmax, base=10)
        sc = ax.scatter(lon[sel], lat[sel], c=g[sel], s=3, cmap="RdBu_r", norm=norm,
                        linewidths=0, rasterized=True)
    else:
        vmax = float(np.percentile(ag, 99)) if sel.any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        sc = ax.scatter(lon[sel], lat[sel], c=g[sel], s=3, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True)
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(f"{s['label']}\n[{s['unit']}]" + ("  (symlog)" if symlog else ""), fontsize=8)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("lon", fontsize=8); ax.set_ylabel("lat", fontsize=8)
    role = ("where-to-tune (Σ→global)" if s["mode"] == "adjoint" else "spatial fingerprint (1 knob)")
    ax.set_title(f"[{s['mode']}] {s['label']}   (climate: 10-yr-mean, K={s['K']} bursts, "
                 f"N={s['N']}={s['N']*1800/86400:.2f} d)\n{role}: scalar={s['scalar']:+.3e}"
                 f"±{s['se']:.1e} {s['unit']}, |max|={s['absmax']:.2e}, {100*s['frac_pos']:.0f}% nodes +",
                 fontsize=8)


def _conv_panel(ax, npz, s):
    rm = np.asarray(npz["running_mean"], dtype=float) if "running_mean" in npz else np.array([])
    x = np.arange(1, rm.size + 1)
    ax.plot(x, rm, "-", color="C0", lw=1.3, label="running mean (MAD-kept)")
    ax.axhline(s["scalar"], color="k", lw=1.0, ls="--", label=f"final {s['scalar']:+.3e}")
    ax.fill_between([1, max(2, rm.size)], s["scalar"] - s["se"], s["scalar"] + s["se"],
                    color="0.8", alpha=0.6, label="±SE")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("bursts averaged", fontsize=8)
    ax.set_ylabel(f"running ⟨{s['label']}⟩ [{s['unit']}]", fontsize=7)
    ax.set_title(f"across-burst convergence (SE/|mean|={abs(s['se']/s['scalar']) if s['scalar'] else float('nan'):.0%})",
                 fontsize=8)
    ax.legend(fontsize=6, loc="best")


def make_figure(maps, out: Path, symlog="auto"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] matplotlib unavailable ({e}); skipping the plot", flush=True)
        return False
    npzs = [(p, np.load(p, allow_pickle=True)) for p in maps if Path(p).exists()]
    if not npzs:
        print("  [fig] no maps found; skipping the plot", flush=True)
        return False
    n = len(npzs)
    fig, axes = plt.subplots(n, 2, figsize=(13.5, 4.2 * n), squeeze=False,
                             gridspec_kw=dict(width_ratios=[2.3, 1.0]))
    for i, (p, npz) in enumerate(npzs):
        s = _summary(npz)
        # auto: symlog only for the heavy-tailed TLM response fields (smooth adjoint maps stay linear)
        sl = (s["mode"] == "tlm") if symlog == "auto" else bool(symlog)
        _map_panel(axes[i, 0], npz, s, symlog=sl)
        _conv_panel(axes[i, 1], npz, s)
    fig.suptitle("Fig D1 — ensemble-averaged climate sensitivity maps (many short frozen-ice bursts "
                 "along the 10-yr reference)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=130)
    print(f"  [fig] wrote {out}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True, help="lw_avgadj_*_map.npz files")
    ap.add_argument("--out", type=Path, default=Path("scripts/fig_avgadj_climate.png"))
    ap.add_argument("--symlog", choices=("auto", "on", "off"), default="auto",
                    help="signed-log map colour: auto = on for tlm (heavy-tailed response), off for adjoint")
    args = ap.parse_args()
    symlog = {"auto": "auto", "on": True, "off": False}[args.symlog]

    print("\n  mode     target      label                          scalar ± SE              K   N")
    print("  " + "-" * 92)
    for p in args.maps:
        if not Path(p).exists():
            print(f"  (missing) {p}"); continue
        s = _summary(np.load(p, allow_pickle=True))
        print(f"  {s['mode']:<8s} {s['target']:<11s} {s['label']:<28s} "
              f"{s['scalar']:+.4e} ± {s['se']:.2e} {s['unit']:<4s} {s['K']:>4d} {s['N']:>3d}")
    ok = make_figure(args.maps, args.out, symlog=symlog)
    print(f"\nAVGADJ_FIG_{'OK' if ok else 'SKIPPED'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
