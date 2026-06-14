"""Fig 2 — instantaneous adjoint sensitivity maps + the adjoint↔EKI agreement (Paper-experiments
Task C1, §1 Sensitivity).

Reads the per-target ``.npz`` maps + the JSONL summary that :mod:`scripts.core2_paper_sensitivity`
wrote, draws the two ``[nod2D]`` sensitivity maps (``∂(mean MLD)/∂c_k(x)`` and
``∂(upper-ocean T)/∂k_gm(x)``, signed diverging colour), annotates the FD spot-check, adds the
adjoint↔EKI agreement inset for ``k_gm``, and emits **SENSITIVITY_MAP_OK** when every target's gate
components passed (adjoint==FD + map finite/nonzero + FD spot-check + the adjoint↔EKI cross-check).

The figure is optional (degrades cleanly if matplotlib is absent — the decision + token still print).
Usage:  python scripts/fig_sensitivity.py --results <file.jsonl> --out scripts/fig_sensitivity.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORDER = ("mld_ck", "ts_kgm")


def load(results: Path):
    recs = {}
    for line in results.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            recs[r["target"]] = r           # last write per target wins
    return recs


def _panel(ax, npz, rec):
    """One signed sensitivity map: wet nodes scattered on lon/lat, diverging colour, the FD
    spot-check node ringed."""
    import matplotlib.pyplot as plt        # noqa: F401 (colormap registration)
    lon = npz["lon"]; lat = npz["lat"]; g = npz["grad"]
    wet = npz["node_wet"].astype(bool) if "node_wet" in npz else np.ones_like(g, bool)
    sel = wet & np.isfinite(g)
    # symmetric limits at the 99th percentile so a few extrema don't wash out the structure
    vmax = np.percentile(np.abs(g[sel]), 99) if sel.any() else 1.0
    vmax = float(vmax) if vmax > 0 else 1.0
    sc = ax.scatter(lon[sel], lat[sel], c=g[sel], s=3, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True)
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(f"{rec['label']}\n[{rec['unit']}]", fontsize=8)
    # ring the FD spot-check node
    si = int(rec.get("spot_idx", -1))
    if 0 <= si < lon.size:
        ax.scatter([lon[si]], [lat[si]], s=70, facecolors="none", edgecolors="k", linewidths=1.2)
        ax.annotate("FD spot-check", (lon[si], lat[si]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("lon", fontsize=8); ax.set_ylabel("lat", fontsize=8)
    ax.set_title(f"{rec['config']}: {rec['label']}  (N={rec['N']}, ~{rec['days']:.2f} d, "
                 f"fast/instantaneous)\nFD spot-check rel={rec.get('spot_rel', float('nan')):.1e}, "
                 f"adjoint==FD plateau={rec.get('plateau', float('nan')):.1e}", fontsize=8)


def make_figure(recs, outdir: Path, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] matplotlib unavailable ({e}); skipping the plot", flush=True)
        return False
    avail = [t for t in ORDER if t in recs and (outdir / f"sensitivity_{t}.npz").exists()]
    if not avail:
        print("  [fig] no .npz maps found; skipping the plot", flush=True)
        return False

    fig, axes = plt.subplots(len(avail), 1, figsize=(8.5, 4.0 * len(avail)), squeeze=False)
    for ax, t in zip(axes[:, 0], avail):
        npz = np.load(outdir / f"sensitivity_{t}.npz", allow_pickle=True)
        _panel(ax, npz, recs[t])

    # adjoint↔EKI agreement inset (k_gm) on the ts_kgm panel
    if "ts_kgm" in avail:
        r = recs["ts_kgm"]
        if r.get("grad_ens") is not None and np.isfinite(r.get("grad_ens", np.nan)):
            ax = axes[avail.index("ts_kgm"), 0]
            ins = ax.inset_axes([0.06, 0.10, 0.26, 0.34])
            vals = [r["grad_scalar"], r["grad_ens"]]
            ins.bar([0, 1], vals, color=["C0", "C1"], width=0.6)
            ins.set_xticks([0, 1]); ins.set_xticklabels(["adjoint", "EKI\nensemble"], fontsize=6)
            ins.axhline(0, color="0.5", lw=0.6)
            ins.set_title(f"dJ/dk_gm  rel={r.get('ens_rel', float('nan')):.1e}", fontsize=6)
            ins.tick_params(labelsize=5)

    fig.suptitle("Fig 2 — instantaneous adjoint sensitivity maps (one backward pass; the fast "
                 "sensitivity, not the equilibrium)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=130)
    print(f"  [fig] wrote {out}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("scripts/fig_sensitivity.png"))
    args = ap.parse_args()
    if not args.results.exists():
        print(f"no results file {args.results}; run scripts/core2_paper_sensitivity.py first")
        return 1
    recs = load(args.results)
    if not recs:
        print("results file empty")
        return 1
    outdir = args.out.parent

    print("\n  target   config       N   d(J)/d(theta)   plateau    spot_rel   EKI_rel   gate")
    print("  " + "-" * 80)
    all_ok = True
    for t in ORDER:
        if t not in recs:
            print(f"  {t:<8s} (missing)")
            all_ok = False
            continue
        r = recs[t]
        comp = (r.get("proof_ok") and r.get("map_finite") and (r.get("map_nonzero", 0) > 0)
                and r.get("spot_ok") and r.get("xcheck_ok", True))
        all_ok = all_ok and bool(comp)
        eki_rel = r.get("ens_rel", float("nan"))
        print(f"  {t:<8s} {r['config']:<11s} {r['N']:<3d} {r['grad_scalar']:+.3e}   "
              f"{r.get('plateau', float('nan')):.2e}   {r.get('spot_rel', float('nan')):.2e}   "
              f"{eki_rel:.2e}   {'OK' if comp else 'FAIL'}")

    print("\n  === §1 sensitivity decision ===")
    print("  One backward pass through the assembled global model yields the full [nod2D] parameter")
    print("  sensitivity field — the FAST/INSTANTANEOUS sensitivity over an ~10-h window (A7 N_max=20),")
    print("  NOT the multi-year equilibrium. The map shows which parameters matter and WHERE → §2 calibration")
    print("  descends it. The adjoint↔EKI agreement on the shared k_gm scalar validates both tools and")
    print("  motivates EKI for the slow GM→T/S equilibrium the adjoint window cannot reach.")

    make_figure(recs, outdir, args.out)

    have_both = all(t in recs for t in ORDER)
    print("\nSENSITIVITY_MAP_OK" if (all_ok and have_both) else "\nSENSITIVITY_MAP_INCOMPLETE",
          flush=True)
    return 0 if (all_ok and have_both) else 1


if __name__ == "__main__":
    raise SystemExit(main())
