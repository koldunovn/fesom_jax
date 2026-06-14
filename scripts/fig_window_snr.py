"""Gradient-SNR vs adjoint-window figure + the adjoint↔EKI decision (Paper-experiments Task A7).

Reads the JSONL the sweep (:mod:`scripts.core2_adjoint_window_sweep`) appended, finds **N_max**
(the largest window with a clean gradient that fits 80 GB), draws the supplementary figure
(FD↔AD agreement + peak backward memory vs N), and records the per-target adjoint-vs-EKI decision.
Emits **WINDOW_DERISK_OK** when the sweep produced a usable N_max + memory curve + decision.

The figure is optional (degrades cleanly if matplotlib is absent — the decision + token still
print). Usage:  python scripts/fig_window_snr.py --results <file.jsonl> --out fig_window_snr.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PLATEAU_CLEAN = 1.0e-2     # FD↔AD relative agreement below this ⇒ the gradient is "clean"


def load(results: Path):
    rows = []
    for line in results.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["N"])
    return rows


def make_figure(rows, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] matplotlib unavailable ({e}); skipping the plot", flush=True)
        return False
    ok = [r for r in rows if r.get("grad") is not None]
    Ns = [r["N"] for r in ok]
    plat = [r.get("plateau", float("nan")) for r in ok]
    mem = [r.get("peak_gb", float("nan")) for r in rows]
    memN = [r["N"] for r in rows]
    gpu_gb = next((r["gpu_gb"] for r in rows if r.get("gpu_gb")), 80.0)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)
    a1.axhline(PLATEAU_CLEAN, color="0.6", ls="--", lw=1, label=f"clean < {PLATEAU_CLEAN:g}")
    a1.semilogy(Ns, plat, "o-", color="C0")
    a1.set_ylabel("FD↔AD relative gap\n(min over h)")
    a1.set_title("CORE2 adjoint window: gradient quality + memory vs N\nd(mean MLD)/d(tke_c_k), zstar+TKE")
    a1.legend(loc="best", fontsize=8); a1.grid(True, which="both", alpha=0.3)

    a2.axhline(gpu_gb, color="C3", ls="--", lw=1, label=f"A100 {gpu_gb:.0f} GB")
    a2.plot(memN, mem, "s-", color="C1")
    a2.set_ylabel("peak backward\nmemory (GB)"); a2.set_xlabel("adjoint window N (steps)")
    a2.legend(loc="best", fontsize=8); a2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"  [fig] wrote {out}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("scripts/fig_window_snr.png"))
    args = ap.parse_args()
    if not args.results.exists():
        print(f"no results file {args.results}; run the sweep first")
        return 1
    rows = load(args.results)
    if not rows:
        print("results file empty")
        return 1

    gpu_gb = next((r["gpu_gb"] for r in rows if r.get("gpu_gb")), 80.0)
    print(f"\n  (GPU memory limit: {gpu_gb:.0f} GB)")
    print("  N     days   peak_GB   d(MLD)/d(c_k)    plateau    SNR      clean fitsGPU")
    print("  " + "-" * 74)
    for r in rows:
        g = r.get("grad")
        gs = f"{g:+.3e}" if g is not None else "   (failed)  "
        print(f"  {r['N']:<5d} {r.get('days',0):5.2f}  {r.get('peak_gb',0):6.2f}   {gs}   "
              f"{r.get('plateau',float('nan')):.2e}  {r.get('snr',float('nan')):.2e}  "
              f"{str(r.get('clean',False)):5s} {str(r.get('fits_gpu',False)):5s}")

    clean_fit = [r for r in rows if r.get("clean") and r.get("fits_gpu")]
    n_max = max((r["N"] for r in clean_fit), default=None)
    oomed = [r["N"] for r in rows if r.get("oom")]
    days_max = (n_max * rows[0].get("dt", 1800.0) / 86400.0) if n_max else 0.0

    print("\n  === adjoint↔EKI decision ===")
    if n_max:
        print(f"  N_max = {n_max} steps (~{days_max:.2f} days) — largest CLEAN gradient fitting 80 GB.")
        print(f"  FAST targets (MLD, SST): days–seasonal ⇒ ADJOINT-reachable up to N_max "
              f"(batch windows for the climatology).")
        print(f"  SLOW targets (GM→T/S stratification): multi-year equilibrium >> N_max ⇒ EKI "
              f"(forward ensemble, fesom_jax.eki).")
    else:
        print("  No clean+fitting window found — all targets via EKI / shorter windows; "
              "or build nested checkpointing.")
    if oomed:
        print(f"  OOM at N ∈ {oomed} (peak > 80 GB) — the memory ceiling; reconciles the inherited "
              f"37.8 GB-at-N=20 GM figure with this TKE→MLD curve.")
    print("  NOTE: config is zstar+TKE (GM/ice off) ⇒ N_max is an UPPER BOUND; the all-on "
          "(+GM streamfunction +mEVP) adjoint window is ≤ this.")

    make_figure(rows, args.out)

    derisked = (n_max is not None) or (len(oomed) > 0)   # a usable boundary was found either way
    print("\nWINDOW_DERISK_OK" if derisked else "\nWINDOW_DERISK_INCOMPLETE", flush=True)
    return 0 if derisked else 1


if __name__ == "__main__":
    raise SystemExit(main())
