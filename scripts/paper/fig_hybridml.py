"""Fig 4 — §3 Hybrid-ML (NN-of-TKE): the twin proof + the obs application + the deployment test.

Composes the four §3 capability pieces into one headline figure:
  (A) NN twin recovery — the T/S evolution misfit drops ≪ its NN→0 start (E1 ``nn_twin.npz``): the
      proof that NN weights train end-to-end through the full global adjoint.
  (B) Obs held-out MLD-misfit reduction per season (E2 ``nn_obs_train.npz``): the NN, trained on TRAIN
      cells over batched seasonal windows, lowers the INDEPENDENT held-out MLD bias vs WOA (NN_OBS_OK;
      train≈held-out ⇒ not overfitting).
  (C) Long-forward drift (E2 ``nn_obs_validate.npz``): the trained NN deploys forward-only with drift ≤
      default (no destabilization; NN→0 is the bit-identical fallback).
  (D) Persisted benefit (E2 ``nn_obs_validate.npz``): the MLD-vs-WOA improvement persists over the long
      forward, not just the training window (the offline-trained/online-deployed test).

Usage:  python scripts/paper/fig_hybridml.py   (after the E1 + E2 runs; skips any missing panel)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SC = ROOT / "scripts"


def _load(name):
    p = SC / name
    return np.load(p, allow_pickle=True) if p.exists() else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    twin = _load("nn_twin.npz") or _load("nn_twin_batched.npz")
    tr = _load("nn_obs_train_run2.npz") or _load("nn_obs_train.npz")
    va = _load("nn_obs_validate_trained.npz") or _load("nn_obs_validate.npz")

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))

    # (A) twin recovery
    if twin is not None and "loss_hist" in twin.files:
        lh = twin["loss_hist"]
        ax[0, 0].semilogy(np.arange(1, len(lh) + 1), lh, ".-", ms=4, lw=1, color="C2")
        ax[0, 0].set(xlabel="Adam iteration", ylabel="T/S evolution loss (J/J0)",
                     title="(A) NN twin: weights recovered through the global adjoint")
        ax[0, 0].grid(True, which="both", alpha=0.25)
    else:
        ax[0, 0].text(0.5, 0.5, "twin npz missing", ha="center", transform=ax[0, 0].transAxes)

    # (B) held-out MLD reduction per season
    if tr is not None:
        months = tr["months"]; x = np.arange(len(months)); w = 0.2
        ax[0, 1].bar(x - 1.5 * w, tr["b_mld_tr"], w, label="train base", color="0.75")
        ax[0, 1].bar(x - 0.5 * w, tr["f_mld_tr"], w, label="train NN", color="C0")
        ax[0, 1].bar(x + 0.5 * w, tr["b_mld_ho"], w, label="held base", color="0.5")
        ax[0, 1].bar(x + 1.5 * w, tr["f_mld_ho"], w, label="held NN", color="C1")
        rho = float(tr["red_mld_ho"]); rtr = float(tr["red_mld_tr"])
        ax[0, 1].set(xlabel="season (month)", ylabel="MLD misfit [m²]",
                     title=f"(B) obs MLD reduced: train {rtr:+.1f}% / held-out {rho:+.1f}%")
        ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels([str(int(m)) for m in months], fontsize=7)
        ax[0, 1].legend(fontsize=6, ncol=2)
    else:
        ax[0, 1].text(0.5, 0.5, "obs-train npz missing", ha="center", transform=ax[0, 1].transAxes)

    # (C) long-forward drift
    if va is not None:
        d = va["days"]
        ax[1, 0].plot(d, va["drift_T_def"], "o-", ms=3, color="0.5", label="default (NN→0)")
        ax[1, 0].plot(d, va["drift_T_tr"], "s-", ms=3, color="C0", label="trained NN")
        ax[1, 0].set(xlabel="forward day", ylabel="vol-wt RMS ΔT [°C]",
                     title="(C) STABLE deploy: trained drift ≈ default (no blow-up)")
        ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.25)
        # (D) the offline/online gap — the short-window benefit does NOT persist
        ax[1, 1].plot(d, va["mld_def"], "o-", ms=3, color="0.5", label="default (NN→0)")
        ax[1, 1].plot(d, va["mld_tr"], "s-", ms=3, color="C3", label="trained NN")
        ax[1, 1].set(xlabel="forward day", ylabel="MLD misfit vs WOA [m²]",
                     title="(D) offline/online gap: short-window benefit does NOT persist")
        ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.25)
    else:
        for a in (ax[1, 0], ax[1, 1]):
            a.text(0.5, 0.5, "validate npz missing", ha="center", transform=a.transAxes)

    fig.suptitle("§3 Hybrid-ML: an NN mixing closure trained through the differentiable FESOM2-JAX — "
                 "twin proof → held-out obs reduction → STABLE deploy, but the short-window benefit\n"
                 "does not persist (the offline/online gap ⇒ the deployed equilibrium is a slow target)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = SC / "fig_hybridml.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
