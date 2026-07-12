"""CORE2 TKE stability + scheme-discrimination — Phase 9b, Task JT.5.

10-day (default) forward run of the assembled CORE2 model with classical-TKE mixing
(``tke_cfg=TkeConfig()``, linfs, KPP/GM/ice OFF — the validated TKE-only config), checking
numerical stability (no NaN, |SSH|<5 m, max|vel|<3 m/s), then the SAME run with KPP to
report the TKE↔KPP surface-RMS scheme difference (must be clearly resolved — the C measured
an 11–18× SST contrast; here we just require it ≫ FP noise). Reuses the KPP stability
script's helpers (``build``/``make_diag``/``stable``/``report``/``surf_rms``).

Usage (GPU; scripts/archive/core2_tke_stability.sbatch):  python scripts/archive/core2_tke_stability.py --steps 480
"""

from __future__ import annotations

import argparse
import time

import fesom_jax  # noqa: F401  (enables x64)
import jax
import numpy as np

from fesom_jax import step as stepmod
from fesom_jax.kpp import KppConfig
from fesom_jax.tke import TkeConfig

# reuse the KPP stability harness (build the model + forcing, the per-step diagnostics,
# the stability predicate, the surface-RMS).
from core2_kpp_stability_run import DT, build, make_diag, report, stable, surf_rms


def run(label, mixing, mesh, state, op, cf, dates, diag, steps, every):
    """One TKE-only / KPP-only / PP forward trajectory (no GM, no ice — isolating the mixing
    scheme, the validated TKE config). Returns (ok, final_state)."""
    fs = cf.static
    tke_cfg = TkeConfig() if mixing == "tke" else None
    kpp_cfg = KppConfig() if mixing == "kpp" else None
    print(f"\n========== {label}: mixing={mixing.upper()} (linfs, no GM/ice) "
          f"({steps} steps, {steps*DT/86400:.2f} days) ==========", flush=True)
    d = jax.device_get(diag(state)); report("init", d)
    worst_vel = 0.0
    for i in range(steps):
        sf = cf.step_forcing(*dates[i])
        ts = time.time()
        state = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=(i == 0),
                                 step_forcing=sf, forcing_static=fs,
                                 tke_cfg=tke_cfg, kpp_cfg=kpp_cfg)
        d = jax.device_get(diag(state)); dts = time.time() - ts
        worst_vel = max(worst_vel, float(d["vel_absmax"]))
        ok, why = stable(d)
        step = i + 1
        day_bdry = int(step * DT // 86400) > int((step - 1) * DT // 86400)
        if (not ok) or step <= 2 or step % every == 0 or step == steps or day_bdry:
            report(f"s{step}·d{int(step*DT//86400)}" if day_bdry else f"s{step}", d, dts)
        if not ok:
            print(f"\nFAIL ({label}) at step {step}: {why}", flush=True)
            return False, state
    print(f"PASS ({label}): {steps} steps stable; worst max|vel|={worst_vel:.3e}", flush=True)
    return True, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=480)         # 10 days at dt=1800
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--every", type=int, default=48)
    args = ap.parse_args()
    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, state0, op, cf, dates = build(args.year, args.steps)
    diag = make_diag(mesh)
    print(f"[setup] built in {time.time()-t0:.1f}s", flush=True)

    ok_tke, fin_tke = run("TKE", "tke", mesh, state0, op, cf, dates, diag, args.steps, args.every)
    ok_kpp, fin_kpp = run("KPP", "kpp", mesh, state0, op, cf, dates, diag, args.steps, args.every)

    discriminates = True
    if ok_tke and ok_kpp:
        sst_rms = surf_rms(mesh, fin_tke.T, fin_kpp.T)
        sss_rms = surf_rms(mesh, fin_tke.S, fin_kpp.S)
        discriminates = sst_rms > 1e-3
        print(f"\n[TKE↔KPP scheme difference after {args.steps*DT/86400:.2f} days] "
              f"surface SST RMS={sst_rms:.4e} °C   SSS RMS={sss_rms:.4e} psu", flush=True)
        print(f"  (TKE is genuinely distinct from KPP — "
              f"{'RESOLVED ✓' if discriminates else 'NOT resolved ✗'})", flush=True)

    ok = ok_tke and ok_kpp and discriminates
    print(f"\n  TKE stable={ok_tke}  KPP stable={ok_kpp}  discriminates={discriminates}")
    print("TKE_STABILITY_OK" if ok else "TKE_STABILITY_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
