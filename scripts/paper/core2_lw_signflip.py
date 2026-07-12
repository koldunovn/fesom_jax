"""Long-window Task C1 — the frozen-ice clean-gradient horizon + the SIGN-FLIP test.

THE make-or-break measurement of the long-window plan. On a single GPU, with the **frozen-ice
adjoint** (the mEVP rheology adjoint is the dominant blow-up source — its forward still runs,
only the backward is stop_gradient'd) and the **full paper config** (zstar + TKE + mEVP + GM),
seed a SHORT adjoint burst at a **developed seasonal state** (a deep-winter-convection snapshot,
or a summer one) and measure, as the window length ``N`` grows:

  (a) ``d(window-mean MLD)/d(c_k)`` — its **value and SIGN** (window-mean = the time-average of
      the area-weighted mean MLD over the N-step burst — the climate-relevant observable D1 averages);
  (b) the gradient norm + FD↔AD agreement (the "is the gradient still clean / not yet chaotic" probe);
  (c) peak backward memory (O(√N) two-level checkpointing via integrate's ``_run_steps`` past N≈20).

**The decisive question.** Does the gradient flip from the FAST (negative / wrong — §3's "less
mixing") sign to the SLOW (positive / right — D2a's validated "more mixing DEEPENS MLD") sign
*before* it blows up? Record the flip window ``N*`` and the blow-up window ``N_blow``:
  * ``N* < N_blow``  ⇒ a clean window reaches the slow sign ⇒ simple ensemble-averaging is viable
    (Part D).  ``SIGNFLIP_HORIZON_OK``.
  * ``N* ≥ N_blow`` (or no flip) ⇒ escalate WITHIN the adjoint family (Part E) — NOT a switch to EKI.
The measured ``N*`` / ``N_blow`` is itself a clean result (it bounds the simple-averaging horizon).

Because the seed is a DEVELOPED snapshot (it already carries the AB2 history), every burst step
runs ``is_first_step=False`` — a uniform scan, no eager first step. The burst reads forcing at its
**seed date** (derived from the snapshot's elapsed step, or set explicitly) — NOT a hardcoded year.

ONE (seed, N) per process (peak memory is a process-global high-water mark — a fresh process per N
is the only clean per-N measurement, as in the A7 sweep). The ``.sbatch`` loops N per seed; each run
appends a JSON line; the figure assembles sign + |grad| vs N. Single-GPU only (the sharded ragged-
halo AD bug). Heavy per-window gradients (if saved) under ``$WORK/longwindow/signflip/``.

Usage (GPU):
  python scripts/paper/core2_lw_signflip.py --snap $WORK/foundation_snaps/snap_step004320.pkl \\
      --season winterNH --n 48 --results scripts/lw_signflip_results.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import ale, core2_forcing, obs_compare, ssh
from fesom_jax.ale import AleConfig
from fesom_jax.calibrate import build_params
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.integrate import _run_steps
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.step import step
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist864"        # the 864r oracle IC (matches the foundation/spin-up)
DT = 1800.0
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
ALE_CFG = AleConfig()
TKE_CFG = TkeConfig()
GM_CFG = GMConfig()
# the full paper config, FROZEN-ICE adjoint (forward = full mEVP, backward skips the ice rheology)
ICE_FROZEN = IceConfig(whichEVP=1, adjoint_mode="frozen")
DEFAULT_RESULTS = ROOT / "scripts" / "lw_signflip_results.jsonl"


def peak_gb():
    pk = 0.0
    for d in jax.devices():
        try:
            pk = max(pk, ((d.memory_stats() or {}).get("peak_bytes_in_use") or 0) / 1e9)
        except Exception:
            pass
    return pk


def gpu_limit_gb():
    for d in jax.devices():
        try:
            lim = (d.memory_stats() or {}).get("bytes_limit")
            if lim:
                return lim / 1e9
        except Exception:
            pass
    return 80.0


def seed_date(snap_path: str, run_start_year: int, month=None, day=None):
    """The burst's forcing (year, month, day): explicit ``--seed-month/--seed-day`` if given, else
    derived from the snapshot's elapsed step (``snap_step<N>.pkl`` ⇒ date = run-start + N·dt)."""
    if month and day:
        return run_start_year, int(month), int(day)
    m = re.search(r"step(\d+)", Path(snap_path).name)
    step_n = int(m.group(1)) if m else 0
    d = datetime.datetime(int(run_start_year), 1, 1) + datetime.timedelta(seconds=step_n * DT)
    return d.year, d.month, d.day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True, help="seed State pickle (a developed seasonal snapshot)")
    ap.add_argument("--season", default="seed", help="label for the seed (e.g. winterNH, summerNH)")
    ap.add_argument("--n", type=int, required=True, help="burst window length (steps)")
    ap.add_argument("--run-start-year", type=int, default=1958, help="JRA year the source run started")
    ap.add_argument("--seed-month", type=int, default=0)
    ap.add_argument("--seed-day", type=int, default=0)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()
    n = args.n

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    import pickle
    with open(args.snap, "rb") as f:
        st0 = jax.device_put(pickle.load(f))
    yr, mo, dy = seed_date(args.snap, args.run_start_year, args.seed_month or None, args.seed_day or None)
    sst0 = np.asarray(core2_initial_state(mesh, IC_DIR).T[:, 0])          # forcing a_ice mask (fixed)
    cf = core2_forcing.build_core_forcing(mesh, yr, sst_ic=sst0)
    dates = core2_forcing.dates_for_steps(yr, DT, n, start_month=mo, start_day=dy)
    sfs = cf.stack(dates)
    fs = cf.static
    segments = max(2, int(round(math.sqrt(max(1, n)))))                   # O(√N) checkpointing
    node_mask = jnp.asarray(mesh.node_layer_mask)
    area0 = jnp.asarray(np.asarray(mesh.area[:, 0]))
    print(f"[setup] built in {time.time()-t0:.1f}s; seed={args.season} ({Path(args.snap).name}) "
          f"date={yr}-{mo:02d}-{dy:02d}  N={n} ({n*DT/86400:.2f} d)  √N-segments={segments}  "
          f"config=zstar+TKE+mEVP+GM frozen-ice-adjoint", flush=True)

    def mld_mean(state):
        """Area-weighted mean mixed-layer depth [m] of a State (de Boyer Montégut density-threshold)."""
        _, Z3d = ale.live_geometry(mesh, state.hnode)
        mld, valid = obs_compare.mld_density_threshold(state.T, state.S, Z3d, node_mask)
        w = area0 * valid
        den = jnp.sum(w)
        return jnp.sum(w * mld) / jnp.where(den > 0, den, 1.0)

    def window_mean_mld(c_k, state0):
        """The climate-relevant observable: the time-mean over the N-step burst of the area-weighted
        mean MLD, as a function of c_k. Seeded at a developed state ⇒ is_first_step=False throughout."""
        p = build_params({"tke_c_k": c_k})

        def body(carry, sf):
            s, acc = carry
            nxt = step(s, mesh, op, None, p, dt=DT, is_first_step=False, step_forcing=sf,
                       forcing_static=fs, ice_cfg=ICE_FROZEN, gm_cfg=GM_CFG, tke_cfg=TKE_CFG,
                       ale_cfg=ALE_CFG)
            return (nxt, acc + mld_mean(nxt)), None

        fin, acc = _run_steps(body, (state0, jnp.asarray(0.0, jnp.float64)), xs=sfs,
                              checkpoint=True, segments=segments)
        return acc / n

    ck0 = jnp.asarray(float(Params.defaults().tke_c_k), jnp.float64)      # 0.1
    gpu_gb = gpu_limit_gb()
    rec = {"season": args.season, "snap": Path(args.snap).name, "date": f"{yr}-{mo:02d}-{dy:02d}",
           "N": n, "days": n * DT / 86400.0, "dt": DT, "segments": segments,
           "config": "zstar+TKE+mEVP+GM", "adjoint": "frozen-ice", "gpu_gb": gpu_gb}
    try:
        loss_jit = jax.jit(window_mean_mld)
        grad_jit = jax.jit(jax.grad(window_mean_mld, argnums=0))
        l0 = float(loss_jit(ck0, st0))
        t1 = time.time()
        ad = float(grad_jit(ck0, st0))
        bwd_s = time.time() - t1
        pk = peak_gb()
        # FD↔AD agreement (the clean-gradient probe): smallest relative gap over h
        rows = []
        for h in H_SWEEP:
            ckp, ckm = ck0 * (1.0 + h), ck0 * (1.0 - h)
            fd = float((loss_jit(ckp, st0) - loss_jit(ckm, st0)) / (ckp - ckm))
            rows.append((h, fd))
        plateau = min(abs(ad - fd) / max(abs(fd), 1e-300) for _, fd in rows)
        best_fd = min(rows, key=lambda r: abs(ad - r[1]) / max(abs(r[1]), 1e-300))[1]
        fd_spread = float(np.std([fd for _, fd in rows]))
        slow_sign = bool(ad > 0.0)                # D2a: more mixing (↑c_k) DEEPENS MLD ⇒ +ve = slow/right
        # The MLD density-threshold crossing is piecewise-constant in the level it picks, so the
        # SECANT FD is noisy across h (kink straddling) — the SIGN is the trustworthy, decisive
        # quantity, and BLOW-UP is |grad| exploding (cf the all3 adjoint |g| 1e4→1e15), NOT a few-%
        # FD mismatch. So gate on |grad| magnitude + best-h FD SIGN agreement, not plateau<1%.
        BLOW = 1.0e2
        fd_sign_agree = bool(np.sign(ad) == np.sign(best_fd))
        blown = bool((not np.isfinite(ad)) or abs(ad) > BLOW)
        clean = bool(np.isfinite(ad) and ad != 0.0 and not blown and fd_sign_agree)
        rec.update(dict(mean_mld=l0, grad=ad, abs_grad=abs(ad), peak_gb=pk, bwd_s=bwd_s,
                        plateau=plateau, best_fd=best_fd, fd_sign_agree=fd_sign_agree,
                        fd_spread=fd_spread, slow_sign=slow_sign, blown=blown,
                        clean=clean, fits_gpu=bool(pk < gpu_gb), oom=False))
        print(f"\n  N={n} ({rec['days']:.2f} d) seed={args.season}: window-mean MLD={l0:.3f} m", flush=True)
        print(f"  d(window-mean MLD)/d(c_k) AD = {ad:+.6e}   SIGN = {'+slow/right' if slow_sign else '-fast/wrong'}"
              f"   |grad|={abs(ad):.3e}", flush=True)
        for h, fd in rows:
            print(f"      h={h:.0e}  FD={fd:+.6e}  rel={abs(ad-fd)/max(abs(fd),1e-300):.2e}")
        print(f"      plateau(min rel)={plateau:.3e}  clean={clean}  peak={pk:.2f}/{gpu_gb:.0f} GB  "
              f"bwd={bwd_s:.1f}s", flush=True)
    except Exception as e:
        msg = type(e).__name__ + ": " + str(e)[:200]
        is_oom = ("RESOURCE" in str(e).upper() or "memory" in str(e).lower() or "OOM" in str(e).upper())
        rec.update(dict(mean_mld=None, grad=None, peak_gb=peak_gb(), oom=is_oom,
                        clean=False, fits_gpu=False, error=msg))
        print(f"\n  N={n} seed={args.season}: FAILED ({msg})", flush=True)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  appended (seed={args.season}, N={n}) → {args.results}", flush=True)
    tok = "OOM" if rec.get("oom") else ("CLEAN" if rec.get("clean") else "BLOWN")
    print(f"SIGNFLIP_{args.season.upper()}_N{n}_{tok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
