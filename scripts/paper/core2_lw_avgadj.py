"""Long-window Task D1 — the ENSEMBLE-AVERAGED climate sensitivity (multi-target, adjoint OR tangent-linear).

The Part-D payoff (C1 found N* ≪ N_blow ⇒ simple averaging is viable). Seed K SHORT frozen-ice
bursts (window N = the C1 clean window, default 24 = 0.5 d) at states spread along a reference
trajectory; each burst returns the **field** sensitivity of a window-mean climate diagnostic;
stream-average them (the :mod:`fesom_jax.longwindow` seam) → the climate-timescale sensitivity
(the [nod2D] map + a global scalar), with across-burst SE + convergence + a MAD robust filter.

**``--mode`` picks the AD direction — the two are TRANSPOSES about the SAME burst** (one reference,
one forward integration, two readouts; the chaotic N_blow horizon + the ensemble-averaging apply to
both — forward vs reverse changes only WHICH dimension is cheap, not the conditioning):
  * ``adjoint`` (reverse-mode ``jax.grad``, DEFAULT) — param is a [nod2D] FIELD, observable a SCALAR ⇒
    map = ``d(scalar diagnostic)/d(param_i)`` ("where to tune the param to move a global metric");
    scalar summary = Σ map. Needs the √N-checkpointed tape (peak ~43 GB at N=48).
  * ``tlm`` (forward-mode ``jax.jvp``) — param is a SCALAR (one global knob), observable a [nod2D] FIELD ⇒
    map = ``d(field_j)/d(scalar param)`` ("the spatial fingerprint of one global knob"); scalar summary =
    the area/volume-weighted mean of the map = ``dO/dθ`` for the SAME ``O``. No tape (leaner). The
    weighted-mean of the tlm map should match the adjoint Σ map — a free adjoint↔tangent (transpose) check.

**One reference run amortizes across MANY sensitivities** (``--target`` × ``--mode``): the all-on config
has TKE+GM+mEVP all live, so every parameter's path is active — only the observable + parameter shape
change. Targets:
  * ``mld_ck``    — ``d(window-mean MLD)/d(c_k)`` (FAST, mixing-driven). The C1-validated headline:
                    the averaged short-window adjoint is trustworthy (slow sign reached by 6 h).
  * ``t100_kgm``  — ``d(window-mean upper-ocean T, 0–`--upper-depth` m)/d(k_gm + redi_kmax)`` (SLOW,
                    eddy/restratification-driven). The adjoint-REACH test: GM redistributes heat over
                    months, so a 0.5–1 d burst barely sees it — this is the §2 adjoint↔EKI boundary at
                    the climate timescale. The averaged result MUST be validated vs brute-force FD
                    (D2-style k_gm±δ 10-yr forwards), with EKI as the cross-check/fallback.

Each burst restarts from a reference-snapshot State and reads forcing at THAT seed's date (forcing-
calendar alignment), seeded developed ⇒ ``is_first_step=False`` throughout. Single-GPU, frozen-ice
adjoint (the sharded ragged-halo AD bug ⇒ single-GPU; parallelism = many INDEPENDENT burst jobs).

Usage (GPU):
  python scripts/paper/core2_lw_avgadj.py --mode adjoint --target mld_ck   --snap-dir $WORK/longwindow/ref10_snaps --K 200
  python scripts/paper/core2_lw_avgadj.py --mode tlm     --target mld_ck   --snap-dir $WORK/longwindow/ref10_snaps --K 200
  python scripts/paper/core2_lw_avgadj.py --mode adjoint --target t100_kgm --snap-dir $WORK/longwindow/ref10_snaps --K 200
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import pickle
import re
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import ale, surface_forcing, longwindow, obs_compare, ssh
from fesom_jax.ale import AleConfig
from fesom_jax.calibrate import build_params
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.integrate import _run_steps
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.step import step
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist864"
DT = 1800.0
ALE_CFG = AleConfig()
TKE_CFG = TkeConfig()
GM_CFG = GMConfig()
ICE_FROZEN = IceConfig(whichEVP=1, adjoint_mode="frozen")
TARGETS = ("mld_ck", "t100_kgm")
DEFAULT_RESULTS = ROOT / "scripts" / "lw_avgadj_results.jsonl"


def peak_gb():
    pk = 0.0
    for d in jax.devices():
        try:
            pk = max(pk, ((d.memory_stats() or {}).get("peak_bytes_in_use") or 0) / 1e9)
        except Exception:
            pass
    return pk


def snap_date(snap_path: str, run_start_year: int):
    m = re.search(r"step(\d+)", Path(snap_path).name)
    step_n = int(m.group(1)) if m else 0
    d = datetime.datetime(int(run_start_year), 1, 1) + datetime.timedelta(seconds=step_n * DT)
    return d.year, d.month, d.day


def make_target(target, mesh, upper_depth, mode):
    """Return ``(params_from, theta0, observable, acc0, w_scalar, label, unit)`` for ``(target, mode)``.

    The two modes are **transposes about the SAME burst** (one reference, one forward, two readouts):

    * ``adjoint`` (reverse-mode, ``jax.grad``): ``theta0`` is the [nod2D(,1)] PARAMETER FIELD and
      ``observable(state) -> scalar`` reduces a State to a SCALAR climate diagnostic => the burst returns
      the [nod2D] sensitivity MAP ``d(scalar)/d(field_i)`` -- "where to tune the param to move a global metric".
    * ``tlm`` (forward-mode, ``jax.jvp``): ``theta0`` is a SCALAR (one global param value) and
      ``observable(state) -> [nod2D] FIELD`` => the burst returns the [nod2D] response MAP
      ``d(field_j)/d(scalar)`` -- "the spatial fingerprint of turning one global knob".

    ``acc0`` is the matching window accumulator (scalar vs [nod2D] field); ``w_scalar`` is the per-node
    weight that reduces a TLM map to its global scalar summary ``Sum w*map / Sum w`` (= ``dO/dtheta`` for the
    same observable ``O`` the adjoint reduces to) -- ``None`` in adjoint mode, where the scalar = ``Sum map``.
    Same all-on config drives both modes; only the parameter shape + the observable's reduction differ."""
    node_mask = jnp.asarray(mesh.node_layer_mask)
    area0 = jnp.asarray(np.asarray(mesh.area[:, 0], dtype=np.float64))
    node_wet = np.asarray(mesh.node_layer_mask[:, 0], dtype=np.float64)
    N2D = int(mesh.nod2D)
    tlm = (mode == "tlm")

    if target == "mld_ck":
        theta_s = float(Params.defaults().tke_c_k)

        def mld_of(state):
            _, Z3d = ale.live_geometry(mesh, state.hnode)
            return obs_compare.mld_density_threshold(state.T, state.S, Z3d, node_mask)  # (mld, valid)

        if tlm:
            theta0 = jnp.asarray(theta_s, jnp.float64)                 # single global c_k

            def params_from(scalar):                                  # broadcast scalar -> field
                return build_params({"tke_c_k": jnp.full((N2D, 1), scalar)})

            def observable(state):                                    # [N2D] MLD field [m]
                mld, valid = mld_of(state)
                return jnp.where(valid, mld, 0.0)

            acc0 = jnp.zeros((N2D,), jnp.float64)
            w_scalar = np.asarray(area0) * node_wet                   # surface area (wet) for the global mean
            return params_from, theta0, observable, acc0, w_scalar, "d(MLD field)/d(c_k)", "m"

        theta0 = jnp.full((N2D, 1), theta_s, jnp.float64)             # [N,1] c_k field

        def params_from(field):                                       # [N,1]
            return build_params({"tke_c_k": field})

        def observable(state):                                        # area-weighted mean MLD [m] (scalar)
            mld, valid = mld_of(state)
            w = area0 * valid
            den = jnp.sum(w)
            return jnp.sum(w * mld) / jnp.where(den > 0, den, 1.0)

        return params_from, theta0, observable, jnp.asarray(0.0, jnp.float64), None, "d(mean MLD)/d(c_k)", "m"

    if target == "t100_kgm":
        theta_s = float(Params.defaults().k_gm)
        # volume weight over the top `upper_depth` m (area x layer thickness, wet layers)
        Z = np.asarray(mesh.Z, dtype=np.float64)
        zbar = np.asarray(mesh.zbar, dtype=np.float64)
        nz1 = Z.size
        dz = zbar[:nz1] - zbar[1:nz1 + 1]
        in_band = (np.abs(Z) <= float(upper_depth)).astype(np.float64)
        lyr_mask = np.asarray(mesh.node_layer_mask)[:, :nz1].astype(np.float64)
        vol_w = jnp.asarray((np.asarray(mesh.area[:, 0])[:, None]) * (dz * in_band)[None, :] * lyr_mask)
        col_den = jnp.sum(vol_w, axis=1)                              # [N2D] per-node band volume
        den_vol = jnp.sum(vol_w)

        if tlm:
            theta0 = jnp.asarray(theta_s, jnp.float64)                # single global k_gm

            def params_from(scalar):                                  # broadcast -> k_gm AND C-synced redi_kmax
                return build_params({"k_gm": jnp.full((N2D,), scalar),
                                     "redi_kmax": jnp.full((N2D,), scalar)})

            def observable(state):                                    # [N2D] column-mean T over the band [degC]
                num = jnp.sum(vol_w * state.T[:, :nz1], axis=1)
                return num / jnp.where(col_den > 0, col_den, 1.0)

            acc0 = jnp.zeros((N2D,), jnp.float64)
            w_scalar = np.asarray(col_den)                            # band volume per node for the global mean
            return params_from, theta0, observable, acc0, w_scalar, \
                f"d(T field 0-{upper_depth:.0f}m)/d(k_gm)", "degC"

        theta0 = jnp.full((N2D,), theta_s, jnp.float64)               # [N] k_gm field

        def params_from(field):                                       # [N]; k_gm AND the C-synced redi_kmax
            return build_params({"k_gm": field, "redi_kmax": field})

        def observable(state):                                        # volume-weighted mean T over 0-upper_depth m
            return jnp.sum(vol_w * state.T[:, :nz1]) / jnp.where(den_vol > 0, den_vol, 1.0)

        return params_from, theta0, observable, jnp.asarray(0.0, jnp.float64), None, \
            f"d(mean T 0-{upper_depth:.0f}m)/d(k_gm)", "degC"

    raise ValueError(f"unknown target {target!r}; choose {TARGETS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS, default="mld_ck")
    ap.add_argument("--mode", choices=("adjoint", "tlm"), default="adjoint",
                    help="adjoint=reverse-mode d(scalar)/d(param field) [where-to-tune]; "
                         "tlm=forward-mode d(response field)/d(scalar param) [spatial fingerprint]. "
                         "Same reference/seeds/window — only grad↔jvp + the observable's reduction differ.")
    ap.add_argument("--snap-dir", required=True, help="reference-trajectory snapshot dir (snap_step*.pkl)")
    ap.add_argument("--K", type=int, default=200, help="number of ensemble bursts (seeds)")
    ap.add_argument("--n", type=int, default=48, help="burst window (steps); 48 = 1 d (within the C1 "
                    "clean horizon N_blow=96; the edge — summer seeds noisier). 24 = 0.5 d (cleanest).")
    ap.add_argument("--run-start-year", type=int, default=1963, help="JRA year the reference run started")
    ap.add_argument("--upper-depth", type=float, default=100.0, help="t100_kgm: upper-ocean band depth [m]")
    ap.add_argument("--amp-factor", type=float, default=5.0, help="drop bursts with |scalar−median| > "
                    "this × MAD (amplified past their clean horizon; ~5 MAD ≈ 3.5 modified-z); 0 ⇒ no filter")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-map", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", action="store_true", help="jitter the seed indices within sub-intervals")
    args = ap.parse_args()
    n = args.n
    out_map = args.out_map or (ROOT / "scripts" / f"lw_avgadj_{args.target}_{args.mode}_map.npz")

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    sst0 = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    geo = np.degrees(np.asarray(mesh.geo_coord_nod2D))
    segments = max(2, int(round(math.sqrt(max(1, n)))))

    params_from, theta0, observable, acc0, w_scalar, label, unit = \
        make_target(args.target, mesh, args.upper_depth, args.mode)
    N2D = int(mesh.nod2D)
    # per-node weight that reduces a [nod2D] MAP to its global scalar summary:
    #   adjoint -> Σ map (u=1); tlm -> Σ ŵ·map (u = normalized w_scalar) = dO/dθ for the same O.
    u = (np.ones(N2D) if w_scalar is None else np.asarray(w_scalar) / np.sum(w_scalar))

    files = sorted(glob.glob(str(Path(args.snap_dir) / "snap_step*.pkl")))
    if not files:
        print(f"no snapshots in {args.snap_dir}"); return 1
    rng = np.random.default_rng(args.seed) if args.jitter else None
    idx = longwindow.spread_indices(len(files), args.K, rng)
    seeds = [files[int(i)] for i in idx]
    print(f"[setup] built in {time.time()-t0:.1f}s; mode={args.mode} target={args.target} ({label}); "
          f"{len(files)} ref snaps → K={len(seeds)} bursts; N={n} ({n*DT/86400:.2f} d) √N-seg={segments}  "
          f"frozen-ice {args.mode}, all-on config", flush=True)

    def window_mean(theta, state0, fs, sfs):
        """Time-mean over the N-step burst of the target observable. ``theta`` and the observable's
        shape follow the mode: adjoint = (param FIELD → SCALAR obs); tlm = (param SCALAR → FIELD obs).
        The differentiation transform (grad vs jvp) reads the matching [nod2D] map off this one fn."""
        p = params_from(theta)

        def body(carry, sf):
            s, acc = carry
            nxt = step(s, mesh, op, None, p, dt=DT, is_first_step=False, step_forcing=sf,
                       forcing_static=fs, ice_cfg=ICE_FROZEN, gm_cfg=GM_CFG, tke_cfg=TKE_CFG,
                       ale_cfg=ALE_CFG)
            return (nxt, acc + observable(nxt)), None

        # adjoint needs the √N-checkpointed tape for the backward; tlm (forward-mode) carries the
        # tangent alongside the primal ⇒ no tape ⇒ a plain scan (checkpoint=False) is leaner.
        _, acc = _run_steps(body, (state0, acc0), xs=sfs,
                            checkpoint=(args.mode == "adjoint"), segments=segments)
        return acc / n

    if args.mode == "adjoint":
        _burst = jax.jit(jax.grad(window_mean, argnums=0))          # reverse-mode: d(scalar)/d(field)

        def burst_map(st0, fs, sfs):
            return np.asarray(_burst(theta0, st0, fs, sfs)).reshape(-1)
    else:
        @jax.jit
        def _burst(theta, st0, fs, sfs):                            # forward-mode: d(field)/d(scalar)
            _, tan = jax.jvp(lambda t: window_mean(t, st0, fs, sfs), (theta,), (jnp.ones_like(theta),))
            return tan

        def burst_map(st0, fs, sfs):
            return np.asarray(_burst(theta0, st0, fs, sfs)).reshape(-1)

    cf_cache = {}

    def forcing_for(yr, mo, dy):
        if yr not in cf_cache:
            cf_cache[yr] = surface_forcing.build_surface_forcing(mesh, yr, sst_ic=sst0)
        cf = cf_cache[yr]
        return cf.static, cf.stack(surface_forcing.dates_for_steps(yr, DT, n, start_month=mo, start_day=dy))

    maps, scalars, dates = [], [], []
    t_bursts = time.time()
    for bi, sp in enumerate(seeds):
        yr, mo, dy = snap_date(sp, args.run_start_year)
        with open(sp, "rb") as f:
            st0 = jax.device_put(pickle.load(f))
        fs, sfs = forcing_for(yr, mo, dy)
        try:
            g = burst_map(st0, fs, sfs)
        except Exception as e:
            print(f"  burst {bi} ({Path(sp).name}): FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            del st0
            continue
        s = float(np.sum(u * g))
        maps.append(g); scalars.append(s); dates.append(f"{yr}-{mo:02d}-{dy:02d}")
        del st0
        if bi < 3 or (bi + 1) % 10 == 0:
            print(f"  burst {bi+1}/{len(seeds)} {dates[-1]}: scalar={s:+.4e}  "
                  f"running⟨g⟩={np.mean(scalars):+.4e}  |map|max={np.max(np.abs(g)):.2e}  "
                  f"peak={peak_gb():.0f}GB  ({(time.time()-t_bursts)/(bi+1):.0f}s/burst)", flush=True)

    if not maps:
        print("no successful bursts"); return 1
    # ROBUST FILTER — only bursts WITHIN their clean horizon are valid ensemble-adjoint samples.
    # N_blow=96 (C1) is the AVERAGE blow-up; individual (unstable summer) seeds amplify earlier, so
    # at N=48 (1 d, the horizon edge) a few bursts blow up and would dominate the plain mean (the
    # smoke saw mean +52 vs median +3). Drop |g| ≫ median (×amp_factor); a no-op at N=24 (all clean).
    scal_all = np.asarray(scalars)
    med = float(np.median(scal_all))
    mad = float(np.median(np.abs(scal_all - med)))          # robust scale (median abs deviation)
    keep = (np.ones(scal_all.size, bool) if args.amp_factor <= 0
            else (np.abs(scal_all - med) <= args.amp_factor * max(mad, 1e-300)))
    n_drop = int((~keep).sum())
    kept_maps = [mp for mp, k in zip(maps, keep) if k]
    kept_scal = scal_all[keep]

    stats, nb = longwindow.average_grads(iter(kept_maps))
    mean_map = np.asarray(stats["mean"]).reshape(-1)
    se_map = np.asarray(stats["stderr"]).reshape(-1)
    scalar_grad = float(np.sum(u * mean_map))                # adjoint: Σ map; tlm: Σ ŵ·map (global mean)
    scalar_se = float(np.sqrt(np.sum((u * se_map) ** 2)))
    conv = longwindow.convergence(kept_scal, tol=0.05)
    finite = bool(np.isfinite(mean_map).all())
    nz = int(np.sum(mean_map != 0.0))
    # the (filtered) mean should now agree with the robust median ⇒ well-determined
    mean_med_agree = bool(abs(scalar_grad - med) <= 0.25 * max(abs(med), 1e-300))

    scalar_kind = ("Σ map (global scalar)" if w_scalar is None else "area/vol-weighted mean (global)")
    print(f"\n=== D1 ensemble-averaged {args.mode} — {label} (K={nb}/{len(scal_all)} kept, "
          f"N={n}={n*DT/86400:.2f} d) ===", flush=True)
    print(f"  {label} = {scalar_grad:+.5e} {unit}  ± {scalar_se:.2e} (across-burst SE) [{scalar_kind}]", flush=True)
    print(f"  map: finite={finite} nonzero={nz}/{mean_map.size} |max|={np.max(np.abs(mean_map)):.3e}", flush=True)
    print(f"  per-burst (all {len(scal_all)}): mean={np.mean(scal_all):+.4e} median={med:+.4e} "
          f"std={np.std(scal_all):.4e} min={np.min(scal_all):+.3e} max={np.max(scal_all):+.3e}", flush=True)
    print(f"  robust filter (|g−med|>{args.amp_factor:g}×MAD): dropped {n_drop}/{len(scal_all)} amplifying; "
          f"kept-mean≈median={mean_med_agree}", flush=True)
    print(f"  convergence: n_stable={conv['n_stable']}/{nb} (running mean within 5% of final)", flush=True)
    slow_sign = bool(scalar_grad > 0)
    stabilized = bool(conv["n_stable"] < nb)
    ok = bool(finite and nz > 0 and stabilized and mean_med_agree)   # sign is target-dependent (not gated)

    out_map.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_map, mean_map=mean_map, se_map=se_map, lon=geo[:, 0], lat=geo[:, 1],
             node_wet=np.asarray(mesh.node_layer_mask[:, 0]), mode=args.mode, target=args.target,
             label=label, unit=unit, scalar_grad=scalar_grad, scalar_se=scalar_se,
             scalars=np.asarray(scalars), running_mean=conv["running_mean"], K=nb, N=n)
    rec = dict(task="D1", mode=args.mode, target=args.target, label=label, unit=unit, K_kept=nb,
               K_total=len(scal_all), N=n, days=n * DT / 86400.0, snap_dir=str(args.snap_dir),
               scalar_grad=scalar_grad, scalar_se=scalar_se, sign_pos=slow_sign, n_stable=conv["n_stable"],
               map_finite=finite, map_nonzero=nz, burst_mean_all=float(np.mean(scal_all)),
               burst_median=med, burst_std=float(np.std(scal_all)), n_dropped=n_drop,
               amp_factor=args.amp_factor, mean_med_agree=mean_med_agree,
               t_sec=round(time.time() - t0, 1), ok=ok)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  wrote map → {out_map}; jsonl → {args.results}", flush=True)
    tag = "ADJ" if args.mode == "adjoint" else "TLM"
    print(f"AVG_{tag}_SENS_{args.target.upper()}_{'OK' if ok else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
