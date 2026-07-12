"""CORE2 GM→T/S stratification calibration via EKI — Paper-experiments Task D2b (§2, the
SLOW-target half: the gradient-free pillar).

D1/D2a proved the **fast** targets (GM twin, TKE→MLD) are adjoint-reachable. The eddy
(GM/Redi) effect on **T/S stratification** is the opposite regime — a slow, near-equilibrium
quantity the short-window adjoint cannot reach (review MAJOR-1; the C1 adjoint↔EKI cross-check
measured the boundary). So this calibrates ``k_gm`` (the GM thickness diffusivity, with the
C-synced ``redi_kmax`` riding it) with **Ensemble Kalman Inversion** (:mod:`fesom_jax.eki`):
**forward-only**, ``vmap``/loop-parallel, and — crucially — **immune to the mEVP sea-ice
rheology adjoint instability** (A8) that forces the twins to freeze the ice. EKI runs the FULL
all-on model (zstar+TKE+mEVP+GM) with NO gradient at all, so the ice is fully live.

**Observable = basin-mean upper-ocean/thermocline T/S profiles.** The model T/S is regridded onto
the WOA grid by the tested differentiable operator (:func:`fesom_jax.obs_compare.to_obs`, live
zstar z-interp) then reduced to **5 latitude-band basins × 8 standard depths × {T,S}** = an
80-vector (:func:`fesom_jax.obs_compare.basin_mean_profiles`). The SAME fixed reduction (fixed
basin weights + a fixed common validity mask) hits the model and the obs target ⇒ the EKI
observable is a clean fixed *linear* readout of the regridded field; members differ only through
the physics. The low dimension keeps the EKI ensemble covariances (``C_gg`` 80×80) trivial.

Three modes:
  * ``--mode budget`` — time the all-on **forward** over ``--n`` steps (steps/s, peak GB) and
    project the GPU-hours for a production ``members × n_total × iters`` run. The A4/D2b compute
    budget (the EKI analogue of A7's adjoint-window de-risking). Forward-only ⇒ memory is ~flat
    in N (no tape), so the constraint is wall-clock, not memory — this measures it.
  * ``--mode twin`` — **perfect-model proof**: inject a truth ``--truth`` k_gm, freeze its
    basin-mean T/S profiles as the synthetic obs, and recover from a prior ensemble centred at
    ``--k0`` via EKI. A forward-only misfit-bowl scan confirms the argmin sits at the truth FIRST
    (well-posedness), then EKI recovers it. Token **GM_EKI_TWIN_OK**.
  * ``--mode obs`` — the **application**: EKI ``k_gm`` to **WOA basin-mean T/S** over a warm-started
    (spin-up, no AD) window. Reports the recovered VALUE + physical plausibility + misfit reduction.
    Token **GM_CALIB_OK**. ⚠️ The few-year equilibrium adjustment is a production run (its budget is
    the ``--mode budget`` output); an overnight window is a scoped **demonstration** of the obs
    machinery, framed (warm-start caveat) as a few-window adjustment, not a multi-decade equilibrium.

⚠️ Forcing memory caps a single pre-stacked window at ~weeks (``StepForcing`` is ~10 MB/step);
a multi-year window needs chunked re-stacking (the production path — noted, not built tonight).
Single-GPU (the sharded ragged-halo bug is forward-safe but we keep one device for clean timing).

Usage (GPU):  python scripts/paper/core2_paper_calib_gm_eki.py --mode twin --n 240 --members 16
              python scripts/paper/core2_paper_calib_gm_eki.py --mode budget --n 240
              python scripts/paper/core2_paper_calib_gm_eki.py --mode obs  --n 480 --n-spin 480
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import ale, calibrate, core2_forcing, eki, ice, obs_compare, ssh
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
KGM0 = float(Params.defaults().k_gm)                  # 1000 (K_GM_MAX)
WOA_TS_NPZ = ROOT / "scripts" / "woa_ts_targets.npz"
DEFAULT_RESULTS = ROOT / "scripts" / "calib_gm_eki_results.jsonl"

# 5 latitude-band basins (the GM/stratification observable; SO is where GM matters most)
LAT_BANDS = [(-90.0, -45.0), (-45.0, -15.0), (-15.0, 15.0), (15.0, 45.0), (45.0, 90.0)]
BASIN_NAMES = ["SO", "S-mid", "Trop", "N-mid", "N-high"]


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


def is_oom(e):
    s = str(e).upper()
    return "RESOURCE_EXHAUSTED" in s or "OUT_OF_MEMORY" in s or "memory" in str(e).lower()


def build(year):
    """Mesh + ice-seeded all-on state + forcing + the FULL all-on config dict (forward-only ⇒
    the mEVP ice is fully live; no frozen-ice adjoint needed — that is the whole point of EKI)."""
    mesh = load_mesh(MESH_DIR)
    base = core2_initial_state(mesh, IC_DIR)
    state = ice.seed_ice(base, mesh, np.asarray(base.T[:, 0]))
    cfgs = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(),
                ice_cfg=IceConfig(whichEVP=1), ale_cfg=AleConfig())   # exact ice (forward-only)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, year, sst_ic=np.asarray(base.T[:, 0]))
    return mesh, state, op, cf, cfgs


def obs_grid_and_basins(mesh):
    """The WOA obs grid (8 chosen levels) → Hmap + the fixed ``[n_basins, n_cells]`` basin
    weights (cell area × latitude-band membership)."""
    d = np.load(WOA_TS_NPZ)
    lat, lon, depth = d["lat"], d["lon"], d["depth"]
    n_lat, n_lon = lat.size, lon.size
    obs_grid = obs_compare.ObsGrid(lat=lat, lon=lon, depth=depth)
    Hmap = obs_compare.build_h_map(mesh, obs_grid)
    cell_lat = np.repeat(lat, n_lon)                      # [n_cells] (lat-major, C-order)
    cell_area = np.cos(np.radians(cell_lat))
    basin_w = np.zeros((len(LAT_BANDS), cell_lat.size))
    for b, (lo, hi) in enumerate(LAT_BANDS):
        basin_w[b] = cell_area * ((cell_lat >= lo) & (cell_lat < hi))
    return Hmap, jnp.asarray(basin_w), (lat, lon, depth, n_lat, n_lon)


def load_woa_ts_cells(n_lat, n_lon):
    """WOA T/S on the chosen levels → cell arrays ``[n_cells, n_depth]`` + validity (C-order,
    matching :func:`fesom_jax.obs_compare.build_h_map` cell ids ``lat*n_lon+lon``)."""
    d = np.load(WOA_TS_NPZ)
    T, S, valid = d["T"], d["S"], d["valid"]              # [n_depth, lat, lon]
    cT = np.transpose(T, (1, 2, 0)).reshape(n_lat * n_lon, -1)
    cS = np.transpose(S, (1, 2, 0)).reshape(n_lat * n_lon, -1)
    cv = np.transpose(valid, (1, 2, 0)).reshape(n_lat * n_lon, -1)
    return jnp.asarray(cT), jnp.asarray(cS), jnp.asarray(cv)


def make_observable(state0, mesh, op, fs, sfs, n, cfgs, Hmap, basin_w, common_valid):
    """``model_obs(k_gm) -> [d]`` (jitted, reused across members + iters): run the all-on forward
    ``n`` steps from ``state0`` with the candidate ``k_gm`` (redi_kmax synced), regrid T & S onto
    the WOA grid (live z-interp), reduce to basin-mean profiles with the FIXED common mask, and
    concatenate ``[basin_T | basin_S]``. ``checkpoint=False`` — forward-only, no AD tape."""
    nd = Hmap.obs_z.shape[0]

    @jax.jit
    def model_obs(k_gm):
        p = calibrate.build_params({"k_gm": k_gm, "redi_kmax": k_gm})
        fin = integrate(state0, mesh, op, None, n_steps=n, dt=DT, step_forcings=sfs,
                        forcing_static=fs, params=p, checkpoint=False, **cfgs)
        _, Z3d = ale.live_geometry(mesh, fin.hnode)
        cT, _ = obs_compare.to_obs(fin.T, Z3d, Hmap)
        cS, _ = obs_compare.to_obs(fin.S, Z3d, Hmap)
        pT = obs_compare.basin_mean_profiles(cT, common_valid, basin_w)   # [n_basins, nd]
        pS = obs_compare.basin_mean_profiles(cS, common_valid, basin_w)
        return jnp.concatenate([pT.reshape(-1), pS.reshape(-1)])          # [2*n_basins*nd]

    return model_obs, nd


def gamma_vector(nd, sigma_t, sigma_s):
    """Per-component obs-noise covariance diag: ``σ_T²`` on the T half, ``σ_S²`` on the S half."""
    nb = len(LAT_BANDS)
    gt = np.full(nb * nd, sigma_t ** 2)
    gs = np.full(nb * nd, sigma_s ** 2)
    return jnp.asarray(np.concatenate([gt, gs]))


def spin_up(state0, mesh, op, cf, fs, dates, cfgs):
    """Forward spin-up (no AD) to a model-consistent warm start S0 (default k_gm)."""
    if not len(dates):
        return state0
    sfs = cf.stack(dates)
    S0 = integrate(state0, mesh, op, None, n_steps=len(dates), dt=DT, step_forcings=sfs,
                   forcing_static=fs, params=calibrate.build_params({}), checkpoint=False, **cfgs)
    S0.T.block_until_ready()
    return jax.lax.stop_gradient(S0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("budget", "twin", "obs"), default="twin")
    ap.add_argument("--n", type=int, default=240, help="EKI/forward window (steps); 240=5 d")
    ap.add_argument("--n-spin", type=int, default=480, help="forward spin-up steps (no AD); 480=10 d")
    ap.add_argument("--members", type=int, default=16, help="EKI ensemble size")
    ap.add_argument("--iters", type=int, default=6, help="EKI iterations (1-D param converges fast)")
    ap.add_argument("--k0", type=float, default=1000.0, help="prior ensemble centre for k_gm")
    ap.add_argument("--sigma-lnk", type=float, default=0.30, help="prior log-k_gm spread")
    ap.add_argument("--truth", type=float, default=1500.0, help="twin: injected k_gm truth")
    ap.add_argument("--sigma-t", type=float, default=0.5, help="obs T noise σ [°C] (Γ weight)")
    ap.add_argument("--sigma-s", type=float, default=0.1, help="obs S noise σ [psu] (Γ weight)")
    ap.add_argument("--gamma-auto", action="store_true",
                    help="twin: set Γ = (gamma-eps · bowl signal)² per group (auto-scale ≪ signal)")
    ap.add_argument("--gamma-eps", type=float, default=0.05)
    ap.add_argument("--perturb-obs", action="store_true", help="stochastic EKI (perturbed obs)")
    ap.add_argument("--budget-members", type=int, default=16, help="budget: members to project")
    ap.add_argument("--budget-iters", type=int, default=8, help="budget: iters to project")
    ap.add_argument("--budget-years", type=float, default=3.0, help="budget: production window [yr]")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--outdir", type=Path, default=ROOT / "scripts")
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, st0, op, cf, cfgs = build(args.year)
    fs = cf.static
    gpu_gb = gpu_limit_gb()
    Hmap, basin_w, (lat, lon, depth, n_lat, n_lon) = obs_grid_and_basins(mesh)
    all_dates = core2_forcing.dates_for_steps(args.year, DT, args.n_spin + args.n)
    rec = {"target": "gm_eki", "mode": args.mode, "config": "zstar+TKE+mEVP+GM (forward-only)",
           "N": args.n, "n_spin": args.n_spin, "members": args.members, "iters": args.iters,
           "k0": args.k0, "depth": depth.astype(int).tolist(), "gpu_gb": gpu_gb}
    print(f"[setup] built in {time.time()-t0:.1f}s; mode={args.mode}  N={args.n} "
          f"({args.n*DT/86400:.2f} d)  spin={args.n_spin} ({args.n_spin*DT/86400:.1f} d)  "
          f"basins={BASIN_NAMES}  levels={depth.astype(int).tolist()}", flush=True)

    # ---------- warm-start spin-up (forward, no AD) ----------
    try:
        t1 = time.time()
        S0 = spin_up(st0, mesh, op, cf, fs, all_dates[:args.n_spin], cfgs)
        print(f"[spin] {args.n_spin} steps in {time.time()-t1:.1f}s  peak={peak_gb():.1f} GB",
              flush=True)
    except Exception as e:
        return _bail(args, rec, "spinup", e)

    sfs_win = cf.stack(all_dates[args.n_spin:args.n_spin + args.n]) if args.n > 0 else None

    # ---------- budget mode: time the forward, project GPU-hours, exit ----------
    if args.mode == "budget":
        # common_valid is irrelevant for timing — an all-ones mask is fine.
        cv = jnp.ones((Hmap.n_cells, Hmap.obs_z.shape[0]))
        model_obs, nd = make_observable(S0, mesh, op, fs, sfs_win, args.n, cfgs,
                                        Hmap, basin_w, cv)
        try:
            _ = model_obs(jnp.asarray(KGM0, jnp.float64)).block_until_ready()   # compile
            t1 = time.time()
            for _ in range(2):
                _ = model_obs(jnp.asarray(KGM0, jnp.float64)).block_until_ready()
            sec_per_fwd = (time.time() - t1) / 2.0
        except Exception as e:
            return _bail(args, rec, "budget", e)
        sps = args.n / sec_per_fwd
        n_prod = int(round(args.budget_years * 365 * 86400 / DT))
        fwds = args.budget_members * args.budget_iters
        gpu_h = fwds * n_prod / sps / 3600.0
        pk = peak_gb()
        print(f"\n  --- EKI compute budget (all-on forward, single A100) ---")
        print(f"      window N={args.n} ({args.n*DT/86400:.2f} d): {sec_per_fwd:.1f} s/forward  "
              f"⇒ {sps:.1f} steps/s  peak={pk:.1f}/{gpu_gb:.0f} GB", flush=True)
        print(f"      production: {args.budget_members} members × {args.budget_iters} iters × "
              f"{args.budget_years:g} yr ({n_prod} steps) = {fwds*n_prod:,} step-evals", flush=True)
        print(f"      ⇒ ~{gpu_h:.1f} GPU-h  (forward-only; memory ~flat in N — chunked re-stacking "
              f"needed past ~weeks of pre-stacked forcing)", flush=True)
        # also project a scoped overnight window for reference
        for yr in (0.25, 0.5, 1.0):
            ns = int(round(yr * 365 * 86400 / DT))
            print(f"        ({args.budget_members}×{args.budget_iters}, {yr:g} yr): "
                  f"~{fwds*ns/sps/3600.0:.1f} GPU-h", flush=True)
        rec.update(dict(sec_per_fwd=sec_per_fwd, steps_per_s=sps, peak_gb=pk,
                        budget_gpu_h=gpu_h, n_prod=n_prod, ok=True))
        _write(args, rec)
        print("GM_EKI_BUDGET_OK", flush=True)
        return 0

    # ---------- build the target y_obs + the fixed common validity mask ----------
    # reference run (default k_gm) gives the model-side validity; intersect with the obs validity.
    nd = Hmap.obs_z.shape[0]
    try:
        # model-side obs validity (≈run-independent: fixed bottom) from a single default forward
        p_ref = calibrate.build_params({})
        fin_ref = integrate(S0, mesh, op, None, n_steps=args.n, dt=DT, step_forcings=sfs_win,
                            forcing_static=fs, params=p_ref, checkpoint=False, **cfgs)
        _, Z3d_ref = ale.live_geometry(mesh, fin_ref.hnode)
        _, ref_valid = obs_compare.to_obs(fin_ref.T, Z3d_ref, Hmap)        # [n_cells, nd] bool
        ref_valid.block_until_ready()
    except Exception as e:
        return _bail(args, rec, "refrun", e)

    if args.mode == "twin":
        common_valid = ref_valid.astype(jnp.float64)
        model_obs, _ = make_observable(S0, mesh, op, fs, sfs_win, args.n, cfgs,
                                       Hmap, basin_w, common_valid)
        try:
            y_obs = model_obs(jnp.asarray(args.truth, jnp.float64))         # synthetic obs
            y_obs.block_until_ready()
        except Exception as e:
            return _bail(args, rec, "truth", e)
        print(f"[twin] injected truth k_gm={args.truth:.0f}; observable d={y_obs.shape[0]}", flush=True)
    else:  # obs
        cT, cS, woa_valid = load_woa_ts_cells(n_lat, n_lon)
        common_valid = (ref_valid & woa_valid).astype(jnp.float64)
        model_obs, _ = make_observable(S0, mesh, op, fs, sfs_win, args.n, cfgs,
                                       Hmap, basin_w, common_valid)
        pT = obs_compare.basin_mean_profiles(cT, common_valid, basin_w)
        pS = obs_compare.basin_mean_profiles(cS, common_valid, basin_w)
        y_obs = jnp.concatenate([pT.reshape(-1), pS.reshape(-1)])
        ncov = int(np.asarray(common_valid).sum())
        print(f"[obs] WOA basin T/S target; observable d={y_obs.shape[0]}  "
              f"common-valid cell-levels={ncov}", flush=True)

    # ---------- (1) forward-only misfit-bowl scan over k_gm (well-posedness + signal) ----------
    gamma = gamma_vector(nd, args.sigma_t, args.sigma_s)
    Ginv0 = 1.0 / np.asarray(gamma)
    grid = np.unique(np.r_[np.arange(600.0, 1801.0, 200.0), args.k0,
                           (args.truth if args.mode == "twin" else KGM0)]).astype(np.float64)

    def wmisfit(g):                                          # Γ-weighted obs misfit
        r = np.asarray(g) - np.asarray(y_obs)
        return float(np.sum(Ginv0 * r * r) / r.size)

    try:
        t1 = time.time()
        bowl_g = {k: np.asarray(model_obs(jnp.asarray(k, jnp.float64))) for k in grid}
    except Exception as e:
        return _bail(args, rec, "bowl", e)
    Js = np.array([wmisfit(bowl_g[k]) for k in grid])
    kmin = float(grid[int(np.argmin(Js))])
    print(f"\n  --- misfit bowl ({len(grid)} pts, {time.time()-t1:.1f}s) ---", flush=True)
    for k, J in zip(grid, Js):
        mark = "  <-- argmin" if abs(k - kmin) < 1e-9 else ""
        tag = "  (truth)" if (args.mode == "twin" and k == args.truth) else ""
        print(f"      k_gm={k:7.1f}  J={J:.6e}{mark}{tag}", flush=True)
    # signal scale across the bowl (per group) — used for auto-Γ + as a diagnostic
    G = np.stack([bowl_g[k] for k in grid])                  # [n_grid, d]
    half = G.shape[1] // 2
    sig_t = float(np.sqrt(np.mean(np.var(G[:, :half], axis=0))))
    sig_s = float(np.sqrt(np.mean(np.var(G[:, half:], axis=0))))
    print(f"      bowl signal RMS: T={sig_t:.3e} °C  S={sig_s:.3e} psu  "
          f"(vs Γ σ_T={args.sigma_t} σ_S={args.sigma_s})", flush=True)
    bowl_ok = bool(abs(kmin - args.truth) < 1e-6) if args.mode == "twin" else True
    rec.update(dict(grid_kmin=kmin, bowl_ok=bowl_ok, sig_t=sig_t, sig_s=sig_s))

    if args.gamma_auto:                                      # twin: Γ ≪ signal ⇒ full EKI steps
        st = max(args.gamma_eps * sig_t, 1e-6)
        ss = max(args.gamma_eps * sig_s, 1e-7)
        gamma = gamma_vector(nd, st, ss)
        print(f"      [gamma-auto] σ_T={st:.3e} σ_S={ss:.3e}", flush=True)

    # ---------- (2) EKI recovery ----------
    key = jax.random.PRNGKey(args.seed)
    key, sk = jax.random.split(key)
    theta0 = (np.log(args.k0) + args.sigma_lnk *
              np.asarray(jax.random.normal(sk, (args.members,), jnp.float64)))   # log k_gm
    theta0 = jnp.asarray(theta0[:, None])                    # [J, 1]

    hist = []

    def on_step(r):
        kg = float(np.exp(r["theta_mean"][0]))
        print(f"    it={r['it']:2d}  k_gm={kg:8.2f}  spread(lnk)={r['spread']:.3e}  "
              f"misfit={r['misfit']:.4e}", flush=True)
        hist.append({"it": r["it"], "k_gm": kg, "spread": r["spread"], "misfit": r["misfit"]})

    try:
        t1 = time.time()
        print(f"\n  --- EKI recover k_gm ({args.members} members, {args.iters} iters, "
              f"prior k0={args.k0:.0f}±lnσ{args.sigma_lnk}; sequential forward) ---", flush=True)
        theta_mean, eki_hist, ens = eki.eki_run(
            theta0, lambda th: model_obs(jnp.exp(th[0])), y_obs, gamma,
            n_iters=args.iters, key=key, perturb_obs=args.perturb_obs,
            on_step=on_step, map_fn=eki.sequential_eval)
        k_rec = float(np.exp(np.asarray(theta_mean)[0]))
        k_members = np.exp(np.asarray(ens)[:, 0])
        # final misfit (post-update ensemble mean prediction)
        g_fin = np.asarray(model_obs(jnp.asarray(k_rec, jnp.float64)))
        eki_s = time.time() - t1
    except Exception as e:
        return _bail(args, rec, "eki", e)

    J0 = float(Js[int(np.argmin(np.abs(grid - args.k0)))])   # bowl misfit at the prior centre
    Jf = wmisfit(g_fin)
    pk = peak_gb()
    misfit_ratio = Jf / J0 if J0 > 0 else float("nan")
    print(f"\n  recovered k_gm = {k_rec:.2f}  (members {k_members.mean():.1f}±{k_members.std():.1f})  "
          f"in {len(eki_hist)} iters / {eki_s:.1f}s  peak={pk:.1f}/{gpu_gb:.0f} GB", flush=True)
    print(f"  misfit J(k0={args.k0:.0f}) {J0:.4e} -> J(k_rec) {Jf:.4e}  (ratio {misfit_ratio:.3e})",
          flush=True)

    # ---------- gate ----------
    if args.mode == "twin":
        rel = abs(k_rec - args.truth) / abs(args.truth)
        recover_ok = bool(rel < 0.05)                        # within 5% of the planted truth
        ok = bool(bowl_ok and recover_ok and misfit_ratio < 0.5)
        print(f"  twin: recovered rel err {rel:.3%}  (truth {args.truth:.0f})  "
              f"bowl_ok={bowl_ok} recover_ok={recover_ok}", flush=True)
        rec.update(dict(truth=args.truth, k_rec=k_rec, rel=rel, recover_ok=recover_ok,
                        J0=J0, Jf=Jf, misfit_ratio=misfit_ratio, peak_gb=pk, eki_s=eki_s, ok=ok))
        token = "GM_EKI_TWIN"
    else:
        kgm_ok = bool(300.0 <= k_rec <= 3000.0)              # physical GM diffusivity range
        red_ok = bool(misfit_ratio < 1.0)                    # improved on the prior centre
        ok = bool(kgm_ok and red_ok)
        print(f"  obs: recovered k_gm={k_rec:.1f}  plausible(300–3000)={kgm_ok}  "
              f"misfit reduced={red_ok}", flush=True)
        rec.update(dict(k_rec=k_rec, k_members_mean=float(k_members.mean()),
                        k_members_std=float(k_members.std()), kgm_ok=kgm_ok, red_ok=red_ok,
                        J0=J0, Jf=Jf, misfit_ratio=misfit_ratio, peak_gb=pk, eki_s=eki_s, ok=ok))
        token = "GM_CALIB"

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez(args.outdir / f"calib_gm_eki_{args.mode}.npz",
             grid_k=grid, grid_J=Js, hist_it=np.array([h["it"] for h in hist]),
             hist_k=np.array([h["k_gm"] for h in hist]),
             hist_misfit=np.array([h["misfit"] for h in hist]),
             k_members=k_members, k_rec=k_rec, J0=J0, Jf=Jf, y_obs=np.asarray(y_obs),
             depth=depth, sig_t=sig_t, sig_s=sig_s, N=args.n, mode=args.mode,
             truth=(args.truth if args.mode == "twin" else np.nan))
    _write(args, rec)
    print(f"{token}_{'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def _write(args, rec):
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _bail(args, rec, phase, e):
    oom = is_oom(e)
    rec.update(dict(ok=False, phase=phase, oom=oom, peak_gb=peak_gb(),
                    error=f"{type(e).__name__}: {str(e)[:240]}"))
    _write(args, rec)
    print(f"\n  FAILED in {phase}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print(f"GM_EKI_{'OOM' if oom else 'FAIL'}", flush=True)
    return 2 if oom else 1


if __name__ == "__main__":
    raise SystemExit(main())
