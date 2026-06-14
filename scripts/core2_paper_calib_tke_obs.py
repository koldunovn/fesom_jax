"""CORE2 TKE→obs calibration — Paper-experiments Task D2a (the obs-application half).

Calibrate the TKE closure ``{tke_c_k, tke_c_eps}`` so the model's **MLD and SST** match WOA, via the
global adjoint through the FULL paper model (zstar+TKE+mEVP+GM) with the frozen-ice adjoint (A8). The
perfect-model twin (``core2_paper_calib_tke.py``) proved the inversion is well-posed; this applies it
to real observations through the differentiable obs operator (:mod:`fesom_jax.obs_compare`).

**STATUS (2026-06-14): WORKS in the all-on model.** Earlier the loop died at iteration 2 with
``RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN`` — which looked like a memory OOM but was a
**weak-type recompile**: ``jnp.asarray(1.0)`` makes ``u0`` *weakly* typed, the first Adam update
strengthens it, and that type change re-triggers ``jit_step`` compilation at it=2 → a 2nd CUBIN
loaded atop it=1's → OOM. (The twin never hit it: its init was explicitly ``jnp.float64``.) Fix:
strong-typed ``u0`` (below) ⇒ no recompile ⇒ the loop runs (job 25601631, all-on N=12, 32 iters).
⚠️ The **2-parameter** ``{c_k, c_eps}`` fit overfits: ``c_eps`` collapses to ~0 (unphysical) to
squeeze out misfit — the structural-bias compensation the plausibility report is designed to catch
(over a short window MLD is mainly controlled by ``c_k``, so ``c_eps`` is weakly constrained). Use
``--params ck`` (well-constrained) for the headline, or bound ``c_eps``; held-out validation (D2c)
also catches it.

**Obs target = WOA monthly-derived MLD + SST** (de Boyer Montégut is IFREMER-blocked). The MLD is
computed from **monthly** WOA T/S with the model's *own* diagnostic and averaged
(``scripts/make_woa_targets.py`` → ``woa_targets.npz``) — the per-month-then-average that avoids the
Jensen bias of the annual-mean profile. ⚠️ WOA's smoothed monthly climatology still under-represents
sporadic deep convection vs dBM's individual profiles — documented; the ice mask drops the
(winter-ice-covered) deep-convection cells from the surface misfit anyway.

**Spin-up/adjoint split (the methodology).** The PHC initial state is not model-consistent and a 6-h
window cannot build a climatological MLD, so we (1) **spin up forward, no AD** (``--n-spin`` steps,
default params) to a model-consistent state ``S0``, then (2) take a **short adjoint window** from
``S0`` and tune ``{c_k, c_eps}`` to the WOA misfit. ``S0`` is ``stop_gradient``-ed (we never backprop
the spin-up). ⚠️ Spin-up detachment: ``S0`` was made with default ``c_k`` — this is a single
inner calibration; one outer spin-up↔calibrate iteration (re-spin with the recovered ``c_k``) is the
consistency check (reported, not yet looped). The short window calibrates the **fast** MLD/SST
*response* (the deep-convection equilibrium is the slow target — EKI / a longer integration).

Loss = ``misfit_MLD/J0_MLD + w·misfit_SST/J0_SST`` (each normalized by its baseline ⇒ O(1), balanced,
well-conditioned), area-weighted on the WOA grid with a consistent open-water (ice) mask. We report
**recovered parameter VALUES + physical plausibility** (catches structural-bias compensation) and the
baseline→calibrated misfit. Token **TKE_CALIB_OK** = MLD (and SST) misfit reduced + ``c_k`` plausible.
Single-GPU only. Usage:  python scripts/core2_paper_calib_tke_obs.py --n-spin 480 --n 12
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (x64)
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fesom_jax import ale, calibrate, core2_forcing, ice, obs_compare, ssh
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
CK0 = float(Params.defaults().tke_c_k)              # 0.1
CEPS0 = float(Params.defaults().tke_c_eps)
WOA_NPZ = ROOT / "scripts" / "woa_targets.npz"
DEFAULT_RESULTS = ROOT / "scripts" / "calib_tke_obs_results.jsonl"


def peak_gb():
    pk = 0.0
    for d in jax.devices():
        try:
            pk = max(pk, ((d.memory_stats() or {}).get("peak_bytes_in_use") or 0) / 1e9)
        except Exception:
            pass
    return pk


def is_oom(e):
    s = str(e).upper()
    return "RESOURCE_EXHAUSTED" in s or "OUT_OF_MEMORY" in s or "memory" in str(e).lower()


def build(year, config):
    mesh = load_mesh(MESH_DIR)
    base = core2_initial_state(mesh, IC_DIR)
    if config == "all3":                    # full model; ⚠️ obs-calib backward OOMs (mEVP recompute)
        state = ice.seed_ice(base, mesh, np.asarray(base.T[:, 0]))
        cfgs = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(),
                    ice_cfg=IceConfig(whichEVP=1, adjoint_mode="frozen"), ale_cfg=AleConfig())
    elif config == "tkegm":                 # ocean-only: no mEVP forward-recompute in the backward
        state = base                        # (the obs MLD/SST calib is over ice-masked open water)
        cfgs = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(), ale_cfg=AleConfig())
    else:
        raise ValueError(f"unknown --config {config!r}")
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, year, sst_ic=np.asarray(base.T[:, 0]))
    return mesh, state, op, cf, cfgs        # return cf so spin/window forcing stack separately


def load_woa(mesh, month):
    """WOA MLD+SST target for ``month`` (1-12) on the WOA grid → Hmap + flattened cell targets."""
    d = np.load(WOA_NPZ)
    lat, lon = d["lat"], d["lon"]
    mi = month - 1
    mld = d["mld_monthly"][mi].reshape(-1)                       # [n_cells] (lat*lon, C-order)
    sst = d["sst_monthly"][mi].astype(np.float64).reshape(-1)
    mld_ok = d["mld_valid_monthly"][mi].reshape(-1)
    sst_ok = d["sst_valid"].reshape(-1)
    # AD masked-NaN rule: WOA land cells are NaN; even at weight 0, 0·NaN=NaN poisons the misfit.
    # Zero them (the *_ok masks drop them from the weighted sum anyway).
    mld = np.where(np.isfinite(mld) & mld_ok, mld, 0.0)
    sst = np.where(np.isfinite(sst) & sst_ok, sst, 0.0)
    obs_grid = obs_compare.ObsGrid(lat=lat, lon=lon, depth=np.array([10.0, 100.0]))  # depth unused (surface)
    Hmap = obs_compare.build_h_map(mesh, obs_grid)
    return Hmap, (jnp.asarray(mld), jnp.asarray(mld_ok), jnp.asarray(sst), jnp.asarray(sst_ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="adjoint calibration window (steps)")
    ap.add_argument("--n-spin", type=int, default=480, help="forward spin-up steps (no AD); 480=10 d")
    ap.add_argument("--month", type=int, default=1, help="WOA month to match (model starts Jan)")
    ap.add_argument("--config", choices=("all3", "tkegm"), default="tkegm",
                    help="tkegm = zstar+TKE+GM (ocean-only; obs calib fits — over ice-masked water); "
                         "all3 = +mEVP (the twins' config; obs-calib backward OOMs on mEVP recompute)")
    ap.add_argument("--params", choices=("ck", "ck_ceps"), default="ck_ceps")
    ap.add_argument("--w-sst", type=float, default=1.0, help="SST weight in the combined loss")
    ap.add_argument("--ice-sst", type=float, default=-1.0,
                    help="obs ice mask: drop cells with WOA SST below this °C (freezing ⇒ ice)")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--mode", choices=("spinup", "calibrate", "both"), default="both",
                    help="run the spin-up + calibration in SEPARATE processes (fresh allocator pool "
                         "for the backward — the spin-up's retained pool OOMs an in-process backward)")
    ap.add_argument("--s0", type=Path,
                    default=Path("/work/ab0995/a270088/port_jax/calib_tke_obs_S0.pkl"),
                    help="path to save/load the spun-up state S0 between processes (~1.9 GB → /work)")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--outdir", type=Path, default=ROOT / "scripts")
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, st0, op, cf, cfgs = build(args.year, args.config)
    fs = cf.static
    node_mask = jnp.asarray(mesh.node_layer_mask)
    all_dates = core2_forcing.dates_for_steps(args.year, DT, args.n_spin + args.n)
    Hmap, (woa_mld, woa_mld_ok, woa_sst, woa_sst_ok) = load_woa(mesh, args.month)
    rec = {"target": "tke_obs", "config": "zstar+TKE+mEVP+GM(frozen-ice)", "N": args.n,
           "n_spin": args.n_spin, "month": args.month, "params": args.params}
    print(f"[setup] built in {time.time()-t0:.1f}s; spin {args.n_spin} ({args.n_spin*DT/86400:.1f} d) "
          f"+ window {args.n}; WOA month {args.month:02d}", flush=True)

    # ---------- spin-up (forward, no AD) → S0 ----------
    # Run as a SEPARATE process (--mode spinup) so the spin-up's retained allocator pool (~24 GB,
    # which JAX's BFC never returns) doesn't starve the heavy all-on per-step backward (~36 GiB) —
    # the in-process version OOMs at every N. The calibrate process loads S0 with a fresh pool.
    if args.mode in ("spinup", "both"):
        try:
            t1 = time.time()
            sfs_spin = cf.stack(all_dates[:args.n_spin])
            S0 = integrate(st0, mesh, op, None, n_steps=args.n_spin, dt=DT, step_forcings=sfs_spin,
                           forcing_static=fs, params=calibrate.build_params({}), **cfgs)
            S0 = jax.lax.stop_gradient(S0)
            S0.T.block_until_ready()
            print(f"[spin] {args.n_spin} steps in {time.time()-t1:.1f}s  peak={peak_gb():.1f} GB",
                  flush=True)
        except Exception as e:
            return _bail(args, rec, "spinup", e)
        if args.mode == "spinup":
            import pickle
            with open(args.s0, "wb") as fpk:
                pickle.dump(jax.device_get(S0), fpk)
            print(f"[spin] saved S0 -> {args.s0}\nTKE_CALIB_SPINUP_OK", flush=True)
            return 0
        del sfs_spin
        gc.collect()
    else:                                            # --mode calibrate: load S0 (fresh pool)
        import pickle
        with open(args.s0, "rb") as fpk:
            S0 = jax.device_put(pickle.load(fpk))
        print(f"[calib] loaded S0 <- {args.s0}", flush=True)

    sfs_win = cf.stack(all_dates[args.n_spin:args.n_spin + args.n])
    # obs-based open-water (ice) mask — drop WOA cells below freezing (ice-covered; MLD/SST
    # unreliable + the ocean-only config has no ice physics there). Config-independent.
    _, cell_valid = obs_compare.to_obs_surface(jnp.ones((mesh.nod2D,)), Hmap)  # cell has a wet node
    ice_ok = woa_sst > args.ice_sst
    w_mld = Hmap.cell_area * woa_mld_ok * ice_ok * cell_valid
    w_sst = Hmap.cell_area * woa_sst_ok * ice_ok * cell_valid

    def model_surf(tun):
        fin = integrate(S0, mesh, op, None, n_steps=args.n, dt=DT, step_forcings=sfs_win,
                        forcing_static=fs, params=calibrate.build_params(tun), **cfgs)
        _, Z3d = ale.live_geometry(mesh, fin.hnode)
        mld, _ = obs_compare.mld_density_threshold(fin.T, fin.S, Z3d, node_mask)
        cell_mld, _ = obs_compare.to_obs_surface(mld, Hmap)
        cell_sst, _ = obs_compare.to_obs_surface(fin.T[:, 0], Hmap)
        return cell_mld, cell_sst

    def tun_of(u):                                   # scale-free leaves → the Params dict
        d = {"tke_c_k": CK0 * u["ck"]}
        if args.params == "ck_ceps":
            d["tke_c_eps"] = CEPS0 * u["ceps"]
        return d

    def misfits(u):
        cell_mld, cell_sst = model_surf(tun_of(u))
        return (obs_compare.misfit(cell_mld, woa_mld, w_mld),
                obs_compare.misfit(cell_sst, woa_sst, w_sst))

    # ---------- calibrate {c_k(, c_eps)} via the global adjoint ----------
    # ONE fwd+bwd executable (has_aux returns the MLD/SST misfits per iter) — NO separate baseline
    # forward, which would compile its own program + retain a ~24 GB pool and OOM the backward.
    # Fixed O(1) scales balance MLD (~1e3 m²) and SST (~1 °C²) without a baseline pre-pass; the
    # baseline is just the FIRST iter's misfits (default params).
    SCALE_MLD, SCALE_SST = 1000.0, 1.0
    # ⚠️ STRONG float64 (not jnp.asarray(1.0), which is WEAK): a weak→strong type change after the
    # first Adam update re-triggers jit_step compilation at it=2 → a 2nd CUBIN load on top of it=1's
    # → the "Failed to load in-memory CUBIN" OOM. (The twin's init was explicitly float64 ⇒ no recompile.)
    one = jnp.asarray(1.0, jnp.float64)
    u0 = {"ck": one} | ({"ceps": one} if args.params == "ck_ceps" else {})

    def loss_aux(u):
        jm, js = misfits(u)
        return jm / SCALE_MLD + args.w_sst * js / SCALE_SST, (jm, js)

    opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.iters))

    @jax.jit
    def step(u, opt_state):
        # ONE executable for the WHOLE update (value_and_grad + Adam) — separate optax executables
        # OOM on CUBIN-load when the fwd+bwd pool already fills the card (the v5 failure).
        (lval, (jm, js)), grads = jax.value_and_grad(loss_aux, has_aux=True)(u)
        gn = optax.tree.norm(grads)
        updates, opt_state = opt.update(grads, opt_state, u)
        return optax.apply_updates(u, updates), opt_state, lval, jm, js, gn

    try:
        t1 = time.time()
        print(f"\n  --- calibrate {args.params} to WOA MLD+SST (Adam lr={args.lr}, ≤{args.iters} it; "
              f"single jitted step) ---", flush=True)
        u, opt_state, hist = u0, opt.init(u0), []
        m0_mld = m0_sst = None
        for it in range(1, args.iters + 1):
            ck = CK0 * float(u["ck"])                          # the iterate being EVALUATED this step
            ceps = CEPS0 * float(u["ceps"]) if args.params == "ck_ceps" else CEPS0
            u, opt_state, lval, jm, js, gn = step(u, opt_state)
            jm, js, lval = float(jm), float(js), float(lval)
            if m0_mld is None:
                m0_mld, m0_sst = jm, js
            extra = f" c_eps={ceps:.4f}" if args.params == "ck_ceps" else ""
            print(f"    it={it:3d}  c_k={ck:.5f}{extra}  MLD={jm:.4e} m²  SST={js:.4e} °C²  "
                  f"|g|={float(gn):.2e}", flush=True)
            hist.append({"it": it, "ck": ck, "ceps": ceps, "jm": jm, "js": js, "loss": lval})
            # plateau early-stop (the obs misfit bottoms out at the irreducible model-obs gap)
            if it > 8 and hist[-9]["loss"] - lval < 1e-3 * hist[0]["loss"]:
                print(f"    (plateau — stop at it={it})", flush=True)
                break
        ck_fin, ceps_fin = hist[-1]["ck"], hist[-1]["ceps"]   # last EVALUATED iterate (jm/js consistent)
        mf_mld, mf_sst = hist[-1]["jm"], hist[-1]["js"]
        bwd_s = time.time() - t1
    except Exception as e:
        return _bail(args, rec, "calibrate", e)
    if not (m0_mld > 0):
        return _bail(args, rec, "calibrate", ValueError(f"degenerate baseline MLD {m0_mld}"))

    # plausibility bounds (physical TKE coefficient ranges)
    ck_ok = bool(0.05 <= ck_fin <= 0.30)
    ceps_ok = bool(0.5 * CEPS0 <= ceps_fin <= 2.0 * CEPS0) if args.params == "ck_ceps" else True
    mld_red = mf_mld < m0_mld
    sst_red = mf_sst < m0_sst
    pk = peak_gb()
    print(f"\n  recovered c_k={ck_fin:.5f}"
          f"{' c_eps=%.4f'%ceps_fin if args.params=='ck_ceps' else ''}  "
          f"(plausible: c_k={ck_ok}{', c_eps=%s'%ceps_ok if args.params=='ck_ceps' else ''})", flush=True)
    print(f"  MLD misfit {m0_mld:.4e} -> {mf_mld:.4e} m²  ({100*(1-mf_mld/m0_mld):+.1f}%)", flush=True)
    print(f"  SST misfit {m0_sst:.4e} -> {mf_sst:.4e} °C²  ({100*(1-mf_sst/m0_sst):+.1f}%)  "
          f"in {len(hist)} it / {bwd_s:.1f}s  peak={pk:.1f} GB", flush=True)

    ok = bool(mld_red and ck_ok)            # MLD reduced + c_k physically plausible (D2c sets the bar)
    rec.update(dict(ck0=CK0, ceps0=CEPS0, ck_fin=ck_fin, ceps_fin=ceps_fin, ck_ok=ck_ok,
                    ceps_ok=ceps_ok, m0_mld=m0_mld, mf_mld=mf_mld, m0_sst=m0_sst, mf_sst=mf_sst,
                    mld_red=mld_red, sst_red=sst_red, n_iters=len(hist), peak_gb=pk, ok=ok))
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez(args.outdir / "calib_tke_obs.npz",
             hist_it=np.array([r["it"] for r in hist]),
             hist_ck=np.array([r["ck"] for r in hist]),
             hist_mld=np.array([r["jm"] for r in hist]),
             hist_sst=np.array([r["js"] for r in hist]),
             hist_loss=np.array([r["loss"] for r in hist]),
             ck0=CK0, ck_fin=ck_fin, ceps_fin=ceps_fin, m0_mld=m0_mld, mf_mld=mf_mld,
             m0_sst=m0_sst, mf_sst=mf_sst, month=args.month, n_spin=args.n_spin)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  gate: MLD reduced={mld_red}  c_k plausible={ck_ok}  (SST reduced={sst_red})", flush=True)
    print(f"TKE_CALIB_{'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def _bail(args, rec, phase, e):
    oom = is_oom(e)
    rec.update(dict(ok=False, phase=phase, oom=oom, peak_gb=peak_gb(),
                    error=f"{type(e).__name__}: {str(e)[:240]}"))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  FAILED in {phase}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print(f"TKE_CALIB_{'OOM' if oom else 'FAIL'}", flush=True)
    return 2 if oom else 1


if __name__ == "__main__":
    raise SystemExit(main())
