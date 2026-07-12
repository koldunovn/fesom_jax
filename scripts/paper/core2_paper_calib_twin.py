"""CORE2 perfect-model ``k_gm`` twin — Paper-experiments Task D1 (§2 Calibration, the
adjoint-as-optimizer proof).

This is the first half of the §2 calibration pillar: a **perfect-model OSSE twin**. We inject a
"truth" by running the assembled global model with a known GM thickness-diffusivity
``k_gm = --truth`` (default 1500, the C-synced ``redi_kmax`` riding along), freeze that final
upper-ocean temperature field as the synthetic observation, then **recover** the injected value
from a deliberately wrong start (``--init`` = 800) by minimizing the model↔truth misfit with the
**global adjoint** (cosine-decay Adam through :func:`fesom_jax.calibrate.optimize`). Recovering
the planted parameter to ~2 % *is* the proof that the differentiable model's gradient is an
optimizer-grade signal — the short-window twin target IS adjoint-reachable (unlike the slow
*obs* GM→T/S equilibrium of D2b, which needs EKI; the adjoint↔EKI boundary A7/C1 measured).

**Proper ocean model, not a toy (user steer 2026-06-14).** The default config is the FULL
paper model — **zstar + TKE + mEVP + GM/Redi all live** (``--config all3``), sea ice seeded
(:func:`fesom_jax.ice.seed_ice`) — so the FORWARD is the real all-on ocean. ⚠️ The naive all-on
*adjoint* explodes through the **mEVP sea-ice rheology** (the 120-iter plastic-yield solver;
``|g|~9e51`` at N=12 — the classic VP/EVP sea-ice adjoint instability), even though the ocean-only
adjoint is clean (C1). So the twin uses the **frozen-ice adjoint** (``IceConfig.adjoint_mode="frozen"``,
Task A8): the full mEVP runs in the forward, but ``stop_gradient`` keeps the backward out of the ice
rheology — the gradient threads GM through the TKE mixing while treating the (forward-real) ice as
fixed. ``stop_gradient`` is the identity in the forward ⇒ the bowl is bit-identical to ``"exact"``.
(``--config tkegm`` = ocean-only no-ice control; ``gm`` = the C1 isolated path. The planned
free-drift adjoint will restore ice-momentum sensitivity — A8 Option B.) The window ``--n`` is small
and the ``.sbatch`` descends N on OOM; a perfect-model twin stays well-posed at any N (J=0 at truth).

Two pieces, both reusing the calibrate seam (:mod:`fesom_jax.calibrate`):
  1. **grid-scan the misfit bowl FIRST** (forward-only): confirm ``argmin_k J(k)`` sits at the
     injected truth — a well-posedness check before any backward is trusted (the bowl is convex
     near the truth: J(k)=‖T(k)−T(truth)‖²_w ≥ 0, =0 at k=truth).
  2. **recover** ``k_gm`` from ``--init`` via Adam on the normalized leaf ``u = k_gm/1000`` (so a
     scale-free cosine-decay schedule recovers 0.8→1.5), early-stopping on the misfit plateau.

J = area-weighted MSE of the **upper-ocean temperature** (top ``--band`` layers — the GM
heat-redistribution / stratification observable) between the candidate run and the frozen truth.
The model→obs operator (:mod:`fesom_jax.obs_compare`) is wired and ready, so swapping this
model-vs-model J for ``misfit(T, T_obs, …)`` vs WOA is a one-line change (the D2 obs application).

Outputs ``scripts/calib_twin_kgm.npz`` (the bowl + the recovery trace) and appends a JSON summary
to ``--results``; emits **TWIN_RECOVER_OK** when the bowl argmin sits at the truth AND the
recovered ``k_gm`` is within ``--tol`` of it AND the final misfit ≪ the initial. Single-GPU only
(the sharded ragged-halo AD bug is forward-safe only — docs/JAX_RAGGED_A2A_BUG.md).

Usage (GPU):  python scripts/paper/core2_paper_calib_twin.py --n 8 --config all3
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
import optax

from fesom_jax import calibrate, core2_forcing, ice, ssh
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
DEFAULT_RESULTS = ROOT / "scripts" / "calib_results.jsonl"


def peak_gb():
    pk = 0.0
    for d in jax.devices():
        try:
            st = d.memory_stats() or {}
            pk = max(pk, (st.get("peak_bytes_in_use") or 0) / 1e9)
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


def is_oom(e: Exception) -> bool:
    s = str(e).upper()
    return ("RESOURCE_EXHAUSTED" in s or "OUT_OF_MEMORY" in s or "memory" in str(e).lower())


def build(year, n, config):
    """Mesh + (ice-seeded) state + forcing + the live-config dict for the chosen model."""
    mesh = load_mesh(MESH_DIR)
    base = core2_initial_state(mesh, IC_DIR)
    if config == "all3":
        sst = np.asarray(base.T[:, 0])
        state = ice.seed_ice(base, mesh, sst)                       # mEVP needs an ice IC
        # frozen-ice adjoint: full mEVP ice in the forward, gradient skips the unstable rheology
        # adjoint (the mitgcm/ECCO-style cheap path; see IceConfig.adjoint_mode + the §2 finding).
        cfgs = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(),
                    ice_cfg=IceConfig(whichEVP=1, adjoint_mode="frozen"), ale_cfg=AleConfig())
    elif config == "tkegm":
        state = base
        cfgs = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(), ale_cfg=AleConfig())  # ocean-only, no ice
    elif config == "gm":
        state = base
        cfgs = dict(gm_cfg=GMConfig(), ale_cfg=AleConfig())         # C1-style isolation / fallback
    else:
        raise ValueError(f"unknown --config {config!r}")
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, year, sst_ic=np.asarray(base.T[:, 0]))
    sfs = cf.stack(core2_forcing.dates_for_steps(year, DT, n))
    return mesh, state, op, cf.static, sfs, cfgs


def make_model(mesh, state, op, fs, sfs, n, cfgs):
    """Return ``model_T(k_gm) -> T[nod2D, nl]`` (jitted forward, reused for truth + grid + loss).

    ``k_gm`` and the C-synced ``redi_kmax`` both derive from the scalar (the grad-gate / namelist
    convention). One jitted forward serves the truth, the forward-only bowl scan, AND the recovery
    loss (differentiated through it) — the C1 "reuse one jitted forward" memory lesson."""

    @jax.jit
    def model_T(k_gm):
        p = calibrate.build_params({"k_gm": k_gm, "redi_kmax": k_gm})
        fin = integrate(state, mesh, op, None, n_steps=n, dt=DT, step_forcings=sfs,
                        forcing_static=fs, params=p, **cfgs)
        return fin.T

    return model_T


def band_weights(mesh, band):
    """Area×wet weights over the top ``band`` layers (the upper-ocean T observable)."""
    node_mask = jnp.asarray(mesh.node_layer_mask)                   # [N, nl]
    klev = jnp.arange(mesh.nl)[None, :]
    bmask = node_mask & (klev < band)
    return jnp.where(bmask, jnp.asarray(mesh.area), 0.0)            # [N, nl]


def wmse(T, truth, w):
    """Area-weighted mean-squared (T − truth) over the band (guarded all-zero → 0)."""
    d = T - truth
    num = jnp.sum(w * d * d)
    den = jnp.sum(w)
    return num / jnp.where(den > 0.0, den, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="adjoint window (steps); all-on is heavy")
    ap.add_argument("--config", choices=("all3", "tkegm", "gm"), default="all3",
                    help="all3 = zstar+TKE+mEVP+GM (proper model); tkegm = ocean-only (no ice); "
                         "gm = isolated zstar+GM (fallback)")
    ap.add_argument("--grad-check", action="store_true",
                    help="diagnostic: report |grad| of the normalized loss at init, then exit "
                         "(probes adjoint sanity per config/N — no grid scan / recovery)")
    ap.add_argument("--truth", type=float, default=1500.0, help="injected k_gm truth")
    ap.add_argument("--init", type=float, default=800.0, help="recovery start k_gm")
    ap.add_argument("--band", type=int, default=10, help="# top layers for the upper-ocean T metric")
    # lr/iters from the pi convergence probe (clean normalized quadratic bowl, mesh-independent):
    # lr=0.05/80 lands at rel=2.2% (misses the 2% bar); lr=0.1/80 -> 0.75%, lr=0.1/100 -> tighter.
    ap.add_argument("--iters", type=int, default=100, help="max Adam iterations")
    ap.add_argument("--lr", type=float, default=0.1, help="Adam lr on the normalized leaf u=k_gm/1000")
    ap.add_argument("--tol", type=float, default=0.02, help="recovered-value bar |k-truth|/truth")
    ap.add_argument("--misfit-tol", type=float, default=0.05, help="J_final/J_init bar")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--outdir", type=Path, default=ROOT / "scripts")
    args = ap.parse_args()
    n, cfg = args.n, args.config

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, state, op, fs, sfs, cfgs = build(args.year, n, cfg)
    cfg_name = "zstar+TKE+mEVP+GM" if cfg == "all3" else "zstar+GM"
    gpu_gb = gpu_limit_gb()
    print(f"[setup] built in {time.time()-t0:.1f}s; config={cfg_name}  N={n} "
          f"({n*DT/86400:.3f} days)  GPU limit={gpu_gb:.0f} GB", flush=True)

    model_T = make_model(mesh, state, op, fs, sfs, n, cfgs)
    w = band_weights(mesh, args.band)
    rec = {"target": "kgm_twin", "config": cfg_name, "N": n, "days": n * DT / 86400.0, "dt": DT,
           "truth": args.truth, "init": args.init, "band": args.band, "gpu_gb": gpu_gb}

    # ---------- truth injection (the synthetic observation) ----------
    try:
        t1 = time.time()
        truth = model_T(jnp.asarray(args.truth, jnp.float64))       # [N, nl]
        truth.block_until_ready()
        print(f"[truth] ran k_gm={args.truth:.0f} forward in {time.time()-t1:.1f}s  "
              f"peak={peak_gb():.1f}/{gpu_gb:.0f} GB", flush=True)
    except Exception as e:
        return _bail(args, rec, "truth", e)

    # ---------- diagnostic: adjoint sanity (|grad| of the normalized loss at init), then exit ----
    if args.grad_check:
        try:
            J0 = float(wmse(model_T(jnp.asarray(args.init, jnp.float64)), truth, w))

            def loss_norm_gc(uu):
                return wmse(model_T(1000.0 * uu), truth, w) / J0

            t1 = time.time()
            g = float(jax.grad(loss_norm_gc)(jnp.asarray(args.init / 1000.0, jnp.float64)))
            dt_g = time.time() - t1
        except Exception as e:
            return _bail(args, rec, "gradcheck", e)
        sane = bool(np.isfinite(g) and abs(g) < 1e6)        # a well-conditioned normalized grad is O(1)
        pk = peak_gb()
        print(f"\n  [grad-check] config={cfg_name} N={n}  J0={J0:.3e}  d(J/J0)/du@init={g:+.4e}  "
              f"|g|={abs(g):.3e}  finite={np.isfinite(g)}  sane(|g|<1e6)={sane}  "
              f"peak={pk:.1f}/{gpu_gb:.0f} GB  ({dt_g:.1f}s)", flush=True)
        rec.update(dict(grad_check=True, J0=J0, grad_u=g, grad_sane=sane, peak_gb=pk))
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"TWIN_GRADCHECK_{'OK' if sane else 'BAD'}", flush=True)
        return 0 if sane else 1

    # ---------- (1) grid-scan the misfit bowl FIRST (forward-only) ----------
    # bracket both init and truth; include the truth exactly so argmin can land on it.
    grid = np.unique(np.concatenate([
        np.arange(500.0, 2201.0, 100.0), [args.truth, args.init]])).astype(np.float64)
    grid.sort()
    try:
        t1 = time.time()
        Js = np.array([float(wmse(model_T(jnp.asarray(k, jnp.float64)), truth, w)) for k in grid])
    except Exception as e:
        return _bail(args, rec, "gridscan", e)
    kmin = float(grid[int(np.argmin(Js))])
    bowl_ok = bool(abs(kmin - args.truth) < 1e-6)
    print(f"\n  --- misfit bowl ({len(grid)} pts, {time.time()-t1:.1f}s) ---")
    for k, J in zip(grid, Js):
        mark = "  <-- argmin" if abs(k - kmin) < 1e-9 else ("  (truth)" if k == args.truth else "")
        print(f"      k_gm={k:7.1f}  J={J:.6e}{mark}")
    print(f"      argmin at k_gm={kmin:.1f}  (truth={args.truth:.0f})  bowl_ok={bowl_ok}", flush=True)
    rec.update(dict(grid_kmin=kmin, bowl_ok=bowl_ok))

    # ---------- (2) recover k_gm via cosine-decay Adam (the global adjoint) ----------
    # Two normalizations make the recovery well-conditioned for the TINY GM short-window misfit
    # (C1: the GM→T signal is ~1e-9, so J ~ 1e-12):
    #   * the leaf is u = k_gm/1000   ⇒ a scale-free Adam lr (0.8→1.5);
    #   * the loss is J/J0 (J0 = the misfit at the start) ⇒ an O(1) loss whose gradient is O(1),
    #     so Adam's eps (1e-8) does NOT swamp a ~1e-11 raw gradient (the CPU test caught this:
    #     without it `u` barely moved). redi_kmax rides k_gm inside model_T.
    J0 = float(Js[int(np.argmin(np.abs(grid - args.init)))])    # raw misfit at the start (bowl pt)
    if not (J0 > 0.0):
        return _bail(args, rec, "recover", ValueError(f"degenerate start misfit J0={J0}"))

    def raw_loss(d):
        return wmse(model_T(1000.0 * d["u"]), truth, w)

    def loss_norm(d):
        return raw_loss(d) / J0

    init = {"u": jnp.asarray(args.init / 1000.0, jnp.float64)}
    opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.iters))

    def on_step(r):
        u = float(r["params"]["u"])
        print(f"    it={r['it']:3d}  k_gm={1000.0*u:8.2f}  J/J0={r['loss']:.6e}  |g|={r['gnorm']:.2e}",
              flush=True)

    def stop_fn(r):                                 # plateau early-stop (loss-based, truth-agnostic)
        # J/J0 = J(u)/J(u_init); for a parabolic bowl J/J0 ∝ rel² ⇒ J/J0 < 1e-4 implies the
        # recovered value is well inside the 2% bar (so firing this never pre-empts a gate pass).
        return r["loss"] < 1e-4

    try:
        t1 = time.time()
        print(f"\n  --- recover k_gm: {args.init:.0f} -> {args.truth:.0f}  (Adam lr={args.lr}, "
              f"cosine decay, ≤{args.iters} it; loss J/J0, J0={J0:.3e}) ---", flush=True)
        rec_params, hist = calibrate.optimize(loss_norm, init, opt, n_iters=args.iters,
                                              on_step=on_step, stop_fn=stop_fn)
        k_rec = 1000.0 * float(rec_params["u"])
        J_final = float(raw_loss(rec_params))
        bwd_s = time.time() - t1
    except Exception as e:
        return _bail(args, rec, "recover", e)

    J_init = J0
    its = np.array([r["it"] for r in hist])
    Jtr = np.array([r["loss"] for r in hist])        # normalized loss trace (J/J0, starts at 1)
    ktr = np.array([1000.0 * float(r["params"]["u"]) for r in hist])
    rel = abs(k_rec - args.truth) / abs(args.truth)
    misfit_ratio = J_final / J_init if J_init > 0 else 0.0
    recover_ok = bool(rel < args.tol)
    misfit_ok = bool(misfit_ratio < args.misfit_tol)
    pk = peak_gb()
    print(f"\n  recovered k_gm = {k_rec:.2f}  (truth {args.truth:.0f}, rel {rel:.3%})  "
          f"in {len(hist)} it / {bwd_s:.1f}s  peak={pk:.1f}/{gpu_gb:.0f} GB", flush=True)
    print(f"  misfit J: {J_init:.6e} -> {J_final:.6e}  (ratio {misfit_ratio:.2e})", flush=True)

    ok = bool(bowl_ok and recover_ok and misfit_ok)
    rec.update(dict(k_rec=k_rec, rel=rel, J_init=J_init, J_final=J_final,
                    misfit_ratio=misfit_ratio, n_iters=len(hist), peak_gb=pk, bwd_s=bwd_s,
                    recover_ok=recover_ok, misfit_ok=misfit_ok, ok=ok))

    # ---------- persist the bowl + recovery trace + the diagnostic figure ----------
    args.outdir.mkdir(parents=True, exist_ok=True)
    npz = args.outdir / "calib_twin_kgm.npz"
    np.savez(npz, grid_k=grid, grid_J=Js, hist_it=its, hist_k=ktr, hist_J=Jtr,
             truth=args.truth, init=args.init, k_rec=k_rec, J_init=J_init, J_final=J_final,
             N=n, config=cfg_name, band=args.band)
    _maybe_figure(args.outdir / "calib_twin_kgm.png", grid, Js, ktr, Jtr, args, k_rec, cfg_name)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  wrote {npz}; appended summary -> {args.results}", flush=True)
    print(f"  gate components: bowl={bowl_ok} recover={recover_ok} (rel<{args.tol}) "
          f"misfit={misfit_ok} (ratio<{args.misfit_tol})", flush=True)
    print(f"TWIN_RECOVER_{'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def _bail(args, rec, phase, e):
    """OOM → exit 2 (the .sbatch descends N); any other failure → exit 1."""
    oom = is_oom(e)
    msg = f"{type(e).__name__}: {str(e)[:240]}"
    rec.update(dict(ok=False, phase=phase, oom=oom, peak_gb=peak_gb(), error=msg))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  FAILED in {phase}: {msg}", flush=True)
    print(f"TWIN_RECOVER_{'OOM' if oom else 'FAIL'}", flush=True)
    return 2 if oom else 1


def _maybe_figure(path, grid, Js, ktr, Jtr, args, k_rec, cfg_name):
    """Quick review diagnostic (not the paper Fig 3 — that's D2c): bowl + recovery trace."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] skipped ({e})", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(grid, Js, "o-", lw=1, ms=4)
    ax[0].axvline(args.truth, color="g", ls="--", label=f"truth {args.truth:.0f}")
    ax[0].axvline(args.init, color="r", ls=":", label=f"init {args.init:.0f}")
    ax[0].axvline(k_rec, color="k", ls="-", lw=1, label=f"recovered {k_rec:.0f}")
    ax[0].set(xlabel="k_gm [m$^2$/s]", ylabel="misfit J (upper-ocean T, weighted MSE)",
              title=f"D1 twin misfit bowl ({cfg_name})")
    ax[0].legend(fontsize=8)
    ax[1].plot(ktr, "o-", ms=3)
    ax[1].axhline(args.truth, color="g", ls="--", label=f"truth {args.truth:.0f}")
    ax[1].set(xlabel="Adam iteration", ylabel="k_gm [m$^2$/s]", title="recovery trace")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  [fig] wrote {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
