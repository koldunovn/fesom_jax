"""CORE2 TKE→MLD calibration — Paper-experiments Task D2a (§2, the FAST adjoint target).

§2's mixing-calibration deliverable: tune the TKE closure coefficient ``tke_c_k`` (the
mixing-length→diffusivity constant, ``KappaM = c_k·mxl·√tke``) so the model's **mixed-layer
depth** matches a target. MLD is the mixing-sensitive OMIP metric — exactly what ``c_k`` controls
— and it is **fast** (responds in hours–days), so it is **adjoint-reachable** at short windows
(C1 measured ``∂(mean MLD)/∂c_k = +2.91`` at N=12, a strong clean signal — unlike the tiny GM→T/S
signal that needs EKI). This is the obs-application counterpart to D1's perfect-model proof.

**This driver — the perfect-model TWIN (the proof, no obs dependency).** It injects a truth
``tke_c_k=--truth`` (default 0.15), freezes the resulting **MLD field** as the synthetic
observation (:func:`fesom_jax.obs_compare.mld_density_threshold`, the AD-safe density-threshold
MLD), then **recovers** ``tke_c_k`` from a wrong start (``--init`` 0.08) by minimizing the
MLD↔truth misfit through the global adjoint (:func:`fesom_jax.calibrate.optimize`). Recovering the
planted value proves the TKE→MLD inversion is well-posed and adjoint-reachable in the **full paper
model**. The real-obs step (MLD-vs-WOA-derived / de Boyer Montégut once IFREMER is back, batched
over windows) is a one-line swap of the target field for ``obs_compare.misfit`` vs the obs MLD.

**Proper ocean model + frozen-ice adjoint (D1 finding).** Config = the FULL paper model
(zstar+TKE+mEVP+GM, ``--config all3``) with the **frozen-ice adjoint** (``IceConfig.adjoint_mode
="frozen"`` — full mEVP forward, the backward skips the unstable ice rheology; see Task A8). The
TKE→MLD gradient threads through the live mixing while treating the (forward-real) ice as fixed —
a sound approximation for the *direct* mixing→MLD control (the ice-mediated polar-MLD path is the
free-drift adjoint's job, Option B). ``--config tkegm`` drops ice entirely (ocean-only control).

Same two conditioning tricks as D1: a scale-free leaf ``u = c_k/0.1`` and a loss normalized by its
start value ``J/J0`` (robust even though the MLD misfit is O(m²), not tiny like GM). Grid-scan the
bowl first (argmin at the truth), then cosine-decay-Adam recovery → **TKE_TWIN_OK** (recovered
``c_k`` within ``--tol`` of the truth, misfit ≪ initial). Single-GPU only.

Usage (GPU):  python scripts/paper/core2_paper_calib_tke.py --n 12 --config all3
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

from fesom_jax import ale, calibrate, surface_forcing, ice, obs_compare, ssh
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
CK0 = float(Params.defaults().tke_c_k)              # 0.1 — the leaf normalizer / default
DEFAULT_RESULTS = ROOT / "scripts" / "calib_tke_results.jsonl"


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


def is_oom(e: Exception) -> bool:
    s = str(e).upper()
    return ("RESOURCE_EXHAUSTED" in s or "OUT_OF_MEMORY" in s or "memory" in str(e).lower())


def build(year, n, config):
    """Mesh + (ice-seeded) state + forcing + the live-config dict (mirrors the D1 twin)."""
    mesh = load_mesh(MESH_DIR)
    base = core2_initial_state(mesh, IC_DIR)
    if config == "all3":
        state = ice.seed_ice(base, mesh, np.asarray(base.T[:, 0]))
        cfgs = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(),
                    ice_cfg=IceConfig(whichEVP=1, adjoint_mode="frozen"), ale_cfg=AleConfig())
    elif config == "tkegm":
        state = base
        cfgs = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(), ale_cfg=AleConfig())   # ocean-only control
    else:
        raise ValueError(f"unknown --config {config!r} (use all3 / tkegm)")
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, year, sst_ic=np.asarray(base.T[:, 0]))
    sfs = cf.stack(surface_forcing.dates_for_steps(year, DT, n))
    return mesh, state, op, cf.static, sfs, cfgs


def make_mld_model(mesh, state, op, fs, sfs, n, cfgs):
    """Return ``model_mld(c_k) -> mld[nod2D]`` (jitted): forward the model with ``tke_c_k=c_k``,
    then the AD-safe density-threshold MLD from the LIVE zstar geometry. One jitted forward reused
    for the truth, the bowl scan, and (differentiated) the recovery loss (the C1 memory lesson)."""
    node_mask = jnp.asarray(mesh.node_layer_mask)

    @jax.jit
    def model_mld(c_k):
        p = calibrate.build_params({"tke_c_k": c_k})
        fin = integrate(state, mesh, op, None, n_steps=n, dt=DT, step_forcings=sfs,
                        forcing_static=fs, params=p, **cfgs)
        _, Z3d = ale.live_geometry(mesh, fin.hnode)
        mld, valid = obs_compare.mld_density_threshold(fin.T, fin.S, Z3d, node_mask)
        return mld, valid

    return model_mld


def wmse(field, truth, w):
    """Area-weighted mean-squared (field − truth) (guarded all-zero → 0)."""
    d = field - truth
    return jnp.sum(w * d * d) / jnp.where(jnp.sum(w) > 0.0, jnp.sum(w), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="adjoint window (steps); A7/C1 reachable")
    ap.add_argument("--config", choices=("all3", "tkegm"), default="all3",
                    help="all3 = zstar+TKE+mEVP+GM (frozen-ice adjoint); tkegm = ocean-only control")
    ap.add_argument("--truth", type=float, default=0.15, help="injected tke_c_k truth")
    ap.add_argument("--init", type=float, default=0.08, help="recovery start tke_c_k")
    ap.add_argument("--iters", type=int, default=100, help="max Adam iterations")
    ap.add_argument("--lr", type=float, default=0.1, help="Adam lr on the normalized leaf u=c_k/0.1")
    ap.add_argument("--tol", type=float, default=0.02, help="recovered-value bar |c_k-truth|/truth")
    ap.add_argument("--misfit-tol", type=float, default=0.05, help="J_final/J_init bar")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--outdir", type=Path, default=ROOT / "scripts")
    args = ap.parse_args()
    n, cfg = args.n, args.config

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, state, op, fs, sfs, cfgs = build(args.year, n, cfg)
    cfg_name = "zstar+TKE+mEVP+GM" if cfg == "all3" else "zstar+TKE+GM"
    gpu_gb = gpu_limit_gb()
    print(f"[setup] built in {time.time()-t0:.1f}s; config={cfg_name}  N={n} "
          f"({n*DT/86400:.3f} days)  GPU limit={gpu_gb:.0f} GB", flush=True)

    model_mld = make_mld_model(mesh, state, op, fs, sfs, n, cfgs)
    area = jnp.asarray(mesh.area[:, 0])                  # surface CV area (MLD is a 2-D field)
    rec = {"target": "tke_mld_twin", "config": cfg_name, "N": n, "days": n * DT / 86400.0,
           "dt": DT, "truth": args.truth, "init": args.init, "gpu_gb": gpu_gb}

    # ---------- truth injection (the synthetic MLD observation) ----------
    try:
        t1 = time.time()
        truth_mld, truth_valid = model_mld(jnp.asarray(args.truth, jnp.float64))
        truth_mld.block_until_ready()
        w = area * truth_valid                           # weight: surface area on truth-valid nodes
        print(f"[truth] ran c_k={args.truth:g} forward in {time.time()-t1:.1f}s  "
              f"valid MLD nodes={int(np.sum(np.asarray(truth_valid)))}  "
              f"mean MLD={float(jnp.sum(w*truth_mld)/jnp.sum(w)):.1f} m  peak={peak_gb():.1f} GB",
              flush=True)
    except Exception as e:
        return _bail(args, rec, "truth", e)

    # ---------- (1) grid-scan the misfit bowl FIRST (forward-only) ----------
    grid = np.unique(np.concatenate([
        np.arange(0.04, 0.221, 0.01), [args.truth, args.init]])).astype(np.float64)
    grid.sort()
    try:
        t1 = time.time()
        Js = np.array([float(wmse(model_mld(jnp.asarray(c, jnp.float64))[0], truth_mld, w))
                       for c in grid])
    except Exception as e:
        return _bail(args, rec, "gridscan", e)
    kmin = float(grid[int(np.argmin(Js))])
    bowl_ok = bool(abs(kmin - args.truth) < 1e-9)
    print(f"\n  --- MLD misfit bowl ({len(grid)} pts, {time.time()-t1:.1f}s) ---")
    for c, J in zip(grid, Js):
        mark = "  <-- argmin" if abs(c - kmin) < 1e-12 else ("  (truth)" if c == args.truth else "")
        print(f"      c_k={c:.3f}  J={J:.6e}{mark}")
    print(f"      argmin at c_k={kmin:.3f}  (truth={args.truth:g})  bowl_ok={bowl_ok}", flush=True)
    rec.update(dict(grid_kmin=kmin, bowl_ok=bowl_ok))

    # ---------- (2) recover tke_c_k via cosine-decay Adam (the global adjoint) ----------
    # scale-free leaf u = c_k/0.1; loss J/J0 (O(1) ⇒ robust conditioning, the D1 lesson).
    J0 = float(Js[int(np.argmin(np.abs(grid - args.init)))])
    if not (J0 > 0.0):
        return _bail(args, rec, "recover", ValueError(f"degenerate start misfit J0={J0}"))

    def raw_loss(d):
        return wmse(model_mld(CK0 * d["u"])[0], truth_mld, w)

    init = {"u": jnp.asarray(args.init / CK0, jnp.float64)}
    opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.iters))

    def on_step(r):
        print(f"    it={r['it']:3d}  c_k={CK0*float(r['params']['u']):.5f}  "
              f"J/J0={r['loss']:.6e}  |g|={r['gnorm']:.2e}", flush=True)

    def stop_fn(r):
        return r["loss"] < 1e-4              # J/J0 ∝ rel² ⇒ well inside the 2% bar (see D1)

    try:
        t1 = time.time()
        print(f"\n  --- recover tke_c_k: {args.init:g} -> {args.truth:g}  (Adam lr={args.lr}, "
              f"cosine decay, ≤{args.iters} it; loss J/J0, J0={J0:.3e}) ---", flush=True)
        rec_params, hist = calibrate.optimize(lambda d: raw_loss(d) / J0, init, opt,
                                              n_iters=args.iters, on_step=on_step, stop_fn=stop_fn)
        k_rec = CK0 * float(rec_params["u"])
        J_final = float(raw_loss(rec_params))
        bwd_s = time.time() - t1
    except Exception as e:
        return _bail(args, rec, "recover", e)

    its = np.array([r["it"] for r in hist])
    Jtr = np.array([r["loss"] for r in hist])
    ktr = np.array([CK0 * float(r["params"]["u"]) for r in hist])
    rel = abs(k_rec - args.truth) / abs(args.truth)
    misfit_ratio = J_final / J0
    recover_ok = bool(rel < args.tol)
    misfit_ok = bool(misfit_ratio < args.misfit_tol)
    pk = peak_gb()
    print(f"\n  recovered tke_c_k = {k_rec:.5f}  (truth {args.truth:g}, rel {rel:.3%})  "
          f"in {len(hist)} it / {bwd_s:.1f}s  peak={pk:.1f}/{gpu_gb:.0f} GB", flush=True)
    print(f"  MLD misfit J: {J0:.6e} -> {J_final:.6e}  (ratio {misfit_ratio:.2e})", flush=True)

    ok = bool(bowl_ok and recover_ok and misfit_ok)
    rec.update(dict(k_rec=k_rec, rel=rel, J_init=J0, J_final=J_final, misfit_ratio=misfit_ratio,
                    n_iters=len(hist), peak_gb=pk, bwd_s=bwd_s, recover_ok=recover_ok,
                    misfit_ok=misfit_ok, ok=ok))

    args.outdir.mkdir(parents=True, exist_ok=True)
    npz = args.outdir / "calib_tke_twin.npz"
    np.savez(npz, grid_k=grid, grid_J=Js, hist_it=its, hist_k=ktr, hist_J=Jtr,
             truth=args.truth, init=args.init, k_rec=k_rec, J_init=J0, J_final=J_final,
             N=n, config=cfg_name)
    _maybe_figure(args.outdir / "calib_tke_twin.png", grid, Js, ktr, args, k_rec, cfg_name)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  wrote {npz}; appended summary -> {args.results}", flush=True)
    print(f"  gate components: bowl={bowl_ok} recover={recover_ok} (rel<{args.tol}) "
          f"misfit={misfit_ok} (ratio<{args.misfit_tol})", flush=True)
    print(f"TKE_TWIN_{'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def _bail(args, rec, phase, e):
    oom = is_oom(e)
    msg = f"{type(e).__name__}: {str(e)[:240]}"
    rec.update(dict(ok=False, phase=phase, oom=oom, peak_gb=peak_gb(), error=msg))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  FAILED in {phase}: {msg}", flush=True)
    print(f"TKE_TWIN_{'OOM' if oom else 'FAIL'}", flush=True)
    return 2 if oom else 1


def _maybe_figure(path, grid, Js, ktr, args, k_rec, cfg_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] skipped ({e})", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(grid, Js, "o-", lw=1, ms=4)
    ax[0].axvline(args.truth, color="g", ls="--", label=f"truth {args.truth:g}")
    ax[0].axvline(args.init, color="r", ls=":", label=f"init {args.init:g}")
    ax[0].axvline(k_rec, color="k", lw=1, label=f"recovered {k_rec:.3f}")
    ax[0].set(xlabel="tke_c_k", ylabel="MLD misfit J (weighted MSE, m$^2$)",
              title=f"D2a TKE→MLD twin bowl ({cfg_name})")
    ax[0].legend(fontsize=8)
    ax[1].plot(ktr, "o-", ms=3)
    ax[1].axhline(args.truth, color="g", ls="--", label=f"truth {args.truth:g}")
    ax[1].set(xlabel="Adam iteration", ylabel="tke_c_k", title="recovery trace")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  [fig] wrote {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
