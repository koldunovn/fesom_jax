"""CORE2 NN-of-TKE perfect-model twin — Paper-experiments Task E1 (§3 hybrid-ML, the headline proof).

The §3 capability claim: **a neural network embedded in the mixing closure can be trained end-to-end
through the FULL differentiable global ocean model.** This driver proves it the rigorous way — a
perfect-model twin with NO representational error:

  truth = the model run with a KNOWN instance of the **same** ``tke_nn`` architecture (seeded weights →
          a non-trivial, bounded per-column multiplier on ``c_k/c_eps/c_d``) ⇒ a synthetic T/S evolution.
  recover = train a trainee ``tke_nn`` **from NN→0** (zero last layer ⇒ multiplier ≡ 1 ⇒ default TKE, the
            deployment-safe start) to reproduce the truth's T/S evolution over a short window, through the
            global adjoint (:func:`fesom_jax.calibrate.optimize` over the NN-weight pytree).

Because the trainee has the SAME architecture, the truth is exactly representable — so a failure is the
optimizer/adjoint, not representational error (the clean test the plan's review asked for). The proof is
TWO-fold: **(1) evolution recovered** — the T/S misfit drops ≪ its NN→0 start; **(2) induced-mixing field
recovered** — the trainee's learned per-column multiplier matches the truth's where the short window
constrains it (the c_k-sensitive deep-mixing columns; cf. the D2c spatial-signal finding). A FD↔AD spot
check on the largest-|grad| output weight proves the **NN-weight gradient through the global model is
correct** (the actual novel capability). Token **NN_TWIN_OK**.

Proper model + frozen-ice adjoint (D1/D2a precedent): config ``all3`` = zstar+TKE+mEVP+GM with
``IceConfig.adjoint_mode="frozen"`` (full mEVP forward, the backward skips the unstable ice rheology;
Task A8). The NN reads the LIVE per-column features each step; the multiplier is bounded+positive
(structural stability) and exactly 1 at the NN→0 start (bit-identical to default TKE). Single-GPU only.

Usage (GPU):  python scripts/paper/core2_paper_nn_twin.py --n 12 --config all3
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

from fesom_jax import ale, calibrate, surface_forcing, eos, ice, pp, ssh, tke_nn
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.tke import DENSITY_0, TkeConfig, _layer_center_Z, _safe_sqrt, _shift_down

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
DEFAULT_RESULTS = ROOT / "scripts" / "nn_twin_results.jsonl"


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
    """Mesh + (ice-seeded) state + forcing + the live-config dict (mirrors the D2a TKE twin)."""
    mesh = load_mesh(MESH_DIR)
    base = phc_initial_state(mesh, IC_DIR)
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
    dates = surface_forcing.dates_for_steps(year, DT, n)
    sfs = cf.stack(dates)
    sf_last = jax.tree.map(lambda x: x[n - 1], sfs)              # last step's forcing (for diagnostics)
    return mesh, state, op, cf.static, sfs, sf_last, cfgs


def make_truth_nn(seed, amp, bias, m_max=3.0):
    """A seeded ``tke_nn`` with a NON-trivial bounded multiplier (the synthetic truth). Start from a
    random non-zero net, then amplify the OUTPUT layer (``amp``) and add an output ``bias`` so the
    induced multiplier has both a mean offset and a spatial pattern — a real 'wrong' mixing the
    trainee (starting at multiplier ≡ 1) must learn. Exactly representable by the trainee net."""
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(seed), zero_last=False, m_max=m_max)
    Ws, bs = list(nn.Ws), list(nn.bs)
    Ws[-1] = Ws[-1] * amp
    bs[-1] = bs[-1] + bias
    return tke_nn.TkeNN(Ws=tuple(Ws), bs=tuple(bs), log_m_max=nn.log_m_max)


def reconstruct_features(mesh, st, sf_last, fs):
    """Reassemble the NN column features at state ``st`` — the EXACT mirror of the per-column
    assembly inside :func:`fesom_jax.tke.mixing_tke` (uses the model's own eos/pp/forcing fns;
    no halo exch needed on a single dense GPU). For the induced-mixing-field comparison only."""
    nl = mesh.nl
    k = jnp.arange(nl)[None, :]
    nzmin = (mesh.ulevels_nod2D - 1)[:, None]
    nzmax = (mesh.nlevels_nod2D - 1)[:, None]
    is_interior = (k >= nzmin + 1) & (k <= nzmax - 1)
    _, Z3d = ale.live_geometry(mesh, st.hnode)
    _, _, bvfreq = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode, Z3d=Z3d)
    bvfreq2 = jnp.where(is_interior, bvfreq, 0.0)
    uvnode = pp.compute_vel_nodes(mesh, st.uv)
    Z = _layer_center_Z(mesh, Z3d)
    dZ = _shift_down(Z) - Z
    dZ_safe = jnp.where(dZ == 0.0, 1.0, dZ)
    du = _shift_down(uvnode[..., 0]) - uvnode[..., 0]
    dv = _shift_down(uvnode[..., 1]) - uvnode[..., 1]
    vshear2 = jnp.where(is_interior, (du * du + dv * dv) / (dZ_safe * dZ_safe), 0.0)
    sfx = surface_forcing.compute_surface_fluxes(mesh, st, sf_last, fs, dt=DT)
    sx, sy = sfx.stress_node_surf[:, 0], sfx.stress_node_surf[:, 1]
    forc = _safe_sqrt(sx * sx + sy * sy) / DENSITY_0
    return tke_nn.column_features(forc, mesh.coriolis_node, mesh.depth, bvfreq2, vshear2,
                                 st.tke, is_interior)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="adjoint window (steps); A7/C1 reachable")
    ap.add_argument("--config", choices=("all3", "tkegm"), default="all3",
                    help="all3 = zstar+TKE+mEVP+GM (frozen-ice adjoint); tkegm = ocean-only control")
    ap.add_argument("--seed", type=int, default=0, help="truth-NN PRNG seed")
    ap.add_argument("--truth-amp", type=float, default=4.0, help="output-layer amplitude of the truth NN")
    ap.add_argument("--truth-bias", type=float, default=0.3, help="output-layer bias of the truth NN")
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 16], help="MLP hidden sizes")
    ap.add_argument("--iters", type=int, default=200, help="max Adam iterations")
    ap.add_argument("--lr", type=float, default=0.03, help="Adam lr on the NN weights")
    ap.add_argument("--misfit-tol", type=float, default=0.15,
                    help="gate: mean(J_T/J_T0, J_S/J_S0) below this = evolution recovered")
    ap.add_argument("--corr-tol", type=float, default=0.9,
                    help="gate: corr(m_trainee, m_truth) over active cols = induced-mixing recovered")
    ap.add_argument("--grad-check", action="store_true", default=True,
                    help="FD↔AD spot check on the largest-|grad| output weight (default on)")
    ap.add_argument("--no-grad-check", dest="grad_check", action="store_false")
    ap.add_argument("--fd-h", type=float, default=1e-3, help="FD step for the grad check")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--from-s0", type=Path, default=None,
                    help="start truth+trainee from a spun-up state pickle (a strong, immediate "
                         "mixing signal vs the cold IC) — reuse the D2a tkegm S0 (config must match)")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--outdir", type=Path, default=ROOT / "scripts")
    ap.add_argument("--remat-blocks", dest="remat_blocks", action="store_true", default=True,
                    help="nested in-step rematerialization (cuts the single-step VJP working set "
                         "~3× so the CORE2 NN backward fits the 80 GB A100); on by default")
    ap.add_argument("--no-remat-blocks", dest="remat_blocks", action="store_false",
                    help="disable nested remat (the un-checkpointed baseline; OOMs on CORE2)")
    ap.add_argument("--remat-segments", type=int, default=0,
                    help="O(√N) two-level checkpointing for LONG continuous windows: -1=auto S≈√N "
                         "(stores ~2√N carries not N ⇒ N≫10 fits one GPU), >1=explicit S, 0=per-step")
    args = ap.parse_args()
    n, cfg = args.n, args.config

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, state, op, fs, sfs, sf_last, cfgs = build(args.year, n, cfg)
    if args.from_s0 is not None:
        import pickle
        with open(args.from_s0, "rb") as fpk:
            state = jax.device_put(pickle.load(fpk))
        print(f"[setup] loaded spun-up S0 <- {args.from_s0}  (strong-signal start)", flush=True)
    cfg_name = "zstar+TKE+mEVP+GM" if cfg == "all3" else "zstar+TKE+GM"
    gpu_gb = gpu_limit_gb()
    node_mask = jnp.asarray(mesh.node_layer_mask)
    w3d = jnp.asarray(mesh.area[:, 0])[:, None] * node_mask          # volume-ish weight [N, nl]
    area0 = jnp.asarray(mesh.area[:, 0])                             # surface area (per-node weight)
    print(f"[setup] built in {time.time()-t0:.1f}s; config={cfg_name}  N={n} ({n*DT/86400:.3f} d)  "
          f"GPU limit={gpu_gb:.0f} GB", flush=True)

    sf_first = jax.tree.map(lambda x: x[0], sfs)        # forcing for the induced-mixing feature probe

    @jax.jit
    def model_ts(nn):                       # the ONE model program (truth + loss) — only T/S.
        # Returning just T/S (not the full State) keeps a single jitted forward/backward program;
        # the induced-mixing features come from the SHARED initial state S0 (forward-only, below),
        # so no second 'full-state' executable coexists with the grad ⇒ the all3 backward fits.
        p = calibrate.build_params({"tke_nn": nn})
        fin = integrate(state, mesh, op, None, n_steps=n, dt=DT, step_forcings=sfs,
                        forcing_static=fs, params=p, remat_blocks=args.remat_blocks,
                        remat_segments=args.remat_segments, **cfgs)
        return fin.T, fin.S

    def wmse(a, b):
        d = a - b
        return jnp.sum(w3d * d * d) / jnp.sum(w3d)

    rec = {"target": "nn_twin", "config": cfg_name, "N": n, "days": n * DT / 86400.0, "dt": DT,
           "seed": args.seed, "truth_amp": args.truth_amp, "truth_bias": args.truth_bias,
           "hidden": list(args.hidden), "from_s0": bool(args.from_s0), "gpu_gb": gpu_gb,
           "remat_blocks": bool(args.remat_blocks), "remat_segments": int(args.remat_segments)}

    # ---------- truth injection (seeded NN → synthetic T/S evolution) ----------
    try:
        t1 = time.time()
        truth_nn = make_truth_nn(args.seed, args.truth_amp, args.truth_bias)
        truth_T, truth_S = model_ts(truth_nn)
        truth_T.block_until_ready()
        feats_ref = reconstruct_features(mesh, state, sf_first, fs)  # shared S0 features (forward-only)
        m_truth = tke_nn.multiplier(truth_nn, feats_ref)            # [N, 3] on the shared initial state
        mck_t = np.asarray(m_truth[:, 0]); wet = np.asarray(node_mask[:, 0]).astype(bool)
        print(f"[truth] ran seeded NN forward in {time.time()-t1:.1f}s  peak={peak_gb():.1f} GB", flush=True)
        print(f"[truth] induced c_k multiplier (wet nodes): mean={mck_t[wet].mean():.3f} "
              f"min={mck_t[wet].min():.3f} max={mck_t[wet].max():.3f} std={mck_t[wet].std():.3f}", flush=True)
    except Exception as e:
        return _bail(args, rec, "truth", e)

    # ---------- trainee init: NN→0 (multiplier ≡ 1 = default TKE) ----------
    trainee0 = tke_nn.init_tke_nn(jax.random.PRNGKey(args.seed + 777), hidden=tuple(args.hidden),
                                  zero_last=True)
    init = {"tke_nn": trainee0}

    def raw_loss_TS(d):
        T, S = model_ts(d["tke_nn"])
        return wmse(T, truth_T), wmse(S, truth_S)

    try:
        # Baseline misfit via the ALREADY-compiled model_ts executable (NOT a separate jitted
        # raw_loss program). A second large executable retains its own ~11 GiB BFC region with
        # XLA_PYTHON_CLIENT_PREALLOCATE=false (BFC never returns it), starving the backward arena —
        # the measured N=12 OOM root cause (the grad couldn't grow a 38 GiB region because the truth
        # forward + baseline executables already held ~47 GiB). Reusing model_ts (same shapes ⇒ no
        # recompile) + an eager wmse keeps ONE forward arena resident. Same trick for JTf/JSf below.
        T0, S0 = model_ts(trainee0)
        JT0 = float(wmse(T0, truth_T)); JS0 = float(wmse(S0, truth_S))
        del T0, S0
    except Exception as e:
        return _bail(args, rec, "baseline", e)
    if not (JT0 > 0 and JS0 > 0):
        return _bail(args, rec, "baseline", ValueError(f"degenerate baseline JT0={JT0} JS0={JS0}"))
    print(f"\n[baseline] NN→0 (default TKE) vs truth:  J_T0={JT0:.4e} °C²  J_S0={JS0:.4e} psu²", flush=True)

    def loss(d):                            # balanced, O(1) at start (=2) — the D2a normalization
        jt, js = raw_loss_TS(d)
        return jt / JT0 + js / JS0

    # ---------- (rigor) FD↔AD spot check on the largest-|grad| output weight ----------
    if args.grad_check:
        try:
            t1 = time.time()
            g = jax.jit(jax.grad(loss))(init)
            gW = np.asarray(g["tke_nn"].Ws[-1])                    # output-layer weight grads
            i, j = np.unravel_index(int(np.argmax(np.abs(gW))), gW.shape)
            g_ad = float(gW[i, j])

            def bump(delta):
                Ws = list(trainee0.Ws)
                Wl = Ws[-1].at[i, j].add(delta)
                Ws[-1] = Wl
                nn2 = tke_nn.TkeNN(Ws=tuple(Ws), bs=trainee0.bs, log_m_max=trainee0.log_m_max)
                return float(loss({"tke_nn": nn2}))

            h = args.fd_h
            g_fd = (bump(h) - bump(-h)) / (2 * h)
            rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
            gc_ok = bool(rel < 0.05)
            print(f"\n[grad-check] output W[{i},{j}] AD={g_ad:.6e} FD={g_fd:.6e} rel={rel:.2e} "
                  f"ok={gc_ok}  ({time.time()-t1:.1f}s)", flush=True)
            rec.update(dict(gc_ad=g_ad, gc_fd=g_fd, gc_rel=rel, gc_ok=gc_ok))
        except Exception as e:
            return _bail(args, rec, "gradcheck", e)
    else:
        gc_ok = True

    # ---------- recover the NN via cosine-decay Adam (the global adjoint) ----------
    opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.iters))
    losshist: list[float] = []

    def on_step(r):
        losshist.append(r["loss"])
        if r["it"] % 10 == 0 or r["it"] == 1:
            print(f"    it={r['it']:3d}  loss(J_T/J_T0+J_S/J_S0)={r['loss']:.6e}  "
                  f"|g|={r['gnorm']:.2e}", flush=True)

    def stop_fn(r):
        if r["loss"] < 2.0 * args.misfit_tol:                     # both channels ~at the bar
            return True
        return len(losshist) > 20 and (losshist[-20] - r["loss"]) < 1e-3 * losshist[0]   # plateau

    try:
        t1 = time.time()
        print(f"\n  --- recover the NN: NN→0 → truth  (Adam lr={args.lr}, cosine, ≤{args.iters} it; "
              f"loss J_T/J_T0+J_S/J_S0, start=2.0) ---", flush=True)
        rec_params, hist = calibrate.optimize(loss, init, opt, n_iters=args.iters,
                                              on_step=on_step, stop_fn=stop_fn, keep_params=False)
        rec_nn = rec_params["tke_nn"]
        Tf, Sf = model_ts(rec_nn)                       # reuse model_ts (no separate executable)
        JTf = float(wmse(Tf, truth_T)); JSf = float(wmse(Sf, truth_S)); del Tf, Sf
        bwd_s = time.time() - t1
    except Exception as e:
        return _bail(args, rec, "recover", e)

    # ---------- induced-mixing field recovered? (trainee vs truth multiplier) ----------
    try:
        m_rec = tke_nn.multiplier(rec_nn, feats_ref)               # trainee m on the shared S0 features
        mck_r = np.asarray(m_rec[:, 0])
        wa = np.asarray(area0)
        # restrict to columns the short window actually constrains: |truth deviation| in the top
        # quartile of area-weighted mixing perturbation (the D2c "signal is spatially concentrated").
        dev = np.abs(mck_t - 1.0) * wet
        thr = np.quantile(dev[wet], 0.75) if wet.any() else 0.0
        act = wet & (dev >= thr)
        def wcorr(a, b, w):
            w = w / w.sum()
            am, bm = (w * a).sum(), (w * b).sum()
            ca, cb = a - am, b - bm
            return float((w * ca * cb).sum() / np.sqrt((w * ca * ca).sum() * (w * cb * cb).sum() + 1e-300))
        corr_all = wcorr(mck_t[wet], mck_r[wet], wa[wet])
        corr_act = wcorr(mck_t[act], mck_r[act], wa[act])
        relerr = float(np.sqrt((wa[wet] * (mck_r[wet] - mck_t[wet]) ** 2).sum()
                               / (wa[wet] * (mck_t[wet] - 1.0) ** 2).sum() + 1e-300))
    except Exception as e:
        return _bail(args, rec, "induced", e)

    JTr, JSr = JTf / JT0, JSf / JS0
    misfit_mean = 0.5 * (JTr + JSr)
    misfit_ok = bool(misfit_mean < args.misfit_tol)
    induced_ok = bool(corr_act > args.corr_tol)
    pk = peak_gb()
    print(f"\n  recovered NN in {len(hist)} it / {bwd_s:.1f}s  peak={pk:.1f}/{gpu_gb:.0f} GB", flush=True)
    print(f"  EVOLUTION: J_T {JT0:.4e}→{JTf:.4e} ({JTr:.2e}×)  J_S {JS0:.4e}→{JSf:.4e} ({JSr:.2e}×)  "
          f"mean ratio={misfit_mean:.2e}  (tol {args.misfit_tol})", flush=True)
    print(f"  INDUCED MIXING (c_k mult): corr_active={corr_act:.3f} corr_all={corr_all:.3f} "
          f"relerr={relerr:.3f}  (corr tol {args.corr_tol})", flush=True)

    ok = bool(gc_ok and misfit_ok and induced_ok)
    rec.update(dict(JT0=JT0, JS0=JS0, JTf=JTf, JSf=JSf, JT_ratio=JTr, JS_ratio=JSr,
                    misfit_mean=misfit_mean, misfit_ok=misfit_ok, corr_active=corr_act,
                    corr_all=corr_all, induced_relerr=relerr, induced_ok=induced_ok,
                    m_truth_mean=float(mck_t[wet].mean()), m_truth_std=float(mck_t[wet].std()),
                    n_iters=len(hist), peak_gb=pk, bwd_s=bwd_s, ok=ok))

    args.outdir.mkdir(parents=True, exist_ok=True)
    npz = args.outdir / "nn_twin.npz"
    np.savez(npz, loss_hist=np.array(losshist), m_truth_ck=mck_t, m_rec_ck=mck_r,
             wet=wet, active=act, area=wa, JT0=JT0, JS0=JS0, JTf=JTf, JSf=JSf,
             corr_active=corr_act, corr_all=corr_all, N=n, config=cfg_name)
    _maybe_figure(args.outdir / "nn_twin.png", losshist, mck_t, mck_r, act, wet, args,
                  corr_act, misfit_mean, cfg_name)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  wrote {npz}; appended summary -> {args.results}", flush=True)
    print(f"  gate: grad-check={gc_ok}  evolution={misfit_ok} (mean ratio<{args.misfit_tol})  "
          f"induced-mixing={induced_ok} (corr>{args.corr_tol})", flush=True)
    print(f"NN_TWIN_{'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def _bail(args, rec, phase, e):
    oom = is_oom(e)
    msg = f"{type(e).__name__}: {str(e)[:240]}"
    rec.update(dict(ok=False, phase=phase, oom=oom, peak_gb=peak_gb(), error=msg))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  FAILED in {phase}: {msg}", flush=True)
    print(f"NN_TWIN_{'OOM' if oom else 'FAIL'}", flush=True)
    return 2 if oom else 1


def _maybe_figure(path, losshist, mck_t, mck_r, act, wet, args, corr_act, misfit_mean, cfg_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] skipped ({e})", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].semilogy(np.arange(1, len(losshist) + 1), losshist, ".-", ms=4, lw=1)
    ax[0].set(xlabel="Adam iteration", ylabel="loss  J_T/J_T0 + J_S/J_S0",
              title=f"E1 NN-twin recovery ({cfg_name})\nmean misfit ratio {misfit_mean:.1e}")
    ax[0].grid(True, which="both", alpha=0.25)
    a = act.astype(bool)
    ax[1].scatter(mck_t[wet & ~a], mck_r[wet & ~a], s=3, c="0.7", linewidths=0,
                  rasterized=True, label="other wet cols")
    ax[1].scatter(mck_t[a], mck_r[a], s=5, c="C0", linewidths=0, rasterized=True,
                  label="constrained (top-quartile)")
    lo = float(min(mck_t[wet].min(), mck_r[wet].min())); hi = float(max(mck_t[wet].max(), mck_r[wet].max()))
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax[1].set(xlabel="truth c_k multiplier", ylabel="recovered c_k multiplier",
              title=f"induced-mixing field recovered\ncorr(active)={corr_act:.3f}")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  [fig] wrote {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
