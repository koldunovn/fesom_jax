"""Batched-window NN-of-TKE twin — §3 / E2 capability (the fix for the single-window field limit).

The single-window E1 twin (`core2_paper_nn_twin.py`) proved the NN-weight gradient through the full
global model is correct and recovers the T/S evolution, BUT the induced-mixing FIELD is under-recovered:
the adjoint window is memory-capped at N≈10 (~5 h) on the 80 GB A100, and 5 h of evolution does not
uniquely constrain a rich per-column NN multiplier (short-window non-uniqueness). The principled fix is
**batched windows** (the plan's D2a/E2 capability): the memory wall is PER-WINDOW (one backward through N
steps), so train on K short chunks that start from K DIFFERENT ocean states and **accumulate the gradient
across chunks** — each backward stays at the fitting N, but the NN sees K× more of the feature space, which
is what makes the field identifiable.

Design (teacher-forced chunked trajectory / TBPTT):
  1. Run the truth NN forward as K chained N-step chunks; snapshot the truth start-state ``S_k`` of each
     chunk (forward-only ⇒ cheap; one full State on device at a time, snapshots staged to HOST).
  2. Train: each iter, for each chunk k, take the gradient of the per-chunk misfit between
     ``model(S_k, forcing_k, traineeNN)`` and the truth target, ACCUMULATE ``G=Σ_k g_k`` (only one chunk's
     backward live at a time — device_put S_k, use, free), then one Adam step. One jitted ``value_and_grad``
     (fixed shapes) is reused across all K chunks.

Token **NN_TWIN_BATCHED_OK** (induced-mixing field recovered across the batched windows + per-chunk
evolution recovered). Single-GPU; reuses the E1 remat_blocks fix so each chunk backward fits.

Usage (GPU):  python scripts/paper/core2_paper_nn_twin_batched.py --n 10 --windows 8 --config all3 --from-s0 <pkl>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (x64)
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fesom_jax import calibrate, tke_nn
from fesom_jax.integrate import integrate

import core2_paper_nn_twin as twin  # reuse build/make_truth_nn/reconstruct_features/helpers

DT = twin.DT


def _wcorr(a, b, w):
    w = w / w.sum()
    am, bm = (w * a).sum(), (w * b).sum()
    ca, cb = a - am, b - bm
    return float((w * ca * cb).sum() / np.sqrt((w * ca * ca).sum() * (w * cb * cb).sum() + 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="per-chunk steps (the memory-feasible window)")
    ap.add_argument("--windows", type=int, default=8, help="K chained chunks (the batch)")
    ap.add_argument("--config", choices=("all3", "tkegm"), default="all3")
    ap.add_argument("--iters", type=int, default=120, help="max Adam iterations (full-batch grad-accum)")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truth-amp", type=float, default=4.0)
    ap.add_argument("--truth-bias", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 16])
    ap.add_argument("--misfit-tol", type=float, default=0.15, help="per-chunk mean evolution-ratio gate")
    ap.add_argument("--corr-tol", type=float, default=0.9)
    ap.add_argument("--from-s0", type=Path, default=None, help="spun-up state pickle (chunk-0 start)")
    ap.add_argument("--snapshots-dir", type=Path, default=None,
                    help="SEASONAL mode: dir of snap_step<N>.pkl (from core2_nn_snapshots_1yr.sbatch). "
                         "Chunk-start states are loaded from DIFFERENT months (winter/summer ⇒ diverse "
                         "N²/MLD/mixing) instead of consecutive chunks — the diversity the field needs.")
    ap.add_argument("--skip-before", type=int, default=4320,
                    help="seasonal: ignore snapshots before this step (spin-up transient; ~90 d default)")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--results", type=Path, default=twin.ROOT / "scripts" / "nn_twin_batched_results.jsonl")
    ap.add_argument("--outdir", type=Path, default=twin.ROOT / "scripts")
    ap.add_argument("--remat-blocks", dest="remat_blocks", action="store_true", default=True)
    ap.add_argument("--no-remat-blocks", dest="remat_blocks", action="store_false")
    args = ap.parse_args()
    N, K = args.n, args.windows
    t0 = time.time()

    seasonal = args.snapshots_dir is not None
    nbuild = N if seasonal else N * K
    mesh, state, op, fs, sfs, sf_last, cfgs = twin.build(args.year, nbuild, args.config)
    if args.from_s0 is not None and not seasonal:
        import pickle
        with open(args.from_s0, "rb") as fpk:
            state = jax.device_put(pickle.load(fpk))
        print(f"[setup] loaded spun-up S0 <- {args.from_s0}", flush=True)

    gpu_gb = twin.gpu_limit_gb()
    node_mask = jnp.asarray(mesh.node_layer_mask)
    w3d = jnp.asarray(mesh.area[:, 0])[:, None] * node_mask
    area0 = np.asarray(mesh.area[:, 0])
    wet = np.asarray(node_mask[:, 0]).astype(bool)

    def wmse(a, b):
        d = a - b
        return jnp.sum(w3d * d * d) / jnp.sum(w3d)

    def chunk_full(nn, s0, frc_k):
        p = calibrate.build_params({"tke_nn": nn})
        return integrate(s0, mesh, op, None, n_steps=N, dt=DT, step_forcings=frc_k,
                         forcing_static=fs, params=p, remat_blocks=args.remat_blocks, **cfgs)

    def chunk_ts(nn, s0, frc_k):                       # T/S only ⇒ memory-lean backward (the E1 lesson)
        fin = chunk_full(nn, s0, frc_k)
        return fin.T, fin.S

    rec = {"target": "nn_twin_batched", "config": args.config, "N": N, "windows": K, "mode":
           "seasonal" if seasonal else "chained", "dt": DT, "seed": args.seed,
           "truth_amp": args.truth_amp, "truth_bias": args.truth_bias, "hidden": list(args.hidden),
           "gpu_gb": gpu_gb, "remat_blocks": bool(args.remat_blocks)}

    # ---------- chunk starts + per-chunk forcing (SEASONAL: snapshots from different months;
    #            CHAINED: consecutive N-step chunks from one S0) ----------
    try:
        t1 = time.time()
        truth_nn = twin.make_truth_nn(args.seed, args.truth_amp, args.truth_bias)
        starts_host = []                                # chunk-start States on HOST (one on device at a time)
        frc = []                                        # per-chunk forcing
        targets = []                                    # (T,S) truth targets (device)
        feats_list = []                                 # truth column features at each chunk start
        if seasonal:
            import glob
            import pickle
            import re
            from fesom_jax import surface_forcing
            from fesom_jax.phc_ic import phc_initial_state
            files = sorted(glob.glob(str(args.snapshots_dir / "snap_step*.pkl")))
            pairs = [(int(re.search(r"snap_step(\d+)", f).group(1)), f) for f in files
                     if re.search(r"snap_step(\d+)", f)]
            pairs = [p for p in pairs if p[0] >= args.skip_before]
            if not pairs:
                raise RuntimeError(f"no snapshots >= step {args.skip_before} in {args.snapshots_dir}")
            if len(pairs) > K:                          # evenly subsample to K across the year
                idx = sorted(set(np.linspace(0, len(pairs) - 1, K).round().astype(int).tolist()))
                pairs = [pairs[i] for i in idx]
            K = len(pairs)
            base = phc_initial_state(mesh, twin.IC_DIR)
            cf = surface_forcing.build_surface_forcing(mesh, args.year, sst_ic=np.asarray(base.T[:, 0]))
            for s, f in pairs:
                with open(f, "rb") as fp:
                    starts_host.append(pickle.load(fp))  # host State (saved via device_get)
                dts = surface_forcing.dates_for_steps(args.year, DT, s + N)[s:s + N]
                frc.append(cf.stack(dts))
            days = [round(s * DT / 86400) for s, _ in pairs]
            print(f"[setup] SEASONAL: {K} snapshots at days {days} (steps {[s for s,_ in pairs]})",
                  flush=True)
            for k in range(K):
                s_dev = jax.device_put(starts_host[k])
                sf0 = jax.tree.map(lambda x: x[0], frc[k])
                feats_list.append(twin.reconstruct_features(mesh, s_dev, sf0, fs))
                end = chunk_full(truth_nn, s_dev, frc[k])
                end.T.block_until_ready()
                targets.append((end.T, end.S))
                del s_dev, end
        else:
            frc = [jax.tree.map(lambda x, k=k: x[k * N:(k + 1) * N], sfs) for k in range(K)]
            st = state
            for k in range(K):
                starts_host.append(jax.device_get(st))
                sf0 = jax.tree.map(lambda x: x[0], frc[k])
                feats_list.append(twin.reconstruct_features(mesh, st, sf0, fs))
                end = chunk_full(truth_nn, st, frc[k])
                end.T.block_until_ready()
                targets.append((end.T, end.S))
                st = end
            del st
        rec["span_days"] = K * N * DT / 86400.0
        m_truth0 = np.asarray(tke_nn.multiplier(truth_nn, feats_list[0])[:, 0])
        m_truth_ex = np.stack([np.asarray(tke_nn.multiplier(truth_nn, f)[:, 0]) for f in feats_list])
        print(f"[truth] {('seasonal' if seasonal else 'chained')} {K} chunks in {time.time()-t1:.1f}s  "
              f"peak={twin.peak_gb():.1f} GB", flush=True)
        print(f"[truth] c_k mult @chunk0: mean={m_truth0[wet].mean():.3f} "
              f"[{m_truth0[wet].min():.3f},{m_truth0[wet].max():.3f}] std={m_truth0[wet].std():.3f}", flush=True)
    except Exception as e:
        return _bail(args, rec, "truth", e)

    # ---------- trainee NN→0 + per-chunk baselines (reuse model on each start) ----------
    trainee0 = tke_nn.init_tke_nn(jax.random.PRNGKey(args.seed + 777), hidden=tuple(args.hidden),
                                  zero_last=True)
    try:
        JT0, JS0 = [], []
        for k in range(K):
            s_dev = jax.device_put(starts_host[k])
            T0, S0_ = chunk_ts(trainee0, s_dev, frc[k])
            JT0.append(float(wmse(T0, targets[k][0]))); JS0.append(float(wmse(S0_, targets[k][1])))
            del s_dev, T0, S0_
        if not (min(JT0) > 0 and min(JS0) > 0):
            raise ValueError(f"degenerate baseline JT0={JT0} JS0={JS0}")
        print(f"[baseline] NN→0 per-chunk J_T0 mean={np.mean(JT0):.3e}  J_S0 mean={np.mean(JS0):.3e}",
              flush=True)
    except Exception as e:
        return _bail(args, rec, "baseline", e)

    # ---------- batched recovery: per-chunk value_and_grad, accumulate G=Σ g_k, one Adam step ----------
    @jax.jit
    def chunk_vg(nn, s0, frc_k, tT, tS, jt0, js0):
        def L(nn_):
            T, S = chunk_ts(nn_, s0, frc_k)
            return wmse(T, tT) / jt0 + wmse(S, tS) / js0
        return jax.value_and_grad(L)(nn)

    opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.iters))
    params = trainee0
    opt_state = opt.init(params)
    losshist = []
    try:
        t1 = time.time()
        print(f"\n  --- batched recovery: NN→0 → truth over K={K} chunks (Adam lr={args.lr}, cosine, "
              f"≤{args.iters} it; per-chunk loss J_T/J_T0+J_S/J_S0, start≈2.0) ---", flush=True)
        for it in range(args.iters):
            Gsum = None
            Lsum = 0.0
            gnorm = 0.0
            for k in range(K):
                s_dev = jax.device_put(starts_host[k])  # only ONE chunk-start on device at a time
                l, g = chunk_vg(params, s_dev, frc[k], targets[k][0], targets[k][1], JT0[k], JS0[k])
                Lsum += float(l)
                Gsum = g if Gsum is None else jax.tree.map(jnp.add, Gsum, g)
                del s_dev
            Gavg = jax.tree.map(lambda x: x / K, Gsum)
            updates, opt_state = opt.update(Gavg, opt_state)
            params = optax.apply_updates(params, updates)
            losshist.append(Lsum / K)
            if it % 10 == 0 or it == args.iters - 1:
                gnorm = float(optax.global_norm(Gavg))
                print(f"    it={it:3d}  mean loss={Lsum/K:.6e}  |g|={gnorm:.2e}  "
                      f"peak={twin.peak_gb():.1f} GB", flush=True)
            if Lsum / K < 2.0 * args.misfit_tol:
                print(f"    (reached batch-mean loss < {2*args.misfit_tol}) at it={it}", flush=True)
                break
            if len(losshist) > 25 and (losshist[-25] - losshist[-1]) < 1e-3 * losshist[0]:
                print(f"    (plateau) at it={it}", flush=True)
                break
        rec_nn = params
        bwd_s = time.time() - t1
    except Exception as e:
        return _bail(args, rec, "recover", e)

    # ---------- final per-chunk evolution misfit + induced-field recovery (across exercised states) ------
    try:
        JTf, JSf = [], []
        for k in range(K):
            s_dev = jax.device_put(starts_host[k])
            Tf, Sf = chunk_ts(rec_nn, s_dev, frc[k])
            JTf.append(float(wmse(Tf, targets[k][0]))); JSf.append(float(wmse(Sf, targets[k][1])))
            del s_dev, Tf, Sf
        JTr = float(np.mean([f / b for f, b in zip(JTf, JT0)]))
        JSr = float(np.mean([f / b for f, b in zip(JSf, JS0)]))
        misfit_mean = 0.5 * (JTr + JSr)
        misfit_ok = bool(misfit_mean < args.misfit_tol)

        # induced multiplier: trainee vs truth at S0 (apples-to-apples with E1) AND over ALL exercised
        # chunk-start states (the feature space the batch actually constrains).
        m_rec0 = np.asarray(tke_nn.multiplier(rec_nn, feats_list[0])[:, 0])
        m_rec_ex = np.stack([np.asarray(tke_nn.multiplier(rec_nn, f)[:, 0]) for f in feats_list])
        wa = area0
        wet_b = np.broadcast_to(wet, m_truth_ex.shape)
        dev0 = np.abs(m_truth0 - 1.0) * wet
        thr0 = np.quantile(dev0[wet], 0.75) if wet.any() else 0.0
        act0 = wet & (dev0 >= thr0)
        corr_S0 = _wcorr(m_truth0[act0], m_rec0[act0], wa[act0]) if act0.any() else 0.0
        # exercised-space corr: stack all chunks' active columns
        devx = np.abs(m_truth_ex - 1.0) * wet_b
        thrx = np.quantile(devx[wet_b], 0.75)
        actx = wet_b & (devx >= thrx)
        wax = np.broadcast_to(wa, m_truth_ex.shape)
        corr_ex = _wcorr(m_truth_ex[actx], m_rec_ex[actx], wax[actx]) if actx.any() else 0.0
        corr_all = _wcorr(m_truth_ex[wet_b], m_rec_ex[wet_b], wax[wet_b])
        induced_ok = bool(corr_ex > args.corr_tol)
    except Exception as e:
        return _bail(args, rec, "induced", e)

    pk = twin.peak_gb()
    ok = bool(misfit_ok and induced_ok)
    print(f"\n  recovered NN in {len(losshist)} it / {bwd_s:.1f}s  peak={pk:.1f}/{gpu_gb:.0f} GB", flush=True)
    print(f"  EVOLUTION (per-chunk mean): J_T ratio={JTr:.2e}  J_S ratio={JSr:.2e}  "
          f"mean={misfit_mean:.2e}  (tol {args.misfit_tol})", flush=True)
    print(f"  INDUCED MIXING (c_k mult): corr_exercised={corr_ex:.3f}  corr_S0={corr_S0:.3f}  "
          f"corr_all={corr_all:.3f}  (corr tol {args.corr_tol})", flush=True)
    print(f"  gate: evolution={misfit_ok} (mean<{args.misfit_tol})  "
          f"induced={induced_ok} (corr_exercised>{args.corr_tol})", flush=True)

    rec.update(dict(JT_ratio=JTr, JS_ratio=JSr, misfit_mean=misfit_mean, misfit_ok=misfit_ok,
                    corr_exercised=corr_ex, corr_S0=corr_S0, corr_all=corr_all, induced_ok=induced_ok,
                    n_iters=len(losshist), peak_gb=pk, bwd_s=bwd_s,
                    m_truth_mean=float(m_truth0[wet].mean()), m_truth_std=float(m_truth0[wet].std()),
                    JT0_mean=float(np.mean(JT0)), JS0_mean=float(np.mean(JS0)), ok=ok))

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez(args.outdir / "nn_twin_batched.npz", loss_hist=np.array(losshist),
             m_truth_ex=m_truth_ex, m_rec_ex=m_rec_ex, active_ex=actx, wet=wet, area=wa,
             corr_exercised=corr_ex, corr_S0=corr_S0, N=N, windows=K, config=args.config)
    _figure(args.outdir / "nn_twin_batched.png", losshist, m_truth_ex, m_rec_ex, actx, wet_b,
            corr_ex, misfit_mean, args.config, K, N)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  appended summary -> {args.results}", flush=True)
    print(f"NN_TWIN_BATCHED_{'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def _bail(args, rec, phase, e):
    oom = twin.is_oom(e)
    msg = f"{type(e).__name__}: {str(e)[:240]}"
    rec.update(dict(ok=False, phase=phase, oom=oom, peak_gb=twin.peak_gb(), error=msg))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  FAILED in {phase}: {msg}", flush=True)
    print(f"NN_TWIN_BATCHED_{'OOM' if oom else 'FAIL'}", flush=True)
    return 2 if oom else 1


def _figure(path, losshist, m_truth_ex, m_rec_ex, actx, wet_b, corr_ex, misfit_mean, cfg, K, N):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] skipped ({e})", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].semilogy(np.arange(1, len(losshist) + 1), losshist, ".-", ms=4, lw=1)
    ax[0].set(xlabel="Adam iteration", ylabel="batch-mean loss",
              title=f"E2 batched NN-twin ({cfg}, K={K}×N={N})\nmean misfit ratio {misfit_mean:.1e}")
    ax[0].grid(True, which="both", alpha=0.25)
    a = actx.astype(bool)
    mt, mr = m_truth_ex[wet_b & ~a], m_rec_ex[wet_b & ~a]
    ax[1].scatter(mt, mr, s=2, c="0.7", linewidths=0, rasterized=True, label="other wet cols")
    ax[1].scatter(m_truth_ex[a], m_rec_ex[a], s=4, c="C0", linewidths=0, rasterized=True,
                  label="constrained (top-quartile)")
    lo = float(min(m_truth_ex[wet_b].min(), m_rec_ex[wet_b].min()))
    hi = float(max(m_truth_ex[wet_b].max(), m_rec_ex[wet_b].max()))
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax[1].set(xlabel="truth c_k multiplier", ylabel="recovered c_k multiplier",
              title=f"induced field over {K} exercised states\ncorr(exercised)={corr_ex:.3f}")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  [fig] wrote {path}", flush=True)


if __name__ == "__main__":
    import sys
    sys.exit(main())
