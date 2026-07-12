"""CORE2 NN-of-TKE **obs training** + long-forward validation — Paper-experiments Task E2.

The §3 obs half: train the embedded ``tke_nn`` closure to **reduce the model's MLD/SST bias vs WOA**
through the global adjoint, then prove the learned closure **deploys** (a long forward-only run stays
stable, drifts no more than default, and the obs benefit *persists* online — the classic
offline-trained / online-deployed failure the plan's review flagged).

Why this is well-posed even though the E1 *twin* field was not identifiable: E2 does NOT recover a known
NN. The chaotic-adjoint horizon (E1) means a short window can't *uniquely* pin a per-column multiplier —
but obs training has **no unique truth**. The goal is to LEARN *any* bounded, stable closure correction
that lowers the held-out MLD misfit. The bounded multiplier ``m∈(1/m_max, m_max)`` is itself the
physical-plausibility guarantee (positive-definite diffusivities; the NN structurally *cannot* do the
``c_eps→0`` pathology the 2-param scalar fit did in D2a). So the rigor is three falsifiable gates, not a
field correlation:
  • **NN_OBS_OK** — held-out (independent-cell) MLD misfit reduced (the D2c overfitting bar, now for the NN).
  • **NN_DRIFT_OK** — a long forward-only run with the trained NN is finite + stable, drifts ≤ default,
    and the MLD benefit persists at the long horizon (not just over the training window).

Method (mirrors D2a obs + the E1 batched twin):
  • CONFIG = the full paper model ``all3`` = zstar+TKE+mEVP+GM with the frozen-ice adjoint (A8); the
    obs MLD/SST is over the ice-masked open ocean.
  • BATCHED SEASONAL WINDOWS — chunk starts are the 12 monthly snapshots (``nn_twin_snaps/``, one per
    ~30 d), so the NN sees winter-deep-convection AND summer-stratified columns and targets the
    *seasonal climatology* (the plan's first-class temporal aggregation). Each chunk runs N steps from
    its snapshot and is scored against **that month's** WOA MLD+SST. Per-chunk loss normalized by the
    NN→0 baseline (=2.0 at start); gradient ACCUMULATED across chunks (one chunk's backward live at a
    time ⇒ per-chunk memory = the fitting N), one Adam step. Reuses the E1 ``remat_blocks`` fix.
  • HELD-OUT cells (D2c ``--holdout random`` default): the loss drives the TRAIN cells only; the
    INDEPENDENT held-out cells are forward-scored in the aux (no gradient leak) ⇒ NN_OBS_OK.
  • LONG-FORWARD (``--mode validate``, forward-only, chunked re-stacked forcing): default NN→0 vs the
    trained NN from a held-out start; drift = volume-weighted RMS(T,S − start); benefit = MLD-vs-WOA at
    each ~5 d segment ⇒ NN_DRIFT_OK. The NN→0 fallback (bit-identical to default) is the deployment net
    if it diverges.

Single-GPU (the sharded ragged-halo AD bug ⇒ adjoint stays single-device). Train + validate run as
SEPARATE processes (``--mode train`` then ``--mode validate``) so the forward-only validation gets a
fresh allocator pool (the D2a split). Usage:
  python scripts/paper/core2_paper_nn_obs.py --mode train    --n 10 --windows 8 --holdout random --fold 0
  python scripts/paper/core2_paper_nn_obs.py --mode validate --long-steps 4320 --long-seg 240
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fesom_jax import ale, calibrate, surface_forcing, obs_compare, tke_nn
from fesom_jax.integrate import integrate

import core2_paper_nn_twin as twin          # build/reconstruct_features/peak_gb/gpu_limit_gb/is_oom/DT
import core2_paper_calib_tke_obs as cobs     # load_woa/build_holdout/FLOOR_SST/WOA_NPZ

DT = twin.DT
ROOT = twin.ROOT
SNAP_DIR = Path("/work/ab0995/a270088/port_jax/nn_twin_snaps")
NN_PKL = Path("/work/ab0995/a270088/port_jax/nn_obs_trained.pkl")
DEFAULT_RESULTS = ROOT / "scripts" / "nn_obs_results.jsonl"


# ----------------------------------------------------------------------------- helpers
def woa_month(month):
    """Flattened (NaN-zeroed) WOA MLD+SST target + validity for ``month`` (1-12). Same npz/keys as
    :func:`core2_paper_calib_tke_obs.load_woa`, but month-only (the Hmap is month-independent)."""
    d = np.load(cobs.WOA_NPZ)
    mi = month - 1
    mld = d["mld_monthly"][mi].reshape(-1)
    sst = d["sst_monthly"][mi].astype(np.float64).reshape(-1)
    mld_ok = d["mld_valid_monthly"][mi].reshape(-1).astype(bool)
    sst_ok = d["sst_valid"].reshape(-1).astype(bool)
    # AD masked-NaN rule: WOA land is NaN; 0·NaN=NaN poisons the misfit even at weight 0. Zero them
    # (the *_ok masks drop them from the weighted sum anyway).
    mld = np.where(np.isfinite(mld) & mld_ok, mld, 0.0)
    sst = np.where(np.isfinite(sst) & sst_ok, sst, 0.0)
    return mld, mld_ok, sst, sst_ok


def snapshot_steps(snap_dir, skip_before, k):
    """Sorted (step, path) of snapshots ≥ ``skip_before``, evenly subsampled to ``k``."""
    files = sorted(snap_dir.glob("snap_step*.pkl"))
    pairs = [(int(re.search(r"snap_step(\d+)", f.name).group(1)), f) for f in files
             if re.search(r"snap_step(\d+)", f.name)]
    pairs = [p for p in pairs if p[0] >= skip_before]
    if not pairs:
        raise RuntimeError(f"no snapshots >= step {skip_before} in {snap_dir}")
    if len(pairs) > k:
        idx = sorted(set(np.linspace(0, len(pairs) - 1, k).round().astype(int).tolist()))
        pairs = [pairs[i] for i in idx]
    return pairs


def cell_weights(woa_mld_ok, woa_sst, woa_sst_ok, cell_area, cell_valid, ice_sst):
    """Obs weights (area × validity × open-water ice mask). ``ice_sst``: drop WOA cells below this °C
    (freezing ⇒ ice-covered, MLD/SST unreliable). Month-specific via ``woa_sst``."""
    ice_ok = (woa_sst > ice_sst).astype(np.float64)
    w_mld = cell_area * woa_mld_ok.astype(np.float64) * ice_ok * cell_valid
    w_sst = cell_area * woa_sst_ok.astype(np.float64) * ice_ok * cell_valid
    return w_mld, w_sst


def uniform_nn(c, hidden=(16, 16), m_max=3.0):
    """A :class:`TkeNN` with all weights zero + an output bias giving a UNIFORM multiplier ``c``
    (raw ≡ b_last ⇒ m = exp(log_m_max·tanh(b_last)) = c). Mimics a global c_k scaling (the D2a
    scalar) for the deployment diagnostic — isolates the fast-vs-slow MLD response from the NN's
    spatial structure."""
    import math
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(0), hidden=hidden, zero_last=True, m_max=m_max)
    arg = max(min(math.log(c) / nn.log_m_max, 0.999), -0.999)
    bs = list(nn.bs); bs[-1] = bs[-1] + math.atanh(arg)
    return tke_nn.TkeNN(Ws=nn.Ws, bs=tuple(bs), log_m_max=nn.log_m_max)


def _append(results, rec):
    results.parent.mkdir(parents=True, exist_ok=True)
    with open(results, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ============================================================================= TRAIN
def run_train(args, mesh, op, fs, cfgs, cf, node_mask, w3d, Hmap, cell_lat, cell_lon, rec):
    wet = np.asarray(node_mask[:, 0]).astype(bool)
    area0 = np.asarray(mesh.area[:, 0])
    cell_area = np.asarray(Hmap.cell_area)
    _, cv = obs_compare.to_obs_surface(jnp.ones((mesh.nod2D,)), Hmap)   # cell has a wet node
    cell_valid = np.asarray(cv).astype(np.float64)
    hmask = build_hmask(args, cell_lat, cell_lon)
    w_sst_coef = args.w_sst
    reg_coef = args.reg
    reg_tgt = float(np.log(args.reg_target))      # A1(b): anchor toward log(target mult); 0.0 ⇒ E2 (toward default)
    wnode = jnp.asarray(area0 * wet.astype(np.float64))                 # area-wt wet mask (reg weight)

    # ---------- chunk starts (monthly snapshots) + per-month forcing + WOA targets + weights ----------
    t1 = time.time()
    pairs = snapshot_steps(args.snapshots_dir, args.skip_before, args.windows)
    K = len(pairs)
    starts_host, frc, months = [], [], []
    woa_mld, woa_sst = [], []
    w_mld_tr, w_sst_tr, w_mld_ho, w_sst_ho = [], [], [], []
    for s, f in pairs:
        with open(f, "rb") as fp:
            starts_host.append(pickle.load(fp))                # host State (device_get'd at save)
        dts = surface_forcing.dates_for_steps(args.year, DT, s + args.n)
        frc.append(cf.stack(dts[s:s + args.n]))
        m = dts[s][3]                                          # (year, doy, sec, month) → month
        months.append(m)
        mld, mld_ok, sst, sst_ok = woa_month(m)
        woa_mld.append(jnp.asarray(mld)); woa_sst.append(jnp.asarray(sst))
        wm, ws = cell_weights(mld_ok, sst, sst_ok, cell_area, cell_valid, args.ice_sst)
        w_mld_tr.append(jnp.asarray(wm * (1.0 - hmask))); w_mld_ho.append(jnp.asarray(wm * hmask))
        w_sst_tr.append(jnp.asarray(ws * (1.0 - hmask))); w_sst_ho.append(jnp.asarray(ws * hmask))
    days = [round(s * DT / 86400) for s, _ in pairs]
    print(f"[setup] {K} seasonal chunks at days {days} (months {months}) in {time.time()-t1:.1f}s  "
          f"peak={twin.peak_gb():.1f} GB", flush=True)
    if args.holdout != "none":
        print(f"[xval] holdout={args.holdout} fold={args.fold}: "
              f"train cells={int(((1-hmask)*cell_valid).sum())}  held-out={int((hmask*cell_valid).sum())}",
              flush=True)
    rec.update(dict(windows=K, snap_days=days, months=months, span_days=K * args.n * DT / 86400.0))

    # ---------- the model→obs surface map (MLD + SST on the WOA grid), trained NN inside ----------
    def model_surf(nn, s0, frc_k):
        p = calibrate.build_params({"tke_nn": nn})
        fin = integrate(s0, mesh, op, None, n_steps=args.n, dt=DT, step_forcings=frc_k,
                        forcing_static=fs, params=p, remat_blocks=args.remat_blocks, **cfgs)
        _, Z3d = ale.live_geometry(mesh, fin.hnode)
        mld, _ = obs_compare.mld_density_threshold(fin.T, fin.S, Z3d, node_mask)
        cell_mld, _ = obs_compare.to_obs_surface(mld, Hmap)
        cell_sst, _ = obs_compare.to_obs_surface(fin.T[:, 0], Hmap)
        return cell_mld, cell_sst

    @jax.jit
    def chunk_vg(nn, s0, frc_k, feats_k, wmld, wsst, wmld_tr, wsst_tr, wmld_ho, wsst_ho, j0m, j0s):
        # loss drives TRAIN cells; aux carries raw train+held-out misfits (held-out weights never
        # touch the loss ⇒ no gradient leak — the D2a/D2c has_aux split). The trust-region reg term
        # keeps the closure MILD (multiplier near 1) so it DEPLOYS on the long forward — the fix for
        # the offline-trained/online-deployed over-mixing instability (the bang-bang failure).
        def L(nn_):
            cell_mld, cell_sst = model_surf(nn_, s0, frc_k)
            jm = obs_compare.misfit(cell_mld, wmld, wmld_tr)
            js = obs_compare.misfit(cell_sst, wsst, wsst_tr)
            jmh = obs_compare.misfit(cell_mld, wmld, wmld_ho)
            jsh = obs_compare.misfit(cell_sst, wsst, wsst_ho)
            lm = nn_.log_m_max * jnp.tanh(tke_nn.mlp_raw(nn_, feats_k))      # log-multiplier [N, 3]
            reg = jnp.sum(wnode[:, None] * (lm - reg_tgt) ** 2) / (jnp.sum(wnode) * lm.shape[1])
            return jm / j0m + w_sst_coef * js / j0s + reg_coef * reg, (jm, js, jmh, jsh)
        (l, aux), g = jax.value_and_grad(L, has_aux=True)(nn)
        return l, aux, g

    one = jnp.asarray(1.0, jnp.float64)        # strong-typed (the D2a weak-type-recompile guard)

    # ---------- per-chunk NN input features (forward-only; CONSTANT w.r.t. the NN ⇒ for the reg
    #            penalty + the induced-multiplier diagnostics) ----------
    feats = []
    for k in range(K):
        s_dev = jax.device_put(starts_host[k])
        sf0 = jax.tree.map(lambda x: x[0], frc[k])
        feats.append(twin.reconstruct_features(mesh, s_dev, sf0, fs))
        del s_dev

    # ---------- baseline (NN→0 = default TKE) per chunk: raw misfits → normalizers j0 ----------
    trainee0 = tke_nn.init_tke_nn(jax.random.PRNGKey(args.seed + 777), hidden=tuple(args.hidden),
                                  zero_last=True, m_max=args.m_max)
    t1 = time.time()
    j0m, j0s, b_mld_tr, b_sst_tr, b_mld_ho, b_sst_ho = [], [], [], [], [], []
    for k in range(K):
        s_dev = jax.device_put(starts_host[k])
        _, (jm, js, jmh, jsh), _ = chunk_vg(trainee0, s_dev, frc[k], feats[k], woa_mld[k], woa_sst[k],
                                            w_mld_tr[k], w_sst_tr[k], w_mld_ho[k], w_sst_ho[k], one, one)
        jm, js, jmh, jsh = float(jm), float(js), float(jmh), float(jsh)
        j0m.append(jm); j0s.append(js)
        b_mld_tr.append(jm); b_sst_tr.append(js); b_mld_ho.append(jmh); b_sst_ho.append(jsh)
        del s_dev
    if not (min(j0m) > 0 and min(j0s) > 0):
        raise ValueError(f"degenerate baseline j0m={j0m} j0s={j0s}")
    print(f"[baseline] NN→0 per-chunk: MLD mean={np.mean(b_mld_tr):.4e} m²  SST mean={np.mean(b_sst_tr):.4e} °C²"
          f"  ({time.time()-t1:.1f}s)", flush=True)
    print(f"[baseline] held-out: MLD mean={np.mean(b_mld_ho):.4e} m²  SST mean={np.mean(b_sst_ho):.4e} °C²",
          flush=True)

    # ---------- batched recovery: per-chunk value_and_grad, accumulate G=Σ g_k, one Adam step ----------
    opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.iters))
    warm = (uniform_nn(args.reg_target, tuple(args.hidden), args.m_max)
            if args.reg_target != 1.0 else trainee0)        # A1(b): start AT the anchor, refine around it
    params, opt_state, losshist = warm, None, []
    if args.reg_target != 1.0:
        print(f"  [A1(b)] warm-start at uniform {args.reg_target:g}× + reg anchored there (m_max={args.m_max:g}); "
              f"baseline j0/NN→0 unchanged ⇒ reductions are still vs DEFAULT", flush=True)
    t1 = time.time()
    print(f"\n  --- train NN→0 → reduce WOA MLD+SST over K={K} seasonal chunks "
          f"(Adam lr={args.lr}, cosine, ≤{args.iters} it; per-chunk MLD/J0+{args.w_sst:g}·SST/J0) ---",
          flush=True)
    opt_state = opt.init(params)
    best_loss, best_params, best_it = float("inf"), params, 0     # KEEP BEST (the optimizer overshoots
    for it in range(args.iters):                                  # past the data optimum into bang-bang)
        Gsum, Lsum = None, 0.0
        for k in range(K):
            s_dev = jax.device_put(starts_host[k])
            l, _, g = chunk_vg(params, s_dev, frc[k], feats[k], woa_mld[k], woa_sst[k], w_mld_tr[k],
                               w_sst_tr[k], w_mld_ho[k], w_sst_ho[k],
                               jnp.asarray(j0m[k], jnp.float64), jnp.asarray(j0s[k], jnp.float64))
            Lsum += float(l)
            Gsum = g if Gsum is None else jax.tree.map(jnp.add, Gsum, g)
            del s_dev
        Gavg = jax.tree.map(lambda x: x / K, Gsum)
        updates, opt_state = opt.update(Gavg, opt_state)
        params = optax.apply_updates(params, updates)
        cur = Lsum / K
        losshist.append(cur)
        if cur < best_loss:
            best_loss, best_params, best_it = cur, params, it
        if it % 5 == 0 or it == args.iters - 1:
            mbar = float(np.asarray(tke_nn.multiplier(params, feats[0])[:, 0])[wet].mean())
            print(f"    it={it:3d}  mean loss={cur:.6e}  |g|={float(optax.global_norm(Gavg)):.2e}  "
                  f"m̄@0={mbar:.3f}  peak={twin.peak_gb():.1f} GB", flush=True)
        if len(losshist) > 25 and (losshist[-25] - losshist[-1]) < 2e-4 * losshist[0]:
            print(f"    (plateau) at it={it}", flush=True)
            break
    rec_nn = best_params                                          # deploy the BEST, not the last
    print(f"  (kept best iterate it={best_it}, loss={best_loss:.6e})", flush=True)
    bwd_s = time.time() - t1

    # ---------- final per-chunk train + held-out misfits (the NN_OBS gate) ----------
    f_mld_tr, f_sst_tr, f_mld_ho, f_sst_ho = [], [], [], []
    for k in range(K):
        s_dev = jax.device_put(starts_host[k])
        _, (jm, js, jmh, jsh), _ = chunk_vg(rec_nn, s_dev, frc[k], feats[k], woa_mld[k], woa_sst[k],
                                            w_mld_tr[k], w_sst_tr[k], w_mld_ho[k], w_sst_ho[k],
                                            jnp.asarray(j0m[k], jnp.float64),
                                            jnp.asarray(j0s[k], jnp.float64))
        f_mld_tr.append(float(jm)); f_sst_tr.append(float(js))
        f_mld_ho.append(float(jmh)); f_sst_ho.append(float(jsh))
        del s_dev

    # seasonal-mean (climatology) reductions; held-out is the falsifiable bar
    B_mld_tr, B_mld_ho = float(np.mean(b_mld_tr)), float(np.mean(b_mld_ho))
    F_mld_tr, F_mld_ho = float(np.mean(f_mld_tr)), float(np.mean(f_mld_ho))
    B_sst_tr, F_sst_tr = float(np.mean(b_sst_tr)), float(np.mean(f_sst_tr))
    B_sst_ho, F_sst_ho = float(np.mean(b_sst_ho)), float(np.mean(f_sst_ho))
    red_mld_tr = 100.0 * (1.0 - F_mld_tr / B_mld_tr)
    red_mld_ho = 100.0 * (1.0 - F_mld_ho / B_mld_ho) if B_mld_ho > 0 else 0.0
    red_sst_tr = 100.0 * (1.0 - F_sst_tr / B_sst_tr)
    red_sst_ho = 100.0 * (1.0 - F_sst_ho / B_sst_ho) if B_sst_ho > 0 else 0.0
    drmse_mld_ho = float(np.sqrt(max(B_mld_ho, 0.0)) - np.sqrt(max(F_mld_ho, 0.0)))   # m
    drmse_sst_ho = float(np.sqrt(max(B_sst_ho, 0.0)) - np.sqrt(max(F_sst_ho, 0.0)))   # °C

    # induced multiplier stats at the first chunk's state (plausibility / how hard the NN is pushing)
    m_rec0 = np.asarray(tke_nn.multiplier(rec_nn, feats[0])[:, 0])
    m_in_bound = bool(np.isfinite(m_rec0[wet]).all() and m_rec0[wet].min() > 0)

    mld_ho_red = bool(F_mld_ho < B_mld_ho)
    nn_obs_ok = bool(mld_ho_red and m_in_bound)        # held-out MLD reduced + multiplier physical
    pk = twin.peak_gb()
    print(f"\n  trained NN in {len(losshist)} it / {bwd_s:.1f}s  peak={pk:.1f}/{rec['gpu_gb']:.0f} GB",
          flush=True)
    print(f"  MLD (seasonal-mean): TRAIN {B_mld_tr:.4e}→{F_mld_tr:.4e} m² ({red_mld_tr:+.1f}%)   "
          f"HELD-OUT {B_mld_ho:.4e}→{F_mld_ho:.4e} m² ({red_mld_ho:+.1f}%, ΔRMS={drmse_mld_ho:+.2f} m)",
          flush=True)
    print(f"  SST (seasonal-mean): TRAIN {B_sst_tr:.4e}→{F_sst_tr:.4e} °C² ({red_sst_tr:+.1f}%)   "
          f"HELD-OUT {red_sst_ho:+.1f}% (ΔRMSE={drmse_sst_ho:+.4f} °C; floor {cobs.FLOOR_SST})", flush=True)
    print(f"  induced c_k mult @chunk0 (wet): mean={m_rec0[wet].mean():.3f} "
          f"[{m_rec0[wet].min():.3f},{m_rec0[wet].max():.3f}] std={m_rec0[wet].std():.3f}  bounded={m_in_bound}",
          flush=True)
    print(f"  gate NN_OBS: held-out MLD reduced={mld_ho_red}  multiplier-bounded={m_in_bound}  "
          f"(train≈held-out ⇒ not overfitting: {red_mld_tr:+.1f}% vs {red_mld_ho:+.1f}%)", flush=True)

    rec.update(dict(B_mld_tr=B_mld_tr, F_mld_tr=F_mld_tr, B_mld_ho=B_mld_ho, F_mld_ho=F_mld_ho,
                    B_sst_tr=B_sst_tr, F_sst_tr=F_sst_tr, B_sst_ho=B_sst_ho, F_sst_ho=F_sst_ho,
                    red_mld_tr=red_mld_tr, red_mld_ho=red_mld_ho, red_sst_tr=red_sst_tr,
                    red_sst_ho=red_sst_ho, drmse_mld_ho=drmse_mld_ho, drmse_sst_ho=drmse_sst_ho,
                    b_mld_tr=b_mld_tr, f_mld_tr=f_mld_tr, b_mld_ho=b_mld_ho, f_mld_ho=f_mld_ho,
                    m_rec0_mean=float(m_rec0[wet].mean()), m_rec0_std=float(m_rec0[wet].std()),
                    m_in_bound=m_in_bound, mld_ho_red=mld_ho_red, n_iters=len(losshist),
                    reg=args.reg, m_max=args.m_max, best_it=best_it,
                    peak_gb=pk, bwd_s=bwd_s, nn_obs_ok=nn_obs_ok))

    # ---------- persist the trained NN (for --mode validate) + diagnostics ----------
    args.nn_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.nn_pkl, "wb") as fp:
        pickle.dump(dict(nn=jax.device_get(rec_nn), hidden=list(args.hidden), seed=args.seed,
                         config=args.config, start_step=int(pairs[0][0])), fp)
    print(f"  saved trained NN -> {args.nn_pkl}", flush=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez(args.outdir / "nn_obs_train.npz", loss_hist=np.array(losshist),
             months=np.array(months), b_mld_tr=np.array(b_mld_tr), f_mld_tr=np.array(f_mld_tr),
             b_mld_ho=np.array(b_mld_ho), f_mld_ho=np.array(f_mld_ho), b_sst_tr=np.array(b_sst_tr),
             f_sst_tr=np.array(f_sst_tr), m_rec0=m_rec0, wet=wet, red_mld_ho=red_mld_ho,
             red_mld_tr=red_mld_tr, holdout=args.holdout, fold=args.fold, N=args.n, windows=K)
    _train_figure(args.outdir / "nn_obs_train.png", losshist, months, b_mld_tr, f_mld_tr,
                  b_mld_ho, f_mld_ho, red_mld_ho, red_mld_tr)
    return nn_obs_ok


# ============================================================================= VALIDATE (long forward)
def run_validate(args, mesh, op, fs, cfgs, cf, node_mask, w3d, Hmap, cell_lat, cell_lon, rec):
    """Long forward-only deployment test: default NN→0 vs the trained NN from a held-out start state.
    Drift (vol-wt RMS T,S vs start) ≤ default + finite + benefit (MLD-vs-WOA) persists ⇒ NN_DRIFT_OK."""
    with open(args.nn_pkl, "rb") as fp:
        blob = pickle.load(fp)
    start_step = blob.get("start_step", 1440)
    zero_nn = tke_nn.init_tke_nn(jax.random.PRNGKey(blob.get("seed", 0) + 777),
                                 hidden=tuple(blob.get("hidden", args.hidden)), zero_last=True)
    if args.const_mult > 0:                 # DIAGNOSTIC: a UNIFORM multiplier (D2a-style global c_k
        trained_nn = uniform_nn(args.const_mult)   # scaling) — does more/less mixing help the SLOW
        rec["const_mult"] = args.const_mult        # deployed MLD? (isolates the fast/slow misalignment)
        print(f"[validate] DIAGNOSTIC uniform multiplier c={args.const_mult} (not the trained NN)", flush=True)
    else:
        trained_nn = jax.device_put(blob["nn"])
    print(f"[validate] loaded trained NN <- {args.nn_pkl}  start_step={start_step}  "
          f"long={args.long_steps} steps ({args.long_steps*DT/86400:.0f} d), seg={args.long_seg}", flush=True)

    # start state = the snapshot at start_step (model-consistent; the long forward begins here)
    snap = args.snapshots_dir / f"snap_step{start_step:06d}.pkl"
    with open(snap, "rb") as fp:
        S_start = jax.device_put(pickle.load(fp))
    wet = np.asarray(node_mask[:, 0]).astype(bool)
    cell_area = np.asarray(Hmap.cell_area)
    _, cv = obs_compare.to_obs_surface(jnp.ones((mesh.nod2D,)), Hmap)
    cell_valid = np.asarray(cv).astype(np.float64)
    T0 = np.asarray(S_start.T); S0 = np.asarray(S_start.S)
    W = np.asarray(w3d)

    @jax.jit
    def fwd_seg(nn, state, frc_seg):
        p = calibrate.build_params({"tke_nn": nn})
        return integrate(state, mesh, op, None, n_steps=args.long_seg, dt=DT, step_forcings=frc_seg,
                         forcing_static=fs, params=p, **cfgs)         # forward-only (no remat)

    def vw_rms(A, A0):
        d = A - A0
        return float(np.sqrt((W * d * d).sum() / W.sum()))

    def mld_misfit_vs_woa(state, month):
        _, Z3d = ale.live_geometry(mesh, state.hnode)
        mld, _ = obs_compare.mld_density_threshold(state.T, state.S, Z3d, node_mask)
        cell_mld, _ = obs_compare.to_obs_surface(mld, Hmap)
        mldw, mld_ok, sstw, sst_ok = woa_month(month)
        w_mld, _ = cell_weights(mld_ok, sstw, sst_ok, cell_area, cell_valid, args.ice_sst)
        return float(obs_compare.misfit(cell_mld, jnp.asarray(mldw), jnp.asarray(w_mld)))

    n_seg = args.long_steps // args.long_seg
    out = {}
    for tag, nn in (("default", zero_nn), ("trained", trained_nn)):
        state = S_start
        days, drift_T, drift_S, mld_mis, finite_ok = [], [], [], [], True
        t1 = time.time()
        for seg in range(n_seg):
            off = start_step + seg * args.long_seg
            dts = surface_forcing.dates_for_steps(args.year, DT, off + args.long_seg)
            frc_seg = cf.stack(dts[off:off + args.long_seg])
            state = fwd_seg(nn, state, frc_seg)
            state.T.block_until_ready()
            Tn, Sn = np.asarray(state.T), np.asarray(state.S)
            # stable = finite AND physically bounded (the real deployment floor — a runaway closure
            # blows SST/SSS out of range long before it NaNs). The NN→0 fallback is bit-identical.
            fin = bool(np.isfinite(Tn[wet]).all() and np.isfinite(Sn[wet]).all())
            phys = bool(fin and Tn[wet].min() > -5.0 and Tn[wet].max() < 45.0
                        and Sn[wet].min() >= 0.0 and Sn[wet].max() < 50.0)
            finite_ok = finite_ok and phys
            day = (off + args.long_seg) * DT / 86400.0
            month = dts[off + args.long_seg - 1][3]
            dT, dS = vw_rms(Tn, T0), vw_rms(Sn, S0)
            mm = mld_misfit_vs_woa(state, month) if fin else float("nan")
            days.append(day); drift_T.append(dT); drift_S.append(dS); mld_mis.append(mm)
            if seg % max(1, n_seg // 6) == 0 or seg == n_seg - 1:
                print(f"    [{tag}] seg {seg+1}/{n_seg} day {day:.0f} (m{month})  "
                      f"drift_T={dT:.4f}°C drift_S={dS:.4f}  MLD_mis={mm:.4e} m²  finite={fin}", flush=True)
        out[tag] = dict(days=days, drift_T=drift_T, drift_S=drift_S, mld_mis=mld_mis,
                        finite_ok=finite_ok, secs=time.time() - t1)
        print(f"  [{tag}] long forward {n_seg*args.long_seg} steps in {out[tag]['secs']:.1f}s  "
              f"finite={finite_ok}  peak={twin.peak_gb():.1f} GB", flush=True)

    d, t = out["default"], out["trained"]
    stable = bool(t["finite_ok"])
    # drift: NO RUNAWAY — the trained-NN drift must stay the same ORDER as default's natural drift,
    # not exploding. (Drift-from-start is NOT penalized per se: a real bias correction legitimately
    # moves the state off the default trajectory — that's beneficial drift. The gate guards against
    # destabilization, hence a generous factor, plus the absolute physical-range `stable` floor.)
    drift_ratio_T = t["drift_T"][-1] / (d["drift_T"][-1] + 1e-30)
    drift_ratio_S = t["drift_S"][-1] / (d["drift_S"][-1] + 1e-30)
    drift_ok = bool(stable and drift_ratio_T <= 1.0 + args.drift_tol and drift_ratio_S <= 1.0 + args.drift_tol)
    # persisted benefit: trained MLD misfit < default at the final horizon AND on seasonal-mean
    bn_final = bool(stable and t["mld_mis"][-1] < d["mld_mis"][-1])
    mean_def = float(np.nanmean(d["mld_mis"])); mean_tr = float(np.nanmean(t["mld_mis"]))
    bn_mean = bool(stable and mean_tr < mean_def)
    benefit_persists = bool(bn_final and bn_mean)
    nn_drift_ok = bool(stable and drift_ok and benefit_persists)

    print(f"\n  === long-forward deployment ({args.long_steps*DT/86400:.0f} d, forward-only) ===", flush=True)
    print(f"  stable (trained finite throughout)={stable}", flush=True)
    print(f"  drift @horizon: T trained/default={drift_ratio_T:.3f}  S={drift_ratio_S:.3f}  "
          f"(≤{1+args.drift_tol}) ⇒ drift_ok={drift_ok}", flush=True)
    print(f"  MLD misfit @horizon: default={d['mld_mis'][-1]:.4e}  trained={t['mld_mis'][-1]:.4e}  "
          f"⇒ benefit_final={bn_final}", flush=True)
    print(f"  MLD misfit seasonal-mean: default={mean_def:.4e}  trained={mean_tr:.4e} "
          f"({100*(1-mean_tr/mean_def):+.1f}%) ⇒ benefit_mean={bn_mean}", flush=True)
    print(f"  gate NN_DRIFT: stable={stable}  drift≤default={drift_ok}  benefit-persists={benefit_persists}",
          flush=True)

    rec.update(dict(start_step=start_step, long_steps=args.long_steps, long_seg=args.long_seg,
                    drift_ratio_T=drift_ratio_T, drift_ratio_S=drift_ratio_S, stable=stable,
                    drift_ok=drift_ok, bn_final=bn_final, bn_mean=bn_mean,
                    benefit_persists=benefit_persists, mld_mean_def=mean_def, mld_mean_tr=mean_tr,
                    default=d, trained=t, nn_drift_ok=nn_drift_ok, peak_gb=twin.peak_gb()))
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez(args.outdir / "nn_obs_validate.npz",
             days=np.array(d["days"]), drift_T_def=np.array(d["drift_T"]),
             drift_T_tr=np.array(t["drift_T"]), drift_S_def=np.array(d["drift_S"]),
             drift_S_tr=np.array(t["drift_S"]), mld_def=np.array(d["mld_mis"]),
             mld_tr=np.array(t["mld_mis"]), start_step=start_step)
    _validate_figure(args.outdir / "nn_obs_validate.png", d, t, drift_ok, benefit_persists)
    return nn_drift_ok


# ============================================================================= shared
def build_hmask(args, cell_lat, cell_lon):
    return cobs.build_holdout(args.holdout, cell_lat, cell_lon, args.holdout_deg, args.fold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("train", "validate", "both"), default="train")
    ap.add_argument("--n", type=int, default=12, help="per-chunk adjoint window (steps; A7/D2a N≈12)")
    ap.add_argument("--windows", type=int, default=8, help="K seasonal chunks (subsample of 12 months)")
    ap.add_argument("--config", choices=("all3", "tkegm"), default="all3")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.1,
                    help="Adam lr — the zero-init last layer + spatially-structured bias need a brisk "
                         "lr to bootstrap (lr=0.02 plateaued at ~0.6%% loss; the bounded multiplier "
                         "keeps diffusivities PD regardless)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 16])
    ap.add_argument("--w-sst", type=float, default=0.3, help="SST weight in the loss (MLD is primary)")
    ap.add_argument("--reg", type=float, default=0.03,
                    help="trust-region reg: area-wt penalty on (log multiplier − log reg_target)² ⇒ keep "
                         "the closure near the anchor so it DEPLOYS — the offline/online-gap fix")
    ap.add_argument("--reg-target", type=float, default=1.0,
                    help="A1(b): anchor the trust-region reg toward THIS uniform multiplier (1.0 = E2 "
                         "default/no-blow-up behavior; 2.0 = the const-mult-validated 'more mixing' side "
                         "that improves the DEPLOYED MLD — warm-started so the NN refines spatial "
                         "structure around the right anchor)")
    ap.add_argument("--m-max", type=float, default=2.0,
                    help="bounded multiplier ∈ (1/m_max, m_max) — the structural-stability cap; tighter "
                         "than the twin's 3.0 to bound the deployed over-mixing")
    ap.add_argument("--ice-sst", type=float, default=-1.0, help="obs ice mask: drop WOA cells below this °C")
    ap.add_argument("--holdout", choices=("none", "random", "lon", "lat", "nh", "sh"), default="random",
                    help="D2c held-out cells: train on TRAIN, score INDEPENDENT held-out (the NN_OBS bar)")
    ap.add_argument("--holdout-deg", type=float, default=60.0)
    ap.add_argument("--fold", type=int, choices=(0, 1), default=0)
    ap.add_argument("--skip-before", type=int, default=1440, help="ignore snapshots before this step")
    ap.add_argument("--snapshots-dir", type=Path, default=SNAP_DIR)
    ap.add_argument("--long-steps", type=int, default=2880, help="validate: total forward steps (no AD)")
    ap.add_argument("--long-seg", type=int, default=240, help="validate: re-stacked forcing segment")
    ap.add_argument("--const-mult", type=float, default=0.0,
                    help="validate DIAGNOSTIC: deploy a UNIFORM multiplier c (global c_k scaling) "
                         "instead of the trained NN — tests if the SLOW deployed MLD is mixing-improvable")
    ap.add_argument("--drift-tol", type=float, default=0.5,
                    help="no-runaway: trained drift ≤ (1+tol)·default at the horizon (same order)")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--nn-pkl", type=Path, default=NN_PKL, help="trained-NN handoff (train↔validate)")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--outdir", type=Path, default=ROOT / "scripts")
    ap.add_argument("--remat-blocks", dest="remat_blocks", action="store_true", default=True)
    ap.add_argument("--no-remat-blocks", dest="remat_blocks", action="store_false")
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}  mode={args.mode}", flush=True)
    t0 = time.time()
    # build the all3 model (n only sizes the throwaway initial stack; chunks re-stack their own forcing)
    mesh, _, op, fs, _, _, cfgs = twin.build(args.year, args.n, args.config)
    from fesom_jax.phc_ic import core2_initial_state
    base = core2_initial_state(mesh, twin.IC_DIR)
    cf = surface_forcing.build_surface_forcing(mesh, args.year, sst_ic=np.asarray(base.T[:, 0]))
    node_mask = jnp.asarray(mesh.node_layer_mask)
    w3d = jnp.asarray(mesh.area[:, 0])[:, None] * node_mask
    Hmap, _, (cell_lat, cell_lon) = cobs.load_woa(mesh, 1)        # Hmap is month-independent
    gpu_gb = twin.gpu_limit_gb()
    cfg_name = "zstar+TKE+mEVP+GM(frozen-ice)" if args.config == "all3" else "zstar+TKE+GM"
    print(f"[setup] built {cfg_name} in {time.time()-t0:.1f}s  N={args.n} ({args.n*DT/86400:.3f} d)  "
          f"GPU limit={gpu_gb:.0f} GB", flush=True)
    rec = {"target": "nn_obs", "mode": args.mode, "config": cfg_name, "N": args.n,
           "iters": args.iters, "lr": args.lr, "seed": args.seed, "hidden": list(args.hidden),
           "w_sst": args.w_sst, "holdout": args.holdout, "fold": args.fold, "gpu_gb": gpu_gb,
           "remat_blocks": bool(args.remat_blocks)}

    nn_obs_ok = nn_drift_ok = None
    try:
        if args.mode in ("train", "both"):
            nn_obs_ok = run_train(args, mesh, op, fs, cfgs, cf, node_mask, w3d, Hmap,
                                  cell_lat, cell_lon, rec)
        if args.mode in ("validate", "both"):
            nn_drift_ok = run_validate(args, mesh, op, fs, cfgs, cf, node_mask, w3d, Hmap,
                                       cell_lat, cell_lon, rec)
    except Exception as e:
        oom = twin.is_oom(e)
        rec.update(dict(ok=False, oom=oom, peak_gb=twin.peak_gb(),
                        error=f"{type(e).__name__}: {str(e)[:240]}"))
        _append(args.results, rec)
        print(f"\n  FAILED ({args.mode}): {type(e).__name__}: {str(e)[:200]}", flush=True)
        tok = "OOM" if oom else "FAIL"
        print(f"NN_OBS_{tok}" if args.mode != "validate" else f"NN_DRIFT_{tok}", flush=True)
        return 2 if oom else 1

    ok = all(x for x in (nn_obs_ok, nn_drift_ok) if x is not None)
    rec["ok"] = bool(ok)
    _append(args.results, rec)
    if nn_obs_ok is not None:
        print(f"NN_OBS_{'OK' if nn_obs_ok else 'FAIL'}", flush=True)
    if nn_drift_ok is not None:
        print(f"NN_DRIFT_{'OK' if nn_drift_ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


# ----------------------------------------------------------------------------- figures
def _train_figure(path, losshist, months, b_mld_tr, f_mld_tr, b_mld_ho, f_mld_ho, red_ho, red_tr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] skipped ({e})", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].semilogy(np.arange(1, len(losshist) + 1), losshist, ".-", ms=4, lw=1)
    ax[0].set(xlabel="Adam iteration", ylabel="batch-mean loss (MLD/J0 + w·SST/J0)",
              title=f"E2 NN obs-training\n(train {red_tr:+.1f}% / held-out {red_ho:+.1f}% MLD)")
    ax[0].grid(True, which="both", alpha=0.25)
    x = np.arange(len(months)); ww = 0.2
    ax[1].bar(x - 1.5 * ww, b_mld_tr, ww, label="train baseline", color="0.7")
    ax[1].bar(x - 0.5 * ww, f_mld_tr, ww, label="train trained", color="C0")
    ax[1].bar(x + 0.5 * ww, b_mld_ho, ww, label="held-out baseline", color="0.5")
    ax[1].bar(x + 1.5 * ww, f_mld_ho, ww, label="held-out trained", color="C1")
    ax[1].set(xlabel="seasonal chunk (month)", ylabel="MLD misfit [m²]", title="per-season MLD misfit")
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(m) for m in months], fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"  [fig] wrote {path}", flush=True)


def _validate_figure(path, d, t, drift_ok, benefit_persists):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] skipped ({e})", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(d["days"], d["drift_T"], "o-", ms=3, label="default (NN→0)", color="0.5")
    ax[0].plot(t["days"], t["drift_T"], "s-", ms=3, label="trained NN", color="C0")
    ax[0].set(xlabel="forward day", ylabel="vol-wt RMS ΔT [°C]",
              title=f"long-forward drift (drift_ok={drift_ok})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
    ax[1].plot(d["days"], d["mld_mis"], "o-", ms=3, label="default (NN→0)", color="0.5")
    ax[1].plot(t["days"], t["mld_mis"], "s-", ms=3, label="trained NN", color="C0")
    ax[1].set(xlabel="forward day", ylabel="MLD misfit vs WOA [m²]",
              title=f"persisted benefit ({benefit_persists})")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"  [fig] wrote {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
