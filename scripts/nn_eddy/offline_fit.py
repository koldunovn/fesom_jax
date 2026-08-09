"""NN-GM child-plan M3 — Stage 0: fit ``gm_nn`` offline to the diffusivities the teacher's
eddy fluxes imply.

This produces an INITIALISATION, not a closure. The project's own recorded lesson (parent plan,
the E2 offline/online gap) is that offline skill does not predict online skill, which is why
Task 9 trains through the adjoint from the start. What Stage 0 buys is a starting point that is
already in the right region, so the online phase spends its updates on the part that matters.

What is fitted
--------------
Per accepted column, :mod:`kappa_hat` gives the diffusivities the teacher's fluxes imply, and
:mod:`features_from_product` gives the coefficient the default formula would have produced there
from the SAME derived fields. The network's job is the ratio between them, in log space:

    target = log(kappa_hat / kappa_formula),   prediction = log m = log_m_max * tanh(mlp(feats))

Log space is not a convenience. The network's output is bounded by construction and the bound is
symmetric in log — ``m in [1/3, 3]`` — so a squared error on ``log m`` weights a factor-of-two
error the same whether it is up or down, while a squared error on ``m`` would care four times
more about the upward half.

Three things the fit reports rather than hides
----------------------------------------------
* **Reachability.** A bounded multiplier cannot represent every target. Columns whose ratio is
  negative (the teacher's flux is UP-gradient there — real, and outside Arm A by construction)
  are dropped, and columns outside ``[1/3, 3]`` are CLIPPED to the bound with the clipped
  fraction reported. That fraction is the honest answer to "can this architecture express the
  teacher at all", and it feeds the parent's Arm-B decision.
* **The trust region.** The regulariser pulls ``log m`` toward 0, i.e. toward the default
  closure. Its weight is a real choice: too small and Stage 0 hands the online phase a network
  far from anything validated; too large and Stage 0 does nothing. It is reported alongside the
  fit so the two are never separated.
* **What was held out, named before fitting.** ``--holdout`` is printed and written into the
  output BEFORE the first gradient step, and the gate is read on those months only.

Not decided here: the architecture screen (that is ``apriori_eval.py``, restricted to the
seam-deployable rungs D0 and D2 — D1's smoothed features need a mesh-Laplacian filter running
INSIDE the differentiated step, which M1 does not build).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fesom_jax  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_got = Path(fesom_jax.__file__).resolve()
if ROOT not in _got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {_got}, not from {ROOT}")

from fesom_jax import gm_nn  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features_from_product as _fp  # noqa: E402
import kappa_hat as _kh  # noqa: E402
import nn_data as _nd  # noqa: E402

# Jan-Mar 1958 are excluded by default: the teacher is spinning its eddy field up from a cold
# start there, so the flux AMPLITUDES are not yet representative even where the pattern is. The
# amplitude-trend check below (fitted m against teacher month) is what validates or moves that
# cut — it is a decision to be measured, not assumed.
SPINUP_EXCLUDE = ("1958_01", "1958_02", "1958_03")


def month_list(coarse_dir: Path) -> list:
    return sorted(p.name[len("coarse_ng5_"):-3]
                  for p in Path(coarse_dir).glob("coarse_ng5_[0-9]*.nc"))


def samples_from_mark(mesh, path: Path, record: int, *, rng, per_mark: int,
                      guard: dict, verbose=False):
    """``(feats, target, base_gm, base_redi, depth, lat)`` for one teacher mark.

    Returns the ACCEPTED cells only, subsampled to at most ``per_mark``. Two targets come back
    stacked as ``target[:, 0] = log(kappa_hat_GM / fer_K)`` and ``[:, 1]`` the Redi one; a cell
    is kept only where BOTH survive their guards, because the network emits both multipliers
    from one feature vector and a per-output mask would make the two heads see different data."""
    T47, S47, have47, tsec = _fp.load_product_mark(path, record)
    d = _fp.derive(mesh, T47, S47, have47, verbose=verbose)
    F_T, F_S = _kh.load_fluxes(path, record, mesh.nl)
    kh = _kh.project(mesh, d, F_T, F_S, verbose=verbose, **guard)

    nl = mesh.nl
    base_gm = np.asarray(d.fer_K_lay)
    base_redi = np.asarray(d.Ki)[:, :nl - 1]
    ok = (np.asarray(kh.ok_gm, bool) & np.asarray(kh.ok_redi, bool)
          & (base_gm > 0.0) & (base_redi > 0.0))
    r_gm = np.where(ok, np.asarray(kh.kappa_gm) / np.maximum(base_gm, 1e-300), np.nan)
    r_re = np.where(ok, np.asarray(kh.kappa_redi) / np.maximum(base_redi, 1e-300), np.nan)
    # UP-gradient columns are real physics and are outside Arm A by construction: a positive
    # bounded multiplier on a positive base cannot change the sign of the flux. Dropping them
    # is a decision, so it is counted.
    pos = ok & (r_gm > 0.0) & (r_re > 0.0)
    n_neg = int((ok & ~pos).sum())

    idx = np.flatnonzero(pos.ravel())
    if idx.size == 0:
        return None
    take = idx if idx.size <= per_mark else rng.choice(idx, per_mark, replace=False)
    feats = np.asarray(d.feats).reshape(-1, np.asarray(d.feats).shape[-1])[take]
    tgt = np.stack([np.log(r_gm.ravel()[take]), np.log(r_re.ravel()[take])], axis=-1)
    z = (np.abs(np.asarray(mesh.Z))[None, :nl - 1] * np.ones((mesh.nod2D, 1))).ravel()[take]
    lat = (_nd.geo_latitude_deg(mesh)[:, None]        # GEOGRAPHIC, see nn_data
           * np.ones((1, nl - 1))).ravel()[take]
    return dict(feats=feats, target=tgt, depth=z, lat=lat, mark=_nd.mark_index(tsec),
                n_accepted=int(pos.sum()), n_upgradient=n_neg,
                n_usable=int(np.asarray(d.usable, bool).sum()), reject=kh.reject)


def assemble(mesh, months, coarse_dir: Path, *, per_mark: int, seed: int, guard: dict,
             verbose=True):
    """Sample every mark of every month in ``months`` into one training set."""
    rng = np.random.default_rng(seed)
    parts, meta = [], []
    import netCDF4
    for m in months:
        p = Path(coarse_dir) / f"coarse_ng5_{m}.nc"
        with netCDF4.Dataset(p) as ds:
            nrec = len(ds.dimensions["time"])
        for r in range(nrec):
            t = time.time()
            s = samples_from_mark(mesh, p, r, rng=rng, per_mark=per_mark, guard=guard)
            if s is None:
                print(f"  {m}[{r}]: NOTHING ACCEPTED — skipped", flush=True)
                continue
            parts.append(s)
            meta.append(dict(month=m, record=r, mark=s["mark"], n_accepted=s["n_accepted"],
                             n_usable=s["n_usable"], n_upgradient=s["n_upgradient"],
                             kept=s["feats"].shape[0]))
            if verbose:
                print(f"  {m}[{r}] mark {s['mark']:>3d}: accepted {s['n_accepted']:>8d} of "
                      f"{s['n_usable']:>8d} usable ({100*s['n_accepted']/max(s['n_usable'],1):4.1f}%), "
                      f"up-gradient {s['n_upgradient']:>7d}, kept {s['feats'].shape[0]:>6d}"
                      f"  [{time.time()-t:.0f}s]", flush=True)
    if not parts:
        raise SystemExit("no samples at all — check the guard floors with kappa_hat.py --report")
    out = {k: np.concatenate([p[k] for p in parts], axis=0)
           for k in ("feats", "target", "depth", "lat")}
    out["meta"] = meta
    return out


def cache_key(train, hold, guard, per_mark, seed) -> str:
    """Identity of an assembled sample set. Anything that changes which cells are drawn, or
    which values they carry, has to be in here — a cache hit on a different guard would compare
    two architectures on two different oceans and call the difference architecture."""
    import hashlib
    payload = json.dumps(dict(train=list(train), hold=list(hold),
                              guard={k: float(v) for k, v in sorted(guard.items())},
                              per_mark=int(per_mark), seed=int(seed),
                              version=1), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16], payload


def load_cache(path: Path, key: str):
    f = path / f"assembled_{key}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    if str(z["key"]) != key:
        raise SystemExit(f"cache {f} carries key {z['key']!r}, expected {key!r}")
    def unpack(tag):
        return {k: z[f"{tag}_{k}"] for k in ("feats", "target", "depth", "lat")} | {
            "meta": json.loads(str(z[f"{tag}_meta"]))}
    return unpack("tr"), unpack("ho"), str(f)


def save_cache(path: Path, key: str, payload: str, d_tr, d_ho) -> str:
    path.mkdir(parents=True, exist_ok=True)
    f = path / f"assembled_{key}.npz"
    out = dict(key=key, spec=payload)
    for tag, d in (("tr", d_tr), ("ho", d_ho)):
        for k in ("feats", "target", "depth", "lat"):
            out[f"{tag}_{k}"] = d[k]
        out[f"{tag}_meta"] = json.dumps(d["meta"])
    np.savez_compressed(f, **out)
    return str(f)


def clip_targets(tgt, log_m_max: float):
    """Clip to what a bounded multiplier can express, and report how much was clipped.

    A target outside ``[1/m_max, m_max]`` is not reachable; leaving it in would pull every
    weight toward saturation to chase something the architecture cannot represent, and the
    saturated network then has almost no gradient left for the columns it CAN fit."""
    lo, hi = -log_m_max, log_m_max
    clipped = np.clip(tgt, lo, hi)
    frac = float(np.mean(np.any(np.abs(tgt) > log_m_max, axis=-1)))
    return clipped, frac


def fit(data, *, hidden, seed, epochs, batch, lr, reg, log_m_max, holdout=None, verbose=True):
    """Minibatch adam on ``log m`` against the clipped log-ratio targets."""
    import optax
    key = jax.random.PRNGKey(seed)
    nn = gm_nn.init_gm_nn(key, hidden=hidden, two_scale=False, m_max=float(np.exp(log_m_max)),
                          zero_last=True)
    # zero_last=True means the fit STARTS at m == 1 exactly, i.e. at the default closure. The
    # regulariser then measures departure from where it started, and a fit that learns nothing
    # returns the validated model rather than noise.
    X = jnp.asarray(data["feats"])
    Y = jnp.asarray(data["target"])
    Xh = jnp.asarray(holdout["feats"]) if holdout is not None else None
    Yh = jnp.asarray(holdout["target"]) if holdout is not None else None

    def loss_fn(nn_, x, y):
        pred = nn_.log_m_max * jnp.tanh(gm_nn.mlp_raw(nn_, x))       # = log m
        mis = jnp.mean((pred - y) ** 2)
        return mis + reg * jnp.mean(pred ** 2), mis

    opt = optax.adam(lr)
    state = opt.init(nn)

    @jax.jit
    def step(nn_, state_, x, y):
        (l, mis), g = jax.value_and_grad(loss_fn, has_aux=True)(nn_, x, y)
        upd, state_ = opt.update(g, state_, nn_)
        return optax.apply_updates(nn_, upd), state_, l, mis

    n = X.shape[0]
    hist = []
    rng = np.random.default_rng(seed + 1)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = cnt = 0.0
        for i in range(0, n - batch + 1, batch):
            j = jnp.asarray(perm[i:i + batch])
            nn, state, l, mis = step(nn, state, X[j], Y[j])
            tot += float(mis) * batch
            cnt += batch
        tr = tot / max(cnt, 1)
        ho = None
        if Xh is not None:
            _, ho = loss_fn(nn, Xh, Yh)
            ho = float(ho)
        hist.append(dict(epoch=ep, train_mse=tr, holdout_mse=ho))
        if verbose:
            print(f"  epoch {ep:>3d}  train MSE(log m) {tr:.5f}"
                  + (f"   holdout {ho:.5f}" if ho is not None else ""), flush=True)
    return nn, hist


def skill(nn, X, Y, *, log_m_max):
    """R² of ``log m`` against the target, plus the two baselines the gate compares to.

    ``formula`` is the default closure, i.e. ``log m == 0`` everywhere — the thing the seam
    reduces to with an untrained network. ``constant`` is the best single global multiplier,
    which is the cheapest way to look like you have learned something: a closure that only
    rescales κ globally would beat the formula and mean nothing, so the gate requires beating
    BOTH."""
    pred = np.asarray(nn.log_m_max * jnp.tanh(gm_nn.mlp_raw(nn, jnp.asarray(X))))
    y = np.asarray(Y)
    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
    r2 = lambda p: 1.0 - float(np.sum((y - p) ** 2)) / max(ss_tot, 1e-300)  # noqa: E731
    return dict(nn=r2(pred), formula=r2(np.zeros_like(y)),
                constant=r2(np.broadcast_to(y.mean(axis=0), y.shape)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coarse-dir", type=Path, default=_nd.COARSE_DIR)
    ap.add_argument("--months", default="", help="comma-separated; empty = every landed month")
    ap.add_argument("--holdout", default="",
                    help="comma-separated months held out. NAMED BEFORE FITTING and printed "
                         "before the first gradient step; empty = the last two landed months.")
    ap.add_argument("--exclude", default=",".join(SPINUP_EXCLUDE),
                    help="months excluded entirely (default: the teacher's eddy spin-up)")
    ap.add_argument("--per-mark", type=int, default=50_000,
                    help="accepted cells sampled per teacher mark (a mark holds ~5.9M)")
    ap.add_argument("--hidden", default="16,16")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--reg", type=float, default=0.01,
                    help="weight on mean((log m)^2) — the trust region around the default "
                         "closure. Reported with the fit; the two are not separable.")
    ap.add_argument("--m-max", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-floor", type=float, default=_kh.GRAD_FLOOR)
    ap.add_argument("--slope-floor", type=float, default=_kh.SLOPE_FLOOR)
    ap.add_argument("--resid-max", type=float, default=_kh.RESID_MAX)
    ap.add_argument("--b-sigma-max", type=float, default=_kh.B_SIGMA_MAX)
    ap.add_argument("--out", type=Path, default=_nd.data_path("stage0"))
    ap.add_argument("--cache", type=Path, default=_nd.data_path("stage0_cache"),
                    help="where assembled sample sets are kept. The assembly is ~70 min over "
                         "36 months and identical across architecture rungs, so the screen "
                         "reuses one draw rather than re-sampling per rung.")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    all_months = month_list(args.coarse_dir)
    excl = {m for m in args.exclude.split(",") if m.strip()}
    months = [m for m in (args.months.split(",") if args.months.strip() else all_months)
              if m.strip() and m not in excl]
    hold = ([m for m in args.holdout.split(",") if m.strip()] if args.holdout.strip()
            else months[-2:])
    train = [m for m in months if m not in hold]
    if not train or not hold:
        raise SystemExit(f"nothing to fit: train={train} holdout={hold}")

    # ⚠️ Printed and written BEFORE any fitting. A holdout chosen after seeing results is not
    # a holdout, and this is the plan's explicit requirement for GATE M3.
    print(f"[plan] landed months: {len(all_months)}; excluded (spin-up): {sorted(excl)}")
    print(f"[plan] TRAIN on {len(train)}: {train}")
    print(f"[plan] HOLD OUT {len(hold)}: {hold}   <- named before fitting", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "holdout_declared.json").write_text(json.dumps(
        dict(train=train, holdout=hold, excluded=sorted(excl), declared_at=time.strftime("%F %T"),
             argv=sys.argv[1:]), indent=1))

    guard = dict(grad_floor=args.grad_floor, slope_floor=args.slope_floor,
                 resid_max=args.resid_max, b_sigma_max=args.b_sigma_max)
    key, payload = cache_key(train, hold, guard, args.per_mark, args.seed)
    cached = None if args.no_cache else load_cache(args.cache, key)
    if cached is not None:
        d_tr, d_ho, cf = cached
        print(f"\n[data] REUSED the assembled sample set {cf}")
        print(f"[data] key {key} — same months, same guard, same per-mark draw, so an "
              f"architecture screen over this cache compares rungs on identical data")
    else:
        mesh = load_mesh(_fp.MESH_DIR)
        print(f"\n[data] assembling training marks ({args.per_mark} cells per mark)", flush=True)
        d_tr = assemble(mesh, train, args.coarse_dir, per_mark=args.per_mark, seed=args.seed,
                        guard=guard)
        print(f"\n[data] assembling HOLDOUT marks", flush=True)
        d_ho = assemble(mesh, hold, args.coarse_dir, per_mark=args.per_mark, seed=args.seed + 7,
                        guard=guard)
        if not args.no_cache:
            print(f"[data] cached as {save_cache(args.cache, key, payload, d_tr, d_ho)}",
                  flush=True)

    log_m_max = float(np.log(args.m_max))
    y_tr, clip_tr = clip_targets(d_tr["target"], log_m_max)
    y_ho, clip_ho = clip_targets(d_ho["target"], log_m_max)
    d_tr["target"], d_ho["target"] = y_tr, y_ho
    print(f"\n[targets] train {y_tr.shape[0]} samples, holdout {y_ho.shape[0]}")
    print(f"[targets] OUTSIDE the [1/{args.m_max:g}, {args.m_max:g}] multiplier bound and "
          f"therefore clipped: train {100*clip_tr:.1f}%, holdout {100*clip_ho:.1f}% "
          f"— this is the architecture's reachability, not a fit quality")
    for k, nm in ((0, "GM"), (1, "Redi")):
        v = y_tr[:, k]
        print(f"[targets] log m_{nm:4s}: median {np.median(v):+.3f}  "
              f"(m = {np.exp(np.median(v)):.3f})   p10 {np.percentile(v, 10):+.3f}  "
              f"p90 {np.percentile(v, 90):+.3f}", flush=True)

    # The best single global multiplier on the TRAINING targets. Least squares on log m makes
    # it the mean, and it is recorded here rather than recomputed at evaluation time because a
    # constant fitted on the data it is scored against is not a baseline — it is a free
    # parameter. apriori_eval.py reads this key.
    train_mean_log_m = [float(y_tr[:, 0].mean()), float(y_tr[:, 1].mean())]
    print(f"[baseline] best global constant from TRAIN: m_GM = "
          f"{np.exp(train_mean_log_m[0]):.4f}, m_Redi = {np.exp(train_mean_log_m[1]):.4f}")

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    print(f"\n[fit] D0, hidden={hidden}, reg={args.reg}, lr={args.lr}, batch={args.batch}",
          flush=True)
    nn, hist = fit(d_tr, hidden=hidden, seed=args.seed, epochs=args.epochs, batch=args.batch,
                   lr=args.lr, reg=args.reg, log_m_max=log_m_max, holdout=d_ho)

    s_tr = skill(nn, d_tr["feats"], d_tr["target"], log_m_max=log_m_max)
    s_ho = skill(nn, d_ho["feats"], d_ho["target"], log_m_max=log_m_max)
    print(f"\n[skill] R^2 of log m — the gate is read on the HOLDOUT row")
    for nm, s in (("train", s_tr), ("HOLDOUT", s_ho)):
        print(f"  {nm:8s}  NN {s['nn']:+.4f}   formula (m=1) {s['formula']:+.4f}   "
              f"best global constant {s['constant']:+.4f}")
    margin = min(s_ho["nn"] - s_ho["formula"], s_ho["nn"] - s_ho["constant"])
    print(f"  holdout margin over the WORSE baseline: {margin:+.4f} (gate wants >= 0.05 "
          f"globally, and not worse in any of 5 depth bands — depth bands are apriori_eval.py)")

    wpath = args.out / f"gm_nn_stage0_seed{args.seed}.npz"
    gm_nn.save_gm_nn(wpath, nn)
    rec = dict(train_months=train, holdout_months=hold, excluded=sorted(excl),
               hidden=list(hidden), reg=args.reg, lr=args.lr, batch=args.batch,
               epochs=args.epochs, seed=args.seed, m_max=args.m_max, guard=guard,
               per_mark=args.per_mark, n_train=int(y_tr.shape[0]), n_holdout=int(y_ho.shape[0]),
               clipped_frac_train=clip_tr, clipped_frac_holdout=clip_ho,
               train_mean_log_m=train_mean_log_m,
               skill_train=s_tr, skill_holdout=s_ho, holdout_margin=margin,
               history=hist, weights=str(wpath), marks_train=d_tr["meta"],
               marks_holdout=d_ho["meta"], seconds=time.time() - t0)
    (args.out / f"stage0_seed{args.seed}.json").write_text(json.dumps(rec, indent=1))
    print(f"\n[out] weights {wpath}")
    print(f"[out] record  {args.out / f'stage0_seed{args.seed}.json'}")
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    print("M3_OFFLINE_FIT_DONE", flush=True)


if __name__ == "__main__":
    main()
