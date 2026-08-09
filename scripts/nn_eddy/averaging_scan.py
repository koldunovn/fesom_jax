"""How much of the teacher's eddy flux divergence becomes PREDICTABLE as the target is averaged.

The decision this exists for. On a single five-day mark the measurements are unambiguous: the best
constant coefficient pair over the whole ocean explains **+0.0000**, a trained network reaches
**+0.0029**, and a per-cell oracle reaches **+0.5976**. The oracle's advantage is eddy phase — its
coefficient in the local slope frame is centred on zero with a median 4 % of its spread and the GM
sign winning in 54.6 % of cells — and no closure mapping local mean-state features to coefficients
can carry that.

A parameterization represents a statistically AVERAGED transport, so the honest question is not
"how much of one snapshot" but "how fast does the predictable part emerge as we average". That is
a curve, not an argument, and every ingredient for it is already cached.

**What is averaged, and why it has to be this and not the other thing.** The fields are averaged
FIRST and the coefficient is fitted to the average — one coefficient for the whole window. Fitting
each mark separately and averaging the coefficients is a different and much more permissive object:
it describes a closure allowed to change its answer every five days, which is not a
parameterization. Same distinction ``apriori_eval.evaluate_time_averaged`` draws.

The divergence is linear in the flux, so averaging the cached per-mark divergences is identical to
taking the divergence of the averaged flux, and no mesh operator is needed here.

**Three numbers per window length.**

* the best single global ``(a, b)`` — an exact 2x2 solve, the feature-free floor. If this stays at
  zero as the window grows, averaging is not producing a systematic part that a closure could use.
* the deployable per-cell oracle — the ceiling, which also moves with averaging, and the gap
  between the two is what any feature set is competing for.
* the oracle's coefficient distribution in the slope frame — median against robust spread, and the
  fraction carrying the GM sign. This is the direct readout of signal-to-noise, and it is the one
  that says WHY the other two move.

⚠️ **Five-day marks are not independent samples of the eddy field.** Mesoscale eddies decorrelate
over 10-30 days, so six marks within a month are worth on the order of one to two independent
draws, and the curve should be read as a function of the averaging window in days rather than of
the sample count. Whatever it shows at 30 days is a lower bound on what a seasonal or annual mean
would give, which is the target a deployed closure is actually judged on.
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
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apriori_eval as _ae  # noqa: E402
import armb_fit as _ab  # noqa: E402

CACHE = Path("/work/ba1583/a270088/nn_eddy/armb_cache")


def load_mark(path: Path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in _ab._CACHED}


_SUMMED = ("F_T", "F_S", "dvT", "dvS", "dzT", "dzS", "scale_iface", "feats_iface")


class Accum:
    """Streaming equal-weight mean over marks.

    ⚠️ Written because the obvious shortcut is wrong: folding each new member in with
    ``average([acc, new])`` halves the weight of everything already accumulated, so after three
    members the weights are 1/4, 1/4, 1/2 rather than equal. A climatology built that way is
    dominated by whichever month happened to come last."""

    def __init__(self):
        self.n = 0
        self.sums = None
        self.shat = None
        self.usable = None
        self.wet = None
        self.free = None

    def add(self, m):
        f = {k: m[k].astype(np.float64) for k in _SUMMED}
        s = m["shat_iface"].astype(np.float64)
        if self.sums is None:
            self.sums = f
            self.shat = s
            self.usable = m["usable"].copy()
            self.wet, self.free = m["wet"], m["free"]
        else:
            for k in _SUMMED:
                self.sums[k] += f[k]
            self.shat += s
            self.usable &= m["usable"]
        self.n += 1
        return self

    def finish(self):
        out = {k: v / self.n for k, v in self.sums.items()}
        nrm = np.linalg.norm(self.shat, axis=-1, keepdims=True)
        out["shat_iface"] = np.where(nrm > 0, self.shat / np.where(nrm > 0, nrm, 1.0), 0.0)
        out["usable"] = self.usable
        out["wet"], out["free"] = self.wet, self.free
        out["flux_w"] = np.float32(_ae.flux_weight(out["F_T"], out["F_S"]))
        u = self.usable
        rt = float(np.sqrt(np.mean(out["dvT"][u] ** 2)))
        rs = float(np.sqrt(np.mean(out["dvS"][u] ** 2)))
        out["div_w"] = np.float32(rt / max(rs, 1e-300))
        return out


def average(marks):
    """Equal-weight mean of a short window. One implementation, shared with the climatology
    path, so a window of 6 and a window of 120 cannot mean two different things."""
    acc = Accum()
    for m in marks:
        acc.add(m)
    return acc.finish()


def ev_about_zero(target, pred, mask):
    sse = float((((target - pred) ** 2)[mask]).sum())
    tot = float(((target ** 2)[mask]).sum())
    return 1.0 - sse / max(tot, 1e-300)


def measure(batch, n_wet):
    """The three readouts for one averaged window."""
    import jax.numpy as jnp
    dev = {k: jnp.asarray(v) for k, v in batch.items()
           if k not in ("flux_w", "div_w")}
    dev["flux_w"] = float(batch["flux_w"])
    dev["div_w"] = float(batch["div_w"])
    m = np.asarray(batch["usable"])

    # (1) the feature-free floor: best single (a, b), exact
    g = _ab.global_ab(_MESH, [dev], _HNODE, _HELEM)

    # (2) the ceiling: deployable per-cell oracle, fitted on the averaged flux
    pT, pS = _ae.oracle_psi_constrained(batch["dzT"], batch["dzS"], batch["F_T"], batch["F_S"],
                                        float(batch["flux_w"]), n_wet)
    qT, qS = _ab.div_pair(_MESH, jnp.asarray(pT), jnp.asarray(pS), _HNODE, _HELEM, _MESH.nl)
    ceil_T = ev_about_zero(batch["dvT"], np.asarray(qT), m)

    # (3) why: the oracle's coefficient in the slope frame, non-dimensionalised
    psi = _ae.oracle_psi_vec(batch["dzT"], batch["dzS"], batch["F_T"], batch["F_S"],
                             float(batch["flux_w"]))
    sc = batch["scale_iface"]
    sx, sy = batch["shat_iface"][..., 0], batch["shat_iface"][..., 1]
    sel = m & (sc > 0) & (np.abs(sx) + np.abs(sy) > 0)
    safe = np.where(sc > 0, sc, 1.0)
    a = ((psi[..., 0] * (-sy) + psi[..., 1] * sx) / safe)[sel]
    b = ((psi[..., 0] * sx + psi[..., 1] * sy) / safe)[sel]
    qa = np.percentile(a, [25, 50, 75])
    sd_a = (qa[2] - qa[0]) / 1.349
    return dict(const_a=g["a"], const_b=g["b"], const_ev=g["ev_joint"],
                ceiling_ev_T=ceil_T,
                oracle_a_median=float(qa[1]), oracle_a_robust_sd=float(sd_a),
                oracle_a_snr=float(qa[1] / sd_a) if sd_a > 0 else float("nan"),
                frac_gm_sign=float(np.mean(a > 0)),
                n_cells=int(sel.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--months", default="", help="comma-separated; empty = every month cached "
                                                 "with all six marks")
    ap.add_argument("--windows", default="1,2,3,6",
                    help="how many consecutive 5-day marks to average per window")
    ap.add_argument("--climatology", action="store_true",
                    help="also average EVERY mark of EVERY month into one window — the limit the "
                         "curve is heading towards, and the target a deployed closure is judged on")
    ap.add_argument("--out", type=Path,
                    default=Path("/work/ba1583/a270088/nn_eddy/averaging_scan.json"))
    args = ap.parse_args()

    from fesom_jax import ale
    from fesom_jax.mesh import load_mesh
    from fesom_jax.state import State
    import features_from_product as _fp

    global _MESH, _HNODE, _HELEM
    _MESH = load_mesh(_fp.MESH_DIR)
    _HNODE = State.rest(_MESH).hnode
    _, _HELEM = ale.commit_thickness(_MESH, _HNODE)
    n_wet = _ae.wet_layer_count(_MESH)

    files = sorted(args.cache.glob("armb_mark_v4_*.npz"))
    by_month = {}
    for f in files:
        stem = f.stem[len("armb_mark_v4_"):]
        month, rec = stem.rsplit("_r", 1)
        by_month.setdefault(month, {})[int(rec)] = f
    # ⚠️ Record indices are NOT 0..5. The assembler picked `marks-per-month` evenly spaced marks
    # from whatever the month's file held, so a month with SEVEN records cached 0,1,2,4,5,6 —
    # 1959_03 does exactly that. Work off the indices actually present, never off a range.
    full = {m: sorted(d) for m, d in by_month.items() if len(d) >= 2}
    if args.months.strip():
        want = {x for x in args.months.split(",") if x.strip()}
        full = {m: v for m, v in full.items() if m in want}
    months = sorted(full)
    counts = sorted({len(v) for v in full.values()})
    print(f"[plan] {len(months)} months, {counts} marks each: {months[0]}..{months[-1]}",
          flush=True)

    rows = []
    for w in [int(x) for x in args.windows.split(",")]:
        t0 = time.time()
        per = []
        for m in months:
            recs = full[m]
            for lo in range(0, len(recs) - w + 1, w):   # non-overlapping windows of what exists
                marks = [load_mark(by_month[m][r]) for r in recs[lo:lo + w]]
                per.append(measure(average(marks), n_wet))
                del marks
        agg = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k != "n_cells"}
        agg |= dict(window_marks=w, window_days=5 * w, n_windows=len(per))
        rows.append(agg)
        print(f"  {w:>2d} marks ({5*w:>2d} d), {len(per):>3d} windows: "
              f"const EV {agg['const_ev']:+.4f}   ceiling(T) {agg['ceiling_ev_T']:+.4f}   "
              f"oracle a: median {agg['oracle_a_median']:+.4f} / sd {agg['oracle_a_robust_sd']:.4f} "
              f"= {agg['oracle_a_snr']:+.4f}   GM sign {agg['frac_gm_sign']:.4f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    if args.climatology:
        t0 = time.time()
        acc = Accum()
        n_added = 0
        for m in months:
            for rec in full[m]:
                acc.add(load_mark(by_month[m][rec]))
                n_added += 1
        r = measure(acc.finish(), n_wet)
        r |= dict(window_marks=n_added, window_days=5 * n_added,
                  n_windows=1, climatology=True)
        rows.append(r)
        print(f"  CLIMATOLOGY ({n_added} marks over {len(months)} months): const EV {r['const_ev']:+.4f}   "
              f"ceiling(T) {r['ceiling_ev_T']:+.4f}   "
              f"oracle a: median {r['oracle_a_median']:+.4f} / sd {r['oracle_a_robust_sd']:.4f} "
              f"= {r['oracle_a_snr']:+.4f}   GM sign {r['frac_gm_sign']:.4f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(months=months, rows=rows), indent=2))
    print(f"[done] wrote {args.out}", flush=True)
    print("M3_AVERAGING_SCAN_DONE", flush=True)


if __name__ == "__main__":
    main()
