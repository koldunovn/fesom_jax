"""Control-transform A/B for the TKE-constant calibration: raw vs linear vs log controls.

The calibration (``tke_lbfgs.py``) optimises log controls u, p = p0·exp(σu) (``fesom_jax.controls``).
This driver asks whether that choice matters, by optimising the SAME cost through a host-side change
of variables v ↦ p and nothing else:

    log      p = p0·exp(σv)      dp/dv = σp     (v is u: the calibration's own control)
    linear   p = p0·(1 + σv)     dp/dv = σp0    (domain v > −1/σ)
    raw      p = v               dp/dv = 1      (domain v > 0)

The model and its adjoint always run in u = ln(p/p0)/σ, so every arm evaluates the same compiled
program; the gradient follows from the chain rule on the host,

    ∂J/∂v = (dp/dv)·∂J/∂p = (dp/dv)/(σp) · ∂J/∂u,

which is the elementwise "adjoint of the scaling". A transform cannot change the adjoint itself
(it is a fixed diagonal change of variables), only the geometry the optimiser sees.

Contract of :func:`reparam` (what the tests pin):

* log returns the base function's result object unchanged (no arithmetic), so a log run is bitwise
  the plain driver's run;
* linear and raw form the ratio (dp/dv)/(σp) before multiplying, so linear at v = 0 (p = p0, u = 0
  exactly, ratio exactly 1.0) returns the base gradient bitwise;
* a point with any p ≤ 0 is rejected WITHOUT running the model: ``(nan, nan-vector, aux)``, which the
  L-BFGS treats as a failed Armijo test (it backtracks);
* ``max_model_evals`` limits MODEL runs (rejections are free) by raising :class:`ModelBudget`;
* every value added to aux is plain JSON (``optim_lbfgs.minimize`` writes aux with ``json.dumps``).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from fesom_jax import optim_lbfgs
from fesom_jax.controls import ControlSet, tke_control_set  # noqa: F401  (re-exported for tests)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tke_lbfgs as T  # noqa: E402

KINDS = ("log", "linear", "raw")


class ModelBudget(RuntimeError):
    """Raised instead of starting a model evaluation beyond ``max_model_evals``."""


@dataclasses.dataclass(frozen=True)
class Transform:
    """The optimiser variable v of one arm, mapped to the physical constants p."""

    kind: str
    p0: tuple
    sigma: tuple

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        p0 = tuple(float(x) for x in self.p0)
        sigma = tuple(float(x) for x in self.sigma)
        if len(p0) != len(sigma) or not p0:
            raise ValueError("p0 and sigma need the same, non-zero length")
        if not all(math.isfinite(x) and x > 0.0 for x in p0 + sigma):
            raise ValueError("p0 and sigma must be finite and > 0")
        object.__setattr__(self, "p0", p0)
        object.__setattr__(self, "sigma", sigma)

    @property
    def _p0(self) -> np.ndarray:
        return np.array(self.p0, np.float64)

    @property
    def _sig(self) -> np.ndarray:
        return np.array(self.sigma, np.float64)

    def to_p(self, v) -> np.ndarray:
        v = np.asarray(v, np.float64)
        if self.kind == "log":
            return self._p0 * np.exp(self._sig * v)
        if self.kind == "linear":
            return self._p0 * (1.0 + self._sig * v)
        return v.copy()

    def to_v(self, p) -> np.ndarray:
        p = np.asarray(p, np.float64)
        if not (np.all(np.isfinite(p)) and np.all(p > 0.0)):
            raise ValueError(f"physical values must be finite and > 0, got {p.tolist()}")
        if self.kind == "log":
            return np.log(p / self._p0) / self._sig
        if self.kind == "linear":
            return (p / self._p0 - 1.0) / self._sig
        return p.copy()

    def dp_dv(self, v) -> np.ndarray:
        v = np.asarray(v, np.float64)
        if self.kind == "log":
            return self._sig * self.to_p(v)
        if self.kind == "linear":
            return self._sig * self._p0
        return np.ones_like(v)

    def ratio(self, p) -> np.ndarray:
        """(dp/dv)/(σp) at the physical point p: the factor from ∂J/∂u to ∂J/∂v."""
        p = np.asarray(p, np.float64)
        if self.kind == "log":
            return np.ones_like(p)
        if self.kind == "linear":
            return self._p0 / p                  # σ cancels: exactly 1.0 at p = p0
        return 1.0 / (self._sig * p)


def transform_for(cs: ControlSet, kind: str) -> Transform:
    """The ``kind`` transform with the calibration controls' own first guess and σ."""
    return Transform(kind, tuple(cs.p0.tolist()), tuple(cs.sigma.tolist()))


def reparam(base_fg: Callable, tr: Transform, *, max_model_evals: int | None = None,
            n_done: int = 0) -> Callable:
    """``fun_and_grad`` in the arm's variable v from the calibration's ``base_fg(u) -> (J, ∇_u J, aux)``
    (see the module docstring for the contract). ``n_done``: model evaluations already spent
    (a resumed run's log), counted against ``max_model_evals``."""
    p0, sig = tr._p0, tr._sig
    n_model = [int(n_done)]

    def run_base(u):
        if max_model_evals is not None and n_model[0] >= max_model_evals:
            raise ModelBudget(f"model-evaluation budget {max_model_evals} used up")
        n_model[0] += 1
        return base_fg(u)

    if tr.kind == "log":
        return run_base

    def fg(v):
        v = np.asarray(v, np.float64)
        p = tr.to_p(v)
        if not (np.all(np.isfinite(p)) and np.all(p > 0.0)):
            return (math.nan, np.full(v.shape, math.nan),
                    {"out_of_domain": True, "transform": tr.kind,
                     "p": [float(x) for x in p], "v": [float(x) for x in v],
                     "n_model": int(n_model[0])})
        u = np.log(p / p0) / sig
        f, g_u, aux = run_base(u)
        g_v = tr.ratio(p) * np.asarray(g_u, np.float64)
        aux = dict(aux)
        aux.update({"out_of_domain": False, "transform": tr.kind,
                    "v": [float(x) for x in v], "u": [float(x) for x in u],
                    "n_model": int(n_model[0])})
        return f, g_v, aux

    return fg


# ---------------------------------------------------------------------------
# S0: conditioning of a Gauss-Newton Hessian under each transform


def transform_hessians(H_u, p, p0, sigma) -> dict:
    """The Hessian ``H_u`` (log-u coordinates, at the physical point p) in every arm's coordinates:
    raw H_p = H_u ⊘ (σp)(σp)ᵀ, linear H_p ⊙ (σp0)(σp0)ᵀ, log H_u. Gauss-Newton only: the log arm's
    extra diagonal term σ²p·∂J/∂p vanishes at a twin truth and is left out everywhere."""
    H_u = np.asarray(H_u, np.float64)
    sp = np.asarray(sigma, np.float64) * np.asarray(p, np.float64)
    sp0 = np.asarray(sigma, np.float64) * np.asarray(p0, np.float64)
    H_p = H_u / np.outer(sp, sp)
    return {"raw": H_p, "linear": H_p * np.outer(sp0, sp0), "log": H_u}


def kappa(H, prior: float = 0.0) -> float:
    """Condition number λ_max/λ_min of the symmetric ``H + prior·I``."""
    w = np.linalg.eigvalsh(np.asarray(H, np.float64) + prior * np.eye(len(H)))
    return float(w[-1] / w[0])


def cond_rows_from_records(records) -> list:
    """One row per ``ident.jsonl`` record: κ per transform without a prior, and with the unit prior
    for log and linear only (the two unit priors are DIFFERENT priors — relative errors σ in log,
    absolute σ·p0 in linear; raw units have no meaningful unit prior)."""
    rows = []
    for r in records:
        p, p0, sig = r["p"], r["calib"]["p0"], r["calib"]["sigma"]
        hs = transform_hessians(r["H_data"], p, p0, sig)
        rows.append({"n_steps": r["n_steps"], "point": r["point"], "p": list(map(float, p)),
                     "kappa": {k: kappa(H) for k, H in hs.items()},
                     "kappa_unit_prior": {k: kappa(hs[k], 1.0) for k in ("linear", "log")},
                     "eigvals": {k: np.linalg.eigvalsh(H).tolist() for k, H in hs.items()},
                     "diag": {k: np.diag(H).tolist() for k, H in hs.items()}})
    return rows


def cond_rows(path) -> list:
    with open(path) as fh:
        return cond_rows_from_records([json.loads(line) for line in fh if line.strip()])


def _write_new(path: Path, text: str) -> None:
    """Write a NEW file; refuse to overwrite one that exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as fh:
        fh.write(text)


def run_cond(idents, out_dir) -> list:
    out_dir = Path(out_dir)
    rows = []
    for f in idents:
        for r in cond_rows(f):
            rows.append({"file": str(f), **r})
    lines = ["| window (steps) | point | κ raw | κ linear | κ log | κ linear + unit prior | "
             "κ log + unit prior |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        k, kp = r["kappa"], r["kappa_unit_prior"]
        lines.append(f"| {r['n_steps']} | {r['point']} | {k['raw']:.4g} | {k['linear']:.4g} | "
                     f"{k['log']:.4g} | {kp['linear']:.4g} | {kp['log']:.4g} |")
    lines.append("")
    lines.append("Gauss-Newton data Hessians from track F's ident runs, transformed exactly; the two "
                 "unit priors are different priors (relative σ in log, absolute σ·p0 in linear).")
    _write_new(out_dir / "cond.json", json.dumps(rows, indent=1) + "\n")
    _write_new(out_dir / "cond.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return rows


# ---------------------------------------------------------------------------
# S1: optimiser A/B on a surrogate cost with the measured Hessian

#: 1 % of a constant in log-u at σ = ln 2 (track F's "units of 1 %", 0.0144 in u)
U_PCT = math.log(1.01) / math.log(2.0)
STARTS = {"p0": np.ones(4), "plus": np.array([2 ** 0.5, 2 ** -0.5, 2 ** 0.5, 2 ** -0.5]),
          "minus": np.array([2 ** -0.5, 2 ** 0.5, 2 ** -0.5, 2 ** 0.5])}
ADAM_LRS = {"raw": (1e-3, 1e-2, 1e-1, 1.0), "linear": (0.03, 0.1, 0.3), "log": (0.03, 0.1, 0.3)}


def evals_to(f, thr: float):
    """First evaluation (1-based) with f/f[0] ≤ thr, or None."""
    f0 = float(f[0])
    for k, x in enumerate(f):
        if math.isfinite(x) and x / f0 <= thr:
            return k + 1
    return None


def evals_to_error(errs, tol: float = 1.0):
    """First evaluation (1-based) at which every identifiable-direction error is ≤ tol, or None."""
    for k, e in enumerate(errs):
        if np.all(np.asarray(e, np.float64) <= tol):
            return k + 1
    return None


class Surrogate:
    """J(p) = ½(p − p*)ᵀH_p(p − p*): the measured Gauss-Newton curvature as a cost in physical units.
    Quadratic in p, hence fair between raw and linear (an affine pair) and NOT between linear and
    log. No domain rule: S1 asks about units, not positivity. ``V_ident``: columns = identifiable
    eigenvectors of H in log-u (track F's λ ≥ 10)."""

    def __init__(self, H_p, p_star, p0, sigma, V_ident):
        self.H = np.asarray(H_p, np.float64)
        self.p_star = np.asarray(p_star, np.float64)
        self.p0 = np.asarray(p0, np.float64)
        self.sig = np.asarray(sigma, np.float64)
        self.V = np.asarray(V_ident, np.float64)
        self.u_star = np.log(self.p_star / self.p0) / self.sig

    def J(self, p):
        d = np.asarray(p, np.float64) - self.p_star
        Hd = self.H @ d
        return 0.5 * float(d @ Hd), Hd

    def fg(self, tr: Transform) -> Callable:
        def fg(v):
            v = np.asarray(v, np.float64)
            p = tr.to_p(v)
            f, gp = self.J(p)
            return f, tr.dp_dv(v) * gp, {"p": [float(x) for x in p]}
        return fg

    def errors(self, p) -> np.ndarray:
        p = np.asarray(p, np.float64)
        if not np.all(p > 0.0):
            return np.full(self.V.shape[1], math.inf)
        u = np.log(p / self.p0) / self.sig
        return np.abs(self.V.T @ (u - self.u_star)) / U_PCT


def run_surrogate_arm(sur: Surrogate, tr: Transform, p_start, optimizer: str, out_dir, *,
                      maxeval: int = 30, lr: float | None = None) -> dict:
    """One optimiser run of one arm from a physical start; log in ``out_dir/iterations.jsonl``
    (refuses an existing one)."""
    log = Path(out_dir) / "iterations.jsonl"
    if log.exists():
        raise FileExistsError(f"{log} exists")
    fs, ps = [], []
    base = sur.fg(tr)

    def fg(v):
        out = base(v)
        fs.append(float(out[0]))
        ps.append(out[2]["p"])
        return out

    v0 = tr.to_v(np.asarray(p_start, np.float64))
    if optimizer == "lbfgs":
        optim_lbfgs.minimize(fg, v0, maxiter=maxeval, maxeval=maxeval, log_path=log, resume=False,
                             options=optim_lbfgs.LBFGSOptions(expand=T.LBFGS_EXPAND))
    elif optimizer == "adam":
        T.adam_minimize(fg, v0, n_evals=maxeval, lr=float(lr), log_path=log)
    else:
        raise ValueError(f"optimizer must be lbfgs or adam, got {optimizer!r}")
    errs = [sur.errors(p).tolist() for p in ps]
    return {"transform": tr.kind, "optimizer": optimizer, "lr": lr, "f": fs, "p": ps, "err": errs,
            "evals_to_1e-2": evals_to(fs, 1e-2), "evals_to_1e-3": evals_to(fs, 1e-3),
            "evals_to_1pct": evals_to_error(errs), "f_ratio_min": min(fs) / fs[0]}


def _truth_row(path, n_steps=24):
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("point") == "truth" and r.get("n_steps") == n_steps:
                return r
    raise ValueError(f"{path}: no truth row with n_steps={n_steps}")


def run_surrogate(ident, out_dir, *, maxeval: int = 30) -> dict:
    out_dir = Path(out_dir)
    r = _truth_row(ident)
    p0, sig = np.array(r["calib"]["p0"], np.float64), np.array(r["calib"]["sigma"], np.float64)
    p_star = np.array(r["p"], np.float64)
    H_u = np.array(r["H_data"], np.float64)
    w, V = np.linalg.eigh(H_u)
    sur = Surrogate(transform_hessians(H_u, p_star, p0, sig)["raw"], p_star, p0, sig,
                    V[:, w >= 10.0])
    runs = []
    for kind in KINDS:
        tr = Transform(kind, tuple(p0), tuple(sig))
        for sname, fac in STARTS.items():
            runs.append({"start": sname, **run_surrogate_arm(
                sur, tr, p0 * fac, "lbfgs", out_dir / f"lbfgs_{kind}_{sname}", maxeval=maxeval)})
        probe = {}
        for lr in ADAM_LRS[kind]:
            res = run_surrogate_arm(sur, tr, p0, "adam", out_dir / f"adam_{kind}_p0_lr{lr:g}",
                                    maxeval=maxeval, lr=lr)
            probe[str(lr)] = {"lr": lr, "f": res["f"], "run": res}
        lr_best = T.choose_lr(probe)
        runs.append({"start": "p0", **probe[str(lr_best)]["run"]})
        for sname in ("plus", "minus"):
            runs.append({"start": sname, **run_surrogate_arm(
                sur, tr, p0 * STARTS[sname], "adam", out_dir / f"adam_{kind}_{sname}_lr{lr_best:g}",
                maxeval=maxeval, lr=lr_best)})
    verdict = surrogate_verdict(runs)
    lines = ["| optimiser | transform | start | lr | evals to J/J0 ≤ 1e-2 | ≤ 1e-3 | ≤ 1 % error | "
             "min J/J0 |", "|---|---|---|---|---|---|---|---|"]
    for x in runs:
        lr = "" if x["lr"] is None else f"{x['lr']:g}"
        lines.append(f"| {x['optimizer']} | {x['transform']} | {x['start']} | {lr} | "
                     f"{x['evals_to_1e-2']} | {x['evals_to_1e-3']} | {x['evals_to_1pct']} | "
                     f"{x['f_ratio_min']:.2e} |")
    lines += ["", f"verdict: {json.dumps(verdict)}",
              "log is shown for completeness only: the surrogate is quadratic in p, which favours "
              "linear by construction."]
    _write_new(out_dir / "surrogate.json",
               json.dumps({"ident": str(ident), "runs": runs, "verdict": verdict}) + "\n")
    _write_new(out_dir / "surrogate.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return {"runs": runs, "verdict": verdict}


def surrogate_verdict(runs, margin: int = 3) -> dict:
    """Per optimiser: "units absorbed" if raw reaches J/J0 ≤ 1e-2 within ``margin`` evaluations of
    linear at every start, else "units not absorbed"."""
    out = {}
    for opt in sorted({x["optimizer"] for x in runs}):
        by = {(x["transform"], x["start"]): x["evals_to_1e-2"] for x in runs if x["optimizer"] == opt}
        ok = True
        for s in STARTS:
            raw, lin = by.get(("raw", s)), by.get(("linear", s))
            if lin is None:
                continue
            if raw is None or raw > lin + margin:
                ok = False
        out[opt] = "units absorbed" if ok else "units not absorbed"
    return out


# ---------------------------------------------------------------------------
# The real model: the Calibration exactly as tke_lbfgs.py builds it


def build_calib(argv):
    """``(calib, cs, args)`` from tke_lbfgs command-line options, constructed exactly as
    ``tke_lbfgs.main()`` does (tke_lbfgs.py:1499-1531), so the defaults cannot drift. ``--out``
    is required (it holds the twin targets)."""
    args = T._parse(["calib"] + list(argv))
    if not args.out:
        raise ValueError("--out is required")
    sig = T._floats(args.sigma) if args.sigma else None
    cs = T.tke_control_set() if sig is None else T.tke_control_set(
        sigma=sig if len(sig) > 1 else sig[0])
    starts = [datetime.datetime(args.year, int(m), 1) for m in args.months.split(",")]
    out = Path(args.out)
    states_dir = Path(args.states_dir)
    first = states_dir / T.C.restart_tag(starts[0])
    model = T.build_calib_model(starts[0], args.window_steps, args.chunk_steps, cs=cs,
                                mld_invalid=args.mld_invalid, adjoint_cfg=T.C._adjoint_cfg(args),
                                ice_mode=args.ice,
                                save_names=tuple(s for s in args.save_names.split(",") if s),
                                rematerialize=not args.no_remat,
                                restart=first if first.is_dir() else None)
    u_truth = cs.to_u(T._floats(args.truth)) if args.mode == "twin" else None
    calib = T.Calibration(model, starts, cs=cs, mode=args.mode, woa=args.woa,
                          sigma_obs=args.sigma_obs, n_eff=args.n_eff, prior=args.prior == "on",
                          u_truth=u_truth, ice_sst=args.ice_sst, states={starts[0]: model.s0},
                          states_dir=states_dir, out_dir=out, mld_invalid=args.mld_invalid,
                          first_check=args.first_check, targets_from=args.twin_targets)
    return calib, cs, args


# ---------------------------------------------------------------------------
# Gate: linear and log at the first guess agree within the GPU repeat spread

#: relative floors of the gate tolerance (two evaluations of the same GPU program differ at
#: round-off; track F measured a 6-hour gradient repeat spread up to 3e-5)
GATE_G_FLOOR = 1e-4
GATE_F_FLOOR = 1e-8
#: fallback trigger: the twin noise floor J_data(p*) relative to J_data(p̄)
FLOOR_MAX = 1e-5


def gate_verdict(log1, log2, lin) -> dict:
    """``(f, g)`` of log twice and linear once at the first guess: linear passes if it is within
    max(3·log-log spread, floor) of the first log evaluation, per gradient component."""
    f1, g1 = float(log1[0]), np.asarray(log1[1], np.float64)
    f2, g2 = float(log2[0]), np.asarray(log2[1], np.float64)
    fl, gl = float(lin[0]), np.asarray(lin[1], np.float64)
    tol_g = np.maximum(3.0 * np.abs(g2 - g1), GATE_G_FLOOR * np.max(np.abs(g1)))
    tol_f = max(3.0 * abs(f2 - f1), GATE_F_FLOOR * abs(f1))
    ok = bool(np.all(np.abs(gl - g1) <= tol_g) and abs(fl - f1) <= tol_f)
    return {"pass": ok, "f": [f1, f2, fl], "g_log1": g1.tolist(), "g_log2": g2.tolist(),
            "g_lin": gl.tolist(), "spread_g": np.abs(g2 - g1).tolist(),
            "diff_lin_g": np.abs(gl - g1).tolist(), "tol_g": tol_g.tolist(), "tol_f": tol_f}


def run_gate(calib, cs, out_dir) -> dict:
    out_dir = Path(out_dir)
    if (out_dir / "gate.json").exists():
        raise FileExistsError(f"{out_dir / 'gate.json'} exists")
    u0 = np.zeros(cs.n, np.float64)
    secs, res = [], []
    for fg in (calib.fun_and_grad, calib.fun_and_grad,
               reparam(calib.fun_and_grad, transform_for(cs, "linear"))):
        t0 = time.time()
        res.append(fg(u0.copy()))
        secs.append(time.time() - t0)
    verdict = gate_verdict(res[0][:2], res[1][:2], res[2][:2])
    t0 = time.time()
    J_star, aux_star = calib.fun(calib.u_truth)
    secs.append(time.time() - t0)
    ratio = float(aux_star["J_data"]) / float(res[0][2]["J_data"])
    fallback = (not math.isfinite(J_star)) or (not math.isfinite(ratio)) or ratio > FLOOR_MAX
    out = {"gate": verdict, "seconds": secs, "J_data_p0": res[0][2]["J_data"],
           "J_data_truth": aux_star["J_data"], "floor_ratio": ratio, "fallback_needed": fallback,
           "u_truth": np.asarray(calib.u_truth).tolist(), "p_truth": calib.physical(calib.u_truth)}
    _write_new(out_dir / "gate.json", json.dumps(T.C._jsonable(out), indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("seconds", "floor_ratio", "fallback_needed")}
                     | {"pass": verdict["pass"]}), flush=True)
    return out


# ---------------------------------------------------------------------------
# S2: forward-only cost profiles through the truth, one constant at a time

PROFILE_K = tuple(range(-4, 5))                 # p*_i · 2^(k/2): ×1/4 … ×4


def profile_points(p_star) -> list:
    """``(i, k, p)`` for each constant i and k in PROFILE_K, the centre (k = 0) once as i = -1."""
    p_star = np.asarray(p_star, np.float64)
    pts = [(-1, 0, p_star.copy())]
    for i in range(len(p_star)):
        for k in PROFILE_K:
            if k == 0:
                continue
            p = p_star.copy()
            p[i] = p_star[i] * 2.0 ** (k / 2.0)
            pts.append((i, k, p))
    return pts


def _r2(x, y) -> float:
    c = np.polyfit(x, y, 2)
    res = y - np.polyval(c, x)
    tot = y - y.mean()
    return float(1.0 - (res @ res) / (tot @ tot))


def profile_metrics(fac, J) -> dict:
    """Shape of one profile: ``fac`` = p/p* (from 2^(k/2)), ``J`` the cost. Primary: the ratios
    J(2p*)/J(p*/2) and J(4p*)/J(p*/4) (quadratic in ln p ⇒ 1 and 1; quadratic in p ⇒ 4 and 16).
    Context: R² of a quadratic fit in p and in ln p over ×½…×2 and ×¼…×4."""
    fac, J = np.asarray(fac, np.float64), np.asarray(J, np.float64)
    at = {round(math.log2(f) * 2): j for f, j in zip(fac, J)}
    out = {"ratio_2": at[2] / at[-2], "ratio_4": at[4] / at[-4]}
    for name, kmax in (("x2", 2), ("x4", 4)):
        m = np.abs(np.round(np.log2(fac) * 2)) <= kmax
        out[f"r2_lin_{name}"] = _r2(fac[m], J[m])
        out[f"r2_log_{name}"] = _r2(np.log(fac[m]), J[m])
    return out


def run_profile(calib, cs, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "profile.jsonl"
    done = {}
    if log.exists():                             # resume: skip the points already evaluated
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done[(r["i"], r["k"])] = r
    p_star = np.asarray(calib.physical(calib.u_truth), np.float64)
    with open(log, "a") as fh:
        for i, k, p in profile_points(p_star):
            if (i, k) in done:
                continue
            t0 = time.time()
            J, aux = calib.fun(cs.to_u(p))
            r = {"i": i, "k": k, "p": p.tolist(), "J": float(J), "J_data": float(aux["J_data"]),
                 "seconds": time.time() - t0}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done[(i, k)] = r
    centre = done[(-1, 0)]["J_data"]
    res = {}
    for i, name in enumerate(cs.names):
        ks = sorted(k for (ii, k) in done if ii == i) + [0]
        ks = sorted(ks)
        J = [centre if k == 0 else done[(i, k)]["J_data"] for k in ks]
        res[name] = profile_metrics([2.0 ** (k / 2.0) for k in ks], J)
    out = {"p_truth": p_star.tolist(), "J_data_truth": centre, "metrics": res}
    _write_new(out_dir / "profile.json", json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)
    return out


# ---------------------------------------------------------------------------
# S3: the far-truth twin with L-BFGS in one arm's variable


def _model_evals_logged(log: Path) -> int:
    """Model evaluations already in an optimiser log (domain rejections excluded)."""
    if not log.exists():
        return 0
    n = 0
    for line in log.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "eval" and not (r.get("aux") or {}).get("out_of_domain", False):
            n += 1
    return n


def twin_meta(cs, kind: str, start: str, replicate: int, model_budget: int, maxeval: int) -> dict:
    tr = transform_for(cs, kind)
    p_start = cs.p0 * STARTS[start]
    return {"transform": kind, "p0": list(tr.p0), "sigma": list(tr.sigma), "start": start,
            "p_start": p_start.tolist(), "v0": tr.to_v(p_start).tolist(),
            "replicate": int(replicate), "model_budget": int(model_budget),
            "maxeval": int(maxeval)}


def check_or_write_meta(out_dir: Path, meta: dict) -> None:
    """``norm_ab.json``: written once; a rerun must match field by field (at the p̄ start v0 = 0
    in every arm, so the optimiser's own header cannot tell the arms apart)."""
    path = out_dir / "norm_ab.json"
    if path.exists():
        old = json.loads(path.read_text())
        if json.dumps(old, sort_keys=True) != json.dumps(meta, sort_keys=True):
            raise ValueError(f"{path} belongs to another arm: {old} vs {meta}")
        return
    _write_new(path, json.dumps(meta, indent=1, sort_keys=True) + "\n")


def run_twin(calib, cs, out_dir, *, kind: str, start: str, replicate: int = 0,
             model_budget: int = 25, maxeval: int = 40, wall_budget_s=None,
             first_eval_s: float = 900.0) -> dict:
    out_dir = Path(out_dir)
    meta = twin_meta(cs, kind, start, replicate, model_budget, maxeval)
    check_or_write_meta(out_dir, meta)
    tr = transform_for(cs, kind)
    n_done = _model_evals_logged(out_dir / "iterations.jsonl")

    def wrap(f):
        g = reparam(f, tr, max_model_evals=model_budget, n_done=n_done)
        return g if wall_budget_s is None else T.wall_budgeted(g, wall_budget_s, first_eval_s)

    status = None
    try:
        r = T.run_lbfgs(calib, out_dir, u0=np.array(meta["v0"], np.float64), maxiter=maxeval,
                        maxeval=maxeval, options=optim_lbfgs.LBFGSOptions(expand=T.LBFGS_EXPAND),
                        wrap=wrap)
        status = r.status
    except ModelBudget:
        status = "model_budget"
    except T.WallBudget:
        status = "paused_wall_budget"
    print(json.dumps({"out": str(out_dir), "status": status,
                      "model_evals": _model_evals_logged(out_dir / "iterations.jsonl")}),
          flush=True)
    return {"status": status}


# ---------------------------------------------------------------------------
# Task 7: read the twin runs, apply the decision rule, write the report

MILESTONES = (1e-2, 1e-3)


def read_run(run_dir) -> dict:
    """One twin run: its meta, and the MODEL-evaluation trace (domain rejections excluded) of f
    and physical p from ``iterations.jsonl``; ``status`` = the optimiser's stop row, else
    "model_budget" (the wrapper ended it) or "incomplete"."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "norm_ab.json").read_text())
    f, ps, rej, status = [], [], 0, None
    for line in (run_dir / "iterations.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "eval":
            aux = r.get("aux") or {}
            if aux.get("out_of_domain", False):
                rej += 1
                continue
            f.append(float(r["f"]))
            ps.append([float(x) for x in aux["p"]])
        elif r.get("type") == "stop":
            status = r.get("status")
    if status is None:
        status = "model_budget" if len(f) >= int(meta.get("model_budget", 10 ** 9)) else "incomplete"
    return {"dir": str(run_dir), **meta, "f": f, "p": ps, "n_rejected": rej, "status": status}


def eig_errors(ps, ident_row) -> list:
    """Errors in the identifiable eigen-directions (λ ≥ 10) of H(p*) in log-u, units of 1 %."""
    V = np.array(ident_row["eigvecs"], np.float64)          # rows = eigenvectors
    lam = np.array(ident_row["eigvals"], np.float64)
    V = V[lam >= 10.0]
    p0 = np.array(ident_row["calib"]["p0"], np.float64)
    sig = np.array(ident_row["calib"]["sigma"], np.float64)
    u_star = np.array(ident_row["u"], np.float64)
    out = []
    for p in ps:
        u = np.log(np.asarray(p, np.float64) / p0) / sig
        out.append((np.abs(V @ (u - u_star)) / U_PCT).tolist())
    return out


def decide(mil: dict, stalled: dict, *, budget: int) -> dict:
    """The S3 decision rule at one milestone. ``mil = {"arms": {start: {"log": n, "linear": n}},
    "replicate": n}`` (model evaluations to reach it, None = never); ``stalled = {start: bool}``
    (the linear run ended line_search_stalled after domain rejections without reaching it)."""
    arms = mil["arms"]
    starts = list(arms)
    if sum(arms[s]["log"] is None and arms[s]["linear"] is None for s in starts) >= 2:
        return {"verdict": "inconclusive", "reason": "neither arm reaches it at >= 2 starts"}
    eff = lambda n: budget + 1 if n is None else n                       # noqa: E731
    lp0, rep = arms["p0"]["log"], mil.get("replicate")
    noise = (1 if lp0 is None and rep is None else
             budget if (lp0 is None) != (rep is None) else max(1, abs(lp0 - rep)))
    need = max(3, 2 * noise)
    gaps = {s: eff(arms[s]["linear"]) - eff(arms[s]["log"]) for s in starts}
    med = float(np.median(list(gaps.values())))
    base = {"gaps": gaps, "median_gap": med, "noise": noise, "need": need}
    if all(g > 0 for g in gaps.values()) and med >= need:
        return {"verdict": "log better", "reason": "fewer evaluations", **base}
    if all(g < 0 for g in gaps.values()) and -med >= need:
        return {"verdict": "linear better", "reason": "fewer evaluations", **base}
    if any(stalled.get(s, False) and arms[s]["log"] is not None for s in starts):
        return {"verdict": "log better", "reason": "linear stalled at p = 0", **base}
    return {"verdict": "no practical difference", "reason": "gap below the rule", **base}


def run_report(runs_root, ident, out_dir, *, budget: int = 25) -> dict:
    runs_root, out_dir = Path(runs_root), Path(out_dir)
    ident_row = _truth_row(ident, n_steps=12)
    runs = [read_run(d) for d in sorted(runs_root.glob("norm_ab_twin_*"))
            if (d / "norm_ab.json").is_file() and (d / "iterations.jsonl").is_file()]
    for r in runs:
        r["err"] = eig_errors(r["p"], ident_row)
        r["evals_to"] = {str(m): evals_to(r["f"], m) for m in MILESTONES}
        r["evals_to_1pct"] = evals_to_error(r["err"])
        r["f_ratio_min"] = min(r["f"]) / r["f"][0]
    main = {(r["transform"], r["start"]): r for r in runs if r["replicate"] == 0}
    rep = [r for r in runs if r["replicate"] != 0 and r["transform"] == "log" and r["start"] == "p0"]
    verdicts = {}
    for m in MILESTONES:
        mil = {"arms": {s: {k: main[(k, s)]["evals_to"][str(m)] if (k, s) in main else None
                            for k in ("log", "linear")} for s in STARTS},
               "replicate": rep[0]["evals_to"][str(m)] if rep else None}
        stalled = {s: bool((("linear", s) in main)
                           and main[("linear", s)]["n_rejected"] > 0
                           and main[("linear", s)]["status"] == "line_search_stalled"
                           and main[("linear", s)]["evals_to"][str(m)] is None) for s in STARTS}
        verdicts[str(m)] = decide(mil, stalled, budget=budget)
    lines = ["| transform | start | rep | model evals | rejections | status | to 1e-2 | to 1e-3 | "
             "≤ 1 % error | min J/J0 | final p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["transform"], r["start"], r["replicate"])):
        lines.append(f"| {r['transform']} | {r['start']} | {r['replicate']} | {len(r['f'])} | "
                     f"{r['n_rejected']} | {r['status']} | {r['evals_to']['0.01']} | "
                     f"{r['evals_to']['0.001']} | {r['evals_to_1pct']} | {r['f_ratio_min']:.2e} | "
                     f"{', '.join(f'{x:.4g}' for x in r['p'][-1])} |")
    lines += ["", f"identifiable directions (λ ≥ 10) at p*: "
              f"{[round(x, 3) for x in ident_row['eigvals'] if x >= 10.0]}"]
    for m, v in verdicts.items():
        lines.append(f"milestone J/J0 ≤ {m}: **{v['verdict']}** ({v['reason']}; "
                     f"{ {k: v[k] for k in v if k not in ('verdict', 'reason')} })")
    _write_new(out_dir / "report.json",
               json.dumps({"runs": runs, "verdicts": verdicts}, default=float) + "\n")
    _write_new(out_dir / "report.md", "\n".join(lines) + "\n")
    _plot_report(runs, out_dir / "jj0_vs_eval.png")
    print("\n".join(lines), flush=True)
    return {"runs": runs, "verdicts": verdicts}


def _plot_report(runs, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if path.exists():
        raise FileExistsError(path)
    color = {"log": "#1f77b4", "linear": "#d62728", "raw": "#7f7f7f"}
    marker = {"p0": "o", "plus": "^", "minus": "v"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for r in runs:
        f = np.asarray(r["f"]) / r["f"][0]
        ax.semilogy(np.arange(1, len(f) + 1), f, color=color[r["transform"]],
                    marker=marker[r["start"]], ms=3.5, lw=1.2,
                    ls="--" if r["replicate"] else "-",
                    label=f"{r['transform']} {r['start']}" + (" (replicate)" if r["replicate"] else ""))
    for m in MILESTONES:
        ax.axhline(m, color="0.6", lw=0.6, ls=":")
    ax.set(xlabel="model evaluation", ylabel="J / J0", title="Far-truth twin, L-BFGS: log vs linear controls")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI


def _parse(argv=None):
    """``(args, rest)``: this driver's options; for gate/profile/twin ``rest`` is the tke_lbfgs
    command line that builds the Calibration (its --out, --maxeval, --wall-budget-s,
    --first-eval-s are used as they are there)."""
    # allow_abbrev=False everywhere: the forwarded tke_lbfgs option --mode would otherwise be read
    # as an abbreviation of --model-budget (it was, and killed five twin jobs at start-up)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest="action", required=True)
    c = sub.add_parser("cond", allow_abbrev=False, help="S0: κ of track F's Gauss-Newton Hessians per transform")
    c.add_argument("--ident", nargs="+", required=True)
    c.add_argument("--out", required=True)
    s1 = sub.add_parser("surrogate", allow_abbrev=False, help="S1: optimiser A/B on the measured-Hessian surrogate")
    s1.add_argument("--ident", required=True, help="track F's 12-hour ident.jsonl (truth row)")
    s1.add_argument("--out", required=True)
    s1.add_argument("--maxeval", type=int, default=30)
    sub.add_parser("gate", allow_abbrev=False, help="linear vs log at the first guess on the GPU; twin targets")
    sub.add_parser("profile", allow_abbrev=False, help="S2: forward-only cost profiles through the truth")
    tw = sub.add_parser("twin", allow_abbrev=False, help="S3: L-BFGS in one arm's variable from one start")
    tw.add_argument("--transform", choices=KINDS, required=True)
    tw.add_argument("--start", choices=tuple(STARTS), required=True)
    tw.add_argument("--replicate", type=int, default=0)
    tw.add_argument("--model-budget", type=int, default=25)
    rp = sub.add_parser("report", allow_abbrev=False, help="Task 7: read the twin runs, decide, write table + figure")
    rp.add_argument("--runs-root", required=True, help="directory holding norm_ab_twin_*")
    rp.add_argument("--ident", required=True, help="ident.jsonl at the far truth (12-step rows)")
    rp.add_argument("--out", required=True)
    rp.add_argument("--model-budget", type=int, default=25)
    args, rest = ap.parse_known_args(argv)
    if args.action in ("cond", "surrogate", "report") and rest:
        ap.error(f"unrecognized arguments: {rest}")
    return args, rest


def main(argv=None) -> int:
    args, rest = _parse(argv)
    if args.action == "cond":
        run_cond(args.ident, args.out)
    elif args.action == "surrogate":
        run_surrogate(args.ident, args.out, maxeval=args.maxeval)
    elif args.action == "report":
        run_report(args.runs_root, args.ident, args.out, budget=args.model_budget)
    else:
        calib, cs, targs = build_calib(rest)
        out = Path(targs.out)
        if args.action == "gate":
            run_gate(calib, cs, out)
        elif args.action == "profile":
            run_profile(calib, cs, out)
        else:
            run_twin(calib, cs, out, kind=args.transform, start=args.start,
                     replicate=args.replicate, model_budget=args.model_budget,
                     maxeval=targs.maxeval, wall_budget_s=targs.wall_budget_s,
                     first_eval_s=targs.first_eval_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
