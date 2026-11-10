"""FD Gauss–Newton steps of the sea-ice thermodynamic calibration: the cross-check of the adjoint
L-BFGS of S3 (Nikolay 2026-10-15), with no derivative of the model.

Cost J(u) = ½ (F(u) − o)ᵀ W (F(u) − o) [+ ½ uᵀu with the prior], F the cost's observation-space
vector (:class:`s1sum.ObsSpace`: C̄ at the valid concentration super-observations, H̄ at the valid
northern thickness ones). Every run's J_data is recomputed from its ``fields.npz`` and must equal
its ``fwd.json``. Only the ACTIVE controls (``--controls``) move; the others keep their value.

``step`` at the current point u_k:

* r = F̄ − o, F̄ the mean of the runs AT u_k (``--base``: fwd or fdbatch run directories);
* the Jacobian A in u (one column per active control) from the central fdbatch pairs under
  ``--jac`` (their centre u_c may differ from u_k: a kept Jacobian), at the largest h of each
  control: A_i = (F(u_c + d_i e_i) − F(u_c − d_i e_i)) / (2 d_i), d_i = h_i/σ_i;
* GPU noise: s² = Σ_k ‖F_k − F̄‖²_W / (n − 1) over the repeats ``--noise`` (≥ 2 runs at one point)
  is one run's expected ‖n‖²_W; a central column carries E‖e_i‖²_W = s²/(2 d_i²), taken off the
  diagonal of AᵀWA as in s1sum, and the data block is then clipped to positive semi-definite;
* the Gauss–Newton model m(δ) = J_k + gᵀδ + ½ δᵀHδ with g = AᵀWr (+ u_a), H = AᵀWA − N (+ I);
* one trial per trust radius Δ of ``--radii`` (prior σ units; ``inf`` = no limit): δ_GN = −H⁻¹g
  if |δ_GN|₂ ≤ Δ, else the trust-region step δ(μ) = −(H + μI)⁻¹g with |δ(μ)|₂ = Δ (μ > 0 by
  bisection; it turns towards steepest descent as Δ shrinks, so a direction whose data curvature
  drowned in the noise cannot steer a shortened step, as it would a scaled-down δ_GN); a trial
  outside the physical domain (``ControlSet.out_of_domain``) gets no file;
* writes ``<out>/step.json`` and ``<out>/trial_r<Δ>.u.json`` per trial (for ``icecal fwd --u``).
  With ``--eval`` (an adjoint ``eval.json`` at u_k) the step also reports the angle and the scale
  of the adjoint gradient against the Gauss–Newton one.

``accept``: the trial runs' ``fwd.json`` → J = J_data (+ ½ uᵀu), the gain ratio
ρ = (J_k − J) / (J_k − m(δ)) and the best trial (accepted if its J < J_k) →
``<step dir>/accept.json``.

    python scripts/icecal/fdgn.py step --out DIR --base RUN [RUN ...] --jac FD_ROOT
        --noise RUN RUN [...] --controls con,consn,... [--prior on] [--radii 1,0.5]
    python scripts/icecal/fdgn.py accept --step DIR --trial r1=RUN --trial r0.5=RUN
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import s1sum as S  # noqa: E402
from fesom_jax import seaice_cost  # noqa: E402
from fesom_jax.controls import ice_thermo_control_set  # noqa: E402

#: provenance keys every run of one window must share
WINDOW_KEYS = ("start", "n_steps", "chunk_steps", "obs_sha256", "controls", "setting")


# ---------------------------------------------------------------------------
# reading


def read_run(d) -> dict:
    d = Path(d)
    with open(d / "fwd.json") as f:
        j = json.load(f)
    p = j["provenance"]
    return {"dir": str(d), "u": np.asarray(p["u"], np.float64), "fd": p.get("fd"),
            "J_data": float(j["J_data"]), "prov": p}


def check_window(runs: list) -> dict:
    ref = None
    for r in runs:
        v = {k: r["prov"].get(k) for k in WINDOW_KEYS}
        if ref is None:
            ref = v
        elif v != ref:
            diff = {k: (ref[k], v[k]) for k in ref if ref[k] != v[k]}
            raise ValueError(f"{r['dir']} is not the same window as the others: {diff}")
    return ref


def common_u(runs: list, what: str) -> np.ndarray:
    u = runs[0]["u"]
    for r in runs[1:]:
        if not np.array_equal(r["u"], u):
            raise ValueError(f"{what}: {r['dir']} is at u {r['u'].tolist()}, not {u.tolist()}")
    return u


def jac_pairs(jac_dir, controls: list, names: list) -> dict:
    """``{control: (plus run, minus run)}``: the central fdbatch pair at the largest h with both
    signs, all around one centre u_c."""
    pert = {}
    for fj in sorted(glob.glob(os.path.join(str(jac_dir), "*", "fwd.json"))):
        r = read_run(Path(fj).parent)
        info = r["fd"]
        if info is None or info["control"] is None:
            continue
        key = (S._short(info["control"]), float(info["h_lnp"]), int(info["sign"]))
        if key in pert:
            raise ValueError(f"two FD runs for {key}: {pert[key]['dir']} and {r['dir']}")
        pert[key] = r
    out = {}
    for c in controls:
        hs = [h for (cc, h, s) in pert if cc == c and s == 1 and (c, h, -1) in pert]
        if not hs:
            raise ValueError(f"{jac_dir}: no central pair for {c}")
        h = max(hs)
        rp, rm = pert[(c, h, 1)], pert[(c, h, -1)]
        uc_p, uc_m = np.asarray(rp["fd"]["u0"]), np.asarray(rm["fd"]["u0"])
        if not np.array_equal(uc_p, uc_m):
            raise ValueError(f"{c}: the pair has two centres")
        i = names.index(c)
        for r in (rp, rm):
            off = np.delete(r["u"] - uc_p, i)
            if np.any(off != 0.0):
                raise ValueError(f"{r['dir']} moves controls other than {c}")
        out[c] = (rp, rm)
    centres = {tuple(p[0]["fd"]["u0"]) for p in out.values()}
    if len(centres) != 1:
        raise ValueError(f"{jac_dir}: the pairs have different centres {centres}")
    return out


class Fields:
    """F of every run, J_data recomputed and checked against ``fwd.json``."""

    def __init__(self, obs_path, runs: list, j_rtol: float = 1e-9):
        with np.load(obs_path) as z:
            ob = seaice_cost.cost_obs({k: np.asarray(z[k]) for k in z.files})
        self.space = None
        self.F = {}
        self.worst = 0.0
        have_ref = None
        for r in runs:
            if r["dir"] in self.F:
                continue
            C, H, n = S.read_fields(r["dir"])
            have = n > 0
            if self.space is None:
                self.space, have_ref = S.ObsSpace(ob, have), have
            elif not np.array_equal(have, have_ref):
                raise ValueError(f"{r['dir']}: months with steps differ from the other runs")
            F = self.space.F(C, H)
            rel = abs(self.space.J(F) - r["J_data"]) / max(abs(r["J_data"]), 1e-300)
            if rel > j_rtol:
                raise ValueError(f"{r['dir']}: J_data from fields.npz != fwd.json (rel {rel:.2e})")
            self.worst = max(self.worst, rel)
            self.F[r["dir"]] = F


# ---------------------------------------------------------------------------
# the Gauss–Newton model


def noise_s2(Fs: list, W) -> float | None:
    if len(Fs) < 2:
        return None
    Fbar = np.mean(Fs, axis=0)
    return math.fsum(math.fsum((W * (F - Fbar) ** 2).tolist()) for F in Fs) / (len(Fs) - 1)


def gn_model(A, W, r, u_a, d, s2, *, prior: bool) -> dict:
    """g, H (u-space, active block) of m(δ) = J_k + gᵀδ + ½ δᵀHδ; the noise term s²/(2 d_i²) off
    the data block's diagonal, then the data block clipped to positive semi-definite."""
    A = np.asarray(A, np.float64)
    W = np.asarray(W, np.float64)
    k = A.shape[1]
    H_raw = A.T @ (A * W[:, None])
    N = np.zeros(k) if s2 is None else s2 / (2.0 * np.asarray(d, np.float64) ** 2)
    H_cor = H_raw - np.diag(N)
    lam, V = np.linalg.eigh(0.5 * (H_cor + H_cor.T))
    H_data = (V * np.clip(lam, 0.0, None)) @ V.T
    g = A.T @ (W * np.asarray(r, np.float64))
    H = H_data.copy()
    if prior:
        g = g + np.asarray(u_a, np.float64)
        H = H + np.eye(k)
    return {"g": g, "H": H, "H_raw": H_raw, "noise_diag": N, "data_eigenvalues": lam[::-1]}


def tr_step(g, H, radius: float) -> tuple[np.ndarray, float]:
    """(δ, μ): δ_GN = −H⁻¹g (least squares if H is singular, μ = 0) if |δ_GN| ≤ radius, else
    δ(μ) = −(H + μI)⁻¹g with |δ(μ)| = radius (H symmetric positive semi-definite)."""
    g = np.asarray(g, np.float64)
    delta = -np.linalg.lstsq(H, g, rcond=None)[0]
    if not math.isfinite(radius) or np.linalg.norm(delta) <= radius:
        return delta, 0.0
    lam, V = np.linalg.eigh(0.5 * (H + H.T))
    lam = np.clip(lam, 0.0, None)
    q = V.T @ g

    def norm(mu):
        return float(np.linalg.norm(q / (lam + mu)))

    lo, hi = 0.0, max(1.0, float(np.linalg.norm(g)) / radius)
    while norm(hi) > radius:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm(mid) > radius:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * hi:
            break
    return -V @ (q / (lam + hi)), hi


def model_J(J_k: float, g, H, delta) -> float:
    return float(J_k + g @ delta + 0.5 * delta @ H @ delta)


def angle(a, b) -> tuple[float, float]:
    """(angle in degrees, scale = a·b/|b|²) of a against b."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos)))), float(a @ b / (b @ b))


# ---------------------------------------------------------------------------
# actions


def step(out, *, base, jac, noise, controls, prior=True, radii=(1.0, 0.5),
         eval_path=None, obs=None) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    cs = ice_thermo_control_set()
    names = [S._short(n) for n in cs.names]
    bad = [c for c in controls if c not in names]
    if bad:
        raise ValueError(f"unknown controls {bad}; known {names}")
    act = np.array([names.index(c) for c in controls])
    base_runs = [read_run(d) for d in base]
    noise_runs = [read_run(d) for d in noise]
    pairs = jac_pairs(jac, controls, names)
    pair_runs = [r for p in pairs.values() for r in p]
    win = check_window(base_runs + noise_runs + pair_runs)
    u_k = common_u(base_runs, "--base")
    common_u(noise_runs, "--noise")
    u_c = np.asarray(next(iter(pairs.values()))[0]["fd"]["u0"], np.float64)
    fl = Fields(obs or base_runs[0]["prov"]["obs"], base_runs + noise_runs + pair_runs)
    sp = fl.space
    sigma = cs.sigma
    d = np.array([(pairs[c][0]["u"][i] - pairs[c][1]["u"][i]) / 2.0
                  for c, i in zip(controls, act)])
    A = np.stack([(fl.F[pairs[c][0]["dir"]] - fl.F[pairs[c][1]["dir"]]) / (2.0 * di)
                  for c, di in zip(controls, d)], axis=1)
    Fbar = np.mean([fl.F[r["dir"]] for r in base_runs], axis=0)
    r = Fbar - sp.o
    s2 = noise_s2([fl.F[x["dir"]] for x in noise_runs], sp.W)
    J_data_k = sp.J(Fbar)
    J_prior_k = 0.5 * float(u_k @ u_k) if prior else 0.0
    J_k = J_data_k + J_prior_k
    m = gn_model(A, sp.W, r, u_k[act], d, s2, prior=prior)
    delta, _ = tr_step(m["g"], m["H"], math.inf)
    trials = [(float(rad),) + tr_step(m["g"], m["H"], float(rad)) for rad in radii]
    lam, V = np.linalg.eigh(m["H"])
    order = np.argsort(lam)[::-1]
    res = {
        "action": "step", "controls": controls, "active_index": act.tolist(), "prior": prior,
        "window": win, "u_k": u_k.tolist(), "u_c": u_c.tolist(),
        "jacobian_is_current": bool(np.array_equal(u_c, u_k)),
        "base": [x["dir"] for x in base_runs], "noise": [x["dir"] for x in noise_runs],
        "pairs": {c: [p[0]["dir"], p[1]["dir"]] for c, p in pairs.items()},
        "d_u": d.tolist(), "h_lnp": (d * sigma[act]).tolist(),
        "J_recompute_max_rel": fl.worst, "n_obs": int(sp.W.size),
        "J_data_k": J_data_k, "J_prior_k": J_prior_k, "J_k": J_k,
        "J_data_base_runs": [x["J_data"] for x in base_runs],
        "noise_s2": s2, "noise_diag": m["noise_diag"].tolist(),
        "column_norm2_W": [float(sp.norm2(A[:, j])) for j in range(len(controls))],
        "g": m["g"].tolist(), "H": m["H"].tolist(), "H_raw": m["H_raw"].tolist(),
        "data_eigenvalues_after_noise": m["data_eigenvalues"].tolist(),
        "H_spectrum": [{"eigenvalue": float(lam[i]),
                        "vector": dict(zip(controls, V[:, i].tolist()))} for i in order],
        "posterior_cov_raw": np.linalg.inv(m["H"]).tolist() if prior else None,
        "delta_gn": delta.tolist(), "delta_gn_norm": float(np.linalg.norm(delta)),
        "radii": [float(x) for x in radii], "trials": [],
    }
    if np.array_equal(u_c, u_k):
        # the FD of J itself along each pair (central; the prior added analytically)
        gJ = np.array([(pairs[c][0]["J_data"] - pairs[c][1]["J_data"]) / (2.0 * di)
                       for c, di in zip(controls, d)])
        if prior:
            gJ = gJ + u_k[act]
        res["g_fd_of_J"] = gJ.tolist()
        res["g_fd_of_J_vs_gn"] = dict(zip(("angle_deg", "scale"), angle(gJ, m["g"])))
    if eval_path is not None:
        with open(eval_path) as f:
            ev = json.load(f)
        if not np.allclose(np.asarray(ev["provenance"]["u"], np.float64), u_k, atol=0.0):
            raise ValueError(f"{eval_path} is not at u_k")
        ga = sigma[act] * np.asarray(ev["g_lnp_data"], np.float64)[act]
        if prior:
            ga = ga + u_k[act]
        res["g_adjoint"] = ga.tolist()
        res["g_adjoint_vs_gn"] = dict(zip(("angle_deg", "scale"), angle(ga, m["g"])))
    os.makedirs(out.parent, exist_ok=True)
    os.mkdir(out)
    for rad, dl, mu in trials:
        u_t = u_k.copy()
        u_t[act] = u_t[act] + dl
        oob = cs.out_of_domain(u_t)
        tag = f"r{rad:g}"
        rec = {"tag": tag, "radius": rad, "mu": mu, "delta": dl.tolist(),
               "delta_norm": float(np.linalg.norm(dl)), "u": u_t.tolist(),
               "p": (cs.p0 * np.exp(sigma * u_t)).tolist(), "out_of_domain": oob,
               "J_model": model_J(J_k, m["g"], m["H"], dl), "file": None}
        if not oob:
            rec["file"] = str(out / f"trial_{tag}.u.json")
            with open(rec["file"], "x") as f:
                json.dump({"u": u_t.tolist()}, f)
        res["trials"].append(rec)
    with open(out / "step.json", "x") as f:
        json.dump(S._jsonable(res), f, indent=1)
    return res


def accept(step_dir, trials: dict) -> dict:
    step_dir = Path(step_dir)
    with open(step_dir / "step.json") as f:
        st = json.load(f)
    by_tag = {t["tag"]: t for t in st["trials"]}
    rows = []
    for tag, d in trials.items():
        if tag not in by_tag:
            raise ValueError(f"no trial {tag!r} in {step_dir}; known {sorted(by_tag)}")
        t = by_tag[tag]
        run = read_run(d)
        if not np.array_equal(run["u"], np.asarray(t["u"], np.float64)):
            raise ValueError(f"{d} is at u {run['u'].tolist()}, not the trial's {t['u']}")
        Jp = 0.5 * float(run["u"] @ run["u"]) if st["prior"] else 0.0
        J = run["J_data"] + Jp
        pred = st["J_k"] - t["J_model"]
        rows.append({"tag": tag, "dir": str(d), "J_data": run["J_data"], "J_prior": Jp, "J": J,
                     "J_model": t["J_model"], "actual_decrease": st["J_k"] - J,
                     "predicted_decrease": pred,
                     "rho": (st["J_k"] - J) / pred if pred != 0 else None, "u": t["u"]})
    best = min(rows, key=lambda x: x["J"])
    res = {"action": "accept", "step": str(step_dir), "J_k": st["J_k"], "trials": rows,
           "best": best["tag"], "accepted": bool(best["J"] < st["J_k"]), "u_next": best["u"]}
    with open(step_dir / "accept.json", "x") as f:
        json.dump(S._jsonable(res), f, indent=1)
    return res


def _print_step(res: dict) -> None:
    c = res["controls"]
    print(f"u_k {dict(zip(c, [round(res['u_k'][i], 4) for i in res['active_index']]))}  "
          f"J_k {res['J_k']:.2f} (data {res['J_data_k']:.2f}, prior {res['J_prior_k']:.3f})  "
          f"s2 {res['noise_s2']}")
    print("control    d_u     |A|_W^2     noise     g          delta_gn")
    for j, n in enumerate(c):
        print(f"{n:9s} {res['d_u'][j]:6.3f} {res['column_norm2_W'][j]:10.4g} "
              f"{res['noise_diag'][j]:9.3g} {res['g'][j]:10.4g} {res['delta_gn'][j]:9.4f}")
    print("H eigenvalues", [round(x["eigenvalue"], 1) for x in res["H_spectrum"]])
    for k in ("g_fd_of_J_vs_gn", "g_adjoint_vs_gn"):
        if k in res:
            print(f"{k}: angle {res[k]['angle_deg']:.1f} deg, scale {res[k]['scale']:.3f}")
    print(f"|delta_gn| {res['delta_gn_norm']:.3f}")
    for t in res["trials"]:
        print(f"trial {t['tag']}: |delta| {t['delta_norm']:.3f} mu {t['mu']:.4g} "
              f"J_model {t['J_model']:.2f}  out_of_domain {t['out_of_domain']}  -> {t['file']}")
        print("   delta " + " ".join(f"{n}:{x:+.3f}" for n, x in zip(c, t["delta"])))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    s = sub.add_parser("step")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--base", required=True, nargs="+")
    s.add_argument("--jac", required=True)
    s.add_argument("--noise", required=True, nargs="+")
    s.add_argument("--controls", required=True)
    s.add_argument("--prior", choices=("on", "off"), default="on")
    s.add_argument("--radii", default="1,0.5", help="trust radii in prior sigma (inf: none)")
    s.add_argument("--eval", default=None, help="adjoint eval.json at u_k (reported only)")
    s.add_argument("--obs", default=None)
    a_ = sub.add_parser("accept")
    a_.add_argument("--step", required=True, type=Path)
    a_.add_argument("--trial", action="append", required=True, help="TAG=RUN_DIR")
    a = ap.parse_args(argv)
    if a.action == "step":
        res = step(a.out, base=a.base, jac=a.jac, noise=a.noise,
                   controls=[c.strip() for c in a.controls.split(",") if c.strip()],
                   prior=a.prior == "on",
                   radii=[float(x) for x in a.radii.split(",") if x.strip()],
                   eval_path=a.eval, obs=a.obs)
        _print_step(res)
        print(f"wrote {a.out}/step.json")
    else:
        res = accept(a.step, dict(t.split("=", 1) for t in a.trial))
        for r in res["trials"]:
            print(f"{r['tag']}: J {r['J']:.2f} (model {r['J_model']:.2f}), rho {r['rho']}")
        print(f"best {res['best']}, accepted {res['accepted']}; wrote {a.step}/accept.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
