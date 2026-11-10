#!/usr/bin/env python
"""S1 summary of one window of the sea-ice calibration (Task 13): the finite-difference bar, the
noise floor, the amplification screen of the reverse traces, the Gauss–Newton Hessian from the
FD fields and the parity verdict.

    python scripts/icecal/s1sum.py --window runs/icecal/s1/jan365 --out <new dir> \\
        [--parity runs/icecal/parity/fwd2013_u0_vs_default/parity.json] [--mesh-dir data/mesh_core2]

Inputs (all written by ``scripts/icecal/icecal.py``): ``<window>/fd/<tag>/{fwd.json,fields.npz}``
of ``fdbatch`` (the unperturbed repeats ``base_r<i>`` and the runs perturbed by ``sign·h`` in
ln p of one control; the control, h and sign are read from ``fwd.json``, not from the tag) and
every ``<window>/eval*/eval.json`` (+ ``trace.json``) of ``eval``, with or without ``--terms``.
All runs must share the window (start, steps, State, observations, controls, u0); a mismatch
is refused. Numpy only (no model build); run it on a compute node (CORE2-sized fields).

**Noise floor.** ``noise_J[t]`` = the sample standard deviation (ddof 1) of cost term ``t`` over
the base repeats (at least two). The repeats differ only by GPU round-off (same program, same
inputs).

**Bar** per control × term × window. FD at each h: central ``(J(+h) − J(−h)) / 2h`` in ln p (the
units of ``g_lnp`` in ``eval.json``), plus the one-sided ``(J(±h) − J̄_base) / ±h``, whose
difference shows a kink or jump. A FD counts as RESOLVED at h if ``|FD| ≥ k·noise_J/h`` (k = 3).
Verdict: ``unresolved`` if no h is resolved; ``pass`` if ``|adj/FD − 1| ≤ tol`` (0.10) at the
reference h (1e-2) when that FD is resolved, else at the smallest resolved h (the plan's 3e-2
runs exist for the controls whose 1e-2 FD may drown in the noise), or if the adjoint lies
within the range of the resolved FDs over at least two h; ``fail`` otherwise; ``no adjoint`` where no evaluation gave
that term's gradient (the FD is still reported). Only resolved FDs can pass a control: a match
to a noise-dominated FD is luck, not evidence.

**Amplification screen** of every reverse trace (``trace.json``: the cotangent norm per chunk
boundary, seed first): growth (largest norm / seed norm), finiteness, and one-boundary spikes,
i.e. a boundary whose norm exceeds ``spike`` (1.5) × the larger of its two neighbours, with its
date and the state fields whose norms rise most there. Which NODES carry a spike (and whether
they hold observations) needs a localisation run; the screen only flags the boundaries.

**Gauss–Newton Hessian.** F = the cost's observation-space vector (C̄ at the valid concentration
super-observations, H̄ at the valid northern thickness ones, months the window reaches), W =
w/σ² of :func:`fesom_jax.seaice_cost.cost_obs` on the same mask, so ``½ (F − o)ᵀ W (F − o)``
is ``J_data``; every run's ``J_data`` is recomputed from its ``fields.npz`` this way and must match
``fwd.json`` (the check that F and W are the cost's). Columns ``J_i = (F(+h_i) − F(−h_i)) / 2h_i``
at the LARGEST h with both signs (3e-2 where it exists), ``H_lnp = Jᵀ W J``. GPU noise inflates
the diagonal: with ``s² = Σ_k ‖F_k − F̄‖²_W / (r − 1)`` over the r base repeats (the expected
``‖n‖²_W`` of one run's noise), a central-difference column carries ``E‖e_i‖²_W = s² / (2h_i²)``,
which is subtracted from ``H_lnp[i, i]`` (two independent runs per column, so no off-diagonal
bias). (The plan wrote ``‖δF‖²_W / (2h²)`` with δF the difference of two repeats; since
``E‖δF‖² = 2 s²`` that is twice the bias, so the formula here is the plan's with δF read as one
run's noise.) ``H_u = Σ H_lnp Σ``, ``Σ = diag(σ)``; its spectrum relative to the largest
eigenvalue; ``identifiable`` = eigenvalue ≥ 10 (ten times the unit prior) and ≥ 1e-3 of the
largest.

**Parity verdict** (with ``--parity``, the ``parity.json`` of ``parity.py``): per control, the
response of the hemispheric area, volume (and the node-area RMS of C̄, H̄ over the ice domain)
to ONE prior σ (Δu = 1, i.e. Δln p = σ_i), from the same FD columns, against the largest monthly
JAX–Fortran difference of the same quantity: ``ratio = max_m |parity| / max_m |1σ response|``,
``negligible`` if every ratio ≤ ``parity_frac`` (0.1).

Writes ``<out>/s1sum.json`` (``<out>`` must not exist) and prints the bar table.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fesom_jax import seaice_cost  # noqa: E402

TERMS = ("J_data",) + seaice_cost.TERMS
#: provenance keys every run of one window must share
WINDOW_KEYS = ("start", "n_steps", "chunk_steps", "obs_sha256", "controls", "setting")


# ---------------------------------------------------------------------------
# reading


def _short(name: str) -> str:
    return name.split(".", 1)[1] if name.startswith("ice.") else name


def load_fd(fd_dir) -> dict:
    """``{"base": [run], "pert": {(control, h, sign): run}}``; run = ``{"tag", "dir", "J",
    "prov"}`` with ``J`` the per-term values."""
    base, pert = [], {}
    for fj in sorted(glob.glob(os.path.join(str(fd_dir), "*", "fwd.json"))):
        with open(fj) as f:
            d = json.load(f)
        info = d["provenance"].get("fd")
        if info is None:
            raise ValueError(f"{fj} is not an fdbatch run (no provenance.fd)")
        run = {"tag": Path(fj).parent.name, "dir": str(Path(fj).parent),
               "J": {t: float(d["terms"][t]) for t in TERMS}, "prov": d["provenance"]}
        if info["control"] is None:
            base.append(run)
        else:
            key = (_short(info["control"]), float(info["h_lnp"]), int(info["sign"]))
            if key in pert:
                raise ValueError(f"two FD runs for {key}: {pert[key]['dir']} and {run['dir']}")
            if not math.isclose(float(info["lnp_delta"]), key[1] * key[2], rel_tol=1e-9):
                raise ValueError(f"{fj}: lnp_delta {info['lnp_delta']} is not sign*h")
            pert[key] = run
    return {"base": base, "pert": pert}


def load_evals(paths) -> list:
    """Each ``eval.json`` as ``{"dir", "g_lnp": {term: [n]}, "J": {term}, "prov", "trace"}``.
    Without ``--terms`` only ``J_data`` has a gradient; with it, every listed term (and J_data
    as their sum when all three are present)."""
    out = []
    for ej in paths:
        with open(ej) as f:
            d = json.load(f)
        # with --terms, g_lnp_data is the sum of the listed terms: J_data's gradient only if
        # every unlisted term is zero in the window (g_data_is_full)
        g = {"J_data": d["g_lnp_data"]} if d.get("g_data_is_full", True) else {}
        for t, r in (d.get("per_term") or {}).items():
            g[t] = r["g_lnp"]
        tr_path = Path(ej).parent / "trace.json"
        trace = None
        if tr_path.exists():
            with open(tr_path) as f:
                trace = json.load(f)
        out.append({"dir": str(Path(ej).parent), "g_lnp": g, "prov": d["provenance"],
                    "J": {t: float(d["terms"][t]) for t in TERMS}, "trace": trace,
                    "linear_check": d.get("linear_check")})
    return out


def check_window(runs: list) -> dict:
    """All runs share the window; returns the common provenance values."""
    ref = None
    for r in runs:
        p = r["prov"]
        vals = {k: p.get(k) for k in WINDOW_KEYS}
        u0 = p.get("fd", {}).get("u0") if p.get("fd") else p.get("u")
        vals["u0"] = None if u0 is None else [float(x) for x in u0]
        if ref is None:
            ref = vals
        elif vals != ref:
            diff = {k: (ref[k], vals[k]) for k in ref if ref[k] != vals[k]}
            raise ValueError(f"{r['dir']} is not the same window as the others: {diff}")
    return ref


# ---------------------------------------------------------------------------
# noise, FD, verdicts


def noise_floor(base: list) -> dict:
    """Per term, the sample std (ddof 1) of J over the base repeats; None with < 2 repeats."""
    if len(base) < 2:
        return {t: None for t in TERMS}
    return {t: float(np.std([r["J"][t] for r in base], ddof=1)) for t in TERMS}


def fd_estimates(fd: dict, controls: list) -> dict:
    """``{control: {term: {h: {"central", "plus", "minus", "J_plus", "J_minus"}}}}`` (one-sided
    estimates against the base mean; central only where both signs exist)."""
    base_mean = {t: float(np.mean([r["J"][t] for r in fd["base"]])) if fd["base"] else None
                 for t in TERMS}
    out = {}
    for c in controls:
        hs = sorted({h for (cc, h, _s) in fd["pert"] if cc == c})
        per_t = {}
        for t in TERMS:
            per_h = {}
            for h in hs:
                p = fd["pert"].get((c, h, 1))
                m = fd["pert"].get((c, h, -1))
                e = {"J_plus": None if p is None else p["J"][t],
                     "J_minus": None if m is None else m["J"][t]}
                e["central"] = (None if p is None or m is None
                                else (p["J"][t] - m["J"][t]) / (2.0 * h))
                b = base_mean[t]
                e["plus"] = None if p is None or b is None else (p["J"][t] - b) / h
                e["minus"] = None if m is None or b is None else (b - m["J"][t]) / h
                per_h[h] = e
            per_t[t] = per_h
        out[c] = per_t
    return out


def verdict(adj, fd_by_h: dict, noise, *, h_ref: float = 1e-2, tol: float = 0.10,
            k: float = 3.0) -> dict:
    """The bar for one control × term. ``fd_by_h``: ``{h: central FD or None}``; ``noise``:
    the single-run std of the term (None = unknown: every FD then counts as resolved and the
    result says so)."""
    fds = {h: v for h, v in fd_by_h.items() if v is not None}
    resolved = {h: (noise is None or abs(v) >= k * noise / h) for h, v in fds.items()}
    ratio = {h: (adj / v if (adj is not None and v != 0.0) else None) for h, v in fds.items()}
    res = {"adjoint": adj, "fd": fds, "resolved": resolved, "ratio": ratio,
           "noise_known": noise is not None}
    if adj is None:
        res["verdict"] = "no adjoint"
    elif not any(resolved.values()):
        res["verdict"] = "unresolved"
    else:
        # the ratio is judged at the reference h, or, where that FD is noise-dominated, at the
        # smallest resolved h (the larger steps exist for exactly those controls)
        href = next((h for h in fds if math.isclose(h, h_ref, rel_tol=1e-9)), None)
        if href is None or not resolved[href]:
            href = min((h for h in fds if resolved[h]), default=None)
        ok_ref = (href is not None and ratio[href] is not None
                  and abs(ratio[href] - 1.0) <= tol)
        rs = [v for h, v in fds.items() if resolved[h]]
        ok_range = len(rs) >= 2 and min(rs) <= adj <= max(rs)
        res["verdict"] = "pass" if (ok_ref or ok_range) else "fail"
        res["ratio_h"] = href
        res["pass_by"] = ("ratio" if ok_ref else "range") if (ok_ref or ok_range) else None
    return res


# ---------------------------------------------------------------------------
# the amplification screen


def screen_trace(trace, start, chunk_seconds: float, *, spike: float = 1.5,
                 field_trace=None, top: int = 3) -> dict:
    """``trace``: norms seed first (boundary n, n−1, ..., 0). Growth, finiteness and the
    one-boundary spikes (norm > spike × max of the two neighbours) with their dates and the
    fields that rise most there (field norm / max of its neighbours)."""
    tr = np.asarray(trace, np.float64)
    n = tr.size - 1
    finite = bool(np.all(np.isfinite(tr)))
    out = {"n_boundaries": int(tr.size), "finite": finite,
           "seed": float(tr[0]), "max": float(np.nanmax(tr)) if finite else None,
           "growth": float(np.nanmax(tr) / tr[0]) if finite and tr[0] > 0 else None,
           "end_over_seed": float(tr[-1] / tr[0]) if finite and tr[0] > 0 else None,
           "spikes": []}
    for j in range(1, tr.size - 1):
        nb = max(tr[j - 1], tr[j + 1])
        if np.isfinite(tr[j]) and nb > 0 and tr[j] > spike * nb:
            b = n - j
            s = {"boundary": int(b), "date": (start + datetime.timedelta(
                seconds=b * chunk_seconds)).isoformat(), "ratio": float(tr[j] / nb),
                "norm": float(tr[j])}
            if field_trace is not None and field_trace[j] is not None:
                fr = []
                for leaf, v in field_trace[j].items():
                    a = max(field_trace[j - 1].get(leaf, 0.0) or 0.0,
                            field_trace[j + 1].get(leaf, 0.0) or 0.0)
                    if v and v > 0:
                        fr.append((leaf, float(v / a) if a > 0 else math.inf, float(v)))
                fr.sort(key=lambda x: -x[1])
                s["fields"] = [{"field": a, "ratio": r, "norm": v} for a, r, v in fr[:top]]
            out["spikes"].append(s)
    return out


def screen_eval(ev: dict, *, spike: float) -> dict:
    tr = ev["trace"]
    if tr is None:
        return {"missing": True}
    p = ev["prov"]
    start = datetime.datetime.fromisoformat(p["start"])
    chunk_s = float(p["window_days"]) * 86400.0 / float(p["n_steps"]) * float(p["chunk_steps"])
    if "traces" in tr:                       # eval --terms: one trace per cost term
        return {t: screen_trace(tr["traces"][t], start, chunk_s, spike=spike,
                                field_trace=(tr.get("field_traces") or {}).get(t))
                for t in tr["traces"]}
    return {"J_data": screen_trace(tr["trace"], start, chunk_s, spike=spike,
                                   field_trace=tr.get("field_trace"))}


# ---------------------------------------------------------------------------
# the Gauss–Newton Hessian


class ObsSpace:
    """The cost's observation-space selection: F (C̄ then H̄ at the valid super-observations of
    the months with steps), the observations o and the weights W = w/σ² on the same mask."""

    def __init__(self, ob: "seaice_cost.CostObs", have):
        have = np.asarray(have, bool)
        self.mC = np.asarray(ob.validC, bool) & have[:, None]
        self.mH = np.asarray(ob.validH, bool) & np.asarray(ob.nh, bool)[None, :] & have[:, None]
        self.o = np.concatenate([ob.C[self.mC], ob.H[self.mH]])
        self.W = np.concatenate([ob.WC[self.mC], ob.WH[self.mH]])

    def F(self, C_mean, H_mean):
        return np.concatenate([np.asarray(C_mean, np.float64)[self.mC],
                               np.asarray(H_mean, np.float64)[self.mH]])

    def J(self, F) -> float:
        r = F - self.o
        return 0.5 * math.fsum((self.W * r * r).tolist())

    def norm2(self, x) -> float:
        return math.fsum((self.W * x * x).tolist())


def read_fields(run_dir):
    with np.load(Path(run_dir) / "fields.npz") as z:
        return (np.asarray(z["C_mean"], np.float64), np.asarray(z["H_mean"], np.float64),
                np.asarray(z["n_mon"], np.float64))


def hessian_h(fd: dict, c: str) -> float | None:
    """The largest h with both signs for control c."""
    hs = [h for (cc, h, s) in fd["pert"] if cc == c and s == 1 and (c, h, -1) in fd["pert"]]
    return max(hs) if hs else None


def gauss_newton(cols: dict, W, sigma: dict, hs: dict, s2: float | None, controls: list, *,
                 ident_abs: float = 10.0, ident_rel: float = 1e-3) -> dict:
    """``cols``: ``{control: J_i}`` (observation-space columns in ln p); ``hs``: the h of each
    column; ``s2``: one run's expected noise ``‖n‖²_W`` (None: no correction). Returns H_lnp
    raw and noise-corrected, H_u, the spectra and eigenvectors."""
    names = [c for c in controls if c in cols]
    k = len(names)
    Jm = np.stack([cols[c] for c in names], axis=1) if k else np.zeros((len(W), 0))
    WJ = Jm * np.asarray(W)[:, None]
    H_raw = Jm.T @ WJ
    corr = np.array([0.0 if s2 is None else s2 / (2.0 * hs[c] ** 2) for c in names])
    H_cor = H_raw - np.diag(corr)
    S = np.diag([sigma[c] for c in names])
    res = {"controls": names, "h": [hs[c] for c in names], "noise_s2": s2,
           "diag_noise": corr.tolist(), "H_lnp_raw": H_raw.tolist(),
           "H_lnp": H_cor.tolist()}
    for tag, H in (("raw", S @ H_raw @ S), ("corrected", S @ H_cor @ S)):
        lam, V = np.linalg.eigh(0.5 * (H + H.T))
        order = np.argsort(lam)[::-1]
        lam, V = lam[order], V[:, order]
        lmax = lam[0] if k else 0.0
        res[f"H_u_{tag}"] = H.tolist()
        res[f"spectrum_{tag}"] = [{
            "eigenvalue": float(lam[i]),
            "relative": float(lam[i] / lmax) if lmax > 0 else None,
            "identifiable": bool(lam[i] >= ident_abs and lmax > 0
                                 and lam[i] >= ident_rel * lmax),
            "vector": {names[j]: float(V[j, i]) for j in range(k)}} for i in range(k)]
    return res


# ---------------------------------------------------------------------------
# the parity verdict


def parity_verdict(parity: dict, fd: dict, controls: list, sigma: dict, hs: dict, mesh_dir, *,
                   frac: float = 0.1) -> dict:
    import parity as PA
    import score as SC
    _, lat, wet, area = SC.mesh_nodes(mesh_dir)
    hems = {"NH": wet & (lat > 0), "SH": wet & (lat < 0)}
    months = [m - 1 for m in parity["months"]]
    largest = {}
    for hem in hems:
        rows = parity["monthly"][hem]
        largest[hem] = {q: max(abs(r[q]) for r in rows if r[q] is not None)
                        for q in ("area_diff", "volume_diff", "rms_C", "rms_H")}
    out = {"parity_largest": largest, "controls": {}, "frac": frac}
    all_ok = True
    for c in controls:
        h = hs.get(c)
        if h is None:
            continue
        Cp, Hp, _ = read_fields(fd["pert"][(c, h, 1)]["dir"])
        Cm, Hm, _ = read_fields(fd["pert"][(c, h, -1)]["dir"])
        scale = sigma[c] / (2.0 * h)            # (X(+h) − X(−h)) / 2h per ln p, times σ
        per_hem = {}
        for hem, sel in hems.items():
            resp = {"area_diff": 0.0, "volume_diff": 0.0, "rms_C": 0.0, "rms_H": 0.0}
            for m in months:
                tp = PA.hemisphere_totals(Cp[m], Hp[m], area, sel)
                tm = PA.hemisphere_totals(Cm[m], Hm[m], area, sel)
                resp["area_diff"] = max(resp["area_diff"], abs(tp["area"] - tm["area"]) * scale)
                resp["volume_diff"] = max(resp["volume_diff"],
                                          abs(tp["volume"] - tm["volume"]) * scale)
                dom = sel & ((Cp[m] > 0) | (Cm[m] > 0))
                if dom.any():
                    w = area[dom] / np.sum(area[dom])
                    dC = (Cp[m] - Cm[m])[dom] * scale
                    dH = (Hp[m] - Hm[m])[dom] * scale
                    resp["rms_C"] = max(resp["rms_C"], float(np.sqrt(np.sum(w * dC * dC))))
                    resp["rms_H"] = max(resp["rms_H"], float(np.sqrt(np.sum(w * dH * dH))))
            ratio = {q: (largest[hem][q] / v if v > 0 else math.inf) for q, v in resp.items()}
            per_hem[hem] = {"response_1sigma": resp, "ratio": ratio}
            all_ok &= all(r <= frac for r in ratio.values())
        out["controls"][c] = per_hem
    out["negligible_for_every_control"] = bool(all_ok)
    return out


# ---------------------------------------------------------------------------
# driver


def summarize(window, *, fd_dir=None, eval_paths=None, obs=None, parity=None, mesh_dir=None,
              h_ref=1e-2, tol=0.10, k_noise=3.0, spike=1.5, ident_abs=10.0, ident_rel=1e-3,
              parity_frac=0.1, j_rtol=1e-9) -> dict:
    window = Path(window)
    fd = load_fd(fd_dir or window / "fd")
    if eval_paths is None:
        eval_paths = sorted(glob.glob(str(window / "eval*" / "eval.json")))
    evals = load_evals(eval_paths)
    runs = fd["base"] + list(fd["pert"].values()) + evals
    if not runs:
        raise ValueError(f"{window}: no FD runs and no evaluations")
    win = check_window(runs)
    p0 = runs[0]["prov"]
    controls = [_short(c) for c in p0["controls"]]
    sigma = dict(zip(controls, [float(s) for s in p0["sigma"]]))

    noise = noise_floor(fd["base"])
    fds = fd_estimates(fd, controls)
    bar = {}
    for c in controls:
        bar[c] = {}
        for t in TERMS:
            adjs = [ev["g_lnp"][t][controls.index(c)] for ev in evals if t in ev["g_lnp"]]
            adj = float(adjs[0]) if adjs else None
            v = verdict(adj, {h: e["central"] for h, e in fds[c][t].items()}, noise[t],
                        h_ref=h_ref, tol=tol, k=k_noise)
            v["one_sided"] = {h: {"plus": e["plus"], "minus": e["minus"]}
                              for h, e in fds[c][t].items()}
            if len(adjs) > 1:
                v["adjoint_all_evals"] = [float(a) for a in adjs]
            bar[c][t] = v

    # observation space: every run's J_data recomputed from its fields
    obs_path = obs or p0["obs"]
    with np.load(obs_path) as z:
        ob = seaice_cost.cost_obs({kk: np.asarray(z[kk]) for kk in z.files})
    ref_have = None
    space = None
    Fs = {}
    worst = 0.0
    for r in fd["base"] + list(fd["pert"].values()):
        Cm, Hm, n_mon = read_fields(r["dir"])
        have = n_mon > 0
        if ref_have is None:
            ref_have = have
            space = ObsSpace(ob, have)
        elif not np.array_equal(have, ref_have):
            raise ValueError(f"{r['dir']}: months with steps differ from the other runs")
        F = space.F(Cm, Hm)
        Jre = space.J(F)
        rel = abs(Jre - r["J"]["J_data"]) / max(abs(r["J"]["J_data"]), 1e-300)
        worst = max(worst, rel)
        if rel > j_rtol:
            raise ValueError(f"{r['dir']}: J_data from fields.npz {Jre!r} != fwd.json "
                             f"{r['J']['J_data']!r} (rel {rel:.3e}): F/W are not the cost's")
        Fs[r["tag"]] = F
    hess = None
    if space is not None:
        base_F = [Fs[r["tag"]] for r in fd["base"]]
        s2 = None
        if len(base_F) >= 2:
            Fbar = np.mean(base_F, axis=0)
            s2 = math.fsum(space.norm2(F - Fbar) for F in base_F) / (len(base_F) - 1)
        hs = {c: hessian_h(fd, c) for c in controls}
        hs = {c: h for c, h in hs.items() if h is not None}
        cols = {c: (Fs[fd["pert"][(c, h, 1)]["tag"]] - Fs[fd["pert"][(c, h, -1)]["tag"]])
                / (2.0 * h) for c, h in hs.items()}
        hess = gauss_newton(cols, space.W, sigma, hs, s2, controls, ident_abs=ident_abs,
                            ident_rel=ident_rel)
        hess["n_obs"] = int(space.W.size)
        hess["base_noise_norm2"] = [space.norm2(F - np.mean(base_F, axis=0)) for F in base_F] \
            if len(base_F) >= 2 else None
        par = None
        if parity is not None:
            with open(parity) as f:
                par = parity_verdict(json.load(f), fd, controls, sigma, hs,
                                     mesh_dir or ROOT / "data" / "mesh_core2", frac=parity_frac)
    else:
        par = None

    return {
        "window": str(window), "window_keys": win, "controls": controls, "sigma": sigma,
        "rules": {"h_ref": h_ref, "tol": tol, "k_noise": k_noise, "spike": spike,
                  "ident_abs": ident_abs, "ident_rel": ident_rel, "parity_frac": parity_frac},
        "base": [{"tag": r["tag"], "J": r["J"]} for r in fd["base"]],
        "noise_J": noise, "fd_noise_floor": {
            t: (None if noise[t] is None else
                {h: k_noise * noise[t] / h for h in sorted({h for (_, h, _) in fd["pert"]})})
            for t in TERMS},
        "evals": [{"dir": ev["dir"], "J": ev["J"], "terms_with_gradient": sorted(ev["g_lnp"]),
                   "linear_check": ev["linear_check"]} for ev in evals],
        "bar": bar, "screen": {ev["dir"]: screen_eval(ev, spike=spike) for ev in evals},
        "J_recompute_max_rel": worst, "hessian": hess, "parity": par,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _fmt(x, w=10):
    return " " * (w - 1) + "-" if x is None else f"{x:{w}.4g}"


def print_table(res: dict) -> None:
    print(f"window {res['window']}   noise_J (single-run std): "
          + ", ".join(f"{t} {'-' if v is None else f'{v:.3g}'}" for t, v in res["noise_J"].items()))
    for t in TERMS:
        rows = [(c, res["bar"][c][t]) for c in res["controls"]
                if res["bar"][c][t]["fd"] or res["bar"][c][t]["adjoint"] is not None]
        if not rows:
            continue
        print(f"\n{t}: control      adjoint     FD(h)...                 adj/FD(h)...      verdict")
        for c, v in rows:
            fdt = " ".join(f"{h:g}:{_fmt(f, 9).strip()}{'' if v['resolved'][h] else '*'}"
                           for h, f in sorted(v["fd"].items()))
            rt = " ".join(f"{h:g}:{'-' if r is None else f'{r:.3f}'}"
                          for h, r in sorted(v["ratio"].items()))
            print(f"   {c:12s} {_fmt(v['adjoint'])}   {fdt:28s} {rt:22s} {v['verdict']}")
    print("   (* = FD below k·noise/h, unresolved)")
    h = res.get("hessian")
    if h:
        print("\nH_u spectrum (noise-corrected): "
              + ", ".join(f"{s['eigenvalue']:.3g}{'' if s['identifiable'] else '(n)'}"
                          for s in h["spectrum_corrected"]))
    for d, s in res["screen"].items():
        for t, sc in s.items():
            if isinstance(sc, dict) and "spikes" in sc:
                print(f"screen {Path(d).name} {t}: growth {sc['growth']}, "
                      f"{len(sc['spikes'])} spike(s)"
                      + "".join(f"; {x['date'][:10]} x{x['ratio']:.2f}" for x in sc["spikes"][:5]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fd", type=Path, default=None, help="FD root (default <window>/fd)")
    ap.add_argument("--eval", action="append", default=None,
                    help="eval.json (repeatable; default <window>/eval*/eval.json)")
    ap.add_argument("--obs", default=None, help="observation file (default: the runs' own)")
    ap.add_argument("--parity", default=None, help="parity.json of parity.py")
    ap.add_argument("--mesh-dir", type=Path, default=ROOT / "data" / "mesh_core2")
    ap.add_argument("--h-ref", type=float, default=1e-2)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--k-noise", type=float, default=3.0)
    ap.add_argument("--spike", type=float, default=1.5)
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"refusing to write into an existing directory: {a.out}", file=sys.stderr)
        return 3
    res = summarize(a.window, fd_dir=a.fd, eval_paths=a.eval, obs=a.obs, parity=a.parity,
                    mesh_dir=a.mesh_dir, h_ref=a.h_ref, tol=a.tol, k_noise=a.k_noise,
                    spike=a.spike)
    res["argv"] = sys.argv if argv is None else list(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(a.out)
    with open(a.out / "s1sum.json", "x") as f:
        json.dump(_jsonable(res), f, indent=1)
    print_table(res)
    print(f"\nwrote {a.out}/s1sum.json")
    return 0


def _jsonable(x):
    if isinstance(x, dict):
        return {(f"{k:g}" if isinstance(k, float) else str(k)): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


if __name__ == "__main__":
    sys.exit(main())
