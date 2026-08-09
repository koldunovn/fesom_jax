"""κ̂ from the eddy flux DIVERGENCE instead of from the flux itself (plan M3b).

**The question.** Projecting the teacher's eddy flux onto ``-κ∇σ`` per column keeps only 12.5 %
of wet layer cells, puts the median κ̂/κ_formula at 0.070, leaves 40.7 % of what it keeps
pointing UP-gradient, and fits the [1/3, 3] multiplier bound in 25.8 %. The standing explanation
is a large ROTATIONAL component in the diagnosed flux — a part of ``F`` that moves no tracer.
That explanation has never been measured; it was inferred from the residuals it would explain.

**Why the divergence settles it.** A rotational flux has ``∇·F ≡ 0``. So fitting κ against the
divergence removes that component EXACTLY rather than approximately, and the comparison between
the two fits, on the same cells, is the measurement:

  * if the divergence fit keeps far more cells, loses the up-gradient population and tightens
    around the formula, the rotational reading was right and the bounded-multiplier arm survives;
  * if it behaves the same or worse, the rotational story is not what is wrong and the GM/Redi
    ansatz itself is the problem — evidence for the unbounded arm.

**The estimator.** With κ locally constant, ``∇·(-κ∇σ) = -κ∇²σ``, so

    κ̂_div = -(∇·F_σ) / ∇²σ

per layer cell. That is the same locality assumption the flux fit already makes (it solves
``F = -κ∇σ`` pointwise), so the two are comparable; what differs is only that the rotational
part of ``F`` is annihilated on the left. Both the numerator and the denominator go through the
SAME discrete operator (:mod:`flux_divergence`, itself gated bit-identical against the model's
own ``gm_redi.diff_part_hor_redi``), so their discretisation errors partly cancel.

⚠️ **PARTIAL CONSTRAINT — say so wherever these numbers appear.** Only the HORIZONTAL divergence
exists here: the coarse products carry ``uT_eddy``/``vT_eddy`` but no ``wT_eddy``, so the
vertical eddy flux convergence is missing and any κ fitted against this absorbs its omission.

⚠️ **AVERAGE FIRST, THEN DIFFERENTIATE.** Differentiating amplifies noise. The six 5-day marks
in a month are effectively independent (5 days ≫ the eddy decorrelation time), so averaging them
cuts the random rotational part by ~√6 at no cost to the teacher signal. This module averages
the marks and derives ONE state from the averaged T and S — it never averages per-mark κ̂, which
is a different and worse estimator (a ratio of means is not the mean of ratios, and the
per-mark denominators go through zero).

Usage:
    python scripts/nn_eddy/divergence_fit.py --months 1958_01 1958_02 ...
    python scripts/nn_eddy/divergence_fit.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fesom_jax  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402

if ROOT not in Path(fesom_jax.__file__).resolve().parents:
    raise SystemExit(f"FATAL: fesom_jax from {fesom_jax.__file__}, not {ROOT}")

from fesom_jax import ale  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.state import State  # noqa: E402

import features_from_product as _fp  # noqa: E402
import flux_divergence as _fd  # noqa: E402
import kappa_hat as _kh  # noqa: E402
import nn_data as _nd  # noqa: E402

FLUX_VARS = ("uT_eddy", "vT_eddy", "uS_eddy", "vS_eddy")
MEAN_VARS = ("u", "v")


def month_mean(path: Path, nl: int):
    """Time-mean T, S and eddy fluxes over EVERY mark in the month, plus the mark count.

    ``have`` is the AND over marks, not the OR: a cell that the coarse-graining could not fill
    at one mark has no business contributing a mean, and a mean over a varying subset of marks
    would put a time-varying weight into a field we are about to differentiate.
    """
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        n = ds.dimensions["time"].size
        have = np.ones(ds.variables["temp"].shape[1:], bool)
        acc = {k: 0.0 for k in ("temp", "salt") + FLUX_VARS + MEAN_VARS}
        for r in range(n):
            have &= np.asarray(ds.variables["cell_area"][r]) > 0.0
            for k in acc:
                acc[k] = acc[k] + np.asarray(ds.variables[k][r], np.float64)
    for k in acc:
        acc[k] = acc[k] / n
    T47, S47 = acc["temp"], acc["salt"]
    F_T = np.stack([_fp.expand_levels(acc["uT_eddy"], nl)[:, :nl - 1],
                    _fp.expand_levels(acc["vT_eddy"], nl)[:, :nl - 1]], axis=-1)
    F_S = np.stack([_fp.expand_levels(acc["uS_eddy"], nl)[:, :nl - 1],
                    _fp.expand_levels(acc["vS_eddy"], nl)[:, :nl - 1]], axis=-1)
    Umean = np.stack([_fp.expand_levels(acc["u"], nl)[:, :nl - 1],
                      _fp.expand_levels(acc["v"], nl)[:, :nl - 1]], axis=-1)
    return T47, S47, have, jnp.asarray(F_T), jnp.asarray(F_S), Umean, n


def spice_flux(d: _fp.Derived, F_T, F_S, nl: int, gm_cfg=_fp.GM_CFG):
    """The eddy flux OF SPICE, ``F_χ = ρ₀(+α F_T + β F_S)`` — the Redi arm's tracer.

    The projection fits κ̂_GM against the DENSITY flux and κ̂_Redi against the SPICE flux, so a
    conclusion drawn from density alone covers only half of Arm A. `kappa_hat.project` builds
    ``∇χ`` by calling ``compute_sigma_xy(mesh, -T, S, …)`` — linear in T, so negating T flips
    the sign of the α term and leaves β alone. Same trick here, applied to the flux.
    """
    a = d.sw_alpha[:, :nl - 1, None]
    b = d.sw_beta[:, :nl - 1, None]
    return gm_cfg.rho_ref * (a * F_T + b * F_S)


def density_flux(d: _fp.Derived, F_T, F_S, nl: int, gm_cfg=_fp.GM_CFG):
    """The eddy flux OF DENSITY, ``F_σ = ρ₀(-α F_T + β F_S)``, at layer centres ``(N, nl-1, 2)``.

    Same combination ``gm.compute_sigma_xy`` uses to build ``∇σ`` from ``∇T`` and ``∇S``, so the
    numerator and the denominator of the fit describe the same σ. α and β are already LAYER
    fields, so they are trimmed, never averaged across layers.
    """
    a = d.sw_alpha[:, :nl - 1, None]
    b = d.sw_beta[:, :nl - 1, None]
    return gm_cfg.rho_ref * (-a * F_T + b * F_S)


def fit(mesh, d: _fp.Derived, F_T, F_S, *, lap_floor: float, gm_cfg=_fp.GM_CFG):
    """``κ̂_div`` per layer cell, with the pieces the caller needs to judge it."""
    nl = mesh.nl
    hnode = State.rest(mesh).hnode
    _, helem = ale.commit_thickness(mesh, hnode)

    Fs = density_flux(d, F_T, F_S, nl, gm_cfg)
    divF = _fd.divergence_of_node_flux(mesh, Fs[..., 0], Fs[..., 1], hnode, helem)[:, :nl - 1]

    # ∇²σ through the SAME operator applied to ∇σ — not a separate Laplacian. Two different
    # discretisations in the numerator and the denominator would put their mismatch straight
    # into κ̂, and the mismatch is largest exactly where the guard is already marginal.
    sig = d.sigma_xy[:, :nl - 1, :]
    lap = _fd.divergence_of_node_flux(mesh, sig[..., 0], sig[..., 1], hnode, helem)[:, :nl - 1]

    divF, lap = np.asarray(divF), np.asarray(lap)
    ok = np.asarray(d.usable, bool) & (np.abs(lap) > lap_floor)
    kappa = np.where(ok, -divF / np.where(ok, lap, 1.0), np.nan)
    return dict(kappa=kappa, ok=ok, div_F=divF, lap_sigma=lap)


def divergent_fraction(mesh, d: _fp.Derived, F_T, F_S, *, gm_cfg=_fp.GM_CFG):
    """How DIVERGENT is the teacher's eddy flux, measured against a down-gradient flux?

    This is the test of the rotational-flux inference itself, and it needs no κ estimator, so
    none of the estimator's own weaknesses (the dropped ``∇κ·∇σ``, the noise in ``∇²σ``) enter.

    For any flux varying on a length scale ``L``, ``|∇·F| ~ |F|/L``, so ``η = |∇·F|·L/|F|`` is
    a dimensionless measure of how much of the field actually carries a divergence: order 1 for
    a flux that is essentially all divergent, and ≪1 for one that is mostly rotational or
    cancels at the grid scale. ``L`` is taken as ``√(dual-cell area)``, the local mesh scale.

    The number alone means little — the discrete operator has its own damping at the grid scale.
    So it is measured for the teacher's flux AND for the reference down-gradient flux
    ``G = -κ_formula ∇σ``, through the SAME operator on the SAME cells. Their RATIO is the
    result: near 1 means the teacher's flux is as divergent as a down-gradient one, and the
    rotational explanation has nothing to explain; ≪1 means a large part of the teacher's flux
    genuinely moves no tracer.
    """
    nl = mesh.nl
    hnode = State.rest(mesh).hnode
    _, helem = ale.commit_thickness(mesh, hnode)
    L = np.sqrt(np.asarray(mesh.areasvol)[:, :nl - 1])

    Fs = density_flux(d, F_T, F_S, nl, gm_cfg)
    kap = np.asarray(d.fer_K_lay)[:, :, None]
    Gs = -kap * np.asarray(d.sigma_xy[:, :nl - 1, :])          # the down-gradient reference

    def eta(vec):
        div = np.asarray(_fd.divergence_of_node_flux(
            mesh, jnp.asarray(vec[..., 0]), jnp.asarray(vec[..., 1]), hnode, helem))[:, :nl - 1]
        mag = np.linalg.norm(np.asarray(vec), axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.abs(div) * L / mag, mag

    e_t, mag_t = eta(np.asarray(Fs))
    e_g, mag_g = eta(Gs)
    use = np.asarray(d.usable, bool) & (mag_t > 0) & (mag_g > 0) & np.isfinite(e_t) & np.isfinite(e_g)
    if not use.any():
        return dict(n=0)
    return dict(n=int(use.sum()),
                eta_teacher=float(np.median(e_t[use])),
                eta_downgrad=float(np.median(e_g[use])),
                ratio=float(np.median(e_t[use]) / np.median(e_g[use])),
                ratio_p25=float(np.percentile(e_t[use] / e_g[use], 25)),
                ratio_p75=float(np.percentile(e_t[use] / e_g[use], 75)))


def alignment(mesh, d: _fp.Derived, F_T, F_S, Umean, *, gm_cfg=_fp.GM_CFG):
    """What is the eddy density flux actually ALIGNED with?

    This is the follow-up the divergent-fraction result forces. If the teacher's flux carries
    nearly as much divergence as a down-gradient flux (η ratio ≈ 1) and yet the projection onto
    ``-∇σ`` fails, then the flux genuinely moves tracer — it just does not move it down the mean
    density gradient. Two directions worth testing it against, both free here:

      * ``-∇σ``: the GM/Redi ansatz's assumption. ``cos θ = +1`` is perfectly down-gradient,
        ``-1`` perfectly up-gradient, ``0`` no relationship. The flux fit's 40 % up-gradient
        population says this distribution is broad; this measures it directly, without a
        conditioning guard in the way.
      * the mean velocity ``(u, v)``: an eddy flux that is really the mean flow advecting eddy
        variance would lie along the flow rather than along the gradient. FESOM2 carries u,v on
        elements but the coarse product puts them on nodes with the tracers, so this is a
        node-to-node comparison with no interpolation between staggerings.

    Reported as medians of ``|cos|`` and of the signed ``cos`` over usable cells.
    """
    nl = mesh.nl
    Fs = np.asarray(density_flux(d, F_T, F_S, nl, gm_cfg))
    G = -np.asarray(d.sigma_xy[:, :nl - 1, :])
    U = np.asarray(Umean)
    use = np.asarray(d.usable, bool)

    def cos_with(A, B):
        na = np.linalg.norm(A, axis=-1)
        nb = np.linalg.norm(B, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.sum(A * B, axis=-1) / (na * nb)
        m = use & np.isfinite(c) & (na > 0) & (nb > 0)
        if not m.any():
            return None
        v = c[m]
        return dict(n=int(m.sum()), median=float(np.median(v)),
                    median_abs=float(np.median(np.abs(v))),
                    frac_downgrad=float(np.mean(v > 0)))

    # ⚠️ WITHOUT THIS ROW THE FLOW ALIGNMENT CANNOT BE READ. Thermal wind puts the mean flow
    # nearly PERPENDICULAR to the mean density gradient, so "the eddy flux lies along the flow"
    # and "the eddy flux lies across ∇σ" would be the same statement, and quoting both as
    # separate evidence would be double counting. This measures the degeneracy directly: if
    # |cos(U, ∇σ)| is near 0 the two directions are interchangeable and only ONE of them may be
    # claimed; if it is appreciable, the flow alignment is independent information.
    return dict(vs_minus_grad_sigma=cos_with(Fs, G), vs_mean_flow=cos_with(Fs, U),
                flow_vs_grad_sigma=cos_with(U, G))


def _stats(kappa, ok, base):
    """The four numbers the two fits are compared on, over the cells each one keeps."""
    m = ok & (base > 0) & np.isfinite(kappa)
    if not m.any():
        return dict(n=0)
    r = kappa[m] / base[m]
    return dict(n=int(m.sum()),
                frac_kept=float(m.sum()),
                median_ratio=float(np.median(r)),
                frac_negative=float(np.mean(r < 0)),
                frac_in_bound=float(np.mean((r >= 1.0 / 3.0) & (r <= 3.0))),
                p10=float(np.percentile(r, 10)), p90=float(np.percentile(r, 90)))


def run_month(mesh, month: str, coarse_dir: Path, *, lap_floor: float, verbose=True):
    """Both fits on one month's time mean, on the same cells."""
    nl = mesh.nl
    path = Path(coarse_dir) / f"coarse_ng5_{month}.nc"
    T47, S47, have47, F_T, F_S, Umean, n_marks = month_mean(path, nl)
    d = _fp.derive(mesh, T47, S47, have47, verbose=False)

    fd_ = fit(mesh, d, F_T, F_S, lap_floor=lap_floor)
    kh = _kh.project(mesh, d, F_T, F_S, verbose=False)

    usable = np.asarray(d.usable, bool)
    ok_flux = np.asarray(kh.ok_gm, bool)
    base_gm = np.asarray(d.fer_K_lay)
    kap_div, lap = fd_["kappa"], fd_["lap_sigma"]

    # ⚠️ THE ONLY LIKE-FOR-LIKE COMPARISON IS ON THE SAME CELLS. The flux fit applies a real
    # guard (conditioning, adequacy, and the b_sigma decoupling test) and keeps ~12 %; the
    # divergence fit as posed has no guard at all and would "keep" 100 %. Reporting those two
    # side by side compares an estimator against a different POPULATION and says nothing about
    # either. So the headline row restricts the divergence fit to the cells the flux fit kept —
    # then the only thing that differs between the two columns is the estimator.
    kap_same = np.where(ok_flux, -np.asarray(fd_["div_F"]) / np.where(np.abs(lap) > 0, lap, np.nan),
                        np.nan)
    out = dict(month=month, n_marks=n_marks, n_usable=int(usable.sum()),
               flux=_stats(np.asarray(kh.kappa_gm), ok_flux, base_gm),
               div_same_cells=_stats(kap_same, ok_flux & np.isfinite(kap_same), base_gm),
               div_all_usable=_stats(kap_div, fd_["ok"], base_gm))
    for k in ("flux", "div_same_cells", "div_all_usable"):
        if out[k].get("n"):
            out[k]["frac_kept"] = out[k]["n"] / max(int(usable.sum()), 1)

    # A FLOOR SWEEP, not a chosen floor. If |∇²σ| is mostly noise, the up-gradient fraction sits
    # near 50 % (a ratio of two noisy numbers has a random sign) and does not move as the floor
    # rises. If there is signal under the noise, raising the floor must drive it down. That
    # trend, not any single row, is what says whether the divergence carries information here.
    lap_abs = np.abs(lap)
    sweep = []
    for p in (0, 50, 75, 90, 95, 99):
        thr = float(np.percentile(lap_abs[usable], p)) if p else 0.0
        sel = usable & (lap_abs > thr)
        s = _stats(np.where(sel, kap_div, np.nan), sel & np.isfinite(kap_div), base_gm)
        s["pctl"], s["floor"] = p, thr
        sweep.append(s)
    out["floor_sweep"] = sweep
    out["divergent"] = divergent_fraction(mesh, d, F_T, F_S)
    out["alignment"] = alignment(mesh, d, F_T, F_S, Umean)

    # The REDI arm. Everything above is the density flux, i.e. the GM half of Arm A; the
    # projection fits κ̂_Redi against the SPICE flux, and a conclusion about bounded multipliers
    # has to cover both. Same two measurements, same operator, spice in place of density and the
    # isoneutral diffusivity Ki in place of fer_K.
    from fesom_jax import gm as _gm
    chi_grad = np.asarray(_gm.compute_sigma_xy(mesh, -d.T, d.S, d.sw_alpha, d.sw_beta,
                                               _fp.GM_CFG))[:, :nl - 1, :]
    Fchi = np.asarray(spice_flux(d, F_T, F_S, nl))
    Ki_lay = np.asarray(d.Ki)[:, :nl - 1]
    hnode = State.rest(mesh).hnode
    _, helem = ale.commit_thickness(mesh, hnode)
    L = np.sqrt(np.asarray(mesh.areasvol)[:, :nl - 1])

    def _eta(vec):
        div = np.asarray(_fd.divergence_of_node_flux(
            mesh, jnp.asarray(vec[..., 0]), jnp.asarray(vec[..., 1]), hnode, helem))[:, :nl - 1]
        mag = np.linalg.norm(vec, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.abs(div) * L / mag, mag

    e_t, mag_t = _eta(Fchi)
    e_g, mag_g = _eta(-Ki_lay[..., None] * chi_grad)
    sel = usable & (mag_t > 0) & (mag_g > 0) & np.isfinite(e_t) & np.isfinite(e_g)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.sum(Fchi * (-chi_grad), axis=-1) / (
            np.linalg.norm(Fchi, axis=-1) * np.linalg.norm(chi_grad, axis=-1))
    csel = usable & np.isfinite(c)
    out["redi"] = dict(
        n=int(sel.sum()),
        eta_teacher=float(np.median(e_t[sel])) if sel.any() else None,
        eta_downgrad=float(np.median(e_g[sel])) if sel.any() else None,
        ratio=float(np.median(e_t[sel]) / np.median(e_g[sel])) if sel.any() else None,
        cos_median=float(np.median(c[csel])) if csel.any() else None,
        cos_median_abs=float(np.median(np.abs(c[csel]))) if csel.any() else None,
        frac_downgrad=float(np.mean(c[csel] > 0)) if csel.any() else None)
    out["lap_abs_pctl"] = {f"p{p}": float(np.percentile(lap_abs[usable], p))
                           for p in (1, 10, 25, 50, 75, 90)}
    if verbose:
        f, v = out["flux"], out["div_same_cells"]
        print(f"  {month}  marks={n_marks}  usable={out['n_usable']:>8d}  "
              f"| FLUX med {f.get('median_ratio', float('nan')):7.3f} up-grad "
              f"{f.get('frac_negative', float('nan')):5.1%} in[1/3,3] {f.get('frac_in_bound', float('nan')):5.1%} "
              f"(kept {f.get('frac_kept', 0):5.1%})  "
              f"| DIV same cells med {v.get('median_ratio', float('nan')):7.3f} up-grad "
              f"{v.get('frac_negative', float('nan')):5.1%} in[1/3,3] {v.get('frac_in_bound', float('nan')):5.1%}",
              flush=True)
        print("      floor sweep on |∇²σ| (all usable cells): " + "  ".join(
            f"p{s['pctl']:02d}:up-grad {s.get('frac_negative', float('nan')):.0%}" for s in sweep),
            flush=True)
        dv = out["divergent"]
        if dv.get("n"):
            print(f"      divergent fraction: teacher η={dv['eta_teacher']:.4f}, "
                  f"down-gradient reference η={dv['eta_downgrad']:.4f}, "
                  f"ratio {dv['ratio']:.3f} (p25 {dv['ratio_p25']:.3f}, p75 {dv['ratio_p75']:.3f})",
                  flush=True)
        al = out["alignment"]
        g, u = al.get("vs_minus_grad_sigma"), al.get("vs_mean_flow")
        fg = al.get("flow_vs_grad_sigma")
        if g and u and fg:
            print(f"      alignment of F_sigma: with -grad(sigma) cos med {g['median']:+.3f} "
                  f"(|cos| {g['median_abs']:.3f}, down-gradient {g['frac_downgrad']:.1%})   "
                  f"with mean flow cos med {u['median']:+.3f} (|cos| {u['median_abs']:.3f})   "
                  f"[degeneracy check: |cos(U, -grad sigma)| {fg['median_abs']:.3f}]",
                  flush=True)
        rd = out["redi"]
        if rd.get("n"):
            print(f"      REDI arm (spice): eta ratio {rd['ratio']:.3f}   cos(F_chi, -grad chi) "
                  f"med {rd['cos_median']:+.3f} (|cos| {rd['cos_median_abs']:.3f}, "
                  f"down-gradient {rd['frac_downgrad']:.1%})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="every coarse_ng5_*.nc in --coarse-dir")
    ap.add_argument("--coarse-dir", type=Path, default=_nd.COARSE_DIR)
    ap.add_argument("--lap-floor", type=float, default=0.0,
                    help="reject a cell whose |∇²σ| is at or below this. Default 0 = keep every "
                         "usable cell, which is deliberate: the floor must be FROZEN from the "
                         "measured distribution this run prints, not guessed before it.")
    ap.add_argument("--results", type=Path,
                    default=Path("/work/ba1583/a270088/nn_eddy/m3b_divergence_fit.jsonl"))
    args = ap.parse_args()

    months = args.months or []
    if args.all:
        months = sorted(p.name[len("coarse_ng5_"):-3]
                        for p in Path(args.coarse_dir).glob("coarse_ng5_*.nc"))
    if not months:
        raise SystemExit("nothing to do: pass --months or --all")

    t0 = time.time()
    mesh = load_mesh(_fp.MESH_DIR)
    print(f"[setup] mesh {mesh.nod2D} nodes x {mesh.nl} levels; {len(months)} month(s); "
          f"lap_floor={args.lap_floor:g}", flush=True)
    print("  month     marks    usable      the FLUX fit                          "
          "   the DIVERGENCE fit", flush=True)
    rows = []
    for m in months:
        try:
            rows.append(run_month(mesh, m, args.coarse_dir, lap_floor=args.lap_floor))
        except Exception as e:                      # one bad month must not lose the rest
            print(f"  {m}  FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)

    if rows:
        print(f"\n=== pooled over {len(rows)} month(s) ===")
        for key, lab in (("flux", "flux fit"), ("div_same_cells", "div, same cells"), ("div_all_usable", "div, all usable")):
            kept = [r[key]["frac_kept"] for r in rows if r[key].get("n")]
            med = [r[key]["median_ratio"] for r in rows if r[key].get("n")]
            neg = [r[key]["frac_negative"] for r in rows if r[key].get("n")]
            inb = [r[key]["frac_in_bound"] for r in rows if r[key].get("n")]
            if kept:
                print(f"  {lab:16s} kept {np.mean(kept):6.1%}  median κ̂/κ_formula "
                      f"{np.median(med):7.3f}  up-gradient {np.mean(neg):5.1%}  "
                      f"inside [1/3,3] {np.mean(inb):5.1%}")
        print("\n⚠️ HORIZONTAL divergence only — the products carry no wT_eddy, so the vertical "
              "eddy flux convergence is missing and κ̂_div absorbs it.")
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"[out] {len(rows)} row(s) -> {args.results}")
    print(f"[done] {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
