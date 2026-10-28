"""Mode study of the mEVP tangent: which mode makes the sea-ice adjoint blow up, and which
Newtonian damping of the tangent stress equations (Panteleev et al. 2020, The Cryosphere 14,
4427-4451, doi:10.5194/tc-14-4427-2020) tames it while keeping the response to the ice
parameters.

The map studied is one ice step of the mEVP momentum solver on the CORE2 reference model,
``x = (u, v, σ11, σ12, σ22) -> 120 sweeps of ice_mevp.mevp_iterate``, with the anchor of the
backward-Euler rhs = the entry velocity and the wind stress on the ice recomputed from the entry
velocity (``forcing.atm_ice_stress``), exactly as ``ice_step.ice_surface_step`` steps it. The
ice state (a, m, m_snow), the ocean surface velocity / SSH and the atmosphere are held at their
captured values. The inputs are CAPTURED from one real model step (``core2_adj.build_model`` +
``step.step``), and the script checks that its own 120 sweeps reproduce the step's ``u_ice``.

Measurements (``run``), per variant of the tangent (rows of the decision table):
  * the step map: top singular value (power iteration JVP+VJP) and dominant eigenvalues
    (Arnoldi) in a scaled norm (u, v by the ice-node RMS speed, σ by the ice-element RMS stress);
  * the single-sweep map at sweeps 1/30/60/120: dominant eigenvalues (power-filtered Arnoldi),
    type (oscillating = negative real, steady = positive real, complex pair) and where the
    dominant vector sits (thickness, concentration, divergence, Δ/δmin, the local mEVP
    stability number γ against αβ, lat/lon, ice-mask edge, coast);
  * retained response, one step: r_c = (dD/dc from the variant's tangent) / (central FD of D
    over the nonlinear 120-sweep map) for c in {ln P*, ln C_d}, at h = 1e-3 and 1e-2, and the
    weighted pattern correlation of the tangent and FD (δu, δv) fields; D = Σ w a |u| / Σ w a
    over ice nodes at lat ≥ 70N (w = node area, a = a_ice);
  * the 1-day dynamics-only chain (48 steps, u and σ carried, everything else held): reverse
    growth of the cotangent of D(u_48), r_c and corr_c of the chain, and the per-step cost of
    the unrolled backward.

Variants are built from three derivative-only factors that are jit ARGUMENTS of the script's
own loop (one compiled program serves all of them): ``in_f`` [elem2D] (the explicit form,
``mevp_iterate(sigma_in_f=)``), ``out_f`` [elem2D] (the implicit form, ``sigma_out_f=``) and
``carry_f`` [n_sweeps] (a scale of the whole carry tangent after each sweep: the whole-carry
scaling of the earlier attempt, and the last-K truncation as a single zero).

  run:  python scripts/adjoint/ice_modes.py run --date 2013-01-01 --out <dir> [options]
  plot: python scripts/adjoint/ice_modes.py plot --json <dir>/modes_20130101.json
  why:  python scripts/adjoint/ice_modes.py why --date 2013-01-01 --out <dir> [--forward const
        --alpha-const 3000] --n-sweeps 24000   (Task 25; then `why-summary --dir <dir>`)

``--out`` must lie outside the checkout. Writes ``modes_<YYYYMMDD>.json`` (rewritten after each
variant), ``modes_<YYYYMMDD>.md`` (decision table), ``modes_<YYYYMMDD>_fields.npz`` (mode
energies for the figures); ``plot`` writes ``modes_<YYYYMMDD>_*.png``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import socket
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "core2_full.yaml"

# ---------------------------------------------------------------------------------------------
# generic linear algebra on flat vectors (numpy or jax arrays); tested on synthetic maps


def _dot(a, b) -> float:
    return float((a * b).sum())


def _norm(a) -> float:
    return math.sqrt(max(_dot(a, a), 0.0))


def _check_start(v0):
    n = _norm(v0)
    if not (n > 0.0 and math.isfinite(n)):
        raise ValueError(f"start vector has norm {n}: need a finite non-zero vector")
    return n


def power_iteration(apply, v0, n_iter: int):
    """Plain power iteration. Returns ``(rates, rayleigh, v)``: the growth ``|A v|/|v|`` per
    iteration, the Rayleigh quotients ``<v, A v>`` (sign of a real dominant eigenvalue) and the
    last normalised vector. A complex dominant pair shows as non-converging rates."""
    v = v0 / _check_start(v0)
    rates, ray = [], []
    for _ in range(int(n_iter)):
        w = apply(v)
        r = _norm(w)
        rates.append(r)
        ray.append(_dot(v, w))
        if not (r > 0.0 and math.isfinite(r)):
            break
        v = w / r
    return rates, ray, v


def power_svd(jvp, vjp, v0, n_iter: int):
    """Top singular value of a linear map by power iteration on ``AᵀA`` (``jvp`` = A·, ``vjp`` =
    Aᵀ·). Returns ``(sigma, history, v)`` with ``history`` the estimates ``|A v|``."""
    v = v0 / _check_start(v0)
    hist = []
    for _ in range(int(n_iter)):
        u = jvp(v)
        s = _norm(u)
        hist.append(s)
        if not (s > 0.0 and math.isfinite(s)):
            break
        w = vjp(u / s)
        nw = _norm(w)
        if not (nw > 0.0 and math.isfinite(nw)):
            break
        v = w / nw
    return (hist[-1] if hist else float("nan")), hist, v


def arnoldi(apply, v0, k: int):
    """``k``-step Arnoldi (modified Gram-Schmidt, re-orthogonalised once). Returns a dict with
    the Ritz ``values`` (sorted by decreasing modulus), relative ``residuals``
    ``|h_{m+1,m} y_m| / |θ|``, the reduced dimension ``m`` (< k on breakdown) and
    ``vector(i) -> (re, im)``, the i-th Ritz vector as two real vectors."""
    V = [v0 / _check_start(v0)]
    H = np.zeros((k + 1, k))
    m = k
    for j in range(k):
        w = apply(V[j])
        for _ in range(2):
            for i in range(j + 1):
                h = _dot(V[i], w)
                H[i, j] += h
                w = w - h * V[i]
        hn = _norm(w)
        H[j + 1, j] = hn
        scale = max(1.0, float(np.abs(H[: j + 2, : j + 1]).max()))
        if not math.isfinite(hn):
            raise FloatingPointError("Arnoldi: non-finite vector (the map overflowed)")
        if hn <= 1e-13 * scale:
            m = j + 1
            break
        V.append(w / hn)
    theta, Y = np.linalg.eig(H[:m, :m])
    order = np.argsort(-np.abs(theta), kind="stable")
    theta, Y = theta[order], Y[:, order]
    res = np.abs(H[m, m - 1]) * np.abs(Y[-1, :]) / np.maximum(np.abs(theta), 1e-300)

    def vector(i):
        re = sum(V[j] * float(Y[j, i].real) for j in range(m))
        im = sum(V[j] * float(Y[j, i].imag) for j in range(m))
        return re, im
    return {"values": theta, "residuals": res, "m": m, "vector": vector}


def lanczos_extremes(matvec, v0, k: int, checkpoints=(25, 50, 100, 200, 400, 800)):
    """Symmetric Lanczos with full reorthogonalization (twice per step) from ``v0`` (host
    numpy). At every checkpoint step m (and at the last one) it records the three smallest
    and the largest Ritz values of the m-step tridiagonal with their residual norms
    |β_m·y_m[-1]|. Returns ``(records, x)``, x the Ritz vector of the smallest value at the
    last record. Ritz values lie inside the spectrum, so the smallest one bounds λ_min from
    above; on a spectrum ~1e7 wide a small isolated λ_min converges slowly."""
    n = int(v0.size)
    k = int(min(k, n))
    Q = np.zeros((k + 1, n))
    al = np.zeros(k)
    be = np.zeros(k + 1)
    Q[0] = v0 / _check_start(v0)
    rec, y_last, m_last = [], None, 0
    for j in range(k):
        w = np.asarray(matvec(Q[j]), dtype=float)
        if not np.all(np.isfinite(w)):
            raise FloatingPointError("Lanczos: non-finite matvec")
        if j > 0:
            w = w - be[j] * Q[j - 1]
        al[j] = Q[j] @ w
        w = w - al[j] * Q[j]
        for _ in range(2):
            w = w - Q[:j + 1].T @ (Q[:j + 1] @ w)
        be[j + 1] = np.linalg.norm(w)
        m = j + 1
        brk = be[j + 1] <= 1e-13 * max(float(np.abs(al[:m]).max()), 1e-300)
        if m in checkpoints or m == k or brk:
            T = np.diag(al[:m]) + np.diag(be[1:m], 1) + np.diag(be[1:m], -1)
            th, Y = np.linalg.eigh(T)
            res = np.abs(be[m] * Y[-1, :])
            rec.append({"m": m, "ritz_min": th[:3].tolist(), "res_min": res[:3].tolist(),
                        "ritz_max": float(th[-1]), "res_max": float(res[-1])})
            y_last, m_last = Y[:, 0], m
        if brk:
            break
        Q[j + 1] = w / be[j + 1]
    return rec, Q[:m_last].T @ y_last


def classify(lam, tol: float = 1e-6) -> str:
    """Type of an eigenvalue: 'oscillating' (negative real: sign flip per application),
    'steady' (positive real) or 'complex pair'."""
    lam = complex(lam)
    if abs(lam.imag) <= tol * max(abs(lam), 1e-300):
        return "oscillating" if lam.real < 0 else "steady"
    return "complex pair"


def dominant(ritz: dict) -> dict:
    """The dominant Ritz value of an :func:`arnoldi` result, its type and residual."""
    lam = complex(ritz["values"][0])
    return {"re": lam.real, "im": lam.imag, "abs": abs(lam), "type": classify(lam),
            "residual": float(ritz["residuals"][0]),
            "next_abs": [float(abs(x)) for x in ritz["values"][1:6]]}


def weighted_corr(t, f, w, centered: bool = True) -> float:
    """Weighted pattern correlation of two fields (same shape; ``w`` broadcast to them)."""
    t, f, w = (np.asarray(a, np.float64) for a in (t, f, np.broadcast_to(w, np.shape(t))))
    sw = w.sum()
    if centered:
        t = t - (w * t).sum() / sw
        f = f - (w * f).sum() / sw
    den = math.sqrt(float((w * t * t).sum()) * float((w * f * f).sum()))
    return float((w * t * f).sum()) / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------------------------
# decision table (markdown) — written and parsed back (round-trip tested)

TABLE_COLUMNS = (
    ("variant", "variant"), ("form", "form"), ("tn", "T_N"),
    ("sigma1_step", "σ1 step map"), ("lam_step", "max abs λ step map"),
    ("growth_step_chain", "chain growth/step"), ("growth_day", "reverse growth/day"),
    ("mode_type", "dominant sweep mode (sweep 60)"), ("mode_where", "mode location"),
    ("r_lnp_1", "r P* 1 step"), ("c2_lnp_1", "corr P* 1 step (FD h 1e-2)"),
    ("c_lnp_1", "corr P* 1 step (FD h of r)"),
    ("r_lncd_1", "r C_d 1 step"), ("c2_lncd_1", "corr C_d 1 step (FD h 1e-2)"),
    ("c_lncd_1", "corr C_d 1 step (FD h of r)"),
    ("r_lnp_d", "r P* 1 day"), ("c2_lnp_d", "corr P* 1 day (FD h 1e-2)"),
    ("c_lnp_d", "corr P* 1 day (FD h of r)"),
    ("r_lncd_d", "r C_d 1 day"), ("c2_lncd_d", "corr C_d 1 day (FD h 1e-2)"),
    ("c_lncd_d", "corr C_d 1 day (FD h of r)"),
    ("bwd_s", "backward s/step"), ("verdict", "PASS/FAIL"),
)


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not math.isfinite(x):
            return str(x)
        return f"{x:.3g}" if (x == 0 or 1e-3 <= abs(x) < 1e4) else f"{x:.2e}"
    return str(x).replace("|", "/").replace("\n", " ")


def table_markdown(rows, columns=TABLE_COLUMNS) -> str:
    head = "| " + " | ".join(h for _, h in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(r.get(k)) for k, _ in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body]) + "\n"


def parse_markdown_table(text: str, columns=TABLE_COLUMNS) -> list:
    """Rows of the first table in ``text`` as dicts of the formatted strings (keys = column
    keys). Inverse of :func:`table_markdown` up to the formatting."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return []
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    keys = {h: k for k, h in columns}
    out = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        out.append({keys.get(h, h): c for h, c in zip(head, cells)})
    return out


# ---------------------------------------------------------------------------------------------
# provenance helpers (no git on the aarch64 nodes: read HEAD through the .git file)


def git_head(root=ROOT) -> str:
    try:
        g = Path(root) / ".git"
        gd = Path(g.read_text().split("gitdir:", 1)[1].strip()) if g.is_file() else g
        head = (gd / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(None, 1)[1]
        common = gd
        if (gd / "commondir").is_file():
            common = (gd / (gd / "commondir").read_text().strip()).resolve()
        for base in (gd, common):
            if (base / ref).is_file():
                return (base / ref).read_text().strip()
        packed = common / "packed-refs"
        if packed.is_file():
            for ln in packed.read_text().splitlines():
                if ln.endswith(" " + ref):
                    return ln.split()[0]
        return f"<unresolved {ref}>"
    except Exception as e:  # noqa: BLE001
        return f"<unknown: {e}>"


def _refuse_inside_checkout(path) -> Path:
    p = Path(path).resolve()
    root = ROOT.resolve()
    if p == root or root in p.parents:
        raise SystemExit(f"output {p} is inside the checkout {root}; write to /work")
    return p


# ---------------------------------------------------------------------------------------------
# TLf: the frozen-coefficient tangent of a fixed-point iteration


def frozen_tangent(lin, dx0, danc, dc, n: int, omega: float = 1.0):
    """``n`` applications of one FROZEN affine sweep tangent, δx ← lin(δx, δanchor, δc), from
    δx0. ``lin`` is the linear map of ``jax.linearize(sweep, x_N, anchor, c)``: the sweep
    linearized once, at the final iterate x_N, instead of at every iterate.

    ``omega`` < 1 damps the iteration (Krasnosel'skii–Mann averaging), δx ← (1−ω)·δx +
    ω·lin(δx, …): the same fixed point, every eigenvalue λ of the sweep map moved to
    1 − ω + ωλ. It adds dissipation to the forward-backward pseudo-time iteration without
    changing what it converges to (Task 23, T2); ω = 1 is the plain frozen map."""
    import jax
    from jax import lax

    if omega == 1.0:
        def body(d, _):
            return lin(d, danc, dc), None
    else:
        def body(d, _):
            y = lin(d, danc, dc)
            return jax.tree.map(lambda a, b: (1.0 - omega) * a + omega * b, d, y), None
    return lax.scan(body, dx0, None, length=n)[0]


# ---------------------------------------------------------------------------------------------
# the mEVP stress target with derivative-only symmetrizations (TLfS / TLfP)

STRESS_MODES = ("exact", "S", "P")


def stress_terms(eps11, eps22, eps12, pressure_fac, vale, delta_min, mode="exact"):
    """det2·S(ε), the new-stress term of the mEVP update σ' = det1·σ + det2·S (pressure_fac has
    det2 folded), split into the viscous part ζ·A·ε and the replacement pressure P_r:
        S11 = ½ζ(ε1 + ε2/e²) − ½P_r,  S22 = ½ζ(ε1 − ε2/e²) − ½P_r,  S12 = ζ ε12/e²,
    ζ = P/(Δ+δmin), P_r = PΔ/(Δ+δmin) (algebraically the model's ½·P/(Δ+δmin)·(ε1 − Δ ± ε2/e²)).
    Derivative-only variants (the values are unchanged):
      "S": P_r from stop_gradient(Δ). The rest, ζ·A·ε = D'(Δ)∇Δ with D'(Δ) = PΔ/(2(Δ+δmin)), is
           the gradient of a convex potential: its Jacobian, pairing (σ11, σ22, 2σ12) with
           (ε11, ε22, ε12), is symmetric positive semi-definite.
      "P": ζ and P_r both from stop_gradient(Δ) (Picard / frozen viscosity): δS = ζ·A·δε.
      "Tθ" (e.g. "T0.5"): P_r frozen as in P, ζ from Δ_θ = sg(Δ) + θ·(Δ − sg(Δ)): the value is
           Δ, the viscosity derivative is scaled by θ (θ = 0 is P, θ = 1 is S).
    The P* dependence (through pressure_fac) stays live in every mode."""
    import jax
    from fesom_jax.ice_evp import _safe_sqrt
    theta = None
    if mode.startswith("T"):
        theta = float(mode[1:])
    elif mode not in STRESS_MODES:
        raise ValueError(f"stress mode {mode!r}")
    eps1, eps2 = eps11 + eps22, eps11 - eps22
    delta = _safe_sqrt(eps1 * eps1 + vale * (eps2 * eps2 + 4.0 * eps12 * eps12))
    dsg = jax.lax.stop_gradient(delta)
    if theta is not None:
        dz = dsg + theta * (delta - dsg)
    else:
        dz = dsg if mode == "P" else delta
    dr = dsg if (mode in ("S", "P") or theta is not None) else delta
    zeta = pressure_fac / (dz + delta_min)
    pr = pressure_fac * dr / (dr + delta_min)
    t12 = zeta * eps12 * vale
    t11 = 0.5 * zeta * (eps1 + eps2 * vale) - 0.5 * pr
    t22 = 0.5 * zeta * (eps1 - eps2 * vale) - 0.5 * pr
    return t11, t12, t22


def mevp_iterate_mode(cfg, mesh, u_aux, v_aux, s11, s12, s22, st, u_anchor, v_anchor,
                      srfoce_u, srfoce_v, stress_ax, stress_ay, rdt, mode,
                      sigma_in_f=None, sigma_out_f=None):
    """:func:`fesom_jax.ice_mevp.mevp_iterate` with the stress target of :func:`stress_terms`
    (same algebra, other rounding). Used only for tangents (TLa/TLai/TLf with mode S or P)."""
    import jax.numpy as jnp
    from fesom_jax.adjoint_stab import scale_tangent
    from fesom_jax.config import DENSITY_0
    from fesom_jax.forcing import _safe_speed
    from fesom_jax.ice_evp import strain_rates, stress_div_scatter
    if sigma_in_f is not None:
        s11, s12, s22 = (scale_tangent(q, sigma_in_f) for q in (s11, s12, s22))
    eps11, eps22, eps12 = strain_rates(mesh, u_aux, v_aux)
    t11, t12, t22 = stress_terms(eps11, eps22, eps12, st.pressure_fac, cfg.vale, cfg.delta_min,
                                 mode)
    det1 = cfg.mevp_det1
    s11 = jnp.where(st.ice_el, det1 * s11 + t11, s11)
    s12 = jnp.where(st.ice_el, det1 * s12 + t12, s12)
    s22 = jnp.where(st.ice_el, det1 * s22 + t22, s22)
    if sigma_out_f is not None:
        s11, s12, s22 = (scale_tangent(q, sigma_out_f) for q in (s11, s12, s22))
    u_rhs, v_rhs = stress_div_scatter(mesh, s11, s12, s22, st.ice_el)
    u_rhs = u_rhs * st.mass + st.tilt_u
    v_rhs = v_rhs * st.mass + st.tilt_v
    umod = _safe_speed(u_aux - srfoce_u, v_aux - srfoce_v)
    drag = rdt * cfg.cd_oce_ice * umod * DENSITY_0 * st.inv_thickness
    rhsu = u_anchor + drag * srfoce_u + rdt * (st.inv_thickness * stress_ax + u_rhs) \
        + cfg.beta_evp * u_aux
    rhsv = v_anchor + drag * srfoce_v + rdt * (st.inv_thickness * stress_ay + v_rhs) \
        + cfg.beta_evp * v_aux
    obd = 1.0 + cfg.beta_evp + drag
    rf = rdt * mesh.coriolis_node
    det = st.bc_index / (obd * obd + rf * rf)
    un = det * (obd * rhsu + rf * rhsv)
    vn = det * (obd * rhsv - rf * rhsu)
    u_aux = jnp.where(st.ice_nod, un, u_aux)
    v_aux = jnp.where(st.ice_nod, vn, v_aux)
    u_aux = jnp.where(st.bc_index > 0.0, u_aux, 0.0)
    v_aux = jnp.where(st.bc_index > 0.0, v_aux, 0.0)
    return u_aux, v_aux, s11, s12, s22


def element_pairing_jacobians(e11, e22, e12, pf, vale, delta_min, mode):
    """Per element, the Jacobian of (S11, S22, 2·S12) w.r.t. (ε11, ε22, ε12) of
    :func:`stress_terms` in ``mode``: the pairing σ:ε = σ11ε11 + σ22ε22 + 2σ12ε12, in which
    the stiffness Σ_e A_e·δε(v)ᵀ·E_e·δε(w) is the stress divergence's bilinear form (the
    divergence scatter is minus the transpose of the strain rate, metric term included).
    Returns an (n, 3, 3) numpy array."""
    import jax
    import jax.numpy as jnp

    def f(e, p):
        t11, t12, t22 = stress_terms(e[0], e[1], e[2], p, vale, delta_min, mode)
        return jnp.stack([t11, t22, 2.0 * t12])
    E = jnp.stack([jnp.asarray(e11), jnp.asarray(e22), jnp.asarray(e12)], axis=1)
    return np.asarray(jax.vmap(jax.jacfwd(f), in_axes=(0, 0))(E, jnp.asarray(pf)))


def element_strain_matrix(gs_e, mfac_e):
    """The 3×6 map from one element's vertex velocities (U0, U1, U2, V0, V1, V2) to
    (ε11, ε22, ε12), as :func:`fesom_jax.ice_evp.strain_rates` computes it."""
    g = np.asarray(gs_e, float)
    m3 = float(mfac_e) / 3.0
    return np.array([[g[0], g[1], g[2], -m3, -m3, -m3],
                     [0.0, 0.0, 0.0, g[3], g[4], g[5]],
                     [0.5 * (g[3] + m3), 0.5 * (g[4] + m3), 0.5 * (g[5] + m3),
                      0.5 * g[0], 0.5 * g[1], 0.5 * g[2]]])


# ---------------------------------------------------------------------------------------------
# aEVP (FESOM2 whichEVP=2): per-element α from the local strength, β at nodes


AEVP_C = 0.15            # c_aevp, MOD_ICE.F90:199
AEVP_ALPHA_MIN = 50.0    # ice_maEVP.F90:1027


def aevp_alpha_beta(cfg, mesh, a_ice, m_ice, u, v, *, c_aevp=AEVP_C, alpha_min=AEVP_ALPHA_MIN,
                    pstar_mult=1.0):
    """``find_alpha_field_a`` + ``find_beta_field_a`` (ice_maEVP.F90:941-1031, 1291-1326).

    α_e = max(α_min, sqrt(ice_dt·c_aevp·p/ρ_ice/A_e)), p = P*·exp(-C(1-ā))/(Δ+δmin) with NO
    thickness factor (the Fortran's own comment), Δ from the given velocities (the Fortran calls
    it after the 120 sweeps on the final u_ice_aux, so the next step uses it: the entry velocity
    of that step). Elements with mean₃ m ≤ 0.01 or a cavity node keep their old α; here (no α
    state) they take ``cfg.alpha_evp``, the Fortran's initial value. β_n = max of α over the
    elements around n. ``pstar_mult`` scales P* (the ln P* control acts on α as well)."""
    import jax.numpy as jnp
    from jax import ops as jops
    from fesom_jax.ice_evp import _safe_sqrt, strain_rates
    en = mesh.elem_nodes
    val3 = 1.0 / 3.0
    m_v, a_v = m_ice[en], a_ice[en]
    msum = (m_v[:, 0] + m_v[:, 1] + m_v[:, 2]) * val3
    asum = (a_v[:, 0] + a_v[:, 1] + a_v[:, 2]) * val3
    e11, e22, e12 = strain_rates(mesh, u, v)
    eps1, eps2 = e11 + e22, e11 - e22
    delta = _safe_sqrt(eps1 * eps1 + cfg.vale * (eps2 * eps2 + 4.0 * e12 * e12))
    pressure = cfg.pstar * pstar_mult * jnp.exp(-cfg.c_pressure * (1.0 - asum)) \
        / (delta + cfg.delta_min)
    alpha = jnp.maximum(alpha_min, jnp.sqrt(cfg.ice_dt * c_aevp * pressure / cfg.rhoice
                                            / mesh.elem_area))
    upd = (msum > 0.01) & (jnp.asarray(mesh.ulevels) <= 1)
    alpha = jnp.where(upd, alpha, cfg.alpha_evp)
    beta = jops.segment_max(jnp.repeat(alpha, 3), jnp.asarray(en).reshape(-1),
                            num_segments=int(mesh.nod2D))
    return alpha, beta


# ---------------------------------------------------------------------------------------------
# the CORE2 ice map


class IceMap:
    """The captured mEVP inputs of one model step and the compiled programs of the study."""

    def __init__(self, date: str, n_sweeps: int | None = None, forward: str = "mevp",
                 aevp_c: float = AEVP_C, aevp_alpha_min: float = AEVP_ALPHA_MIN,
                 alpha_const: float | None = None):
        """``forward``: "mevp" (the model's map), "aevp" (the same sweep with per-element α and
        per-node β from :func:`aevp_alpha_beta` at the step's entry velocity; ``aevp_c = 0``
        with ``aevp_alpha_min = alpha_evp`` makes α ≡ alpha_evp, the mEVP map) or "const" (the
        mEVP sweep with the scalar α = β = ``alpha_const``: the same fixed point, the implicit VP
        step, reached by a slower iteration that is stable where α = 250 is not; the TLa base
        α of the tangent rows is ``alpha_const`` too)."""
        if forward not in ("mevp", "aevp", "const"):
            raise ValueError(f"forward {forward!r}")
        if forward == "const" and not (alpha_const and alpha_const > 0):
            raise ValueError("forward 'const' needs alpha_const > 0")
        self.forward = forward
        self.alpha_const = float(alpha_const) if forward == "const" else None
        sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
        import core2_adj as C  # noqa: E402  (checks that fesom_jax is this checkout's)
        import jax
        import jax.numpy as jnp
        from jax import lax
        from fesom_jax import forcing as _forcing
        from fesom_jax import ice_mevp
        from fesom_jax import step as stepmod
        from fesom_jax.adjoint_stab import scale_tangent
        from fesom_jax.ice_evp import _safe_sqrt, boundary_node_mask, strain_rates

        self.jax, self.jnp = jax, jnp
        t0 = time.time()
        model = C.build_model(start=date, n_steps=1, chunk_steps=1, ice_mode="prescribed")
        self.model, self.mesh = model, model.mesh
        mesh = model.mesh
        x0f = jax.tree.map(lambda a: a[0], model.stack_forcing([model.date(model.start, 0)]))
        s0, u0 = model.s0, model.params0
        phys = model.phys
        self.build_s = time.time() - t0

        # ---- capture the mEVP call (and the wind stress call) of one real model step ----
        cap = {}
        orig_mevp, orig_stress = ice_mevp.mevp_dynamics, _forcing.atm_ice_stress

        def spy_mevp(cfg, mesh_, **kw):
            out = orig_mevp(cfg, mesh_, **kw)
            cap["cfg"] = cfg
            cap["adjoint_subcycles"] = kw.pop("adjoint_subcycles", None)
            cap["exch"] = kw.pop("exch", None)
            cap["kw"] = kw
            cap["out"] = out
            return out

        def spy_stress(u_air, v_air, u_ice, v_ice):
            cap["air"] = (u_air, v_air, u_ice, v_ice)
            return orig_stress(u_air, v_air, u_ice, v_ice)

        def one_step(s, x, u):
            ice_mevp.mevp_dynamics, _forcing.atm_ice_stress = spy_mevp, spy_stress
            try:
                new = stepmod.step(s, mesh, model.op, None, model.to_params(u), dt=model.dt,
                                   is_first_step=False, step_forcing=x,
                                   forcing_static=model.forcing.static,
                                   adjoint_cfg=model.adjoint_cfg, **phys)
            finally:
                ice_mevp.mevp_dynamics, _forcing.atm_ice_stress = orig_mevp, orig_stress
            kw = {k: v for k, v in cap["kw"].items() if v is not None}
            return new.u_ice, new.v_ice, kw, cap["out"], cap["air"]

        t1 = time.time()
        u_new, v_new, kw, out, air = jax.jit(one_step)(s0, x0f, u0)
        jax.block_until_ready(u_new)
        self.capture_s = time.time() - t1
        # The dense step hands ice_surface_step an identity `exch` (step.py), so exchange_pair
        # only stacks/unstacks the pair: a value no-op. A real halo exchange would fail the
        # bitwise self-check below.
        self.exch_kind = "None" if cap["exch"] is None else type(cap["exch"]).__name__
        if jax.device_count() != 1:
            raise RuntimeError("run the mode study on one device (dense model)")
        cfg = cap["cfg"]
        self.cfg = cfg
        self.n_sweeps = int(cfg.evp_rheol_steps) if n_sweeps is None else int(n_sweeps)
        self.step_u_ice = (u_new, v_new)
        self.mevp_out = out
        self.kw = kw
        bn = kw.get("boundary_node")
        bn = boundary_node_mask(mesh) if bn is None else bn
        a = kw["a_ice"]
        # The captured fields are jit ARGUMENTS of every program (`inp`), as they are of the
        # model step: closed over, they become compile-time constants and XLA's constant folding
        # evaluates mevp_setup with different rounding (the forward amplifies one ulp to ~1e-7).
        inp = {"a": kw["a_ice"], "m": kw["m_ice"], "ms": kw["m_snow"], "elev": kw["elevation"],
               "srf_u": kw["srfoce_u"], "srf_v": kw["srfoce_v"], "u_air": air[0],
               "v_air": air[1], "bn": bn}
        en_ = np.asarray(mesh.elem_nodes)
        m_e = np.asarray(kw["m_ice"])[en_].mean(axis=1)
        ms_e = np.asarray(kw["m_snow"])[en_].mean(axis=1)
        M_e = cfg.rhoice * m_e + cfg.rhosno * ms_e
        inp["gam_c"] = jnp.asarray(cfg.ice_dt * (M_e / (1.0 + M_e * M_e))
                                   / (np.asarray(mesh.elem_area) * cfg.mevp_det2
                                      * cfg.alpha_evp * cfg.beta_evp))
        self.inp = inp
        self.air_check = {
            "u_ice_arg_is_entry": bool(np.array_equal(np.asarray(air[2]), np.asarray(kw["u_ice"]))
                                       and np.array_equal(np.asarray(air[3]), np.asarray(kw["v_ice"]))),
        }
        self.x0 = tuple(kw[k] for k in ("u_ice", "v_ice", "sigma11", "sigma12", "sigma22"))
        nod, el = int(mesh.nod2D), int(mesh.elem2D)
        self.nod, self.el = nod, el
        rdt = cfg.ice_dt
        self.rdt = rdt

        # ---- the step map F(x, c; in_f, out_f, carry_f) ----
        # The controls enter as multipliers: P* acts only through pressure_fac (linear in it),
        # C_d only through the drag rdt·C_d·|Δu|·ρ0·inv_thickness, so it is applied to
        # inv_thickness and the other use of inv_thickness (the air-stress term
        # inv_thickness·τ_air) is divided back. Any traced control changes XLA's rounding of the
        # sweep (the multipliers are exactly 1 at c = 0, but the program is not the model's), and
        # this forward amplifies a one-ulp change to ~1e-7 m/s within 120 sweeps (CPU, 2013-01-01,
        # at one Arctic node). So the bitwise self-check uses `p_fwd0`, the same map with no
        # control at all; the control programs are used for FD and tangents only.
        def cfg_of(inp, u, v, ep=1.0, gk=None, fwd=forward):
            """The config of the sweep: mEVP's, or aEVP's α/β fields from the entry velocity.
            gk[2] > 0 holds α/β fixed in the tangent (the forward is unchanged)."""
            if fwd == "mevp":
                return cfg
            if fwd == "const":
                return cfg._replace(alpha_evp=float(alpha_const), beta_evp=float(alpha_const))
            al, be = aevp_alpha_beta(cfg, mesh, inp["a"], inp["m"], u, v, c_aevp=aevp_c,
                                     alpha_min=aevp_alpha_min, pstar_mult=ep)
            if gk is not None:
                al = jnp.where(gk[2] > 0, jax.lax.stop_gradient(al), al)
                be = jnp.where(gk[2] > 0, jax.lax.stop_gradient(be), be)
            return cfg._replace(alpha_evp=al, beta_evp=be)
        self.cfg_of = cfg_of

        def setup(inp, c, u, v, gk=None, fwd=forward):
            ep, ec = jnp.exp(c[0]), jnp.exp(c[1])
            cx = cfg_of(inp, u, v, ep, gk, fwd)
            st = ice_mevp.mevp_setup(cx, mesh, inp["a"], inp["m"], inp["ms"], inp["elev"],
                                     inp["bn"])
            return ec, st._replace(pressure_fac=st.pressure_fac * ep,
                                   inv_thickness=st.inv_thickness * ec), cx

        def air(inp, u, v, ec):
            sax, say = _forcing.atm_ice_stress(inp["u_air"], inp["v_air"], u, v)
            return sax / ec, say / ec

        def local_factor(inp, st, x, gk):
            """Per-element damping of the new-stress tangent from the LOCAL mEVP stability number
            of the current iterate (forward values only): γ/(αβ) = P/(Δ+δmin)·rdt·M/(1+M²)/A/(αβ).
            gk = (κ_loc, κ_cut): f = min(1, κ_loc/(γ/αβ)) if κ_loc > 0, and f = 0 where
            γ/αβ > κ_cut if κ_cut > 0; 1 on ice-free elements. (0, 0) ⇒ f ≡ 1."""
            xs = jax.lax.stop_gradient(x)
            e11, e22, e12 = strain_rates(mesh, xs[0], xs[1])
            eps1, eps2 = e11 + e22, e11 - e22
            delta = _safe_sqrt(eps1 * eps1 + cfg.vale * (eps2 * eps2 + 4.0 * e12 * e12))
            gr = jax.lax.stop_gradient(st.pressure_fac) / (delta + cfg.delta_min) * inp["gam_c"]
            f_loc = jnp.where(gk[0] > 0, jnp.minimum(1.0, gk[0] / jnp.maximum(gr, 1e-300)), 1.0)
            f_cut = jnp.where((gk[1] > 0) & (gr > gk[1]), 0.0, 1.0)
            return jnp.where(st.ice_el, f_loc * f_cut, 1.0)

        def sweep(inp, st, cx, x, anchor, sax, say, in_f, out_f, cf, gk=None, tla_coef=None,
                  mode="exact"):
            if gk is not None:
                out_f = out_f * local_factor(inp, st, x, gk)
            if isinstance(tla_coef, str):              # "iter": α_t from this sweep's input
                tla_coef = tla_coefficients(inp, st, x[0], x[1], gk[3])
            if tla_coef is None:
                y = ice_mevp.mevp_iterate(cx, mesh, *x, st, anchor[0], anchor[1], inp["srf_u"],
                                          inp["srf_v"], sax, say, rdt, sigma_in_f=in_f,
                                          sigma_out_f=out_f)
            else:
                y = tla_core(cx, mode, x, st, anchor, sax, say, (inp["srf_u"], inp["srf_v"]),
                             in_f, out_f, *tla_coef)
            return tuple(scale_tangent(q, cf) for q in y)

        # ---- "TLa": the mEVP sweep, linearized with tangent-only per-element α_t / node β_t ----
        @partial(jax.custom_jvp, nondiff_argnums=(0, 1))
        def tla_core(cx, mode, x, st, anchor, sax, say, srf, in_f, out_f, al, be):
            return ice_mevp.mevp_iterate(cx, mesh, *x, st, anchor[0], anchor[1], srf[0], srf[1],
                                         sax, say, rdt, sigma_in_f=in_f, sigma_out_f=out_f)

        @tla_core.defjvp
        def _tla_core_jvp(cx, mode, primals, tangents):
            """Primal: the mEVP sweep. Tangent: the JVP of the same sweep at the same point with
            α_t (det1_t, det2_t in the stress update; det2 rescales pressure_fac) and β_t."""
            x, st, anchor, sax, say, srf, in_f, out_f, al, be = primals
            out = tla_core(cx, mode, *primals)
            ct = cx._replace(alpha_evp=al, beta_evp=be)
            r2 = ct.mevp_det2 / cx.mevp_det2                      # exactly 1 where α_t = α

            def g(x_, st_, anchor_, sax_, say_, srf_):
                st2 = st_._replace(pressure_fac=st_.pressure_fac * r2)
                if mode == "exact":
                    return ice_mevp.mevp_iterate(ct, mesh, *x_, st2, anchor_[0], anchor_[1],
                                                 srf_[0], srf_[1], sax_, say_, rdt,
                                                 sigma_in_f=in_f, sigma_out_f=out_f)
                return mevp_iterate_mode(ct, mesh, *x_, st2, anchor_[0], anchor_[1], srf_[0],
                                         srf_[1], sax_, say_, rdt, mode, sigma_in_f=in_f,
                                         sigma_out_f=out_f)
            _, t = jax.jvp(g, primals[:6], tangents[:6])
            return out, t

        def tla_coefficients(inp, st, u, v, kappa0):
            """α_t = max(α, α·sqrt(s/κ0)) per ice element, s = γ/(αβ) at (u, v) (forward values
            only), so that γ/(α_t β_t) ≤ κ0; β_t = max of α_t over the node's elements."""
            u, v = jax.lax.stop_gradient(u), jax.lax.stop_gradient(v)
            st = jax.lax.stop_gradient(st)
            e11, e22, e12 = strain_rates(mesh, u, v)
            eps1, eps2 = e11 + e22, e11 - e22
            delta = _safe_sqrt(eps1 * eps1 + cfg.vale * (eps2 * eps2 + 4.0 * e12 * e12))
            sn = st.pressure_fac / (delta + cfg.delta_min) * inp["gam_c"]
            a0 = cfg.alpha_evp if forward != "const" else float(alpha_const)
            al = jnp.where(st.ice_el, jnp.maximum(a0, a0 * jnp.sqrt(sn / kappa0)), a0)
            be = jax.ops.segment_max(jnp.repeat(al, 3), jnp.asarray(mesh.elem_nodes).reshape(-1),
                                     num_segments=int(mesh.nod2D))
            return jax.lax.stop_gradient(al), jax.lax.stop_gradient(be)
        self._tla_coefficients = tla_coefficients

        def sweeps(inp, st, cx, x, anchor, sax, say, in_f, out_f, carry_f, gk=None,
                   tla_coef=None, mode="exact"):
            def body(carry, cf):
                return sweep(inp, st, cx, carry, anchor, sax, say, in_f, out_f, cf, gk,
                             tla_coef, mode), None
            return lax.scan(jax.checkpoint(body), x, carry_f)[0]

        def step_map(inp, x, c, in_f, out_f, carry_f, gk, fwd=forward, tla=False,
                     mode="exact"):
            ec, st, cx = setup(inp, c, x[0], x[1], gk, fwd)
            sax, say = air(inp, x[0], x[1], ec)
            coef = (tla_coefficients(inp, st, x[0], x[1], gk[3]) if tla == "entry"
                    else "iter" if tla == "iter" else None)
            return sweeps(inp, st, cx, x, (x[0], x[1]), sax, say, in_f, out_f, carry_f, gk,
                          coef, mode)

        def step_map0(inp, x, sax, say, in_f, out_f, carry_f, _gk=None, fwd=forward):
            # no control and the wind stress as an argument (`p_air`): the model's own rounding
            cx = cfg_of(inp, x[0], x[1], fwd=fwd)
            st = ice_mevp.mevp_setup(cx, mesh, inp["a"], inp["m"], inp["ms"], inp["elev"],
                                     inp["bn"])
            return sweeps(inp, st, cx, x, (x[0], x[1]), sax, say, in_f, out_f, carry_f)

        def iterates(inp, x, c, fwd=forward):
            ec, st, cx = setup(inp, c, x[0], x[1], None, fwd)
            sax, say = air(inp, x[0], x[1], ec)
            ones = jnp.ones(el)

            def body(carry, _):
                y = sweep(inp, st, cx, carry, (x[0], x[1]), sax, say, ones, ones, 1.0)
                return y, carry                                  # ys[p] = input of sweep p+1
            return lax.scan(body, x, None, length=self.n_sweeps)[1]

        _p_air = jax.jit(lambda inp, u, v: _forcing.atm_ice_stress(inp["u_air"], inp["v_air"], u, v))
        self.p_air = partial(_p_air, inp)
        sx, sy = self.p_air(self.x0[0], self.x0[1])
        self.air0 = (sx, sy)
        self.air_check["stress_bitwise"] = bool(
            np.array_equal(np.asarray(sx), np.asarray(kw["stress_ax"]))
            and np.array_equal(np.asarray(sy), np.asarray(kw["stress_ay"])))

        # ---- scaling of the flat state (u, v by the RMS ice speed, σ by the RMS ice stress) ----
        st0 = setup(inp, jnp.zeros(2), self.x0[0], self.x0[1])[1]
        self.st0 = st0
        ice_nod, ice_el = np.asarray(st0.ice_nod), np.asarray(st0.ice_el)
        u0n, v0n = np.asarray(self.x0[0]), np.asarray(self.x0[1])
        U = float(np.sqrt(np.mean((u0n ** 2 + v0n ** 2)[ice_nod]))) if ice_nod.any() else 0.0
        S = float(np.sqrt(np.mean(sum(np.asarray(q) ** 2 for q in self.x0[2:])[ice_el] / 3.0))) \
            if ice_el.any() else 0.0
        U = U if U > 0 else 0.1
        S = S if S > 0 else 1e4
        self.U, self.S = U, S
        dscale = jnp.concatenate([jnp.full(2 * nod, U), jnp.full(3 * el, S)])
        self.dscale = dscale
        sizes = (nod, nod, el, el, el)
        offs = np.cumsum((0,) + sizes)

        def flat(x):
            return jnp.concatenate(x)

        def unflat(v):
            return tuple(v[offs[i]:offs[i + 1]] for i in range(5))
        self.flat, self.unflat, self.offs = flat, unflat, offs

        # ---- drift functional ----
        lon, lat = C._lonlat_deg(mesh)
        self.lon_n, self.lat_n = lon, lat
        box = (lat >= 70.0) & ice_nod
        self.box = box
        w = np.asarray(mesh.area[:, 0]) * np.asarray(a)
        wa = jnp.asarray(np.where(box, w, 0.0))
        self.wa = np.asarray(wa)

        def drift(u, v):
            return jnp.sum(wa * _forcing._safe_speed(u, v)) / jnp.sum(wa)
        self.drift = drift

        # ---- compiled programs (factors are arguments: one program per kind) ----
        self.p_fwd = partial(jax.jit(step_map), inp)
        self.p_fwd0 = partial(jax.jit(step_map0), inp)
        # the other forward, for the aEVP-vs-mEVP comparison (forward only)
        other = "mevp" if forward in ("aevp", "const") else "aevp"
        self.p_fwd0_other = partial(jax.jit(partial(step_map0, fwd=other)), inp)
        self.p_iterates_other = partial(jax.jit(partial(iterates, fwd=other)), inp)
        self.p_alpha_beta = jax.jit(lambda inp, u, v: aevp_alpha_beta(
            cfg, mesh, inp["a"], inp["m"], u, v, c_aevp=aevp_c, alpha_min=aevp_alpha_min))

        def jvp_x(inp, xf, c, in_f, out_f, carry_f, gk, vf, dc, tla=False, mode="exact"):
            x = unflat(xf)
            y, t = jax.jvp(lambda x_, c_: step_map(inp, x_, c_, in_f, out_f, carry_f, gk,
                                                   tla=tla, mode=mode),
                           (x, c), (unflat(vf * dscale), dc))
            return flat(y), flat(t) / dscale
        self.p_jvp = partial(jax.jit(jvp_x), inp)

        def vjp_x(inp, xf, c, in_f, out_f, carry_f, gk, wf, tla=False, mode="exact"):
            x = unflat(xf)
            _, back = jax.vjp(lambda x_: step_map(inp, x_, c, in_f, out_f, carry_f, gk, tla=tla,
                                                  mode=mode), x)
            return flat(back(unflat(wf / dscale))[0]) * dscale
        self.p_vjp = partial(jax.jit(vjp_x), inp)

        def jvp_sweep(inp, xf, anchor_f, c, in_f, out_f, cf, gk, vf, tla=False, mode="exact"):
            anc = unflat(anchor_f)
            ec, st, cx = setup(inp, c, anc[0], anc[1], gk)
            sax, say = air(inp, anc[0], anc[1], ec)
            coef = (tla_coefficients(inp, st, anc[0], anc[1], gk[3]) if tla == "entry"
                    else "iter" if tla == "iter" else None)
            _, t = jax.jvp(lambda x_: sweep(inp, st, cx, x_, (anc[0], anc[1]), sax, say,
                                            in_f, out_f, cf, gk, coef, mode), (unflat(xf),),
                           (unflat(vf * dscale),))
            return flat(t) / dscale
        self.p_jvp_sweep = partial(jax.jit(jvp_sweep), inp)
        # TLa programs (mEVP forward only; κ0 = gk[3] is an argument)
        self.p_jvp_tla = partial(jax.jit(partial(jvp_x, tla="entry")), inp)
        self.p_vjp_tla = partial(jax.jit(partial(vjp_x, tla="entry")), inp)
        self.p_jvp_sweep_tla = partial(jax.jit(partial(jvp_sweep, tla="entry")), inp)
        # TLf: the sweep linearized once at the final iterate x_N (TLa coefficients from x_N),
        # applied n_sweeps times with the anchor and control forcing in every application
        def tlf_lin(inp, x, c, in_f, out_f, carry_f, gk, mode="exact"):
            y = step_map(inp, x, c, in_f, out_f, carry_f, gk)
            xN = jax.lax.stop_gradient(y)
            _, st, _ = setup(inp, c, x[0], x[1])
            coef = tla_coefficients(inp, st, xN[0], xN[1], gk[3])

            def g(xs, a, c_):
                ec, st_, cx_ = setup(inp, c_, a[0], a[1])
                sax, say = air(inp, a[0], a[1], ec)
                return sweep(inp, st_, cx_, xs, a, sax, say, in_f, out_f, 1.0, None, coef, mode)
            _, lin = jax.linearize(g, xN, (jax.lax.stop_gradient(x[0]),
                                           jax.lax.stop_gradient(x[1])), c)
            return y, lin

        def jvp_tlf(inp, xf, c, in_f, out_f, carry_f, gk, vf, dc, mode="exact", omega=1.0):
            x = unflat(xf)
            y, lin = tlf_lin(inp, x, c, in_f, out_f, carry_f, gk, mode)
            dx = unflat(vf * dscale)
            t = frozen_tangent(lin, dx, (dx[0], dx[1]), dc, self.n_sweeps, omega)
            return flat(y), flat(t) / dscale

        def vjp_tlf(inp, xf, c, in_f, out_f, carry_f, gk, wf, mode="exact", omega=1.0):
            x = unflat(xf)
            y, lin = tlf_lin(inp, x, c, in_f, out_f, carry_f, gk, mode)

            def T(dx):
                return frozen_tangent(lin, dx, (dx[0], dx[1]), jnp.zeros(2), self.n_sweeps,
                                      omega)
            ct = jax.linear_transpose(T, y)(unflat(wf / dscale))[0]
            return flat(ct) * dscale
        self.p_jvp_tlf = partial(jax.jit(jvp_tlf), inp)
        self.p_vjp_tlf = partial(jax.jit(vjp_tlf), inp)
        # with the symmetrized (S) or Picard (P) stress tangent: built on first use
        self._programs = {}

        def programs(kind, mode, omega=1.0):
            """(jvp, vjp, jvp_sweep) of a tangent kind ("tlf" | "tlai") and stress mode; the
            jvp_sweep with tla="iter" is the one-sweep map with α_t from the sweep input.
            ``omega`` < 1 (tlf only): the damped frozen iteration of :func:`frozen_tangent`."""
            if kind != "tlf" and omega != 1.0:
                raise ValueError(f"the damped iteration (omega {omega}) is a TLf variant")
            key = (kind, mode, omega)
            if key not in self._programs:
                js = partial(jax.jit(partial(jvp_sweep, tla="iter", mode=mode)), inp)
                if kind == "tlf":
                    self._programs[key] = (
                        partial(jax.jit(partial(jvp_tlf, mode=mode, omega=omega)), inp),
                        partial(jax.jit(partial(vjp_tlf, mode=mode, omega=omega)), inp), js)
                else:
                    self._programs[key] = (
                        partial(jax.jit(partial(jvp_x, tla="iter", mode=mode)), inp),
                        partial(jax.jit(partial(vjp_x, tla="iter", mode=mode)), inp), js)
            return self._programs[key]
        self.programs = programs

        # TLai: α_t recomputed from each sweep's input iterate
        self.p_jvp_tlai = partial(jax.jit(partial(jvp_x, tla="iter")), inp)
        self.p_vjp_tlai = partial(jax.jit(partial(vjp_x, tla="iter")), inp)
        self.p_jvp_sweep_tlai = partial(jax.jit(partial(jvp_sweep, tla="iter")), inp)

        def stress_target_jvp(inp, xf, c, vf, dc):
            """Tangent of the stress the sweep relaxes towards, S(u) = P/(Δ+δmin)·(...) (the
            mEVP fixed point σ = S), at the flat raw state xf along the raw tangent vf and dc."""
            def S(x_, c_):
                ec, st, _ = setup(inp, c_, x_[0], x_[1])
                e11, e22, e12 = strain_rates(mesh, x_[0], x_[1])
                eps1, eps2 = e11 + e22, e11 - e22
                delta = _safe_sqrt(eps1 * eps1 + cfg.vale * (eps2 * eps2 + 4.0 * e12 * e12))
                pr = st.pressure_fac / cfg.mevp_det2 / (delta + cfg.delta_min)
                z = jnp.zeros_like(pr)
                return (jnp.where(st.ice_el, 0.5 * pr * (eps1 - delta + eps2 * cfg.vale), z),
                        jnp.where(st.ice_el, pr * e12 * cfg.vale, z),
                        jnp.where(st.ice_el, 0.5 * pr * (eps1 - delta - eps2 * cfg.vale), z))
            x = unflat(xf)
            return jax.jvp(S, (x, c), (unflat(vf), dc))[1]
        self.p_stress_target_jvp = partial(jax.jit(stress_target_jvp), inp)

        # ---- Task 23 (T1): the linearized STEADY operator of the sweep's fixed point ----
        def steady_u(inp, x_e, uv, mode):
            """One sweep's velocity update with the stress input at the sweep's own fixed point,
            σ_in = S_mode(u) (then σ' = S(u)): u' = G(u). Its fixed point is the implicit
            (backward-Euler) VP step; α and β cancel there. Anchor and wind stress at the
            step's entry state x_e, controls at 0."""
            ec, st, cx = setup(inp, jnp.zeros(2), x_e[0], x_e[1])
            sax, say = air(inp, x_e[0], x_e[1], ec)
            e11, e22, e12 = strain_rates(mesh, uv[0], uv[1])
            t11, t12, t22 = stress_terms(e11, e22, e12, st.pressure_fac, cx.vale, cx.delta_min,
                                         mode)
            d2 = cx.mevp_det2
            y = mevp_iterate_mode(cx, mesh, uv[0], uv[1], t11 / d2, t12 / d2, t22 / d2, st,
                                  x_e[0], x_e[1], inp["srf_u"], inp["srf_v"], sax, say, rdt,
                                  mode)
            return y[0], y[1]

        def steady_coef(inp, x_e, uv):
            """obd = 1 + β + drag and rf = rdt·f at the nodes, as in the sweep's node solve
            C·u' = rhs, C = [[obd, −rf], [rf, obd]]; and the node mass weight W = 1/mass."""
            from fesom_jax.config import DENSITY_0
            from fesom_jax.forcing import _safe_speed
            ec, st, cx = setup(inp, jnp.zeros(2), x_e[0], x_e[1])
            umod = _safe_speed(uv[0] - inp["srf_u"], uv[1] - inp["srf_v"])
            drag = rdt * cx.cd_oce_ice * umod * DENSITY_0 * st.inv_thickness
            w = jnp.where(st.mass > 0.0, 1.0 / jnp.where(st.mass > 0.0, st.mass, 1.0), 0.0)
            return 1.0 + cx.beta_evp + drag, rdt * mesh.coriolis_node, w

        def steady_apply(inp, x_e, uvN, d, mode):
            """A·d with A = C(u_N)·(I − G'(u_N)): the Jacobian of the steady residual
            C(u)·u − rhs(u) at its fixed point (u_N = the step's final iterate), with the
            stress derivative of ``mode`` (:func:`stress_terms`)."""
            gu, gv = jax.jvp(lambda uv: steady_u(inp, x_e, uv, mode), (uvN,), (d,))[1]
            obd, rf, _ = steady_coef(inp, x_e, uvN)
            ru, rv = d[0] - gu, d[1] - gv
            return obd * ru - rf * rv, rf * ru + obd * rv

        def steady_apply_t(inp, x_e, uvN, w, mode):
            """Aᵀ·w = (I − G'ᵀ)·Cᵀ·w."""
            obd, rf, _ = steady_coef(inp, x_e, uvN)
            cu, cv = obd * w[0] + rf * w[1], -rf * w[0] + obd * w[1]
            _, back = jax.vjp(lambda uv: steady_u(inp, x_e, uv, mode), uvN)
            bu, bv = back((cu, cv))[0]
            return cu - bu, cv - bv
        self._steady = {}

        def steady_programs(mode):
            """(A·d, Aᵀ·w, G) of the steady operator for a stress mode, compiled on first use."""
            if mode not in self._steady:
                self._steady[mode] = (
                    partial(jax.jit(partial(steady_apply, mode=mode)), inp),
                    partial(jax.jit(partial(steady_apply_t, mode=mode)), inp),
                    partial(jax.jit(partial(steady_u, mode=mode)), inp))
            return self._steady[mode]
        self.steady_programs = steady_programs
        self.p_steady_coef = partial(jax.jit(steady_coef), inp)
        self.p_tla_alpha = partial(jax.jit(lambda inp, x, kappa0: tla_coefficients(
            inp, setup(inp, jnp.zeros(2), x[0], x[1])[1], x[0], x[1], kappa0)), inp)
        self.p_iterates = partial(jax.jit(iterates), inp)
        self.p_grad_drift = jax.jit(jax.grad(lambda xf: drift(*unflat(xf)[:2])))
        self.p_drift = jax.jit(lambda xf: drift(*unflat(xf)[:2]))
        self.c0 = jnp.zeros(2)

        # ---- element geometry / diagnostics at the entry state ----
        en = np.asarray(mesh.elem_nodes)
        self.en = en
        self.lon_e, self.lat_e = lon[en[:, 0]], lat[en[:, 0]]
        e11, e22, _ = strain_rates(mesh, self.x0[0], self.x0[1])
        self.div_entry = np.asarray(e11 + e22)
        self.ice_nod, self.ice_el = ice_nod, ice_el
        self._strain_rates, self._safe_sqrt = strain_rates, _safe_sqrt

        # ---- aEVP α/β at the entry state; the per-element αβ of the stability number ----
        al0, be0 = (np.asarray(q) for q in self.p_alpha_beta(inp, self.x0[0], self.x0[1]))
        self.alpha0, self.beta0 = al0, be0
        if forward == "aevp":
            self.ab_e = al0 * be0[en].mean(axis=1)
            # γ/(αβ) of local_factor with this step's α (det2 folded in pressure_fac) and β
            inp["gam_c"] = jnp.asarray(cfg.ice_dt * (M_e / (1.0 + M_e * M_e))
                                       / (np.asarray(mesh.elem_area) / (1.0 + al0) * self.ab_e))
        elif forward == "const":
            A_ = float(alpha_const)
            self.ab_e = np.full(el, A_ * A_)
            # γ/(αβ) with α = β = A (pressure_fac carries det2 = 1/(1+A))
            inp["gam_c"] = jnp.asarray(cfg.ice_dt * (M_e / (1.0 + M_e * M_e))
                                       / (np.asarray(mesh.elem_area) / (1.0 + A_) * A_ * A_))
        else:
            self.ab_e = np.full(el, cfg.alpha_evp * cfg.beta_evp)

    # ---- factor sets -----------------------------------------------------------------------
    def ones(self):
        jnp = self.jnp
        return jnp.ones(self.el), jnp.ones(self.el), jnp.ones(self.n_sweeps), jnp.zeros(4)

    def switch_mask(self, thick_h=3.0, div_star=1e-7):
        msum = np.asarray(self.kw["m_ice"])[self.en].mean(axis=1)
        return self.ice_el & (msum > thick_h) & (self.div_entry < -div_star)

    def factors(self, form: str, tn=None, *, k=None, tn_map=None):
        """``(in_f, out_f, carry_f, gk)`` of a variant. ``tn_map`` = per-element T_N (switch);
        ``tn`` = κ for the local forms (``local``: f = min(1, κ/(γ/αβ)); ``local_cut``: f = 0
        where γ/αβ > κ), evaluated per sweep from the current iterate."""
        gk = np.zeros(4)
        jnp = self.jnp
        in_f, out_f, carry_f = (np.ones(self.el), np.ones(self.el), np.ones(self.n_sweeps))
        det1 = float(self.cfg.mevp_det1)
        if form in ("implicit", "explicit"):
            T = np.full(self.el, float(tn)) if tn_map is None else np.asarray(tn_map, float)
            if form == "implicit":
                out_f = np.where(self.ice_el, 1.0 / (1.0 + 1.0 / T), 1.0)
            else:
                if np.any(T * det1 <= 1.0):
                    raise ValueError("explicit damping needs T_N·det1 > 1")
                in_f = np.where(self.ice_el, 1.0 - 1.0 / (T * det1), 1.0)
        elif form == "whole":
            carry_f = np.full(self.n_sweeps, 1.0 - 1.0 / float(tn))
        elif form == "truncate":
            if 0 < k < self.n_sweeps:
                carry_f[self.n_sweeps - k - 1] = 0.0
        elif form == "zero_stress":
            out_f = np.zeros(self.el)
        elif form == "local":
            gk[0] = float(tn)
        elif form == "local_cut":
            gk[1] = float(tn)
        elif form == "frozen_alpha":
            gk[2] = 1.0
        elif form.startswith("tl"):
            if self.forward not in ("mevp", "const"):
                raise ValueError("TLa linearizes the mEVP forward")
            gk[3] = float(tn)
        elif form != "exact":
            raise ValueError(f"unknown form {form!r}")
        return jnp.asarray(in_f), jnp.asarray(out_f), jnp.asarray(carry_f), jnp.asarray(gk)

    # ---- element / node diagnostics of a vector ----------------------------------------------
    def element_table(self, x_in, idx):
        """Physical diagnostics of elements ``idx`` at the sweep input ``x_in`` (flat)."""
        cfg, mesh, en = self.cfg, self.mesh, self.en
        u, v = (np.asarray(q) for q in self.unflat(x_in)[:2])
        e11, e22, e12 = (np.asarray(q) for q in self._strain_rates(mesh, u, v))
        eps1, eps2 = e11 + e22, e11 - e22
        delta = np.sqrt(np.maximum(eps1 ** 2 + cfg.vale * (eps2 ** 2 + 4 * e12 ** 2), 0.0))
        a = np.asarray(self.kw["a_ice"])[en].mean(axis=1)
        mi = np.asarray(self.kw["m_ice"])[en].mean(axis=1)
        msn = np.asarray(self.kw["m_snow"])[en].mean(axis=1)
        P = float(cfg.pstar) * mi * np.exp(-cfg.c_pressure * (1.0 - a))
        M = cfg.rhoice * mi + cfg.rhosno * msn
        minv = M / (1.0 + M * M)
        A = np.asarray(mesh.elem_area)
        gamma = (P / (delta + cfg.delta_min)) * self.rdt * minv / A
        ab = self.ab_e
        bc = np.asarray(self.st0.bc_index)
        rows = []
        for i in idx:
            i = int(i)
            nodes = en[i]
            rows.append({
                "elem": i, "lon": float(self.lon_e[i]), "lat": float(self.lat_e[i]),
                "m_ice": float(mi[i]), "a_ice": float(a[i]),
                "h_ice": float(mi[i] / a[i]) if a[i] > 0 else None,
                "div": float(eps1[i]), "div_entry": float(self.div_entry[i]),
                "delta": float(delta[i]), "delta_over_dmin": float(delta[i] / cfg.delta_min),
                "gamma": float(gamma[i]), "gamma_over_ab": float(gamma[i] / ab[i]),
                "ice_el": bool(self.ice_el[i]),
                "mask_edge": bool(not self.ice_nod[nodes].all()),
                "coast": bool((bc[nodes] == 0).any()),
            })
        g_ice = gamma[self.ice_el]
        ab_ice = ab[self.ice_el]
        stats = {"gamma_over_ab_ice_pcts": {str(q): float(np.percentile(g_ice / ab_ice, q))
                                            for q in (50, 90, 99, 99.9)} if g_ice.size else {},
                 "n_ice_el_gamma_gt_ab": int((g_ice > ab_ice).sum()),
                 "gamma_percentile_of_rows": [float((g_ice < gamma[int(i)]).mean() * 100)
                                              for i in idx] if g_ice.size else []}
        return rows, stats, gamma

    def localize(self, vec, x_in, n_top=10):
        """Where a (scaled, possibly complex = (re, im)) vector lives."""
        re, im = vec
        re = np.asarray(re)
        im = np.zeros_like(re) if im is None else np.asarray(im)
        e = re * re + im * im
        o = self.offs
        e_n = e[o[0]:o[1]] + e[o[1]:o[2]]
        e_e = e[o[2]:o[3]] + e[o[3]:o[4]] + e[o[4]:o[5]]
        tot = float(e.sum())
        top_e = np.argsort(-e_e)[:n_top]
        top_n = np.argsort(-e_n)[:n_top]
        rows, stats, _ = self.element_table(x_in, top_e)
        ee_sorted = np.sort(e_e)[::-1]
        edge_el = ~self.ice_nod[self.en].all(axis=1) & self.ice_el
        bc = np.asarray(self.st0.bc_index)
        coast_el = (bc[self.en] == 0).any(axis=1)
        out = {
            "sigma_energy_fraction": float(e_e.sum() / tot) if tot > 0 else None,
            "top10_elem_energy_fraction": float(ee_sorted[:10].sum() / max(e_e.sum(), 1e-300)),
            "top100_elem_energy_fraction": float(ee_sorted[:100].sum() / max(e_e.sum(), 1e-300)),
            "mask_edge_elem_energy_fraction": float(e_e[edge_el].sum() / max(e_e.sum(), 1e-300)),
            "coast_elem_energy_fraction": float(e_e[coast_el].sum() / max(e_e.sum(), 1e-300)),
            "noice_elem_energy_fraction": float(e_e[~self.ice_el].sum() / max(e_e.sum(), 1e-300)),
            "north_energy_fraction": float(e_e[self.lat_e > 0].sum() / max(e_e.sum(), 1e-300)),
            "top_elements": rows, "gamma_stats": stats,
            "top_nodes": [{"node": int(i), "lon": float(self.lon_n[i]), "lat": float(self.lat_n[i]),
                           "a_ice": float(np.asarray(self.kw["a_ice"])[i]),
                           "m_ice": float(np.asarray(self.kw["m_ice"])[i]),
                           "ice_nod": bool(self.ice_nod[i]), "coast": bool(bc[i] == 0)}
                          for i in top_n],
        }
        return out, e_e.astype(np.float32), e_n.astype(np.float32)


def _where_str(loc: dict) -> str:
    """One-line location summary for the table."""
    if not loc:
        return "—"
    t = loc["top_elements"][0] if loc.get("top_elements") else {}
    return (f"σ-energy {loc['sigma_energy_fraction']:.2f}, top10 {loc['top10_elem_energy_fraction']:.2f}, "
            f"edge {loc['mask_edge_elem_energy_fraction']:.2f}; top el lat {t.get('lat', 0):.1f} "
            f"lon {t.get('lon', 0):.1f} m {t.get('m_ice', 0):.2f} a {t.get('a_ice', 0):.2f} "
            f"Δ/δmin {t.get('delta_over_dmin', 0):.2g} γ/αβ {t.get('gamma_over_ab', 0):.2g}")


# ---------------------------------------------------------------------------------------------
# the study


VARIANTS = (
    ("exact", "exact", None),
    ("imp3", "implicit", 3), ("imp7", "implicit", 7), ("imp15", "implicit", 15),
    ("imp30", "implicit", 30),
    ("exp3", "explicit", 3), ("exp7", "explicit", 7), ("exp15", "explicit", 15),
    ("exp30", "explicit", 30),
    ("imp_switch", "implicit", "switch"), ("exp_switch", "explicit", "switch"),
    ("whole4", "whole", 4), ("whole6", "whole", 6), ("whole7", "whole", 7),
    ("trunc8", "truncate", 8), ("trunc32", "truncate", 32),
    ("zero_stress", "zero_stress", None),
    # aEVP: α/β held fixed in the tangent (pseudo-time parameters); "exact" differentiates them
    ("afrozen", "frozen_alpha", None),
    # mEVP forward, tangent with per-element α_t = max(α, α·sqrt(s/κ0)), β_t = max α_t at nodes
    *((f"TLa{k:g}", "tla", k) for k in (0.1, 0.25, 0.5)),
    # the same with α_t from each sweep's input iterate
    *((f"TLai{k:g}", "tlai", k) for k in (0.05, 0.1, 0.25)),
    # the sweep tangent frozen at the final iterate (TLa coefficients there), applied 120 times;
    # κ0 = inf: the plain mEVP coefficients, frozen
    *((f"TLf{k:g}", "tlf", k) for k in (0.05, 0.1, 0.25, 1.0, float("inf"))),
    # TLf with the symmetrized (replacement pressure from stop_gradient(Δ)) and the Picard
    # (ζ, η and P_r all from stop_gradient(Δ)) stress tangent; TLai symmetrized
    *((f"TLfS{k:g}", "tlfS", k) for k in (0.1, 0.25, 1.0, float("inf"))),
    *((f"TLfP{k:g}", "tlfP", k) for k in (0.1, 0.25, 1.0, float("inf"))),
    *((f"TLaiS{k:g}", "tlaiS", k) for k in (0.1, 0.25)),
    # Picard with per-sweep coefficients; partial viscosity derivative θ (frozen and per-sweep)
    *((f"TLaiP{k:g}", "tlaiP", k) for k in (0.1, 0.25, 1.0)),
    *((f"TLfT{t:g}", f"tlfT{t:g}", 1.0) for t in (0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99)),
    # Task 23 (T2): the frozen iteration damped by averaging (frozen_tangent's ω) at θ = 1, with
    # the replacement pressure frozen (S) and live (exact); and the production θ 0.97 damped
    *((f"TLfS1w{w:g}", f"tlfSw{w:g}", 1.0) for w in (0.5, 0.8)),
    *((f"TLf1w{w:g}", f"tlfw{w:g}", 1.0) for w in (0.5, 0.8)),
    ("TLfT0.97w0.8", "tlfT0.97w0.8", 1.0), ("TLfT0.99w0.8", "tlfT0.99w0.8", 1.0),
    *((f"TLaiT{t:g}k{k:g}", f"tlaiT{t:g}", k) for k in (0.25, 1.0) for t in (0.25, 0.5)),
    # local damping from the per-sweep mEVP stability number (a preview of fallback B)
    ("Bloc1", "local", 1.0), ("Bloc0.25", "local", 0.25),
    *((f"Bloc{k:g}", "local", k) for k in (0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8)),
    ("Bcut1", "local_cut", 1.0), ("Bcut4", "local_cut", 4.0),
)
CONTROLS = ("lnp", "lncd")
PASS_GROWTH_DAY = 10.0
# The verdict's pattern correlation is taken against the central-FD field at this h: at
# h = 1e-3 the FD field itself is ill-defined (it correlates 0.13-0.27 with the h = 1e-2 one;
# a one-ulp change of P* already moves u by 1e-7 m/s at single nodes). The correlation against
# the FD field of the scalar ratio's h (``c_*``) is still reported, for continuity.
VERDICT_FD_H = 1e-2


def _finite(x):
    return x is not None and isinstance(x, float) and math.isfinite(x)


def verdict(row) -> str:
    stable = _finite(row.get("growth_day")) and row["growth_day"] <= PASS_GROWTH_DAY \
        and not row.get("chain_blowup")
    resp = all(_finite(row.get(f"r_{c}_1")) and 0.5 <= row[f"r_{c}_1"] <= 2.0
               and _finite(row.get(f"c2_{c}_1")) and row[f"c2_{c}_1"] > 0.8 for c in CONTROLS)
    return "PASS" if (stable and resp) else "FAIL"


def tl_kind_mode(form: str):
    """("tla" | "tlai" | "tlf", stress mode) of a TL* form, e.g. "tlaiP" → ("tlai", "P"),
    "tlfT0.5" → ("tlf", "T0.5"), "tlfSw0.8" → ("tlf", "S") (the damping suffix "w<ω>" is
    :func:`tl_omega`'s); any other form → (form, "exact")."""
    for k in ("tlai", "tla", "tlf"):
        if form.startswith(k):
            rest = re.sub(r"w[0-9.]+$", "", form[len(k):])
            return k, {"": "exact", "S": "S", "P": "P"}.get(rest, rest)
    return form, "exact"


def tl_omega(form: str) -> float:
    """The damping ω of a TL* form's frozen iteration: "tlfSw0.8" → 0.8; 1.0 without a suffix."""
    m = re.search(r"w([0-9.]+)$", form) if form.startswith("tl") else None
    return float(m.group(1)) if m else 1.0


def aevp_report(M, args) -> dict:
    """α/β statistics, the aEVP forward against mEVP's, and both forwards' last-sweep residual."""
    ice_el, ice_nod = M.ice_el, M.ice_nod
    al, be = M.alpha0, M.beta0
    a_ice = al[ice_el]
    gamma = M.element_table(M.flat(M.x0), [])[2]
    out = {"c_aevp": args.aevp_c, "alpha_min": args.aevp_alpha_min,
           "alpha_from": "entry velocity of the captured step (= the previous step's final u); "
                         "a_ice, m_ice are the step's entry values (post-thermo/advection)",
           "alpha_ice_pcts": {str(q): float(np.percentile(a_ice, q))
                              for q in (0, 10, 50, 90, 99, 99.9, 100)},
           "frac_ice_el_at_alpha_min": float((a_ice <= args.aevp_alpha_min).mean()),
           "beta_ice_pcts": {str(q): float(np.percentile(be[ice_nod], q))
                             for q in (0, 50, 90, 99, 100)},
           "gamma_over_ab_aevp_ice_pcts": {str(q): float(np.percentile(
               (gamma / M.ab_e)[ice_el], q)) for q in (50, 90, 99, 99.9, 100)}}
    ones = M.ones()
    ya = [np.asarray(q) for q in M.p_fwd0(M.x0, *M.air0, *ones)]
    ym = [np.asarray(q) for q in M.p_fwd0_other(M.x0, *M.air0, *ones)]
    du = np.hypot(ya[0] - ym[0], ya[1] - ym[1])[ice_nod]
    sig = lambda y: np.sqrt(sum(q ** 2 for q in y[2:])[ice_el] / 3.0)          # noqa: E731
    out["forward_vs_mevp"] = {
        "D_aevp": float(M.drift(ya[0], ya[1])), "D_mevp": float(M.drift(ym[0], ym[1])),
        "rms_du_ice_nod": float(np.sqrt(np.mean(du ** 2))), "max_du": float(du.max()),
        "rms_speed_mevp": float(np.sqrt(np.mean((ym[0] ** 2 + ym[1] ** 2)[ice_nod]))),
        "rms_sigma_aevp": float(np.sqrt(np.mean(sig(ya) ** 2))),
        "rms_sigma_mevp": float(np.sqrt(np.mean(sig(ym) ** 2))),
        "rms_dsigma": float(np.sqrt(np.mean(sum((p - q) ** 2 for p, q in zip(ya[2:], ym[2:]))
                                            [ice_el] / 3.0)))}
    conv = {}
    for name, prog, y in (("aevp", M.p_iterates, ya), ("mevp", M.p_iterates_other, ym)):
        its = [np.asarray(q) for q in prog(M.x0, M.c0)]          # its[p] = input of sweep p+1
        n = its[0].shape[0]

        def res(k, nxt):   # change made by sweep k (1-based): nxt − input of sweep k
            d = np.hypot(nxt[0] - its[0][k - 1], nxt[1] - its[1][k - 1])[ice_nod]
            ds = np.sqrt(sum((nxt[j] - its[j][k - 1]) ** 2 for j in (2, 3, 4))[ice_el] / 3.0)
            return {"rms_du": float(np.sqrt(np.mean(d ** 2))), "max_du": float(d.max()),
                    "rms_dsigma": float(np.sqrt(np.mean(ds ** 2)))}
        tot = np.hypot(y[0] - its[0][0], y[1] - its[1][0])[ice_nod]
        conv[name] = {"last_sweep": res(n, y),
                      "sweep_60": res(60, [q[60] for q in its]) if n > 60 else None,
                      "rms_du_whole_step": float(np.sqrt(np.mean(tot ** 2)))}
        conv[name]["last_over_whole"] = conv[name]["last_sweep"]["rms_du"] / max(
            conv[name]["rms_du_whole_step"], 1e-300)
        del its
    out["convergence"] = conv
    return out


def run(args) -> int:
    if VERDICT_FD_H not in args.fd_h:
        raise SystemExit(f"--fd-h must include the verdict's h = {VERDICT_FD_H:g}")
    out_dir = _refuse_inside_checkout(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.date.replace("-", "")
    json_path = out_dir / f"modes_{tag}.json"
    t_start = time.time()
    if args.forward == "aevp" and args.variants and any(
            f in ("local", "local_cut") for v, f, _ in VARIANTS if v in args.variants):
        raise SystemExit("the local-damping rows are an mEVP study; not with --forward aevp")
    M = IceMap(args.date, n_sweeps=args.n_sweeps, forward=args.forward, aevp_c=args.aevp_c,
               aevp_alpha_min=args.aevp_alpha_min)
    jax, jnp = M.jax, M.jnp
    dev = jax.devices()[0]
    on_cpu = dev.platform == "cpu"
    doc = {"provenance": {
        "script": "scripts/adjoint/ice_modes.py", "git_head": git_head(), "checkout": str(ROOT),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "device": getattr(dev, "device_kind", str(dev)), "platform": dev.platform,
        "jax": jax.__version__, "date": args.date, "restart": str(M.model.restart),
        "config": str(M.model.config_path), "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_sweeps": M.n_sweeps, "rdt": M.rdt, "alpha_evp": M.cfg.alpha_evp,
        "beta_evp": M.cfg.beta_evp, "det1": M.cfg.mevp_det1, "pstar": M.cfg.pstar,
        "cd_oce_ice": M.cfg.cd_oce_ice, "settings": vars(args),
        "scale_U_m_per_s": M.U, "scale_S_N_per_m": M.S,
        "n_box_nodes": int(M.box.sum()), "n_ice_nodes": int(M.ice_nod.sum()),
        "n_ice_elems": int(M.ice_el.sum()), "build_s": M.build_s, "capture_s": M.capture_s,
        "captured_exch": M.exch_kind,
        "drift_definition": "D = sum(w a |u|)/sum(w a), ice nodes lat>=70N, w = node area",
        "gamma_definition": "gamma = P/(Delta+dmin) * rdt * M/(1+M^2) / A_e (P = P* m e^{-C(1-a)}, "
                            "M = rho_i m + rho_s m_s, element means), compared with alpha*beta; "
                            "an O(1) mesh constant is not included",
        "corr_definition": "weighted (w a) Pearson correlation of the stacked (du, dv) at box nodes",
        "verdict_fd_h": VERDICT_FD_H,
    }, "variants": {}}

    def dump():
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=lambda o: o.tolist()
                                  if hasattr(o, "tolist") else str(o)))
        os.replace(tmp, json_path)

    # ---- self-check: own 120 sweeps == the model step's u_ice ----
    ones = M.ones()
    x0f = M.flat(M.x0)
    t0 = time.time()
    y = M.p_fwd0(M.x0, *M.air0, *ones)
    jax.block_until_ready(y)
    fwd_compile_s = time.time() - t0
    t0 = time.time()
    y2 = M.p_fwd0(M.x0, *M.air0, *ones)
    jax.block_until_ready(y2)
    fwd_s = time.time() - t0
    yc = [np.asarray(q) for q in M.p_fwd(M.x0, M.c0, *ones)]
    su, sv = (np.asarray(q) for q in M.step_u_ice)
    mo = [np.asarray(q) for q in M.mevp_out]
    yn = [np.asarray(q) for q in y]
    y2n = [np.asarray(q) for q in y2]
    sc = {
        "u_bitwise_vs_step": bool(np.array_equal(yn[0], su) and np.array_equal(yn[1], sv)),
        "all_bitwise_vs_mevp_in_step": bool(all(np.array_equal(a, b) for a, b in zip(yn, mo))),
        "max_abs_diff_u_vs_step": float(max(np.max(np.abs(yn[0] - su)), np.max(np.abs(yn[1] - sv)))),
        "max_abs_u": float(max(np.max(np.abs(su)), np.max(np.abs(sv)))),
        "repeat_max_abs_diff_u": float(max(np.max(np.abs(yn[0] - y2n[0])), np.max(np.abs(yn[1] - y2n[1])))),
        "control_program_at_c0_max_abs_diff_u": float(max(np.max(np.abs(yc[0] - su)),
                                                          np.max(np.abs(yc[1] - sv)))),
        "control_program_at_c0_D_diff": float(abs(float(M.drift(yc[0], yc[1]))
                                                  - float(M.drift(su, sv)))),
        "wind_stress_arg_is_entry_velocity": M.air_check["u_ice_arg_is_entry"],
        "wind_stress_recomputed_bitwise": M.air_check["stress_bitwise"],
        "fwd_compile_s": fwd_compile_s, "fwd_s_per_step": fwd_s,
    }
    doc["self_check"] = sc
    print(f"[self-check] {sc}", flush=True)
    dump()
    if M.forward == "aevp":
        doc["aevp"] = aevp_report(M, args)
        print(f"[aevp] {json.dumps(doc['aevp'])}", flush=True)
        dump()
    if on_cpu and args.n_sweeps is None and M.forward == "mevp" and not sc["u_bitwise_vs_step"]:
        print("FATAL: on CPU the script's 120 sweeps do not reproduce the step's u_ice bitwise",
              flush=True)
        return 6

    # ---- iterates of the exact forward (inputs of each sweep) ----
    its = M.p_iterates(M.x0, M.c0)
    sweep_ids = [s for s in args.sweeps if 1 <= s <= M.n_sweeps]
    x_in = {s: M.flat(tuple(q[s - 1] for q in its)) for s in sweep_ids}
    del its
    rng = np.random.default_rng(args.seed)
    nflat = int(M.offs[-1])

    def rnd():
        return jnp.asarray(rng.standard_normal(nflat))

    # ---- FD references (nonlinear, independent of the variant) ----
    fd = {}
    base1 = M.p_fwd(M.x0, M.c0, *ones)
    xN_f = M.flat(base1)                    # the final iterate of the step (TLf's freezing point)
    D_base1 = float(M.p_drift(M.flat(base1)))
    D_rep = float(M.p_drift(M.flat(M.p_fwd(M.x0, M.c0, *ones))))
    fd["one_step"] = {"D_base": D_base1, "D_repeat_diff": abs(D_rep - D_base1)}
    unit = {"lnp": jnp.asarray([1.0, 0.0]), "lncd": jnp.asarray([0.0, 1.0])}
    fd_fields1, fd_fieldsd = {}, {}

    def chain_fwd(c, n):
        x = M.x0
        xs = [M.flat(x)]
        for _ in range(n):
            x = M.p_fwd(x, c, *ones)
            xs.append(M.flat(x))
        return xs
    n_chain = args.chain_steps
    t0 = time.time()
    chain_base = chain_fwd(M.c0, n_chain)
    jax.block_until_ready(chain_base[-1])
    chain_fwd_s = (time.time() - t0) / max(n_chain, 1)
    D_based = float(M.p_drift(chain_base[-1]))
    D_based_rep = float(M.p_drift(chain_fwd(M.c0, n_chain)[-1])) if args.repeat_chain else None
    fd["chain"] = {"D_base": D_based, "D_repeat_diff": None if D_based_rep is None
                   else abs(D_based_rep - D_based), "fwd_s_per_step": chain_fwd_s}
    for cname in CONTROLS:
        for h in args.fd_h:
            yp = M.flat(M.p_fwd(M.x0, M.c0 + h * unit[cname], *ones))
            ym = M.flat(M.p_fwd(M.x0, M.c0 - h * unit[cname], *ones))
            g = (float(M.p_drift(yp)) - float(M.p_drift(ym))) / (2 * h)
            fd_fields1[(cname, h)] = np.asarray((yp - ym) / (2 * h))
            cp = chain_fwd(M.c0 + h * unit[cname], n_chain)[-1]
            cm = chain_fwd(M.c0 - h * unit[cname], n_chain)[-1]
            gd = (float(M.p_drift(cp)) - float(M.p_drift(cm))) / (2 * h)
            fd_fieldsd[(cname, h)] = np.asarray((cp - cm) / (2 * h))
            fd.setdefault(cname, {})[str(h)] = {"one_step": g, "chain": gd}
        vals1 = [fd[cname][str(h)]["one_step"] for h in args.fd_h]
        valsd = [fd[cname][str(h)]["chain"] for h in args.fd_h]

        def pick(vals, kind):
            spread = abs(vals[0] - vals[-1]) / max(abs(vals[0]), abs(vals[-1]), 1e-300)
            noise = fd[kind]["D_repeat_diff"]
            sig = abs(vals[0]) * 2 * args.fd_h[0]
            if spread <= 0.10:
                h_used = args.fd_h[0]
            else:
                h_used = args.fd_h[-1] if (noise is not None and noise > 0.01 * sig) else args.fd_h[0]
            return {"spread": spread, "flag_gt_10pct": spread > 0.10, "h_used": h_used,
                    "value": vals[list(args.fd_h).index(h_used)]}
        fd[cname]["one_step_ref"] = pick(vals1, "one_step")
        fd[cname]["chain_ref"] = pick(valsd, "chain")
    doc["fd"] = fd
    print(f"[fd] {json.dumps(fd)}", flush=True)
    dump()

    o = M.offs
    box = M.box
    wbox = M.wa[box]

    def uv_box(f):
        f = np.asarray(f)
        return np.concatenate([f[o[0]:o[1]][box], f[o[1]:o[2]][box]])

    def corr(t_flat_raw, fd_flat):
        return weighted_corr(uv_box(t_flat_raw), uv_box(fd_flat), np.concatenate([wbox, wbox]))

    # the FD fields' own consistency across h (weighted correlation of each pair)
    hs = list(args.fd_h)
    doc["fd_field_consistency"] = {
        c: {kind: {f"{a:g}~{b:g}": corr(F[(c, a)], F[(c, b)])
                   for i, a in enumerate(hs) for b in hs[i + 1:]}
            for kind, F in (("one_step", fd_fields1), ("chain", fd_fieldsd))}
        for c in CONTROLS}
    print(f"[fd-consistency] {json.dumps(doc['fd_field_consistency'])}", flush=True)

    # ---- switch statistics ----
    sw = M.switch_mask(3.0, args.div_star)
    msum = np.asarray(M.kw["m_ice"])[M.en].mean(axis=1)
    doc["switch"] = {"div_star": args.div_star, "thick_h": 3.0, "n_selected": int(sw.sum()),
                     "n_thick": int((M.ice_el & (msum > 3.0)).sum()),
                     "n_converging": int((M.ice_el & (M.div_entry < -args.div_star)).sum()),
                     "n_ice_el": int(M.ice_el.sum()),
                     "div_entry_pcts_thick": {str(q): float(np.percentile(
                         M.div_entry[M.ice_el & (msum > 3.0)], q)) for q in (1, 10, 50, 90, 99)}
                     if (M.ice_el & (msum > 3.0)).any() else {}}
    tn_switch = np.where(sw, 3.0, 7.0)

    fields = {"lon_e": M.lon_e.astype(np.float32), "lat_e": M.lat_e.astype(np.float32),
              "lon_n": M.lon_n.astype(np.float32), "lat_n": M.lat_n.astype(np.float32),
              "ice_el": M.ice_el, "switch": sw}
    grad_end = M.p_grad_drift  # flat raw-space gradient

    names = [v for v in VARIANTS if (not args.variants or v[0] in args.variants)]
    rows = []
    for vname, form, tn in names:
        tv = time.time()
        if tn == "switch":
            fac = M.factors(form, tn_map=tn_switch)
        elif form == "truncate":
            fac = M.factors(form, k=tn)
        else:
            fac = M.factors(form, tn)
        row = {"variant": vname, "form": form, "tn": "3/7 switch" if tn == "switch" else tn}
        res = {"row": row}
        kind, smode = tl_kind_mode(form)
        omega = tl_omega(form)
        js_iter = M.p_jvp_sweep_tlai
        if kind in ("tla", "tlai", "tlf"):
            if form == "tla":
                p_jvp, p_vjp, p_jvp_sweep = M.p_jvp_tla, M.p_vjp_tla, M.p_jvp_sweep_tla
            elif form == "tlf":
                p_jvp, p_vjp, p_jvp_sweep = M.p_jvp_tlf, M.p_vjp_tlf, None
            elif form == "tlai":
                p_jvp, p_vjp, p_jvp_sweep = M.p_jvp_tlai, M.p_vjp_tlai, M.p_jvp_sweep_tlai
            else:
                p_jvp, p_vjp, js_iter = M.programs(kind, smode, omega)
                p_jvp_sweep = js_iter if kind == "tlai" else None
            res["stress_mode"] = smode
            res["omega"] = omega

            def raised(xs):
                al_t = np.asarray(M.p_tla_alpha(xs, float(tn))[0])
                up = M.ice_el & (al_t > (M.alpha_const or M.cfg.alpha_evp))
                return {"frac_ice_el_raised": float(up.sum() / max(M.ice_el.sum(), 1)),
                        "n_raised": int(up.sum()),
                        "range_raised": [float(al_t[up].min()), float(al_t[up].max())]
                        if up.any() else None,
                        "pcts_raised": {str(q): float(np.percentile(al_t[up], q))
                                        for q in (50, 90, 99)} if up.any() else {}}
            res["alpha_t_entry"] = raised(M.x0)
            if kind == "tlf":
                res["alpha_t_at_xN"] = raised(M.unflat(xN_f))
            if kind == "tlai":
                res["alpha_t_by_sweep_input"] = {str(s_): raised(M.unflat(x_in[s_]))
                                                 for s_ in sweep_ids}
            print(f"[{vname}] alpha_t {res['alpha_t_entry']}", flush=True)
        else:
            p_jvp, p_vjp, p_jvp_sweep = M.p_jvp, M.p_vjp, M.p_jvp_sweep

        def J(v, x=x0f, fac=fac):
            return p_jvp(x, M.c0, *fac, v, jnp.zeros(2))[1]

        def JT(w, x=x0f, fac=fac):
            return p_vjp(x, M.c0, *fac, w)

        # step map: top singular value and dominant eigenvalues
        try:
            s1, hist, vsv = power_svd(J, JT, rnd(), args.n_pow)
            row["sigma1_step"] = s1
            res["svd_history"] = hist
            loc_sv, ee, _ = M.localize((vsv, None), x0f)
            res["svd_vector_location"] = loc_sv
            fields[f"{vname}_svd_e"] = ee
        except (ValueError, FloatingPointError) as e:
            res["svd_error"] = str(e)
        try:
            v = rnd()
            for _ in range(args.arnoldi_warm_step):
                v = J(v)
                v = v / _norm(v)
            rz = arnoldi(J, v, args.arnoldi_k)
            dm = dominant(rz)
            row["lam_step"] = dm["abs"]
            res["step_eig"] = dm
        except (ValueError, FloatingPointError) as e:
            res["step_eig_error"] = str(e)

        # single-sweep spectra (the variant's tangent sweep map)
        if form in ("exact", "implicit", "explicit", "zero_stress", "local", "local_cut") \
                or args.all_sweep_spectra:
            res["sweep_eig"] = {}
            for s in sweep_ids:
                cf = fac[2][s - 1]

                def Js(v, s=s, cf=cf):
                    return p_jvp_sweep(x_in[s], x0f, M.c0, fac[0], fac[1], cf, fac[3], v)
                try:
                    v = rnd()
                    for _ in range(args.arnoldi_warm_sweep):
                        v = Js(v)
                        v = v / _norm(v)
                    rz = arnoldi(Js, v, args.arnoldi_k)
                    dm = dominant(rz)
                    vec = rz["vector"](0)
                    loc, ee, en_ = M.localize(vec, x_in[s])
                    dm["location"] = loc
                    res["sweep_eig"][str(s)] = dm
                    if vname == "exact" or s == args.table_sweep:
                        fields[f"{vname}_sweep{s}_e"] = ee
                        fields[f"{vname}_sweep{s}_n"] = en_
                except (ValueError, FloatingPointError) as e:
                    res["sweep_eig"][str(s)] = {"error": str(e)}
            ts = res["sweep_eig"].get(str(args.table_sweep), {})
            if "type" in ts:
                row["mode_type"] = f"{ts['type']} λ={ts['re']:.4f}{ts['im']:+.4f}i"
                row["mode_where"] = _where_str(ts["location"])

        # TLf: spectrum of the one frozen sweep map at x_N (the map the step tangent iterates)
        if kind in ("tlf", "tlai"):
            def Jf(v, omega=omega):          # the damped map (1 − ω)·I + ω·J for ω < 1
                y = js_iter(xN_f, x0f, M.c0, fac[0], fac[1], 1.0, fac[3], v)
                return y if omega == 1.0 else (1.0 - omega) * v + omega * y
            try:
                v = rnd()
                for _ in range(20):
                    v = Jf(v)
                    v = v / _norm(v)
                rz = arnoldi(Jf, v, args.arnoldi_k)
                dm = dominant(rz)
                loc, ee, _ = M.localize(rz["vector"](0), xN_f)
                dm["location"] = loc
                res["frozen_sweep_eig"] = dm
                row["mode_type"] = f"frozen {dm['type']} λ={dm['re']:.4f}{dm['im']:+.4f}i"
                row["mode_where"] = _where_str(loc)
            except (ValueError, FloatingPointError) as e:
                res["frozen_sweep_eig"] = {"error": str(e)}

        # retained response, one step (TL w.r.t. each control, from the entry state)
        g1 = np.asarray(grad_end(M.flat(base1)))
        for cname in CONTROLS:
            _, t = p_jvp(x0f, M.c0, *fac, jnp.zeros(nflat), unit[cname])
            t_raw = np.asarray(t) * np.asarray(M.dscale)
            tl = float(np.dot(g1, t_raw))
            ref = fd[cname]["one_step_ref"]
            row[f"r_{cname}_1"] = tl / ref["value"] if ref["value"] != 0 else None
            row[f"c_{cname}_1"] = corr(t_raw, fd_fields1[(cname, ref["h_used"])])
            row[f"c2_{cname}_1"] = corr(t_raw, fd_fields1[(cname, VERDICT_FD_H)])
            res[f"tl_{cname}_1"] = tl
            res[f"r_{cname}_1_all_h"] = {str(h): tl / fd[cname][str(h)]["one_step"]
                                         for h in args.fd_h if fd[cname][str(h)]["one_step"] != 0}
            res[f"c_{cname}_1_all_h"] = {str(h): corr(t_raw, fd_fields1[(cname, h)])
                                         for h in args.fd_h}

        # 1-day chain: TL for each control
        for cname in CONTROLS:
            dx = jnp.zeros(nflat)
            blow = None
            for k in range(n_chain):
                _, dx = p_jvp(chain_base[k], M.c0, *fac, dx, unit[cname])
                nk = _norm(dx)
                if not math.isfinite(nk) or nk > 1e200:
                    blow = k + 1
                    break
            if blow is None:
                t_raw = np.asarray(dx) * np.asarray(M.dscale)
                gd = np.asarray(grad_end(chain_base[-1]))
                tl = float(np.dot(gd, t_raw))
                ref = fd[cname]["chain_ref"]
                row[f"r_{cname}_d"] = tl / ref["value"] if ref["value"] != 0 else None
                row[f"c_{cname}_d"] = corr(t_raw, fd_fieldsd[(cname, ref["h_used"])])
                row[f"c2_{cname}_d"] = corr(t_raw, fd_fieldsd[(cname, VERDICT_FD_H)])
                res[f"tl_{cname}_d"] = tl
                # how far the chain's δσ has relaxed to the stress the sweep converges to
                dS = [np.asarray(q) for q in M.p_stress_target_jvp(
                    chain_base[-1], M.c0, jnp.asarray(t_raw), unit[cname])]
                o5 = M.offs
                ie = M.ice_el
                dsig = [t_raw[o5[k]:o5[k + 1]] for k in (2, 3, 4)]
                nS = math.sqrt(sum(float(np.sum(q[ie] ** 2)) for q in dS))
                nd = math.sqrt(sum(float(np.sum((p_[ie] - q[ie]) ** 2)) for p_, q in zip(dsig, dS)))
                ns = math.sqrt(sum(float(np.sum(p_[ie] ** 2)) for p_ in dsig))
                res[f"stress_relax_{cname}_d"] = {"rel_misfit": nd / nS if nS > 0 else None,
                                                  "norm_ratio": ns / nS if nS > 0 else None}
                res[f"r_{cname}_d_all_h"] = {str(h): tl / fd[cname][str(h)]["chain"]
                                             for h in args.fd_h if fd[cname][str(h)]["chain"] != 0}
                res[f"c_{cname}_d_all_h"] = {str(h): corr(t_raw, fd_fieldsd[(cname, h)])
                                             for h in args.fd_h}
            else:
                res[f"tl_{cname}_d_blowup_step"] = blow

        # 1-day chain: reverse growth of the cotangent of D(u_n), and the backward cost
        lam = np.asarray(grad_end(chain_base[-1])) * np.asarray(M.dscale)   # dual-scaled
        lam = jnp.asarray(lam)
        n_end = _norm(lam)
        trace = [n_end]
        times = []
        blow = None
        for k in range(n_chain - 1, -1, -1):
            t1 = time.time()
            lam = p_vjp(chain_base[k], M.c0, *fac, lam)
            nk = _norm(lam)                                               # syncs
            times.append(time.time() - t1)
            trace.append(nk)
            if not math.isfinite(nk) or nk > 1e200:
                blow = n_chain - k
                break
        res["reverse_trace"] = trace
        row["chain_blowup"] = blow is not None
        if blow is None and n_end > 0 and trace[-1] > 0:
            lg = math.log(trace[-1] / n_end) * 48.0 / n_chain           # ln growth per day
            res["log10_growth_day"] = lg / math.log(10.0)
            row["growth_day"] = math.exp(lg) if lg < 700 else float("inf")
            row["growth_step_chain"] = math.exp(lg / 48.0)
        else:
            row["growth_day"] = float("inf")
            res["reverse_blowup_after_steps"] = blow
            if len(trace) > 1 and trace[0] > 0:
                last = trace[-1] if math.isfinite(trace[-1]) else 1e200
                row["growth_step_chain"] = (last / trace[0]) ** (1.0 / (len(trace) - 1))
        steady = times[1:] if len(times) > 1 else times
        row["bwd_s"] = float(np.median(steady)) if steady else None
        res["bwd_s_first_call"] = times[0] if times else None
        row["verdict"] = verdict(row)
        res["wall_s"] = time.time() - tv
        doc["variants"][vname] = res
        rows.append(row)
        print(f"[{vname}] {json.dumps(row, default=str)}  ({res['wall_s']:.1f}s)", flush=True)
        dump()
        np.savez_compressed(out_dir / f"modes_{tag}_fields.npz", **fields)

    md = [f"# mEVP tangent mode study — {args.date}\n",
          f"git {doc['provenance']['git_head'][:12]}, job {doc['provenance']['slurm_job']}, "
          f"{doc['provenance']['device']}, {M.n_sweeps} sweeps, chain {n_chain} steps.\n",
          f"Self-check: {json.dumps(sc)}\n",
          f"FD (h used per control): " + ", ".join(
              f"{c}: 1 step {fd[c]['one_step_ref']['value']:.4g} (h {fd[c]['one_step_ref']['h_used']},"
              f" spread {fd[c]['one_step_ref']['spread']:.2%}), 1 day {fd[c]['chain_ref']['value']:.4g}"
              f" (h {fd[c]['chain_ref']['h_used']}, spread {fd[c]['chain_ref']['spread']:.2%})"
              for c in CONTROLS) + "\n",
          f"Switch: {json.dumps(doc['switch'])}\n",
          f"Pass: chain reverse growth/day ≤ {PASS_GROWTH_DAY}, no blow-up, and for both controls "
          f"0.5 ≤ r ≤ 2 and corr > 0.8 against the FD field at h {VERDICT_FD_H:g} at one step.\n",
          table_markdown(rows)]
    (out_dir / f"modes_{tag}.md").write_text("\n".join(md))
    doc["table_rows"] = rows
    doc["wall_s"] = time.time() - t_start
    dump()
    print(f"[done] {json_path}  wall {doc['wall_s']:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# Task 23 (T1): is the linearized STEADY problem positive?


def operator_study(args) -> int:
    """Danilov's question: does the full viscosity derivative (θ = 1) break the linearized
    implicit VP problem itself (then a VP solver and its adjoint share the trouble), or only
    the mEVP forward-backward pseudo-time iteration that converges to it?

    At the step's final iterate u_N, for each stress mode (:func:`stress_terms`):
      (a) per ice element, the smallest eigenvalue of the symmetric part of the stress Jacobian
          in the pairing σ:ε (:func:`element_pairing_jacobians`), over the viscous scale
          ζ = P/(Δ+δmin);
      (b) the symmetric part Ŝ of Â = W^{1/2}·A·W^{-1/2} on the active ice nodes, A the
          Jacobian of the steady residual (IceMap.steady_programs), W the node mass weight in
          which the stress divergence is the transpose of the strain rate: Lanczos Ritz values
          (upper bounds of λ_min), and the Rayleigh quotient of a velocity field supported on
          one element that produces that element's softest strain direction (also an upper
          bound: a negative one proves Ŝ indefinite).
    λ_min(Ŝ) ≥ 1 makes the implicit step's tangent δu^n → A^{-1}·δu^n non-expansive in the mass
    norm (the anchor u^n enters with weight 1); λ_min < 0 makes the linear problem indefinite."""
    out_dir = _refuse_inside_checkout(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.date.replace("-", "")
    json_path = out_dir / f"operator_{tag}.json"
    t_start = time.time()
    M = IceMap(args.date, n_sweeps=args.n_sweeps)
    jax, jnp = M.jax, M.jnp
    dev = jax.devices()[0]
    cfg, mesh = M.cfg, M.mesh
    modes = [m for m in args.modes.split(",") if m]
    ones = M.ones()
    xN = M.p_fwd(M.x0, M.c0, *ones)
    uvN = (xN[0], xN[1])
    x_e = (M.x0[0], M.x0[1])
    doc = {"provenance": {
        "script": "scripts/adjoint/ice_modes.py operator", "git_head": git_head(),
        "checkout": str(ROOT), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(), "device": getattr(dev, "device_kind", str(dev)),
        "jax": jax.__version__, "date": args.date, "n_sweeps": M.n_sweeps,
        "time": datetime.datetime.now().isoformat(timespec="seconds"), "settings": vars(args),
        "linearization_point": "u_N, the step's final iterate (120 sweeps from the entry state)",
        "weight": "W = 1/mass = (1+M^2)·area/M at ice nodes (the mEVP node mass)"},
        "modes": {}}

    def dump():
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=lambda o: o.tolist()
                                  if hasattr(o, "tolist") else str(o)))
        os.replace(tmp, json_path)

    e11, e22, e12 = (np.asarray(q) for q in M._strain_rates(mesh, uvN[0], uvN[1]))
    eps1, eps2 = e11 + e22, e11 - e22
    delta = np.sqrt(eps1 ** 2 + cfg.vale * (eps2 ** 2 + 4.0 * e12 ** 2))
    pf = np.asarray(M.st0.pressure_fac)
    zeta = pf / (delta + cfg.delta_min)
    ie = np.flatnonzero(M.ice_el)
    en = M.en
    gs, mfac = np.asarray(mesh.gradient_sca), np.asarray(mesh.metric_factor)
    a_n, m_n = np.asarray(M.kw["a_ice"]), np.asarray(M.kw["m_ice"])
    obd, rf, W = (np.asarray(q) for q in M.p_steady_coef(x_e, uvN))
    act = M.ice_nod & (np.asarray(M.st0.bc_index) > 0) & (W > 0)
    idx = np.flatnonzero(act)
    na = int(idx.size)
    pos = np.full(M.nod, -1)
    pos[idx] = np.arange(na)
    ws = np.sqrt(W[idx])
    idx_j, ws_j = jnp.asarray(idx), jnp.asarray(ws)
    nod = M.nod
    doc["setup"] = {"n_active_nodes": na, "n_ice_elems": int(ie.size),
                    "drag_pcts": {str(q): float(np.percentile(obd[idx] - 1.0 - cfg.beta_evp, q))
                                  for q in (0, 50, 100)},
                    "delta_over_dmin_ice_pcts": {str(q): float(np.percentile(
                        delta[ie] / cfg.delta_min, q)) for q in (0.1, 1, 10, 50, 90)}}
    print(f"[operator] {json.dumps(doc['setup'])}", flush=True)

    def embed(x, s):
        return (jnp.zeros(nod).at[idx_j].set(x[:na] * s), jnp.zeros(nod).at[idx_j].set(x[na:] * s))

    def info(e, rel=None):
        n = en[e]
        d = {"elem": int(e), "lat": float(M.lat_e[e]), "lon": float(M.lon_e[e]),
             "delta_over_dmin": float(delta[e] / cfg.delta_min), "a": float(a_n[n].mean()),
             "m": float(m_n[n].mean())}
        if rel is not None:
            d["min_eig_over_zeta"] = float(rel)
        return d

    for mode in modes:
        tm = time.time()
        A, AT, G = M.steady_programs(mode)

        def mv_core(x, xe, uv, A=A, AT=AT):
            a = A(xe, uv, embed(x, 1.0 / ws_j))
            b = AT(xe, uv, embed(x, ws_j))
            return 0.5 * jnp.concatenate([a[0][idx_j] * ws_j + b[0][idx_j] / ws_j,
                                          a[1][idx_j] * ws_j + b[1][idx_j] / ws_j])
        mvj = jax.jit(mv_core)

        def mv(x):
            return np.asarray(mvj(jnp.asarray(x), x_e, uvN))
        rep = {}
        # the fixed point and the transpose
        g = [np.asarray(q) for q in G(x_e, uvN)]
        un = [np.asarray(q) for q in uvN]
        rep["fixed_point_rms_rel"] = float(np.sqrt(np.mean(
            ((g[0] - un[0]) ** 2 + (g[1] - un[1]) ** 2)[idx])) / max(np.sqrt(np.mean(
                (un[0] ** 2 + un[1] ** 2)[idx])), 1e-300))
        rng = np.random.default_rng(args.seed)
        d, w = rng.standard_normal(2 * na), rng.standard_normal(2 * na)
        Ad = A(x_e, uvN, embed(jnp.asarray(d), 1.0))
        ATw = AT(x_e, uvN, embed(jnp.asarray(w), 1.0))
        lhs = float(w @ np.concatenate([np.asarray(Ad[0])[idx], np.asarray(Ad[1])[idx]]))
        rhs = float(d @ np.concatenate([np.asarray(ATw[0])[idx], np.asarray(ATw[1])[idx]]))
        rep["transpose_rel_err"] = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
        # (a) per element
        E = element_pairing_jacobians(e11[ie], e22[ie], e12[ie], pf[ie], cfg.vale,
                                      cfg.delta_min, mode)
        Es = 0.5 * (E + np.transpose(E, (0, 2, 1)))
        lam, vec = np.linalg.eigh(Es)
        rel = lam[:, 0] / zeta[ie]
        asym = np.abs(E - np.transpose(E, (0, 2, 1))).max(axis=(1, 2)) / zeta[ie]
        order = np.argsort(rel)
        rep["element"] = {"frac_negative": float((rel < -1e-12).mean()),
                          "n_negative": int((rel < -1e-12).sum()),
                          "min_eig_over_zeta_min": float(rel.min()),
                          "min_eig_over_zeta_pcts": {str(q): float(np.percentile(rel, q))
                                                     for q in (0.01, 0.1, 1, 50)},
                          "asym_over_zeta_max": float(asym.max()),
                          "worst": [info(ie[o], rel[o]) for o in order[:5]]}
        # (c) Rayleigh quotients of one-element fields along the softest strain direction
        ray = []
        for o in order[:8]:
            e = int(ie[o])
            nodes = en[e]
            if np.any(pos[nodes] < 0):
                continue
            du = np.linalg.pinv(element_strain_matrix(gs[e], mfac[e])) @ vec[o][:, 0]
            x = np.zeros(2 * na)
            x[pos[nodes]] = du[:3] * ws[pos[nodes]]
            x[na + pos[nodes]] = du[3:] * ws[pos[nodes]]
            R = float(x @ mv(x)) / float(x @ x)
            ray.append({**info(e, rel[o]), "rayleigh": R})
        rep["one_element_rayleigh"] = ray
        rep["one_element_rayleigh_min"] = min((r["rayleigh"] for r in ray), default=None)
        # (b) Lanczos
        t0 = time.time()
        try:
            rec, xmin = lanczos_extremes(mv, rng.standard_normal(2 * na), args.lanczos_k)
            e_n = xmin[:na] ** 2 + xmin[na:] ** 2
            top = np.argsort(-e_n)[:5]
            rep["lanczos"] = {"records": rec, "seconds": time.time() - t0,
                              "ritz_min": rec[-1]["ritz_min"][0], "res_min": rec[-1]["res_min"][0],
                              "ritz_max": rec[-1]["ritz_max"],
                              "min_vector_top5_share": float(e_n[top].sum() / e_n.sum()),
                              "min_vector_top_nodes": [
                                  {"node": int(idx[t]), "lat": float(M.lat_n[idx[t]]),
                                   "lon": float(M.lon_n[idx[t]]), "a": float(a_n[idx[t]]),
                                   "m": float(m_n[idx[t]]), "share": float(e_n[t] / e_n.sum())}
                                  for t in top]}
        except FloatingPointError as err:
            rep["lanczos"] = {"error": str(err)}
        rep["seconds"] = time.time() - tm
        lz = rep["lanczos"]
        bounds = [b for b in (lz.get("ritz_min"), rep["one_element_rayleigh_min"]) if b is not None]
        rep["lambda_min_upper_bound"] = min(bounds) if bounds else None
        rep["indefinite_proven"] = bool(bounds and min(bounds) < 0.0)
        rep["expansive_vp_step_proven"] = bool(bounds and min(bounds) < 1.0)
        doc["modes"][mode] = rep
        print(f"[{mode}] element: {rep['element']['n_negative']} negative of {ie.size} (min "
              f"{rep['element']['min_eig_over_zeta_min']:.3g} ζ); one-element Rayleigh min "
              f"{rep['one_element_rayleigh_min']}; Lanczos ritz min "
              f"{lz.get('ritz_min')} (res {lz.get('res_min')}), max {lz.get('ritz_max')}; "
              f"transpose err {rep['transpose_rel_err']:.1e}; fixed point "
              f"{rep['fixed_point_rms_rel']:.1e} ({rep['seconds']:.0f}s)", flush=True)
        dump()
    lines = [f"# Linearized steady operator at u_N, {args.date}", "",
             "| mode | elements with sym. part < 0 | min eig / ζ | one-element Rayleigh min | "
             "Lanczos smallest Ritz (res) | largest Ritz | λ_min < 0 proven | λ_min < 1 proven |",
             "|---|---|---|---|---|---|---|---|"]
    for mode, rep in doc["modes"].items():
        lz = rep["lanczos"]
        lines.append(f"| {mode} | {rep['element']['n_negative']} | "
                     f"{rep['element']['min_eig_over_zeta_min']:.3g} | "
                     f"{_fmt(rep['one_element_rayleigh_min'])} | {_fmt(lz.get('ritz_min'))} "
                     f"({_fmt(lz.get('res_min'))}) | {_fmt(lz.get('ritz_max'))} | "
                     f"{rep['indefinite_proven']} | {rep['expansive_vp_step_proven']} |")
    json_path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    doc["wall_s"] = time.time() - t_start
    dump()
    print("\n".join(lines), flush=True)
    print(f"[done] {json_path}  wall {doc['wall_s']:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# Task 25: why does the model respond like ~3 % residual viscous stiffness?

WHY_HS = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
WHY_FORMS = ("tlf", "tlfS", "tlfw0.5", "tlfSw0.5", "tlfT0.99", "tlfT0.97", "tlfT0.9", "tlfP")
WHY_FIELD_FORMS = ("tlf", "tlfS", "tlfT0.97", "tlfP")
WHY_REF_H = 1e-2
#: bin edges of the node properties the θ = 1 excess is attributed to (right-open)
WHY_EDGES = {
    "gab250": (0.0, 0.1, 1.0, 5.0, 20.0, math.inf),     # γ/(αβ) at α = β = 250, max over elements
    "gab_fwd": (0.0, 0.1, 1.0, 5.0, 20.0, math.inf),    # γ/(αβ) with the forward's α, β
    "dlt_min": (0.0, 10.0, 1e2, 1e3, 1e4, math.inf),    # Δ/δmin at x_N, min over elements
    "repeat": (0.0, 1e-13, 1e-10, 1e-7, 1e-4, math.inf),  # |Δu| between repeats of the forward
    "residual": (0.0, 1e-10, 1e-8, 1e-6, 1e-4, math.inf),  # |Δu| made by one more sweep
    "m": (0.0, 0.5, 1.0, 2.0, 3.0, math.inf),
    "a": (0.0, 0.8, 0.95, 0.99, math.inf),
}


def fd_record(d0: float, dp: float, dm: float, h: float) -> dict:
    """Central and one-sided differences of a scalar from its values at c0, c0 + h, c0 − h. A
    kink at c0 shows as forward ≠ backward, the central difference being their mean."""
    return {"Dp": dp, "Dm": dm, "central": (dp - dm) / (2.0 * h), "forward": (dp - d0) / h,
            "backward": (d0 - dm) / h}


def node_extreme_of_elements(vals_e, elem_nodes, n_nodes, mask_e=None, kind="max"):
    """Per node, the max (or min) of an element field over the node's elements, counting only
    ``mask_e`` elements; NaN where a node has none."""
    sgn = 1.0 if kind == "max" else -1.0
    v = sgn * np.asarray(vals_e, float)
    if mask_e is not None:
        v = np.where(np.asarray(mask_e, bool), v, -np.inf)
    out = np.full(int(n_nodes), -np.inf)
    en = np.asarray(elem_nodes)
    for k in range(en.shape[1]):
        np.maximum.at(out, en[:, k], v)
    return np.where(np.isfinite(out), sgn * out, np.nan)


def binned_contributions(contribs: dict, prop, edges, mask) -> list:
    """Sums of per-node contributions (``contribs``: name -> [nod]) over the bins ``edges``
    (right-open) of the node property ``prop`` on the ``mask`` nodes; NaN properties fall in no
    bin. One dict per bin: lo, hi, n and the sum of each contribution."""
    p = np.asarray(prop, float)
    m = np.asarray(mask, bool) & np.isfinite(p)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = m & (p >= lo) & (p < hi)
        rows.append({"lo": lo, "hi": hi, "n": int(sel.sum()),
                     **{k: float(np.sum(np.asarray(v, float)[sel])) for k, v in contribs.items()}})
    return rows


def concentration(x, fracs=(0.001, 0.01, 0.1)) -> dict:
    """How much of Σx (and of Σ|x|) the largest-|x| fraction ``f`` of the entries carries."""
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    order = np.argsort(-np.abs(a))
    tot, tot_abs = float(a.sum()), float(np.abs(a).sum())
    out = {}
    for f in fracs:
        k = max(1, int(round(f * a.size)))
        top = a[order[:k]]
        out[f"{f:g}"] = {"k": k,
                         "signed_share": float(top.sum()) / tot if tot != 0 else None,
                         "abs_share": float(np.abs(top).sum()) / tot_abs if tot_abs > 0 else None}
    return out


def why_study(args) -> int:
    """One step of the ice map (``--forward mevp`` or ``const`` with ``--alpha-const``): is the
    θ = 1 overshoot the solver (elements where the α = 250 iteration never settles) or the
    plastic linearization? Measures the repeat noise and the last-sweep residual of the forward,
    the FD of D over a scan of h (central and one-sided), the one-step frozen-sweep tangent of
    every θ against every h, and where (per node, binned by the local stability number γ/αβ,
    Δ/δmin, the repeat noise, the residual, m, a) the θ = 1 excess over the FD sits."""
    out_dir = _refuse_inside_checkout(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    M = IceMap(args.date, n_sweeps=args.n_sweeps, forward=args.forward,
               alpha_const=args.alpha_const)
    jax, jnp = M.jax, M.jnp
    N = M.n_sweeps
    tag = args.tag or (args.forward + (f"{M.alpha_const:g}" if M.alpha_const else "") + f"_n{N}")
    stem = f"why_{args.date.replace('-', '')}_{tag}"
    json_path = out_dir / f"{stem}.json"
    dev = jax.devices()[0]
    doc = {"provenance": {
        "script": "scripts/adjoint/ice_modes.py why", "git_head": git_head(), "checkout": str(ROOT),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "device": getattr(dev, "device_kind", str(dev)), "platform": dev.platform,
        "jax": jax.__version__, "date": args.date, "tag": tag, "forward": args.forward,
        "alpha": M.alpha_const or M.cfg.alpha_evp, "n_sweeps": N, "kappa0": args.kappa0,
        "time": datetime.datetime.now().isoformat(timespec="seconds"), "settings": vars(args),
        "n_box_nodes": int(M.box.sum()), "n_ice_nodes": int(M.ice_nod.sum()),
        "n_ice_elems": int(M.ice_el.sum()),
        "drift_definition": "D = sum(w a |u|)/sum(w a), ice nodes lat>=70N, w = node area"}}

    def dump():
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=lambda o: o.tolist()
                                  if hasattr(o, "tolist") else str(o)))
        os.replace(tmp, json_path)
    dump()

    ones = M.ones()
    unit = {"lnp": jnp.asarray([1.0, 0.0]), "lncd": jnp.asarray([0.0, 1.0])}
    o = M.offs
    ice_nod, box = M.ice_nod, M.box

    # ---- the forward: repeats (round-off amplification) and one more sweep (convergence) ----
    t0 = time.time()
    yb = M.p_fwd(M.x0, M.c0, *ones)
    jax.block_until_ready(yb)
    t_first = time.time() - t0
    ybf = M.flat(yb)
    yb_n = [np.asarray(q) for q in yb]
    D0 = float(M.p_drift(ybf))
    rep_u = np.zeros(M.nod)
    D_rep = []
    t0 = time.time()
    for _ in range(args.repeats):
        r = M.p_fwd(M.x0, M.c0, *ones)
        rn = [np.asarray(q) for q in r]
        rep_u = np.maximum(rep_u, np.hypot(rn[0] - yb_n[0], rn[1] - yb_n[1]))
        D_rep.append(float(M.p_drift(M.flat(r))))
    t_fwd = (time.time() - t0) / max(args.repeats, 1)
    y1 = [np.asarray(q) for q in M.p_fwd(M.x0, M.c0, ones[0], ones[1], jnp.ones(N + 1), ones[3])]
    res_u = np.hypot(y1[0] - yb_n[0], y1[1] - yb_n[1])
    res_s = np.sqrt(sum((y1[k] - yb_n[k]) ** 2 for k in (2, 3, 4)) / 3.0)
    whole_u = np.hypot(yb_n[0] - np.asarray(M.x0[0]), yb_n[1] - np.asarray(M.x0[1]))
    rms = lambda a, m: float(np.sqrt(np.mean(a[m] ** 2))) if m.any() else None  # noqa: E731
    D_noise = max((abs(d - D0) for d in D_rep), default=0.0)
    doc["forward"] = {
        "D0": D0, "fwd_first_s": t_first, "fwd_s": t_fwd,
        "repeat": {"max_du": float(rep_u[ice_nod].max()) if ice_nod.any() else 0.0,
                   "n_ice_nodes_du_gt": {f"{t:g}": int((rep_u[ice_nod] > t).sum())
                                         for t in (1e-12, 1e-9, 1e-6, 1e-4)},
                   "D_diffs": [abs(d - D0) for d in D_rep]},
        "residual": {"rms_du": rms(res_u, ice_nod), "max_du": float(res_u[ice_nod].max()),
                     "rms_dsigma": rms(res_s, M.ice_el),
                     "last_over_whole": (rms(res_u, ice_nod) or 0.0)
                     / max(rms(whole_u, ice_nod) or 0.0, 1e-300)}}
    print(f"[forward] {json.dumps(doc['forward'])}", flush=True)
    dump()

    # ---- FD of D over h ----
    g0 = np.asarray(M.p_grad_drift(ybf))
    fd, fd_fields = {}, {}
    for cname in CONTROLS:
        fd[cname] = {}
        for h in args.hs:
            yp = M.flat(M.p_fwd(M.x0, M.c0 + h * unit[cname], *ones))
            ym = M.flat(M.p_fwd(M.x0, M.c0 - h * unit[cname], *ones))
            rec = fd_record(D0, float(M.p_drift(yp)), float(M.p_drift(ym)), h)
            rec["noise_bound"] = D_noise / h
            fd[cname][f"{h:g}"] = rec
            if cname == "lnp" and (h in args.field_hs or h == WHY_REF_H):
                fd_fields[h] = np.asarray((yp - ym) / (2.0 * h))
        print(f"[fd {cname}] " + ", ".join(f"{k}: {v['central']:.5g} ({v['forward']:.5g}/"
                                           f"{v['backward']:.5g})" for k, v in fd[cname].items()),
              flush=True)
        doc["fd"] = fd
        dump()

    # ---- the one-step frozen-sweep tangent of every θ ----
    wbox = M.wa[box]

    def uv_box(f):
        f = np.asarray(f)
        return np.concatenate([f[o[0]:o[1]][box], f[o[1]:o[2]][box]])

    def corr(a, b):
        return weighted_corr(uv_box(a), uv_box(b), np.concatenate([wbox, wbox]))

    x0f = M.flat(M.x0)
    nflat = int(o[-1])
    tl, tl_fields = {}, {}
    for form in args.forms:
        kind, smode = tl_kind_mode(form)
        if kind != "tlf":
            raise SystemExit(f"why: only frozen-sweep (tlf*) forms, not {form!r}")
        omega = tl_omega(form)
        p_jvp = M.programs("tlf", smode, omega)[0]
        fac = M.factors(form, args.kappa0)
        rec = {"mode": smode, "omega": omega}
        tw = time.time()
        for cname in CONTROLS:
            _, t = p_jvp(x0f, M.c0, *fac, jnp.zeros(nflat), unit[cname])
            t_raw = np.asarray(t) * np.asarray(M.dscale)
            ok = bool(np.all(np.isfinite(t_raw)))
            val = float(np.dot(g0, t_raw)) if ok else None
            r = {"D_resp": val, "finite": ok,
                 "ratio": {k: (val / v["central"] if ok and v["central"] != 0 else None)
                           for k, v in fd[cname].items()}}
            if cname == "lnp" and ok:
                r["corr"] = {f"{h:g}": corr(t_raw, f) for h, f in fd_fields.items()}
                if form in args.field_forms:
                    tl_fields[form] = t_raw
            rec[cname] = r
        rec["wall_s"] = time.time() - tw
        tl[form] = rec
        print(f"[tl {form}] P* {rec['lnp']['ratio'].get(f'{WHY_REF_H:g}')} "
              f"C_d {rec['lncd']['ratio'].get(f'{WHY_REF_H:g}')} ({rec['wall_s']:.1f}s)", flush=True)
        doc["tl"] = tl
        dump()

    # ---- where the θ = 1 excess over the FD sits ----
    gu, gv = g0[o[0]:o[1]], g0[o[1]:o[2]]

    def contrib(t_raw):
        t_raw = np.asarray(t_raw)
        return gu * t_raw[o[0]:o[1]] + gv * t_raw[o[1]:o[2]]

    c_fd = contrib(fd_fields[WHY_REF_H])
    contribs = {"fd": c_fd}
    for form, t_raw in tl_fields.items():
        c = contrib(t_raw)
        contribs[form] = c
        contribs[f"excess_{form}"] = c - c_fd
    for h, f in fd_fields.items():
        if h != WHY_REF_H:
            contribs[f"fd{h:g}_minus_ref"] = contrib(f) - c_fd
    _, _, gamma = M.element_table(ybf, [])
    cfg, en = M.cfg, M.en
    e11, e22, e12 = (np.asarray(q) for q in M._strain_rates(M.mesh, yb_n[0], yb_n[1]))
    eps1, eps2 = e11 + e22, e11 - e22
    delta = np.sqrt(np.maximum(eps1 ** 2 + cfg.vale * (eps2 ** 2 + 4 * e12 ** 2), 0.0))
    props = {
        "gab250": node_extreme_of_elements(gamma / (cfg.alpha_evp * cfg.alpha_evp), en, M.nod,
                                           M.ice_el),
        "gab_fwd": node_extreme_of_elements(gamma / M.ab_e, en, M.nod, M.ice_el),
        "dlt_min": node_extreme_of_elements(delta / cfg.delta_min, en, M.nod, M.ice_el, "min"),
        "repeat": rep_u, "residual": res_u,
        "m": np.asarray(M.kw["m_ice"], float), "a": np.asarray(M.kw["a_ice"], float)}
    doc["attribution"] = {
        "control": "lnp", "ref_h": WHY_REF_H,
        "totals": {k: float(np.sum(v[box])) for k, v in contribs.items()},
        "bins": {p: binned_contributions(contribs, v, WHY_EDGES[p], box) for p, v in props.items()},
        "concentration": {k: concentration(v[box]) for k, v in contribs.items()
                          if k.startswith("excess_")}}
    dump()
    np.savez_compressed(
        out_dir / f"{stem}_fields.npz", lon_n=M.lon_n.astype(np.float32),
        lat_n=M.lat_n.astype(np.float32), box=box, ice_nod=ice_nod,
        u=yb_n[0].astype(np.float32), v=yb_n[1].astype(np.float32),
        **{f"prop_{k}": np.asarray(v, np.float32) for k, v in props.items()},
        **{f"c_{k}": np.asarray(v, np.float32) for k, v in contribs.items()})

    doc["wall_s"] = time.time() - t_start
    dump()
    (out_dir / f"{stem}.md").write_text(why_markdown(doc))
    print(f"[done] {json_path}  wall {doc['wall_s']:.0f}s", flush=True)
    return 0


def why_markdown(doc) -> str:
    """The per-run tables of :func:`why_study`."""
    pv, fw = doc["provenance"], doc.get("forward", {})
    L = [f"# Why ~3 %: {pv['date']}, {pv['forward']} α = {pv['alpha']:g}, {pv['n_sweeps']} sweeps\n",
         f"git {pv['git_head'][:12]}, job {pv['slurm_job']}, {pv['device']}.\n",
         f"Forward: repeat max |Δu| {fw.get('repeat', {}).get('max_du', float('nan')):.3g} m/s, "
         f"one more sweep: rms |Δu| {fw.get('residual', {}).get('rms_du') or float('nan'):.3g} m/s "
         f"(× {fw.get('residual', {}).get('last_over_whole', float('nan')):.3g} of the step's "
         f"change), D repeat {max(fw.get('repeat', {}).get('D_diffs') or [0.0]):.3g}.\n"]
    for cname in CONTROLS:
        f = doc.get("fd", {}).get(cname, {})
        if not f:
            continue
        L.append(f"\n## FD of D, {cname}\n\n| h | central | forward | backward | noise bound |\n"
                 "|---|---|---|---|---|")
        for k, v in f.items():
            L.append(f"| {k} | {v['central']:.5g} | {v['forward']:.5g} | {v['backward']:.5g} | "
                     f"{v['noise_bound']:.2g} |")
    tl = doc.get("tl", {})
    if tl:
        hs = list(doc["fd"]["lnp"].keys())
        L.append("\n## One-step tangent / central FD, ln P*\n\n| form | " + " | ".join(hs)
                 + " | C_d at 0.01 |\n|---|" + "---|" * (len(hs) + 1))
        for form, r in tl.items():
            cells = [_fmt(r["lnp"]["ratio"].get(k)) for k in hs]
            L.append(f"| {form} | " + " | ".join(cells) + f" | {_fmt(r['lncd']['ratio'].get(f'{WHY_REF_H:g}'))} |")
    att = doc.get("attribution")
    if att:
        tot = att["totals"]
        L.append(f"\n## Where the θ = 1 excess sits (P*, FD at h {att['ref_h']:g}); totals: "
                 + ", ".join(f"{k} {v:.4g}" for k, v in tot.items()) + "\n")
        for p, rows in att["bins"].items():
            ex = [k for k in tot if k.startswith("excess_")]
            L.append(f"\n{p}: | bin | n | fd | " + " | ".join(ex) + " |\n|---|---|---|"
                     + "---|" * len(ex))
            for b in rows:
                L.append(f"| [{b['lo']:g}, {b['hi']:g}) | {b['n']} | {b['fd']:.3g} | "
                         + " | ".join(f"{b[k]:.3g}" for k in ex) + " |")
    return "\n".join(L) + "\n"


def why_summary(args) -> int:
    """Tables across the ``why`` runs of a directory (per date: every forward), the FD of the
    constant-α forwards against the mEVP one, the h scan, the excess by γ/αβ and repeat-noise
    bins; figures: the h scan and maps of the θ = 1 excess and of the forwards' FD difference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = Path(args.dir)
    out = _refuse_inside_checkout(args.out or d)
    docs = {}
    for p in sorted(d.glob("why_*.json")):
        doc = json.loads(p.read_text())
        if "tl" not in doc or "attribution" not in doc:
            continue
        docs.setdefault(doc["provenance"]["date"], {})[doc["provenance"]["tag"]] = (doc, p)
    ref = f"{WHY_REF_H:g}"
    L = ["# Task 25: why the model responds like ~3 % residual viscous stiffness\n"]
    for date, runs in docs.items():
        tags = sorted(runs, key=lambda t: (runs[t][0]["provenance"]["forward"] != "mevp",
                                           runs[t][0]["provenance"]["alpha"],
                                           runs[t][0]["provenance"]["n_sweeps"]))
        forms = [f for f in WHY_FORMS if all(f in runs[t][0]["tl"] for t in tags)]
        L.append(f"\n## {date}\n\nOne step; tangent / central FD of the same forward at h {ref}.\n")
        L.append("| forward | sweeps | repeat max \\|Δu\\| | last sweep / step | FD P* | FD C_d | "
                 + " | ".join(forms) + " | C_d θ1 |\n|---|---|---|---|---|---|" + "---|" * (len(forms) + 1))
        for t in tags:
            doc = runs[t][0]
            fw, fd, tl = doc["forward"], doc["fd"], doc["tl"]
            L.append(f"| {doc['provenance']['forward']} α {doc['provenance']['alpha']:g} | "
                     f"{doc['provenance']['n_sweeps']} | {_fmt(fw['repeat']['max_du'])} | "
                     f"{_fmt(fw['residual']['last_over_whole'])} | {_fmt(fd['lnp'][ref]['central'])} | "
                     f"{_fmt(fd['lncd'][ref]['central'])} | "
                     + " | ".join(_fmt(tl[f]["lnp"]["ratio"].get(ref)) for f in forms)
                     + f" | {_fmt(tl['tlf']['lncd']['ratio'].get(ref)) if 'tlf' in tl else '—'} |")
        base = [t for t in tags if runs[t][0]["provenance"]["forward"] == "mevp"]
        if base:
            b = runs[base[0]][0]
            L.append(f"\nAgainst the mEVP forward ({base[0]}): FD ratio and θ = 1 (S) / θ 0.97 tangent "
                     "ratio, ln P* | ln C_d.\n\n| forward | FD P* | FD C_d | TL θ1 S P* | TL θ0.97 P* |\n"
                     "|---|---|---|---|---|")
            for t in tags:
                if t == base[0]:
                    continue
                c = runs[t][0]

                def rr(get):
                    try:
                        return get(c) / get(b)
                    except (KeyError, TypeError, ZeroDivisionError):
                        return None
                L.append(f"| {t} | {_fmt(rr(lambda x: x['fd']['lnp'][ref]['central']))} | "
                         f"{_fmt(rr(lambda x: x['fd']['lncd'][ref]['central']))} | "
                         f"{_fmt(rr(lambda x: x['tl']['tlfS']['lnp']['D_resp']))} | "
                         f"{_fmt(rr(lambda x: x['tl']['tlfT0.97']['lnp']['D_resp']))} |")
        hs = list(runs[tags[0]][0]["fd"]["lnp"].keys())
        L.append("\nh scan, ln P*: central FD(h) / FD(0.01) of the same forward (* = the repeat "
                 "noise bound exceeds 1 % of the FD); last column θ = 1 (S) tangent / FD(0.01).\n\n"
                 "| forward | " + " | ".join(hs) + " | θ1 S |\n|---|" + "---|" * (len(hs) + 1))
        for t in tags:
            fd = runs[t][0]["fd"]["lnp"]
            r0 = fd[ref]["central"]
            cells = []
            for k in hs:
                v = fd[k]
                star = "*" if abs(v["noise_bound"]) > 0.01 * abs(v["central"]) else ""
                cells.append(_fmt(v["central"] / r0 if r0 else None) + star)
            L.append(f"| {t} | " + " | ".join(cells)
                     + f" | {_fmt(runs[t][0]['tl'].get('tlfS', {}).get('lnp', {}).get('ratio', {}).get(ref))} |")
        L.append("\nShare of the θ = 1 (S) excess over the FD (Σ over the box = 1) by node bins.\n")
        for p in ("gab250", "repeat", "dlt_min", "residual"):
            for t in tags:
                att = runs[t][0]["attribution"]
                tot = att["totals"].get("excess_tlfS")
                if not tot:
                    continue
                rows = att["bins"][p]
                L.append(f"- {p}, {t}: " + "; ".join(
                    f"[{b['lo']:g}, {b['hi']:g}) n {b['n']}: {b['excess_tlfS'] / tot:.2f}"
                    for b in rows if b["n"]))

        # figure: the h scan
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        rb = runs[base[0]][0]["fd"]["lnp"][ref]["central"] if base else None
        for i, t in enumerate(tags):
            doc = runs[t][0]
            fd = doc["fd"]["lnp"]
            norm = rb or fd[ref]["central"]
            h = np.array([float(k) for k in hs])
            y = np.array([fd[k]["central"] / norm for k in hs])
            noisy = np.array([abs(fd[k]["noise_bound"]) > 0.01 * abs(fd[k]["central"]) for k in hs])
            col = f"C{i}"
            ax.plot(h, y, "-", color=col, lw=1)
            ax.plot(h[~noisy], y[~noisy], "o", color=col, label=t)
            ax.plot(h[noisy], y[noisy], "o", mfc="none", color=col)
            for f, ls in (("tlfS", "--"), ("tlfT0.97", ":")):
                v = doc["tl"].get(f, {}).get("lnp", {}).get("D_resp")
                if v is not None:
                    ax.axhline(v / norm, color=col, ls=ls, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("h (ln P*)")
        ax.set_ylabel("central FD of D / mEVP FD at h 0.01")
        ax.set_title(f"{date}: FD over h (open: noise > 1 %); dashed θ = 1 S, dotted θ = 0.97",
                     fontsize=9)
        ax.legend(fontsize=7)
        fig.savefig(out / f"why_{date.replace('-', '')}_hscan.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # figure: maps (north, > 65°): the θ = 1 excess per forward, the FD difference to mEVP
        panels = []
        for t in tags:
            F = np.load(runs[t][1].with_name(runs[t][1].stem + "_fields.npz"))
            panels.append((f"θ1 S − FD, {t}", F, F["c_excess_tlfS"] if "c_excess_tlfS" in F.files else None))
        if base and len(tags) > 1:
            Fb = np.load(runs[base[0]][1].with_name(runs[base[0]][1].stem + "_fields.npz"))
            for t in tags:
                if t != base[0]:
                    F = np.load(runs[t][1].with_name(runs[t][1].stem + "_fields.npz"))
                    panels.append((f"FD {t} − FD mEVP", F, F["c_fd"] - Fb["c_fd"]))
            panels.append((f"log10 repeat |Δu|, {base[0]}", Fb, np.log10(np.maximum(Fb["prop_repeat"], 1e-16))))
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), squeeze=False)
        for ax, (title, F, val) in zip(axes[0], panels):
            lat, lon = F["lat_n"], F["lon_n"]
            sel = (lat > 65) & F["ice_nod"]
            r = np.tan(np.radians(90 - lat[sel]) / 2)
            xx, yy = r * np.cos(np.radians(lon[sel])), r * np.sin(np.radians(lon[sel]))
            if val is None:
                continue
            vv = np.asarray(val, float)[sel]
            if title.startswith("log10"):
                sc = ax.scatter(xx, yy, c=vv, s=1, cmap="magma", vmin=-16, vmax=-3, rasterized=True)
            else:
                lim = np.nanpercentile(np.abs(vv), 99.5) or 1.0
                sc = ax.scatter(xx, yy, c=vv, s=1, cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
            fig.colorbar(sc, ax=ax, shrink=0.7)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=7)
        fig.suptitle(f"{date}: per-node contributions to δD/δ ln P* (Arctic box)", fontsize=9)
        fig.savefig(out / f"why_{date.replace('-', '')}_maps.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    (out / "why_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# figures


def plot(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jpath = Path(args.json)
    doc = json.loads(jpath.read_text())
    tag = jpath.stem.replace("modes_", "")
    F = np.load(jpath.with_name(f"modes_{tag}_fields.npz"))
    out = _refuse_inside_checkout(jpath.parent)
    keys = [k for k in F.files if k.endswith("_e") and k.startswith("exact_sweep")]
    keys += [k for k in ("exact_svd_e",) if k in F.files]
    lon, lat = F["lon_e"], F["lat_e"]
    made = []
    for hemi, sgn in (("north", 1), ("south", -1)):
        n = len(keys)
        if n == 0:
            break
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
        for ax, k in zip(axes[0], keys):
            e = F[k].astype(np.float64)
            bg = (sgn * lat > 45) & F["ice_el"]
            rb = np.tan(np.radians(90 - sgn * lat[bg]) / 2)
            ax.scatter(rb * np.cos(np.radians(lon[bg])), rb * np.sin(np.radians(lon[bg])),
                       s=0.2, c="0.85", rasterized=True)               # the ice cover
            sel = (sgn * lat > 45) & (e > 1e-8 * e.max())
            r = np.tan(np.radians(90 - sgn * lat[sel]) / 2)
            th = np.radians(lon[sel])
            xx, yy = r * np.cos(th), r * np.sin(th)
            sc = ax.scatter(xx, yy, c=np.log10(e[sel] / e.max()), s=2, cmap="viridis",
                            vmin=-8, vmax=0, rasterized=True)
            top = np.argsort(-e)[:10]
            tsel = top[sgn * lat[top] > 45]
            rt = np.tan(np.radians(90 - sgn * lat[tsel]) / 2)
            ax.scatter(rt * np.cos(np.radians(lon[tsel])), rt * np.sin(np.radians(lon[tsel])),
                       s=40, facecolors="none", edgecolors="r")
            ax.set_aspect("equal")
            ax.set_xlim(-0.45, 0.45)
            ax.set_ylim(-0.45, 0.45)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(k.replace("exact_", "").replace("_e", ""), fontsize=9)
        fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, label="log10 element energy / max")
        fig.suptitle(f"dominant exact-tangent mode, {hemi} (> 45°), {doc['provenance']['date']} "
                     f"(red: top-10 elements)", fontsize=10)
        p = out / f"modes_{tag}_{hemi}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        made.append(str(p))
    print("[plot] " + " ".join(made), flush=True)
    return 0


def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--date", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--variants", nargs="*", default=None)
    r.add_argument("--n-pow", type=int, default=30)
    r.add_argument("--arnoldi-k", type=int, default=20)
    r.add_argument("--arnoldi-warm-step", type=int, default=10)
    r.add_argument("--arnoldi-warm-sweep", type=int, default=100)
    r.add_argument("--sweeps", type=int, nargs="*", default=[1, 30, 60, 120])
    r.add_argument("--table-sweep", type=int, default=60)
    r.add_argument("--chain-steps", type=int, default=48)
    r.add_argument("--fd-h", type=float, nargs="*", default=[1e-3, 1e-2])
    r.add_argument("--div-star", type=float, default=1e-7)
    r.add_argument("--n-sweeps", type=int, default=None,
                   help="override the 120 sweeps (smoke only; disables the bitwise gate)")
    r.add_argument("--repeat-chain", action="store_true")
    r.add_argument("--all-sweep-spectra", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--forward", choices=("mevp", "aevp"), default="mevp")
    r.add_argument("--aevp-c", type=float, default=AEVP_C)
    r.add_argument("--aevp-alpha-min", type=float, default=AEVP_ALPHA_MIN)
    p = sub.add_parser("plot")
    p.add_argument("--json", required=True)
    o = sub.add_parser("operator", help="Task 23 T1: positivity of the linearized steady problem")
    o.add_argument("--date", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--modes", default="exact,S,T0.97,P")
    o.add_argument("--lanczos-k", type=int, default=600)
    o.add_argument("--n-sweeps", type=int, default=None)
    o.add_argument("--seed", type=int, default=0)
    w = sub.add_parser("why", help="Task 25: solver or plasticity behind the θ = 1 overshoot")
    w.add_argument("--date", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--forward", choices=("mevp", "const"), default="mevp")
    w.add_argument("--alpha-const", type=float, default=None)
    w.add_argument("--n-sweeps", type=int, default=None)
    w.add_argument("--hs", type=float, nargs="*", default=list(WHY_HS))
    w.add_argument("--field-hs", type=float, nargs="*", default=[1e-2, 1e-4])
    w.add_argument("--forms", nargs="*", default=list(WHY_FORMS))
    w.add_argument("--field-forms", nargs="*", default=list(WHY_FIELD_FORMS))
    w.add_argument("--kappa0", type=float, default=1.0)
    w.add_argument("--repeats", type=int, default=2)
    w.add_argument("--tag", default=None)
    s = sub.add_parser("why-summary")
    s.add_argument("--dir", required=True)
    s.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "why":
        if WHY_REF_H not in args.hs:
            ap.error(f"--hs must include {WHY_REF_H:g}")
        if args.forward == "const" and not args.alpha_const:
            ap.error("--forward const needs --alpha-const")
    return args


def main(argv=None) -> int:
    args = _parse(argv)
    return {"run": run, "plot": plot, "operator": operator_study, "why": why_study,
            "why-summary": why_summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
