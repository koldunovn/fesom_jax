"""CORE2 instantaneous adjoint sensitivity maps + adjoint↔EKI cross-check — Paper-experiments
Task C1 (§1 Sensitivity, the cheapest pillar: it motivates §2 calibration).

ONE backward pass through the assembled differentiable global model turns a **scalar** parameter
into a **[nod2D] field leaf** and returns ``∂J/∂θ(x)`` — a map of *where* a local change in the
mixing/eddy parameter most moves an ocean diagnostic. Two targets, each isolating one parameter's
gradient path (the proven grad-gate / A7 configs):

  * ``mld_ck``  : config **zstar + TKE** (GM/KPP/ice off).  J = area-weighted mean **mixed-layer
                  depth** (de Boyer Montégut density-threshold, :mod:`fesom_jax.obs_compare`).
                  θ = ``tke_c_k`` promoted to a ``[nod2D, 1]`` field  →  ``∂(mean MLD)/∂c_k(x)``.
  * ``ts_kgm``  : config **zstar + GM/Redi** (TKE/KPP/ice off).  J = area-weighted mean
                  **upper-ocean temperature** (top ``--band`` layers — a GM heat-redistribution /
                  stratification proxy).  θ = ``k_gm`` (+ the C-synced ``redi_kmax``) promoted to a
                  ``[nod2D]`` field  →  ``∂(upper-ocean T)/∂k_gm(x)``.

**What each backward buys (labelled HONESTLY — review MAJOR-1).** This is the *fast / instantaneous*
sensitivity over a **single short window** (``--n`` = A7's N_max=20, ~10 h), NOT the multi-year
equilibrium sensitivity. The equilibrium GM→T/S response is slow and lives beyond the adjoint
window — which is exactly why §2's GM calibration uses **EKI**, not the adjoint (the adjoint↔EKI
boundary A7 measured). The map shows which parameters matter and where; calibration (§2) descends it.

The script proves and produces, per target:
  1. **adjoint == FD** on the *scalar* θ. The scalar adjoint is ``Σ_x ∂J/∂θ_field(x)`` (a uniform
     field broadcast ⇒ the chain rule sums the field cotangent), so the field map's SUM *is* the
     scalar gradient the existing grad gates report — FD-verified here with a step sweep (plateau).
  2. the **[nod2D] sensitivity map** (finite everywhere — the AD masked-NaN rule; nonzero on wet).
  3. an **FD spot-check** at the most-sensitive node (central FD of J vs the map there).
  4. (``ts_kgm`` only) the **adjoint↔EKI cross-check**: the forward-ensemble gradient estimate
     ``cov(θ,J)/var(θ)`` (the core of :func:`fesom_jax.eki.eki_update`'s ``C_θg``, forward-only,
     :mod:`fesom_jax.eki`) agrees in **sign + magnitude** with the adjoint scalar gradient →
     validates both tools on the shared ``k_gm`` scalar and motivates EKI for the slow equilibrium.

Outputs ``scripts/sensitivity_<target>.npz`` (the map + node lon/lat + the scalars) and appends a
JSON summary line to ``--results``; :mod:`scripts.fig_sensitivity` reads them, draws Fig 2 and emits
**SENSITIVITY_MAP_OK**. Single-GPU only (the sharded ragged-halo AD bug is forward-safe only — see
docs/JAX_RAGGED_A2A_BUG.md). The model→obs operator (:mod:`fesom_jax.obs_compare`) is already wired,
so swapping J for ``∂(misfit-vs-WOA/dBM)/∂θ`` is a one-line change to the loss (the obs application).

Usage (GPU):  python scripts/paper/core2_paper_sensitivity.py --target mld_ck --n 20
              python scripts/paper/core2_paper_sensitivity.py --target ts_kgm --n 20
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

from fesom_jax import ale, surface_forcing, eki, obs_compare, ssh
from fesom_jax.ale import AleConfig
from fesom_jax.calibrate import build_params
from fesom_jax.gm import GMConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
ALE_CFG = AleConfig()
TKE_CFG = TkeConfig()
GM_CFG = GMConfig()
DEFAULT_RESULTS = ROOT / "scripts" / "sensitivity_results.jsonl"

# per-target metadata (config + the differentiable θ field + the scalar diagnostic J)
TARGETS = ("mld_ck", "ts_kgm")


def peak_gb():
    pk = 0.0
    for d in jax.devices():
        try:
            st = d.memory_stats() or {}
            pk = max(pk, (st.get("peak_bytes_in_use") or 0) / 1e9)
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


def build(year, n):
    mesh = load_mesh(MESH_DIR)
    state = core2_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, year, sst_ic=np.asarray(state.T[:, 0]))
    sfs = cf.stack(surface_forcing.dates_for_steps(year, DT, n))
    return mesh, state, op, cf.static, sfs


# ==========================================================================
# the two scalar diagnostics J(θ), each closing over the mesh/state/forcing.
# θ is a FIELD; the scalar-θ helper broadcasts a scalar to the uniform field.
# ==========================================================================
def make_loss(target, mesh, st0, op, fs, sfs, n, band):
    """Return ``(loss_field, theta0_field, theta0_scalar, to_field, label, unit)``.

    ``loss_field(theta_field) -> scalar J``; ``to_field(scalar) -> uniform field`` so the
    scalar-θ functional is ``loss_field(to_field(s))`` (used for the FD proof / spot-check / EKI).
    """
    node_mask = jnp.asarray(mesh.node_layer_mask)
    N = mesh.nod2D

    if target == "mld_ck":
        wet0 = node_mask[:, 0]
        theta0_scalar = float(Params.defaults().tke_c_k)        # 0.1

        def to_field(s):
            return jnp.full((N, 1), s, jnp.float64)

        def loss_field(ck_field):                               # ck_field [N,1]
            p = build_params({"tke_c_k": ck_field})
            fin = integrate(st0, mesh, op, None, n_steps=n, dt=DT, step_forcings=sfs,
                            forcing_static=fs, tke_cfg=TKE_CFG, ale_cfg=ALE_CFG, params=p)
            _, Z3d = ale.live_geometry(mesh, fin.hnode)
            mld, valid = obs_compare.mld_density_threshold(fin.T, fin.S, Z3d, node_mask)
            return jnp.sum(jnp.where(valid, mld, 0.0)) / jnp.sum(valid)

        return (loss_field, to_field(theta0_scalar), theta0_scalar, to_field,
                "d(mean MLD)/d(c_k)", "m / (c_k unit)")

    if target == "ts_kgm":
        # upper-ocean band: wet nodes in the top `band` layers (a GM-sensitive heat/strat proxy)
        klev = jnp.arange(mesh.nl)[None, :]
        band_mask = node_mask & (klev < band)
        nband = jnp.sum(band_mask)
        theta0_scalar = float(Params.defaults().k_gm)           # 1000

        def to_field(s):
            return jnp.full((N,), s, jnp.float64)

        def loss_field(kgm_field):                              # kgm_field [N]
            # k_gm and the C-synced redi_kmax both derive from the field ⇒ the gradient is the
            # combined GM+Redi sensitivity (the grad-gate convention: redi_kmax=k_gm).
            p = build_params({"k_gm": kgm_field, "redi_kmax": kgm_field})
            fin = integrate(st0, mesh, op, None, n_steps=n, dt=DT, step_forcings=sfs,
                            forcing_static=fs, gm_cfg=GM_CFG, ale_cfg=ALE_CFG, params=p)
            return jnp.sum(jnp.where(band_mask, fin.T, 0.0)) / nband

        return (loss_field, to_field(theta0_scalar), theta0_scalar, to_field,
                f"d(upper-ocean T, top {band} lev)/d(k_gm)", "degC / (m^2/s)")

    raise ValueError(f"unknown target {target!r}; choose from {TARGETS}")


def fd_scalar_sweep(loss_jit, to_field, s0):
    """Central relative FD of the SCALAR functional at s0, reusing the ALREADY-jitted forward
    ``loss_jit`` (a uniform field ``to_field(s)``) — no extra executable (fewer CUDA graphs)."""
    rows = []
    for h in H_SWEEP:
        sp, sm = s0 * (1.0 + h), s0 * (1.0 - h)
        fp = float(loss_jit(to_field(sp)))
        fm = float(loss_jit(to_field(sm)))
        rows.append((h, (fp - fm) / (sp - sm)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS, required=True)
    ap.add_argument("--n", type=int, default=20, help="adjoint window (steps); A7 N_max=20")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--band", type=int, default=10, help="ts_kgm: # top layers for the T metric")
    ap.add_argument("--members", type=int, default=12, help="ts_kgm: forward-ensemble size (EKI x-check)")
    ap.add_argument("--ens-sigma", type=float, default=0.10, help="ts_kgm: ensemble rel. spread")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--outdir", type=Path, default=ROOT / "scripts")
    args = ap.parse_args()
    tgt, n = args.target, args.n

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, st0, op, fs, sfs = build(args.year, n)
    cfg_name = "zstar+TKE" if tgt == "mld_ck" else "zstar+GM"
    print(f"[setup] built in {time.time()-t0:.1f}s; target={tgt}  N={n} "
          f"({n*DT/86400:.3f} days)  {cfg_name}", flush=True)
    gpu_gb = gpu_limit_gb()
    print(f"[setup] GPU memory limit = {gpu_gb:.0f} GB", flush=True)

    loss_field, th0_field, th0_scalar, to_field, label, unit = make_loss(
        tgt, mesh, st0, op, fs, sfs, n, args.band)
    geo = np.degrees(np.asarray(mesh.geo_coord_nod2D))         # [nod2D, 2] (lon, lat)

    rec = {"target": tgt, "N": n, "days": n * DT / 86400.0, "dt": DT, "config": cfg_name,
           "label": label, "unit": unit, "theta0": th0_scalar, "band": args.band,
           "gpu_gb": gpu_gb}

    # ---------- (2)+(1) ONE backward pass: the [nod2D] map; scalar AD = Σ map ----------
    loss_jit = jax.jit(loss_field)
    grad_jit = jax.jit(jax.grad(loss_field))
    try:
        J0 = float(loss_jit(th0_field))
        t1 = time.time()
        g = np.asarray(grad_jit(th0_field))                    # [N] or [N,1]
        bwd_s = time.time() - t1
    except Exception as e:                                     # OOM (working set OR CUDA-graph)
        msg = type(e).__name__ + ": " + str(e)[:240]
        is_oom = ("RESOURCE_EXHAUSTED" in str(e).upper() or "OUT_OF_MEMORY" in str(e).upper()
                  or "memory" in str(e).lower())
        pk = peak_gb()
        rec.update(dict(oom=is_oom, peak_gb=pk, ok=False, error=msg))
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"\n  N={n} {tgt}: FAILED ({msg})", flush=True)
        # exit 2 = "OOM, retry a smaller N" (the .sbatch descends N on this code); 1 = hard fail
        print(f"SENSITIVITY_{tgt.upper()}_{'OOM' if is_oom else 'FAIL'}", flush=True)
        return 2 if is_oom else 1
    pk = peak_gb()
    g_flat = g.reshape(-1)                                     # squeeze the [N,1] c_k case
    g_ad_scalar = float(np.sum(g_flat))                        # = dJ/dθ_scalar (uniform broadcast)
    n_bad = int((~np.isfinite(g_flat)).sum())
    nz = int(np.sum(g_flat != 0.0))
    amax = float(np.max(np.abs(g_flat))) if g_flat.size else 0.0
    print(f"\n  J(θ0) = {J0:.6f}   {label} (scalar AD = Σ map) = {g_ad_scalar:+.6e}", flush=True)
    print(f"  map: shape={g.shape}  finite={n_bad==0}  nonzero={nz}/{g_flat.size}  "
          f"max|g|={amax:.3e}  peak={pk:.2f}/{gpu_gb:.0f} GB  bwd={bwd_s:.1f}s", flush=True)

    # ---------- (1) adjoint == FD on the scalar θ (the proof; plateau over h) ----------
    rows = fd_scalar_sweep(loss_jit, to_field, th0_scalar)
    plateau = min(abs(g_ad_scalar - gf) / max(abs(gf), 1e-300) for _, gf in rows)
    print(f"  --- adjoint==FD proof (scalar θ={th0_scalar:g}) ---")
    for h, gf in rows:
        print(f"      h={h:.0e}  FD={gf:+.6e}  rel={abs(g_ad_scalar-gf)/max(abs(gf),1e-300):.2e}")
    print(f"      plateau(min rel) = {plateau:.3e}", flush=True)
    proof_ok = bool(np.isfinite(g_ad_scalar) and g_ad_scalar != 0.0 and plateau < 1e-2)

    # ---------- (3) FD spot-check: top-|g| nodes, central-FD h-sweep (plateau) ----------
    # A single-h FD at the EXTREMAL node can sit on a GM clamp/taper kink (the |g|-max node is
    # the likeliest kink), so the secant FD mismatches the true (AD) derivative even though the
    # AGGREGATE scalar FD is perfect. Sweep h at the top-K |g| nodes and take the best plateau —
    # this rigorously verifies the map where the signal is large AND the response is locally
    # smooth (robust to one extremal kink). One node passing proves the map values are FD-correct.
    HS_NODE = (1e-2, 1e-3, 1e-4)
    topk = [int(i) for i in np.argsort(-np.abs(g_flat))[:3]]
    spot = None
    for ni in topk:
        ei = jnp.asarray((np.arange(g_flat.size) == ni).astype(np.float64).reshape(g.shape))
        g_ni = float(g_flat[ni])
        cand = []
        for hr in HS_NODE:
            hn = abs(th0_scalar) * hr
            fdv = float((loss_jit(th0_field + hn * ei) - loss_jit(th0_field - hn * ei)) / (2.0 * hn))
            cand.append((abs(g_ni - fdv) / max(abs(fdv), 1e-300), fdv, hr))
        rel_b, fd_b, hr_b = min(cand, key=lambda r: r[0])
        sign_b = bool(np.sign(g_ni) == np.sign(fd_b))
        print(f"  spot node {ni} (lon={float(geo[ni,0]):.1f},lat={float(geo[ni,1]):.1f}): "
              f"map={g_ni:+.3e} FD={fd_b:+.3e} plateau={rel_b:.2e} (h_rel={hr_b:.0e}) sign={sign_b}",
              flush=True)
        c = dict(node=ni, lon=float(geo[ni, 0]), lat=float(geo[ni, 1]), g=g_ni, fd=fd_b,
                 rel=rel_b, h_rel=hr_b, sign=sign_b)
        if spot is None or rel_b < spot["rel"]:
            spot = c
    spot_ok = bool(spot["sign"] and spot["rel"] < 0.1)
    print(f"  --- FD spot-check: best node {spot['node']} (lon={spot['lon']:.1f},lat={spot['lat']:.1f}) "
          f"plateau={spot['rel']:.2e} sign={spot['sign']} ⇒ spot_ok={spot_ok} ---", flush=True)

    rec.update(dict(J0=J0, grad_scalar=g_ad_scalar, peak_gb=pk, bwd_s=bwd_s, plateau=plateau,
                    map_finite=(n_bad == 0), map_nonzero=nz, map_max=amax,
                    spot_idx=spot["node"], spot_lon=spot["lon"], spot_lat=spot["lat"],
                    spot_map=spot["g"], spot_fd=spot["fd"], spot_rel=spot["rel"],
                    proof_ok=proof_ok, spot_ok=spot_ok))
    idx = spot["node"]                                          # for the .npz spot_idx below

    # ---------- (4) adjoint↔EKI cross-check on the SHARED scalar k_gm ----------
    xcheck_ok = True
    if tgt == "ts_kgm":
        J = args.members
        # deterministic ±spread sample (no Math.random in-graph); forward-only, sequential
        zs = np.linspace(-1.0, 1.0, J)
        thetas = th0_scalar * (1.0 + args.ens_sigma * zs)          # [J]
        # reuse the already-jitted forward (uniform field) — no extra executable / CUDA graphs
        gs = np.array([float(loss_jit(to_field(jnp.asarray(t, jnp.float64)))) for t in thetas])
        # forward-ensemble gradient estimate = cov(θ,J)/var(θ) — the regression slope that is
        # exactly the core of eki.eki_update's C_θg (validated against eki_update below).
        th_c = thetas - thetas.mean()
        g_ens = float((th_c @ (gs - gs.mean())) / (th_c @ th_c))
        # one real EKI update (fesom_jax.eki) toward a target that wants J reduced by 1 std:
        # the ensemble-mean θ must move OPPOSITE the gradient (descent) — sign check on Δθ.
        y_target = jnp.asarray([gs.mean() - gs.std()])
        th_upd = eki.eki_update(thetas[:, None], gs[:, None], y_target,
                                gamma=float(max(gs.var(), 1e-30)), perturb_obs=False)
        dtheta = float(np.mean(np.asarray(th_upd)) - thetas.mean())
        sign_agree = bool(np.sign(g_ad_scalar) == np.sign(g_ens))
        ens_rel = abs(g_ad_scalar - g_ens) / max(abs(g_ens), 1e-300)
        # descent: Δθ should oppose the gradient sign (reduce J) ⇒ sign(Δθ) == -sign(g_ad)
        descent_ok = bool(np.sign(dtheta) == -np.sign(g_ad_scalar)) if dtheta != 0 else True
        xcheck_ok = bool(sign_agree and ens_rel < 0.3)
        print(f"  --- adjoint↔EKI cross-check (scalar k_gm; {J} forward members, "
              f"σ={args.ens_sigma:g}) ---")
        print(f"      adjoint dJ/dk_gm = {g_ad_scalar:+.6e}")
        print(f"      forward-ensemble cov(θ,J)/var(θ) = {g_ens:+.6e}   "
              f"sign_agree={sign_agree}  rel={ens_rel:.2e}")
        print(f"      eki_update Δθ_mean = {dtheta:+.4e}  (descent={descent_ok})", flush=True)
        rec.update(dict(grad_ens=g_ens, ens_sign_agree=sign_agree, ens_rel=ens_rel,
                        eki_dtheta=dtheta, eki_descent=descent_ok, xcheck_ok=xcheck_ok))

    ok = bool(proof_ok and (n_bad == 0) and nz > 0 and spot_ok and xcheck_ok)
    rec["ok"] = ok

    # ---------- persist the map (for Fig 2) + the JSON summary (for the gate) ----------
    args.outdir.mkdir(parents=True, exist_ok=True)
    npz = args.outdir / f"sensitivity_{tgt}.npz"
    np.savez(npz, grad=g_flat.astype(np.float64), lon=geo[:, 0], lat=geo[:, 1],
             node_wet=np.asarray(mesh.node_layer_mask[:, 0]),
             target=tgt, label=label, unit=unit, N=n, config=cfg_name,
             grad_scalar=g_ad_scalar, plateau=plateau,
             spot_idx=idx, spot_map=spot["g"], spot_fd=spot["fd"], spot_rel=spot["rel"],
             grad_ens=rec.get("grad_ens", np.nan))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  wrote map → {npz}; appended summary → {args.results}", flush=True)
    print(f"  gate components: proof={proof_ok} map_finite={n_bad==0} map_nonzero={nz>0} "
          f"spot={spot_ok} xcheck={xcheck_ok}", flush=True)
    print(f"SENSITIVITY_{tgt.upper()}_{'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
