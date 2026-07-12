#!/usr/bin/env python
"""CORE2-slice gradient gate — Phase 5, Task 5.8 (GATE 5).

Re-runs the project's AD de-risking gate on the **assembled CORE2 model** (PHC IC +
JRA55 bulk + SSS/runoff + shortwave penetration + the static ice mask). The pi gate
(``tests/test_gradient.py``) already proves the hard AD machinery (the CG
``custom_linear_solve`` transpose, the FCT subgradient, the ``eos``/``tracer_diff``/FCT
masked-divide guards) in pi's **smooth** blob regime; this confirms the SAME machinery —
plus the NEW Phase-5 forcing seams — on CORE2, and confirms the checkpointed backward
fits the A100 at 40× the node count.

⚠️ **Where FD↔AD is well-posed on CORE2 (the Task-5.8 finding).** A *multi-step* FD↔AD
plateau of ``d(mean SST)/d(k_ver)`` over the real forced trajectory does **NOT** converge:
the FCT Zalesak limiter and the convective adjustment (``max(N²,0)`` / instabmix) are
*active* under CORE2 forcing (unlike pi's dormant smooth blob), so the loss is genuinely
non-smooth in the physics parameters at the FD scale (a valid (sub)gradient, but FD across
a kink ≠ the slope). The AD gradient is still correct; FD just can't validate it there
(part [4] reports this). So the quantitative gates run in **smooth regimes**:

* **[1] N=1 d(mean SST)/d(k_ver)** — at step 1 ``uv=0`` ⇒ PP shear vanishes ⇒ ``Kv =
  k_ver`` additively on stable columns (convective mask fixed by the IC density), so the
  map is smooth → a clean FD↔AD plateau. Validates the vertical-diffusion + bulk
  ``heat_flux→bc_T`` forcing gradient on the assembled CORE2 step.
* **[2] CG implicit-diff transpose on the CORE2 operator** — for ``d_eta(b)=S⁻¹b`` and
  ``f(b)=½‖d_eta‖²``, ∇_b f = S⁻¹·d_eta (S symmetric); the AD cotangent (tight transpose
  solve) must equal an INDEPENDENT tight solve of ``S⁻¹·d_eta``. Confirms the transpose
  converges on the ~40× bigger matrix, isolated from the model's non-smooth FCT/convection.
* **[3] N d(mean SST)/d(T₀) finite EVERYWHERE incl. masked lanes** — the strong masked-NaN
  probe on the assembled CORE2 model at scale (the new bulk seams in the backward path
  beside the eos/tracer_diff/FCT guards); + the checkpointed N-step backward **peak
  memory** (CORE2 ≈ 40× pi nodes → must fit the A100; pi N=200 was 4.23 GB).
* **[4] diagnostic** — the multi-step ``d(mean SST)/d(k_ver)`` FD, reported (NOT pass/fail)
  to document the non-smoothness above.

Usage (GPU node; see scripts/archive/core2_grad_gate.sbatch):
    python scripts/archive/core2_grad_gate.py --n 20
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import core2_forcing, ssh
from fesom_jax.config import A_VER
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.ssh import solve_ssh, ssh_matvec

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 500.0
TIGHT = 1.0e-13                                   # tight CG forward stop for the solve check
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def build(year, n):
    mesh = load_mesh(MESH_DIR)
    state = core2_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, year, sst_ic=np.asarray(state.T[:, 0]))
    sfs = cf.stack(core2_forcing.dates_for_steps(year, DT, n))
    return mesh, state, op, cf.static, sfs


def report_mem(tag):
    for d in jax.devices():
        try:
            st = d.memory_stats() or {}
            pk, lim = st.get("peak_bytes_in_use"), st.get("bytes_limit")
            if pk:
                msg = f"  [{tag}] {d}: peak {pk / 1e9:.2f} GB"
                if lim:
                    msg += f" / limit {lim / 1e9:.1f} GB ({100 * pk / lim:.0f}%)"
                print(msg, flush=True)
        except Exception:
            pass


def fd_sweep(loss_jit, x0):
    """Central relative FD of ``loss_jit`` about ``x0`` over :data:`H_SWEEP`."""
    rows = []
    for h in H_SWEEP:
        xp, xm = x0 * (1.0 + h), x0 * (1.0 - h)
        rows.append((h, float((loss_jit(xp) - loss_jit(xm)) / (xp - xm))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="multi-step slice (T0 backward + diagnostic)")
    ap.add_argument("--year", type=int, default=1958)
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, st0, op, fs, sfs = build(args.year, args.n)
    sf1 = jax.tree.map(lambda x: x[:1], sfs)              # 1-step forcing slice
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    nwet0 = jnp.sum(wet0)
    mlay = np.asarray(mesh.node_layer_mask)
    print(f"[setup] built mesh+IC+forcing in {time.time()-t0:.1f}s; "
          f"N={args.n} ({args.n*DT/86400:.3f} model days)", flush=True)

    def mean_sst(s):
        return jnp.sum(jnp.where(wet0, s.T[:, 0], 0.0)) / nwet0

    passes = {}

    # ----- [1] N=1 d(mean SST)/d(k_ver): smooth (Kv = k_ver additively at step 1) -----
    def kver_loss1(kver):
        p = Params(k_ver=kver, a_ver=jnp.asarray(A_VER, jnp.float64))
        fin = integrate(st0, mesh, op, None, n_steps=1, params=p, dt=DT,
                        step_forcings=sf1, forcing_static=fs)
        return mean_sst(fin)

    k0 = jnp.asarray(1e-3, jnp.float64)                  # lift the 1-step FD signal
    g_ad = float(jax.jit(jax.grad(kver_loss1))(k0))
    rows = fd_sweep(jax.jit(kver_loss1), k0)
    plat_k = min(abs(g_ad - gf) / max(abs(gf), 1e-300) for _, gf in rows)
    print(f"\n  [1] N=1 d(mean SST)/d(k_ver) AD = {g_ad:+.8e}  (k_ver={float(k0):.0e})")
    for h, gf in rows:
        print(f"      h={h:.0e}  FD={gf:+.6e}  rel={abs(g_ad-gf)/max(abs(gf),1e-300):.2e}")
    print(f"      plateau (min rel) = {plat_k:.3e}", flush=True)
    passes["[1] kver_N1_plateau<1e-6"] = bool(np.isfinite(g_ad) and g_ad != 0.0 and plat_k < 1e-6)

    # ----- [2] CG implicit-diff transpose converges on the CORE2 operator -----
    # The AD cotangent of f(b)=½‖S⁻¹b‖² is S⁻¹·d_eta, so it must SOLVE S·g_ad = d_eta. The
    # residual ‖S·g_ad − d_eta‖/‖d_eta‖ independently confirms the transpose reached the true
    # S⁻¹ on the 40×-bigger matrix (stronger than matching another run of the same solver).
    n = int(op.n_nodes)
    bvec = jnp.asarray(np.cos(np.arange(n) * 0.017) + 0.3 * np.sin(np.arange(n) * 0.0031))

    def fcg(rhs):
        de = solve_ssh(op, rhs, forward_tol=TIGHT)
        return 0.5 * jnp.sum(de * de)

    g_ad_cg = jax.jit(jax.grad(fcg))(bvec)               # tight transpose solve cotangent
    de = solve_ssh(op, bvec, forward_tol=TIGHT)          # = S⁻¹·b (the AD's cotangent)
    rel_cg = float(jnp.linalg.norm(ssh_matvec(op, g_ad_cg) - de) / jnp.linalg.norm(de))
    print(f"\n  [2] CG transpose on CORE2 op: residual ‖S·g_ad−d_eta‖/‖d_eta‖ = {rel_cg:.3e}",
          flush=True)
    passes["[2] cg_transpose_resid<1e-6"] = bool(jnp.isfinite(g_ad_cg).all()) and rel_cg < 1e-6

    # ----- [3] N d(mean SST)/d(T0) finite everywhere incl. masked lanes + memory -----
    def loss_T0(T0):
        s = dataclasses.replace(st0, T=T0)
        fin = integrate(s, mesh, op, None, n_steps=args.n, dt=DT,
                        step_forcings=sfs, forcing_static=fs)
        return mean_sst(fin)

    t1 = time.time()
    gT = np.asarray(jax.jit(jax.grad(loss_T0))(st0.T))
    report_mem(f"T0-grad N={args.n}")
    n_bad = int((~np.isfinite(gT)).sum())
    wet_max = float(np.max(np.abs(gT[mlay]))) if mlay.any() else 0.0
    masked_max = float(np.max(np.abs(gT[~mlay]))) if (~mlay).any() else 0.0
    print(f"\n  [3] N={args.n} d(mean SST)/d(T0): non-finite={n_bad}; wet max|g|={wet_max:.3e}; "
          f"masked max|g|={masked_max:.3e}  ({time.time()-t1:.1f}s)", flush=True)
    passes["[3] T0_masked_NaN_clean"] = bool(n_bad == 0 and wet_max > 0.0 and masked_max == 0.0)

    # ----- [4] multi-step k_ver FD diagnostic (NON-smooth — documents the finding) -----
    def kver_lossN(kver):
        p = Params(k_ver=kver, a_ver=jnp.asarray(A_VER, jnp.float64))
        fin = integrate(st0, mesh, op, None, n_steps=args.n, params=p, dt=DT,
                        step_forcings=sfs, forcing_static=fs)
        return mean_sst(fin)

    kN = jnp.asarray(1e-4, jnp.float64)
    g_adN = float(jax.jit(jax.grad(kver_lossN))(kN))
    rowsN = fd_sweep(jax.jit(kver_lossN), kN)
    platN = min(abs(g_adN - gf) / max(abs(gf), 1e-300) for _, gf in rowsN)
    print(f"\n  [4] DIAGNOSTIC N={args.n} d(mean SST)/d(k_ver) AD = {g_adN:+.6e} "
          f"(multi-step, NON-smooth — FCT limiter / convective adjustment active):")
    for h, gf in rowsN:
        print(f"      h={h:.0e}  FD={gf:+.6e}  rel={abs(g_adN-gf)/max(abs(gf),1e-300):.2e}")
    print(f"      multi-step plateau (min rel) = {platN:.3e}  "
          f"(expected LOOSE — non-smooth forced trajectory; not a gate)", flush=True)

    ok = all(passes.values())
    print(f"\n  gate breakdown: {passes}")
    print(f"{'GRAD_GATE_OK' if ok else 'GRAD_GATE_FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
