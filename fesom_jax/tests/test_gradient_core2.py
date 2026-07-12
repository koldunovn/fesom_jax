"""Task 5.8 — GATE 5: the gradient gate on a CORE2 slice (the assembled model).

The pi gradient gate (:mod:`test_gradient`) proves the hard AD patterns on the small
mesh. This module re-runs the **masked-NaN probe** on the *assembled CORE2 model* — the
real PHC IC + JRA55 bulk + SSS/runoff + shortwave-penetration forcing — so the new
Phase-5 differentiable seams (SST→``heat_flux``→``bc_T`` and current→``stress``) are in
the backward path alongside every previously-guarded trap (the ``eos`` ``bvfreq``
``1/zdiff``, the ``tracer_diff`` ``dZ``, the FCT ``qr4c`` ``Z``-stencil).

This suite test is deliberately **N=1** (one assembled step, no scan): it is the
cheapest backward through the full CORE2 step yet still traverses every masked-divide
lane, so it fits a CPU node and is a fast regression. The *quantitative* AD↔FD plateau
through the CG and the multi-step checkpointed-backward **memory** confirmation run at
a larger CORE2 slice on a GPU — ``scripts/archive/core2_grad_gate.py`` (Task 5.8 GATE-5
deliverable), not in the CPU suite.

SKIPS cleanly if the CORE2 mesh / PHC IC cache / JRA55 NetCDF are absent.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
JRA_DIR = Path("/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0")
DT = 500.0
YEAR = 1958

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").exists()
         and (JRA_DIR / f"uas.{YEAR}.nc").is_file()),
    reason="needs CORE2 mesh export + PHC IC cache + JRA55 NetCDF (Task 5.1/5.2/5.3)",
)


@pytest.fixture(scope="module")
def core2():
    """Build the CORE2 model + 1-step forcing once (host build ~tens of seconds)."""
    import numpy as _np

    from fesom_jax import surface_forcing, ssh
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import phc_initial_state

    mesh = load_mesh(MESH_DIR)
    state = phc_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=_np.asarray(state.T[:, 0]))
    sfs = cf.stack(surface_forcing.dates_for_steps(YEAR, DT, 1))   # leading axis [1]
    return dict(mesh=mesh, state=state, op=op, fs=cf.static, sfs=sfs)


def _mean_sst(state, mesh):
    import jax.numpy as jnp
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    return jnp.sum(jnp.where(wet0, state.T[:, 0], 0.0)) / jnp.sum(wet0)


def test_grad_ic_field_finite_core2(core2):
    """``d(mean SST)/d(T₀)`` over one assembled CORE2 step is finite **everywhere** —
    including the below-bottom / masked lanes (the strong masked-NaN probe a scalar
    gradient misses). This is the CORE2 analog of the pi ``test_grad_ic_field_finite``:
    the same gradient now flows through the real bulk forcing (SST→heat_flux→bc_T) and
    the CORE2 ragged-depth masks. Asserts finite, nonzero on wet layers, and exactly 0
    on the masked lanes (the loss cannot depend on below-bottom T)."""
    import jax
    import jax.numpy as jnp

    from fesom_jax.integrate import integrate

    mesh, st0, op, fs, sfs = (core2[k] for k in ("mesh", "state", "op", "fs", "sfs"))
    mlay = np.asarray(mesh.node_layer_mask)

    def loss(T0):
        s = dataclasses.replace(st0, T=T0)        # keep T_old = the constant base (AB2)
        fin = integrate(s, mesh, op, None, n_steps=1, dt=DT,
                        step_forcings=sfs, forcing_static=fs)
        return _mean_sst(fin, mesh)

    g = np.asarray(jax.grad(loss)(st0.T))
    assert np.all(np.isfinite(g)), f"{int(np.isnan(g).sum())} non-finite grad entries"
    assert np.max(np.abs(g[mlay])) > 0.0, "IC gradient identically zero on wet layers"
    assert np.max(np.abs(g[~mlay])) == 0.0, "below-bottom lanes carry spurious gradient"


def test_grad_kver_finite_core2(core2):
    """``d(mean SST)/d(k_ver)`` over one assembled CORE2 step is finite and nonzero —
    the PP background diffusivity (the ML-hook seam) routes through ``Kv`` → the vertical
    tracer-diffusion solve (whose RHS also carries the bulk ``heat_flux→bc_T`` forcing).
    At step 1 ``uv=0`` ⇒ the PP shear term vanishes ⇒ ``Kv = k_ver`` additively on stable
    columns, so this is a smooth, finite gradient. (k_ver does NOT reach the CG at N=1 —
    the CG is upstream of the diffusion and depends on ``uv``/``du`` i.e. ``a_ver``; the
    CG path is the next test. The quantitative AD↔FD plateaus are the GPU gate
    ``scripts/archive/core2_grad_gate.py``.)"""
    import jax
    import jax.numpy as jnp

    from fesom_jax.config import A_VER
    from fesom_jax.integrate import integrate
    from fesom_jax.params import Params

    mesh, st0, op, fs, sfs = (core2[k] for k in ("mesh", "state", "op", "fs", "sfs"))

    def loss(kver):
        p = Params(k_ver=kver, a_ver=jnp.asarray(A_VER, jnp.float64))
        fin = integrate(st0, mesh, op, None, n_steps=1, params=p, dt=DT,
                        step_forcings=sfs, forcing_static=fs)
        return _mean_sst(fin, mesh)

    g_ad = float(jax.grad(loss)(jnp.asarray(1e-4, jnp.float64)))
    assert np.isfinite(g_ad) and g_ad != 0.0, f"d(mean SST)/d(k_ver) = {g_ad}"


def test_grad_cg_transpose_core2(core2):
    """The CG ``custom_linear_solve`` implicit-diff transpose **converges on the CORE2
    operator** (≈40-iter forward, ~hundreds tight) — the AD-critical solver at 40× pi
    scale. For ``d_eta(b) = S⁻¹·b`` and ``f(b) = ½‖d_eta‖²``, calculus gives ∇_b f =
    S⁻ᵀ(S⁻¹b) = S⁻¹·d_eta (S symmetric). So the AD cotangent (computed by the tight
    transpose solve) must equal an INDEPENDENT tight solve of ``S⁻¹·d_eta``. This isolates
    the CG gradient from the model's non-smooth FCT/convection (no time loop) and confirms
    the transpose tolerance is reached on the big matrix — the CORE2 analog of the pi
    Task-2.7 ssh AD gate. (A *param* gradient can't probe this on CORE2: ``d_eta`` is
    physically insensitive to ``a_ver`` — the barotropic SSH doesn't depend on how vertical
    viscosity redistributes momentum — and ``k_ver`` doesn't reach the CG in one step.)

    The check is the **residual** ``‖S·g_ad − d_eta‖/‖d_eta‖``: the AD cotangent must SOLVE
    the system (``S·g_ad = d_eta``), which independently confirms the transpose converged to
    the true ``S⁻¹`` on the big matrix — strictly stronger than comparing it to another run
    of the same solver (a non-converging / wrong-preconditioner CG would pass that)."""
    import jax
    import jax.numpy as jnp

    from fesom_jax.ssh import solve_ssh, ssh_matvec

    op = core2["op"]
    n = int(op.n_nodes)
    # a representative O(1) smooth RHS (the solve is linear ⇒ the relative check is
    # scale/shape-robust; deterministic so the gate is reproducible).
    b = jnp.asarray(np.cos(np.arange(n) * 0.017) + 0.3 * np.sin(np.arange(n) * 0.0031))

    def f(rhs):                                             # ½‖S⁻¹·rhs‖²  ⇒  ∇ = S⁻¹·d_eta
        d_eta = solve_ssh(op, rhs, forward_tol=1e-13)
        return 0.5 * jnp.sum(d_eta * d_eta)

    g_ad = jax.grad(f)(b)                                   # tight transpose solve cotangent
    d_eta = solve_ssh(op, b, forward_tol=1e-13)             # = S⁻¹·b (the AD's cotangent)
    assert bool(jnp.isfinite(g_ad).all()), "CG transpose cotangent has non-finite lanes"
    rel = float(jnp.linalg.norm(ssh_matvec(op, g_ad) - d_eta) / jnp.linalg.norm(d_eta))
    assert rel < 1e-6, f"CG implicit-diff transpose residual ‖S·g_ad−d_eta‖/‖d_eta‖={rel:.2e}"
