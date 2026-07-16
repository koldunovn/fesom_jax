"""S.6 gate: the distributed SSH CG solve (:mod:`fesom_jax.ssh`).

The load-bearing iteration-count determinism (review #1): the forward PCG early-stops
at the loose ``soltol=1e-5``, so a **single**-iteration drift moves ``d_eta`` by
``~1e-5·‖b‖`` — seven orders above the 1e-12 per-substep budget. The sharded residual
is a ``psum`` (device-identical) so the trip count CAN drift only if a residual lands
within the ~1e-12 reassociation margin of the threshold. We gate on the **real CORE2
KPP+GM+ice** ``ssh_rhs`` (captured by ``scripts/debug/capture_core2_ssh_rhs.py``, NOT pi):

* 2/4-device sharded ``solve_ssh`` (under ``shard_map``, ``custom_linear_solve`` +
  ``all_gather``-in-``while_loop``) == single-device ``d_eta`` on **owned** nodes,
* the forward PCG iteration count is **identical** N-device vs 1-device (cold start
  ``x0=0`` and warm start ``x0=d_eta_step1``),
* the zero-rhs short-circuit still holds,
* the serial ``NP=1`` sharded operator + identity halo == the dense solve (no-op).

The multi-device parts need CPU fake-devices and SKIP otherwise; the ``NP=1`` no-op
runs on a single device.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import PartitionSpec

from fesom_jax import halo, partit, reductions, shard_mesh, ssh
from fesom_jax.config import MAXITER, SOLTOL
from fesom_jax.mesh import load_mesh

CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
SSH_RHS_DIR = Path(__file__).resolve().parents[2] / "data" / "ssh_rhs_core2"
NDEV = len(jax.devices())
DT = 1800.0          # MUST match scripts/debug/capture_core2_ssh_rhs.py (the operator dt)

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir()
    or not (SSH_RHS_DIR / "ssh_rhs_step1.npy").is_file(),
    reason="CORE2 mesh / dist partitions / captured ssh_rhs missing "
           "(run scripts/debug/capture_core2_ssh_rhs.sbatch)",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _gather_pad(arr_global, mylist, Lmax, pad=0.0):
    """Gather a global ``[nod2D]`` field by a device id list, pad to ``[Lmax]``."""
    out = np.full(Lmax, pad, dtype=np.float64)
    out[: mylist.size] = np.asarray(arr_global)[mylist]
    return out


def _single_device_solve(op, ssh_rhs, x0):
    """Single-device ``solve_ssh`` d_eta + forward PCG iteration count (the
    reference). The iteration count uses the diagnostic ``_pcg(return_iters=True)``
    with the same warm-start ``rtol_fwd`` ``solve_ssh`` uses."""
    d_eta = np.asarray(ssh.solve_ssh(op, jnp.asarray(ssh_rhs), x0=jnp.asarray(x0)))
    matvec = lambda x: ssh.ssh_matvec(op, x)        # noqa: E731
    precond = ssh.ssh_precond_for(op)               # MITgcm, or Chebyshev if op.cheb
    x0d = lax.stop_gradient(jnp.asarray(x0))
    b_eff = jnp.asarray(ssh_rhs) - matvec(x0d)
    nrhs = ssh_rhs.shape[0]
    rtol = SOLTOL * jnp.sqrt(jnp.sum(jnp.asarray(ssh_rhs) ** 2) / nrhs)
    _, iters = ssh._pcg(matvec, precond, b_eff, jnp.zeros_like(b_eff), SOLTOL,
                        MAXITER, rtol_abs=rtol, return_iters=True)
    return d_eta, int(iters)


def _shard_inputs(mesh, part, ssh_rhs_global, x0_global, cheb=None):
    """Build the per-device sharded operator, halo map, rhs and warm-start for a
    partition — all ``[P, …]`` then folded to ``[P*…]`` for ``PartitionSpec('p')``.
    ``cheb`` (a degree) enables the Chebyshev preconditioner on the global op first."""
    op = ssh.build_ssh_operator(mesh, dt=DT)
    if cheb:
        op = ssh.enable_cheb_precond(op, cheb)
    sop = ssh.partition_ssh_operator(op, part)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    P, Lmax, nnz = sop.P, sop.Lmax_nod, sop.nnz_max
    src_dev, src_lane = sm.exchange["nod"]          # [P, Lmax] each
    omask = sm.owned_mask["nod"]                    # [P, Lmax] bool
    rhs_PL = np.stack([_gather_pad(ssh_rhs_global, part.myList_nod2D[d], Lmax)
                       for d in range(P)])
    x0_PL = np.stack([_gather_pad(x0_global, part.myList_nod2D[d], Lmax)
                      for d in range(P)])
    return sop, src_dev, src_lane, omask, rhs_PL, x0_PL


def _sharded_solve(mesh, part, ssh_rhs_global, x0_global, npes, cheb=None):
    """Run the sharded ``solve_ssh`` under ``shard_map`` → ``d_eta`` ``[P, Lmax]``
    and (separately) the forward PCG iteration count (replicated scalar)."""
    sop, sdev, slane, omask, rhs_PL, x0_PL = _shard_inputs(
        mesh, part, ssh_rhs_global, x0_global, cheb=cheb)
    P, Lmax, nnz, Ng = sop.P, sop.Lmax_nod, sop.nnz_max, int(mesh.nod2D)
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    spec = PartitionSpec("p")

    def fold(a):                                    # [P, X] -> [P*X]
        a = jnp.asarray(a)
        return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])

    rows = fold(sop.rows); cols = fold(sop.cols)
    sv = fold(sop.stiff_vals); pv = fold(sop.precond_vals); diagv = fold(sop.diag)
    sd = fold(sdev).astype(jnp.int32); sl = fold(slane).astype(jnp.int32)
    om = fold(omask); rhs = fold(rhs_PL); x0 = fold(x0_PL)

    def _ops(rows, cols, sv, pv, diagv, sd, sl, om):
        op_local = ssh.SSHOperator(rows=rows, cols=cols, stiff_vals=sv,
                                   precond_vals=pv, diag=diagv, n_nodes=Lmax,
                                   cheb=sop.cheb)
        h = ssh.SSHHalo(src_dev=sd, src_lane=sl, owned_mask=om, n_global=Ng,
                        axis_name="p")
        return op_local, h

    def solve_body(rows, cols, sv, pv, diagv, sd, sl, om, rhs, x0):
        op_local, h = _ops(rows, cols, sv, pv, diagv, sd, sl, om)
        return ssh.solve_ssh(op_local, rhs, x0=x0, halo=h)

    def iters_body(rows, cols, sv, pv, diagv, sd, sl, om, rhs, x0):
        op_local, h = _ops(rows, cols, sv, pv, diagv, sd, sl, om)
        x0d = lax.stop_gradient(x0)
        matvec = lambda x: ssh.ssh_matvec(op_local, x, h)        # noqa: E731
        precond = ssh.ssh_precond_for(op_local, h)
        b_eff = rhs - matvec(x0d)
        reduce_fn = lambda u, v: reductions.global_dot(u, v, h.owned_mask, "p")  # noqa: E731
        rtol = SOLTOL * jnp.sqrt(reduce_fn(rhs, rhs) / Ng)
        _, it = ssh._pcg(matvec, precond, b_eff, jnp.zeros_like(b_eff), SOLTOL,
                         MAXITER, rtol_abs=rtol, reduce=reduce_fn, n_global=Ng,
                         return_iters=True)
        return it

    in_specs = (spec,) * 10
    d_eta = jax.shard_map(solve_body, mesh=jmesh, in_specs=in_specs, out_specs=spec)(
        rows, cols, sv, pv, diagv, sd, sl, om, rhs, x0)
    iters = jax.shard_map(iters_body, mesh=jmesh, in_specs=in_specs,
                          out_specs=PartitionSpec())(
        rows, cols, sv, pv, diagv, sd, sl, om, rhs, x0)
    return np.asarray(d_eta).reshape(P, Lmax), int(iters)


# --------------------------------------------------------------------------
# 1. N-vs-1 d_eta + iteration count, on the captured CORE2 ssh_rhs
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
@pytest.mark.parametrize("case", ["cold", "warm"])
def test_sharded_solve_matches_single_device(npes, case):
    """Sharded ``solve_ssh`` d_eta == single-device on owned nodes, and the forward
    PCG iteration count is identical — cold start (``x0=0``) and warm start."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    part = partit.read_partition(CORE2_DIST, npes)
    if case == "cold":
        ssh_rhs = np.load(SSH_RHS_DIR / "ssh_rhs_step1.npy")
        x0 = np.zeros(mesh.nod2D)
    else:
        ssh_rhs = np.load(SSH_RHS_DIR / "ssh_rhs_step2.npy")
        x0 = np.load(SSH_RHS_DIR / "d_eta_step1.npy")

    d_eta_ref, iters_1 = _single_device_solve(op, ssh_rhs, x0)
    d_eta_N, iters_N = _sharded_solve(mesh, part, ssh_rhs, x0, npes)

    # iteration-count determinism — the load-bearing assertion (review #1)
    assert iters_N == iters_1, (
        f"{case} npes={npes}: sharded CG took {iters_N} iters vs {iters_1} "
        f"single-device — a drift moves d_eta by ~1e-5·‖b‖")

    # owned-node d_eta match (reduction/scatter reassociation budget)
    scale = max(float(np.max(np.abs(d_eta_ref))), 1e-12)
    worst = 0.0
    for d in range(npes):
        md = int(part.myDim_nod2D[d])
        owned_gids = part.myList_nod2D[d][:md]
        diff = np.max(np.abs(d_eta_N[d, :md] - d_eta_ref[owned_gids]))
        worst = max(worst, diff)
    assert worst <= 1e-9 * scale, (
        f"{case} npes={npes}: owned d_eta max|Δ|={worst:.3e} (scale {scale:.3e}, "
        f"rel {worst/scale:.3e}) exceeds the 1e-9 reassociation budget")


# --------------------------------------------------------------------------
# 2. Zero-rhs short-circuit (sharded)
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_sharded_zero_rhs_short_circuit(npes):
    """A zero ``ssh_rhs`` with ``x0=0`` ⇒ ``d_eta=0`` on owned nodes (the C's
    ``s0==0`` short-circuit, now inside ``shard_map``)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, npes)
    zero = np.zeros(mesh.nod2D)
    d_eta_N, iters_N = _sharded_solve(mesh, part, zero, zero, npes)
    assert iters_N == 0
    for d in range(npes):
        md = int(part.myDim_nod2D[d])
        assert np.allclose(d_eta_N[d, :md], 0.0, atol=1e-14)


# --------------------------------------------------------------------------
# 3. Serial NP=1 sharded operator + identity halo == dense solve (no-op invariant)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not CORE2_MESH.is_dir() or not (SSH_RHS_DIR / "ssh_rhs_step1.npy").is_file(),
                    reason="CORE2 mesh / captured ssh_rhs missing")
def test_serial_sharded_solve_matches_dense():
    """``npes==1`` (synth_serial) sharded ``solve_ssh`` (identity exchange, all-owned
    mask) reproduces the dense ``solve_ssh`` to the reassociation budget — the no-op
    invariant proving the sharded code path collapses to the single-device model.
    Runs on ONE device (no fake-devices needed)."""
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    ssh_rhs = np.load(SSH_RHS_DIR / "ssh_rhs_step1.npy")
    x0 = np.zeros(mesh.nod2D)
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)

    d_eta_ref = np.asarray(ssh.solve_ssh(op, jnp.asarray(ssh_rhs), x0=jnp.asarray(x0)))
    d_eta_N, iters_N = _sharded_solve(mesh, part, ssh_rhs, x0, npes=1)
    md = mesh.nod2D
    scale = max(float(np.max(np.abs(d_eta_ref))), 1e-12)
    worst = float(np.max(np.abs(d_eta_N[0, :md] - d_eta_ref)))
    assert worst <= 1e-9 * scale, f"serial sharded d_eta max|Δ|={worst:.3e} (scale {scale:.3e})"


# --------------------------------------------------------------------------
# 4. partition_ssh_operator loop-bound guard (pure host — needs only mesh + dist)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing")
@pytest.mark.parametrize("npes", [2, 4])
def test_partition_operator_loop_bound(npes):
    """``partition_ssh_operator`` builds without dropping any NONZERO owned-row
    entry (the operator loop-bound holds — far columns are exactly-zero), and the
    serial case is the identity remap."""
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    part = partit.read_partition(CORE2_DIST, npes)
    sop = ssh.partition_ssh_operator(op, part)      # raises if a nonzero is dropped
    assert sop.P == npes
    # every local col index is a valid local node lane (< Lmax_nod), never -1
    assert int(sop.cols.min()) >= 0 and int(sop.cols.max()) < sop.Lmax_nod
    assert int(sop.rows.min()) >= 0 and int(sop.rows.max()) < sop.Lmax_nod


# --------------------------------------------------------------------------
# 5. CGPOLY — the Chebyshev polynomial preconditioner (M7 lever, kokkos Task E.3)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not CORE2_MESH.is_dir() or not (SSH_RHS_DIR / "ssh_rhs_step1.npy").is_file(),
                    reason="CORE2 mesh / captured ssh_rhs missing")
@pytest.mark.parametrize("degree", [2, 3])
def test_cheb_precond_equal_tolerance_fewer_iters(degree):
    """Dense CORE2, the captured real step-1 ``ssh_rhs``: the Chebyshev-preconditioned
    solve stops at the SAME unpreconditioned-residual tolerance (the stop criterion is
    precond-independent ⇒ equal-tolerance by construction, verified on the actual
    residual) with SUBSTANTIALLY fewer iterations. The E.3 verify wants ≥1.8× at
    k=2–3; the hard gate here is ≥1.5× (kill-fast: if this fails, the lever dies per
    spec, not the suite's calibration). Both early-stopped iterates are reported
    against each other at the loose-tolerance level."""
    mesh = load_mesh(CORE2_MESH)
    op0 = ssh.build_ssh_operator(mesh, dt=DT)
    ssh_rhs = np.load(SSH_RHS_DIR / "ssh_rhs_step1.npy")
    x0 = np.zeros(mesh.nod2D)
    opc = ssh.enable_cheb_precond(op0, degree)
    deg, lmin, lmax = opc.cheb
    assert deg == degree and 0.0 < lmin < lmax

    d_ref, it_ref = _single_device_solve(op0, ssh_rhs, x0)
    d_chb, it_chb = _single_device_solve(opc, ssh_rhs, x0)

    # equal tolerance, verified on the actual unpreconditioned residuals
    rms_b = float(np.sqrt(np.mean(ssh_rhs ** 2)))
    for tag, x in (("mitgcm", d_ref), ("cheb", d_chb)):
        r = np.asarray(ssh.ssh_matvec(op0, jnp.asarray(x))) - ssh_rhs
        rms_r = float(np.sqrt(np.mean(r ** 2)))
        assert rms_r <= SOLTOL * rms_b, (
            f"{tag}: RMS residual {rms_r:.3e} above tolerance {SOLTOL * rms_b:.3e}")

    ratio = it_ref / max(it_chb, 1)
    scale = max(float(np.max(np.abs(d_ref))), 1e-12)
    rel = float(np.max(np.abs(d_chb - d_ref))) / scale
    print(f"CHEB_ITERS k={degree}: mitgcm={it_ref} cheb={it_chb} ratio={ratio:.2f} "
          f"lam=[{lmin:.4g},{lmax:.4g}] d_eta rel maxdiff={rel:.2e}")
    assert it_chb < it_ref, "Chebyshev must reduce the CG iteration count"
    assert ratio >= 1.5, (
        f"CGPOLY k={degree}: iters ratio {ratio:.2f} < 1.5 — the lever misses its "
        f"spec (E.3 wants ≥1.8× at k=2–3); kill or retune cheb_kappa")


@avail
@pytest.mark.parametrize("npes", [2])
def test_cheb_sharded_matches_single_device(npes):
    """Sharded Chebyshev-preconditioned solve (k=3): the iteration count is
    device-count-identical (the same load-bearing invariant as the MITgcm path —
    the polynomial adds SpMVs but no new reductions) and owned-node d_eta matches
    the single-device cheb solve to the reassociation budget."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    opc = ssh.enable_cheb_precond(ssh.build_ssh_operator(mesh, dt=DT), 3)
    part = partit.read_partition(CORE2_DIST, npes)
    ssh_rhs = np.load(SSH_RHS_DIR / "ssh_rhs_step1.npy")
    x0 = np.zeros(mesh.nod2D)

    d_eta_ref, iters_1 = _single_device_solve(opc, ssh_rhs, x0)
    d_eta_N, iters_N = _sharded_solve(mesh, part, ssh_rhs, x0, npes, cheb=3)

    assert iters_N == iters_1, (
        f"cheb npes={npes}: sharded CG took {iters_N} iters vs {iters_1} "
        f"single-device — iteration-count drift under the polynomial precond")
    scale = max(float(np.max(np.abs(d_eta_ref))), 1e-12)
    worst = 0.0
    for d in range(npes):
        md = int(part.myDim_nod2D[d])
        owned_gids = part.myList_nod2D[d][:md]
        diff = np.max(np.abs(d_eta_N[d, :md] - d_eta_ref[owned_gids]))
        worst = max(worst, diff)
    assert worst <= 1e-9 * scale, (
        f"cheb npes={npes}: owned d_eta max|Δ|={worst:.3e} (scale {scale:.3e}) "
        f"exceeds the 1e-9 reassociation budget")
    print(f"CHEB_SHARDED_OK npes={npes} iters={iters_N}")
