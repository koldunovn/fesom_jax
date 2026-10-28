"""The ELL layout of the SSH operator (``ssh.to_ell`` / ``ssh.to_ell_sharded``, opt-in ``gpu.ssh_ell``): the
matvecs add each row's products in the COO order of that row instead of a ``segment_sum``. Round-off only, not
bitwise: in the fused add chain XLA:CPU contracts each multiply-add into an FMA (job 28291286: 1 ulp on 42 % of a
CORE2 matvec's rows, max rel 1.4e-11 on near-cancelling rows). The layout itself is exact (the small example)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import partit, ssh
from fesom_jax.mesh import load_mesh

CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
DT = 1800.0

avail = pytest.mark.skipif(not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_4").is_dir(),
                           reason="CORE2 mesh or dist partitions missing")


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(CORE2_MESH)


@pytest.fixture(scope="module")
def ops(mesh):
    op = ssh.build_ssh_operator(mesh, dt=DT)
    return {0: op, 3: ssh.enable_cheb_precond(op, 3, kappa_guess=30.0)}


@avail
@pytest.mark.parametrize("cheb", [0, 3])
def test_ell_dense_matvecs_and_solve_bitwise(ops, cheb):
    op = ops[cheb]
    eop = ssh.to_ell(op)
    assert eop.ell >= 3 and np.asarray(eop.rows).size == eop.ell * op.n_nodes
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.standard_normal(op.n_nodes))
    for f in (ssh.ssh_matvec, ssh.ssh_precond):
        a, b = np.asarray(jax.jit(f)(eop, x)), np.asarray(jax.jit(f)(op, x))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-13 * np.abs(b).max())
    rhs = ssh.ssh_matvec(op, x) * 1e-3
    solve = jax.jit(lambda o, r: ssh.solve_ssh(o, r, dt=DT))
    a, b = np.asarray(solve(eop, rhs)), np.asarray(solve(op, rhs))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9 * np.abs(b).max())     # within the CG tolerance


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_ell_sharded_local_matvecs_bitwise(ops, npes):
    part = partit.read_partition(CORE2_DIST, npes)
    sop = ssh.partition_ssh_operator(ops[3], part)
    esop = ssh.to_ell_sharded(sop)
    assert esop.nnz_max == esop.ell * sop.Lmax_nod and esop.cheb == sop.cheb
    rng = np.random.default_rng(6)
    for d in range(npes):
        lo, le = ssh.local_ssh_operator(sop, d), ssh.local_ssh_operator(esop, d)
        x = jnp.asarray(rng.standard_normal(sop.Lmax_nod))
        for f in (ssh.ssh_matvec, ssh.ssh_precond):
            a, b = np.asarray(f(le, x)), np.asarray(f(lo, x))
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-13 * np.abs(b).max())


def test_ell_layout_small_example():
    """Rows keep their COO order; missing slots are (row, col 0, value 0)."""
    op = ssh.SSHOperator(rows=jnp.asarray([1, 0, 1, 0, 1], jnp.int32),
                         cols=jnp.asarray([2, 0, 1, 1, 0], jnp.int32),
                         stiff_vals=jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
                         precond_vals=jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
                         diag=jnp.ones(3), n_nodes=3)
    e = ssh.to_ell(op)
    assert e.ell == 3
    np.testing.assert_array_equal(np.asarray(e.cols), [0, 2, 0, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(e.stiff_vals), [2.0, 1.0, 0.0, 4.0, 3.0, 0.0, 0.0, 5.0, 0.0])
    x = jnp.asarray([1.0, 10.0, 100.0])
    np.testing.assert_array_equal(np.asarray(ssh.ssh_matvec(e, x)), np.asarray(ssh.ssh_matvec(op, x)))


def test_ssh_ell_enabled(monkeypatch):
    monkeypatch.delenv("FESOM_SSH_ELL", raising=False)
    assert not ssh.ssh_ell_enabled()
    monkeypatch.setenv("FESOM_SSH_ELL", "1")
    assert ssh.ssh_ell_enabled()


@avail
@pytest.mark.parametrize("ell", [False, True])
def test_explicit_increment_matches_matrix_free(mesh, ops, ell):
    """``with_increment``: the zstar ΔA as explicit entries gives the matrix-free ``stiff_increment_matvec``
    to round-off, and the zstar solve to the CG tolerance (COO and ELL layouts)."""
    op = ops[3]
    rng = np.random.default_rng(8)
    hbar = jnp.asarray(rng.standard_normal(op.n_nodes) * 0.5)
    x = jnp.asarray(rng.standard_normal(op.n_nodes))
    iop = ssh.with_increment(op, mesh)
    if ell:
        iop = ssh.to_ell(iop)
    factor = float(ssh.G) * DT * float(ssh.SSH_ALPHA) * float(ssh.SSH_THETA)
    live = ssh.dataclasses.replace(iop, stiff_vals=iop.stiff_vals + ssh._increment_values(iop, mesh, hbar, factor))
    a = np.asarray(ssh.ssh_matvec(live, x))
    b = np.asarray(ssh.ssh_matvec(op, x) + ssh.stiff_increment_matvec(mesh, hbar, x, dt=DT))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * np.abs(b).max())
    rhs = ssh.ssh_matvec(op, x) * 1e-3
    s_exp = np.asarray(jax.jit(lambda r, h: ssh.solve_ssh(iop, r, mesh=mesh, hbar=h, dt=DT))(rhs, hbar))
    s_mf = np.asarray(jax.jit(lambda r, h: ssh.solve_ssh(op, r, mesh=mesh, hbar=h, dt=DT))(rhs, hbar))
    np.testing.assert_allclose(s_exp, s_mf, rtol=0, atol=1e-6 * np.abs(s_mf).max())
