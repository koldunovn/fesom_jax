"""CORE2 GPU speed plan, Task 6 (A4): the SSH solve gathered onto every device (opt-in
``ssh.gathered``, off by default — the user, 2026-10-20: "just an option").

Inside ``shard_map`` every device all-gathers the owned lanes of the right-hand side, the warm
start and (zstar) ``hbar``, runs the DENSE single-device solve on the whole 2-D system, and keeps
its own lanes: one collective per solve instead of ~2 per CG iteration. On identical global inputs
the redundant solve IS the dense solve: gathered sharded ``d_eta`` == the dense ``solve_ssh`` on
every valid lane (owned and halo), with and without the Chebyshev preconditioner, cold and warm
start — BITWISE under linfs, and to round-off under zstar (on identical inputs XLA compiles the zstar
increment's element sums differently inside the shard_map program: measured max |Δ| 1.2e-16 on
CORE2, 2026-10-20). Derivatives go through the dense solve's
implicit rule; the gathered gradient (the all_gather transposes to a reduce-scatter) matches the
dense gradient to round-off on owned lanes and is exactly zero on halo and padding lanes.

Uses the captured CORE2 ``ssh_rhs`` of ``test_ssh_sharded.py`` and CPU fake devices:

  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 PY -m pytest \\
      fesom_jax/tests/test_ssh_gathered.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from fesom_jax import halo, partit, ssh
from fesom_jax.mesh import load_mesh
from fesom_jax.run_config import RunConfig, load_yaml

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
SSH_RHS_DIR = ROOT / "data" / "ssh_rhs_core2"
NDEV = len(jax.devices())
DT = 1800.0          # the operator dt of the captured right-hand sides

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_4").is_dir()
    or not (SSH_RHS_DIR / "ssh_rhs_step2.npy").is_file(),
    reason="CORE2 mesh / dist partitions / captured ssh_rhs missing")


@pytest.fixture(scope="module")
def core2():
    return load_mesh(CORE2_MESH)


def _inputs(mesh, case):
    if case == "cold":
        rhs = np.load(SSH_RHS_DIR / "ssh_rhs_step1.npy")
        x0 = np.zeros(mesh.nod2D)
    else:
        rhs = np.load(SSH_RHS_DIR / "ssh_rhs_step2.npy")
        x0 = np.load(SSH_RHS_DIR / "d_eta_step1.npy")
    # a smooth zstar thickness anomaly [m] (sea-surface-height scale)
    lon, lat = np.asarray(mesh.coord_nod2D[:, 0]), np.asarray(mesh.coord_nod2D[:, 1])
    hbar = 0.3 * np.sin(2.0 * lon) * np.cos(lat)
    return rhs, x0, hbar


def _lanes(a_global, part, Lmax):
    """Global ``[nod2D]`` -> ``[P, Lmax]`` by each device's lane list (owned, then halo; pad 0)."""
    out = np.zeros((part.npes, Lmax))
    for d in range(part.npes):
        ids = part.myList_nod2D[d]
        out[d, :ids.size] = np.asarray(a_global)[ids]
    return out


def _fold(a):
    a = jnp.asarray(a)
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


def _gathered(mesh, op, part, rhs, x0, hbar, *, zstar):
    """``solve_ssh_gathered`` under ``shard_map``; returns a function of the folded
    ``(rhs, hbar)`` (warm start fixed) and its folded inputs."""
    g = ssh.build_ssh_gather(op, mesh, part)
    P, Lmax = part.npes, g.Lmax
    jmesh = halo.device_mesh(devices=jax.devices()[:P])
    spec, rep = PartitionSpec("p"), PartitionSpec()
    gl, gs = ssh.gather_inputs(g)

    def body(gl_, gs_, r, x, hb):
        gg = ssh.local_ssh_gather(gl_, gs_)
        return ssh.solve_ssh_gathered(gg, r, x, hb if zstar else None, dt=DT)

    fn = jax.jit(jax.shard_map(body, mesh=jmesh, in_specs=(rep, spec, spec, spec, spec),
                               out_specs=spec, check_vma=False))
    args = (gl, gs, _fold(_lanes(rhs, part, Lmax)), _fold(_lanes(x0, part, Lmax)),
            _fold(_lanes(hbar, part, Lmax)))
    return fn, args, Lmax


@avail
@pytest.mark.parametrize("npes", [2, 4])
@pytest.mark.parametrize("cheb", [0, 3])
@pytest.mark.parametrize("zstar", [False, True])
@pytest.mark.parametrize("case", ["cold", "warm"])
def test_gathered_matches_dense(core2, npes, cheb, zstar, case):
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = core2
    op = ssh.build_ssh_operator(mesh, dt=DT)
    if cheb:
        op = ssh.enable_cheb_precond(op, cheb)
    part = partit.read_partition(CORE2_DIST, npes)
    rhs, x0, hbar = _inputs(mesh, case)

    dense = jax.jit(lambda r, x, hb: ssh.solve_ssh(
        op, r, x0=x, mesh=mesh if zstar else None, hbar=hb if zstar else None, dt=DT))
    ref = np.asarray(dense(jnp.asarray(rhs), jnp.asarray(x0), jnp.asarray(hbar)))

    fn, args, Lmax = _gathered(mesh, op, part, rhs, x0, hbar, zstar=zstar)
    out = np.asarray(fn(*args)).reshape(npes, Lmax)
    scale = float(np.max(np.abs(ref)))
    for d in range(npes):
        ids = part.myList_nod2D[d]
        if zstar:
            np.testing.assert_allclose(out[d, :ids.size], ref[ids], rtol=0, atol=1e-12 * scale,
                                       err_msg=f"device {d}: gathered != dense (zstar)")
        else:
            np.testing.assert_array_equal(out[d, :ids.size], ref[ids],
                                          err_msg=f"device {d}: gathered != dense (linfs)")
        assert not np.any(out[d, ids.size:]), "padding lanes must stay 0"


@avail
def test_gathered_gradient_matches_dense(core2):
    npes = 4
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = core2
    op = ssh.enable_cheb_precond(ssh.build_ssh_operator(mesh, dt=DT), 3)
    part = partit.read_partition(CORE2_DIST, npes)
    rhs, x0, hbar = _inputs(mesh, "warm")
    w = np.random.default_rng(1).normal(size=mesh.nod2D)

    def j_dense(r, hb):
        d = ssh.solve_ssh(op, r, x0=jnp.asarray(x0), mesh=mesh, hbar=hb, dt=DT)
        return jnp.sum(jnp.asarray(w) * d)

    g_rhs, g_hbar = jax.jit(jax.grad(j_dense, argnums=(0, 1)))(jnp.asarray(rhs), jnp.asarray(hbar))
    g_rhs, g_hbar = np.asarray(g_rhs), np.asarray(g_hbar)

    fn, (gl, gs, r_f, x_f, h_f), Lmax = _gathered(mesh, op, part, rhs, x0, hbar, zstar=True)
    owned = np.zeros((npes, Lmax), bool)
    w_l = np.zeros((npes, Lmax))
    for d in range(npes):
        md = int(part.myDim_nod2D[d])
        owned[d, :md] = True
        w_l[d, :md] = w[part.myList_nod2D[d][:md]]
    owned_f, w_f = jnp.asarray(owned.reshape(-1)), jnp.asarray(w_l.reshape(-1))

    def j_gath(r, hb):
        return jnp.sum(jnp.where(owned_f, w_f * fn(gl, gs, r, x_f, hb), 0.0))

    gr, gh = jax.jit(jax.grad(j_gath, argnums=(0, 1)))(r_f, h_f)
    gr, gh = np.asarray(gr).reshape(npes, Lmax), np.asarray(gh).reshape(npes, Lmax)
    for got, ref, name in ((gr, g_rhs, "rhs"), (gh, g_hbar, "hbar")):
        scale = float(np.max(np.abs(ref)))
        for d in range(npes):
            md = int(part.myDim_nod2D[d])
            np.testing.assert_allclose(got[d, :md], ref[part.myList_nod2D[d][:md]],
                                       rtol=0, atol=1e-12 * scale, err_msg=f"{name} device {d}")
            assert not np.any(got[d, md:]), f"{name}: halo/padding lanes must get no cotangent"


def test_gathered_option_off_by_default():
    """Off by default and in every config but the CORE2 GPU one (small meshes only; user 2026-10-20)."""
    assert RunConfig.defaults().ssh_gathered() is False
    assert load_yaml(ROOT / "configs" / "core2_full.yaml").ssh_gathered() is False
    assert load_yaml(ROOT / "configs" / "core2_gpu.yaml").ssh_gathered() is True
    for other in sorted((ROOT / "configs").glob("*.yaml")):
        if other.name != "core2_gpu.yaml":
            try:
                cfg = load_yaml(other)
            except Exception:              # configs this test does not own may not load standalone
                continue
            assert cfg.ssh_gathered() is False, other.name
    assert RunConfig.from_dict({"ssh": {"gathered": True}}).ssh_gathered() is True
