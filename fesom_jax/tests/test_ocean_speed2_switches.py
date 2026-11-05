"""Lane ocean_speed2's performance switches (``gpu.ice_mm_sum3``, ``ice_fct_extreme``, ``ice_scatter_fuse``, ``ssh_rhs_split``,
``cg_prefix``, ``potit_unroll``, ``column_direct``) against the code paths they replace, on CPU. Each changes how
the GPU program is built (fusions, loop structure, memory access), not the operations, so on CPU the values are the
same bits except the ice FCT switches, around which XLA's CPU program fuses and contracts otherwise (round-off). The
GPU bitwise evidence is the lane's 27-hour State comparison (configs/core3_cpl.yaml comments)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import column_kernel as ck
from fesom_jax import ops

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
avail = pytest.mark.skipif(not CORE2_MESH.is_dir(), reason="CORE2 mesh missing")
RNG = np.random.default_rng(5)


@pytest.fixture(scope="module")
def mesh():
    from fesom_jax import mesh as mesh_mod
    return mesh_mod.with_inverse_maps(mesh_mod.load_mesh(CORE2_MESH))


def _with(monkeypatch, env, fn):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = fn()
    for k in env:
        monkeypatch.delenv(k)
    return jax.tree.map(np.asarray, out)


def test_column_direct_thomas_and_scan(monkeypatch):
    n, K = 300, 48                                          # 300: not a multiple of the 128-column block
    a = RNG.standard_normal((n, K))
    c = RNG.standard_normal((n, K))
    b = 4.0 + np.abs(a) + np.abs(c)
    a[:, 0] = 0.0
    c[:, -1] = 0.0
    a, b, c, d = (jnp.asarray(v) for v in (a, b, c, RNG.standard_normal((n, K))))
    th = lambda: ck.thomas(ops._tdma_forward, ops._tdma_backward, a, b, c, d, interpret=True)  # noqa: E731
    np.testing.assert_array_equal(_with(monkeypatch, {"FESOM_COLUMN_DIRECT": "1"}, th), _with(monkeypatch, {}, th))
    m = jnp.asarray(RNG.random((n, K)) > 0.3)

    def body(prev, xs):
        x, keep = xs
        new = jnp.where(keep, jnp.minimum(x, prev + 1.0), x)
        return new, (new, new * 2.0)

    for rev in (False, True):
        sc = lambda: ck.column_scan(body, jnp.zeros((n,)), (d, m), reverse=rev, interpret=True)  # noqa: E731
        got, ref = _with(monkeypatch, {"FESOM_COLUMN_DIRECT": "1"}, sc), _with(monkeypatch, {}, sc)
        jax.tree.map(np.testing.assert_array_equal, got, ref)


def test_cg_prefix_same_iterate_and_count(monkeypatch):
    from fesom_jax import ssh
    n = 60
    A = RNG.standard_normal((n, n))
    A = jnp.asarray(A @ A.T + n * np.eye(n))
    b = jnp.asarray(RNG.standard_normal(n))
    dinv = 1.0 / jnp.diag(A)
    run = lambda: ssh._pcg(lambda x: A @ x, lambda r: dinv * r, b, jnp.zeros(n), 1e-10, 500,  # noqa: E731
                           return_iters=True)
    ref = _with(monkeypatch, {}, run)
    for env in ({"FESOM_CG_PREFIX": "5"}, {"FESOM_CG_PREFIX": "200"}, {"FESOM_CG_PREFIX": "7", "FESOM_CG_BATCH": "4"}):
        got = _with(monkeypatch, env, run)
        assert int(got[1]) == int(ref[1]) and 5 < int(ref[1]) < 200
        np.testing.assert_array_equal(got[0], ref[0])


def test_potit_unroll():
    from fesom_jax import cavity
    s = jnp.asarray(34.0 + RNG.random(500))
    t = jnp.asarray(-2.0 + 3.0 * RNG.random(500))
    p = jnp.asarray(2000.0 * RNG.random(500))
    ref = np.asarray(jax.jit(cavity.potit)(s, t, p, 0.0))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FESOM_POTIT_UNROLL", "10")
        got = np.asarray(jax.jit(cavity.potit)(s, t, p, 0.0))
    np.testing.assert_array_equal(got, ref)


@avail
def test_ice_scatter_fuse_modes(mesh, monkeypatch):
    from fesom_jax import ice_evp
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    e = mesh.elem2D
    s11, s12, s22 = (jnp.asarray(RNG.standard_normal(e)) for _ in range(3))
    act = jnp.asarray(RNG.random(e) > 0.3)
    f = jax.jit(lambda *a: ice_evp.stress_div_scatter(mesh, *a))
    ref = _with(monkeypatch, {}, lambda: f(s11, s12, s22, act))
    for mode in ("1", "2", "3"):
        f = jax.jit(lambda *a: ice_evp.stress_div_scatter(mesh, *a))
        got = _with(monkeypatch, {"FESOM_ICE_SCATTER_FUSE": mode}, lambda: f(s11, s12, s22, act))
        jax.tree.map(np.testing.assert_array_equal, got, ref)


@avail
def test_ice_fct_switches(mesh, monkeypatch):
    from fesom_jax import ice_adv
    from fesom_jax.ice import IceConfig
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    n = mesh.nod2D
    a = jnp.asarray(RNG.random(n))
    m = jnp.asarray(2.0 * RNG.random(n) * (RNG.random(n) > 0.2))
    ms = jnp.asarray(0.3 * RNG.random(n))
    u, v = (jnp.asarray(0.3 * RNG.standard_normal(n)) for _ in range(2))
    cfg = IceConfig()

    def run():
        return jax.jit(lambda *x: ice_adv.fct_solve(cfg, mesh, *x))(a, m, ms, u, v)

    ref = _with(monkeypatch, {}, run)
    # max/min are exact in any order, but the CPU program around them fuses (and contracts) otherwise: round-off
    for env in ({"FESOM_ICE_FCT_EXTREME": "1"}, {"FESOM_ICE_MM_SUM3": "1"},
                {"FESOM_ICE_MM_SUM3": "1", "FESOM_ICE_FCT_EXTREME": "1"}):
        got = _with(monkeypatch, env, run)
        jax.tree.map(lambda g, r: np.testing.assert_allclose(g, r, rtol=1e-12, atol=1e-14), got, ref)


@avail
def test_ssh_rhs_split(mesh, monkeypatch):
    from fesom_jax import ssh
    e, nl = mesh.elem2D, mesh.nl
    uv = jnp.asarray(RNG.standard_normal((e, nl, 2)))
    uvr = jnp.asarray(1e-3 * RNG.standard_normal((e, nl, 2)))
    he = jnp.asarray(10.0 + RNG.random((e, nl)))
    run = lambda: jax.jit(lambda *a: ssh.compute_ssh_rhs(mesh, *a))(uv, uvr, he)  # noqa: E731
    np.testing.assert_array_equal(_with(monkeypatch, {"FESOM_SSH_RHS_SPLIT": "1"}, run), _with(monkeypatch, {}, run))


@avail
def test_barrier_switches(mesh, monkeypatch):
    """visc_upd_barrier, fct_aux_barrier, fct_flux_barrier: an optimization barrier only."""
    from fesom_jax import momentum, tracer_adv
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    e, n, ed, nl = mesh.elem2D, mesh.nod2D, mesh.edge2D, mesh.nl
    uv = jnp.asarray(0.3 * RNG.standard_normal((e, nl, 2)))
    uvr = jnp.asarray(1e-4 * RNG.standard_normal((e, nl, 2)))
    run = lambda: jax.jit(lambda a, b: momentum.visc_filt_bcksct(mesh, a, b))(uv, uvr)  # noqa: E731
    np.testing.assert_array_equal(_with(monkeypatch, {"FESOM_VISC_UPD_BARRIER": "1"}, run), _with(monkeypatch, {}, run))
    T = jnp.asarray(10.0 + RNG.standard_normal((n, nl)))
    LO = T + jnp.asarray(0.01 * RNG.standard_normal((n, nl)))
    adf_h = jnp.asarray(1e3 * RNG.standard_normal((ed, nl)))
    adf_v = jnp.asarray(1e2 * RNG.standard_normal((n, nl)))
    hn = jnp.asarray(10.0 + RNG.random((n, nl)))
    f = lambda: jax.jit(lambda *a: tracer_adv.zalesak_limit(mesh, *a))(T, LO, adf_h, adf_v, hn)  # noqa: E731
    ref = _with(monkeypatch, {}, f)
    jax.tree.map(np.testing.assert_array_equal, _with(monkeypatch, {"FESOM_FCT_AUX_BARRIER": "1"}, f), ref)
    g = lambda: jax.jit(lambda *a: tracer_adv.flux2dtracer_fct(mesh, *a))(T, LO, adf_h, adf_v, hn, hn)  # noqa: E731
    np.testing.assert_array_equal(_with(monkeypatch, {"FESOM_FCT_FLUX_BARRIER": "1"}, g), _with(monkeypatch, {}, g))
