"""Split-explicit SSH (fesom_jax/ssh_se.py; plan docs/plans/20261021-core2-split-explicit.md).

Task 2 (dense step): the config seam, the host tables against a literal edge loop, the SE block's algebra (eta
compatibility, trim) on synthetic inputs, the invariants over 100 pi steps (eta compatibility, trim, the cumulative
layer sum, volume), single precision, the gradient tests (dot test, FD probe, finite cold-start gradient), and the
oracle gate against the frozen Kokkos binary's dumps (skips while they are absent).
Task 3 (restart): the rings round-trip through the restart writer and a 13 + 13 resume is bitwise the 26-step run;
a CG restart into SE starts flat, an SE restart into the CG is refused.
Task 4 (CFL guard): the characteristic-polynomial stability bound and, on a compute node, CORE2's omega_max / M_min.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import forcing, ic, ssh, ssh_se
from fesom_jax import step as stepmod
from fesom_jax.ale import AleConfig
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.ssh_se import SeConfig

DT = 100.0
ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"

pi_missing = pytest.mark.skipif(not DEFAULT_PI_MESH_DIR.is_dir(),
                                reason=f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")


@pytest.fixture(scope="module")
def pi():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    mesh = ssh_se.with_tables(load_mesh())
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st0 = ssh_se.cold_start(ic.initial_state(mesh), mesh)
    return mesh, op, stress, st0


_step = jax.jit(stepmod.step, static_argnames=("dt", "is_first_step", "ale_cfg"))


def _run(mesh, op, stress, st, n, ale, first=True):
    out = []
    for i in range(n):
        st = _step(st, mesh, op, stress, dt=DT, is_first_step=(first and i == 0), ale_cfg=ale)
        out.append(st)
    return out


# --------------------------------------------------------------------------
# config seam
# --------------------------------------------------------------------------
def test_seconfig_validate_and_hashable():
    cfg = SeConfig()
    assert cfg.validate() is cfg
    assert (cfg.M, cfg.chi, cfg.visc, cfg.gamma0, cfg.gamma1, cfg.bt_drag) == (50, 0.1, "two_term", 10.0, 2750.0,
                                                                               False)
    ale = AleConfig(split_explicit=cfg)
    hash(ale)
    assert ale.validate() is ale
    for bad in (dict(M=0), dict(M=1.5), dict(chi=0.0), dict(chi=0.5), dict(visc="fb"), dict(gamma1=-1.0)):
        with pytest.raises(ValueError):
            SeConfig(**bad).validate()
    with pytest.raises(TypeError):
        AleConfig(split_explicit={"M": 50}).validate()
    # the CG default is unchanged: AleConfig() has no split-explicit
    assert AleConfig().split_explicit is None


def test_sm_coefficients_match_the_kokkos_log():
    """chi = 0.1 → the Kokkos run's printed set (``[ssh_se] subcycling live: ... beta=0.281105 zeta=-0.003689
    gamma=0.032087 delta=0.724708``, port2/m12/t8_s8b_26936251/run.log); the AB3 and AM4 weights sum to 1."""
    b, z, g, d = ssh_se.sm_coeffs(0.1)
    assert (round(b, 6), round(z, 6), round(g, 6), round(d, 6)) == (0.281105, -0.003689, 0.032087, 0.724708)
    (a0, a1, a2), (d3, d0, d1, d2) = ssh_se._weights(0.1)
    assert abs(a0 + a1 + a2 - 1.0) < 1e-15 and abs(d3 + d0 + d1 + d2 - 1.0) < 1e-15


def test_se_requires_rings_and_tables(pi):
    mesh, op, stress, st0 = pi
    ale = AleConfig(split_explicit=SeConfig(M=10))
    cg_state = dataclasses.replace(st0, se_eta=None, se_ubt=None, se_sbar_old=None)
    with pytest.raises(ValueError, match="cold_start"):
        stepmod.step(cg_state, mesh, op, stress, dt=DT, is_first_step=True, ale_cfg=ale)
    bare = dataclasses.replace(mesh, se_div_elem=None, se_div_c=None, se_elem_nb=None, se_kappa=None)
    with pytest.raises(ValueError, match="with_tables"):
        stepmod.step(st0, bare, op, stress, dt=DT, is_first_step=True, ale_cfg=ale)


# --------------------------------------------------------------------------
# host tables
# --------------------------------------------------------------------------
def _hash11(x):
    """The Kokkos self-test's integer hash → [-1, 1] (se_hash11), of the 0-based element id."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xff51afd7ed558ccd)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xc4ceb9fe1a85ec53)
        x ^= x >> np.uint64(33)
    return 2.0 * ((x >> np.uint64(11)).astype(np.float64) / 9007199254740992.0) - 1.0


@pi_missing
def test_divergence_gather_equals_the_edge_scatter(pi):
    """The Kokkos operator self-test (se_selftest): a hashed element field's divergence by the gather CSR equals the
    literal edge scatter of compute_hbar (h = 1, / areasvol) to 1e-14 of its scale."""
    mesh = pi[0]
    E = mesh.elem2D
    e = np.arange(E, dtype=np.uint64)
    U = np.stack([_hash11(2 * e), _hash11(2 * e + 1)], axis=-1)
    edges, et, cr = (np.asarray(mesh.edges), np.asarray(mesh.edge_tri), np.asarray(mesh.edge_cross_dxdy))
    ref = np.zeros(mesh.nod2D)
    for ed in range(edges.shape[0]):
        el1, el2 = et[ed]
        c1 = c2 = 0.0
        if el1 >= 0:
            c1 = U[el1, 1] * cr[ed, 0] - U[el1, 0] * cr[ed, 1]
        if el2 >= 0:
            c2 = -(U[el2, 1] * cr[ed, 2] - U[el2, 0] * cr[ed, 3])
        ref[edges[ed, 0]] += c1 + c2
        ref[edges[ed, 1]] -= c1 + c2
    ref = ref / np.asarray(mesh.areasvol)[:, 0]
    got = np.asarray(ssh_se.divergence(jnp.asarray(U), mesh))
    scale = np.abs(ref).max()
    assert np.abs(got - ref).max() <= 1e-14 * scale, np.abs(got - ref).max() / scale


@pi_missing
def test_tables_structure(pi):
    """Rows sorted by element id, every (node, element) incidence present once, the neighbours symmetric with the
    coasts at -1 and ascending edge order, kappa symmetric in A_e kappa (FESOM_F's sum_e A_e V_e = 0 property)."""
    mesh = pi[0]
    t = ssh_se.build_tables(mesh)
    el, c = t["se_div_elem"], t["se_div_c"]
    live = (c != 0.0).any(axis=-1)
    en = np.asarray(mesh.elem_nodes)
    deg = np.bincount(en.ravel(), minlength=mesh.nod2D)
    np.testing.assert_array_equal(live.sum(axis=1), deg)
    for n in range(0, mesh.nod2D, 97):
        row = el[n][live[n]]
        assert np.all(np.diff(row) > 0)
        assert set(row.tolist()) == set(np.nonzero((en == n).any(axis=1))[0].tolist())
    nb = t["se_elem_nb"]
    for e_ in range(0, mesh.elem2D, 53):
        for s in range(3):
            if nb[e_, s] >= 0:
                assert e_ in nb[nb[e_, s]]
    A = np.asarray(mesh.elem_area)
    k = t["se_kappa"]
    for e_ in range(0, mesh.elem2D, 53):
        for s in range(3):
            j = nb[e_, s]
            if j >= 0:
                s2 = int(np.nonzero(nb[j] == e_)[0][0])
                assert abs(A[e_] * k[e_, s] - A[j] * k[j, s2]) <= 1e-12 * A[e_] * k[e_, s]


# --------------------------------------------------------------------------
# the SE block's algebra on synthetic inputs
# --------------------------------------------------------------------------
def _synthetic(mesh, st0, seed=0, amp_u=0.2, amp_eta=0.3):
    rng = np.random.default_rng(seed)
    lm = np.asarray(mesh.elem_layer_mask)[:, :, None]
    uv = jnp.asarray(np.where(lm, amp_u * rng.standard_normal(st0.uv.shape), 0.0))
    du = jnp.asarray(np.where(lm, 1e-3 * rng.standard_normal(st0.uv.shape), 0.0))
    eta = jnp.asarray(amp_eta * rng.standard_normal(mesh.nod2D))
    st = dataclasses.replace(st0, uv=uv, eta_n=eta, hbar=eta)
    st = ssh_se.cold_start(st, mesh)
    ub = st.se_ubt
    st = dataclasses.replace(
        st, se_ubt=ub * jnp.asarray(1.0 + 0.01 * rng.standard_normal(ub.shape)),
        se_eta=st.se_eta + jnp.asarray(1e-3 * rng.standard_normal(st.se_eta.shape)),
        se_sbar_old=st.se_sbar_old * 0.98)
    wf = jnp.asarray(1e-6 * rng.standard_normal(mesh.nod2D))
    return st, du, wf


@pi_missing
@pytest.mark.parametrize("cfg", [SeConfig(M=30), SeConfig(M=30, bt_drag=True), SeConfig(M=30, visc="linear"),
                                 SeConfig(M=30, visc="off")])
def test_se_block_compatibility_and_trim(pi, cfg):
    """eta^{n+1} - eta^n = tau (T(<<U>>) - W) and sum_k h^n_k u^{n+1}_k = <<U>>, to round-off, on random inputs
    (the Kokkos G1a / G1b invariants), for every switch."""
    mesh, _op, _s, st0 = pi
    st, du, wf = _synthetic(mesh, st0)
    out = jax.jit(lambda s, d, w: ssh_se.se_block(mesh, s, d, w, cfg, dt=DT, is_first_step=False))(st, du, wf)
    T = ssh_se.divergence(out.umean, mesh)
    r_eta = np.asarray(out.eta_n - st.eta_n - DT * (T - wf))
    d_eta = float(np.abs(np.asarray(out.eta_n - st.eta_n)).max())
    U = ssh_se.transport(out.uv, st.helem, mesh)
    r_trim = np.asarray(U - out.umean)
    scale_u = float(np.abs(np.asarray(out.umean)).max())
    print(f"\n[se-block {cfg.visc} drag={cfg.bt_drag}] |eta^n+1 - eta^n| {d_eta:.3e}  eta-compat {np.abs(r_eta).max():.3e} m"
          f"  trim {np.abs(r_trim).max():.3e} m2/s (|<<U>>| {scale_u:.3e})")
    assert d_eta > 1e-4
    assert np.abs(r_eta).max() <= 1e-12 * max(d_eta, 1.0)
    assert np.abs(r_trim).max() <= 1e-13 * scale_u
    assert np.all(np.isfinite(np.asarray(out.se_ubt))) and np.all(np.isfinite(np.asarray(out.se_eta)))


@pi_missing
def test_se_unroll_round_off(pi, monkeypatch):
    """``gpu.se_unroll`` (``FESOM_SE_UNROLL``): 10 substeps per loop trip move the block's outputs by round-off only."""
    mesh, _op, _s, st0 = pi
    st, du, wf = _synthetic(mesh, st0)
    cfg = SeConfig(M=30)
    out = {}
    for u in ("1", "10"):
        monkeypatch.setenv("FESOM_SE_UNROLL", u)
        assert ssh_se.se_unroll() == int(u)
        out[u] = jax.jit(lambda s, d, w: ssh_se.se_block(mesh, s, d, w, cfg, dt=DT, is_first_step=False))(st, du, wf)
    for name in ("eta_n", "uv", "umean", "se_eta", "se_ubt"):
        a, b = np.asarray(getattr(out["1"], name)), np.asarray(getattr(out["10"], name))
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * max(float(np.abs(a).max()), 1e-30), err_msg=name)
    monkeypatch.setenv("FESOM_SE_UNROLL", "0")
    with pytest.raises(ValueError, match="FESOM_SE_UNROLL"):
        ssh_se.se_unroll()


@pi_missing
def test_rest_stays_at_rest(pi):
    """No forcing, no flow, eta = 0: every SE output is exactly zero (no spurious source in the r-term)."""
    mesh, _op, _s, st0 = pi
    du = jnp.zeros_like(st0.uv)
    out = ssh_se.se_block(mesh, st0, du, None, SeConfig(M=10), dt=DT, is_first_step=True)
    for name in ("uv", "eta_n", "se_eta", "se_ubt", "umean"):
        assert float(jnp.abs(getattr(out, name)).max()) == 0.0, name


# --------------------------------------------------------------------------
# invariants over 100 pi steps (the full step, wind forcing, zstar)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pi_run(pi):
    mesh, op, stress, st0 = pi
    ale = AleConfig(split_explicit=SeConfig(M=20))
    states = _run(mesh, op, stress, st0, 100, ale)
    return states


@pi_missing
def test_invariants_100_pi_steps(pi, pi_run, capsys):
    """Over 100 wind-driven pi steps (no freshwater on the pi path): per step eta^{n+1} - eta^n = tau T(sum_k h^n_k
    u^{n+1}_k) (eta compatibility through the trim), the cumulative layer sum sum_k hnode - sum_k hnode^0 = eta, and
    the volume integral of eta constant."""
    mesh, op, stress, st0 = pi
    nocav = np.asarray(mesh.ulevels_nod2D) == 1
    nlm = np.asarray(mesh.node_layer_mask)
    av = np.asarray(mesh.areasvol[:, 0])
    sumh0 = np.where(nlm, np.asarray(st0.hnode), 0.0).sum(axis=1)
    vol0 = float((np.asarray(st0.eta_n) * av).sum())
    prev = st0
    r_c = r_l = 0.0
    for st in pi_run:
        T = np.asarray(ssh_se.divergence(ssh_se.transport(st.uv, prev.helem, mesh), mesh))
        de = np.asarray(st.eta_n) - np.asarray(prev.eta_n)
        r_c = max(r_c, float(np.abs((de - DT * T)[nocav]).max()))
        sumh = np.where(nlm, np.asarray(st.hnode), 0.0).sum(axis=1)
        r_l = max(r_l, float(np.abs((sumh - sumh0 - np.asarray(st.eta_n))[nocav]).max()))
        prev = st
    last = pi_run[-1]
    eta_max = float(np.abs(np.asarray(last.eta_n)).max())
    vol = float((np.asarray(last.eta_n) * av).sum())
    r_v = abs(vol - vol0) / (av.sum() * eta_max)
    with capsys.disabled():
        print(f"\n[se-invariants 100 pi steps] max|eta| {eta_max:.3e} m  max|uv| "
              f"{float(np.abs(np.asarray(last.uv)).max()):.3e} m/s  eta-compat {r_c:.3e} m  layer-sum {r_l:.3e} m  "
              f"volume drift {abs(vol - vol0):.3e} m3 (rel {r_v:.2e})")
    assert np.all(np.isfinite(np.asarray(last.T))) and eta_max > 1e-3
    assert r_c < 1e-14
    # the layer sum accumulates the per-step rounding of a ~4000 m column (1e-16 relative per step): measured
    # 3.7e-12 m after 100 steps (job 28310231); the Kokkos port's G1c gave 1.5e-11 m after 1000 CORE2 steps
    assert r_l < 1e-11
    assert r_v < 1e-13


@pi_missing
def test_se_differs_from_cg_but_stays_close(pi, pi_run):
    """SE is not the CG (a different free-surface time stepping) but on a calm pi run both give the same eta to a
    few per cent of its range after 100 steps."""
    mesh, op, stress, st0 = pi
    cg = _run(mesh, op, stress, st0, 100, AleConfig())[-1]
    se = pi_run[-1]
    a, b = np.asarray(cg.eta_n), np.asarray(se.eta_n)
    rel = np.abs(a - b).max() / np.abs(a).max()
    print(f"\n[se-vs-cg] eta max|SE-CG|/max|CG| after 100 pi steps = {rel:.3e}")
    assert 0.0 < rel < 0.1


@pi_missing
def test_cg_path_unchanged_by_se_seam(pi):
    """AleConfig(split_explicit=None) and a CG State with no rings trace the same program as before the seam: the
    step's jaxpr with ale_cfg=AleConfig() contains no SE op and equals the one of AleConfig(split_explicit=None)."""
    mesh, op, stress, st0 = pi
    cg = dataclasses.replace(st0, se_eta=None, se_ubt=None, se_sbar_old=None)
    bare = dataclasses.replace(mesh, se_div_elem=None, se_div_c=None, se_elem_nb=None, se_kappa=None)
    j1 = jax.make_jaxpr(lambda s: stepmod.step(s, bare, op, stress, dt=DT, is_first_step=True,
                                               ale_cfg=AleConfig()))(cg)
    j2 = jax.make_jaxpr(lambda s: stepmod.step(s, bare, op, stress, dt=DT, is_first_step=True,
                                               ale_cfg=AleConfig(split_explicit=None)))(cg)
    assert str(j1) == str(j2)


# --------------------------------------------------------------------------
# single precision (D5: the SE block in WP)
# --------------------------------------------------------------------------
@pi_missing
def test_se_block_single_precision(pi):
    """Under precision 'single' the SE block runs in float32 (its outputs float32) and agrees with the double run
    to float32 round-off amplified by the M substeps (rel 1e-4 of eta's change)."""
    from fesom_jax import precision as PR
    mesh, _op, _s, st0 = pi
    st, du, wf = _synthetic(mesh, st0, seed=3)
    cfg = SeConfig(M=30)
    ref = ssh_se.se_block(mesh, st, du, wf, cfg, dt=DT, is_first_step=False)
    m32, st32, du32, wf32 = PR.cast_tree((mesh, st, du, wf))
    with PR.scope("single"):
        out = jax.jit(lambda m, s, d, w: ssh_se.se_block(m, s, d, w, cfg, dt=DT, is_first_step=False))(
            m32, st32, du32, wf32)
        assert out.eta_n.dtype == jnp.float32 and out.uv.dtype == jnp.float32
        assert out.se_ubt.dtype == jnp.float32
    de = np.abs(np.asarray(ref.eta_n - st.eta_n)).max()
    rel = np.abs(np.asarray(out.eta_n, np.float64) - np.asarray(ref.eta_n)).max() / de
    print(f"\n[se-sp] float32 vs float64 eta: max diff / max change = {rel:.2e}")
    assert rel < 1e-2


# --------------------------------------------------------------------------
# gradients (as fesom3-jax's test_gradients_se_step / test_gradient_se_cold_start)
# --------------------------------------------------------------------------
@pi_missing
def test_se_block_dot_test(pi):
    """<v, J u> = <J^T v, u> for the SE block's linearisation in (uv, du, eta) (jvp vs vjp), rel 1e-12."""
    mesh, _op, _s, st0 = pi
    st, du, wf = _synthetic(mesh, st0, seed=5)
    cfg = SeConfig(M=20, bt_drag=True)

    def f(x):
        uv, d, eta = x
        s = dataclasses.replace(st, uv=uv, eta_n=eta)
        o = ssh_se.se_block(mesh, s, d, wf, cfg, dt=DT, is_first_step=False)
        return o.eta_n, o.uv
    x0 = (st.uv, du, st.eta_n)
    rng = np.random.default_rng(7)
    lm = np.asarray(mesh.elem_layer_mask)[:, :, None]
    u = (jnp.asarray(np.where(lm, rng.standard_normal(st.uv.shape), 0.0)),
         jnp.asarray(np.where(lm, 1e-3 * rng.standard_normal(st.uv.shape), 0.0)),
         jnp.asarray(1e-2 * rng.standard_normal(mesh.nod2D)))
    v = (jnp.asarray(rng.standard_normal(mesh.nod2D)), jnp.asarray(np.where(lm, rng.standard_normal(st.uv.shape),
                                                                             0.0)))
    _, Ju = jax.jit(lambda x, t: jax.jvp(f, (x,), (t,)))(x0, u)
    _, vjp = jax.vjp(f, x0)
    (JTv,) = vjp(v)
    lhs = sum(float(jnp.vdot(a, b)) for a, b in zip(v, Ju))
    rhs = sum(float(jnp.vdot(a, b)) for a, b in zip(JTv, u))
    print(f"\n[se-dot] <v,Ju> = {lhs:.15e}  <J^T v,u> = {rhs:.15e}")
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


@pi_missing
def test_se_step_fd_probe(pi):
    """d J / d(wind scale) through 3 SE steps (from rest) against a central difference, rel 1e-6;
    J = sum eta^2 + sum uv^2 at the end."""
    mesh, op, stress, st0 = pi
    ale = AleConfig(split_explicit=SeConfig(M=20))

    def J(a):
        st = st0
        for i in range(3):
            st = stepmod.step(st, mesh, op, a * stress, dt=DT, is_first_step=(i == 0), ale_cfg=ale)
        return jnp.sum(st.eta_n ** 2) + jnp.sum(st.uv ** 2)
    g = float(jax.jit(jax.grad(J))(1.0))
    h = 1e-4
    Jf = jax.jit(J)
    fd = (float(Jf(1.0 + h)) - float(Jf(1.0 - h))) / (2 * h)
    print(f"\n[se-fd] grad {g:.10e} fd {fd:.10e} rel {abs(g - fd) / abs(fd):.2e}")
    assert np.isfinite(g) and abs(g - fd) <= 1e-6 * abs(fd)


@pi_missing
def test_se_cold_start_gradient_finite(pi):
    """At the cold start (u = 0: the viscosity's |dU| and the drag's |u_b| at their kinks) the gradient of a scalar
    of one SE step with respect to the IC temperature and the wind is finite (no NaN from sqrt at 0)."""
    mesh, op, stress, st0 = pi
    ale = AleConfig(split_explicit=SeConfig(M=10, bt_drag=True))

    def J(T, s):
        st = dataclasses.replace(st0, T=T)
        out = stepmod.step(st, mesh, op, s, dt=DT, is_first_step=True, ale_cfg=ale)
        return jnp.sum(out.eta_n ** 2) + jnp.sum(out.uv ** 2)
    gT, gs = jax.jit(jax.grad(J, argnums=(0, 1)))(st0.T, stress)
    assert np.all(np.isfinite(np.asarray(gT))) and np.all(np.isfinite(np.asarray(gs)))
    assert float(jnp.abs(gs).max()) > 0.0


# --------------------------------------------------------------------------
# Task 3: restart of the rings, the CG/SE guards
# --------------------------------------------------------------------------
def test_core2_gpu_fast_config_runs_split_explicit():
    """``configs/core2_gpu_fast.yaml`` (user 2026-10-22): SP, ice every second step, split-explicit chi 0.1 + drag + M
    60 with the wide halo and the unroll, the padded halo transport without the compact all_gather, chunks of 96."""
    from fesom_jax.run_config import RunConfig, load_yaml
    from pathlib import Path
    cfg = load_yaml(Path(__file__).resolve().parents[2] / "configs" / "core2_gpu_fast.yaml")
    cfg.validate()
    assert cfg.split_explicit() == SeConfig(M=60, chi=0.1, bt_drag=True)
    assert cfg.precision == "single" and cfg.ice.ice_ave_steps == 2
    assert cfg.halo_transport() == "padded" and cfg.chunk_steps() == 96
    env = cfg.gpu_env()
    assert env["FESOM_SE_WIDE"] == "10" and env["FESOM_SE_UNROLL"] == "10" and env["FESOM_CHUNK_TAIL"] == "32"
    assert "FESOM_HALO_COMPACT" not in env


def test_run_config_split_explicit_seam():
    """``ssh: {split_explicit: {...}}`` reaches the step as AleConfig.split_explicit; absent ⇒ the identical ale
    object; zstar required; unknown keys, gathered/cheb with SE refused; YAML round trip."""
    from fesom_jax.run_config import RunConfig
    cfg = RunConfig.from_dict({"ale": {}, "ssh": {"split_explicit": {"M": 40, "chi": 0.2, "bt_drag": True}}})
    cfg.validate()
    se = cfg.physics_kwargs()["ale_cfg"].split_explicit
    assert se == SeConfig(M=40, chi=0.2, bt_drag=True)
    assert cfg.ale.split_explicit is None                       # the ale block itself is unchanged
    back = RunConfig.from_dict(cfg.to_dict())
    assert back.physics_kwargs()["ale_cfg"] == cfg.physics_kwargs()["ale_cfg"]
    cg = RunConfig.from_dict({"ale": {}})
    assert cg.physics_kwargs()["ale_cfg"] is cg.ale
    assert RunConfig.from_dict({"ale": {}, "ssh": {"split_explicit": {}}}).split_explicit() == SeConfig()
    with pytest.raises(ValueError, match="zstar"):
        RunConfig.from_dict({"ssh": {"split_explicit": {}}}).validate()
    with pytest.raises(KeyError):
        RunConfig.from_dict({"ale": {}, "ssh": {"split_explicit": {"theta": 0.14}}}).validate()
    with pytest.raises(ValueError, match="replaces the CG"):
        RunConfig.from_dict({"ale": {}, "ssh": {"split_explicit": {}, "gathered": True}}).validate()
    with pytest.raises(ValueError):
        RunConfig.from_dict({"ale": {}, "ssh": {"split_explicit": {"M": 0}}}).validate()
    g = RunConfig.from_dict({"ale": {}, "ssh": {"split_explicit": {}}, "gpu": {"se_wide": 10, "se_unroll": 10}})
    g.validate()
    assert g.gpu_env() == {"FESOM_SE_WIDE": "10", "FESOM_SE_UNROLL": "10"}
    for key in ("se_wide", "se_unroll"):
        with pytest.raises(ValueError, match=f"gpu.{key} needs"):
            RunConfig.from_dict({"ale": {}, "gpu": {key: 4}}).validate()


@pi_missing
def test_init_state_guards(pi):
    """The driver's guard: SE rings kept (exact resume); a CG state into SE gets flat rings (announced for a
    restart, bitwise ssh_se.cold_start); an SE state into the CG is refused."""
    mesh, op, stress, st0 = pi
    lm = mesh.elem_layer_mask
    se_state = _run(mesh, op, stress, st0, 2, AleConfig(split_explicit=SeConfig(M=10)))[-1]
    assert ssh_se.init_state(se_state, lm, SeConfig(), restart=True) is se_state
    cg = dataclasses.replace(se_state, se_eta=None, se_ubt=None, se_sbar_old=None)
    msgs = []
    flat = ssh_se.init_state(cg, lm, SeConfig(), restart=True, log=msgs.append)
    assert msgs and "flat rings" in msgs[0]
    ref = ssh_se.cold_start(cg, mesh)
    for f in ssh_se.STATE_FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(flat, f)), np.asarray(getattr(ref, f)), err_msg=f)
    assert float(jnp.abs(flat.se_ubt).max()) > 0.0
    msgs = []
    ssh_se.init_state(cg, lm, SeConfig(), restart=False, log=msgs.append)
    assert not msgs                                             # a cold start is not announced
    assert ssh_se.init_state(cg, lm, None, restart=True) is cg
    with pytest.raises(ValueError, match="split-explicit run"):
        ssh_se.init_state(se_state, lm, None, restart=True)
    # host numpy [P, L] layout (the driver's cold start) gives the same rings
    hp = ssh_se.flat_rings(np.asarray(cg.eta_n)[None], np.asarray(cg.uv)[None], np.asarray(cg.helem)[None],
                           np.asarray(lm)[None])
    for f, a in zip(ssh_se.STATE_FIELDS, hp):
        assert isinstance(a, np.ndarray)
        np.testing.assert_array_equal(a[0], np.asarray(getattr(ref, f)), err_msg=f)


@pi_missing
def test_rings_restart_roundtrip(pi, tmp_path):
    """The rings go through partition_state -> fold -> device -> write_restart -> read_restart (onto another device
    count) bitwise; a CG restart (no rings) reads them as None."""
    import zarr
    from fesom_jax import halo, partit, shard_mesh
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import zarr_output as zo
    mesh, op, stress, st0 = pi
    g0 = _run(mesh, op, stress, st0, 2, AleConfig(split_explicit=SeConfig(M=10)))[-1]
    npes = min(2, len(jax.devices()))
    part = (partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D) if npes == 1 else
            partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, npes))
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    fs, spec = ish.folded_state(shard_mesh.partition_state(g0, part))
    placed = ish._to_global_sharded(fs, spec, halo.device_mesh("p", devices=jax.devices()[:npes]))
    d = tmp_path / "r_se"
    zo.write_restart(d, placed, sm, part, step=2, calendar_date="1958-01-01", dt_stage=DT)
    assert set(ssh_se.STATE_FIELDS) <= set(zarr.open_group(str(d), mode="r").array_keys())
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    loaded, _ = zo.read_restart(d, mesh, ser)
    for f in ssh_se.STATE_FIELDS + ("eta_n", "hbar", "uv"):
        n0 = np.asarray(getattr(g0, f)).shape[0]
        np.testing.assert_array_equal(np.asarray(getattr(loaded, f))[:n0], np.asarray(getattr(g0, f)), err_msg=f)
    cg = dataclasses.replace(g0, se_eta=None, se_ubt=None, se_sbar_old=None)
    fs1, spec1 = ish.folded_state(shard_mesh.partition_state(cg, ser))
    placed1 = ish._to_global_sharded(fs1, spec1, halo.device_mesh("p", devices=jax.devices()[:1]))
    d1 = tmp_path / "r_cg"
    zo.write_restart(d1, placed1, shard_mesh.build_sharded_mesh(mesh, ser), ser, step=2,
                     calendar_date="1958-01-01", dt_stage=DT)
    l1, _ = zo.read_restart(d1, mesh, ser)
    assert all(getattr(l1, f) is None for f in ssh_se.STATE_FIELDS)


IC_DIR = ROOT / "data" / "ic_core2"


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


have_forcing = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists() or not CORE2_MESH.is_dir() or not _have_jra(),
    reason="CORE2 PHC IC / mesh / JRA55 forcing missing (compute node only)")


@have_forcing
def test_core2_se_resume_bitwise(tmp_path):
    """run_from_config with ssh.split_explicit on CORE2 (KPP + mEVP + zstar, dt 1800): 26 steps in chunks of 13 in
    one invocation == 13 steps -> restart (with the rings) -> 13 steps, BITWISE on every owned lane of every State
    field (the chunk boundary is the same bootstrap call in both, so the only difference is the disk round trip).
    Then the guards through the driver: the SE restart into a CG config raises; a CG restart into SE runs and says
    so."""
    import zarr
    from fesom_jax import partit, shard_mesh, surface_forcing
    from fesom_jax.integrate_sharded import unfold_state
    from fesom_jax.ice import IceConfig
    from fesom_jax.kpp import KppConfig
    from fesom_jax.phc_ic import cold_start_state
    from fesom_jax.run import run_from_config
    from fesom_jax.run_config import RunConfig
    from fesom_jax.state import State
    mesh = load_mesh(CORE2_MESH)
    state = cold_start_state(mesh, IC_DIR)
    dt, n, m = 1800.0, 26, 13
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(state.T[:, 0]))
    stack = cf.stack(surface_forcing.dates_for_steps(1958, dt, n))
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=dt), part)
    common = dict(mesh=mesh, part=part, sm=sm, sop=sop, year=1958, forcing=cf)
    kw = dict(kpp=KppConfig(), ice=IceConfig(whichEVP=1), ale=AleConfig(), dt=dt,
              ssh={"split_explicit": {"M": 50}})
    cont = run_from_config(RunConfig(n_steps=n, **kw), state0=state, forcing_stack=stack, chunk_steps=m, **common)
    r1 = tmp_path / "r_se13"
    run_from_config(RunConfig(n_steps=m, restart_out=str(r1), **kw), state0=state,
                    forcing_stack=jax.tree.map(lambda x: x[:m], stack), chunk_steps=m, **common)
    assert set(ssh_se.STATE_FIELDS) <= set(zarr.open_group(str(r1), mode="r").array_keys())
    rest = jax.tree.map(lambda x: x[m:], stack)
    chained = run_from_config(RunConfig(n_steps=n - m, restart_in=str(r1), **kw), forcing_stack=rest,
                              chunk_steps=m, **common)
    assert chained.step == n
    a, b = unfold_state(cont.state_p, 1), unfold_state(chained.state_p, 1)
    checked = 0
    for fld in dataclasses.fields(State):
        A = getattr(a, fld.name)
        if A is None:
            assert getattr(b, fld.name) is None, fld.name
            continue
        A, B = np.asarray(A), np.asarray(getattr(b, fld.name))
        md = int((part.myDim_nod2D if A.shape[1] == sm.Lmax["nod"] else part.myDim_elem2D)[0])
        np.testing.assert_array_equal(A[0, :md], B[0, :md], err_msg=fld.name)
        checked += 1
    eta = np.asarray(a.eta_n)[0]
    print(f"\n[se-resume] 26 = 13 + 13 bitwise over {checked} fields; max|eta| {np.abs(eta).max():.3f} m")
    assert np.isfinite(eta).all() and np.abs(eta).max() > 0.05
    # an SE restart into the CG: refused
    kw_cg = dict(kw, ssh=None)
    with pytest.raises(ValueError, match="split-explicit run"):
        run_from_config(RunConfig(n_steps=1, restart_in=str(r1), **kw_cg), forcing_stack=rest, **common)
    # a CG restart into SE: flat rings, announced, runs
    r2 = tmp_path / "r_cg2"
    run_from_config(RunConfig(n_steps=2, restart_out=str(r2), **kw_cg), state0=state,
                    forcing_stack=jax.tree.map(lambda x: x[:2], stack), **common)
    out = run_from_config(RunConfig(n_steps=2, restart_in=str(r2), **kw),
                          forcing_stack=jax.tree.map(lambda x: x[2:4], stack), **common)
    assert np.isfinite(np.asarray(unfold_state(out.state_p, 1).eta_n)).all()


# --------------------------------------------------------------------------
# Task 4: the CFL guard
# --------------------------------------------------------------------------
def test_cfl_acrit_values():
    """a_crit(chi) of the AB3-AM4 substeps: 1.853 / 1.579 / 1.329 at chi = 0 / 0.1 / 0.25 (FESOM_F's se_cfl_acrit,
    fesom3-jax's cfl_acrit docstring); stable just below, unstable just above."""
    for chi, want in ((1e-12, 1.853), (0.1, 1.579), (0.25, 1.329)):
        a = ssh_se.cfl_acrit(chi)
        assert abs(a - want) < 1.5e-3, (chi, a)
        assert ssh_se.cfl_stable(a * 0.999, chi) and not ssh_se.cfl_stable(a * 1.001, chi)


@pi_missing
def test_cfl_guard_raises_below_m_min(pi):
    mesh = pi[0]
    m_min, omega, _a, line = ssh_se.cfl_guard(mesh, DT, SeConfig(M=20))
    assert m_min == 1 and omega > 0.0 and "M_min = 1" in line
    with pytest.raises(ValueError, match="M_min"):
        ssh_se.cfl_guard(mesh, 1.0e5, SeConfig(M=20))
    m_big, *_ = ssh_se.cfl_guard(mesh, 1.0e5, SeConfig(M=20, m_force=True))
    assert m_big > 20


@pytest.mark.skipif(not CORE2_MESH.is_dir(), reason="CORE2 mesh export missing (compute-node test)")
def test_cfl_guard_core2(capsys):
    """CORE2 (the 2026-05-28 levels): omega_max of the discrete operator and M_min at dt 1800 / 2700 for chi 0.1 /
    0.2. FESOM_F's M_min 40 at dt 1800, chi 0.1 is for the FESOM3 mesh (its omega_max 3.15e-2); ours is recorded
    here. The Kokkos port's resolution estimate (0.5 res / sqrt(gH), M_min 35) under-predicts: M_min > 35."""
    mesh = load_mesh(CORE2_MESH)
    omega, rel = ssh_se.cfl_omega_max(mesh)
    rows = []
    for dt in (1800.0, 2700.0):
        for chi in (0.1, 0.2):
            m_min, om, ac, line = ssh_se.cfl_guard(mesh, dt, SeConfig(chi=chi, m_force=True))
            rows.append((dt, chi, m_min, ac))
            with capsys.disabled():
                print("\n[cfl-core2] " + line)
    with capsys.disabled():
        print(f"\n[cfl-core2] omega_max = {omega:.4e} 1/s (power-iteration last change {rel:.1e})")
    m1800 = rows[0][2]
    assert 35 < m1800 < 120
    assert all(r[2] >= m1800 for r in rows[1:])


# --------------------------------------------------------------------------
# Task 1: the oracle gate against the frozen Kokkos se0 binary (CORE2, the z2_cdump configuration)
# --------------------------------------------------------------------------
SE_ORACLE = Path("/work/ab0995/a270088/fesom_jax_dev/runs/se_oracle")
Z2_CDUMP = Path("/work/ab0995/a270088/port/zstar/z2_cdump/dump")
_ORACLE_RUNS = ("off3", "se3")
oracle_missing = pytest.mark.skipif(
    not all((SE_ORACLE / r / "dump").is_dir() for r in _ORACLE_RUNS),
    reason="se0 oracle dumps absent: runs/se_oracle/{off3,se3}/dump (fesom_port_serial_se0 sha df959e90..., CORE2 "
           "private mesh 2026-05-28 levels, 16 ranks dist_16, dt 1800, PHC winter, JRA55 1958, default physics "
           "KPP+GM+EVP, FESOM_ALE=zstar, FESOM_ALE_DUMP_STEPS=3; off3 = FESOM_SSH_MODE=si, se3 = se, M 50, chi 0.1, "
           "visc 1 (10/2750); see runs/se_oracle/README.md)")


@oracle_missing
def test_se0_ssh_off_matches_the_c_oracle(capsys):
    """The se0 binary with SE off reproduces the C oracle fesom-jax gates against (z2_cdump) at every dumped tag of
    step 1 — the precondition for using its SE-on run as a fesom-jax oracle (plan Task 1). Steps 2-3 are printed only:
    z2_cdump was dumped at the C's Phase Z2, before its Z3-Z6 zstar steps (its job says so), and se0 carries them."""
    from fesom_jax import io_dump
    worst = {}
    for step in (1, 2, 3):
        a, _ = io_dump.load_ale_dump(SE_ORACLE / "off3" / "dump", None, step=step, n_nod=126858, n_elem=244659)
        b, _ = io_dump.load_ale_dump(Z2_CDUMP, list(a), step=step, n_nod=126858, n_elem=244659)
        for tag in a:
            x, y = a[tag], b[tag]
            fin = np.isfinite(x) & np.isfinite(y)
            d = float(np.abs(x - y)[fin].max()) if fin.any() else 0.0
            worst[(step, tag)] = d
    with capsys.disabled():
        for k, d in sorted(worst.items()):
            print(f"[se0-off vs z2_cdump] step {k[0]} {k[1]:10s} max|diff| = {d:.3e}")
    assert all(d == 0.0 for (step, _), d in worst.items() if step == 1), "se0 SE-off is not bitwise the C oracle"


@oracle_missing
def test_se_core2_steps123_vs_se0(capsys):
    """fesom-jax SE (M 50, chi 0.1, two-term viscosity, no drag) on the z2_cdump configuration against the se0 SE-on
    run: eta (hbar tag) at steps 1-3. The predictor matches the C to its forcing round-off (the zstar JZ.7 gate) and
    the substeps are explicit (no CG tolerance), so the comparison is near-bitwise: printed, and asserted at the
    class the JZ.7 gate sees for its explicit fields."""
    from fesom_jax import io_dump, surface_forcing, ice
    from fesom_jax.gm import GMConfig
    from fesom_jax.ice import IceConfig
    from fesom_jax.kpp import KppConfig
    from fesom_jax.phc_ic import phc_initial_state
    ic_dir = ROOT / "data" / "ic_core2_dist16"
    mesh = ssh_se.with_tables(load_mesh(CORE2_MESH))
    state = phc_initial_state(mesh, ic_dir)
    sst = np.asarray(state.T[:, 0])
    st = ssh_se.cold_start(ice.seed_ice(state, mesh, sst), mesh)
    op = ssh.build_ssh_operator(mesh, dt=1800.0)
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=sst)
    dates = surface_forcing.dates_for_steps(1958, 1800.0, 3)
    f = jax.jit(stepmod.step, static_argnames=("dt", "is_first_step", "kpp_cfg", "gm_cfg", "ice_cfg", "ale_cfg",
                                               "bulk_uvnode_lag"))
    zstar = AleConfig()
    rels, incs, d_se, d_cg = [], [], [], []
    st_cg = st
    for i in range(3):
        sf = cf.step_forcing(*dates[i])
        kw = dict(dt=1800.0, is_first_step=(i == 0), step_forcing=sf, forcing_static=cf.static, kpp_cfg=KppConfig(),
                  gm_cfg=GMConfig(), ice_cfg=IceConfig(scale_area=2.0e8), bulk_uvnode_lag=True)
        st = f(st, mesh, op, None, ale_cfg=AleConfig(split_explicit=SeConfig()), **kw)
        st_cg = f(st_cg, mesh, op, None, ale_cfg=zstar, **kw)
        c_se = io_dump.ale_component(io_dump.load_ale_dump(SE_ORACLE / "se3" / "dump", ["hbar"], step=i + 1,
                                                           n_nod=126858)[0], "hbar", "hbar")
        c_si = io_dump.ale_component(io_dump.load_ale_dump(SE_ORACLE / "off3" / "dump", ["hbar"], step=i + 1,
                                                           n_nod=126858)[0], "hbar", "hbar")
        j_se, j_cg = np.asarray(st.hbar), np.asarray(st_cg.hbar)
        d = np.abs(j_se - c_se)
        rels.append(d.max() / np.abs(c_se).max())
        # the SE increment over each code's own semi-implicit step: the chained steps 2-3 carry the CG path's
        # known ~mm divergence from the C (the SSH machinery amplifying round-off, test_ale_zstar JZ.7)
        inc_c = c_se - c_si
        incs.append(float(np.abs((j_se - j_cg) - inc_c).max() / np.abs(inc_c).max()))
        d_se.append(float(d.max()))
        d_cg.append(float(np.abs(j_cg - c_si).max()))
        with capsys.disabled():
            print(f"\n[se-vs-se0] step {i + 1}: max|eta| {np.abs(c_se).max():.4e}  SE max|diff| {d.max():.3e} "
                  f"(rel {rels[-1]:.3e})  CG vs SI max|diff| {np.abs(j_cg - c_si).max():.3e}  "
                  f"max|SE-SI| {np.abs(inc_c).max():.3e}  increment rel {incs[-1]:.3e}")
    assert rels[0] < 1e-6, f"step-1 eta differs from se0 at rel {rels[0]:.2e}"
    # steps 2-3: SE adds nothing to the chained CG-vs-C divergence (2026-10-22: 6.3 / 6.8 mm against 7.6 / 7.7 mm,
    # the SE - SI increment matched to 0.6-0.7 %, job 28333516)
    for i in (1, 2):
        assert d_se[i] <= 1.5 * d_cg[i], (i + 1, d_se[i], d_cg[i])
        assert incs[i] < 0.02, (i + 1, incs[i])
