"""Single precision (``precision: single``, FESOM2 #1207 policy; docs/plans/20261020-core2-single-precision.md).

CPU tests on the packaged pi mesh and synthetic columns: the precision helpers, the host casts, the config switch, the
upstream float64 sites (CG rule, SP preconditioner, point-slope forcing, the rounding carry) and a production-physics
single-precision step whose every float64 use is explicit (no implicit float32/float64 promotion anywhere).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import precision as PR

ROOT = Path(__file__).resolve().parents[2]


# ---- helpers ------------------------------------------------------------------------------------------------------
def test_helpers_are_identity_in_double():
    x = jnp.arange(3.0)
    assert not PR.sp()
    assert PR.wp(x) is x and PR.full(x) is x
    assert PR.wp_tree({"a": x})["a"] is x and PR.full_tree({"a": x})["a"] is x
    assert PR.lit(0.1) == 0.1
    assert PR.compiler_options("double") is None
    f = PR.jit(lambda y: y + 1.0, "double")
    assert PR.scoped(f, "double") is f


def test_helpers_in_single_scope():
    with PR.scope("single"):
        assert PR.sp()
        x = jnp.zeros(3)                          # dtype-less ⇒ WP
        assert x.dtype == jnp.float32
        assert PR.full(x).dtype == jnp.float64
        assert PR.wp(PR.full(x)).dtype == jnp.float32
        assert PR.wp_dtype() == jnp.float32
        assert PR.lit(0.1) == float(np.float32(0.1))
    assert not PR.sp()
    assert PR.compiler_options("single") == {"xla_allow_excess_precision": False}
    with pytest.raises(ValueError):
        PR.check("half")


def test_cast_state_policy():
    from fesom_jax.mesh import load_mesh
    from fesom_jax.state import State
    mesh = load_mesh()
    st = State.zeros(mesh, xp=np)
    sp = PR.cast_state(st)
    for f in dataclasses.fields(sp):
        a = getattr(sp, f.name)
        if a is None:                             # the ICE_AVG_FIELDS of an ice_ave_steps 1 State
            continue
        want = np.float64 if f.name in PR.STATE_F64 else np.float32
        assert np.asarray(a).dtype == want, f.name
    assert sp.carry_T.shape == st.T.shape and not np.any(sp.carry_T)
    assert PR.drop_carry(sp).carry_T is None
    from fesom_jax import ice_step
    from fesom_jax.ice import IceConfig
    from fesom_jax.state import ICE_AVG_FIELDS
    sp2 = PR.cast_state(ice_step.init_ice_avg(st, IceConfig(ice_ave_steps=2), xp=np))
    for name in ICE_AVG_FIELDS:                   # ice_ave_steps 2: the counter and the flag stay integer
        want = np.int32 if name in ("ice_steps_since_upd", "ice_update") else np.float32
        assert np.asarray(getattr(sp2, name)).dtype == want, name
    tree = PR.cast_tree({"a": np.ones(2), "i": np.arange(2), "f": 0.1})
    assert tree["a"].dtype == np.float32 and tree["i"].dtype.kind == "i"
    assert PR.cast_tree(tree, keep64=True)["a"].dtype == np.float64


# ---- config -------------------------------------------------------------------------------------------------------
def test_run_config_precision():
    from fesom_jax.run_config import load_yaml
    for name in ("core2_full.yaml", "core2_gpu.yaml"):
        assert load_yaml(ROOT / "configs" / name).precision == "double"
    cfg = load_yaml(ROOT / "configs" / "core2_gpu.yaml")
    sp = dataclasses.replace(cfg, precision="single")
    sp.validate()
    assert type(cfg).from_dict(sp.to_dict()).precision == "single"
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, precision="float").validate()
    with pytest.raises(NotImplementedError):
        dataclasses.replace(cfg, precision="single", ale=None).validate()       # linfs
    with pytest.raises(NotImplementedError):
        dataclasses.replace(cfg, precision="single", ice=cfg.ice._replace(whichEVP=0)).validate()


def test_core2_adjoint_refuses_single():
    import importlib.util
    spec = importlib.util.spec_from_file_location("core2_adj_t", ROOT / "scripts" / "adjoint" / "core2_adj.py")
    A = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = A
    spec.loader.exec_module(A)
    cfg = A.load_yaml(A.CONFIG)
    A.assert_physics(cfg)                                   # the reference configuration passes
    with pytest.raises(AssertionError, match="precision"):
        A.assert_physics(dataclasses.replace(cfg, precision="single"))


# ---- upstream float64 sites ---------------------------------------------------------------------------------------
def test_reductions_widen_in_single():
    from fesom_jax import reductions
    v = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    m = np.ones(1000, bool)
    with PR.scope("single"):
        s = reductions.global_sum(jnp.asarray(v), jnp.asarray(m))
        d = reductions.global_dot(jnp.asarray(v), jnp.asarray(v), jnp.asarray(m))
    assert s.dtype == jnp.float32 and d.dtype == jnp.float64
    assert float(s) == float(np.float32(np.sum(v.astype(np.float64))))     # f64 sum, rounded once
    assert float(d) == pytest.approx(float(np.sum(v.astype(np.float64) ** 2)), rel=1e-15)


def test_precond_variant_1_is_symmetric():
    from fesom_jax import ssh
    from fesom_jax.mesh import load_mesh
    import scipy.sparse as sp_
    mesh = load_mesh()
    op0 = ssh.build_ssh_operator(mesh, dt=1800.0)
    op1 = ssh.build_ssh_operator(mesh, dt=1800.0, precond_variant=1)
    n = op1.n_nodes
    M1 = sp_.coo_matrix((np.asarray(op1.precond_vals), (np.asarray(op1.rows), np.asarray(op1.cols))),
                        shape=(n, n)).tocsr()
    assert abs(M1 - M1.T).max() <= 1e-12 * abs(M1).max()                    # textbook variant: symmetric
    np.testing.assert_array_equal(np.asarray(op0.stiff_vals), np.asarray(op1.stiff_vals))
    with pytest.raises(ValueError):
        ssh.build_ssh_operator(mesh, dt=1800.0, precond_variant=2)


def test_cg_single_follows_upstream_rule():
    """The SP CG on the pi system: vectors f32, dots/scalars f64, rows summed in f64 — the solution agrees with the
    double solve to the solver tolerance."""
    from fesom_jax import ssh
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh()
    op = ssh.build_ssh_operator(mesh, dt=1800.0, precond_variant=1)
    rhs = np.random.default_rng(1).standard_normal(op.n_nodes) * 1e3
    xd = ssh.solve_ssh(op, jnp.asarray(rhs))
    op32 = PR.cast_tree(op)
    with PR.scope("single"):
        xs = jax.jit(lambda b: ssh.solve_ssh(op32, b))(jnp.asarray(rhs, jnp.float32))
    assert xs.dtype == jnp.float32
    rel = float(jnp.linalg.norm(xs.astype(jnp.float64) - xd) / jnp.linalg.norm(xd))
    assert rel < 1e-3


def test_point_slope_matches_double_to_float32():
    from fesom_jax import surface_forcing
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh()
    cf = surface_forcing.build_surface_forcing(mesh, 1958, dt=1800.0)
    dates = surface_forcing.dates_for_steps(1958, 1800.0, 12)
    dp = [cf.step_forcing(*d, xp=np) for d in dates]
    cf.point_slope = True
    sp = [cf.step_forcing(*d, xp=np) for d in dates]
    for a, b in zip(dp, sp):
        for name in ("Tair", "shortwave", "longwave", "shum", "u_air"):
            x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
            scale = np.max(np.abs(x)) + 1e-30
            assert np.max(np.abs(x - y)) <= 1e-5 * scale, name
    # the on-device tables in point-slope form reproduce the host point-slope values
    cf2 = surface_forcing.build_surface_forcing(mesh, 1958, dt=1800.0)
    cf2.point_slope = True
    tab, sched = cf2.stack_tables(dates, nb=4)
    assert sched.elapsed is not None and tab.coef_a.dtype == np.float32
    const = surface_forcing.forcing_device_const(cf2)
    with PR.scope("single"):
        for i in (0, 5, 11):
            got = surface_forcing.combine_step_forcing(
                PR.cast_tree(tab), sched.rdate[i], sched.fidx[i], sched.midx[i], PR.cast_tree(const),
                elapsed=jnp.asarray(sched.elapsed[i]))
            np.testing.assert_allclose(np.asarray(got.Tair), np.asarray(sp[i].Tair), rtol=0, atol=2e-4)


# ---- the rounding carry ---------------------------------------------------------------------------------------------
def _columns(n=400, nl=48, seed=3):
    rng = np.random.default_rng(seed)
    T = (rng.uniform(-2, 30, (n, nl))).astype(np.float32)
    h = rng.uniform(5.0, 250.0, (n, nl))
    hn = h * rng.uniform(0.999, 1.001, (n, nl))
    d = rng.uniform(-1.0, 1.0, (n, nl)) * 1e-2 * h
    return T, h, hn, d


def test_ale_reconstruct_carry_conserves_content():
    from fesom_jax import tracer_adv
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh()
    n, nl = mesh.nod2D, mesh.nl
    T, h, hn, d = _columns(n, nl)
    m = np.asarray(mesh.node_layer_mask)
    h, hn = np.where(m, h, 0.0), np.where(m, hn, 0.0)
    carry = np.zeros((n, nl), np.float32)
    # placed outside the scope: a host float64 is narrowed to float32 when x64 is off
    args = (jnp.asarray(T), jnp.asarray(d, jnp.float64), jnp.asarray(h, jnp.float64),
            jnp.asarray(hn, jnp.float64), jnp.asarray(carry))
    with PR.scope("single"):
        Tn, cn = tracer_adv.ale_reconstruct_carry(PR.cast_tree(mesh), *args)
    Tn, cn = np.asarray(Tn, np.float64), np.asarray(cn, np.float64)
    before = np.where(m, T.astype(np.float64) * h + d, 0.0).sum()
    after = np.where(m, Tn * hn + cn, 0.0).sum()
    lost = np.where(m, Tn * hn, 0.0).sum()
    assert abs(after - before) <= 1e-13 * abs(before)
    assert abs(lost - before) > 100 * abs(after - before)          # negative control: without the carry
    # the upstream formula, column by column
    acc = T.astype(np.float64) * h + d
    want_T = np.where(m & (hn > 0), (acc / np.where(hn > 0, hn, 1.0)).astype(np.float32), T)
    np.testing.assert_array_equal(np.asarray(Tn, np.float32), want_T)


def test_impl_vert_diff_carry_conserves_content():
    from fesom_jax import tracer_diff
    from fesom_jax.mesh import load_mesh
    from fesom_jax import ale
    mesh = load_mesh()
    n, nl = mesh.nod2D, mesh.nl
    m = np.asarray(mesh.node_layer_mask)
    T, h, hn, _ = _columns(n, nl, seed=5)
    hn = np.where(m, np.asarray(mesh.areasvol) * 0 + hn, 0.0)
    Kv = np.where(np.asarray(mesh.node_iface_mask), 1e-2, 0.0).astype(np.float32)
    carry = np.zeros((n, nl), np.float32)
    mesh32 = PR.cast_tree(mesh)
    hn_d = jnp.asarray(hn, jnp.float64)                     # placed outside the scope (stays float64)
    with PR.scope("single"):
        Z = ale.live_geometry(mesh32, hn_d, keep_f64=True)[1]
        Tn, cn = tracer_diff.impl_vert_diff_one(mesh32, jnp.asarray(T), jnp.asarray(Kv), hn_d,
                                                dt=1800.0, Z3d=Z, carry=jnp.asarray(carry))
    av = np.asarray(mesh.areasvol)
    before = np.where(m, T.astype(np.float64) * hn * av, 0.0).sum()
    after = np.where(m, (np.asarray(Tn, np.float64) * hn + np.asarray(cn, np.float64)) * av, 0.0).sum()
    lost = np.where(m, np.asarray(Tn, np.float64) * hn * av, 0.0).sum()
    assert abs(after - before) <= 1e-12 * abs(before)
    assert abs(lost - before) > 10 * abs(after - before)


# ---- a production-physics step in single precision -----------------------------------------------------------------
@pytest.fixture(scope="module")
def pi_prod():
    from fesom_jax import ssh, surface_forcing
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import cold_start_state, PACKAGED_PI_IC_DIR
    mesh = load_mesh()
    s0 = cold_start_state(mesh, PACKAGED_PI_IC_DIR, xp=np)
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(s0.T[:, 0]), dt=1800.0)
    cf.point_slope = True
    sfs = [PR.cast_tree(cf.step_forcing(*d, xp=np)) for d in surface_forcing.dates_for_steps(1958, 1800.0, 2)]
    op = PR.cast_tree(ssh.build_ssh_operator(mesh, dt=1800.0, precond_variant=1))
    return mesh, s0, cf, sfs, op


def _sp_two_steps(pi_prod, check_mixing=False):
    from fesom_jax import step as stepmod
    from fesom_jax.ale import AleConfig
    from fesom_jax.gm import GMConfig
    from fesom_jax.ice import IceConfig
    from fesom_jax.params import Params
    from fesom_jax.tke import TkeConfig
    mesh, s0, cf, sfs, op = pi_prod
    m32, fs, p = PR.cast_tree(mesh), PR.cast_tree(cf.static), PR.cast_tree(Params.defaults())
    st = jax.device_put(PR.cast_state(s0))
    kw = dict(ale_cfg=AleConfig(), gm_cfg=GMConfig(), tke_cfg=TkeConfig(), ice_cfg=IceConfig(whichEVP=1))

    def go(st, sfs):
        for i, sf in enumerate(sfs):
            st = stepmod.step(st, m32, op, None, p, dt=1800.0, is_first_step=(i == 0), step_forcing=sf,
                              forcing_static=fs, **kw)
        return st

    mixed = []
    if check_mixing:
        from jax._src import dtypes as D
        orig = D.lattice_result_type
        F = {np.dtype("float32"), np.dtype("float64")}

        def chk(*args):
            fl = set()
            for a in args:
                dt = getattr(a, "dtype", None)
                weak = getattr(a, "weak_type", False) or getattr(getattr(a, "aval", None), "weak_type", False)
                if dt is not None and not weak and np.dtype(dt) in F:
                    fl.add(np.dtype(dt))
            if len(fl) == 2:
                import traceback
                fr = [f for f in traceback.extract_stack() if "/fesom_jax/" in f.filename
                      and "/tests/" not in f.filename and "precision.py" not in f.filename]
                mixed.append(f"{fr[-1].filename.split('/')[-1]}:{fr[-1].lineno}" if fr else "?")
            return orig(*args)
        D.lattice_result_type = chk
        try:
            with PR.scope("single"):
                jax.make_jaxpr(go)(st, sfs)
        finally:
            D.lattice_result_type = orig
        return mixed
    with PR.scope("single"):
        return PR.jit(go, "single")(st, sfs)


def test_single_step_has_no_implicit_promotion(pi_prod):
    """Every float64 use in the single-precision step is explicit (PR.full / the WP_full fields): no jnp operation
    mixes a float32 and a float64 operand implicitly (a silent promotion would either drop a rounding or move
    float64 bytes)."""
    mixed = _sp_two_steps(pi_prod, check_mixing=True)
    assert not mixed, sorted(set(mixed))


def test_single_step_dtypes_and_finite(pi_prod):
    out = _sp_two_steps(pi_prod)
    for f in dataclasses.fields(out):
        a = getattr(out, f.name)
        if a is None:                             # the ICE_AVG_FIELDS of an ice_ave_steps 1 State
            continue
        want = jnp.float64 if f.name in PR.STATE_F64 else jnp.float32
        assert a.dtype == want, f.name
        assert bool(jnp.all(jnp.isfinite(a))), f.name
