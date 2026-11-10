"""Sea-ice FCT advection gate — Phase 6, Task 6.5.

Verifies :mod:`fesom_jax.ice_adv` against the isolated C FCT dump
(``data/ice_full_dump_core2/ice_fct_dump_s1_rank0.txt``, written by the
``FESOM_ICE_FCT_DUMP_DIR`` hook at config-C full-ice). The dump records, per all-node at
step 1, the advecting velocity ``u_ice``/``v_ice`` + the 3 ice tracers BEFORE and AFTER
``fct_solve`` (before cut_off/thermo). JAX is fed the C inputs and matched output-for-output
— the element-based FCT (no CSR) reproduces the C to machine precision.

SKIPS cleanly if the mesh / FCT dump are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
FCTD = ROOT / "data" / "ice_full_dump_core2" / "ice_fct_dump_s1_rank0.txt"

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.exists() and FCTD.exists()),
    reason="CORE2 mesh / ice FCT dump missing (run port2/jobs/jax_ice_full_dump_core2.sh)",
)


@pytest.fixture(scope="module")
def fct():
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_adv
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(MESH_DIR)
    # The C FCT dumps predate drift fix 3 (2026-09-18: scale_area is the namelist 5.8e9,
    # not the 2.0e8 module default) — replay the value the dumps were made with.
    cfg = IceConfig(scale_area=2.0e8)
    with open(FCTD) as f:
        names = f.readline().split("var=")[1].split()[0].split(",")
    d = np.loadtxt(FCTD)
    col = {n: d[:, i] for i, n in enumerate(names)}
    a_out, m_out, ms_out = ice_adv.fct_solve(
        cfg, mesh, jnp.asarray(col["a_in"]), jnp.asarray(col["m_in"]),
        jnp.asarray(col["ms_in"]), jnp.asarray(col["u_ice"]), jnp.asarray(col["v_ice"]))
    return mesh, col, (np.asarray(a_out), np.asarray(m_out), np.asarray(ms_out))


@pytest.mark.parametrize("field,out_i", [("a", 0), ("m", 1), ("ms", 2)])
def test_fct_matches_dump(fct, field, out_i):
    """Each advected tracer matches the C FCT output to machine precision (element-based)."""
    _mesh, col, outs = fct
    j = outs[out_i]
    c = col[f"{field}_out"]
    d = np.abs(j - c)
    assert d.max() < 1e-12, f"{field}: max|Δ|={d.max():.3e} rel={d.max()/(np.abs(c).max()+1e-30):.3e}"


def test_fct_is_active(fct):
    """The dump exercises real advection (the FCT moved the tracers — the gate isn't trivial)."""
    _mesh, col, _outs = fct
    assert np.abs(col["a_out"] - col["a_in"]).max() > 1e-4
    assert np.abs(col["m_out"] - col["m_in"]).max() > 1e-4


def test_fct_bounded(fct):
    """The Zalesak limiter keeps the solution BOUNDED — only a small antidiffusive overshoot
    past the input range (the C does the same; cut_off clamps a≤1 afterward). Without the
    limiter the high-order MFCT would overshoot far more. The exact overshoot matches the C
    (the dump gate), so this just confirms boundedness, not a tight bound."""
    _mesh, col, outs = fct
    a_out = outs[0]
    over = a_out.max() - col["a_in"].max()
    under = col["a_in"].min() - a_out.min()
    assert over < 0.05 and under < 0.05, f"overshoot {over:.4f} / undershoot {under:.4f}"
    # and it equals the C overshoot (JAX == C, already gated tight)
    assert abs((a_out.max() - col["a_out"].max())) < 1e-12


def test_fct_ad_finite(fct):
    """``d(Σm_out)/d(m_in)`` and ``d(·)/d(u_ice)`` finite everywhere — the limiter ratio floor
    (1e-12) keeps the min/max/where VJP NaN-safe; no division by a tracer value."""
    import jax
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_adv
    mesh, col, _outs = fct
    # The C FCT dumps predate drift fix 3 (2026-09-18: scale_area is the namelist 5.8e9,
    # not the 2.0e8 module default) — replay the value the dumps were made with.
    cfg = IceConfig(scale_area=2.0e8)
    a_in = jnp.asarray(col["a_in"]); m_in = jnp.asarray(col["m_in"])
    ms_in = jnp.asarray(col["ms_in"]); u = jnp.asarray(col["u_ice"]); v = jnp.asarray(col["v_ice"])

    def loss_m(mm):
        _a, mo, _ms = ice_adv.fct_solve(cfg, mesh, a_in, mm, ms_in, u, v)
        return jnp.sum(mo)
    g = np.asarray(jax.grad(loss_m)(m_in))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)

    def loss_u(uu):
        ao, mo, mso = ice_adv.fct_solve(cfg, mesh, a_in, m_in, ms_in, uu, v)
        return jnp.sum(ao + mo + mso)
    g2 = np.asarray(jax.grad(loss_u)(u))
    assert np.all(np.isfinite(g2)) and np.any(g2 != 0.0)


def test_fct_derivative_options(fct):
    """The derivative-only FCT options (``IceConfig.adjoint_cuts`` "fct_limiter", "fct_flux")
    leave the forward bitwise unchanged. With ``freeze_limited_flux`` the tangent of every
    tracer, in m_in and in u_ice, is the tangent of the low-order solution; with
    ``freeze_limiter`` it differs from the full one only where the limiter is active."""
    import jax
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_adv
    mesh, col, _outs = fct
    cfg = IceConfig(scale_area=2.0e8)        # the dump's value (see test_fct_ad_finite)
    a_in = jnp.asarray(col["a_in"]); m_in = jnp.asarray(col["m_in"])
    ms_in = jnp.asarray(col["ms_in"]); u = jnp.asarray(col["u_ice"]); v = jnp.asarray(col["v_ice"])
    rng = np.random.default_rng(5)
    dm = jnp.asarray(rng.standard_normal(mesh.nod2D)) * 1e-2
    du = jnp.asarray(rng.standard_normal(mesh.nod2D)) * 1e-2
    prim = (a_in, m_in, ms_in, u, v)
    tan = (jnp.zeros_like(a_in), dm, jnp.zeros_like(ms_in), du, jnp.zeros_like(v))

    def jvp(**kw):
        f = lambda a, m, ms, uu, vv: ice_adv.fct_solve(cfg, mesh, a, m, ms, uu, vv, **kw)  # noqa: E731
        return jax.jvp(f, prim, tan)

    def low(a, m, ms, uu, vv):
        rm, ra, rms = ice_adv._tg_rhs(cfg, mesh, uu, vv, a, m, ms)
        return ice_adv._solve_low_order(cfg, mesh, a, m, ms, ra, rm, rms)

    out0, t0 = jvp()
    out1, t1 = jvp(freeze_limited_flux=True)
    out2, t2 = jvp(freeze_limiter=True)
    _, t_low = jax.jvp(low, prim, tan)
    for x, y, z in zip(out0, out1, out2):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes() == np.asarray(z).tobytes()
    for name, x, y in zip(("a", "m", "ms"), t1, t_low):
        assert np.all(np.isfinite(np.asarray(x)))
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-15, err_msg=name)
    m_full, m_frz = np.asarray(t0[1]), np.asarray(t2[1])
    changed = m_full != m_frz
    print(f"[fct options] freeze_limiter changes the m tangent on {changed.sum()} of "
          f"{mesh.nod2D} nodes; low-order tangent max |dm| {np.abs(np.asarray(t1[1])).max():.3e} "
          f"vs full {np.abs(m_full).max():.3e}")
    assert np.all(np.isfinite(m_frz)) and changed.any() and not changed.all()


def test_fct_bound_options(fct):
    """The derivative-only bound options (``IceConfig.adjoint_cuts`` "fct_pin", "fct_bound",
    "fct_hilo") leave the forward bitwise unchanged. "fct_bound" (the bounds tmax/tmin constant in the
    clipping ratios) changes the m tangent wherever a clipping ratio below 1 decides a limiter
    factor; "fct_pin" (only the bounds a node attains itself, tmax = 0 or tmin = 0) changes it on
    a subset of those nodes, and in open water (m_in = 0 on a whole element patch, where every
    clipping ratio is a tie at 0) its tangent is the "fct_bound" one."""
    import jax
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_adv, ice_evp
    mesh, col, _outs = fct
    cfg = IceConfig(scale_area=2.0e8)        # the dump's value (see test_fct_ad_finite)
    a_in = jnp.asarray(col["a_in"]); m_in = jnp.asarray(col["m_in"])
    ms_in = jnp.asarray(col["ms_in"]); u = jnp.asarray(col["u_ice"]); v = jnp.asarray(col["v_ice"])
    rng = np.random.default_rng(7)
    dm = jnp.asarray(rng.standard_normal(mesh.nod2D)) * 1e-2
    prim = (a_in, m_in, ms_in, u, v)
    zero = jnp.zeros_like(a_in)
    tan = (zero, dm, zero, zero, zero)

    def jvp(**kw):
        f = lambda a, m, ms, uu, vv: ice_adv.fct_solve(cfg, mesh, a, m, ms, uu, vv, **kw)  # noqa: E731
        return jax.jvp(f, prim, tan)

    out0, t0 = jvp()
    out_p, t_p = jvp(pin_bounds=True)
    out_b, t_b = jvp(freeze_bounds=True)
    out_h, t_h = jvp(pin_bounds=True, freeze_extremes=True)
    coast = np.asarray(ice_evp.boundary_node_mask(mesh))
    out_c, t_c = jvp(pin_bounds=True, freeze_extremes_at=jnp.asarray(coast))
    cring = np.asarray(ice_adv.coast_ring(mesh, jnp.asarray(coast)))
    out_r, t_r = jvp(pin_bounds=True, freeze_bounds_at=jnp.asarray(cring))
    for x, y, z, w, c, r in zip(out0, out_p, out_b, out_h, out_c, out_r):
        assert (np.asarray(x).tobytes() == np.asarray(y).tobytes() == np.asarray(z).tobytes()
                == np.asarray(w).tobytes() == np.asarray(c).tobytes() == np.asarray(r).tobytes())
    m0, mp, mb, mh = (np.asarray(t[1]) for t in (t0, t_p, t_b, t_h))
    assert all(np.all(np.isfinite(t)) for t in (mp, mb, mh))
    ch_p, ch_b = mp != m0, mb != m0
    # open-water nodes: m_in = 0 within three element rings, so that m_l = 0 on the patch of
    # every vertex of the node's elements and those vertices' bounds are pinned (tmax = tmin = 0)
    en = np.asarray(mesh.elem_nodes)
    deep = np.asarray(m_in) == 0.0
    for _ in range(3):
        ring = np.ones(mesh.nod2D, bool)
        np.logical_and.at(ring, en.reshape(-1), np.repeat(deep[en].all(axis=1), 3))
        deep = ring
    print(f"[fct bound options] fct_pin changes the m tangent on {ch_p.sum()}, fct_bound on "
          f"{ch_b.sum()} of {mesh.nod2D} nodes; open-water nodes {deep.sum()}; max |dm| live "
          f"{np.abs(m0).max():.3e}, pin {np.abs(mp).max():.3e}, bound {np.abs(mb).max():.3e}")
    assert ch_b.any() and not ch_b.all()
    assert ch_p.sum() <= ch_b.sum()
    assert ((mh != m0) | ~ch_p).all()        # pin + extremes changes at least what pin does
    np.testing.assert_array_equal(mp[deep], mb[deep])
    np.testing.assert_array_equal(mh[deep], mb[deep])
    # coast only: the tangent differs from the pinned one only on vertices of coastal elements
    mc = np.asarray(t_c[1])
    near = np.zeros(mesh.nod2D, bool)
    np.logical_or.at(near, en.reshape(-1), np.repeat(coast[en].any(axis=1), 3))
    print(f"[fct bound options] fct_hilo_coast: {coast.sum()} coastal nodes, changes the pinned "
          f"tangent on {(mc != mp).sum()} nodes, max |dm| {np.abs(mc).max():.3e}")
    assert np.all(np.isfinite(mc))
    np.testing.assert_array_equal(mc[~near], mp[~near])
    assert (mc != mp).any()
    # fct_bound_coast: coast_ring = the coastal nodes + every vertex of a cell element with a
    # coastal vertex; the tangent differs from the pinned one only on vertices of elements that
    # touch the ring
    cell = np.asarray(mesh.ulevels) <= 1
    want = coast.copy()
    np.logical_or.at(want, en.reshape(-1), np.repeat(coast[en].any(axis=1) & cell, 3))
    np.testing.assert_array_equal(cring, want)
    mr = np.asarray(t_r[1])
    near2 = np.zeros(mesh.nod2D, bool)
    np.logical_or.at(near2, en.reshape(-1), np.repeat(cring[en].any(axis=1), 3))
    print(f"[fct bound options] fct_bound_coast: {cring.sum()} coast+ring nodes, changes the pinned "
          f"tangent on {(mr != mp).sum()} nodes, max |dm| {np.abs(mr).max():.3e}")
    assert np.all(np.isfinite(mr))
    np.testing.assert_array_equal(mr[~near2], mp[~near2])
    assert (mr != mp).any()
