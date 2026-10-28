"""The derivative-only mEVP damping settings of ``IceConfig`` (``adjoint_damping``,
``adjoint_kappa``, ``adjoint_sweeps``) and the local damping factor.

Two halves. The first needs no data: the trace-time validation (``IceConfig._replace``, the
path every driver takes, skips the validating ``__new__``, so ``ice_surface_step`` re-checks on
every trace), and the damping itself on a small synthetic mEVP mesh — inactive settings trace
the same program, the forward is bitwise unchanged, the damped one-sweep stress tangent is the
factor times the undamped one, and κ = 0 zeroes the factor on every ice element. The second
half needs the CORE2 mesh and the 2013-01-01 State: the library factor
(``ice_mevp.mevp_local_damping``) against the one the mode study measured with
(``scripts/adjoint/ice_modes.py``, the fallback-B ``local`` form) on the captured state of a real
model step, and the forward of the whole mEVP call for every setting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ice_mevp
from fesom_jax import ice_step as ice_step_mod
from fesom_jax.ice import IceConfig

ROOT = Path(__file__).resolve().parents[2]
STATES = Path("/work/ab0995/a270088/fesom_jax_dev/states_2013")


# ---------------------------------------------------------------------------------------------
# validation (the driver path: IceConfig()._replace)


def _cfg(**kw):
    return IceConfig()._replace(**kw)


GOOD = [
    dict(whichEVP=1, adjoint_mode="dynamics", adjoint_cuts=("advect", "ssh", "ocean_vel",
                                                            "ice2ocean", "fct")),
    dict(whichEVP=1, adjoint_mode="dynamics", adjoint_cuts=()),
    dict(whichEVP=1, adjoint_mode="dynamics"),
    dict(whichEVP=1, adjoint_mode="dynamics", adjoint_damping="local", adjoint_kappa=0.25),
    dict(whichEVP=1, adjoint_mode="dynamics", adjoint_damping="local", adjoint_kappa=0.0),
    dict(whichEVP=1, adjoint_mode="dynamics", adjoint_sweeps=8),
    dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="local"),
    dict(whichEVP=1, adjoint_mode="prescribed"),
    dict(whichEVP=0, adjoint_mode="rheology"),
    dict(whichEVP=0, adjoint_mode="exact", adjoint_kappa=3.0),     # κ alone is inert
]

BAD = [
    (dict(whichEVP=1, adjoint_mode="dynamics", adjoint_cuts=("advekt",)), "unknown name"),
    (dict(whichEVP=1, adjoint_mode="dynamics", adjoint_cuts="ssh"), "tuple"),
    (dict(whichEVP=1, adjoint_mode="prescribed", adjoint_cuts=("ssh",)), "dynamics"),
    # adjoint_cuts act under "dynamics" and "exact" (the year-window settings cut fct_flux / strength_conc in exact);
    # any other mode refuses them
    (dict(whichEVP=1, adjoint_mode="rheology", adjoint_cuts=("advect",)), "dynamics"),
    (dict(whichEVP=1, adjoint_mode="dynamics", adjoint_damping="implicit"), "adjoint_damping"),
    (dict(whichEVP=1, adjoint_mode="dynamics", adjoint_damping="local", adjoint_kappa=-0.1),
     "adjoint_kappa"),
    (dict(whichEVP=1, adjoint_mode="dynamics", adjoint_kappa=float("nan")), "adjoint_kappa"),
    (dict(whichEVP=1, adjoint_mode="dynamics", adjoint_kappa=float("inf")), "adjoint_kappa"),
    (dict(whichEVP=1, adjoint_mode="dynamics", adjoint_sweeps=-2), "adjoint_sweeps"),
    (dict(whichEVP=1, adjoint_mode="dynamics", adjoint_sweeps=2.5), "adjoint_sweeps"),
    (dict(whichEVP=0, adjoint_mode="dynamics"), "whichEVP"),
    (dict(whichEVP=0, adjoint_mode="exact", adjoint_damping="local"), "whichEVP"),
    (dict(whichEVP=0, adjoint_mode="exact", adjoint_sweeps=4), "whichEVP"),
    (dict(whichEVP=1, adjoint_mode="prescribed", adjoint_damping="local"), "does not"),
    (dict(whichEVP=1, adjoint_mode="rheology", adjoint_sweeps=4), "does not"),
    (dict(whichEVP=1, adjoint_mode="frozen", adjoint_damping="local"), "does not"),
]


@pytest.mark.parametrize("kw", GOOD)
def test_valid_settings_pass(kw):
    ice_step_mod.validate_adjoint_settings(_cfg(**kw))


@pytest.mark.parametrize("kw,match", BAD)
def test_invalid_settings_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        ice_step_mod.validate_adjoint_settings(_cfg(**kw))


@pytest.mark.parametrize("kw,match", BAD[:7] + BAD[10:12] + BAD[13:14])
def test_the_ice_step_validates_at_trace_time(kw, match):
    """``ice_surface_step`` checks before it reads anything else, so a bad ``_replace``-built
    config fails there with the ValueError even with no mesh or state at all; a good one gets
    past the check and fails only on the missing mesh."""
    with pytest.raises(ValueError, match=match):
        ice_step_mod.ice_surface_step(_cfg(**kw), None, None, None, None, dt=1800.0)
    with pytest.raises((AttributeError, TypeError)):
        ice_step_mod.ice_surface_step(_cfg(**GOOD[3]), None, None, None, None, dt=1800.0)


def test_defaults_are_inactive():
    c = IceConfig()
    assert (c.adjoint_damping, c.adjoint_kappa, c.adjoint_sweeps, c.adjoint_cuts) == \
        ("none", 0.25, -1, ())


# ---------------------------------------------------------------------------------------------
# a synthetic mEVP mesh: a 5x5 node grid on a 100 km plane, 32 triangles


def _synthetic():
    n, dx = 5, 1.0e5
    xy = np.array([(i * dx, j * dx) for j in range(n) for i in range(n)], float)
    tri = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            tri += [(a, b, d), (a, d, c)]
    en = np.array(tri, np.int32)
    gs = np.zeros((len(en), 6))
    area_e = np.zeros(len(en))
    for e, (p, q, r) in enumerate(en):
        (x1, y1), (x2, y2), (x3, y3) = xy[p], xy[q], xy[r]
        det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area_e[e] = 0.5 * abs(det)
        gs[e, :3] = np.array([y2 - y3, y3 - y1, y1 - y2]) / det      # d/dx of the P1 basis
        gs[e, 3:] = np.array([x3 - x2, x1 - x3, x2 - x1]) / det      # d/dy
    area_n = np.zeros(n * n)
    np.add.at(area_n, en.ravel(), np.repeat(area_e / 3.0, 3))
    edge = np.array([(i in (0, n - 1)) or (j in (0, n - 1)) for j in range(n) for i in range(n)])
    mesh = SimpleNamespace(
        nod2D=n * n, elem2D=len(en), elem_nodes=jnp.asarray(en), gradient_sca=jnp.asarray(gs),
        metric_factor=jnp.zeros(len(en)), elem_area=jnp.asarray(area_e),
        area=jnp.asarray(area_n)[:, None], ulevels_nod2D=np.ones(n * n, np.int32),
        ulevels=np.ones(len(en), np.int32), coriolis_node=jnp.full(n * n, 1.4e-4))
    rng = np.random.default_rng(3)
    nod = n * n
    a = np.clip(0.8 + 0.2 * rng.random(nod), 0.0, 1.0)
    a[:2 * n] = 0.0                    # open water on the first two rows: ice-free elements
    m = 2.0 * a + 0.5 * rng.random(nod) * a
    # nearly rigid ice on rows 2-3 (velocity differences 1e-10 m/s over 100 km: Δ ≪ δmin, so
    # the local stability number γ/αβ is O(1) there, as in CORE2's near-rigid elements) and
    # deforming ice on the last row (1e-2 m/s: γ/αβ ≪ κ, no damping)
    amp = np.where(np.arange(nod) < (n - 1) * n, 1e-10, 1e-2)
    fields = dict(
        a_ice=jnp.asarray(a), m_ice=jnp.asarray(m), m_snow=jnp.asarray(0.2 * m),
        u_ice=jnp.asarray(0.05 + amp * rng.standard_normal(nod)),
        v_ice=jnp.asarray(-0.02 + amp * rng.standard_normal(nod)),
        sigma11=jnp.asarray(1e3 * rng.standard_normal(len(en))),
        sigma12=jnp.asarray(1e3 * rng.standard_normal(len(en))),
        sigma22=jnp.asarray(1e3 * rng.standard_normal(len(en))),
        srfoce_u=jnp.asarray(0.1 * rng.standard_normal(nod)),
        srfoce_v=jnp.asarray(0.1 * rng.standard_normal(nod)),
        elevation=jnp.asarray(0.01 * rng.standard_normal(nod)),
        stress_ax=jnp.asarray(0.1 + 0.05 * rng.standard_normal(nod)),
        stress_ay=jnp.asarray(0.05 * rng.standard_normal(nod)),
        boundary_node=jnp.asarray(edge))
    return mesh, fields


SYN = None


def _syn():
    global SYN
    if SYN is None:
        SYN = _synthetic()
    return SYN


def _mevp_cfg(**kw):
    return IceConfig()._replace(whichEVP=1, evp_rheol_steps=12, ice_dt=1800.0, **kw)


def _run(cfg, mesh, fields, adjoint_subcycles=-1):
    def f(u0):
        kw = dict(fields, u_ice=u0)
        return ice_mevp.mevp_dynamics(cfg, mesh, adjoint_subcycles=adjoint_subcycles,
                                      exch=None, **kw)
    return f


def test_inactive_fields_trace_the_same_program():
    mesh, fields = _syn()
    ref = str(jax.make_jaxpr(_run(_mevp_cfg(), mesh, fields))(fields["u_ice"]))
    for kw in (dict(adjoint_kappa=0.7), dict(adjoint_kappa=0.0), dict(adjoint_damping="none")):
        got = str(jax.make_jaxpr(_run(_mevp_cfg(**kw), mesh, fields))(fields["u_ice"]))
        assert got == ref, kw
    damped = str(jax.make_jaxpr(_run(_mevp_cfg(adjoint_damping="local"), mesh, fields))(
        fields["u_ice"]))
    assert damped != ref


@pytest.mark.parametrize("kw,k", [(dict(adjoint_damping="local"), -1),
                                  (dict(adjoint_damping="local", adjoint_kappa=0.0), -1),
                                  (dict(adjoint_damping="local"), 4), ({}, 4)])
def test_the_forward_is_bitwise_unchanged(kw, k):
    mesh, fields = _syn()
    ref = jax.jit(_run(_mevp_cfg(), mesh, fields))(fields["u_ice"])
    got = jax.jit(_run(_mevp_cfg(**kw), mesh, fields, k))(fields["u_ice"])
    for a, b in zip(ref, got):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    # also the primal of a linearization (the gradient programs' forward)
    got_lin, _ = jax.linearize(_run(_mevp_cfg(**kw), mesh, fields, k), fields["u_ice"])
    for a, b in zip(ref, got_lin):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-14)


def _one_sweep_setup(cfg, mesh, fields):
    st = ice_mevp.mevp_setup(cfg, mesh, fields["a_ice"], fields["m_ice"], fields["m_snow"],
                             fields["elevation"], fields["boundary_node"])
    coef = ice_mevp.mevp_stability_coef(cfg, mesh, fields["m_ice"], fields["m_snow"])
    return st, coef


def test_the_damped_one_sweep_stress_tangent_is_f_times_the_undamped_one():
    mesh, fields = _syn()
    cfg = _mevp_cfg()
    st, coef = _one_sweep_setup(cfg, mesh, fields)
    x0 = tuple(fields[k] for k in ("u_ice", "v_ice", "sigma11", "sigma12", "sigma22"))
    rng = np.random.default_rng(5)
    dx = tuple(jnp.asarray(rng.standard_normal(np.shape(q)) * (0.01 if i < 2 else 100.0))
               for i, q in enumerate(x0))
    f = ice_mevp.mevp_local_damping(cfg, mesh, st, x0[0], x0[1], coef, 0.25)
    fn = np.asarray(f)
    ice_el = np.asarray(st.ice_el)
    assert np.all(fn[~ice_el] == 1.0) and np.all((fn > 0) & (fn <= 1))
    assert np.any(fn[ice_el] < 1.0), "pick a synthetic state where the damping acts"

    def sweep(out_f):
        return lambda *x: ice_mevp.mevp_iterate(
            cfg, mesh, *x, st, x0[0], x0[1], fields["srfoce_u"], fields["srfoce_v"],
            fields["stress_ax"], fields["stress_ay"], cfg.ice_dt, sigma_out_f=out_f)

    y0, t0 = jax.jvp(sweep(None), x0, dx)
    y1, t1 = jax.jvp(sweep(f), x0, dx)
    for a, b in zip(y0, y1):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    for k in (2, 3, 4):
        np.testing.assert_array_equal(np.asarray(t1[k]), fn * np.asarray(t0[k]))


def test_kappa_zero_zeroes_the_factor_on_every_ice_element():
    mesh, fields = _syn()
    cfg = _mevp_cfg()
    st, coef = _one_sweep_setup(cfg, mesh, fields)
    ice_el = np.asarray(st.ice_el)
    assert 0 < ice_el.sum() < ice_el.size
    f = np.asarray(ice_mevp.mevp_local_damping(cfg, mesh, st, fields["u_ice"], fields["v_ice"],
                                               coef, 0.0))
    assert np.all(f[ice_el] == 0.0) and np.all(f[~ice_el] == 1.0)


def test_the_stability_number_has_the_documented_form():
    """γ/αβ = P/(Δ+δmin)·rdt·M/(1+M²)/A/(αβ), recomputed in numpy from the definition."""
    mesh, fields = _syn()
    cfg = _mevp_cfg()
    st, coef = _one_sweep_setup(cfg, mesh, fields)
    g = np.asarray(ice_mevp.mevp_stability_number(cfg, mesh, st, fields["u_ice"],
                                                  fields["v_ice"], coef))
    en = np.asarray(mesh.elem_nodes)
    gs = np.asarray(mesh.gradient_sca)
    u, v = np.asarray(fields["u_ice"])[en], np.asarray(fields["v_ice"])[en]
    e11 = (gs[:, :3] * u).sum(1)
    e22 = (gs[:, 3:] * v).sum(1)
    e12 = 0.5 * ((gs[:, 3:] * u).sum(1) + (gs[:, :3] * v).sum(1))
    delta = np.sqrt((e11 + e22) ** 2 + cfg.vale * ((e11 - e22) ** 2 + 4 * e12 ** 2))
    mi = np.asarray(fields["m_ice"])[en].mean(1)
    ai = np.asarray(fields["a_ice"])[en].mean(1)
    M = cfg.rhoice * mi + cfg.rhosno * np.asarray(fields["m_snow"])[en].mean(1)
    P = cfg.pstar * mi * np.exp(-cfg.c_pressure * (1 - ai))
    want = P / (delta + cfg.delta_min) * cfg.ice_dt * M / (1 + M * M) \
        / np.asarray(mesh.elem_area) / (cfg.alpha_evp * cfg.beta_evp)
    ice_el = np.asarray(st.ice_el)
    np.testing.assert_allclose(g[ice_el], want[ice_el], rtol=1e-10)
    assert np.all(g[~ice_el] == 0.0)


def test_the_damping_reaches_the_gradient_and_only_the_gradient():
    """Through the whole call: the damped gradient differs from the undamped one, κ = 0 differs
    again, and K = 0 sweeps cut it to exactly zero whatever the damping."""
    mesh, fields = _syn()

    def g(cfg, k=-1):
        f = _run(cfg, mesh, fields, k)
        return np.asarray(jax.grad(lambda u0: jnp.sum(f(u0)[0] ** 2))(fields["u_ice"]))

    g_ex = g(_mevp_cfg())
    g_l = g(_mevp_cfg(adjoint_damping="local"))
    g_0 = g(_mevp_cfg(adjoint_damping="local", adjoint_kappa=0.0))
    assert np.all(np.isfinite(g_l)) and np.all(np.isfinite(g_0))
    assert not np.array_equal(g_ex, g_l) and not np.array_equal(g_l, g_0)
    assert np.all(g(_mevp_cfg(adjoint_damping="local"), 0) == 0.0)


# ---------------------------------------------------------------------------------------------
# CORE2: the library factor against the mode study's, on a captured model step


def _closure(fn, name):
    for n, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        if n == name:
            return cell.cell_contents
    raise KeyError(f"{name} not in the closure of {fn.__name__}")


@pytest.fixture(scope="module")
def icemap():
    if not ((ROOT / "data" / "mesh_core2").is_dir()
            and (STATES / "fesom.2013.001.00000").is_dir()):
        pytest.skip("CORE2 mesh / 2013 States missing")
    sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
    import ice_modes as IM
    return IM.IceMap("2013-01-01")


def test_core2_library_factor_equals_the_mode_study_factor(icemap):
    """``ice_mevp.mevp_local_damping`` (κ = 0.25) against ``ice_modes.py``'s ``local_factor``
    (the fallback-B ``local`` form the κ scan measured), at the captured entry iterate and at
    sweep 60 of the forward. The per-call coefficient is computed in numpy by the script and
    under jit by the library: equal to rounding (measured: 12.5 % of the elements differ, by at
    most 7.3e-16 relative, job 27987615). Given the same coefficient the factor is bitwise the
    script's, and with the library's own coefficient equal to 1e-14."""
    M = icemap
    cfg, mesh = M.cfg, M.mesh
    step_map = M.p_fwd.func.__wrapped__
    local_factor = _closure(_closure(_closure(step_map, "sweeps"), "sweep"), "local_factor")
    coef = jax.jit(lambda m, ms: ice_mevp.mevp_stability_coef(cfg, mesh, m, ms))(
        M.kw["m_ice"], M.kw["m_snow"])
    np.testing.assert_allclose(np.asarray(coef), np.asarray(M.inp["gam_c"]), rtol=2e-15, atol=0)
    st = M.st0
    its = M.p_iterates(M.x0, M.c0)
    x60 = tuple(q[60] for q in its)
    lib = jax.jit(lambda st_, u, v, c: ice_mevp.mevp_local_damping(cfg, mesh, st_, u, v, c, 0.25))
    scr = jax.jit(local_factor)
    gk = jnp.asarray([0.25, 0.0])
    for x in (M.x0, x60):
        f_scr = np.asarray(scr(M.inp, st, x, gk))
        f_same = np.asarray(lib(st, x[0], x[1], M.inp["gam_c"]))
        f_lib = np.asarray(lib(st, x[0], x[1], coef))
        ice = M.ice_el
        assert 0 < np.count_nonzero(f_lib[ice] < 1.0) < ice.sum()
        np.testing.assert_array_equal(f_same, f_scr)
        np.testing.assert_allclose(f_lib, f_scr, rtol=1e-14, atol=0)
