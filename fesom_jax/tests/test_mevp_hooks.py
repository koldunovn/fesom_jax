"""The derivative-only stress-tangent hooks of ``ice_mevp.mevp_iterate`` (Track I, Task 3).

``sigma_in_f`` scales the tangent of the incoming σ (explicit Newtonian damping);
``sigma_out_f`` scales the tangent of the new σ after the element skip and before the
stress-divergence scatter (implicit form, Panteleev et al. 2020). What is pinned here:

* hooks ``None`` trace NO scaling at all (the unhooked program), hooks set trace exactly three;
* the forward is bitwise the unhooked one for any factor;
* ``sigma_out_f = f``: the σ tangent of one sweep is exactly ``f`` times the undamped one, and
  the SAME sweep's velocity tangent is affine in ``f`` and moves with it (the placement
  before the scatter — a hook after the scatter would leave the velocity tangent unchanged);
* ``sigma_in_f = g``: the whole one-sweep tangent equals the undamped tangent with
  ``δσ_in`` replaced by ``g·δσ_in``;
* ``f = 1`` is plain autodiff; ``f = 0`` removes the σ memory exactly;
* reverse mode is the transpose of the hooked forward mode (dot test).

The point of evaluation is a seeded CORE2 ice cover (PHC IC + ``seed_ice``) spun up by 30
plain mEVP sweeps under a synthetic wind stress, so u, v and σ are all non-zero. CORE2 mesh ⇒
``core2`` suite group; skips without the mesh / IC.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing")

SPIN = 30


@pytest.fixture(scope="module")
def setup():
    from fesom_jax import ice as icemod
    from fesom_jax import ice_mevp
    from fesom_jax.ice import IceConfig
    from fesom_jax.ice_evp import boundary_node_mask
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import phc_initial_state

    mesh = load_mesh(MESH_DIR)
    bare = phc_initial_state(mesh, IC_DIR)
    s = icemod.seed_ice(bare, mesh, bare.T[:, 0])
    cfg = IceConfig(whichEVP=1)._replace(ice_dt=1800.0)
    st = ice_mevp.mevp_setup(cfg, mesh, s.a_ice, s.m_ice, s.m_snow,
                             jnp.zeros(mesh.nod2D), boundary_node_mask(mesh))
    assert int(jnp.count_nonzero(st.ice_el)) > 1000
    lon = jnp.asarray(mesh.geo_coord_nod2D)[:, 0]              # radians
    lat = jnp.asarray(mesh.geo_coord_nod2D)[:, 1]
    sax, say = 0.15 * jnp.cos(2 * lon) * jnp.cos(lat), 0.10 * jnp.sin(3 * lon)
    zu = jnp.zeros(mesh.nod2D)
    ze = jnp.zeros(mesh.elem2D)
    rdt = cfg.ice_dt

    def sweep(x, sigma_in_f=None, sigma_out_f=None):
        return ice_mevp.mevp_iterate(cfg, mesh, *x, st, zu, zu, zu, zu, sax, say, rdt,
                                     sigma_in_f=sigma_in_f, sigma_out_f=sigma_out_f)

    @jax.jit
    def spin(x):
        return jax.lax.scan(lambda c, _: (sweep(c), None), x, None, length=SPIN)[0]

    x0 = spin((zu, zu, ze, ze, ze))
    for name, a in zip(("u", "v", "s11", "s12", "s22"), x0):
        assert float(jnp.max(jnp.abs(a))) > 0.0, f"spin-up left {name} identically zero"
    rng = np.random.default_rng(7)
    dx = tuple(jnp.asarray(rng.standard_normal(a.shape)) * (jnp.std(a) + 1e-12) for a in x0)
    f_el = jnp.where(st.ice_el, jnp.asarray(rng.uniform(0.3, 0.95, mesh.elem2D)), 1.0)
    return dict(mesh=mesh, st=st, sweep=sweep, x0=x0, dx=dx, f_el=f_el)


def _jvp(setup, tangent, **hooks):
    fn = jax.jit(lambda x, t: jax.jvp(lambda y: setup["sweep"](y, **hooks), (x,), (t,)))
    return jax.device_get(fn(setup["x0"], tangent))


def _n_scaled(jaxpr_text: str) -> int:
    return jaxpr_text.count("name=_scaled_tangent")


def _zero_like(x):
    return tuple(jnp.zeros_like(a) for a in x)


def test_hooks_none_trace_the_unhooked_program(setup):
    sw, x0, f = setup["sweep"], setup["x0"], setup["f_el"]
    plain = str(jax.make_jaxpr(lambda x: sw(x))(x0))
    none = str(jax.make_jaxpr(lambda x: sw(x, sigma_in_f=None, sigma_out_f=None))(x0))
    assert plain == none
    assert _n_scaled(plain) == 0
    assert _n_scaled(str(jax.make_jaxpr(lambda x: sw(x, sigma_out_f=f))(x0))) == 3
    assert _n_scaled(str(jax.make_jaxpr(lambda x: sw(x, sigma_in_f=f))(x0))) == 3


def test_forward_is_bitwise_with_hooks(setup):
    sw, x0, f = setup["sweep"], setup["x0"], setup["f_el"]
    ref = jax.device_get(jax.jit(lambda x: sw(x))(x0))
    for hooks in ({"sigma_out_f": f}, {"sigma_in_f": f}, {"sigma_in_f": 0.0, "sigma_out_f": 0.0}):
        got = jax.device_get(jax.jit(lambda x, h=hooks: sw(x, **h))(x0))
        for a, b in zip(ref, got):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), hooks


def test_sigma_out_scales_the_new_stress_tangent_and_the_same_sweep_sees_it(setup):
    dx, f = setup["dx"], setup["f_el"]
    _, t1 = _jvp(setup, dx)
    _, tf = _jvp(setup, dx, sigma_out_f=f)
    for k in (2, 3, 4):
        np.testing.assert_allclose(tf[k], np.asarray(f) * t1[k], rtol=1e-13, atol=0)
    # the velocity tangent is affine in a scalar f and depends on it (hook BEFORE the scatter)
    _, t0 = _jvp(setup, dx, sigma_out_f=0.0)
    _, th = _jvp(setup, dx, sigma_out_f=0.5)
    for k in (0, 1):
        scale = np.max(np.abs(t1[k]))
        np.testing.assert_allclose(th[k], 0.5 * (t0[k] + t1[k]), rtol=0, atol=1e-11 * scale)
        assert np.max(np.abs(t1[k] - t0[k])) > 1e-3 * scale, "velocity tangent ignores the hook"


def test_sigma_in_equals_the_undamped_tangent_of_a_scaled_input(setup):
    dx, g = setup["dx"], setup["f_el"]
    _, tg = _jvp(setup, dx, sigma_in_f=g)
    dx_g = dx[:2] + tuple(g * d for d in dx[2:])
    _, tr = _jvp(setup, dx_g)
    for a, b in zip(tg, tr):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14 * np.max(np.abs(b)))


def test_factor_one_is_plain_autodiff(setup):
    dx = setup["dx"]
    _, t1 = _jvp(setup, dx)
    _, tu = _jvp(setup, dx, sigma_in_f=1.0, sigma_out_f=1.0)
    for a, b in zip(t1, tu):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_factor_zero_removes_the_stress_memory(setup):
    dx = setup["dx"]
    only_sigma = (jnp.zeros_like(dx[0]), jnp.zeros_like(dx[1])) + dx[2:]
    _, t_in = _jvp(setup, only_sigma, sigma_in_f=0.0)
    for a in t_in:
        assert np.all(np.asarray(a) == 0.0)
    _, t_out = _jvp(setup, dx, sigma_out_f=0.0)
    for a in t_out[2:]:
        assert np.all(np.asarray(a) == 0.0)
    _, t_out_u = _jvp(setup, dx[:2] + _zero_like(dx[2:]), sigma_out_f=0.0)
    for a, b in zip(t_out[:2], t_out_u[:2]):              # δσ_in no longer reaches (u, v)
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_reverse_mode_is_the_transpose_of_the_hooked_tangent(setup):
    sw, x0, dx, f = setup["sweep"], setup["x0"], setup["dx"], setup["f_el"]
    rng = np.random.default_rng(11)
    w = tuple(jnp.asarray(rng.standard_normal(a.shape)) for a in x0)
    for hooks in ({"sigma_out_f": f}, {"sigma_in_f": f}):
        _, t = _jvp(setup, dx, **hooks)
        _, vjp = jax.vjp(lambda y, h=hooks: sw(y, **h), x0)
        (ct,) = jax.jit(vjp)(w)
        lhs = sum(float(jnp.vdot(a, b)) for a, b in zip(w, t))
        rhs = sum(float(jnp.vdot(a, b)) for a, b in zip(ct, dx))
        assert abs(lhs - rhs) <= 1e-11 * max(abs(lhs), abs(rhs)), (hooks, lhs, rhs)
