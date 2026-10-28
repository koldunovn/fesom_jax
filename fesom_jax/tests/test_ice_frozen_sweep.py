"""``IceConfig.adjoint_damping="frozen_sweep"``: the frozen-sweep backward of the mEVP solve
(fesom_jax/ice_mevp_frozen.py).

All on the small synthetic mEVP mesh of test_ice_damping_config (no data):
* the trace-time validation of the new form and its two fields;
* the forward is bitwise the model's, both called plainly and as the primal of a vjp;
* the custom_vjp backward is the transpose of the frozen-sweep TANGENT. That tangent is written
  here independently, in forward mode: one sweep linearized at the final iterate, applied n
  times with the anchor and input forcing in every application. It is compared by the dot
  product <ȳ, T·δ> = <Tᵀ·ȳ, δ>, with tangents on the entry state, every input field and a traced
  P* (the ice control's path into the config);
* κ0 → ∞ with θ = 1 keeps the coefficients and the stress derivative of the plain sweep.
The CORE2 check (the library backward equals the mode harness's TLfT on a captured step) is
scripts/tests/test_frozen_sweep_core2.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ice_mevp, ice_mevp_frozen
from fesom_jax import ice_step as ice_step_mod
from fesom_jax.ice import IceConfig
from fesom_jax.tests.test_ice_damping_config import _mevp_cfg, _syn

jax.config.update("jax_enable_x64", True)


def _cfg(**kw):
    return IceConfig()._replace(**kw)


GOOD = [
    dict(whichEVP=1, adjoint_mode="dynamics", adjoint_damping="frozen_sweep"),
    dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
         adjoint_alpha_kappa=0.25, adjoint_theta=0.0),
    dict(whichEVP=1, adjoint_mode="dynamics", adjoint_damping="frozen_sweep", adjoint_theta=1.0,
         adjoint_cuts=("ssh",)),
    dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
         adjoint_cuts=("fct_flux", "strength_conc"), adjoint_sweep_guard=1e4),
]
BAD = [
    (dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
          adjoint_alpha_kappa=0.0), "adjoint_alpha_kappa"),
    (dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
          adjoint_alpha_kappa=-1.0), "adjoint_alpha_kappa"),
    (dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
          adjoint_alpha_kappa=float("inf")), "adjoint_alpha_kappa"),
    (dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
          adjoint_theta=1.5), "adjoint_theta"),
    (dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
          adjoint_theta=-0.1), "adjoint_theta"),
    (dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
          adjoint_sweeps=8), "adjoint_sweeps"),
    (dict(whichEVP=0, adjoint_mode="exact", adjoint_damping="frozen_sweep"), "whichEVP"),
    (dict(whichEVP=1, adjoint_mode="rheology", adjoint_damping="frozen_sweep"), "does not"),
    (dict(whichEVP=1, adjoint_mode="prescribed", adjoint_damping="frozen_sweep"), "does not"),
    (dict(whichEVP=1, adjoint_mode="prescribed", adjoint_cuts=("ssh",)), "dynamics"),
    (dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
          adjoint_sweep_guard=-1.0), "adjoint_sweep_guard"),
    (dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="none",
          adjoint_sweep_guard=10.0), "adjoint_sweep_guard"),
]


@pytest.mark.parametrize("kw", GOOD)
def test_valid_settings_pass(kw):
    ice_step_mod.validate_adjoint_settings(_cfg(**kw))


@pytest.mark.parametrize("kw,match", BAD)
def test_invalid_settings_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        ice_step_mod.validate_adjoint_settings(_cfg(**kw))


def test_defaults():
    c = IceConfig()
    assert (c.adjoint_alpha_kappa, c.adjoint_theta) == (1.0, 0.97)
    assert (c.adjoint_thin_mass, c.adjoint_sweep_guard) == (0.0, 0.0)
    assert IceConfig._fields[-4:] == ("adjoint_alpha_kappa", "adjoint_theta",
                                      "adjoint_thin_mass", "adjoint_sweep_guard")


def test_mevp_dynamics_refuses_subcycles():
    mesh, fields = _syn()
    with pytest.raises(ValueError, match="frozen_sweep"):
        ice_mevp.mevp_dynamics(_mevp_cfg(adjoint_damping="frozen_sweep"), mesh,
                               adjoint_subcycles=4, **fields)


# ---------------------------------------------------------------------------------------------

STATE = ("u_ice", "v_ice", "sigma11", "sigma12", "sigma22")
INPUTS = ("a_ice", "m_ice", "m_snow", "elevation", "srfoce_u", "srfoce_v", "stress_ax",
          "stress_ay")


def _solve(cfg, mesh, fields):
    """(state, inputs, pstar) -> the five outputs of mevp_dynamics with P* = pstar (traced)."""
    def f(state, inputs, pstar):
        kw = dict(fields, **dict(zip(STATE, state)), **dict(zip(INPUTS, inputs)))
        return ice_mevp.mevp_dynamics(cfg._replace(pstar=pstar), mesh, exch=None, **kw)
    return f


def _args(fields, pstar):
    return (tuple(fields[k] for k in STATE), tuple(fields[k] for k in INPUTS),
            jnp.asarray(pstar))


@pytest.mark.parametrize("theta,kappa0", [(0.75, 1.0), (0.0, 0.25), (1.0, 1.0)])
def test_the_forward_is_bitwise_unchanged(theta, kappa0):
    mesh, fields = _syn()
    base = _mevp_cfg()
    frz = _mevp_cfg(adjoint_damping="frozen_sweep", adjoint_theta=theta,
                    adjoint_alpha_kappa=kappa0)
    args = _args(fields, base.pstar)
    ref = jax.jit(_solve(base, mesh, fields))(*args)
    got = jax.jit(_solve(frz, mesh, fields))(*args)
    prim = jax.jit(lambda *a: jax.vjp(_solve(frz, mesh, fields), *a)[0])(*args)
    for a, b, c in zip(ref, got, prim):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes() == np.asarray(c).tobytes()


def _frozen_tangent_reference(cfg, mesh, fields, args, dargs):
    """The frozen-sweep tangent in FORWARD mode, written from the definition: x_N from the plain
    forward; α_t, β_t at x_N; one θ-sweep g(x, anchor, inputs, P*) linearized at x_N; n
    applications δx ← J_x δx + J_a δa + J_in δin + J_P δP* from δx_0 = the entry state tangent."""
    state, inputs, pstar = args
    dstate, dinputs, dpstar = dargs
    c = cfg._replace(pstar=pstar)
    bn = fields["boundary_node"]
    xN = _solve(_mevp_cfg(), mesh, fields)(state, inputs, pstar)
    a, m, ms, elev, su, sv, sax, say = inputs
    stN = ice_mevp.mevp_setup(c, mesh, a, m, ms, elev, bn)
    al, be = ice_mevp_frozen.raised_coefficients(c, mesh, stN, m, ms, xN[0], xN[1],
                                                 float(cfg.adjoint_alpha_kappa))

    def g(x, ua, va, inp, p):
        c_ = cfg._replace(pstar=p)
        a_, m_, ms_, elev_, su_, sv_, sax_, say_ = inp
        st = ice_mevp.mevp_setup(c_, mesh, a_, m_, ms_, elev_, bn)
        ct = c_._replace(alpha_evp=al, beta_evp=be)
        st = st._replace(pressure_fac=st.pressure_fac * (ct.mevp_det2 / c_.mevp_det2))
        return ice_mevp_frozen.theta_sweep(ct, mesh, *x, st, ua, va, su_, sv_, sax_, say_,
                                           c_.ice_dt, float(cfg.adjoint_theta))
    _, lin = jax.linearize(g, xN, state[0], state[1], inputs, pstar)
    dx = dstate
    for _ in range(int(cfg.evp_rheol_steps)):
        dx = lin(dx, dstate[0], dstate[1], dinputs, dpstar)
    return dx


def _dot(a, b):
    return sum(float(jnp.vdot(p, q)) for p, q in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


@pytest.mark.parametrize("theta,kappa0", [(0.75, 1.0), (0.0, 0.25)])
def test_the_backward_is_the_transposed_frozen_tangent(theta, kappa0):
    mesh, fields = _syn()
    cfg = _mevp_cfg(adjoint_damping="frozen_sweep", adjoint_theta=theta,
                    adjoint_alpha_kappa=kappa0)
    args = _args(fields, cfg.pstar)
    rng = np.random.default_rng(11)
    rnd = lambda t: jax.tree.map(lambda q: jnp.asarray(rng.standard_normal(np.shape(q))  # noqa
                                                       * (np.abs(q).max() + 1.0)), t)
    dargs = (rnd(args[0]), rnd(args[1]), jnp.asarray(0.3 * cfg.pstar))
    ybar = rnd(args[0])
    y, back = jax.vjp(_solve(cfg, mesh, fields), *args)
    bar = back(tuple(ybar))
    tan = _frozen_tangent_reference(cfg, mesh, fields, args, dargs)
    lhs = _dot(ybar, tan)
    rhs = _dot(bar, dargs)
    assert np.isfinite(lhs) and abs(lhs) > 0
    assert rhs == pytest.approx(lhs, rel=1e-9)
    # P* really reaches the backward (the traced config field comes out as a cotangent)
    assert float(bar[2]) != 0.0


def test_the_raised_coefficients_respect_kappa0():
    mesh, fields = _syn()
    cfg = _mevp_cfg()
    bn = fields["boundary_node"]
    st = ice_mevp.mevp_setup(cfg, mesh, fields["a_ice"], fields["m_ice"], fields["m_snow"],
                             fields["elevation"], bn)
    coef = ice_mevp.mevp_stability_coef(cfg, mesh, fields["m_ice"], fields["m_snow"])
    s = np.asarray(ice_mevp.mevp_stability_number(cfg, mesh, st, fields["u_ice"],
                                                  fields["v_ice"], coef))
    ice = np.asarray(st.ice_el)
    for k0 in (0.25, 1.0, np.inf):
        al, be = (np.asarray(q) for q in ice_mevp_frozen.raised_coefficients(
            cfg, mesh, st, fields["m_ice"], fields["m_snow"], fields["u_ice"], fields["v_ice"],
            k0))
        assert np.all(al >= cfg.alpha_evp) and np.all(al[~ice] == cfg.alpha_evp)
        # γ/(α_t·α_t) ≤ κ0 wherever α_t was raised
        up = al > cfg.alpha_evp
        if np.isinf(k0):
            assert not up.any()
        else:
            assert up.any()
            np.testing.assert_allclose(s[up] * cfg.alpha_evp ** 2 / al[up] ** 2, k0, rtol=1e-12)
        en = np.asarray(mesh.elem_nodes)
        ref = np.full(int(mesh.nod2D), -np.inf)
        np.maximum.at(ref, en.ravel(), np.repeat(al, 3))
        assert np.array_equal(be, ref)


def test_a_coastal_mask_traced_inside_the_jit_is_not_closed_over():
    """The model computes boundary_node inside its jit (a tracer); the custom_vjp must take it as
    an argument (the first coupled run failed with UnexpectedTracerError)."""
    mesh, fields = _syn()
    cfg = _mevp_cfg(adjoint_damping="frozen_sweep")

    def loss(u0, bn0):
        bn = bn0 | (u0 > 1e9)                                  # a traced bool mask
        kw = dict(fields, u_ice=u0, boundary_node=bn)
        y = ice_mevp.mevp_dynamics(cfg, mesh, exch=None, **kw)
        return jnp.sum(y[0] ** 2 + y[1] ** 2)
    g = jax.jit(jax.grad(loss))(fields["u_ice"], fields["boundary_node"])
    assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0)


def _with_pad_node(mesh, fields):
    """The synthetic mesh plus one node that belongs to no element, with zero fields, as a pad
    lane of a sharded mesh is (area 0, no ice, not coastal)."""
    from types import SimpleNamespace
    n = int(mesh.nod2D)
    def pad(q, v=0.0):
        q = jnp.asarray(q)
        return jnp.concatenate([q, jnp.full((1,) + q.shape[1:], v, q.dtype)])
    m = SimpleNamespace(**vars(mesh))
    m.nod2D = n + 1
    m.area = pad(mesh.area)
    m.coriolis_node = pad(mesh.coriolis_node)
    m.ulevels_nod2D = np.concatenate([mesh.ulevels_nod2D, [1]]).astype(np.int32)
    f = {k: (pad(v, False) if k == "boundary_node" else pad(v)) if np.shape(v)[0] == n else v
         for k, v in fields.items()}
    return m, f


def test_a_node_without_elements_gets_a_finite_backward():
    """β_t is a segment_max over a node's elements, and the max over no elements is −inf. A pad
    lane of a sharded mesh is such a node: its masked node solve then holds 0·(−inf) and the
    backward put NaN on the pad lane and into d/d(C_d) (CORE2 on 2 devices, I18). The pad node
    must get a finite β_t, and the real lanes must not notice it."""
    mesh, fields = _syn()
    pmesh, pfields = _with_pad_node(mesh, fields)
    cfg = _mevp_cfg(adjoint_damping="frozen_sweep")
    n = int(mesh.nod2D)

    def solve(m, f):
        def run(state, inputs, pstar, cd):
            kw = dict(f, **dict(zip(STATE, state)), **dict(zip(INPUTS, inputs)))
            c = cfg._replace(pstar=pstar, cd_oce_ice=cd)
            return ice_mevp.mevp_dynamics(c, m, exch=None, **kw)
        return run

    def args(f):
        return (tuple(f[k] for k in STATE), tuple(f[k] for k in INPUTS),
                jnp.asarray(cfg.pstar), jnp.asarray(cfg.cd_oce_ice))
    rng = np.random.default_rng(5)
    ybar = tuple(jnp.asarray(rng.standard_normal(np.shape(fields[k]))) for k in STATE)
    ybar_p = tuple(jnp.concatenate([y, jnp.zeros(1)]) if np.shape(y)[0] == n else y
                   for y in ybar)
    _, back = jax.vjp(solve(mesh, fields), *args(fields))
    _, back_p = jax.vjp(solve(pmesh, pfields), *args(pfields))
    ref, got = back(ybar), back_p(ybar_p)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
        a, b = np.asarray(a), np.asarray(b)
        assert np.all(np.isfinite(b))
        if b.shape and b.shape[0] == n + 1:
            assert b[n] == 0.0
            b = b[:n]
        np.testing.assert_allclose(b, a, rtol=1e-13, atol=0.0)
    st = ice_mevp.mevp_setup(cfg, pmesh, pfields["a_ice"], pfields["m_ice"], pfields["m_snow"],
                             pfields["elevation"], pfields["boundary_node"])
    _, be = ice_mevp_frozen.raised_coefficients(cfg, pmesh, st, pfields["m_ice"],
                                                pfields["m_snow"], pfields["u_ice"],
                                                pfields["v_ice"], 1.0)
    assert float(be[n]) == cfg.alpha_evp
