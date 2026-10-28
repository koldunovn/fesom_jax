"""The "frozen_sweep" backward of the mEVP sweeps (``IceConfig.adjoint_damping="frozen_sweep"``).

The forward is :func:`fesom_jax.ice_mevp.mevp_dynamics`' unchanged: ``mevp_setup`` and 120 sweeps.
Only the derivative changes. The 120-sweep loop sits inside a ``jax.custom_vjp`` whose backward
does not unroll the forward iterates. It linearizes ONE sweep once, at the step's final iterate
x_N = (u_N, v_N, σ_N), and runs the transposed frozen iteration 120 times. Three changes make that
one-sweep tangent contracting (measured on CORE2, docs/PORTING_LESSONS.md I5e-I5g). In the
coupled CORE2 model (drift cost; 2013-01-01 and 07-01 at 2-30 d, 03-01 at 2-10 d) the gradient at
the default θ = 0.97 is ln P* 0.95-1.02 and ln C_d 0.99-1.00 of the full-model finite difference,
at the cost of the unrolled backward (I5h, I17, I19).

* Frozen at x_N. The exact tangent is linearized around the forward iterates, which in near-rigid
  elements form a non-converged, sign-flipping limit cycle (|λ| 6-18 per sweep). Changing the
  coefficients per sweep is unstable even with the two changes below.
* Raised pseudo-time coefficients. Per element α_t = max(α, α·sqrt(s/κ0)), where s = γ/(αβ) is
  the local mEVP stability number at x_N (:func:`fesom_jax.ice_mevp.mevp_stability_number`).
  β_t at a node is the maximum of α_t over the node's elements, as FESOM2's aEVP does. Then
  γ/(α_t β_t) ≤ κ0. The relaxation parameters change only the pseudo-time path, not the fixed
  point of the stress tangent, δσ = δS.
* Partial plastic softening θ. In the swept stress the replacement pressure P_r = PΔ/(Δ+δmin)
  takes stop_gradient(Δ). The viscosities ζ = P/(Δ+δmin) and η = ζ/e² take
  Δ_θ = sg(Δ) + θ·(Δ − sg(Δ)), so their derivative is scaled by θ while the value is unchanged.
  θ = 0 is the Picard (frozen-viscosity) tangent; θ = 1 is the symmetrized Newton tangent, which
  is unstable. The 1-day harness chain contracts up to θ = 0.95. In the coupled model the
  default θ = 0.97 is clean to 30 d and brings ln P* to the finite difference, while θ = 0.99
  overshoots it (I19). The P* dependence (through pressure_fac) stays live.

Transposed frozen iteration: with g(x, anchor, inputs) the modified sweep and
(J_x, J_a, J_in) its Jacobians at x_N, the step tangent is δx_{k+1} = J_x δx_k + J_a δa + J_in δin
from δx_0 = (δu^n, δσ^n), with δa = (δu^n, δv^n). Its adjoint: λ_0 = ȳ, λ_{k+1} = J_xᵀ λ_k, and
the anchor and input cotangents accumulate Σ_k J_aᵀλ_k and Σ_k J_inᵀλ_k. The entry velocity is
both the initial carry and the anchor, so its cotangent is λ_N[:2] + Σ J_aᵀλ_k.

Traced config fields (the Params controls put ``ice_pstar`` / ``ice_cd`` in place of
``cfg.pstar`` / ``cfg.cd_oce_ice``) are passed through the custom_vjp as explicit arguments and
re-inserted inside, so their cotangents come out. Every other traced value the solve reads (the
coastal mask, computed inside the model's jit) is an explicit argument too: a custom_vjp
backward must not close over a tracer of the enclosing trace. Under ``shard_map`` that includes
the local mesh and the halo maps (:func:`fesom_jax.halo.exchange_arrays`); on one device they
are constants and stay closed over, so the dense program does not change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .config import DENSITY_0
from .forcing import _safe_speed
from .halo import HaloCtx, exchange_arrays, exchange_pair
from .ice import IceConfig
from .ice_evp import _safe_sqrt, strain_rates, stress_div_scatter
from .ice_mevp import (MevpStatic, mevp_iterate, mevp_setup, mevp_stability_coef,
                       mevp_stability_number)
from .mesh import Mesh


def theta_sweep(cfg: IceConfig, mesh: Mesh, u_aux, v_aux, s11, s12, s22, st: MevpStatic,
                u_anchor, v_anchor, srfoce_u, srfoce_v, stress_ax, stress_ay, rdt, theta):
    """One mEVP sweep (:func:`fesom_jax.ice_mevp.mevp_iterate`) with the stress written as
    viscous part plus replacement pressure, S11 = ½ζ(ε1 + ε2/e²) − ½P_r, S22 = ½ζ(ε1 − ε2/e²) − ½P_r,
    S12 = ζ ε12/e² (det2 folded in pressure_fac, as there). P_r uses stop_gradient(Δ) and ζ uses Δ_θ.
    The values are the model's up to rounding, and only the derivative differs. ``cfg`` may carry
    per-element ``alpha_evp`` and per-node ``beta_evp`` arrays (the raised α_t, β_t)."""
    vale = cfg.vale
    eps11, eps22, eps12 = strain_rates(mesh, u_aux, v_aux)
    eps1 = eps11 + eps22
    eps2 = eps11 - eps22
    delta = _safe_sqrt(eps1 * eps1 + vale * (eps2 * eps2 + 4.0 * eps12 * eps12))
    dsg = jax.lax.stop_gradient(delta)
    d_theta = dsg + theta * (delta - dsg)
    zeta = st.pressure_fac / (d_theta + cfg.delta_min)
    p_r = st.pressure_fac * dsg / (dsg + cfg.delta_min)
    det1 = cfg.mevp_det1
    s12n = det1 * s12 + zeta * eps12 * vale
    s11n = det1 * s11 + 0.5 * zeta * (eps1 + eps2 * vale) - 0.5 * p_r
    s22n = det1 * s22 + 0.5 * zeta * (eps1 - eps2 * vale) - 0.5 * p_r
    s11 = jnp.where(st.ice_el, s11n, s11)
    s12 = jnp.where(st.ice_el, s12n, s12)
    s22 = jnp.where(st.ice_el, s22n, s22)
    u_rhs, v_rhs = stress_div_scatter(mesh, s11, s12, s22, st.ice_el)
    u_rhs = u_rhs * st.mass + st.tilt_u
    v_rhs = v_rhs * st.mass + st.tilt_v
    umod = _safe_speed(u_aux - srfoce_u, v_aux - srfoce_v)
    drag = rdt * cfg.cd_oce_ice * umod * DENSITY_0 * st.inv_thickness
    rhsu = u_anchor + drag * srfoce_u + rdt * (st.inv_thickness * stress_ax + u_rhs) \
        + cfg.beta_evp * u_aux
    rhsv = v_anchor + drag * srfoce_v + rdt * (st.inv_thickness * stress_ay + v_rhs) \
        + cfg.beta_evp * v_aux
    obd = 1.0 + cfg.beta_evp + drag
    rf = rdt * mesh.coriolis_node
    det = st.bc_index / (obd * obd + rf * rf)
    un = det * (obd * rhsu + rf * rhsv)
    vn = det * (obd * rhsv - rf * rhsu)
    u_aux = jnp.where(st.ice_nod, un, u_aux)
    v_aux = jnp.where(st.ice_nod, vn, v_aux)
    u_aux = jnp.where(st.bc_index > 0.0, u_aux, 0.0)
    v_aux = jnp.where(st.bc_index > 0.0, v_aux, 0.0)
    return u_aux, v_aux, s11, s12, s22


def raised_coefficients(cfg: IceConfig, mesh: Mesh, st: MevpStatic, m_ice, m_snow, u, v,
                        kappa0):
    """(α_t per element, β_t per node) at the iterate (u, v): α_t = max(α, α·sqrt(s/κ0)) on ice
    elements (α elsewhere), s = γ/(αβ); β_t = max of α_t over the node's elements (α at a node
    with none). Forward values only."""
    coef = mevp_stability_coef(cfg, mesh, m_ice, m_snow)
    s = mevp_stability_number(cfg, mesh, st, u, v, coef)
    a0 = cfg.alpha_evp
    al = jnp.where(st.ice_el, jnp.maximum(a0, a0 * jnp.sqrt(s / kappa0)), a0)
    be = jax.ops.segment_max(jnp.repeat(al, 3), jnp.asarray(mesh.elem_nodes).reshape(-1),
                             num_segments=int(mesh.nod2D))
    # A node with no element (a pad lane of a sharded mesh) gets segment_max's identity −inf; its
    # masked node solve then holds 0·(−inf) and the backward NaN (I18). α_t ≥ α, so the max with
    # α changes nothing at a node that has elements.
    be = jnp.maximum(be, a0)
    return jax.lax.stop_gradient(al), jax.lax.stop_gradient(be)


def _traced_fields(cfg: IceConfig):
    """Names of the config fields that hold jax values (traced controls), in field order."""
    return tuple(k for k, v in zip(cfg._fields, cfg) if isinstance(v, jax.Array))


def mevp_dynamics_frozen_sweep(cfg: IceConfig, mesh: Mesh, *, a_ice, m_ice, m_snow,
                               u_ice, v_ice, sigma11, sigma12, sigma22,
                               srfoce_u, srfoce_v, elevation, stress_ax, stress_ay,
                               boundary_node, exch=None):
    """:func:`fesom_jax.ice_mevp.mevp_dynamics` (all sweeps differentiated) with the frozen-sweep
    backward. The same outputs, and the same forward values bit for bit."""
    names = _traced_fields(cfg)
    cfg_static = cfg._replace(**{k: None for k in names})
    # Sharded: the local mesh and the halo maps are tracers of the shard_map body, explicit
    # arguments of the rule (`aux`); one device: aux = None and both stay closed over.
    hx, rebuild = exchange_arrays(exch)
    aux = (mesh, hx) if hx else None

    def unpack(aux_):
        return (mesh, exch) if aux_ is None else (aux_[0], rebuild(aux_[1]))
    n = int(cfg.evp_rheol_steps)
    kappa0 = float(cfg.adjoint_alpha_kappa)
    theta = float(cfg.adjoint_theta)
    guard = float(getattr(cfg, "adjoint_sweep_guard", 0.0))

    def with_traced(tv):
        return cfg_static._replace(**dict(zip(names, tv)))

    def forward(state, inputs, tv, bn, aux_):
        cfg_ = with_traced(tv)
        mesh, exch = unpack(aux_)
        u0, v0, s11, s12, s22 = state
        a, m, ms, elev, su, sv, sax, say = inputs
        st = mevp_setup(cfg_, mesh, a, m, ms, elev, bn)
        rdt = cfg_.ice_dt

        def body(carry, _):
            u_i, v_i, q11, q12, q22 = carry
            u_i, v_i, q11, q12, q22 = mevp_iterate(
                cfg_, mesh, u_i, v_i, q11, q12, q22, st, u0, v0, su, sv, sax, say, rdt)
            u_i, v_i = exchange_pair(exch, u_i, v_i, "nod")
            return (u_i, v_i, q11, q12, q22), None
        # jax.checkpoint as in mevp_dynamics, so the primal program is the same
        return lax.scan(jax.checkpoint(body), (u0, v0, s11, s12, s22), None, length=n)[0]

    @jax.custom_vjp
    def run(state, inputs, tv, bn, aux_):
        return forward(state, inputs, tv, bn, aux_)

    def run_fwd(state, inputs, tv, bn, aux_):
        y = forward(state, inputs, tv, bn, aux_)
        return y, (y, state[0], state[1], inputs, tv, bn, aux_)

    def run_bwd(res, ybar):
        xN, u0, v0, inputs, tv, bn, aux_ = res
        mesh, exch = unpack(aux_)
        cfg_ = with_traced(tv)
        a, m, ms, elev, su, sv, sax, say = inputs
        st_N = mevp_setup(cfg_, mesh, a, m, ms, elev, bn)
        al, be = raised_coefficients(cfg_, mesh, st_N, m, ms, xN[0], xN[1], kappa0)

        def g(x, ua, va, inputs_, tv_):
            c_ = with_traced(tv_)
            a_, m_, ms_, elev_, su_, sv_, sax_, say_ = inputs_
            st = mevp_setup(c_, mesh, a_, m_, ms_, elev_, bn)
            ct = c_._replace(alpha_evp=al, beta_evp=be)
            st = st._replace(pressure_fac=st.pressure_fac * (ct.mevp_det2 / c_.mevp_det2))
            y = theta_sweep(ct, mesh, *x, st, ua, va, su_, sv_, sax_, say_, c_.ice_dt, theta)
            u_, v_ = exchange_pair(exch, y[0], y[1], "nod")
            return (u_, v_) + tuple(y[2:])

        _, g_vjp = jax.vjp(g, xN, u0, v0, inputs, tv)
        zeros = lambda t: jax.tree.map(jnp.zeros_like, t)        # noqa: E731

        def body(carry, _):
            lam, acc_u, acc_v, acc_in, acc_tv = carry
            l_x, l_u, l_v, l_in, l_tv = g_vjp(lam)
            add = lambda p, q: jax.tree.map(jnp.add, p, q)      # noqa: E731
            return (l_x, acc_u + l_u, acc_v + l_v, add(acc_in, l_in), add(acc_tv, l_tv)), None
        init = (tuple(ybar), jnp.zeros_like(u0), jnp.zeros_like(v0), zeros(inputs), zeros(tv))
        if guard <= 0.0:
            lam, acc_u, acc_v, acc_in, acc_tv = lax.scan(body, init, None, length=n)[0]
            state_bar = (lam[0] + acc_u, lam[1] + acc_v, lam[2], lam[3], lam[4])
            return state_bar, acc_in, acc_tv, _zero_cot(bn), jax.tree.map(_zero_cot, aux_)

        # IceConfig.adjoint_sweep_guard: the largest |λ_k|² of the transposed sweeps against
        # |λ_0|² = |ybar|², over all devices; past guard² (or not finite) the step's mEVP solve
        # enters the tangent as a constant (zero cotangents out).
        def sq(t):
            return sum(jnp.sum(jnp.square(x)) for x in t)

        def body_g(carry, x):
            inner, mx = carry
            inner, _ = body(inner, x)
            return (inner, jnp.maximum(mx, sq(inner[0]))), None
        (lam, acc_u, acc_v, acc_in, acc_tv), mx = lax.scan(
            body_g, (init, jnp.zeros((), u0.dtype)), None, length=n)[0]
        n0 = sq(tuple(ybar))
        ctx = getattr(exch, "__self__", None)
        if isinstance(ctx, HaloCtx):
            n0, mx = lax.psum(n0, ctx.axis_name), lax.pmax(mx, ctx.axis_name)
        ok = mx <= (guard * guard) * n0
        lax.cond(ok, lambda: None, lambda: jax.debug.print(
            "frozen_sweep guard: mEVP derivative of this ice step dropped, max|lam_k|/|lam_0| "
            "{r:.3e}", r=jnp.sqrt(mx / jnp.maximum(n0, 1e-300))))
        keep = lambda t: jax.tree.map(lambda x: jnp.where(ok, x, jnp.zeros_like(x)), t)  # noqa: E731
        state_bar = keep((lam[0] + acc_u, lam[1] + acc_v, lam[2], lam[3], lam[4]))
        return state_bar, keep(acc_in), keep(acc_tv), _zero_cot(bn), jax.tree.map(_zero_cot, aux_)

    run.defvjp(run_fwd, run_bwd)
    tv = tuple(getattr(cfg, k) for k in names)
    inputs = (a_ice, m_ice, m_snow, elevation, srfoce_u, srfoce_v, stress_ax, stress_ay)
    return run((u_ice, v_ice, sigma11, sigma12, sigma22), inputs, tv, boundary_node, aux)


def _zero_cot(x):
    """The zero cotangent of a constant argument (``float0`` for an integer or bool one)."""
    if jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(np.shape(x), jax.dtypes.float0)
