"""Single forward ocean step — the Phase-2 pi timestep (Task 2.11).

Wires the ported substep kernels into one ``step(state, mesh, op, stress_surf)``
mirroring the C driver ``fesom_step.c`` (linfs ALE, PP mixing, FCT tracers, CG SSH,
no GM/KPP/ice, analytical wind). This is **integration + state-threading**, not new
physics — every kernel is already dump-gated in its own module/test.

The substep order (Phase 2, ``gm=ice=0``):

    1  eos.compute_pressure_bv        (density, hpressure, bvfreq)
    3  pgf.pressure_force_linfs       (pgf_x, pgf_y)
    4  pp.mixing_pp                   (Kv, Av, uvnode)
    5  momentum.compute_vel_rhs       (uv_rhs, NEW uv_rhsAB)   ← reads lagged eta_n, w_e
    6  momentum.visc_filt_bidiff      (uv_rhs)
    7  momentum.impl_vert_visc        (du)
    8  ssh.compute_ssh_rhs            (ssh_rhs)                ← uses uv + du
    9  ssh.solve_ssh                  (d_eta)                  ← warm-start x0 = prev d_eta
    10 momentum.update_vel            (uv)
    11 ssh.compute_hbar               (ssh_rhs_old, hbar)      ← hbar_old = prev hbar
    12 ssh.eta_n_update               (eta_n)
    13 ale.thickness_linfs + compute_w + compute_cfl_z + wvel_split (use_wsplit=0 ⇒ w_e=w)
    15 tracer_adv.advect_one_fct ×2 + tracer_diff.impl_vert_diff + salinity floor
    16 ale.commit_thickness           (hnode, helem)

State threading that matters (verified by the multi-step dump gate, not step 1 alone):
the CG **warm-start** (``d_eta`` is never zeroed between steps), the **AB2** slots
(``uv_rhsAB`` for momentum, ``T_old``/``S_old`` for tracers), ``hbar_old`` saved before
``compute_hbar``, and the lagged ``eta_n``/``w_e`` feeding ``compute_vel_rhs``.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp

from . import ale, eos, momentum, pgf, pp, ssh, tracer_adv, tracer_diff
from .config import DT_DEFAULT
from .mesh import Mesh
from .params import Params
from .ssh import SSHOperator
from .state import State

# Salinity floor S = max(S, 0.5) on wet layers (fesom_step.c:382-393) — a stability
# clamp for confined brackish seas; a no-op while S ≈ 35, needed for CORE2.
S_FLOOR = 0.5


def step(state: State, mesh: Mesh, op: SSHOperator, stress_surf, params: Params = None,
         *, dt: float = DT_DEFAULT, is_first_step: bool = False,
         step_forcing=None, forcing_static=None, ice_cfg=None) -> State:
    """Advance ``state`` one ocean timestep. ``op`` is the static linfs SSH operator
    (:func:`ssh.build_ssh_operator`, built once outside the loop); ``stress_surf`` is
    the element wind stress (:func:`forcing.surface_stress`, static analytical, or
    zeros for a rest test). ``is_first_step`` selects the AB2 first-step branch.

    ``params`` (a :class:`fesom_jax.params.Params`) carries the differentiable
    physics tunables (the PP backgrounds ``k_ver``/``a_ver``). ``None`` ⇒
    :meth:`Params.defaults` (the config constants — numerically identical to the
    Phase-2 path). Pass a ``Params`` with traced leaves to take ``d(loss)/d(param)``.

    **CORE2 forcing (Phase 5).** Pass ``step_forcing`` (a
    :class:`fesom_jax.core2_forcing.StepForcing`, this step's atmosphere + SSS/chl) and
    ``forcing_static`` (the :class:`~fesom_jax.core2_forcing.ForcingStatic` constants) to
    drive the **bulk + SSS/runoff + shortwave-penetration** surface BCs: the bulk
    ``stress_surf`` replaces the passed ``stress_surf`` (momentum), and the per-node
    ``bc_T``/``bc_S`` + ``sw_3d`` feed the tracer diffusion. The bulk taps the
    start-of-step ``state.T[:,0]`` (SST) and ``state.uvnode[:,0]`` (surface current), so
    the SST→flux / current→stress feedback is differentiable. ``step_forcing=None`` ⇒ the
    pi analytical path (static ``stress_surf``, zero surface BCs) — **bit-identical** to
    Phase 2 (the 313 pi gates must not move)."""
    st = state
    if params is None:
        params = Params.defaults()

    # CORE2 surface forcing (None ⇒ pi path: keep the passed stress_surf, zero BCs).
    # ``ice_cfg`` (an IceConfig) ⇒ the Phase-6 PROGNOSTIC sea-ice step (ocean2ice → EVP → FCT
    # → cut_off → thermo → oce_fluxes → stress blend → shortwave); else the Phase-5 static-ice
    # surface fluxes. Both produce stress_surf + bc_T/bc_S/sw_3d for the ocean step below.
    bc_T = bc_S = sw_3d = None
    ice_out = None
    if step_forcing is not None:
        if ice_cfg is not None:
            from . import ice_step as _ice_step       # lazy: keep ice deps off the pi path
            ice_out = _ice_step.ice_surface_step(
                ice_cfg, mesh, st, step_forcing, forcing_static, dt=dt)
            stress_surf = ice_out.stress_surf
            bc_T, bc_S, sw_3d = ice_out.bc_T, ice_out.bc_S, ice_out.sw_3d
        else:
            from . import core2_forcing            # lazy: keep netCDF deps off the pi path
            sfx = core2_forcing.compute_surface_fluxes(
                mesh, st, step_forcing, forcing_static, dt=dt)
            stress_surf = sfx.stress_surf
            bc_T, bc_S, sw_3d = sfx.bc_T, sfx.bc_S, sfx.sw_3d

    # 1 — EOS / hydrostatic pressure / N²
    density, hpressure, bvfreq = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode)

    # 3 — pressure-gradient force
    pgf_x, pgf_y = pgf.pressure_force_linfs(mesh, hpressure)

    # 4 — PP vertical mixing (k_ver/a_ver are the ML-hook seam)
    Kv, Av, uvnode = pp.mixing_pp(mesh, st.uv, bvfreq,
                                  k_ver=params.k_ver, a_ver=params.a_ver)

    # 5 — momentum RHS (lagged eta_n, w_e, uv; shifts the OLD AB slot)
    uv_rhs, uv_rhsAB = momentum.compute_vel_rhs(
        mesh, st.uv, st.uv_rhsAB, st.eta_n, pgf_x, pgf_y, st.w_e, st.hnode,
        is_first_step=is_first_step, dt=dt)

    # 6 — biharmonic horizontal viscosity
    uv_rhs = momentum.visc_filt_bidiff(mesh, st.uv, uv_rhs, dt=dt)

    # 7 — implicit vertical viscosity → increment du (the C keeps it in uv_rhs)
    du = momentum.impl_vert_visc(mesh, st.uv, uv_rhs, Av, stress_surf, dt=dt)

    # 8 — SSH RHS (transport divergence of uv + du)
    ssh_rhs = ssh.compute_ssh_rhs(mesh, st.uv, du, st.helem)

    # 9 — CG solve, warm-started from the previous step's d_eta (0 at step 1)
    d_eta = ssh.solve_ssh(op, ssh_rhs, x0=st.d_eta)

    # 10 — velocity update (du + barotropic SSH-gradient correction)
    uv = momentum.update_vel(mesh, st.uv, du, d_eta, dt=dt)

    # 11 — hbar (save hbar_old first); 12 — eta_n blend
    hbar_old = st.hbar
    ssh_rhs_old, hbar = ssh.compute_hbar(mesh, uv, st.helem, hbar_old, dt=dt)
    eta_n = ssh.eta_n_update(mesh, st.eta_n, hbar, hbar_old)

    # 13 — ALE: static hnode_new + vertical velocity; vertical CFL + explicit/implicit
    #      split. use_wsplit=0 in the pi reference config ⇒ the split is the identity
    #      (w_e=w, w_i=0); cfl_z is computed every step as in the C (populates State).
    hnode_new = ale.thickness_linfs(st.hnode)
    w = ale.compute_w(mesh, uv, st.helem)
    cfl_z = ale.compute_cfl_z(mesh, w, hnode_new, dt=dt)
    w_e, w_i = ale.compute_wvel_split(mesh, w, cfl_z)

    # 15 — FCT tracer advection (T then S) + implicit vertical diffusion
    T_adv, T_old = tracer_adv.advect_one_fct(mesh, uv, w_e, st.helem, st.hnode,
                                             hnode_new, st.T, st.T_old, dt=dt)
    S_adv, S_old = tracer_adv.advect_one_fct(mesh, uv, w_e, st.helem, st.hnode,
                                             hnode_new, st.S, st.S_old, dt=dt)
    T_new, S_new = tracer_diff.impl_vert_diff(mesh, T_adv, S_adv, Kv, hnode_new, dt=dt,
                                              bc_T=bc_T, bc_S=bc_S, sw_3d=sw_3d)
    # salinity floor on wet layers only (below-bottom stays 0)
    S_new = jnp.where(mesh.node_layer_mask, jnp.maximum(S_new, S_FLOOR), S_new)

    # 16 — commit thickness (hnode := hnode_new; helem = vertex mean)
    hnode, helem = ale.commit_thickness(mesh, hnode_new)

    new = dataclasses.replace(
        st,
        T=T_new, S=S_new, T_old=T_old, S_old=S_old,
        uv=uv, uv_rhs=du, uv_rhsAB=uv_rhsAB, uvnode=uvnode,
        w=w, w_e=w_e, w_i=w_i, cfl_z=cfl_z,
        eta_n=eta_n, d_eta=d_eta, ssh_rhs=ssh_rhs, ssh_rhs_old=ssh_rhs_old,
        hnode=hnode, hnode_new=hnode_new, helem=helem, hbar=hbar, hbar_old=hbar_old,
        density=density, hpressure=hpressure, bvfreq=bvfreq, Kv=Kv, Av=Av,
        pgf_x=pgf_x, pgf_y=pgf_y,
    )
    # Phase 6: carry the updated prognostic ice state (the ice step ran before the ocean step).
    if ice_out is not None:
        new = dataclasses.replace(
            new, a_ice=ice_out.a_ice, m_ice=ice_out.m_ice, m_snow=ice_out.m_snow,
            u_ice=ice_out.u_ice, v_ice=ice_out.v_ice, t_skin=ice_out.t_skin,
            sigma11=ice_out.sigma11, sigma12=ice_out.sigma12, sigma22=ice_out.sigma22)
    return new


# Jitted entry point (the Task-2.11 deliverable; also what Phase 3's lax.scan wraps).
# ``mesh``/``op``/``state``/``stress_surf`` are pytree args; ``dt``/``is_first_step``
# are static (the latter ⇒ two compiled variants: the step-1 AB2 branch and the rest).
step_jit = jax.jit(step, static_argnames=("dt", "is_first_step", "ice_cfg"))


def run(state: State, mesh: Mesh, op: SSHOperator, stress_surf, n_steps: int,
        params: Params = None, *, dt: float = DT_DEFAULT,
        step_forcings=None, forcing_static=None, ice_cfg=None) -> State:
    """Run ``n_steps`` jitted forward steps from ``state`` (a plain Python loop;
    Phase 3 adds :func:`fesom_jax.integrate.integrate`, a checkpointed ``lax.scan``,
    for the differentiable path). ``is_first_step`` is set on the first iteration
    only (the AB2 first-step branch).

    For CORE2, pass ``step_forcings`` (a :class:`~fesom_jax.core2_forcing.StepForcing`
    with leading axis ``[n_steps]``) + ``forcing_static``; step ``i`` consumes
    ``step_forcings[i]``. ``None`` ⇒ the pi analytical path."""
    for i in range(n_steps):
        sf = None if step_forcings is None else jax.tree.map(lambda x: x[i], step_forcings)
        state = step_jit(state, mesh, op, stress_surf, params, dt=dt,
                         is_first_step=(i == 0), step_forcing=sf,
                         forcing_static=forcing_static, ice_cfg=ice_cfg)
    return state
