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

from . import (ale, eos, gm, gm_redi, halo, kpp, momentum, pgf, pp, ssh, tke,
               tracer_adv, tracer_diff)
from .config import DENSITY_0, DT_DEFAULT
from .mesh import Mesh
from .params import Params
from .ssh import SSHOperator
from .state import State

# Salinity floor S = max(S, 0.5) on wet layers (fesom_step.c:382-393) — a stability
# clamp for confined brackish seas; a no-op while S ≈ 35, needed for CORE2.
S_FLOOR = 0.5


def step(state: State, mesh: Mesh, op: SSHOperator, stress_surf, params: Params = None,
         *, dt: float = DT_DEFAULT, is_first_step: bool = False,
         step_forcing=None, forcing_static=None, ice_cfg=None, gm_cfg=None,
         kpp_cfg=None, tke_cfg=None, ale_cfg=None, halo_ctx=None,
         boundary_node=None) -> State:
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
    Phase 2 (the 313 pi gates must not move).

    **GM/Redi (Phase 6B).** ``gm_cfg`` (a :class:`fesom_jax.gm.GMConfig`, static/hashable —
    the ``ice_cfg`` precedent) turns on the mesoscale eddy parameterization: after EOS the
    GM coefficient/bolus block (:func:`fesom_jax.gm.gm_diagnostics`) builds the eddy bolus
    velocity ``fer_uv`` + the Redi diffusivities ``slope_tapered``/``Ki``; the bolus
    augments the tracer-advecting velocity (``uv+fer_uv``, ``w_e+fer_w``), the Redi explicit
    terms (G7a vertical + G7b horizontal) add to the post-advection tracer, and the K33
    isoneutral term augments ``Kv`` in the vertical diffusion. GM/Redi is **stateless** (no
    new ``State`` fields — recomputed each step from T/S/N²). The differentiable ceilings
    ``k_gm``/``redi_kmax`` enter via ``params`` (the 2nd ML-hook seam). ``gm_cfg=None`` ⇒
    a dead branch (no trace) ⇒ the pi/Phase-5/ice path is **bit-identical**.

    **KPP (Phase 6C).** ``kpp_cfg`` (a :class:`fesom_jax.kpp.KppConfig`, static/hashable —
    the ``gm_cfg``/``ice_cfg`` precedent) selects the **K-Profile Parameterization** — the
    *real* FESOM2 CORE2 default vertical mixing — in place of PP at the mixing seam
    (substep 4): KPP recomputes the OBL profile each step from N²/forcing and emits the same
    ``(Kv, Av)`` PP does. Like GM/Redi it is **stateless** (no new ``State`` fields) and, like
    ice, a **CORE2 forced-path feature** (it needs ``heat_flux``/``water_flux``/wind stress →
    ``ustar``/``Bo`` → the OBL depth, so it raises on the pi path). It recomputes the OBL
    profile each step (:func:`fesom_jax.kpp.mixing_kpp`) and emits the same ``(Kv, Av)`` PP
    does → ``Kv`` feeds the tracer vertical diffusion (still augmented by GM's K33 when
    ``gm_cfg`` is on) and ``Av`` the momentum viscosity. ``kpp_cfg=None`` ⇒ the existing PP
    branch, **byte-identical** (a dead branch — no trace).

    **TKE (Phase 9b).** ``tke_cfg`` (a :class:`fesom_jax.tke.TkeConfig`, static/hashable —
    the ``kpp_cfg`` precedent) selects the **classical-TKE** prognostic mixing closure in
    place of PP/KPP at the mixing seam (substep 4): unlike PP/KPP/GM (all stateless), TKE
    carries ``State.tke`` (interface-indexed turbulent kinetic energy) across steps — it
    joins both ``State`` and the ``lax.scan`` carry. It emits the same ``(Kv, Av, uvnode)``
    PP/KPP do PLUS the advanced ``tke_new`` (written via a conditional ``replace`` so the
    ``None`` path never touches ``state.tke``). Like KPP it is a **forced-path feature**
    (the surface flux ``cd·|stress|^{3/2}`` needs ``stress_node_surf``) so it raises on the
    pi path, and it is an **error to set both ``kpp_cfg`` and ``tke_cfg``** (the C runs
    exactly one mixing scheme per process — fail loudly). The trainable constants
    ``tke_c_k``/``tke_c_eps``/``tke_cd``/``tke_alpha`` enter via ``params`` (the PRIMARY
    ML-hook seam). ``tke_cfg=None`` ⇒ the KPP/PP branch, **byte-identical** (a dead branch).

    **zstar ALE (Phase 9a).** ``ale_cfg`` (an :class:`fesom_jax.ale.AleConfig`, static/hashable —
    the ``gm_cfg``/``kpp_cfg``/``ice_cfg`` precedent) selects the **zstar vertical coordinate**
    (``which_ALE='zstar'``): the SSH change is distributed over the water column each step, so
    layer thicknesses (``hnode``/``helem``/``zbar_3d_n``/``Z_3d_n``) become time-varying and the
    forcing flips to real freshwater/salt fluxes. ``ale_cfg=None`` ⇒ the linfs path
    (``dh/dt=0``), **byte-identical** to the pre-Phase-9 model (a dead branch — no trace).
    Validated at the seam (the C ``fesom_ale_mode_init`` abort parity)."""
    st = state
    if params is None:
        params = Params.defaults()
    # zstar seam (Phase 9a): None ⇒ linfs (the byte-identical dead branch). A non-None
    # AleConfig means zstar — validate the mode here (the C's fesom_ale_mode_init exit(1)
    # parity; static arg ⇒ this is a trace-time Python guard, no runtime cost).
    if ale_cfg is not None:
        ale_cfg.validate()
    # zstar ⇒ real freshwater/salt fluxes (use_virt_salt=False, is_nonlinfs=1); linfs ⇒
    # virtual salt (use_virt_salt=True, is_nonlinfs=0). Drives the forcing flip + the bc terms.
    use_virt_salt = True if ale_cfg is None else ale_cfg.use_virt_salt

    # Phase 8 (S.7): the halo-exchange closure. ``halo_ctx=None`` ⇒ the identity ⇒ the
    # dead branch is not traced ⇒ byte-identical ``v1.0``; a :class:`~fesom_jax.halo.HaloCtx`
    # (built inside ``shard_map``) inserts the broadcast halo refreshes at the C's
    # exchange points (``halo_points.OCEAN_SCHEDULE``) + the CG's ``SSHHalo`` (S.6).
    if halo_ctx is None:
        _exch = lambda f, kind: f                       # noqa: E731
        _ssh_halo = None
        _red_mask = None                                # reductions: plain masked sum
        _red_axis = None
    else:
        _exch = halo_ctx.exchange
        _ssh_halo = halo_ctx.ssh_halo
        # distributed reductions (S.7 part 3): owned-node sum + psum. owned_mask=None ⇒
        # the dense path is byte-identical (plain jnp.sum), so this is a dead-branch gate.
        _red_mask = halo_ctx.owned_mask.get("nod")
        _red_axis = halo_ctx.axis_name

    # zstar live geometry (D1): the per-node zbar_3d_n/Z_3d_n + the per-element zbar_n/Z_n,
    # both derived from the carried (pre-commit) st.hnode/st.helem. Hoisted ONCE here — BEFORE
    # the forcing block, whose shortwave-penetration reads zbar3_live (JZ.6) — and reused by the
    # shchepetkin PGF (JZ.5), vert_vel distribute (JZ.4), and the JZ.6 consumer re-points
    # (EOS/PP/dbsfc/KPP/GM/QR4C/vert-Redi/momentum). At cold start (hbar=0) live == static;
    # linfs ⇒ all None (the consumers keep the static mesh geometry ⇒ byte-identical).
    zbar3_live = Z3d_live = elem_geo_live = None
    if ale_cfg is not None:
        zbar3_live, Z3d_live = ale.live_geometry(mesh, st.hnode)
        elem_geo_live = ale.live_geometry_elem(mesh, st.helem)

    # CORE2 surface forcing (None ⇒ pi path: keep the passed stress_surf, zero BCs).
    # ``ice_cfg`` (an IceConfig) ⇒ the Phase-6 PROGNOSTIC sea-ice step (ocean2ice → EVP → FCT
    # → cut_off → thermo → oce_fluxes → stress blend → shortwave); else the Phase-5 static-ice
    # surface fluxes. Both produce stress_surf + bc_T/bc_S/sw_3d for the ocean step below.
    bc_T = bc_S = sw_3d = None
    # KPP surface-forcing inputs (Phase 6C; None on the pi path ⇒ KPP unavailable, by design).
    heat_flux = water_flux = stress_node_surf = None
    ice_out = None
    if step_forcing is not None:
        if ice_cfg is not None:
            from . import ice_step as _ice_step       # lazy: keep ice deps off the pi path
            ice_out = _ice_step.ice_surface_step(
                ice_cfg, mesh, st, step_forcing, forcing_static, dt=dt,
                owned_mask=_red_mask, axis_name=_red_axis, exch=_exch,
                boundary_node=boundary_node, use_virt_salt=use_virt_salt, zbar3=zbar3_live)
            stress_surf = ice_out.stress_surf
            bc_T, bc_S, sw_3d = ice_out.bc_T, ice_out.bc_S, ice_out.sw_3d
            heat_flux, water_flux = ice_out.heat_flux, ice_out.water_flux
            stress_node_surf = ice_out.stress_node_surf
        else:
            from . import core2_forcing            # lazy: keep netCDF deps off the pi path
            sfx = core2_forcing.compute_surface_fluxes(
                mesh, st, step_forcing, forcing_static, dt=dt,
                owned_mask=_red_mask, axis_name=_red_axis, use_virt_salt=use_virt_salt,
                zbar3=zbar3_live)
            stress_surf = sfx.stress_surf
            bc_T, bc_S, sw_3d = sfx.bc_T, sfx.bc_S, sfx.sw_3d
            heat_flux, water_flux = sfx.heat_flux, sfx.water_flux
            stress_node_surf = sfx.stress_node_surf

    # 1 — EOS / hydrostatic pressure / N² (zstar: density compressibility + N² spacing on
    #     live geometry; hpressure unused under zstar). Z3d_live=None ⇒ static (byte-identical).
    density, hpressure, bvfreq = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode, Z3d=Z3d_live)
    density = _exch(density, "nod")
    hpressure = _exch(hpressure, "nod")
    bvfreq = _exch(bvfreq, "nod")

    # ALE thickness is static in full-cell linfs (hnode_new == hnode, a memcpy). Compute
    # it once here so the GM coefficient block (substep 2) and the Redi reconstruction
    # (substep 15) share it; the pi/Phase-5 None path is unchanged (a pure hoist).
    hnode_new = ale.thickness_linfs(st.hnode)

    # 2 — GM/Redi coefficient + bolus block (fesom_step.c:227-231, after EOS). gm_cfg=None
    #     ⇒ skipped (a dead branch ⇒ the path is bit-identical). When a GMConfig: the eddy
    #     bolus velocity fer_uv + the Redi diffusivities slope_tapered/Ki (k_gm/redi_kmax
    #     from params = the 2nd ML-hook). GM/Redi is stateless ⇒ recomputed from T/S/N².
    gm_diag = None
    if gm_cfg is not None:
        gm_diag = gm.gm_diagnostics(mesh, st.T, st.S, bvfreq, hnode_new, st.helem,
                                    params, gm_cfg, exch=_exch,
                                    Z3d=Z3d_live, zbar3=zbar3_live)

    # 3 — pressure-gradient force. zstar ⇒ the shchepetkin density-Jacobian on live geometry
    #     (NO hpressure under zstar — the C uses none); linfs ⇒ the hpressure gradient (byte-identical).
    if ale_cfg is not None:
        pgf_x, pgf_y = pgf.pressure_force_shchepetkin(mesh, density - DENSITY_0, Z3d_live, st.helem)
    else:
        pgf_x, pgf_y = pgf.pressure_force_linfs(mesh, hpressure)
    pgf_x = _exch(pgf_x, "elem")
    pgf_y = _exch(pgf_y, "elem")

    # 4 — vertical mixing. 3-way: tke_cfg ⇒ classical-TKE (the PROGNOSTIC closure, Phase 9b);
    #     elif kpp_cfg ⇒ KPP (the real CORE2 default); else PP (k_ver/a_ver are the ML-hook
    #     seam). All emit (Kv, Av, uvnode) post-mo_convect (the shared convective adjustment,
    #     fesom_step.c:264) ⇒ a drop-in; TKE additionally returns tke_new. KPP & TKE are
    #     forced-path features (need ustar/Bo resp. |stress|) ⇒ they require CORE2 surface
    #     forcing (locked decision 7); and the C runs ONE scheme per process ⇒ both-set is an
    #     error. tke_new=None on the non-TKE branches (the conditional replace below skips it).
    tke_new = None
    if tke_cfg is not None:
        if kpp_cfg is not None:
            raise ValueError(
                "kpp_cfg and tke_cfg are both set — the C runs exactly one mixing scheme "
                "per process (KPP xor TKE). Pass only one of them.")
        tke_cfg.validate()                       # the C fesom_tke_alloc:247-253 abort parity
        if stress_node_surf is None:
            raise ValueError(
                "TKE (tke_cfg) requires CORE2 surface forcing (step_forcing): the surface "
                "flux cd·|stress_node_surf|^{3/2} needs the ice-blended nodal wind stress. "
                "The pi analytical path has no surface forcing — keep tke_cfg=None there.")
        Kv, Av, uvnode, tke_new = tke.mixing_tke(
            mesh, st.uv, bvfreq, st.tke, stress_node_surf, st.hnode, tke_cfg, params,
            exch=_exch, Z3d=Z3d_live, zbar3=zbar3_live)
    elif kpp_cfg is not None:
        if heat_flux is None:
            raise ValueError(
                "KPP (kpp_cfg) requires CORE2 surface forcing (step_forcing): it needs "
                "heat_flux/water_flux/stress_node_surf/sw_3d for ustar/Bo/bfsfc. The pi "
                "analytical path has no surface forcing — keep kpp_cfg=None there.")
        sw_alpha, sw_beta = eos.compute_sw_alpha_beta(mesh, st.T, st.S, Z3d=Z3d_live)
        dbsfc = eos.compute_dbsfc(mesh, st.T, st.S, Z3d=Z3d_live)
        Kv, Av, uvnode = kpp.mixing_kpp(
            mesh, st.uv, bvfreq, dbsfc, sw_alpha, sw_beta, st.S,
            heat_flux, water_flux, stress_node_surf, sw_3d, st.hnode, kpp_cfg, exch=_exch,
            Z3d=Z3d_live, zbar3=zbar3_live)
    else:
        Kv, Av, uvnode = pp.mixing_pp(mesh, st.uv, bvfreq,
                                      k_ver=params.k_ver, a_ver=params.a_ver, Z3d=Z3d_live)
    uvnode = _exch(uvnode, "nod")
    Kv = _exch(Kv, "nod")
    Av = _exch(Av, "elem")

    # 5 — momentum RHS (lagged eta_n, w_e, uv; shifts the OLD AB slot). Embeds
    #     momentum_adv_scalar, which exchanges its node uvnode_rhs internally (S.7).
    uv_rhs, uv_rhsAB = momentum.compute_vel_rhs(
        mesh, st.uv, st.uv_rhsAB, st.eta_n, pgf_x, pgf_y, st.w_e, st.hnode,
        is_first_step=is_first_step, dt=dt, exch=_exch)
    uv_rhs = _exch(uv_rhs, "elem")
    uv_rhsAB = _exch(uv_rhsAB, "elem")

    # 6 — biharmonic horizontal viscosity (splits internally to exchange u_b/v_b)
    uv_rhs = momentum.visc_filt_bidiff(mesh, st.uv, uv_rhs, dt=dt, exch=_exch)
    uv_rhs = _exch(uv_rhs, "elem")

    # 7 — implicit vertical viscosity → increment du (the C keeps it in uv_rhs). zstar (JZ.6):
    #     the per-element tridiagonal geometry comes from the live st.helem stack (elem_geo_live);
    #     None ⇒ static column-uniform zbar/Z (byte-identical).
    du = momentum.impl_vert_visc(mesh, st.uv, uv_rhs, Av, stress_surf, dt=dt,
                                 elem_geo=elem_geo_live)
    du = _exch(du, "elem")

    # zstar (Phase 9a, JZ.3): the SSH plumbing gains the real-freshwater tail (ssh_rhs +
    # compute_hbar) and the D2 stiffness-as-function-of-hbar increment (the solve matvec).
    # linfs (ale_cfg=None) ⇒ both None ⇒ the dead branch ⇒ byte-identical.
    wf_ssh = water_flux if ale_cfg is not None else None
    hbar_ssh = st.hbar if ale_cfg is not None else None

    # 8 — SSH RHS (transport divergence of uv + du; zstar adds −α·wf·areasvol)
    ssh_rhs = ssh.compute_ssh_rhs(mesh, st.uv, du, st.helem, water_flux=wf_ssh)
    ssh_rhs = _exch(ssh_rhs, "nod")

    # 9 — CG solve, warm-started from the previous step's d_eta (0 at step 1). The CG
    #     exchanges pp/rr internally per iteration (S.6, via _ssh_halo); refresh d_eta after.
    #     zstar: mesh+hbar add ΔA(mean₃(hbar)) to the matvec (no-op at cold start).
    d_eta = ssh.solve_ssh(op, ssh_rhs, x0=st.d_eta, halo=_ssh_halo,
                          mesh=mesh, hbar=hbar_ssh, dt=dt)
    d_eta = _exch(d_eta, "nod")

    # 10 — velocity update (du + barotropic SSH-gradient correction)
    uv = momentum.update_vel(mesh, st.uv, du, d_eta, dt=dt)
    uv = _exch(uv, "elem")

    # 11 — hbar (save hbar_old first); 12 — eta_n blend. zstar: compute_hbar subtracts
    #      wf·areasvol from ssh_rhs_old before the hbar update (so hbar + the next step's
    #      (1−α) term read the wf-modified value).
    hbar_old = st.hbar
    ssh_rhs_old, hbar = ssh.compute_hbar(mesh, uv, st.helem, hbar_old, dt=dt,
                                         water_flux=wf_ssh)
    ssh_rhs_old = _exch(ssh_rhs_old, "nod")
    hbar = _exch(hbar, "nod")
    eta_n = ssh.eta_n_update(mesh, st.eta_n, hbar, hbar_old)
    eta_n = _exch(eta_n, "nod")

    # 13 — ALE: vertical velocity; vertical CFL + explicit/implicit split. use_wsplit=0 in the
    #      reference config ⇒ the split is the identity (w_e=w, w_i=0); cfl_z is computed every
    #      step as in the C (populates State).
    w = ale.compute_w(mesh, uv, st.helem)
    # zstar (Phase 9a, JZ.4): distribute the SSH change over the stretched column (correcting w)
    #      and produce the live hnode_new from the carried (pre-commit) geometry. This OVERRIDES
    #      the hoisted linfs hnode_new — so substeps 15/16 (tracers/Redi/impl-diff/commit) read the
    #      new thickness, while the GM coefficient block (substep 2) already used the OLD committed
    #      st.hnode (the dual-geometry, lesson #6). linfs ⇒ hnode_new stays the hoist (byte-identical).
    if ale_cfg is not None:
        w, hnode_new = ale.vert_vel_zstar_distribute(
            mesh, w, st.hnode, zbar3_live, hbar, hbar_old, wf_ssh, dt=dt)
        hnode_new = _exch(hnode_new, "nod")    # the C's zstar-only exchange_nod(hnode_new)
    w = _exch(w, "nod")
    cfl_z = ale.compute_cfl_z(mesh, w, hnode_new, dt=dt)
    cfl_z = _exch(cfl_z, "nod")
    w_e, w_i = ale.compute_wvel_split(mesh, w, cfl_z)
    w_e = _exch(w_e, "nod")
    w_i = _exch(w_i, "nod")

    # zstar (JZ.6): the about-to-commit ("new") live geometry from hnode_new — the dual-geometry
    # "new" side. Used by the Redi K33 augmentation + the impl vert diff layer-center spacings
    # (the C builds zbar_n/Z_n from hnode_new, tracer_diff.c:134-158). linfs ⇒ None (static, byte-id).
    zbar3_new = Z3d_new = None
    if ale_cfg is not None:
        zbar3_new, Z3d_new = ale.live_geometry(mesh, hnode_new)

    # 13a — GM bolus wrap (fesom_step.c:422-438): feed the bolus-augmented velocity to the
    #       tracer advection. fer_w reuses compute_w driven by fer_uv. In functional JAX the
    #       carried uv/w_e are untouched, so the C's post-diffusion subtract-back is automatic.
    uv_adv, w_e_adv = uv, w_e
    if gm_diag is not None:
        fer_w = ale.compute_w(mesh, gm_diag.fer_uv, st.helem)
        fer_w = _exch(fer_w, "nod")    # compute_w is an edge→node scatter ⇒ refresh halo
        uv_adv = uv + gm_diag.fer_uv   # uv (exch elem) + fer_uv (exch elem) ⇒ halo-complete
        w_e_adv = w_e + fer_w          # w_e (exch nod) + fer_w (exch nod) ⇒ halo-complete

    # 15 — FCT tracer advection (T then S) + Redi explicit terms + implicit vert diffusion.
    #      advect_one_fct exchanges fct_LO + fct_plus/minus internally (S.7 splits).
    T_adv, T_old = tracer_adv.advect_one_fct(mesh, uv_adv, w_e_adv, st.helem, st.hnode,
                                             hnode_new, st.T, st.T_old, dt=dt, exch=_exch,
                                             Z3d=Z3d_live, zbar3=zbar3_live)
    S_adv, S_old = tracer_adv.advect_one_fct(mesh, uv_adv, w_e_adv, st.helem, st.hnode,
                                             hnode_new, st.S, st.S_old, dt=dt, exch=_exch,
                                             Z3d=Z3d_live, zbar3=zbar3_live)
    T_adv = _exch(T_adv, "nod")
    S_adv = _exch(S_adv, "nod")

    # 15a/b — Redi neutral diffusion (fesom_step.c:468-499). The G7a (vertical-explicit) +
    #         G7b (horizontal-edge) gradients read the PRE-step tracer (st.T/st.S = the C
    #         `valuesold` saved during advection), and the deltas apply to the post-advection
    #         T_adv/S_adv. K33 augments Kv before the vertical diffusion (no diffusion-kernel
    #         change — impl_vert_diff already builds a∝Kv[nz], c∝Kv[nz+1]).
    Kv_eff = Kv
    if gm_diag is not None:
        slope_tap, Ki = gm_diag.slope_tapered, gm_diag.Ki
        # zstar (JZ.6): vertical Redi geometry on the OLD (st.hnode) side (zbar3/Z3d_live, the ÷
        # hnode_new divisor stays the new side); horizontal Redi already on st.hnode/helem/hnode_new
        # (no static geom); K33 on the NEW (hnode_new) side (zbar3/Z3d_new — matches impl_vert_diff).
        T_adv = (T_adv
                 + gm_redi.diff_ver_part_redi_expl(mesh, st.T, slope_tap, Ki, hnode_new, dt=dt,
                                                   zbar3=zbar3_live, Z3d=Z3d_live)
                 + gm_redi.diff_part_hor_redi(mesh, st.T, slope_tap, Ki, st.hnode, hnode_new,
                                              st.helem, dt=dt))
        S_adv = (S_adv
                 + gm_redi.diff_ver_part_redi_expl(mesh, st.S, slope_tap, Ki, hnode_new, dt=dt,
                                                   zbar3=zbar3_live, Z3d=Z3d_live)
                 + gm_redi.diff_part_hor_redi(mesh, st.S, slope_tap, Ki, st.hnode, hnode_new,
                                              st.helem, dt=dt))
        Kv_eff = Kv + gm_redi.k33_augmentation(mesh, slope_tap, Ki, zbar3=zbar3_new, Z3d=Z3d_new)
    # zstar bc_T surface term −dt·sval·water_flux (is_nonlinfs=1): sval = the POST-advection
    # (+Redi) surface T, the C's `trarr[surface]` at the diffusion (fesom_tracer_diff.c:292) —
    # NOT the start-of-step T, so it lands HERE, not in the forcing-step bc_T. bc_S has no such
    # term (sign-trap lesson #3). linfs ⇒ skipped (byte-identical; ale_cfg=None dead branch).
    if ale_cfg is not None and bc_T is not None:
        bc_T = bc_T - dt * T_adv[:, 0] * water_flux
    # zstar: the impl vert diff layer-center spacings come from the NEW (about-to-commit)
    # thickness hnode_new (Z3d_new, hoisted after vert_vel above) — the dual-geometry's "new"
    # side (vs QR4C on the committed st.hnode). linfs ⇒ None (static).
    T_new, S_new = tracer_diff.impl_vert_diff(mesh, T_adv, S_adv, Kv_eff, hnode_new, dt=dt,
                                              bc_T=bc_T, bc_S=bc_S, sw_3d=sw_3d, Z3d=Z3d_new)
    # salinity floor on wet layers only (below-bottom stays 0)
    S_new = jnp.where(mesh.node_layer_mask, jnp.maximum(S_new, S_FLOOR), S_new)
    T_new = _exch(T_new, "nod")        # refresh halo for next step's EOS/FCT/Redi reads
    S_new = _exch(S_new, "nod")

    # 16 — commit thickness (hnode := hnode_new; helem = vertex mean)
    hnode, helem = ale.commit_thickness(mesh, hnode_new)
    hnode = _exch(hnode, "nod")
    helem = _exch(helem, "elem")

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
    # Phase 9b: carry the advanced prognostic TKE (a conditional replace keyed on the cfg —
    # the ice precedent — so the tke_cfg=None path never touches state.tke ⇒ byte-identical).
    if tke_cfg is not None:
        new = dataclasses.replace(new, tke=tke_new)
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
step_jit = jax.jit(step,
                   static_argnames=("dt", "is_first_step", "ice_cfg", "gm_cfg", "kpp_cfg",
                                    "tke_cfg", "ale_cfg"))


def run(state: State, mesh: Mesh, op: SSHOperator, stress_surf, n_steps: int,
        params: Params = None, *, dt: float = DT_DEFAULT,
        step_forcings=None, forcing_static=None, ice_cfg=None, gm_cfg=None,
        kpp_cfg=None, tke_cfg=None, ale_cfg=None) -> State:
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
                         forcing_static=forcing_static, ice_cfg=ice_cfg, gm_cfg=gm_cfg,
                         kpp_cfg=kpp_cfg, tke_cfg=tke_cfg, ale_cfg=ale_cfg)
    return state
