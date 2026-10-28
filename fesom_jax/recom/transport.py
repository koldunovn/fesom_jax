"""The REcoM tracer stage: FESOM2's tracer loop for the BGC tracers (``solve_tracers_ale``, the IDs > 2).

The BGC tracers see exactly what T and S see in :func:`fesom_jax.step.step` (step 15): FCT advection with the same
bolus-augmented velocities and w_split, the explicit Redi terms from the pre-step tracer, the K33-augmented ``Kv`` in
the implicit vertical diffusion, the halo refresh after the advection and after the diffusion. They are carried as
one ``(nod2D, nl, ntr)`` stack (``State.bgc``) and run either one tracer at a time (``loop``, the reference) or as
one batched pass (``vmap``: every kernel and every halo collective once for the whole stack), chosen by
``gpu.bgc_layout`` (``FESOM_BGC_LAYOUT``, read at trace time).

Task 1 of the plan: the passive-tracer stage only. REcoM's own additions to the loop (the benthic return flux,
ballasting, sinking, benthos, the ``bc_surface`` cases) join it in Tasks 11-12.

``RecomConfig.bgc_every = k > 1`` (Task 2b, a stated deviation): :func:`bgc_every_k` sums the transport of each
physics step and runs the stage once per k steps with ``k·dt`` on the means (:func:`init_every_k` adds the sums).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from .. import ale, gm_redi, tracer_adv, tracer_diff

LAYOUTS = ("vmap", "loop", "major")
# "major" (the default): the vmap over the stack moved to tracer-major [ntr, entity, level] and held so in memory
# (a layout constraint), so the transport's gathers read each tracer's levels contiguously. "vmap" maps over the
# trailing axis of the entity-leading State layout and lets XLA place the tracer axis: with full REcoM the node
# fields stayed tracer-minor and every gather read the levels at a stride of ntr (CORE2 on one GH200: 1099 against
# 584 GB a step, jobs 28456722 / 28457954). "loop": one tracer at a time (the reference). Same arithmetic.
# The State fields one full-REcoM BGC step updates (fesom_jax.recom.wiring.full_bgc_step; + bgc_budget when carried)
FULL_STEP_FIELDS = ("bgc", "bgc_old", "bgc_benthos", "bgc_carb", "bgc_mstep", "bgc_diag")


def layout() -> str:
    """The tracer-axis layout of the BGC stage (``FESOM_BGC_LAYOUT``; default ``major``: CORE2 + full REcoM on 4 GH200
    119.8 -> 66 ms/step in DP, 76.6 -> 39.5 in SP together with the loop-term and limiter fixes, job 28458068)."""
    v = os.environ.get("FESOM_BGC_LAYOUT", "").strip().lower() or "major"
    if v not in LAYOUTS:
        raise ValueError(f"FESOM_BGC_LAYOUT must be one of {LAYOUTS}, got {v!r}")
    return v


def holds() -> bool:
    """Whether the BGC stacks are held tracer-major (:func:`hold_major`; layout ``major`` and not
    ``FESOM_BGC_HOLD=0``)."""
    return layout() == "major" and os.environ.get("FESOM_BGC_HOLD", "1").strip() != "0"


def hold_major(a):
    """``a`` (entity-leading, the tracer axis last) held tracer-major in memory, the buffer :func:`_tracer_major` makes:
    the move to the front is then free and full REcoM's stage 3 keeps ONE copy of a stack. Layout ``major`` only
    (``FESOM_BGC_HOLD=0`` leaves the placement to XLA). Without it the stage-1 stack was kept in both layouts and
    the T* fusion of the 23 non-detritus tracers read it at a stride of ntr: 17 ms of its 19 a step at 0.68 TB/s on
    one GH200 (job 28457954).

    Only on inputs and intermediates, never on a value a jit returns: a constraint at a jit's output sets the output's
    layout (JAX 0.10.1), and a second jit then reads that buffer as row-major, silently (CPU, the tracers permuted).
    The step's outputs stay unconstrained (a scan carry constrained inside its body comes back row-major)."""
    if not holds():
        return a
    from jax.experimental.layout import Layout, with_layout_constraint
    return with_layout_constraint(a, Layout(major_to_minor=(a.ndim - 1,) + tuple(range(a.ndim - 1))))


def init_passive(state, ntr: int):
    """``state`` with ``ntr`` passive tracers, each a copy of salinity (``bgc = bgc_old = S``, ``S_old``): realistic
    fronts for the FCT limiter to work on. Backend-agnostic (host numpy or jax, dense, ``[P, Lmax]`` or folded): the
    tracer axis is appended last, so the leaves stay entity-leading."""
    import dataclasses
    xp = np if isinstance(state.S, np.ndarray) else jnp
    rep = lambda a: xp.repeat(a[..., None], int(ntr), axis=-1)     # noqa: E731
    return dataclasses.replace(state, bgc=rep(state.S), bgc_old=rep(state.S_old))


def init_every_k(state, k: int):
    """``state`` with the BGC-every-k sums (``State.BGC_MR_FIELDS``) at the start of an interval: zero sums, a zero
    step counter, ``bgc_h0 = hnode``; with ``k == 1`` the fields are stripped (None). Fields already present (a
    restart written mid-interval) are kept, so the run resumes the interval. Backend-agnostic like
    :func:`init_passive` (``x * 0`` keeps a sharded leaf's sharding)."""
    import dataclasses
    from ..state import BGC_MR_FIELDS
    if int(k) == 1:
        return dataclasses.replace(state, **{f: None for f in BGC_MR_FIELDS})
    if state.bgc_steps is not None:
        return state
    z_n = state.hnode * 0
    return dataclasses.replace(
        state, bgc_steps=(state.hnode[..., 0] * 0).astype(np.int32), bgc_uvh=state.uv * 0, bgc_we=z_n, bgc_wi=z_n,
        bgc_kv=z_n, bgc_bc=state.hnode[..., 0] * 0, bgc_h0=state.hnode * 1)


def bgc_transport(mesh, X, X_old, *, uv_adv, w_e_adv, w_i, helem, hnode, hnode_new, zbar3_live, Z3d_live,
                  Z3d_new, gm_diag, Kv_eff, bc, dt, exch, ckpt, adj, sp_estimate: bool = False, helem_flux=None):
    """Advance the BGC stack ``X`` (``(nod2D, nl, ntr)``) through the tracer stage of one step. Returns
    ``(X_new, X_old_new)``, both ``(nod2D, nl, ntr)``, with ``X_old_new = X`` (the AB2 ``valuesold`` save).

    The keyword arguments are the step's own tracer-stage fields (step.py step 15): ``uv_adv``/``w_e_adv`` the
    bolus-augmented advecting velocities, ``w_i`` the w_split implicit part (``None`` without w_split), ``helem``/
    ``hnode`` the pre-step thicknesses, ``hnode_new`` the new ones, ``zbar3_live``/``Z3d_live`` and ``Z3d_new`` the
    zstar geometry on the old and new sides (``None`` under linfs), ``gm_diag`` the GM/Redi diagnostics (``None``
    without GM), ``Kv_eff`` the K33-augmented diffusivity, ``bc`` the surface flux (``None``, ``[nod2D]`` shared by
    every tracer, or ``[nod2D, ntr]``), ``exch``/``ckpt``/``adj`` the step's halo closure, remat wrapper and
    adjoint config. ``helem_flux`` (default ``helem``) is the thickness the FCT multiplies ``uv_adv`` by: the advection
    sees the velocity only through the edge volume flux ``(U·dy − V·dx)·helem`` (:func:`tracer_adv.edge_vflux`), so
    :func:`bgc_every_k` passes the summed ``uv·helem`` with ``helem_flux = 1``.

    ``sp_estimate`` (single precision, plan Task 2 only): the vertical diffusion runs its column in float64 as for
    T/S (``impl_vert_diff_one(carry=...)``) but the rounding residual is dropped instead of carried across steps:
    the cost of FESOM2's SP scheme, not its values (Task 16 ports the scheme)."""
    X_adv, X_old_new = bgc_advect_redi(
        mesh, X, X_old, uv_adv=uv_adv, w_e_adv=w_e_adv, w_i=w_i, helem=helem, hnode=hnode, hnode_new=hnode_new,
        zbar3_live=zbar3_live, Z3d_live=Z3d_live, gm_diag=gm_diag, dt=dt, exch=exch, ckpt=ckpt, adj=adj,
        helem_flux=helem_flux)
    X_new = bgc_diffuse(mesh, X_adv, Kv_eff=Kv_eff, hnode_new=hnode_new, Z3d_new=Z3d_new, bc=bc, dt=dt,
                        exch=exch, ckpt=ckpt, sp_estimate=sp_estimate)
    if sp_estimate:     # the WP State keeps its dtype (the AB2 save comes back in WP_full)
        X_new, X_old_new = X_new.astype(X.dtype), X_old_new.astype(X_old.dtype)
    return X_new, X_old_new


def _tracer_major(a):
    """``a`` with its trailing tracer axis moved to the front, held row-major (tracer-major) in memory."""
    from jax.experimental.layout import Layout, with_layout_constraint
    t = jnp.moveaxis(a, -1, 0)
    return with_layout_constraint(t, Layout(major_to_minor=tuple(range(t.ndim))))

def _over_stack(fn, arrays, axes):
    """``fn`` over the trailing tracer axis of ``arrays`` (``axes[i]`` = -1 for a stacked argument, None for a shared
    one): one tracer at a time (``loop``) or batched (``vmap``, ``major``), as :func:`layout` says. Returns the
    stacked outputs."""
    if layout() == "major":
        # a row-major layout constraint makes XLA write the tracer-major stack once. A bare axis move (job 28455642)
        # or one behind an optimization_barrier (job 28457487: layout {0,2,1}) became a bitcast of the tracer-minor
        # buffer, so every gather of the transport still read the levels at a stride of ntr
        lead = [_tracer_major(a) if ax is not None else a for a, ax in zip(arrays, axes)]
        out = jax.vmap(fn, in_axes=tuple(0 if ax is not None else None for ax in axes), out_axes=0)(*lead)
        return jax.tree.map(lambda o: jnp.moveaxis(o, 0, -1), out)
    if layout() == "loop":
        ntr = next(a.shape[-1] for a, ax in zip(arrays, axes) if ax is not None)
        outs = [fn(*[a[..., i] if ax is not None else a for a, ax in zip(arrays, axes)]) for i in range(ntr)]
        if isinstance(outs[0], tuple):
            return tuple(jnp.stack([o[k] for o in outs], axis=-1) for k in range(len(outs[0])))
        return jnp.stack(outs, axis=-1)
    return jax.vmap(fn, in_axes=tuple(axes), out_axes=-1)(*arrays)


def bgc_advect_redi(mesh, X, X_old, *, uv_adv, w_e_adv, w_i, helem, hnode, hnode_new, zbar3_live, Z3d_live,
                    gm_diag, dt, exch, ckpt, adj, helem_flux=None):
    """The first half of the BGC tracer stage: FCT advection (the T* reconstruction included) + the explicit Redi
    terms, halo-refreshed after the advection. Returns ``(X_adv, X_old_new)`` (``(nod2D, nl, ntr)``); arguments as
    :func:`bgc_transport`."""
    h_flux = helem if helem_flux is None else helem_flux

    def one(x, x_old):
        x_adv, x_old_new = ckpt(lambda: tracer_adv.advect_one_fct(
            mesh, uv_adv, w_e_adv, h_flux, hnode, hnode_new, x, x_old, dt=dt, exch=exch,
            Z3d=Z3d_live, zbar3=zbar3_live, w_i=w_i,
            freeze_limited_flux=adj.freeze_advection, freeze_limiter=adj.freeze_fct_limiter))()
        x_adv = exch(x_adv, "nod")
        if gm_diag is not None:
            # the Redi gradients read the PRE-step tracer (the `valuesold` of the advection), as for T/S
            x_adv = (x_adv
                     + ckpt(lambda: gm_redi.diff_ver_part_redi_expl(
                         mesh, x, gm_diag.slope_tapered, gm_diag.Ki, hnode_new, dt=dt,
                         zbar3=zbar3_live, Z3d=Z3d_live))()
                     + ckpt(lambda: gm_redi.diff_part_hor_redi(
                         mesh, x, gm_diag.slope_tapered, gm_diag.Ki, hnode, hnode_new, helem, dt=dt))())
        return x_adv, x_old_new

    return _over_stack(one, (X, X_old), (-1, -1))


def bgc_diffuse(mesh, X_adv, *, Kv_eff, hnode_new, Z3d_new, bc, dt, exch, ckpt, sp_estimate: bool = False):
    """The second half: the implicit vertical diffusion with the surface BC ``bc`` (``None``, ``[nod2D]`` shared,
    or ``[nod2D, ntr]``), halo-refreshed after. Returns ``(nod2D, nl, ntr)``."""
    def one(x_adv, bc1):
        if sp_estimate:
            x_new = ckpt(lambda: tracer_diff.impl_vert_diff_one(
                mesh, x_adv, Kv_eff, hnode_new, dt=dt, bc_surf=bc1, Z3d=Z3d_new, carry=jnp.zeros_like(x_adv)))()[0]
        else:
            x_new = ckpt(lambda: tracer_diff.impl_vert_diff_one(
                mesh, x_adv, Kv_eff, hnode_new, dt=dt, bc_surf=bc1, Z3d=Z3d_new))()
        return exch(x_new, "nod")

    per_tracer_bc = bc is not None and bc.ndim == 2
    return _over_stack(one, (X_adv, bc), (-1, -1 if per_tracer_bc else None))


def surface_bc(sfc, dt):
    """REcoM's ``bc_surface`` (``oce_ale_tracer.F90:2232-2290``) for the whole stack, ``(nod2D, ntr)``: ``dt`` times
    the stage-1 flux of :data:`fesom_jax.recom.tracers.BC_SURFACE_SOURCE` for the IDs that have one, 0 elsewhere."""
    from .tracers import BC_SURFACE_SOURCE, BGC_IDS
    ref = sfc.GloCO2flux_seaicemask
    cols = [dt * getattr(sfc, BC_SURFACE_SOURCE[i]) if BC_SURFACE_SOURCE.get(i, "zero") != "zero"
            else jnp.zeros_like(ref) for i in BGC_IDS]
    return jnp.stack(cols, axis=-1)


def recom_stage3(mesh, X1, X_old, *, uv_adv, w_e_adv, w_i, helem, hnode, hnode_new, zbar3_live, Z3d_live, Z3d_new,
                 gm_diag, Kv_eff, temp, salt, sfc, benthos, geom_sink, params, switches, dt, exch, ckpt, adj,
                 helem_flux=None, return_terms: bool = False, carry=None):
    """Full REcoM's stage 3: FESOM2's tracer loop for the 31 BGC tracers (``solve_tracers_ale``,
    ``oce_ale_tracer.F90:725-930``). ``X1`` is stage 1's output (the tracers at the start of the loop, ``(nod2D, nl,
    31)``), ``temp``/``salt`` the step's final T and S (S before ``S_FLOOR``), ``sfc``/``benthos`` stage 1's
    :class:`~fesom_jax.recom.forcing.SurfaceFluxes` and benthos, ``geom_sink`` the
    :class:`~fesom_jax.recom.sinking.SinkingGeometry` of this tracer step; the transport arguments as
    :func:`bgc_transport`.

    Per tracer, as Fortran: advection + explicit Redi (batched), then REcoM's increments in tracer order
    (:func:`fesom_jax.recom.sinking.recom_tracer_loop_terms`: the benthic return flux, ballasting from the current
    detritus, sinking, benthos), which enter through the T* update as ``Δ·hnode/hnode_new``, then the implicit vertical
    diffusion with :func:`surface_bc`. Returns ``(X_new, X_old_new, benthos_new)``, with ``return_terms`` also the
    :class:`~fesom_jax.recom.sinking.LoopTermsResult` (its ``values_mod`` give the export and budget diagnostics).

    ``carry`` (single precision, the BGC rounding carry ``State.bgc_carry``): :func:`_recom_stage3_sp`, the #1207
    scheme; the result tuple then ends with the new carry."""
    from . import sinking
    from .tracers import SLOT
    if carry is not None:
        if w_i is not None or helem_flux is not None:
            raise NotImplementedError("single precision + REcoM: no w_split, bgc_every = 1 only")
        return _recom_stage3_sp(mesh, X1, X_old, carry, uv_adv=uv_adv, w_e_adv=w_e_adv, helem=helem, hnode=hnode,
                                hnode_new=hnode_new, zbar3_live=zbar3_live, Z3d_live=Z3d_live, Z3d_new=Z3d_new,
                                gm_diag=gm_diag, Kv_eff=Kv_eff, temp=temp, salt=salt, sfc=sfc, benthos=benthos,
                                geom_sink=geom_sink, params=params, switches=switches, dt=dt, exch=exch, ckpt=ckpt,
                                adj=adj, return_terms=return_terms)
    nl1 = mesh.nl - 1
    X1 = hold_major(X1)             # the advection, the T* of every group and the loop terms read this one buffer
    X_adv, X_old_new = bgc_advect_redi(
        mesh, X1, X_old, uv_adv=uv_adv, w_e_adv=w_e_adv, w_i=w_i, helem=helem, hnode=hnode, hnode_new=hnode_new,
        zbar3_live=zbar3_live, Z3d_live=Z3d_live, gm_diag=gm_diag, dt=dt, exch=exch, ckpt=ckpt, adj=adj,
        helem_flux=helem_flux)
    bc = surface_bc(sfc, dt)
    ratio = jnp.where(hnode_new > 0.0, hnode / jnp.where(hnode_new > 0.0, hnode_new, 1.0), 0.0)[..., None]

    def finish(ids, values_mod):
        """T* with the modified values, then the implicit diffusion, for the tracers ``ids``."""
        idx = jnp.array([SLOT[t] for t in ids])
        vmod = jnp.concatenate([values_mod, X1[:, nl1:, idx]], axis=1)          # (nod2D, nl, n)
        x_star = hold_major(X_adv[..., idx] + (vmod - X1[..., idx]) * ratio)
        x_new = bgc_diffuse(mesh, x_star, Kv_eff=Kv_eff, hnode_new=hnode_new, Z3d_new=Z3d_new, bc=bc[:, idx],
                            dt=dt, exch=lambda f, kind: f, ckpt=ckpt)
        return x_new[:, :nl1]

    res = sinking.recom_tracer_loop_terms(X1[:, :nl1], temp[:, :nl1], salt[:, :nl1], geom_sink,
                                          sfc.GlodecayBenthos, benthos, dt, params, finish, switches=switches)
    X_new = exch(jnp.concatenate([res.bgc, X1[:, nl1:]], axis=1), "nod")
    if return_terms:
        return X_new, X_old_new, exch(res.benthos, "nod"), res
    return X_new, X_old_new, exch(res.benthos, "nod")


def _recom_stage3_sp(mesh, X1, X_old, carry, *, uv_adv, w_e_adv, helem, hnode, hnode_new, zbar3_live, Z3d_live,
                     Z3d_new, gm_diag, Kv_eff, temp, salt, sfc, benthos, geom_sink, params, switches, dt, exch, ckpt,
                     adj, return_terms):
    """Stage 3 in single precision + full REcoM (an opt-in deviation; plan ``20261026-recom-lane-sp``, S1 design).

    FESOM2 #1207's tracer scheme for every BGC tracer, as for T and S (:func:`fesom_jax.step._tracers_advect_sp`),
    with REcoM's code in its own float64 (REcoM's ``WP = 8``: float64 inputs, explicit float64 constants):

    * ``X1`` is stage 1's float64 result; the advection and Redi read its WP rounding ``V1`` (FESOM2's WP ``values``
      after ``recom()``) and give the float64 ``del_ttf`` (:func:`fesom_jax.step.tracer_content_sp`);
    * REcoM's loop terms run in float64 on ``X1`` and the finished WP tracers; their modified values ``vmod`` enter
      the carried reconstruction as float64 content, ``acc = vmod·hnode + del_ttf + carry``, rounded once
      (:func:`fesom_jax.tracer_adv.ale_reconstruct_carry`): stage 1's change, stage 1's rounding residual
      ``(X1 − V1)·hnode`` and the loop terms all reach the ONE carried rounding (deviations R1, R2);
    * the implicit vertical diffusion is T/S's float64 column with the carry
      (:func:`fesom_jax.tracer_diff.impl_vert_diff_one`), REcoM's surface fluxes unrounded in its rhs (R3).

    Returns ``(X_new, X_old_new, benthos_new[, terms], carry_new)``; ``X_new``, ``X_old_new`` and the carry WP."""
    from .. import precision as PR
    from ..step import tracer_content_sp
    from . import sinking
    from .tracers import SLOT
    nl1 = mesh.nl - 1
    X1, carry = hold_major(X1), hold_major(carry)        # one tracer-major buffer each (hold_major)
    V1 = PR.wp(X1)                                        # FESOM2's WP values after recom()
    h_wp = PR.wp(hnode)

    def content(x, x_old):
        return tracer_content_sp(mesh, x, x_old, uv_adv=uv_adv, w_e_adv=w_e_adv, helem=helem, hnode=hnode,
                                 hnode_new=hnode_new, h_wp=h_wp, gm_diag=gm_diag, zbar3_live=zbar3_live,
                                 Z3d_live=Z3d_live, dt=dt, exch=exch, ckpt=ckpt, adj=adj)

    d, X_old_new = _over_stack(content, (V1, X_old), (-1, -1))       # d float64 (nod2D, nl, ntr)
    d = hold_major(d)
    bc = surface_bc(sfc, dt)                          # REcoM's real(8) fluxes · dt, (nod2D, ntr)
    carry_cols = [carry[..., s] for s in range(carry.shape[-1])]

    def finish(ids, values_mod):
        """T* (carried, from the float64 vmod) then the carried implicit diffusion, in FESOM2's WP scheme; returns the
        WP values, widened exactly for REcoM's code that reads them (the later sinking groups)."""
        slots = [SLOT[t] for t in ids]
        idx = jnp.array(slots)
        vmod = hold_major(jnp.concatenate([values_mod, X1[:, nl1:, idx]], axis=1))  # float64 (nod2D, nl, n)
        x_star, c = _over_stack(
            lambda v, dd, cc: tracer_adv.ale_reconstruct_carry(mesh, v, dd, hnode, hnode_new, cc),
            (vmod, d[..., idx], carry[..., idx]), (-1, -1, -1))
        x_new, c = _over_stack(
            lambda x, cc, b: ckpt(lambda: tracer_diff.impl_vert_diff_one(
                mesh, x, Kv_eff, hnode_new, dt=dt, bc_surf=b, Z3d=Z3d_new, carry=cc))(),
            (x_star, c, bc[:, idx]), (-1, -1, -1))
        for k, s in enumerate(slots):
            carry_cols[s] = c[..., k]
        return PR.full(x_new[:, :nl1])

    res = sinking.recom_tracer_loop_terms(X1[:, :nl1], PR.full(temp[:, :nl1]), PR.full(salt[:, :nl1]),
                                          geom_sink, sfc.GlodecayBenthos, benthos, dt, params, finish,
                                          switches=switches)
    # assembled once from its columns, as the loop terms assemble their stack (not a scatter into it per group)
    carry_new = jnp.stack(carry_cols, axis=-1)
    X_new = exch(jnp.concatenate([PR.wp(res.bgc), V1[:, nl1:]], axis=1), "nod")
    benthos_new = exch(res.benthos, "nod")
    if return_terms:
        return X_new, X_old_new, benthos_new, res, carry_new
    return X_new, X_old_new, benthos_new, carry_new


def bgc_every_k(mesh, st, k: int, *, uv_adv, w_e_adv, w_i, helem, hnode_new, Z3d_new, gm_diag, Kv_eff, bc, dt,
                exch, ckpt, adj, zstar: bool, sp_estimate: bool = False, full=None):
    """One physics step of the BGC-every-k scheme. Adds this step's transport (``uv_adv·helem``, ``w_e_adv``, ``w_i``,
    ``Kv_eff``, the surface BC ``bc``) to the interval sums; on the k-th step of the interval runs
    :func:`bgc_transport` once with ``k·dt`` on the means, from the thickness at the interval start (``bgc_h0``, its
    zstar geometry and vertex-mean ``helem``) to this step's ``hnode_new``, and starts a new interval. The Redi
    tensor is the k-th step's (slowly varying; its slopes and ``Ki`` enter as a product, so a mean of each would not
    be the mean flux either). Every mean is linear in what the physics steps used, so the BGC inventory changes by
    exactly the summed fluxes. Returns the updated BGC fields as a dict of ``State`` field values.

    ``full`` (full REcoM, :func:`fesom_jax.recom.wiring.full_bgc_step` with the step's own arguments bound): called on
    the BGC step in place of :func:`bgc_transport` with the interval-start geometry and the mean transport
    (``h0, zbar0, Z0, helem, helem_flux, uv_adv, w_e_adv, w_i, Kv_eff, dt_bgc``); it returns the BGC fields it
    updates, which the hold branch carries unchanged."""
    uvh = st.bgc_uvh + uv_adv * helem[..., None]
    we = st.bgc_we + w_e_adv
    wi = st.bgc_wi + (0.0 if w_i is None else w_i)
    kv = st.bgc_kv + Kv_eff
    bcs = st.bgc_bc + (0.0 if bc is None else bc)
    n = st.bgc_steps + 1
    inv = 1.0 / k

    sums_reset = lambda: dict(bgc_steps=n * 0, bgc_uvh=uvh * 0, bgc_we=we * 0, bgc_wi=wi * 0,   # noqa: E731
                              bgc_kv=kv * 0, bgc_bc=bcs * 0, bgc_h0=hnode_new.astype(st.bgc_h0.dtype))
    if full is not None:
        def run_full(_):
            h0 = st.bgc_h0
            zbar3_0, Z3d_0 = ale.live_geometry(mesh, h0)
            helem0 = exch(ale.commit_thickness(mesh, h0)[1], "elem")
            out = full(h0=h0, zbar0=zbar3_0, Z0=Z3d_0, helem=helem0, helem_flux=jnp.ones_like(helem),
                       uv_adv=uvh * inv, w_e_adv=we * inv, w_i=None if w_i is None else wi * inv, Kv_eff=kv * inv,
                       dt_bgc=k * dt)
            return dict(out, **sums_reset())

        def hold_full(_):
            keys = FULL_STEP_FIELDS + (("bgc_budget",) if st.bgc_budget is not None else ())
            out = {f: getattr(st, f) for f in keys}
            return dict(out, bgc_steps=n, bgc_uvh=uvh, bgc_we=we, bgc_wi=wi, bgc_kv=kv, bgc_bc=bcs, bgc_h0=st.bgc_h0)

        return jax.lax.cond(n[0] >= k, run_full, hold_full, None)

    def run(_):
        h0 = st.bgc_h0
        zbar3_0, Z3d_0 = ale.live_geometry(mesh, h0) if zstar else (None, None)
        helem0 = exch(ale.commit_thickness(mesh, h0)[1], "elem")
        X_new, X_old_new = bgc_transport(
            mesh, st.bgc, st.bgc_old, uv_adv=uvh * inv, helem_flux=jnp.ones_like(helem), w_e_adv=we * inv,
            w_i=None if w_i is None else wi * inv, helem=helem0, hnode=h0, hnode_new=hnode_new,
            zbar3_live=zbar3_0, Z3d_live=Z3d_0, Z3d_new=Z3d_new, gm_diag=gm_diag, Kv_eff=kv * inv,
            bc=None if bc is None else bcs, dt=k * dt, exch=exch, ckpt=ckpt, adj=adj, sp_estimate=sp_estimate)
        return dict(bgc=X_new, bgc_old=X_old_new, bgc_steps=n * 0, bgc_uvh=uvh * 0, bgc_we=we * 0, bgc_wi=wi * 0,
                    bgc_kv=kv * 0, bgc_bc=bcs * 0, bgc_h0=hnode_new.astype(h0.dtype))

    def hold(_):
        return dict(bgc=st.bgc, bgc_old=st.bgc_old, bgc_steps=n, bgc_uvh=uvh, bgc_we=we, bgc_wi=wi, bgc_kv=kv,
                    bgc_bc=bcs, bgc_h0=st.bgc_h0)

    # the counter is the same on every node and device (it starts at 0 everywhere), so lane 0 decides
    return jax.lax.cond(n[0] >= k, run, hold, None)
