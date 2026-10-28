"""Evolving model state for the FESOM2 → JAX port (Task 1.2).

:class:`State` is a flat, frozen dataclass registered as a JAX **pytree**, so it
flows through ``jax.lax.scan`` / ``jax.grad`` as a single structured value. Every
field is a dense array sized to the global mesh (``nl=48`` for pi); ragged
vertical extent is handled by the masks in :mod:`fesom_jax.mesh`, not by ragged
storage.

Layout mirrors the C port's ``fesom_dyn`` (``fesom_dyn.h``) and ``fesom_aux``
(``fesom_aux.h``) structs and the thickness fields on ``fesom_mesh``. Each field
notes its C owner, entity, and whether it is a **layer** quantity (valid
``[ulevels-1, nlevels-1)``) or an **interface** quantity (valid
``[ulevels-1, nlevels-1]``). Velocity vectors are ``[..., 2]`` (u,v), matching
C macro ``FESOM_ELEMVEC``.

This phase only *defines and sizes* the state and provides constructors; the
individual fields are populated as the substep kernels are ported in Phase 2.
The exact bit-level rest initialization is a Phase-2 (Task 2.11) gate; here
:meth:`State.rest` gives a clean, physically-consistent starting point.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from .mesh import Mesh


@dataclasses.dataclass(frozen=True)
class State:
    # --- tracers (node, layer) — fesom_tracers ---
    T: jax.Array            # (nod2D,nl) potential temperature [°C]
    S: jax.Array            # (nod2D,nl) salinity [psu]
    T_old: jax.Array        # (nod2D,nl) previous-step T (AB2 single slot, valuesold)
    S_old: jax.Array        # (nod2D,nl) previous-step S
    del_ttf: jax.Array      # (nod2D,nl) per-step tracer increment accumulator

    # --- momentum (element layer, + node interpolation) — fesom_dyn ---
    uv: jax.Array           # (elem2D,nl,2) horizontal velocity (u,v)
    uv_rhs: jax.Array       # (elem2D,nl,2) momentum RHS
    uv_rhsAB: jax.Array     # (elem2D,nl,2) AB2 RHS history
    uvnode: jax.Array       # (nod2D,nl,2) u,v interpolated to nodes (PP mixing)
    uvnode_rhs: jax.Array   # (nod2D,nl,2) momentum-advection RHS at nodes (momadv_opt=2)

    # --- vertical velocity (node, interface) — fesom_dyn ---
    w: jax.Array            # (nod2D,nl) vertical velocity at interfaces
    w_e: jax.Array          # (nod2D,nl) explicit part (tracer advection)
    w_i: jax.Array          # (nod2D,nl) implicit part (impl_vert_visc)
    cfl_z: jax.Array        # (nod2D,nl) vertical CFL at interfaces

    # --- SSH (node, 2D) — fesom_dyn ---
    eta_n: jax.Array        # (nod2D,) surface elevation η
    d_eta: jax.Array        # (nod2D,) SSH increment (CG solve output)
    ssh_rhs: jax.Array      # (nod2D,) SSH equation RHS
    ssh_rhs_old: jax.Array  # (nod2D,) previous-step ssh_rhs

    # --- ALE thickness — fesom_mesh ---
    hnode: jax.Array        # (nod2D,nl) layer thickness at vertices (layer)
    hnode_new: jax.Array    # (nod2D,nl) thickness after SSH update
    helem: jax.Array        # (elem2D,nl) layer thickness at cells (layer)
    hbar: jax.Array         # (nod2D,) thickness-weighted SSH proxy
    hbar_old: jax.Array     # (nod2D,) previous-step hbar

    # --- EOS / pressure / mixing — fesom_aux ---
    density: jax.Array      # (nod2D,nl) in-situ density − ρ0 (density_m_rho0, layer)
    hpressure: jax.Array    # (nod2D,nl) hydrostatic pressure (layer)
    bvfreq: jax.Array       # (nod2D,nl) Brunt-Väisälä N² (interface)
    Kv: jax.Array           # (nod2D,nl) tracer vertical diffusivity (interface)
    Av: jax.Array           # (elem2D,nl) momentum vertical viscosity (elem interface)
    pgf_x: jax.Array        # (elem2D,nl) pressure-gradient force x (layer)
    pgf_y: jax.Array        # (elem2D,nl) pressure-gradient force y (layer)

    # --- TKE turbulent kinetic energy (Phase 9b, PROGNOSTIC) — g_cvmix_tke ---
    # The ONE prognostic mixing field (KPP/PP/GM are stateless): the classical-TKE
    # closure carries `tke` at interfaces across steps, so it joins both State and the
    # lax.scan carry. IC = 0 (cold start, fesom_tke_alloc calloc): the first TKE call
    # floors the wet column to tke_min and emits Kv/Av ≈ 0 — the exact C mirror. Stays
    # 0 forever outside the wet slice (masked-inert). Unused while tke_cfg is None.
    tke: jax.Array          # (nod2D,nl) turbulent kinetic energy (interface)

    # --- sea ice (Phase 6, surface-only 2-D) — fesom_ice ---
    # Prognostic ice state carried across steps. All default-zero ⇒ the ocean-only
    # (pi / Phase-5 no-ice) path is bit-identical (nothing reads these until Phase 6).
    a_ice: jax.Array        # (nod2D,) ice concentration [0..1]
    m_ice: jax.Array        # (nod2D,) ice volume per area [m]
    m_snow: jax.Array       # (nod2D,) snow volume per area [m]
    u_ice: jax.Array        # (nod2D,) ice velocity x [m/s]
    v_ice: jax.Array        # (nod2D,) ice velocity y [m/s]
    t_skin: jax.Array       # (nod2D,) ice surface (skin) temperature [°C] (Newton warm-start)
    sigma11: jax.Array      # (elem2D,) EVP stress σ11 [N/m] — ELASTIC MEMORY across steps
    sigma12: jax.Array      # (elem2D,) EVP stress σ12 (not re-zeroed each step)
    sigma22: jax.Array      # (elem2D,) EVP stress σ22

    # --- single precision only (fesom_jax.precision; FESOM2 #1206 ``tracers%work%carry``) ---
    # The thickness-weighted rounding residual of T and S, returned at the next update so the
    # tracer content stays exact to float64 while the state is float32. ``None`` in double
    # precision (an empty pytree node: no leaf, the double-precision program is unchanged).
    carry_T: jax.Array = None   # (nod2D,nl) float32 in single precision
    carry_S: jax.Array = None   # (nod2D,nl)
    # --- sea ice every N ocean steps (ice_ave_steps > 1, plan Task 12) — FESOM2 t_ice ---
    # OPTIONAL: all None unless IceConfig.ice_ave_steps > 1, so an ice_ave_steps == 1 program
    # (the DP production step, the adjoint) has exactly the leaves it had before (None is an
    # empty pytree). ice_step.init_ice_avg fills them with the upstream init values. Each is a
    # NODE field so the sharded partition/fold/restart plumbing (entity-leading leaves) carries
    # it unchanged; the counter and the flag hold the same value on every lane.
    ice_steps_since_upd: jax.Array | None = None  # (nod2D,) int32 ice%ice_steps_since_upd
    ice_update: jax.Array | None = None           # (nod2D,) int32 ice%ice_update (0/1)
    srfoce_temp: jax.Array | None = None          # (nod2D,) running mean T(1) since the last ice update
    srfoce_salt: jax.Array | None = None          # (nod2D,) running mean S(1)
    srfoce_ssh: jax.Array | None = None           # (nod2D,) running mean hbar
    srfoce_u: jax.Array | None = None             # (nod2D,) u_w (upstream quirk: uw/(k+1), ice_step.py)
    srfoce_v: jax.Array | None = None             # (nod2D,) v_w
    ice_flx_h: jax.Array | None = None            # (nod2D,) last ice-thermo outputs that oce_fluxes
    ice_flx_fw: jax.Array | None = None           #   reads on the steps without an ice update
    ice_evaporation: jax.Array | None = None
    ice_sublimation: jax.Array | None = None
    ice_thdgr: jax.Array | None = None
    ice_thdgrsn: jax.Array | None = None
    ice_real_salt_flux: jax.Array | None = None
    ice_a_old: jax.Array | None = None            # thermo-entry a_ice (ice%data(1)%values_old)
    # --- split-explicit SSH (AleConfig.split_explicit, fesom_jax.ssh_se; plan 20261021-core2-split-explicit) ---
    # OPTIONAL: all None on the CG path (the ICE_AVG_FIELDS precedent), so a CG program has exactly the leaves it had
    # before. ssh_se.cold_start fills them with flat rings. Entity-leading, so the sharded partition/fold/restart
    # plumbing carries them unchanged.
    se_eta: jax.Array | None = None               # (nod2D,2) eta^{n-1/M}, eta^{n-2/M} (eta^n is eta_n)
    se_ubt: jax.Array | None = None               # (elem2D,3,2) U^n, U^{n-1/M}, U^{n-2/M} barotropic transport
    se_sbar_old: jax.Array | None = None          # (elem2D,2) last step's sum_k h u (the AB2 Coriolis of the r-term)
    # --- REcoM biogeochemistry (RunConfig.recom, fesom_jax.recom; plan 20261025-recom-port) ---
    # OPTIONAL: all None unless REcoM is on (the SE_FIELDS precedent). The tracer axis is TRAILING so every leaf stays
    # entity-leading and the sharded partition/fold/restart plumbing carries them unchanged.
    bgc: jax.Array | None = None                  # (nod2D,nl,ntr) BGC tracers, in recom slot order (tracers%data(3:))
    bgc_old: jax.Array | None = None              # (nod2D,nl,ntr) previous-step BGC tracers (AB2 valuesold)
    # single precision + full REcoM only (an opt-in deviation, plan 20261026-recom-lane-sp): the #1206 rounding carry of
    # every BGC tracer, as carry_T/carry_S (float32; None in double precision and in the passive mode)
    bgc_carry: jax.Array | None = None            # (nod2D,nl,ntr)
    # BGC every k-th step (RecomConfig.bgc_every > 1, a stated deviation; None at k = 1): the running sums of the
    # transport the physics steps of the current interval used, and the thickness at the interval start
    bgc_steps: jax.Array | None = None            # (nod2D,) int32 physics steps summed so far in this interval
    bgc_uvh: jax.Array | None = None              # (elem2D,nl,2) sum of the advecting uv * helem (incl. GM bolus)
    bgc_we: jax.Array | None = None               # (nod2D,nl) sum of the advecting explicit w (incl. GM bolus)
    bgc_wi: jax.Array | None = None               # (nod2D,nl) sum of the implicit w (w_split; 0 without)
    bgc_kv: jax.Array | None = None               # (nod2D,nl) sum of the K33-augmented Kv
    bgc_bc: jax.Array | None = None               # (nod2D,) sum of the surface BC increments (passive "salt" mode)
    bgc_h0: jax.Array | None = None               # (nod2D,nl) hnode at the interval start
    # full REcoM (RecomConfig.mode == "full"; None in the passive mode): FESOM2-REcoM's own carried arrays
    bgc_benthos: jax.Array | None = None          # (nod2D,4) Benthos: benthic N, C, Si, Calc [mmol m-2] (restarted)
    bgc_alk_surf: jax.Array | None = None         # (nod2D,) Alk_surf, the alkalinity restoring target (restarted;
    #                                               Fortran rebuilds it from the IC every run: the same values)
    bgc_carb: jax.Array | None = None             # (nod2D,nl-1,8) the cached carbonate system CO23D..rhoSW3D (sms.CARB_FIELDS);
    #                                               NOT restarted (as Fortran; the mstep == 1 refresh fills every wet cell)
    bgc_mstep: jax.Array | None = None            # (nod2D,) int32 FESOM2's mstep - 1: BGC steps done in this run segment;
    #                                               0 at EVERY run start (never from a restart), as fesom_module.F90's n
    bgc_diag: jax.Array | None = None             # (nod2D,len(recom.wiring.DIAG2D)) the last BGC step's 2-D diagnostics
    #                                               (output only, not restarted; GloHplus = surface pH among them)
    bgc_budget: jax.Array | None = None           # (nod2D,4,ntr) RecomConfig.budget only: per-column content sums
    #                                               (recom.wiring.BUDGET_TERMS), not restarted

    # ---- constructors -----------------------------------------------------
    @classmethod
    def zeros(cls, mesh: Mesh, *, xp=jnp) -> "State":
        """All-zero state with every field correctly shaped & float64.

        ``xp`` selects the array backend: the default ``jnp`` is byte-identical to before
        (builds on the default device); ``xp=np`` builds the whole State on the HOST as
        numpy (Phase 8b B.3 — so a global IC for a big mesh is never materialized on GPU 0
        before sharding, the dars/NG5 setup-OOM fix). Either way the result is a valid
        ``State`` pytree; the data pipeline (``partition_state``) consumes both."""
        n, e, nl = mesh.nod2D, mesh.elem2D, mesh.nl
        f = xp.float64

        def Z(*shape):
            return xp.zeros(shape, f)

        return cls(
            T=Z(n, nl), S=Z(n, nl), T_old=Z(n, nl), S_old=Z(n, nl), del_ttf=Z(n, nl),
            uv=Z(e, nl, 2), uv_rhs=Z(e, nl, 2), uv_rhsAB=Z(e, nl, 2),
            uvnode=Z(n, nl, 2), uvnode_rhs=Z(n, nl, 2),
            w=Z(n, nl), w_e=Z(n, nl), w_i=Z(n, nl), cfl_z=Z(n, nl),
            eta_n=Z(n), d_eta=Z(n), ssh_rhs=Z(n), ssh_rhs_old=Z(n),
            hnode=Z(n, nl), hnode_new=Z(n, nl), helem=Z(e, nl), hbar=Z(n), hbar_old=Z(n),
            density=Z(n, nl), hpressure=Z(n, nl), bvfreq=Z(n, nl), Kv=Z(n, nl),
            Av=Z(e, nl), pgf_x=Z(e, nl), pgf_y=Z(e, nl),
            tke=Z(n, nl),
            a_ice=Z(n), m_ice=Z(n), m_snow=Z(n), u_ice=Z(n), v_ice=Z(n), t_skin=Z(n),
            sigma11=Z(e), sigma12=Z(e), sigma22=Z(e),
        )

    @classmethod
    def rest(cls, mesh: Mesh, T0: float = 10.0, S0: float = 35.0, *, xp=jnp,
             ale_cfg=None) -> "State":
        """Rest state: zero flow/SSH, constant ``T=T0``/``S=S0``, reference layer
        thicknesses (``zbar`` differences). The exact C-matching rest init is a
        Phase-2 gate; this is a clean physical starting point.

        ``xp`` selects the array backend (see :meth:`zeros`): the default ``jnp`` is
        byte-identical to before; ``xp=np`` builds on the HOST (the dars/NG5 setup-OOM fix).

        ``ale_cfg`` (Phase 9a): an :class:`~fesom_jax.ale.AleConfig` ⇒ the thickness IC is
        built by :func:`fesom_jax.ale.init_thickness_zstar` instead of the bare ``zbar``
        differences. At cold start (``hbar=0``) the two are **bit-for-bit identical** (the
        free Z1 degeneracy gate) — the zstar stretch factor ``(1+hbar/dd)`` is 1 — so this
        is a no-op on the values, just the explicit zstar entry point (the carried thickness
        becomes time-varying only once the run starts). ``None`` ⇒ the linfs path (unchanged).
        Only the ``jnp`` backend is supported with ``ale_cfg`` (the host-numpy path is the
        big-mesh sharding setup; its zstar IC is out of scope here)."""
        st = cls.zeros(mesh, xp=xp)
        n, e, nl = mesh.nod2D, mesh.elem2D, mesh.nl
        T = xp.full((n, nl), float(T0), xp.float64)
        S = xp.full((n, nl), float(S0), xp.float64)

        if ale_cfg is not None:
            ale_cfg.validate()
            if xp is not jnp:
                raise NotImplementedError(
                    "State.rest(ale_cfg=...) supports the jnp backend only (the host-numpy "
                    "path is the big-mesh sharding setup; zstar IC there is out of scope)")
            from . import ale as _ale       # lazy: keep ale off the host-numpy import path
            hbar0 = jnp.zeros((n,), jnp.float64)
            hnode, helem, eta_n, ssh_rhs_old = _ale.init_thickness_zstar(mesh, hbar0, hbar0)
            return dataclasses.replace(
                st, T=T, S=S, T_old=T, S_old=S, hnode=hnode, hnode_new=hnode, helem=helem,
                eta_n=eta_n, ssh_rhs_old=ssh_rhs_old)

        # reference layer thickness h[k] = z(k) - z(k+1) > 0, masked to valid layers. The
        # jnp path uses the original .at[].set(); the numpy path the equivalent slice-assign
        # (bit-identical values — same subtraction, last column left 0).
        z_n = mesh.zbar_3d_n                                   # (n, nl)
        zbar = mesh.zbar                                       # (nl,)
        if xp is jnp:
            dz_n = jnp.zeros((n, nl)).at[:, :-1].set(z_n[:, :-1] - z_n[:, 1:])
            dz = jnp.zeros((nl,)).at[:-1].set(zbar[:-1] - zbar[1:])
            node_mask, elem_mask = mesh.node_layer_mask, mesh.elem_layer_mask
        else:
            z_n, zbar = np.asarray(z_n), np.asarray(zbar)
            dz_n = np.zeros((n, nl))
            dz_n[:, :-1] = z_n[:, :-1] - z_n[:, 1:]
            dz = np.zeros((nl,))
            dz[:-1] = zbar[:-1] - zbar[1:]
            node_mask = np.asarray(mesh.node_layer_mask)
            elem_mask = np.asarray(mesh.elem_layer_mask)
        hnode = xp.where(node_mask, dz_n, 0.0)
        helem = xp.where(elem_mask, dz[None, :], 0.0)

        return dataclasses.replace(
            st, T=T, S=S, T_old=T, S_old=S, hnode=hnode, hnode_new=hnode, helem=helem
        )


# The optional ice_ave_steps > 1 fields, in State order (None when ice_ave_steps == 1).
ICE_AVG_FIELDS = ("ice_steps_since_upd", "ice_update", "srfoce_temp", "srfoce_salt",
                  "srfoce_ssh", "srfoce_u", "srfoce_v", "ice_flx_h", "ice_flx_fw",
                  "ice_evaporation", "ice_sublimation", "ice_thdgr", "ice_thdgrsn",
                  "ice_real_salt_flux", "ice_a_old")
# The optional split-explicit SSH fields, in State order (None on the CG path; = ssh_se.STATE_FIELDS).
SE_FIELDS = ("se_eta", "se_ubt", "se_sbar_old")
# The optional REcoM fields, in State order (None unless RunConfig.recom is set).
BGC_MR_FIELDS = ("bgc_steps", "bgc_uvh", "bgc_we", "bgc_wi", "bgc_kv", "bgc_bc", "bgc_h0")  # BGC-every-k sums (None at k=1)
BGC_FULL_FIELDS = ("bgc_benthos", "bgc_alk_surf", "bgc_carb", "bgc_mstep", "bgc_diag", "bgc_budget")  # full REcoM only
BGC_FIELDS = ("bgc", "bgc_old", "bgc_carry") + BGC_MR_FIELDS + BGC_FULL_FIELDS
# Present but never written to a restart: FESOM2 restarts neither the carbonate cache nor mstep (both restart with
# the run segment, fesom_module.F90:849,983,1049; recom_sms.F90:489); the diagnostics and budget sums are output.
RESTART_SKIP = ("bgc_carb", "bgc_mstep", "bgc_diag", "bgc_budget")


def present_fields(state) -> tuple[str, ...]:
    """The names of ``state``'s array fields, in State order: every field except an optional
    one that is None (``ICE_AVG_FIELDS`` when ``ice_ave_steps == 1``, ``SE_FIELDS`` on the CG path, ``BGC_FIELDS``
    without REcoM)."""
    return tuple(f.name for f in dataclasses.fields(State) if getattr(state, f.name) is not None)


# Register State as a pytree: every field is a data leaf (no static metadata). An optional
# field that is None is an empty subtree, so it adds no leaf.
_STATE_FIELDS = [f.name for f in dataclasses.fields(State)]
tree_util.register_dataclass(State, data_fields=_STATE_FIELDS, meta_fields=[])
