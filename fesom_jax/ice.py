"""Sea-ice static config + cold-start initial condition (Phase 6, Task 6.1).

Two host-side pieces, both faithful ports of the C ice setup
(``port2/fesom2_port/src/fesom_ice.c``):

* :class:`IceConfig` — the static CORE2 ``namelist.ice`` constants hardcoded in
  ``fesom_ice_init`` (``fesom_ice.c:53-111``) + the over-ice exchange coefficients
  (``fesom_constants.h:110-113``). A bundle of compile-time constants **closed over** the
  time loop. Two of them are also differentiable parameters on the ``params.py`` path: the ice
  strength ``pstar`` and the ice-ocean drag ``cd_oce_ice`` (``Params.ice_pstar`` /
  ``Params.ice_cd``; when set, ``ice_step.ice_surface_step`` replaces the field by the traced
  value; unset ⇒ the constant, nothing traced).
* :func:`ice_initial_state` / :func:`seed_ice` — the cold-start ice IC
  (``fesom_ice_initial_state``, ``fesom_ice.c:246-277``): where (non-cavity & IC SST < 0),
  ``a_ice=0.9`` with hemisphere-split ``m_ice``/``m_snow`` (NH 1.0/0.1, SH 2.0/0.5); else
  open water. This **generalizes** the Phase-5 static-mask :func:`surface_forcing.ice_ic_aice`
  (which produced only the ``a_ice`` mask) to the full prognostic ``(a_ice, m_ice, m_snow)``.

⚠️ Scope (``fesom_ice_types.h:6-18``): single ice class, **standard EVP only**
(``whichEVP=0``), 3 tracers, virtual-salt path (``use_virt_salt=1``), ``ref_sss_local=1``,
no ridging/meltponds/icebergs/cavities/wiso. The IC is host setup (not in the AD path,
though the seeded fields are valid gradient targets).
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .config import CD_ATM_ICE, CE_ATM_ICE, CH_ATM_ICE
from .mesh import Mesh
from .state import State


class IceConfig(NamedTuple):
    """Static sea-ice constants (CORE2 ``namelist.ice``, ``fesom_ice.c:53-111``). SI units.

    ``ice_dt`` defaults to the CORE2 ocean step (500 s); set it to the actual ocean ``dt``
    when building for a run. The EVP timing (``Tevp_inv``, ``dte``) and the derived heat
    capacities (``cc``, ``cl``) are properties — note ``Tevp_inv = 3.0/ice_dt`` is the
    ``fesom_ice.c:233`` *setup* value, NOT the stale ``types.h:177`` comment
    "evp_rheol_steps/ice_dt".

    Choosing the ice adjoint. These fields act only in the backward pass; the forward is the
    same for every choice, and the defaults are unchanged (``adjoint_mode="exact"``,
    ``adjoint_damping="none"``):

    * Ice controls (P*, C_d) or ice-velocity sensitivity, with mEVP (``whichEVP=1``): set
      ``adjoint_mode`` to "exact" (ice state live) or "dynamics" (ice state given), and
      ``adjoint_damping`` to one of
        - "frozen_sweep" (``adjoint_alpha_kappa=1``, ``adjoint_theta=0.97``): both controls.
          Coupled CORE2, TKE and GM adjoint frozen, 2013-01-01 and 07-01 at 2-30 d, 03-01 at
          2-10 d: ln P* 0.95-1.02 and ln C_d 0.99-1.00 of the full-model finite difference;
        - "local" with ``adjoint_kappa=0`` (the zero-stress tangent): C_d only, at 0.93-0.96.
          The cheapest option; P* gets no path.
    * No ice controls: "rheology" (ice velocity dark; thermodynamics, advection and coupling
      differentiated) or "prescribed" (ice state given, fluxes differentiable).
    * "frozen" removes the ice from the gradient entirely, including the atmosphere-to-ocean
      fluxes.
    * "exact" with ``adjoint_damping="none"`` unrolls the exact mEVP tangent, which grows 6-18
      per sweep in near-rigid elements: NaN within a day.

    Measurements: docs/ADJOINT_HORIZON.md §5 and docs/PORTING_LESSONS.md I5-I5h, I10, I17, I19."""

    # --- EVP rheology / dynamics (fesom_ice.c:54-67) ---
    pstar: float = 30000.0        # ice strength parameter [N/m²]
    ellipse: float = 2.0          # yield-curve axis ratio (vale = 1/ellipse²)
    c_pressure: float = 20.0      # strength concentration exponent C
    delta_min: float = 1.0e-11    # EVP delta floor [1/s] — ⚠️ match C; safe-sqrt for AD
    evp_rheol_steps: int = 120    # fixed EVP subcycles per ice step
    cd_oce_ice: float = 5.5e-3    # ocean-ice drag coefficient
    theta_io: float = 0.0         # ice-ocean rotation angle [rad]
    ice_free_slip: int = 0
    # --- mEVP rheology (Phase 9c; fesom_ice_maevp.c) — whichEVP=0 ⇒ byte-identical ---
    whichEVP: int = 0             # 0 = standard EVP (default); 1 = mEVP. aEVP (=2) NOT ported.
    alpha_evp: float = 250.0      # mEVP stress relaxation constant (Bouillon et al. 2013)
    beta_evp: float = 250.0       # mEVP momentum relaxation constant
    # --- FCT advection (fesom_ice.c:61-64) ---
    ice_gamma_fct: float = 0.5    # CORE2 namelist value (NOT the 0.25 module default)
    ice_diff: float = 10.0        # stabilising diffusion
    # scale_area is a &oce_dyn NAMELIST parameter (oce_modules.F90:193), 5.8e9 in every
    # CORE2 reference run; the ice FCT diffusion ice_diff*sqrt(elem_area/scale_area)
    # (ice_fct.F90:175) is its only live consumer. The C, Kokkos and JAX ports carried the
    # module default 2.0e8 until 2026-09-18 = 5.4x too much ice diffusion at every ice
    # edge — the systematic ice loss behind the 63-yr deep-ocean drift (fesom2_port_zstar
    # docs/plans/20260918-longrun-drift-strategy.md; fix fesom2_port_drift@21da79e).
    scale_area: float = 5.8e9     # CORE2 namelist.oce value (NOT the 2.0e8 module default)
    # --- timestep (fesom_ice.c:69, 231 ice_setup) ---
    ice_ave_steps: int = 1
    ice_dt: float = 500.0         # = ice_ave_steps * ocean dt (CORE2 dt=500)
    # --- thermo densities (fesom_ice.c:77-81) ---
    rhoair: float = 1.3
    rhowat: float = 1025.0
    rhofwt: float = 1000.0
    rhoice: float = 910.0
    rhosno: float = 290.0
    # --- specific heats J/(kg·K) (fesom_ice.c:82-84) ---
    cpair: float = 1005.0
    cpice: float = 2106.0
    cpsno: float = 2090.0
    # --- latent heats J/kg (fesom_ice.c:87-88) ---
    clhw: float = 2.501e6         # water -> vapour
    clhi: float = 2.835e6         # ice -> vapour
    # --- conductivities W/m/K (fesom_ice.c:94-95) ---
    con: float = 2.1656
    consn: float = 0.31
    # --- radiation / albedo / misc (fesom_ice.c:89-110) ---
    tmelt: float = 273.15
    boltzmann: float = 5.67e-8
    emiss_ice: float = 0.97
    emiss_wat: float = 0.97
    albsn: float = 0.81           # snow albedo (frozen)
    albsnm: float = 0.77          # snow albedo (melting)
    albi: float = 0.70            # ice albedo (frozen)
    albim: float = 0.68           # ice albedo (melting)
    albw: float = 0.1             # open-water albedo (CORE2 namelist, overrides LY2004 0.066)
    Sice: float = 4.0             # ice salinity [ppt]
    h0: float = 0.5               # lead-closing N hemi [m]
    h0_s: float = 0.5             # lead-closing S hemi [m]
    h_ml: float = 2.5             # upper-layer thickness for heat available [m]
    iclasses: int = 7             # ice-thickness gradations (growth-rate average)
    hmin: float = 0.01            # cutoff ice thickness [m]
    armin: float = 0.01           # minimum ice concentration
    c_melt: float = 0.5           # compactness melt coefficient
    snowdist: int = 1
    open_water_albedo: int = 0
    use_meltponds: int = 0
    use_virt_salt: int = 1        # linfs virtual-salt path
    ref_sss_local: int = 1        # rsss = local S_top (not a global ref_sss)
    ref_sss: float = 34.7         # global rsss when ref_sss_local=0 (oce_modules.F90:37 default)
    # --- over-ice exchange coeffs (fesom_constants.h:110-113; from config.py) ---
    ch_atm_ice: float = CH_ATM_ICE  # 1.75e-3 sensible-heat over ice
    ce_atm_ice: float = CE_ATM_ICE  # 1.75e-3 evaporation over ice
    cd_atm_ice: float = CD_ATM_ICE  # 1.2e-3  atm-ice momentum drag
    # --- adjoint mode: how the ice block is treated in the BACKWARD pass -----------------
    # The FORWARD is identical in all three modes — `stop_gradient` is the identity, so the ice
    # trajectory is bit-for-bit the validated model whichever you pick. Only the gradient
    # changes. Measured numbers and provenance: docs/ADJOINT_HORIZON.md.
    #
    #   "rheology"  : RECOMMENDED WITHOUT ICE CONTROLS (with them: see the class docstring).
    #                 Differentiate ice thermodynamics, advection, cut-off and the
    #                 ocean-to-ice coupling, but unroll ZERO momentum-solver subcycles in the
    #                 backward, so only the ice VELOCITY is dark. This is the one cut that is
    #                 actually needed: freezing each piece of an ice step in turn showed the
    #                 momentum solve is solely responsible (dL/dm 3.4e-05 frozen against
    #                 2.0e+39 exact) while every other component is harmless. Over a 5-day
    #                 window it is clean and reproducible and changes an eddy-diffusivity
    #                 gradient by 0.21% against "frozen" — so it costs nothing in stability and
    #                 recovers the pathway that matters: eddy transport changes SST, which
    #                 changes ice growth, which changes the surface heat and salt fluxes back
    #                 into the ocean.
    #
    #   "prescribed": the ice TRAJECTORY is given. The nine prognostic ice fields carry no
    #                 tangent into a step and none out of it, so the ice-state recurrence is
    #                 cut — including its unbounded gain at an advancing ice edge, which
    #                 "rheology" keeps (a x462 jump in the a_ice tangent in one day at the
    #                 Okhotsk edge, 2014-01-04) — while every flux the ocean
    #                 receives stays differentiable in the ocean state and the atmosphere. What
    #                 the ECCO adjoint does when the sea-ice package's adjoint is disabled. Use
    #                 this, not "frozen", when the ice must be excluded but dJ/d(forcing) and
    #                 the SST -> flux feedback are wanted.
    #
    #   "frozen"    : stop_gradient the WHOLE ice update, so no gradient flows through ice at
    #                 all. ⚠️ "The whole ice update" is IceStepOut, and that carries the surface
    #                 fluxes: under "frozen" the gradient with respect to the ATMOSPHERE is
    #                 identically zero and the SST -> flux damping is gone from the adjoint
    #                 (measured over 30 d: the forcing gradient 2.4e-03 -> 0.0 exactly, the
    #                 temperature gradient +16 %). A calibration whose controls are ocean
    #                 parameters can live with that; a forcing sensitivity cannot. The most
    #                 conservative option, and the one every published adjoint sea-ice study
    #                 fell back to before free-drift approximations existed.
    #
    #   "exact"     : differentiate everything. With adjoint_damping="none" it **returns NaN on
    #                 realistic configurations**: the unrolled mEVP tangent has a local
    #                 near-rigid stress mode that flips sign each sweep and grows 6-18 per
    #                 sweep where γ/αβ = 6-130 (4e8 or more per model step). Published
    #                 adjoint sea-ice work discards the rheology, reduces it to free drift
    #                 (Koldunov et al. 2017), simplifies it (Usui et al. 2016; Toyoda et al.
    #                 2015, 2019; Lyu et al. 2023) or damps the tangent (Panteleev et al.
    #                 2020). Usable with adjoint_damping="frozen_sweep" (both ice controls) or
    #                 "local" with κ 0 (C_d only). The default, because it is the
    #                 mathematically faithful one.
    #
    #   "dynamics"  : (mEVP only, whichEVP=1) the ice DYNAMICS are differentiated, the ice STATE
    #                 is given. a_ice, m_ice, m_snow and t_skin carry no tangent into a step and
    #                 none out of it (as in "prescribed"), while u_ice, v_ice and the stresses
    #                 stay live across steps and through the mEVP sweeps (all of them, or the last
    #                 `adjoint_sweeps`), with the derivative-only damping `adjoint_damping`.
    #                 Within a step the live ice velocity still reaches advection -> a_co ->
    #                 thermodynamics, the fluxes, the freshwater balance and the friction
    #                 velocity, and the post-thermo a_ice still feeds the stress blend and the
    #                 shortwave gate of the SAME step: the exit cut sits after those uses, on the
    #                 carried state only. This is exactly the derivative of a forward whose ice
    #                 state is overwritten by a base trajectory after every step (the "replay"
    #                 finite difference of scripts/adjoint/ice_cbar.py). Use it with
    #                 adjoint_damping="frozen_sweep" or "local": undamped, the exact mEVP
    #                 tangent has a local near-rigid stress mode that flips sign each sweep with
    #                 |λ| 6-18 (measured on CORE2 2013-01-01 and 03-01, PORTING_LESSONS.md I5).
    #
    #   "free_drift": (PLANNED, now low priority) drop the internal stress in the backward while
    #                 keeping wind, ocean drag, Coriolis and tilt. adjoint_damping="local" with
    #                 κ 0 under "exact"/"dynamics" is that zero-stress tangent within mEVP.
    adjoint_mode: str = "exact"
    # --- derivative-only settings of the mEVP sweeps (whichEVP=1; the forward is unchanged) ---
    # Allowed with adjoint_mode "exact" or "dynamics" (the modes that unroll the sweeps);
    # validated at trace time by ice_step.validate_adjoint_settings (``_replace`` skips __new__).
    #   adjoint_damping "none": plain autodiff. "local": every sweep multiplies the tangent of
    #       the NEW stress by f = min(1, κ/(γ/αβ)) per ice element, γ/αβ the local mEVP stability
    #       number of the current iterate (forward values, ice_mevp.mevp_stability_number): the
    #       implicit Newtonian damping of Panteleev et al. (2020) with a local timescale, acting
    #       only where the iteration is stiff (near-rigid ice). κ = 0 removes the stress tangent
    #       on every ice element (the zero-stress tangent). Measured (fallback-B κ scan, CORE2
    #       one-day chains): stable at both dates for κ ≤ 0.3; ln C_d retained (r ≈ 1, pattern
    #       correlation 0.99); ln P* biased (one-day r 0.73-0.89, pattern correlation 0.42-0.62).
    #       In the coupled CORE2 bar (2-10 d), κ = 0 keeps C_d at 0.93-0.96 of the FD, and κ > 0
    #       leaves P* biased (κ 0.1: 0.45-1.08 of the FD). For P*, use "frozen_sweep".
    #   adjoint_kappa κ ≥ 0 (default 0.25).
    #   adjoint_sweeps: −1 = differentiate all sweeps; K ≥ 0 = only the last K (the earlier ones
    #       run in the forward with their tangent cut).
    #   adjoint_cuts: derivative cuts inside the "dynamics" or "exact" step, a tuple of names
    #       (ice_step.ADJOINT_CUTS; default () = none, the unchanged program). The first five
    #       are DIAGNOSTIC, each one stop_gradient on a coupling path between the ice velocity
    #       and the ocean:
    #       "advect" — the new u_ice/v_ice into the ice advection (fct_solve) AND the thermodynamic
    #                  friction velocity (everything downstream of the new velocity except the
    #                  ice-ocean stress, the carried state and the cost);
    #       "fct"    — the new u_ice/v_ice into fct_solve only (a subset of "advect");
    #       "ssh"    — the sea-surface height (hbar) the ice momentum solve sees (ssh tilt);
    #       "ocean_vel" — the ocean surface velocity the ice momentum solve sees;
    #       "ice2ocean" — the ice velocity in the ice-ocean stress handed to the ocean;
    #       "fct_limiter" — the Zalesak limiter factors of the ice advection (a, m, m_snow) enter
    #                  the tangent as constants (ice_adv._fem_fct), the ice counterpart of
    #                  AdjointConfig.freeze_fct_limiter;
    #       "fct_flux" — the whole limited antidiffusive correction enters as a constant, so the
    #                  advection tangent is the low-order one (the ice counterpart of
    #                  AdjointConfig.freeze_advection).
    #       Why the last two: farc, January, θ 0.9, the reverse trace grows x240 in m_ice in 6 h
    #       on ten coastal nodes of the Gulf of Anadyr (u_ice = 0 on the coast, 0.3 m/s
    #       offshore), alternating in sign. The one-step tangent of the ice FCT (m_ice, u_ice
    #       held) at that state has a top singular value of 3e3 with the limiter derivative
    #       live, 1.0007 with the limiter frozen and 1.003 for the low-order tangent. Yet
    #       "fct_limiter" alone blows the 10 d farc gradient up (x1.7e6): with the factors
    #       frozen, the unlimited response of the antidiffusive flux to u_ice feeds the
    #       ice-dynamics loop.
    #       "strength_conc" — mEVP only: the concentration factor exp(−C(1−A)) of the ice
    #                  strength (ice_mevp.mevp_setup, A the element mean) enters the tangent as a
    #                  constant; its thickness factor and P* stay live.
    #       Why: farc, January start, θ 0.9, cuts fct_flux + thin mass 100: the 365 d reverse
    #       trace grows x2 per 6 h chunk for nine days in late July (days 204-213, x1.7e5 in
    #       all) in thick compact summer pack (Chukchi Sea, west of Banks Island; A 0.8-0.98,
    #       h 2.6-3.2 m), a_ice- and u_ice-led. Per step (the sharded one-step VJP over one
    #       chunk, x1.85 with the model's derivative): this cut x0.84, the whole strength
    #       constant x0.83, the mEVP solve dark x0.74; its thickness factor frozen x1.80, θ 0.5
    #       or 0 x1.25, cut "advect" x1.22, adjoint_alpha_kappa 0.5 or 0.25 x1.83-1.90. A loop
    #       A -> strength -> stress -> u_ice -> advection -> A, through the steep exp(−20(1−A)).
    adjoint_damping: str = "none"
    adjoint_kappa: float = 0.25
    adjoint_sweeps: int = -1
    adjoint_cuts: tuple = ()
    #   adjoint_damping "frozen_sweep": the forward is unchanged. The backward does not unroll the
    #       120 forward iterates. It linearizes ONE sweep at the final iterate x_N and runs that
    #       frozen map's transpose 120 times (custom_vjp, fesom_jax/ice_mevp_frozen.py). Two
    #       derivative-only changes to that sweep:
    #       * raised pseudo-time coefficients α_t = max(α, α·sqrt(s/κ0)) per element (s = γ/αβ
    #         at x_N), with β_t = the maximum of α_t over a node's elements;
    #       * partial plastic softening: the replacement pressure takes stop_gradient(Δ), and
    #         the viscosities' Δ-derivative is scaled by θ (0 = Picard, 1 = Newton, unstable).
    #       Measured with the CORE2 mode harness (docs/PORTING_LESSONS.md I5f/I5g, edge scan
    #       2026-10-09): at κ0 = 1 the 1-day ice-only chain contracts at both dates (2013-01-01,
    #       -03-01) for θ ≤ 0.95, at 0.75-0.86 per step; θ = 1 grows. Coupled CORE2 drift
    #       gradients, "exact" vs the full-model FD (I11, I17, I19): θ = 0.95 leaves ln P*
    #       5-12 % low; θ = 0.97 gives ln P* 0.95-1.02 and ln C_d 0.99-1.00 at 2-30 d (January,
    #       July; March to 10 d) with a clean amplification screen; θ = 0.99 overshoots P*
    #       (1.08). θ sets the size of the P* response, κ0 only the pseudo-time path (κ0 = 4
    #       diverges). Cost: as the unrolled backward.
    #   adjoint_alpha_kappa κ0 > 0 (default 1.0); adjoint_theta θ in [0, 1] (default 0.97).
    #   adjoint_thin_mass [kg/m²] ≥ 0 (default 0 = off, the unchanged program), mEVP under
    #       "dynamics" or "exact": on ice nodes whose mean ice+snow load (ρᵢ m + ρₛ m_snow)/a is
    #       below it, the momentum solve's inverse load (inv_thickness, which scales the
    #       ice-ocean drag and the air stress) and its regularized mass enter the tangent as
    #       constants (ice_mevp.mevp_setup: the forward values, and the frozen_sweep backward,
    #       which calls it again).
    #       Why: farc, January, θ 0.9, cut "fct_flux", 90 d: the reverse trace bursts at day
    #       35.75 on one nearly ice-free polynya node (Franz Josef Land, a 0.002-0.05, m
    #       0.03-0.7 mm, forming and melting every ~6 steps): its m_ice cotangent grows x34 in
    #       24 steps. Measured per step (the sharded one-step VJP): the growth is local to the
    #       node and runs m -> the node's free-drift velocity through its load -> the advection
    #       of thicker neighbour ice into the node -> m. The tangent with the mEVP solve dark
    #       decays at that node (thermodynamics and the top-layer T are contractive there); the
    #       cuts "advect" or "fct", this one at 50 or 200 kg/m², or the same everywhere all
    #       remove the growth, while "ice2ocean", "ssh" and "ocean_vel" do nothing. The floor of
    #       inv_thickness (9 kg/m²) does not reach a load of 15-35 kg/m². A counterpart of the
    #       MITgcm adjoint's special treatment of thin ice.
    #   adjoint_sweep_guard G ≥ 0 (default 0 = off, the unchanged program), frozen_sweep only:
    #       the backward tracks the largest |λ_k| of its 120 transposed sweeps against |λ_0| (the
    #       incoming cotangent; global over the devices). Past G, or not finite, that ice step's
    #       mEVP solve enters the tangent as a constant (zero cotangents out), and the step is
    #       reported on stdout ("frozen_sweep guard: ...").
    #       Why: farc, January start, θ 0.9, cuts fct_flux + thin mass 100: the 365 d reverse
    #       trace jumps x3.5e49 in one chunk at day 279.5 (thick compact ice west of Banks
    #       Island, u_ice-led). Per step it is single ice steps of x1e21-1e34 (one sweep's
    #       tangent |λ| ~1.5-2 at the frozen point, over 120 sweeps), and it is a lottery: the
    #       same setting bursts in one sharded run and not in the next (GPU round-off moves the
    #       final iterate of the near-rigid, unconverged sweep), at θ 0.9, 0.95 and 0 and at
    #       κ0 1 and 0.25, while the mEVP solve dark never does.
    adjoint_alpha_kappa: float = 1.0
    adjoint_theta: float = 0.97
    adjoint_thin_mass: float = 0.0
    adjoint_sweep_guard: float = 0.0

    # ---- derived (fesom_ice.c:85-86, 233, 84) ----
    @property
    def cc(self) -> float:
        """Volumetric heat capacity of seawater for the ice budget = rhowat·4190."""
        return self.rhowat * 4190.0

    @property
    def cl(self) -> float:
        """Volumetric latent heat of fusion of ice = rhoice·3.34e5."""
        return self.rhoice * 3.34e5

    @property
    def Tevp_inv(self) -> float:
        """EVP inverse relaxation time = 3.0/ice_dt (fesom_ice.c:233 setup value)."""
        return 3.0 / self.ice_dt

    @property
    def dte(self) -> float:
        """EVP subcycle timestep = ice_dt/evp_rheol_steps (fesom_ice_evp.c:84)."""
        return self.ice_dt / self.evp_rheol_steps

    @property
    def vale(self) -> float:
        """Yield-curve factor 1/ellipse² (fesom_ice_evp.c:83)."""
        return 1.0 / (self.ellipse * self.ellipse)

    # ---- mEVP relaxation weights (fesom_ice_maevp.c:110-111) ----
    @property
    def mevp_det2(self) -> float:
        """mEVP stress-update weight = 1/(1+alpha_evp) (fesom_ice_maevp.c:110)."""
        return 1.0 / (1.0 + self.alpha_evp)

    @property
    def mevp_det1(self) -> float:
        """mEVP stress-memory weight = alpha_evp·det2 (fesom_ice_maevp.c:111)."""
        return self.alpha_evp * self.mevp_det2


# ``typing.NamedTuple`` forbids a ``__new__`` in the class body and has no ``__init__`` hook,
# so we patch a validating ``__new__`` on post-creation: a DIRECT ``IceConfig(whichEVP=2)``
# raises (C parity — ``fesom_ice_init`` aborts on aEVP, which has no reference oracle). Honored
# at call time; ``_replace`` rebuilds via ``tuple.__new__`` and cleanly skips it (the dispatch
# in ``ice_step`` re-guards), and the NamedTuple stays a pytree.
_ice_config_new = IceConfig.__new__


def _validating_ice_config_new(cls, *args, **kwargs):
    self = _ice_config_new(cls, *args, **kwargs)
    if self.whichEVP not in (0, 1):
        raise ValueError(
            f"IceConfig.whichEVP={self.whichEVP} unsupported — only 0 (standard EVP) or "
            "1 (mEVP). aEVP (whichEVP=2) is not ported (no C reference oracle).")
    return self


IceConfig.__new__ = _validating_ice_config_new


def ice_initial_state(mesh: Mesh, sst, *, xp=jnp):
    """Cold-start ice IC (``fesom_ice_initial_state``, ``fesom_ice.c:246-277``).

    Where (non-cavity ``ulevels_nod2D<=1`` AND IC ``sst < 0``): ``a_ice=0.9``;
    Northern hemisphere (geo lat > 0) ``m_ice=1.0, m_snow=0.1``; Southern ``m_ice=2.0,
    m_snow=0.5``. Else open water (all 0). ``u_ice=v_ice=0`` (caller seeds via State).

    ``sst`` is the IC surface temperature [°C] (``[nod2D]``, e.g. ``state.T[:, 0]`` from
    :func:`fesom_jax.phc_ic.phc_initial_state`). ``xp`` selects the backend (mirrors
    :meth:`State.zeros`): default ``jnp`` (device, byte-identical to before); ``xp=np`` keeps
    the result on the HOST — needed for the big-mesh (dars/NG5) host-build path so the 2-D ice
    fields are not staged on GPU 0. Returns ``(a_ice, m_ice, m_snow)`` ``[nod2D]`` float64."""
    sst = np.asarray(sst, dtype=np.float64)
    non_cavity = np.asarray(mesh.ulevels_nod2D) <= 1
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]      # radians; > 0 ⇒ Northern hemisphere
    cold = (sst < 0.0) & non_cavity
    nh = lat > 0.0
    a_ice = np.where(cold, 0.9, 0.0)
    m_ice = np.where(cold, np.where(nh, 1.0, 2.0), 0.0)
    m_snow = np.where(cold, np.where(nh, 0.1, 0.5), 0.0)
    return (xp.asarray(a_ice, xp.float64),
            xp.asarray(m_ice, xp.float64),
            xp.asarray(m_snow, xp.float64))


def seed_ice(state: State, mesh: Mesh, sst, *, xp=jnp) -> State:
    """Return ``state`` with the cold-start ice IC seeded into ``a_ice``/``m_ice``/
    ``m_snow`` (:func:`ice_initial_state`). ``u_ice``/``v_ice``/``t_skin``/``sigma*`` stay
    at their (zero) values — the C cold-start sets ``u_ice=v_ice=0`` and ``t_skin`` starts
    0 (the first thermo step's Newton warm-starts from it). ``xp=np`` keeps the seeded fields
    on the HOST (the big-mesh host-build path; the rest of ``state`` must match the backend)."""
    a_ice, m_ice, m_snow = ice_initial_state(mesh, sst, xp=xp)
    return dataclasses.replace(state, a_ice=a_ice, m_ice=m_ice, m_snow=m_snow)
