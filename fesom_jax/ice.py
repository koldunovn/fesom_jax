"""Sea-ice static config + cold-start initial condition (Phase 6, Task 6.1).

Two host-side pieces, both faithful ports of the C ice setup
(``port2/fesom2_port/src/fesom_ice.c``):

* :class:`IceConfig` — the static CORE2 ``namelist.ice`` constants hardcoded in
  ``fesom_ice_init`` (``fesom_ice.c:53-111``) + the over-ice exchange coefficients
  (``fesom_constants.h:110-113``). A bundle of compile-time constants **closed over** the
  time loop (not a trainable param — ice stays off the ``params.py`` ML-hook path for now).
* :func:`ice_initial_state` / :func:`seed_ice` — the cold-start ice IC
  (``fesom_ice_initial_state``, ``fesom_ice.c:246-277``): where (non-cavity & IC SST < 0),
  ``a_ice=0.9`` with hemisphere-split ``m_ice``/``m_snow`` (NH 1.0/0.1, SH 2.0/0.5); else
  open water. This **generalizes** the Phase-5 static-mask :func:`core2_forcing.ice_ic_aice`
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
    "evp_rheol_steps/ice_dt"."""

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
    # --- over-ice exchange coeffs (fesom_constants.h:110-113; from config.py) ---
    ch_atm_ice: float = CH_ATM_ICE  # 1.75e-3 sensible-heat over ice
    ce_atm_ice: float = CE_ATM_ICE  # 1.75e-3 evaporation over ice
    cd_atm_ice: float = CD_ATM_ICE  # 1.2e-3  atm-ice momentum drag
    # --- adjoint mode (paper §2; how the ice block is treated in the BACKWARD pass) ---
    #   "exact"      : autodiff the full ice step (default; forward == backward transpose).
    #   "frozen"     : stop_gradient the ice update in the backward — the FORWARD still runs the
    #                  full mEVP ice (identity under stop_gradient), only the gradient skips it.
    #                  Tames the mEVP rheology adjoint blow-up for OCEAN-parameter calibration
    #                  (the ice-mediated sensitivity is dropped). The MITgcm/ECCO-style cheap path.
    #   "free_drift" : (PLANNED) custom_vjp free-drift adjoint — drop the internal-stress ∇·σ in
    #                  the backward, keep wind/ocean-drag/Coriolis/tilt; retains ice-momentum
    #                  sensitivity (and enables adjoint sea-ice calibration). NOT yet implemented.
    adjoint_mode: str = "exact"

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
    :func:`fesom_jax.phc_ic.core2_initial_state`). ``xp`` selects the backend (mirrors
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
