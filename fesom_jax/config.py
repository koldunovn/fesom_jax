"""Global configuration + physical constants for the FESOM2 → JAX port.

This module does two things:

1. **Enables float64 ("x64") in JAX.** This must happen before any JAX array is
   created, so it runs at import time and ``fesom_jax/__init__.py`` imports this
   module first. Idempotent — safe to import repeatedly.

2. **Defines the constants, mesh-rotation Euler angles, and Phase-1 namelist
   defaults, mirroring the C port's ``fesom_constants.h`` literally** (the
   algorithmic source of truth, ``/home/a/a270088/port2/fesom2_port/src/``).
   Each value carries the same provenance comment as the C header.

Fidelity note (the Golden Rule). FESOM uses a **truncated** value of π,
``3.14159265358979`` (14 significant figures) — **not** ``math.pi`` /
``jnp.pi`` (``3.141592653589793``). ``RAD``, ``OMEGA``, and the cyclic length
all derive from it; using full-precision π would seed a ~1e-13 relative error
into every coordinate rotation and Coriolis term and break the fidelity gates.
Keep the truncated literal.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

# --------------------------------------------------------------------------
# (1) float64 everywhere. Must precede any JAX array creation.
# --------------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# (2) Physical constants — FRESH_START.md §17 / fesom_constants.h:7-13
# --------------------------------------------------------------------------
PI = 3.14159265358979  # FESOM truncated π — do NOT replace with math.pi/jnp.pi
RAD = PI / 180.0  # degrees → radians
DENSITY_0 = 1030.0  # reference density [kg/m³]
G = 9.81  # gravitational acceleration [m/s²]
R_EARTH = 6367500.0  # Earth radius [m]
OMEGA = 2.0 * PI / 86400.0  # Earth rotation rate [rad/s]
VCPW = 4.2e6  # volumetric heat capacity of seawater [J/(m³·K)]


# --------------------------------------------------------------------------
# Mesh rotation — fesom_constants.h:21-31
# FESOM default pole at (lon=50°, lat=15°), γ=−90°; almost all computation is
# done in rotated coordinates (FRESH_START §2). cyclic_length=360° × rad = 2π.
# --------------------------------------------------------------------------
ALPHA_EULER_DEG = 50.0
BETA_EULER_DEG = 15.0
GAMMA_EULER_DEG = -90.0
FORCE_ROTATION = True  # apply geo→rot (g2r) at mesh load
CYCLIC_LENGTH_RAD = 2.0 * PI


# --------------------------------------------------------------------------
# Phase-1 namelist defaults — fesom_constants.h:34-106
# Config: which_ALE='linfs', PP mixing, opt_visc=7 (biharmonic), GM/Redi off.
# These are compile-time constants in the C port (no namelist parser yet); we
# mirror them until a config layer is needed.
# --------------------------------------------------------------------------

# Background mixing / drag (FRESH_START §14.7)
K_VER = 1.0e-5  # background tracer vertical diffusivity [m²/s]
A_VER = 1.0e-4  # background momentum vertical viscosity [m²/s]
C_D = 0.0025  # bottom-drag coefficient

# GM/Redi eddy diffusivity ceilings — the 2nd ML-hook seam (Phase 6B). These two
# are the differentiable `params.py` leaves (default = the namelist.oce values);
# the rest of the GM constant bundle lives in `gm.GMConfig`. Redi_Kmax auto-syncs
# to K_GM_max in the C (fesom_gm.c:362). [m²/s].
K_GM_MAX = 1000.0   # GM thickness-diffusivity ceiling (namelist K_GM_max)
REDI_KMAX = 1000.0  # Redi isoneutral-diffusivity ceiling (auto-sync = K_GM_max)

# CVMix classical-TKE tunable constants — the PRIMARY ML-hook seam (Phase 9b). The 4
# differentiable `params.py` leaves (default = the namelist.cvmix reference values,
# echo-verified in docs/tke_reference_namelists/PROVENANCE.md:43-58). The structural
# switches (mxl_min/tke_min/kappaM_max/mxl_choice/…) stay static in `tke.TkeConfig`;
# the Pr-law literal 6.6 stays a static double in `cvmix_tke.py` (TKE_C66). These are
# exactly the constants Phase 7a tunes / Phase 7 NN-replaces (fesom_tke.c:227-241).
TKE_C_K = 0.1      # mixing-length→KappaM coefficient c_k (KappaM = c_k·mxl·√tke)
TKE_C_EPS = 0.7    # dissipation coefficient c_eps (Patankar quasi-implicit term)
TKE_CD = 3.75      # surface-flux coefficient cd (namelist beats module default 1.0)
TKE_ALPHA = 30.0   # TKE self-diffusivity multiplier alpha_tke (NOT in mxl)

# Adams-Bashforth order for Coriolis (single-slot history — FRESH_START §14.4)
AB_ORDER = 2

# SSH solver implicitness — oce_modules.F90:90 (alpha=theta=1.0)
SSH_ALPHA = 1.0
SSH_THETA = 1.0

# CG SSH solver — MOD_DYN.F90:13-25
SOLTOL = 1.0e-5
MAXITER = 500

# Timestep [s]. Default = pi step_per_day=36 → 86400/36 = 2400; CORE2 ≤ 500-600.
DT_DEFAULT = 2400.0

# Vertical-velocity CFL splitter — OFF for the CORE2/linfs target (see header).
USE_WSPLIT = False
WSPLIT_MAXCFL = 1.0

# Horizontal viscosity coefficients (opt_visc=7 biharmonic; gamma*_h=0 → pure
# biharmonic). visc_gamma0 = 0.003, NOT 0.03 (FRESH_START §14.7).
VISC_GAMMA0 = 0.003
VISC_GAMMA1 = 0.1
VISC_GAMMA2 = 0.285
VISC_GAMMA0_H = 0.0
VISC_GAMMA1_H = 0.0
VISC_EASYBSRETURN = 1.0  # opt_visc=5 backscatter return (unused at opt_visc=7)


class ViscConfig(NamedTuple):
    """Per-run horizontal-viscosity γ coefficients (opt_visc=7 biharmonic).

    A **promoted** view of the ``VISC_GAMMA*`` module constants so a run can override them
    from config — NG5 wants ``gamma1 = 0.2`` (CORE2 uses the 0.1 default). The defaults
    reproduce the module constants EXACTLY ⇒ ``ViscConfig()`` is bit-identical to the bare
    module path (the regression-guard invariant). Static/hashable (a ``NamedTuple``) ⇒ a valid
    ``jax.jit`` static arg, like the other ``*Config`` sub-configs. Note there is **no
    ``gamma2_h``** (only ``0_h``/``1_h``) — mirrors ``fesom_momentum.c``."""
    gamma0: float = VISC_GAMMA0       # floor inside max(g0, inner)·len
    gamma1: float = VISC_GAMMA1       # |Δu| flow-aware coefficient (NG5: 0.2)
    gamma2: float = VISC_GAMMA2       # |Δu|² flow-aware coefficient
    gamma0_h: float = VISC_GAMMA0_H   # harmonic-stage floor (0 ⇒ pure biharmonic)
    gamma1_h: float = VISC_GAMMA1_H   # harmonic-stage |Δu| coefficient


# Tracer advection scheme (Phase 4 MFCT/QR4C/FCT). num_ord is HARD-CODED into the
# reconstruction kernels (horizontal MFCT num_ord=0, vertical QR4C num_ord=1) — NOT a
# runtime flag — so this config is currently a SELECTOR/validator, not a kernel switch:
# the implemented (and NG5) path is exactly ('MFCT', 'QR4C', 'FCT') / (0, 1). A different
# num_ord would be a kernel change (new reconstruction), flagged via TracerConfig.validate.
TRACER_HOR_SCHEME = "MFCT"
TRACER_VER_SCHEME = "QR4C"
TRACER_LIMITER = "FCT"
TRACER_NUM_ORD_HOR = 0
TRACER_NUM_ORD_VER = 1


class TracerConfig(NamedTuple):
    """Tracer-advection scheme selector (the implemented MFCT/QR4C/FCT, num_ord (0,1)).

    ``num_ord`` is hard-coded in the reconstruction kernels (not a runtime flag), so this is a
    **validator** over the implemented path, not a kernel switch — :meth:`validate` raises
    ``NotImplementedError` for any other ``(scheme, num_ord)`` (which would require new kernel
    work). The defaults == the implemented = the NG5 ``nml_tracer_list`` ``'MFCT','QR4C','FCT'``
    / ``0., 1.`` ⇒ NG5 needs zero tracer work; CORE2's scheme is OPEN (confirm before its run)."""
    hor_scheme: str = TRACER_HOR_SCHEME
    ver_scheme: str = TRACER_VER_SCHEME
    limiter: str = TRACER_LIMITER
    num_ord_hor: int = TRACER_NUM_ORD_HOR
    num_ord_ver: int = TRACER_NUM_ORD_VER

    def validate(self) -> None:
        impl = (TRACER_HOR_SCHEME, TRACER_VER_SCHEME, TRACER_LIMITER,
                TRACER_NUM_ORD_HOR, TRACER_NUM_ORD_VER)
        got = (self.hor_scheme, self.ver_scheme, self.limiter,
               self.num_ord_hor, self.num_ord_ver)
        if got != impl:
            raise NotImplementedError(
                f"tracer scheme {got} != the implemented {impl}: num_ord is hard-coded into "
                "the MFCT/QR4C reconstruction kernels, so a different scheme/num_ord is a "
                "kernel change (new reconstruction), not a config toggle. The implemented "
                "path matches NG5; confirm the CORE2 matched config before its Fortran run.")

# PP mixing — oce_modules.F90:25-78
MIX_COEFF_PP = 0.01  # PP scaling coefficient
INSTABMIX_KV = 0.1  # convective-adjustment Kv [m²/s]
USE_MOMIX = False  # Monin-Obukhov mixing off in Phase 1 (needs ice+forcing)
USE_INSTABMIX = True  # convective adjustment on
USE_WINDMIX = False

# Shortwave penetration — oce_shortwave_pene.F90 (constant chlorophyll fallback)
USE_SW_PENE = True
CHL_CONST = 0.1  # [mg/m³]

# Sea-ice exchange coefficients (gen_modules_forcing.F90:17-19) — used from Phase 6
CH_ATM_ICE = 1.75e-3  # sensible-heat over ice
CE_ATM_ICE = 1.75e-3  # evaporation over ice
CD_ATM_ICE = 1.2e-3  # atm-ice momentum drag (CORE2 namelist, not module default)
