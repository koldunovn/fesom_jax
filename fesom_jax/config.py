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
