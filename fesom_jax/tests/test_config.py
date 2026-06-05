"""Task 0.1 gate: float64 is active and the constants mirror FRESH_START §17.

These constants are the numerical foundation of the whole port; a wrong π (e.g.
``math.pi`` instead of FESOM's truncated literal) would break every fidelity
gate downstream, so the test pins the *exact* float literals from the C port's
``fesom_constants.h``.
"""

import numpy as np
import jax.numpy as jnp

import fesom_jax  # noqa: F401 — import enables x64 as a side effect
from fesom_jax import config


def test_x64_enabled():
    # Default float dtype must be float64 once the package is imported.
    assert jnp.ones(1).dtype == np.float64
    assert jnp.asarray(1.0).dtype == np.float64
    assert jnp.zeros(3).sum().dtype == np.float64


def test_physical_constants_match_fresh_start_section_17():
    # Exact literals — FRESH_START.md §17 / fesom_constants.h:7-13.
    assert config.PI == 3.14159265358979
    # The Golden Rule: FESOM's π is *truncated*, not full-precision.
    assert config.PI != np.pi
    assert config.RAD == config.PI / 180.0
    assert config.DENSITY_0 == 1030.0
    assert config.G == 9.81
    assert config.R_EARTH == 6367500.0
    assert config.OMEGA == 2.0 * config.PI / 86400.0
    assert config.VCPW == 4.2e6


def test_rotation_defaults():
    # fesom_constants.h:21-31 — FESOM default rotated grid.
    assert config.ALPHA_EULER_DEG == 50.0
    assert config.BETA_EULER_DEG == 15.0
    assert config.GAMMA_EULER_DEG == -90.0
    assert config.FORCE_ROTATION is True
    assert config.CYCLIC_LENGTH_RAD == 2.0 * config.PI


def test_phase1_namelist_defaults():
    # fesom_constants.h:34-106 — locked Phase-1 (linfs, PP, opt_visc=7) defaults.
    assert config.AB_ORDER == 2
    assert config.SSH_ALPHA == 1.0
    assert config.SSH_THETA == 1.0
    assert config.K_VER == 1.0e-5
    assert config.A_VER == 1.0e-4
    assert config.C_D == 0.0025
    assert config.SOLTOL == 1.0e-5
    assert config.MAXITER == 500
    assert config.VISC_GAMMA0 == 0.003  # NOT 0.03 (FRESH_START §14.7)
    assert config.MIX_COEFF_PP == 0.01
    assert config.INSTABMIX_KV == 0.1
